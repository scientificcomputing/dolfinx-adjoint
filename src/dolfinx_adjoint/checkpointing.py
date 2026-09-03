"""Snapshot checkpointing of DOLFINx functions to disk.

A checkpoint schedule that uses {py:class}`checkpoint_schedules.schedule.StorageType` ``DISK`` needs somewhere
to put a function's values. This module provides that as a *snapshot* checkpoint: it is written
and read within a single run, by the same processes, against an unchanged mesh and partition.
Under those assumptions the whole payload is a process's local values, so no mesh, geometry or
permutation data is stored and the file is a flat array per stored value. Ghost values are
stored alongside the owned ones, which keeps restoring free of communication -- see `_layout`.

Snapshot checkpoints are therefore not portable. They cannot be reopened by a later run, or on a
different number of processes. For a checkpoint that outlives the run, use {py:mod}`io4dolfinx`.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
import typing
import weakref

from mpi4py import MPI

import h5py
import numpy as np
import pyadjoint.checkpointing
from pyadjoint.tape import TapePackageData, get_working_tape

if typing.TYPE_CHECKING:
    # Annotations only. Importing at runtime would be circular: dolfinx_adjoint.types imports
    # this module to decide where a checkpoint goes.
    from .types.function import Function

__all__ = ["enable_disk_checkpointing", "disable_disk_checkpointing", "SnapshotCheckpoint"]

#: Key under which the disk checkpointer registers itself in ``Tape._package_data``.
#:
#: That dictionary is the only place a checkpointer is held. It is deliberately not also
#: tracked in a module global: a checkpoint file must outlive every
#: {py:class}`SnapshotCheckpoint` written into it, and those belong to one tape's block
#: variables. A global would make the file's lifetime follow whichever tape was configured
#: most recently instead, so enabling disk checkpointing on a second tape would close the
#: first tape's file out from under a reduced functional that is still perfectly usable.
_PACKAGE_KEY = "dolfinx_adjoint"

# Message pyadjoint shows when a schedule wants disk storage but none is configured.
pyadjoint.checkpointing.disk_checkpointing_callback[_PACKAGE_KEY] = (
    "Call dolfinx_adjoint.enable_disk_checkpointing() before enabling a schedule that uses disk storage."
)


def _layout(function: Function, shared_file: bool, comm: MPI.Intracomm) -> tuple[int, int, int]:
    """Describe where this process's values sit in a stored dataset.

    The whole local array is stored, ghost values included, not just the locally owned values.
    Owned values alone would be smaller, but restoring them requires a forward scatter to
    refill the ghosts, and that is collective. Restores are driven by whichever blocks happen
    to need a value, and are additionally filtered by a cache whose lifetime depends on when
    the garbage collector runs -- which is not the same moment on every process. A collective
    call on that path deadlocks as soon as one process takes a cached value while another
    reads. Storing the ghosts makes restoring purely local, so it cannot deadlock.

    Args:
        function: The function whose values are about to be stored.
        shared_file: Whether the dataset spans every process's values (one shared file) or
            only this process's (one file per process).
        comm: The communicator the checkpoint files are shared over.

    Returns:
        A tuple of the number of values this process stores, the length of the whole dataset,
        and this process's offset into it.
    """
    n_local = function.x.array.size
    if not shared_file:
        return n_local, n_local, 0
    # Collective, but reached only from the write path, which every process reaches together.
    # An exclusive scan rather than gathering every size and summing a prefix: it is the
    # operation this actually is, and its cost does not grow with the number of processes.
    offset = comm.exscan(n_local, op=MPI.SUM)
    if comm.rank == 0:
        offset = 0
    return n_local, comm.allreduce(n_local, op=MPI.SUM), offset


class _CheckpointFile:
    """One HDF5 file holding snapshot checkpoints.

    The file is opened once and closed explicitly. It must not be closed from a finaliser:
    with MPI-IO, opening and closing are collective, and Python's garbage collector does not
    run at the same moment on every process, so a close driven by collection deadlocks. Every
    call here therefore happens at a point all processes reach together -- creating the file,
    rolling to a new one when the tape resets, and tearing down.
    """

    def __init__(self, path: pathlib.Path, comm: MPI.Intracomm, use_mpio: bool, cleanup: bool):
        """
        Args:
            path: Where to create the file.
            comm: The communicator the file is shared over.
            use_mpio: Whether to open one shared file with MPI-IO, so that every process writes
                its own slice of each dataset. Without it each process gets its own file.
            cleanup: Whether to delete the file when it is closed. False keeps it on disk for
                inspection, which is only useful for debugging.
        """
        self._path = path
        self._comm = comm
        # One shared file that every process writes a slice of, or one file per process.
        self._shared_file = use_mpio or comm.size == 1
        # Only a shared file is written by more than one process, so only then does deleting it
        # belong to a single one of them.
        self._deleted_by_this_process = cleanup and (comm.rank == 0 or not self._shared_file)
        kwargs = {"driver": "mpio", "comm": comm} if use_mpio else {}
        self._handle = h5py.File(path, "w", **kwargs)
        self._next_index = 0
        self._closed = False

    @property
    def path(self) -> pathlib.Path:
        """Where this file lives."""
        return self._path

    @property
    def comm(self) -> MPI.Intracomm:
        """The communicator this file is shared over."""
        return self._comm

    @property
    def shared_file(self) -> bool:
        """Whether one file holds every process's values, rather than one file per process."""
        return self._shared_file

    def next_key(self) -> str:
        """Return a dataset name that every process agrees on.

        Safe because checkpoints are taken in the same order on every process: pyadjoint holds
        the checkpointable state in an insertion-ordered set, and all processes run the same
        schedule.
        """
        key = f"checkpoint_{self._next_index}"
        self._next_index += 1
        return key

    def write(self, key: str, values: np.ndarray, n_global: int, offset: int) -> None:
        """Store one process's values in a new dataset.

        Args:
            key: Dataset name, from {py:meth}`next_key`.
            values: The values this process contributes, ghost values included.
            n_global: Length of the whole dataset, across every process.
            offset: Where this process's values start in it.
        """
        dataset = self._handle.create_dataset(key, (n_global,), dtype=values.dtype)
        dataset[offset : offset + values.size] = values

    def read(self, key: str, n_local: int, offset: int) -> np.ndarray:
        """Read this process's values back out of a dataset.

        Args:
            key: Dataset name, as passed to {py:meth}`write`.
            n_local: How many values this process stored.
            offset: Where this process's values start in the dataset.

        Returns:
            The stored values, ghost values included.
        """
        return self._handle[key][offset : offset + n_local]

    def close(self) -> None:
        """Close the file, deleting it unless it is being kept for inspection.

        Collective when the file was opened with MPI-IO, so every process must call it.
        """
        if self._closed:
            return
        self._closed = True
        self._handle.close()
        if self._deleted_by_this_process:
            try:
                os.remove(self._path)
            except OSError:  # pragma: no cover - another process may have removed it first
                pass


class SnapshotCheckpoint:
    """A stored checkpoint, holding a reference to its data rather than the data itself.

    Returned by :meth:`Function._ad_create_checkpoint` while disk checkpointing is active, and
    turned back into a function by :meth:`Function._ad_restore_at_checkpoint`.
    """

    __slots__ = ("_file", "_key", "_space", "_cls", "_n_local", "_offset", "_name", "_cache", "__weakref__")

    def __init__(self, file: _CheckpointFile, key: str, function: Function, n_local: int, offset: int):
        # Holding the file keeps it alive for exactly as long as some checkpoint needs it.
        self._file = file
        self._key = key
        self._space = function.function_space
        self._cls = type(function)
        self._n_local = n_local
        self._offset = offset
        self._name = function.name
        # Weak, so that repeated restores during one block evaluation hand back the *same*
        # object -- the blocks build replacement maps across several `saved_output` accesses
        # and a fresh object each time makes those maps miss. Weak rather than strong so the
        # values are released again once the block is done with them, which is the point of
        # storing them on disk in the first place.
        self._cache: weakref.ReferenceType | None = None

    def restore(self) -> Function:
        """Read the stored values back into a function of the original type."""
        from .types.function import Function

        if self._cache is not None:
            cached = self._cache()
            if cached is not None:
                return cached

        # Mirrors Function._ad_new_like: going through __new__ preserves the concrete subclass
        # (Constant takes a different constructor signature).
        restored = self._cls.__new__(self._cls, self._space)  # type: ignore[call-arg]
        Function.__init__(restored, self._space)
        restored.name = self._name
        # Purely local: the stored array already includes the ghost values, so no scatter.
        restored.x.array[:] = self._file.read(self._key, self._n_local, self._offset)
        self._cache = weakref.ref(restored)
        return restored


class _DiskCheckpointer(TapePackageData):
    """Tape-attached state owning the checkpoint files for one tape."""

    def __init__(
        self,
        directory: pathlib.Path,
        comm: MPI.Intracomm,
        use_mpio: bool,
        cleanup: bool,
        owns_directory: bool,
    ):
        """
        Args:
            directory: Where the checkpoint files are written.
            comm: The communicator the files are shared over.
            use_mpio: Whether to write one shared file with MPI-IO.
            cleanup: Whether to delete the files, and the directory, on teardown.
            owns_directory: Whether this object created the directory and so should remove it.
                Must agree across processes, or teardown deadlocks.
        """
        self._directory = directory
        self._comm = comm
        self._use_mpio = use_mpio
        self._cleanup = cleanup
        self._owns_directory = owns_directory
        self._generation = 0
        self._storing = False
        self._file = self._roll_to_new_file()

    def _roll_to_new_file(self) -> _CheckpointFile:
        # Reached by every process together (pyadjoint resets package data on all of them), so
        # it is safe to close the superseded file here.
        previous = getattr(self, "_file", None)
        if previous is not None:
            previous.close()
        rank_suffix = "" if (self._use_mpio or self._comm.size == 1) else f"_rank{self._comm.rank}"
        path = self._directory / f"checkpoint_{self._generation}{rank_suffix}.h5"
        self._generation += 1
        return _CheckpointFile(path, self._comm, self._use_mpio, self._cleanup)

    @property
    def storing(self) -> bool:
        """Whether values should currently be written to disk rather than kept in memory."""
        return self._storing

    def store(self, function: Function) -> SnapshotCheckpoint:
        """Write a function's values to the current checkpoint file.

        Args:
            function: The function to store.

        Returns:
            A handle that reads the values back.
        """
        n_local, n_global, offset = _layout(function, self._file.shared_file, self._comm)
        key = self._file.next_key()
        self._file.write(key, function.x.array, n_global, offset)
        return SnapshotCheckpoint(self._file, key, function, n_local, offset)

    # -- TapePackageData ------------------------------------------------------------------

    def clear(self):
        # The tape is being discarded, so no checkpoint taken so far can still be wanted.
        self._file = self._roll_to_new_file()

    def reset(self):
        # Deliberately not rolling to a new file. pyadjoint resets package data before
        # recomputing the forward, but then restores the initial condition from a checkpoint
        # written while taping, so data from before the reset is still live. Rolling the file
        # here deletes it and the restore fails. Checkpoints therefore accumulate in one file
        # for as long as disk checkpointing is enabled, and are removed together at teardown.
        self._storing = False

    def checkpoint(self):
        return self._file

    def restore_from_checkpoint(self, state):
        self._file = state

    def copy(self):
        other = _DiskCheckpointer.__new__(_DiskCheckpointer)
        other.__dict__.update(self.__dict__)
        return other

    def close(self) -> None:
        """Close the current file and remove the directory if this object created it."""
        self._file.close()
        self._storing = False
        if self._owns_directory:
            self._comm.Barrier()
            if self._comm.rank == 0:
                try:
                    os.rmdir(self._directory)
                except OSError:  # pragma: no cover - non-empty when cleanup was disabled
                    pass

    def continue_checkpointing(self):
        self._storing = True

    def pause_checkpointing(self):
        self._storing = False


def _checkpointer_for(tape) -> "_DiskCheckpointer | None":
    """Return the checkpointer registered on ``tape``, or None if it has none."""
    checkpointer = tape._package_data.get(_PACKAGE_KEY)
    return checkpointer if isinstance(checkpointer, _DiskCheckpointer) else None


def maybe_disk_checkpoint(function: Function) -> SnapshotCheckpoint | None:
    """Store ``function`` on disk if disk checkpointing is active, otherwise return None.

    Returning None tells the caller to fall back to an in-memory copy. Disk storage is only
    active inside the windows pyadjoint opens around writing checkpoint data, so most calls
    return None even when disk checkpointing is enabled.

    Args:
        function: The function pyadjoint is asking to checkpoint.

    Returns:
        A handle to the stored values, or None to keep them in memory.
    """
    checkpointer = _checkpointer_for(get_working_tape())
    if checkpointer is None or not checkpointer.storing:
        return None
    return checkpointer.store(function)


def enable_disk_checkpointing(
    dirname: str | os.PathLike | None = None,
    comm: MPI.Intracomm | None = None,
    cleanup: bool = True,
    use_mpio: bool | None = None,
) -> None:
    """Store the working tape's checkpoints on disk rather than in memory.

    Must be called before any operation is recorded on the working tape, and before enabling a
    checkpoint schedule on it.

    Disk checkpointing is a property of one tape, not of the process: enabling it on a second
    tape leaves the first tape's checkpoints, and the file holding them, alone. Each tape's
    files live until {py:func}`disable_disk_checkpointing` is called with that tape as the
    working tape, or until the process exits -- nothing can close them implicitly, because
    closing a shared file is collective and so cannot be driven by garbage collection.

    Args:
        dirname: Directory to hold the checkpoint files. A temporary directory is created if
            this is not given.
        comm: MPI communicator. Defaults to ``MPI.COMM_WORLD``.
        cleanup: Whether to delete the checkpoint files, and the temporary directory, on
            teardown. Pass False to keep them for inspection; they are unreadable by any later
            run either way.
        use_mpio: Whether to write one shared file with MPI-IO. The default chooses it when
            running on more than one process with an MPI-enabled h5py, and falls back to one
            file per process otherwise. Pass False to force the per-process layout.
    """
    tape = get_working_tape()
    if tape.get_blocks():
        raise RuntimeError(
            "Disk checkpointing must be enabled before any blocks are added to the tape, "
            "so that every checkpoint is stored the same way."
        )
    if _checkpointer_for(tape) is not None:
        # Re-enabling on the same tape, to change the directory or the layout. Closing the old
        # files is safe here and only here: the guard above has just established that this tape
        # holds no blocks, so it holds no checkpoint that could still point into them.
        disable_disk_checkpointing(tape)

    comm = MPI.COMM_WORLD if comm is None else comm
    if use_mpio is None:
        use_mpio = comm.size > 1 and h5py.get_config().mpi
    elif use_mpio and not h5py.get_config().mpi:
        raise RuntimeError(
            "use_mpio=True requires an MPI-enabled build of h5py. Use use_mpio=False to write "
            "one checkpoint file per process instead."
        )
    # Whether we created the directory decides whether teardown removes it, and teardown
    # synchronises the processes before doing so. Every process must therefore agree: if some
    # were given a `dirname` and others were not, teardown would deadlock.
    without_dirname = comm.allreduce(int(dirname is None), op=MPI.SUM)
    if without_dirname not in (0, comm.size):
        raise ValueError(
            "dirname must be given on every process or on none of them, "
            f"but it was omitted on {without_dirname} of {comm.size}."
        )

    owns_directory = without_dirname == comm.size
    if owns_directory:
        # Every process must agree on the directory, even in the per-process layout.
        created = tempfile.mkdtemp(prefix="dolfinx_adjoint_checkpoints_") if comm.rank == 0 else None
        directory = pathlib.Path(comm.bcast(created, root=0))
    else:
        directory = pathlib.Path(typing.cast("str | os.PathLike", dirname))
        if comm.rank == 0:
            directory.mkdir(parents=True, exist_ok=True)
        comm.Barrier()

    tape._package_data[_PACKAGE_KEY] = _DiskCheckpointer(directory, comm, use_mpio, cleanup, owns_directory)


def disable_disk_checkpointing(tape=None) -> None:
    """Stop storing a tape's checkpoints on disk and delete its checkpoint files.

    Every {py:class}`SnapshotCheckpoint` written by this tape becomes unreadable, so call it
    only once nothing will evaluate a reduced functional built on the tape again.

    Collective: every process must call it, because closing a shared checkpoint file is.

    Args:
        tape: The tape to stop checkpointing to disk. Defaults to the working tape. Pass one
            explicitly to tear down a tape that is no longer current -- popping the key off
            whichever tape happens to be working instead would leave the real owner holding a
            checkpointer whose file is gone, which still satisfies pyadjoint's "disk storage is
            configured" check and so fails much later, at the first restore.
    """
    tape = get_working_tape() if tape is None else tape
    checkpointer = _checkpointer_for(tape)
    if checkpointer is None:
        return
    del tape._package_data[_PACKAGE_KEY]
    checkpointer.close()
