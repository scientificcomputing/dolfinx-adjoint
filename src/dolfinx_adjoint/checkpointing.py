"""Snapshot checkpointing of DOLFINx functions to disk.

A checkpoint schedule that uses :class:`checkpoint_schedules.StorageType.DISK` needs somewhere
to put a function's values. This module provides that as a *snapshot* checkpoint: it is written
and read within a single run, by the same processes, against an unchanged mesh and partition.
Under those assumptions the whole payload is a process's local values, so no mesh, geometry or
permutation data is stored and the file is a flat array per stored value. Ghost values are
stored alongside the owned ones, which keeps restoring free of communication -- see `_layout`.

Snapshot checkpoints are therefore not portable. They cannot be reopened by a later run, or on a
different number of processes. For a checkpoint that outlives the run, use ``io4dolfinx``.
"""

from __future__ import annotations

import os
import tempfile
import typing
import weakref

from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint.checkpointing
from pyadjoint.tape import TapePackageData, get_working_tape

__all__ = ["enable_disk_checkpointing", "disable_disk_checkpointing", "SnapshotCheckpoint"]

#: Key under which the disk checkpointer registers itself in ``Tape._package_data``.
_PACKAGE_KEY = "dolfinx_adjoint"

#: The active checkpointer, or None when disk checkpointing is not enabled.
_checkpointer: typing.Optional["_DiskCheckpointer"] = None

# Message pyadjoint shows when a schedule wants disk storage but none is configured.
pyadjoint.checkpointing.disk_checkpointing_callback[_PACKAGE_KEY] = (
    "Call dolfinx_adjoint.enable_disk_checkpointing() before enabling a schedule that uses disk storage."
)


def _import_h5py():
    try:
        import h5py
    except ImportError as e:  # pragma: no cover - exercised only without h5py
        raise ImportError("Disk checkpointing requires h5py. Install it with 'pip install h5py'.") from e
    return h5py


def _layout(function: dolfinx.fem.Function, shared_file: bool, comm: MPI.Comm) -> tuple[int, int, int]:
    """Describe where this process's values sit in a stored dataset.

    The whole local array is stored, ghost values included, not just the locally owned values.
    Owned values alone would be smaller, but restoring them requires a forward scatter to
    refill the ghosts, and that is collective. Restores are driven by whichever blocks happen
    to need a value, and are additionally filtered by a cache whose lifetime depends on when
    the garbage collector runs -- which is not the same moment on every process. A collective
    call on that path deadlocks as soon as one process takes a cached value while another
    reads. Storing the ghosts makes restoring purely local, so it cannot deadlock.

    Returns:
        A tuple of the number of values this process stores, the length of the whole dataset,
        and this process's offset into it.
    """
    n_local = function.x.array.size
    if not shared_file:
        return n_local, n_local, 0
    # Collective, but called only from the write path, which every process reaches together.
    sizes = comm.allgather(n_local)
    return n_local, sum(sizes), sum(sizes[: comm.rank])


class _CheckpointFile:
    """One HDF5 file holding snapshot checkpoints.

    The file is opened once and closed explicitly. It must not be closed from a finaliser:
    with MPI-IO, opening and closing are collective, and Python's garbage collector does not
    run at the same moment on every process, so a close driven by collection deadlocks. Every
    call here therefore happens at a point all processes reach together -- creating the file,
    rolling to a new one when the tape resets, and tearing down.
    """

    def __init__(self, path: str, comm: MPI.Comm, use_mpio: bool, cleanup: bool):
        h5py = _import_h5py()
        self.path = path
        self.comm = comm
        # A shared file is a single file that every process writes its own slice of. Without
        # MPI-IO each process gets a file to itself instead.
        # One shared file that every process writes a slice of, or one file per process.
        self.shared_file = use_mpio or comm.size == 1
        # Only a shared file is written by more than one process, so only then does deleting it
        # belong to a single one of them.
        self._deleted_by_this_process = cleanup and (comm.rank == 0 or not self.shared_file)
        kwargs = {"driver": "mpio", "comm": comm} if use_mpio else {}
        self._handle = h5py.File(path, "w", **kwargs)
        self._next_index = 0
        self._closed = False

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
        dataset = self._handle.create_dataset(key, (n_global,), dtype=values.dtype)
        dataset[offset : offset + values.size] = values

    def read(self, key: str, n_local: int, offset: int) -> np.ndarray:
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
                os.remove(self.path)
            except OSError:  # pragma: no cover - another process may have removed it first
                pass


class SnapshotCheckpoint:
    """A stored checkpoint, holding a reference to its data rather than the data itself.

    Returned by :meth:`Function._ad_create_checkpoint` while disk checkpointing is active, and
    turned back into a function by :meth:`Function._ad_restore_at_checkpoint`.
    """

    __slots__ = ("_file", "_key", "_space", "_cls", "_n_local", "_offset", "_name", "_cache", "__weakref__")

    def __init__(self, file: _CheckpointFile, key: str, function: dolfinx.fem.Function, n_local: int, offset: int):
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
        self._cache: typing.Optional[weakref.ReferenceType] = None

    def restore(self):
        """Read the stored values back into a function of the original type."""
        from .types.function import Function

        if self._cache is not None:
            cached = self._cache()
            if cached is not None:
                return cached

        # Mirrors Function._ad_new_like: going through __new__ preserves the concrete subclass
        # (Constant takes a different constructor signature).
        restored = self._cls.__new__(self._cls, self._space)
        Function.__init__(restored, self._space)
        restored.name = self._name
        # Purely local: the stored array already includes the ghost values, so no scatter.
        restored.x.array[:] = self._file.read(self._key, self._n_local, self._offset)
        self._cache = weakref.ref(restored)
        return restored


class _DiskCheckpointer(TapePackageData):
    """Tape-attached state owning the checkpoint files for one tape."""

    def __init__(self, directory: str, comm: MPI.Comm, use_mpio: bool, cleanup: bool, owns_directory: bool):
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
        path = os.path.join(self._directory, f"checkpoint_{self._generation}{rank_suffix}.h5")
        self._generation += 1
        return _CheckpointFile(path, self._comm, self._use_mpio, self._cleanup)

    @property
    def storing(self) -> bool:
        """Whether values should currently be written to disk rather than kept in memory."""
        return self._storing

    def store(self, function: dolfinx.fem.Function) -> SnapshotCheckpoint:
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


def maybe_disk_checkpoint(function: dolfinx.fem.Function) -> typing.Optional[SnapshotCheckpoint]:
    """Store ``function`` on disk if disk checkpointing is active, otherwise return None.

    Returning None tells the caller to fall back to an in-memory copy. Disk storage is only
    active inside the windows pyadjoint opens around writing checkpoint data, so most calls
    return None even when disk checkpointing is enabled.
    """
    if _checkpointer is None or not _checkpointer.storing:
        return None
    return _checkpointer.store(function)


def enable_disk_checkpointing(
    dirname: typing.Optional[str] = None,
    comm: typing.Optional[MPI.Comm] = None,
    cleanup: bool = True,
    use_mpio: typing.Optional[bool] = None,
) -> None:
    """Store checkpoints on disk rather than in memory.

    Must be called before any operation is recorded on the working tape, and before enabling a
    checkpoint schedule on it.

    Args:
        dirname: Directory to hold the checkpoint files. A temporary directory is created if
            this is not given.
        comm: MPI communicator. Defaults to ``MPI.COMM_WORLD``.
        cleanup: Whether to delete checkpoint files once nothing refers to them. Pass False to
            keep them for inspection.
        use_mpio: Whether to write one shared file with MPI-IO. The default chooses it when
            running on more than one process with an MPI-enabled h5py, and falls back to one
            file per process otherwise. Pass False to force the per-process layout.
    """
    global _checkpointer

    if _checkpointer is not None:
        # Enabling twice would otherwise strand the previous files, open and undeleted.
        disable_disk_checkpointing()

    tape = get_working_tape()
    if tape.get_blocks():
        raise RuntimeError(
            "Disk checkpointing must be enabled before any blocks are added to the tape, "
            "so that every checkpoint is stored the same way."
        )

    comm = MPI.COMM_WORLD if comm is None else comm
    h5py = _import_h5py()
    if use_mpio is None:
        use_mpio = comm.size > 1 and h5py.get_config().mpi
    elif use_mpio and not h5py.get_config().mpi:
        raise RuntimeError(
            "use_mpio=True requires an MPI-enabled build of h5py. Use use_mpio=False to write "
            "one checkpoint file per process instead."
        )
    owns_directory = dirname is None
    if dirname is None:
        # Every process must agree on the directory, even in the per-process layout.
        created = tempfile.mkdtemp(prefix="dolfinx_adjoint_checkpoints_") if comm.rank == 0 else None
        directory = typing.cast(str, comm.bcast(created, root=0))
    else:
        directory = dirname
        if comm.rank == 0:
            os.makedirs(directory, exist_ok=True)
        comm.Barrier()

    _checkpointer = _DiskCheckpointer(directory, comm, use_mpio, cleanup, owns_directory)
    tape._package_data[_PACKAGE_KEY] = _checkpointer


def disable_disk_checkpointing() -> None:
    """Stop storing checkpoints on disk and delete the checkpoint files.

    Collective: every process must call it, because closing a shared checkpoint file is.
    """
    global _checkpointer

    tape = get_working_tape()
    tape._package_data.pop(_PACKAGE_KEY, None)
    if _checkpointer is not None:
        _checkpointer.close()
    _checkpointer = None
