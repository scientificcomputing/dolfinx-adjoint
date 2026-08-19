"""Pointwise observation of a finite element function.

Inverse problems are frequently posed against data that lives at *points* rather than on
the computational mesh: sensor readings, well measurements, or the voxel centres of an
image.  This module provides the discrete observation operator

.. math::

    d = B u, \\qquad B \\in \\mathbb{R}^{n_d \\times N},
    \\qquad B_{ij} = \\phi_j(x_i),

whose :math:`i`-th row evaluates a finite element function at the point :math:`x_i`, and a
differentiable least-squares misfit built on top of it,

.. math::

    J(u) = \\frac{1}{2\\sigma^2} \\lVert W (B u - d) \\rVert^2 .

Composed with a PDE solve, this is the *parameter-to-observable map* of PDE-constrained
optimization and Bayesian inversion: :math:`m \\mapsto B u(m)`.

Parallel semantics
------------------
``points`` must be *replicated* on every MPI rank -- observation points typically come from
a file or an instrument layout that every rank can read.  Each point is assigned to exactly
one owning rank: the lowest-numbered rank whose *owned* cells contain it.  Points that no
rank can locate are reported in :attr:`PointObservation.found` and excluded from the
operator, rather than becoming zero rows -- a zero row in a misfit silently contributes
:math:`d_i^2` and biases the result.
"""

from __future__ import annotations

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt
import pyadjoint
from fenicsx_ii import create_interpolation_matrix
from fenicsx_ii.quadrature import Quadrature
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

from .blocks.observation import PointObservationBlock

__all__ = ["PointObservation", "point_observation_misfit"]


class _PointCloudTrace:
    """Reduction operator handing :mod:`fenicsx_ii` the coordinates of a point cloud.

    ``fenicsx_ii.PointwiseTrace`` is written for 1D line meshes and obtains the physical
    coordinates by compiling a ``SpatialCoordinate`` expression, which FFCx cannot do on a
    point cell. On a point mesh each cell *is* a single geometry node, so the coordinates
    can simply be read off the geometry.
    """

    def __init__(self, mesh: dolfinx.mesh.Mesh) -> None:
        self._mesh = mesh

    def compute_quadrature(self, cells: npt.NDArray[np.int32], reference_points: npt.NDArray[np.floating]):
        gdim = self._mesh.geometry.dim
        nodes = np.asarray(self._mesh.geometry.dofmaps[0])[cells].reshape(-1)
        points = self._mesh.geometry.x[nodes][:, :gdim]
        return Quadrature(
            name="PointCloud",
            points=points,
            weights=np.ones((points.shape[0], 1), dtype=points.dtype),
            scales=np.ones(points.shape[0], dtype=points.dtype),
        )

    @property
    def num_points(self) -> int:
        return 1

    def __str__(self) -> str:
        return f"PointCloudTrace({self._mesh})"


def _default_padding(mesh: dolfinx.mesh.Mesh) -> float:
    """Padding on the scale of rounding error relative to the mesh extent.

    Reduced over the *global* bounding box rather than each process's own, so that the
    operator locates the same points however the mesh happens to be partitioned.
    """
    x = mesh.geometry.x
    empty = x.shape[0] == 0
    local_lower = np.full(3, np.inf) if empty else x.min(axis=0)
    local_upper = np.full(3, -np.inf) if empty else x.max(axis=0)
    lower = np.empty(3)
    upper = np.empty(3)
    mesh.comm.Allreduce(np.ascontiguousarray(local_lower, dtype=np.float64), lower, op=MPI.MIN)
    mesh.comm.Allreduce(np.ascontiguousarray(local_upper, dtype=np.float64), upper, op=MPI.MAX)
    diagonal = float(np.linalg.norm(upper - lower)) if np.all(np.isfinite(lower)) else 0.0
    return 1e-10 * max(diagonal, 1.0)


def _pad_points(points: npt.ArrayLike) -> np.ndarray:
    """Return points as a contiguous ``(num_points, 3)`` float64 array.
    Add zero padding for points in 1D or 2D, so that the bounding-box tree can be
    built once and used for all points.

    Args:
        points: Input points, shape ``(num_points, dim)`` with ``dim <= 3``.

    Returns:
        Padded points, shape ``(num_points, 3)``.
    """
    padded_input = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if padded_input.ndim != 2:
        raise ValueError(f"Points must be 2D with shape (num_points, dim), got shape {padded_input.shape}")
    if padded_input.shape[1] > 3:
        raise ValueError(f"Points can have at most 3 components, got {padded_input.shape[1]}")
    padded = np.zeros((padded_input.shape[0], 3), dtype=np.float64)
    padded[:, : padded_input.shape[1]] = padded_input
    return padded


def _pad_points_collective(comm: MPI.Comm, points: npt.ArrayLike) -> np.ndarray:
    """``_pad_points``, but validated on every rank before any rank can raise.

    A bad ``points`` array is exactly the kind of per-rank data bug the shape checks in
    ``_pad_points`` exist to catch, and it need not affect every rank alike. Raising locally,
    before the replication check below has run, would let one rank exit the constructor while
    the others block forever on that check's collective reduction.
    """
    try:
        padded = _pad_points(points)
        message = ""
    except ValueError as exc:
        padded = np.zeros((0, 3), dtype=np.float64)
        message = str(exc)

    failures = [message for message in comm.allgather(message) if message]
    if failures:
        raise ValueError(failures[0])
    return padded


class PointObservation:
    """The operator :math:`B` evaluating a finite element function at a set of points.

    Args:
        V: Function space the observed state lives in.
        points: Observation points, ``shape=(num_points, gdim)`` or ``(num_points, 3)``.
            Must be identical on every MPI rank.
        padding: Absolute padding of the mesh bounding boxes used when searching for the
            cell containing each point. Defaults to a small multiple of the mesh extent,
            which makes points sitting exactly on the boundary robustly detectable. Large
            values are expensive -- keep this at cell scale at most.

    Attributes:
        num_points: Total (global) number of input points.
        local_indices: Indices into the global point array of the points owned by this
            rank. This is the row ordering used by :meth:`apply`.

    Note:
        For a vector space with block size ``bs``, row ``i * bs + c`` observes component
        ``c`` at point ``i``.

    Example:
        .. code-block:: python

            V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
            B = PointObservation(V, np.array([[0.25, 0.25], [0.75, 0.5]]))
            values = B.evaluate(u)  # u(0.25, 0.25) and u(0.75, 0.5)
    """

    def __init__(
        self,
        V: dolfinx.fem.FunctionSpace,
        points: npt.ArrayLike,
        padding: float | None = None,
    ) -> None:
        mesh = V.mesh
        comm = mesh.comm
        self.function_space = V
        self.comm = comm

        # Validate the element up front, on every process. Doing it lazily during assembly
        # would raise only on the processes that end up owning points, and a rank-divergent
        # exception deadlocks at the next collective call instead of surfacing cleanly.
        if V.element.needs_dof_transformations:
            raise NotImplementedError(
                "PointObservation does not support elements requiring DOF transformations "
                "(for instance high-order elements on oriented, non-simplex cells)."
            )
        try:
            V.element.basix_element
        except RuntimeError as exc:
            raise NotImplementedError(
                "PointObservation needs a function space backed by a Basix element, which mixed "
                "elements are not. Collapse the sub-space you want to observe, for example "
                "`V.sub(0).collapse()[0]`, and build the operator on that."
            ) from exc

        # Make sure all points are 3D. For dim < 3, pad with zeros so that the bounding-box
        # tree can be built once and used for all points.
        padded_points = _pad_points_collective(comm, points)
        num_points = padded_points.shape[0]
        if comm.allreduce(num_points, op=MPI.MIN) != comm.allreduce(num_points, op=MPI.MAX):
            raise ValueError("`points` must be replicated on all processes (got differing lengths).")
        # A checksum costs one scalar reduction and catches the far nastier case of equally
        # many but *different* points, which would otherwise silently corrupt the operator.
        # Weighting each entry by its position makes the checksum sensitive to permuted rows
        # too, not just to changed coordinates.
        position_weights = np.arange(1, padded_points.size + 1, dtype=np.float64)
        checksum = float(np.dot(padded_points.ravel(), position_weights))
        if comm.allreduce(checksum, op=MPI.MIN) != comm.allreduce(checksum, op=MPI.MAX):
            raise ValueError("`points` must be replicated on all processes (got differing coordinates).")
        self.num_points = num_points
        self.points = padded_points

        tdim = mesh.topology.dim
        self.padding = _default_padding(mesh) if padding is None else float(padding)

        # Locate the points in the mesh
        num_owned_cells = mesh.topology.index_map(tdim).size_local
        first_cell = np.full(num_points, -1, dtype=np.int32)
        if num_owned_cells > 0 and num_points > 0:
            owned_cells = np.arange(num_owned_cells, dtype=np.int32)
            tree = dolfinx.geometry.bb_tree(mesh, tdim, padding=self.padding, entities=owned_cells)
            candidates = dolfinx.geometry.compute_collisions_points(tree, padded_points)
            colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, padded_points)
            offsets = colliding.offsets
            has_cell = offsets[1:] > offsets[:-1]
            first_cell[has_cell] = colliding.array[offsets[:-1][has_cell]]

        # Searching owned cells only means a point interior to a cell is found on exactly one
        # process; break the remaining ties, on shared facets and vertices, by lowest rank so
        # that every process agrees on a single owner per point.
        candidate_owner = np.where(first_cell >= 0, comm.rank, comm.size).astype(np.int32)
        owner = np.empty_like(candidate_owner)
        comm.Allreduce(candidate_owner, owner, op=MPI.MIN)

        self._found = owner < comm.size
        self._owner = np.where(self._found, owner, -1).astype(np.int32)
        self.num_found = int(self._found.sum())
        self.local_indices = np.flatnonzero(owner == comm.rank).astype(np.int32)

        self._build_matrix(padded_points[self.local_indices])

    def __repr__(self) -> str:
        return type(self).__name__ + f"(V={self.function_space}, points={self.points}, padding={self.padding})"

    def __str__(self) -> str:
        return f"PointObservation({self.num_found} points, {self.function_space})"

    @property
    def found(self) -> npt.NDArray[np.bool_]:
        """Boolean array of length ``num_points``, ``True`` where the point was located in the mesh on some rank.

        Identical on every rank.
        """
        return self._found

    @property
    def owner(self) -> npt.NDArray[np.int32]:
        """Rank owning each point, ``-1`` where it was not found.

        Identical on every rank.
        """
        return self._owner

    def _build_matrix(self, local_points: np.ndarray) -> None:
        """Build the interpolation matrix from ``V`` onto a point mesh of the owned points.

        The point mesh carries one cell per observation point this process owns, and a
        DG-0 space on it has exactly one degree of freedom per point (``block_size`` of
        them for a vector space), so the interpolation matrix from ``V`` onto that space
        *is* :math:`B`. :mod:`fenicsx_ii` assembles it, including all the cross-process
        communication, and its transpose gives the adjoint for free.
        """
        V = self.function_space
        gdim = V.mesh.geometry.dim
        bs = V.dofmap.bs

        self.point_mesh = dolfinx.mesh.create_point_mesh(
            self.comm, np.ascontiguousarray(local_points[:, :gdim], dtype=V.mesh.geometry.x.dtype)
        )
        element = ("DG", 0) if bs == 1 else ("DG", 0, (bs,))
        self.observation_space = dolfinx.fem.functionspace(self.point_mesh, element)

        self._matrix, _, _ = create_interpolation_matrix(
            V,
            self.observation_space,
            _PointCloudTrace(self.point_mesh),
            tol=self.padding,
            use_petsc=True,
        )
        # Reusable work vectors, so that apply/apply_transpose do not allocate per call.
        self._observation_function = dolfinx.fem.Function(self.observation_space)
        self._state_function = dolfinx.fem.Function(V)
        dm = self.observation_space.dofmap
        self._num_local_rows = dm.index_map.size_local * dm.index_map_bs

    # --------------------------------------------------------------------- actions ---
    @property
    def block_size(self) -> int:
        """Block size of the observed function space."""
        return self.function_space.dofmap.bs

    @property
    def num_local_rows(self) -> int:
        """Number of rows owned by this process, including the block-size unrolling."""
        return self._num_local_rows

    @property
    def matrix(self):
        """The interpolation matrix :math:`B`, as a distributed ``PETSc.Mat``."""
        return self._matrix

    @property
    def dtype(self) -> np.dtype:
        """Scalar dtype of the observed function space.

        Independent of the mesh geometry's dtype, which is always real -- a complex-valued
        ``V`` still has real geometry.
        """
        return self._state_function.x.array.dtype

    def apply(self, u: dolfinx.fem.Function) -> npt.NDArray[np.float64]:
        """Evaluate :math:`Bu` for the rows owned by this process.

        Args:
            u: Function in :attr:`function_space`.

        Returns:
            The point values of the rows owned by this process, ordered as
            :attr:`local_indices` (component-fastest for vector spaces).
        """
        u.x.scatter_forward()
        self._matrix.mult(u.x.petsc_vec, self._observation_function.x.petsc_vec)
        self._observation_function.x.scatter_forward()
        return np.asarray(self._observation_function.x.array[: self.num_local_rows], dtype=np.float64)

    def evaluate(self, u: dolfinx.fem.Function, fill: float = np.nan) -> npt.NDArray[np.float64]:
        """Evaluate ``u`` at every observation point.

        The convenience combination of :meth:`apply` and :meth:`gather`: the result has one
        entry per point (``block_size`` entries per point for a vector space) and is the
        same on every process, so it can be used directly as a data vector.

        Args:
            u: Function in :attr:`function_space`.
            fill: Value used for points that lie outside the mesh.

        Note:
            This does not touch the tape. Use :func:`point_observation_misfit` to build a
            differentiable functional out of the observations.
        """
        return self.gather(self.apply(u), fill=fill)

    def apply_transpose(self, values: npt.ArrayLike, out: dolfinx.la.Vector | None = None) -> dolfinx.la.Vector:
        """Accumulate :math:`B^T v` into a DOF vector.

        Args:
            values: Row values in the layout produced by :meth:`apply`.
            out: Optional vector to accumulate into; created when not supplied.

        Returns:
            The vector holding :math:`B^T v`, with ghost contributions reduced onto their
            owners and scattered back out.
        """
        V = self.function_space
        if out is None:
            out = dolfinx.la.vector(V.dofmap.index_map, V.dofmap.bs, dtype=self.dtype)
            out.array[:] = 0.0

        self._observation_function.x.array[: self.num_local_rows] = np.asarray(values, dtype=np.float64)
        self._observation_function.x.scatter_forward()
        # PETSc performs the reverse communication itself, so no scatter_reverse is needed.
        self._matrix.multTranspose(self._observation_function.x.petsc_vec, self._state_function.x.petsc_vec)
        self._state_function.x.scatter_forward()

        out.array[:] += self._state_function.x.array
        out.scatter_forward()
        return out

    def restrict(self, data: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Select the entries of a global, replicated array belonging to this rank's rows.

        Args:
            data: Array of length ``num_points * block_size``, identical on every rank.

        Returns:
            Array of length ``num_local_rows``, in the layout produced by :meth:`apply`.
        """
        values = np.asarray(data, dtype=np.float64)
        bs = self.block_size
        if values.shape[0] != self.num_points * bs:
            raise ValueError(f"Expected data of length {self.num_points * bs}, got {values.shape[0]}")
        return values.reshape(self.num_points, bs)[self.local_indices].reshape(-1)

    def gather(self, values: npt.ArrayLike, fill: float = np.nan) -> npt.NDArray[np.float64]:
        """Assemble local row values into a global array replicated on every rank.

        Args:
            values: Row values in the layout produced by :meth:`apply`.
            fill: Value used for points that were not found in the mesh.
        """
        bs = self.block_size
        buffer = np.zeros(self.num_points * bs, dtype=np.float64)
        local = np.asarray(values, dtype=np.float64)
        buffer.reshape(self.num_points, bs)[self.local_indices] = local.reshape(-1, bs)
        total = np.empty_like(buffer)
        self.comm.Allreduce(buffer, total, op=MPI.SUM)
        if not np.all(self.found):
            total.reshape(self.num_points, bs)[~self.found] = fill
        return total


def point_observation_misfit(
    u: dolfinx.fem.Function,
    observation: PointObservation,
    data: npt.ArrayLike,
    noise_variance: float = 1.0,
    weights: npt.ArrayLike | None = None,
    ad_block_tag: str | None = None,
    **kwargs,
) -> pyadjoint.AdjFloat:
    """Least-squares misfit between a state observed at points and measured data.

    Computes :math:`\\frac{1}{2\\sigma^2}\\lVert W(Bu - d)\\rVert^2` and records it on the
    tape, so that it can be differentiated with respect to anything ``u`` depends on.

    Args:
        u: The state to observe.
        observation: The observation operator :math:`B`.
        data: The measured values :math:`d`. Either a global array of length
            ``observation.num_points * observation.block_size``, replicated on every rank,
            or an array already restricted to this rank's rows.
        noise_variance: :math:`\\sigma^2`. Use ``1.0`` for a purely deterministic inverse
            problem; for a Bayesian one this is the variance of the additive Gaussian
            observation noise.
        weights: Optional per-row weights :math:`W`, in the same layout as ``data``. A 0/1
            array masks individual observations out.
        ad_block_tag: Optional tag for the tape block.
        kwargs: ``annotate`` may be passed to control whether the tape records this.

    Returns:
        The misfit, as a ``pyadjoint.AdjFloat``.

    Raises:
        ZeroDivisionError: If ``noise_variance`` is zero.
    """
    if noise_variance == 0:
        raise ZeroDivisionError("noise_variance must not be 0.0; use 1.0 for a deterministic inverse problem.")

    local_data = _as_local(observation, data)
    local_weights = None if weights is None else _as_local(observation, weights)

    annotate = annotate_tape(kwargs)
    with stop_annotating():
        output = misfit_value(observation, u, local_data, noise_variance, local_weights)

    overloaded = create_overloaded_object(output)

    if annotate:
        block = PointObservationBlock(
            u, observation, local_data, noise_variance, local_weights, ad_block_tag=ad_block_tag
        )
        get_working_tape().add_block(block)
        block.add_output(overloaded.block_variable)

    return overloaded


def misfit_value(
    observation: PointObservation,
    u: dolfinx.fem.Function,
    data: npt.NDArray[np.float64],
    noise_variance: float,
    weights: npt.NDArray[np.float64] | None,
) -> float:
    """The misfit value, summed over ranks. Used by both the forward pass and the block."""
    residual = observation.apply(u) - data
    if weights is not None:
        residual = weights * residual
    local = 0.5 * float(np.dot(residual, residual)) / noise_variance
    return observation.comm.allreduce(local, op=MPI.SUM)


def _as_local(observation: PointObservation, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Accept either a global (replicated) array or one already restricted to local rows."""
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    expected_global = observation.num_points * observation.block_size
    if array.shape[0] == expected_global:
        return observation.restrict(array)
    if array.shape[0] == observation.num_local_rows:
        return array
    raise ValueError(
        f"Expected an array of length {expected_global} (global) "
        f"or {observation.num_local_rows} (local), got {array.shape[0]}"
    )
