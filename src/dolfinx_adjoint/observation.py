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
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

from .blocks.observation import PointObservationBlock

__all__ = ["PointObservation", "point_observation_misfit"]


def _geometry_dofmap(mesh: dolfinx.mesh.Mesh):
    """``mesh.geometry.dofmap`` was renamed to ``dofmaps[0]``."""
    try:
        return mesh.geometry.dofmaps[0]
    except AttributeError:
        return mesh.geometry.dofmap


def _coordinate_map(mesh: dolfinx.mesh.Mesh):
    """``mesh.geometry.cmap`` is deprecated in favour of ``cmaps[0]``."""
    try:
        return mesh.geometry.cmaps[0]
    except (AttributeError, IndexError):
        return mesh.geometry.cmap


#: Cell types whose degree-1 coordinate map is affine. Tensor-product cells
#: (quadrilateral, hexahedron, prism, pyramid) are multilinear rather than affine even at
#: degree 1, so they take the general Newton pull-back.
AFFINE_CELL_TYPES = frozenset(
    {
        dolfinx.mesh.CellType.point,
        dolfinx.mesh.CellType.interval,
        dolfinx.mesh.CellType.triangle,
        dolfinx.mesh.CellType.tetrahedron,
    }
)


def _is_affine_simplex(mesh: dolfinx.mesh.Mesh) -> bool:
    """True if every cell map is affine, i.e. a degree-1 simplex geometry."""
    return _coordinate_map(mesh).degree == 1 and mesh.topology.cell_type in AFFINE_CELL_TYPES


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
    """Return points as a contiguous ``(num_points, 3)`` float64 array."""
    padded_input = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if padded_input.ndim != 2:
        raise ValueError(f"Points must be 2D with shape (num_points, dim), got shape {padded_input.shape}")
    if padded_input.shape[1] > 3:
        raise ValueError(f"Points can have at most 3 components, got {padded_input.shape[1]}")
    padded = np.zeros((padded_input.shape[0], 3), dtype=np.float64)
    padded[:, : padded_input.shape[1]] = padded_input
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
        found: Boolean array of length ``num_points``, ``True`` where the point was located
            in the mesh on some rank. Identical on every rank.
        owner: Rank owning each point, ``-1`` where it was not found. Identical on every
            rank.
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
            self._basix_element = V.element.basix_element
        except RuntimeError as exc:
            raise NotImplementedError(
                "PointObservation needs a function space backed by a Basix element, which mixed "
                "elements are not. Collapse the sub-space you want to observe, for example "
                "`V.sub(0).collapse()[0]`, and build the operator on that."
            ) from exc

        padded_points = _pad_points(points)
        num_points = padded_points.shape[0]
        if comm.allreduce(num_points, op=MPI.MIN) != comm.allreduce(num_points, op=MPI.MAX):
            raise ValueError("`points` must be replicated on all processes (got differing lengths).")
        # A checksum costs one scalar reduction and catches the far nastier case of equally
        # many but *different* points, which would otherwise silently corrupt the operator.
        checksum = float(padded_points.sum())
        if comm.allreduce(checksum, op=MPI.MIN) != comm.allreduce(checksum, op=MPI.MAX):
            raise ValueError("`points` must be replicated on all processes (got differing coordinates).")
        self.num_points = num_points
        self.points = padded_points

        tdim = mesh.topology.dim
        gdim = mesh.geometry.dim
        self.padding = _default_padding(mesh) if padding is None else float(padding)

        # Search owned cells only: a point interior to a cell is then found on exactly one
        # rank, and only points on a shared facet or vertex remain ambiguous.
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

        # Break ties by lowest rank, so every rank agrees on a single owner per point.
        candidate_owner = np.where(first_cell >= 0, comm.rank, comm.size).astype(np.int32)
        owner = np.empty_like(candidate_owner)
        comm.Allreduce(candidate_owner, owner, op=MPI.MIN)

        self.found = owner < comm.size
        self.owner = np.where(self.found, owner, -1).astype(np.int32)
        self.num_found = int(self.found.sum())
        self.local_indices = np.flatnonzero(owner == comm.rank).astype(np.int32)

        self._assemble(padded_points[self.local_indices], first_cell[self.local_indices], gdim)

    # -------------------------------------------------------------------- assembly ---
    def _assemble(self, points: np.ndarray, cells: np.ndarray, gdim: int) -> None:
        """Tabulate the basis functions of the containing cell for every owned point.

        The operator is stored as two dense ``(num_local_rows, num_dofs_per_cell)`` arrays
        -- the basis values and the corresponding *ghosted* local DOF indices -- rather
        than a sparse matrix. Every row has exactly the same number of entries, so this is
        both smaller and faster than a general sparse format, and keeps the package free
        of a sparse-matrix dependency.
        """
        V = self.function_space
        dofmap = V.dofmap
        bs = dofmap.bs
        self._num_local_dofs = (dofmap.index_map.size_local + dofmap.index_map.num_ghosts) * bs

        if len(points) == 0:
            num_dofs_per_cell = dofmap.dof_layout.num_dofs
            self._basis = np.zeros((0, num_dofs_per_cell))
            self._cols = np.zeros((0, num_dofs_per_cell), dtype=np.int32)
            return

        reference_points = self._pull_back(points, cells, gdim)

        # The reference basis is shared by all cells, so a single tabulate call suffices.
        basis = np.asarray(self._basix_element.tabulate(0, reference_points))[0, :, :, 0]
        cell_dofs = np.asarray(dofmap.list)[cells]

        if bs == 1:
            self._basis = basis
            self._cols = cell_dofs.astype(np.int32)
        else:
            # Unroll to one row per (point, component): row i*bs + c reads the DOFs of
            # component c, weighted by the same scalar basis values.
            num_rows, num_dofs_per_cell = basis.shape
            self._basis = np.repeat(basis, bs, axis=0)
            components = np.arange(bs, dtype=np.int32)
            cols = cell_dofs[:, None, :] * bs + components[None, :, None]
            self._cols = cols.reshape(num_rows * bs, num_dofs_per_cell).astype(np.int32)

    def _pull_back(self, points: np.ndarray, cells: np.ndarray, gdim: int) -> np.ndarray:
        """Map physical points to the reference coordinates of their containing cell."""
        mesh = self.function_space.mesh
        geometry_dofmap = _geometry_dofmap(mesh)
        geometry_x = mesh.geometry.x
        tdim = mesh.topology.dim
        dtype = geometry_x.dtype

        if _is_affine_simplex(mesh) and gdim == tdim:
            # X = x0 + J xi, so xi = J^{-1} (X - x0) with a constant Jacobian per cell.
            coords = geometry_x[np.asarray(geometry_dofmap)[cells]][:, : tdim + 1, :gdim]
            jacobian = np.swapaxes(coords[:, 1:, :] - coords[:, :1, :], 1, 2)
            offsets = (points[:, :gdim] - coords[:, 0, :])[..., None]
            return np.linalg.solve(jacobian, offsets)[..., 0].astype(dtype)

        # Otherwise fall back to the coordinate element's Newton iteration, which handles
        # one cell at a time; group the points by cell to call it as rarely as possible.
        cmap = _coordinate_map(mesh)
        reference_points = np.zeros((len(points), tdim), dtype=dtype)
        order = np.argsort(cells, kind="stable")
        boundaries = np.flatnonzero(np.diff(cells[order])) + 1
        for group in np.split(order, boundaries):
            cell_coords = geometry_x[geometry_dofmap[cells[group[0]]]][:, :gdim]
            physical = np.ascontiguousarray(points[group, :gdim], dtype=dtype)
            reference_points[group] = cmap.pull_back(physical, cell_coords)
        return reference_points

    # --------------------------------------------------------------------- actions ---
    @property
    def block_size(self) -> int:
        """Block size of the observed function space."""
        return self.function_space.dofmap.bs

    @property
    def num_local_rows(self) -> int:
        """Number of rows owned by this rank, including the block-size unrolling."""
        return self._basis.shape[0]

    def apply(self, u: dolfinx.fem.Function) -> npt.NDArray[np.float64]:
        """Evaluate :math:`Bu` for the rows owned by this rank.

        Args:
            u: Function in :attr:`function_space`.

        Returns:
            The point values of the rows owned by this rank, ordered as
            :attr:`local_indices` (component-fastest for vector spaces).
        """
        u.x.scatter_forward()
        if self.num_local_rows == 0:
            return np.zeros(0)
        return np.einsum("ij,ij->i", self._basis, u.x.array[self._cols])

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
            out = dolfinx.la.vector(V.dofmap.index_map, V.dofmap.bs, dtype=V.mesh.geometry.x.dtype)
            out.array[:] = 0.0
        if self.num_local_rows > 0:
            weights = self._basis * np.asarray(values, dtype=np.float64)[:, None]
            contributions = np.bincount(
                self._cols.reshape(-1), weights=weights.reshape(-1), minlength=self._num_local_dofs
            )
            out.array[:] += contributions
        out.scatter_reverse(dolfinx.la.InsertMode.add)
        out.scatter_forward()
        return out

    def restrict(self, data: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Select the entries of a global, replicated array belonging to this rank's rows.

        Args:
            data: Array of length ``num_points * block_size``, identical on every rank.
        """
        values = np.asarray(data, dtype=np.float64)
        bs = self.block_size
        if values.shape[0] != self.num_points * bs:
            raise ValueError(f"Expected data of length {self.num_points * bs}, got {values.shape[0]}")
        if bs == 1:
            return values[self.local_indices]
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
        if bs == 1:
            buffer[self.local_indices] = local
        else:
            buffer.reshape(self.num_points, bs)[self.local_indices] = local.reshape(-1, bs)
        total = np.empty_like(buffer)
        self.comm.Allreduce(buffer, total, op=MPI.SUM)
        if not np.all(self.found):
            missing = ~self.found
            if bs == 1:
                total[missing] = fill
            else:
                total.reshape(self.num_points, bs)[missing] = fill
        return total

    def to_scipy(self):
        """This rank's block of :math:`B` as a ``scipy.sparse`` CSR matrix.

        Columns index the ghosted local DOF array (``u.x.array``). Requires ``scipy``,
        which is not a dependency of this package; :meth:`apply` and
        :meth:`apply_transpose` do not need it.
        """
        import scipy.sparse

        num_rows, num_dofs_per_cell = self._basis.shape
        rows = np.repeat(np.arange(num_rows), num_dofs_per_cell)
        return scipy.sparse.csr_matrix(
            (self._basis.reshape(-1), (rows, self._cols.reshape(-1))),
            shape=(num_rows, self._num_local_dofs),
        )


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
