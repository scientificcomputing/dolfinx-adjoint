"""Exact evaluation of source-mesh expressions at another mesh's points."""

from mpi4py import MPI
from mpi4py.util import dtlib

import dolfinx
import numpy as np
import numpy.typing as npt
import ufl


class SourcePointEvaluator:
    """Expressions on a source mesh, evaluated at the interpolation points of a target space.

    Each point is pulled back into the source cell holding it and evaluated there with a
    ``COMM_SELF`` :py:class:`dolfinx.fem.Expression`, as
    ``fenicsx_ii.interpolation_utils.evaluate_basis_function`` does. This is exact on curved and
    non-affine cells, unlike staging the expression in a discontinuous space. Points outside the
    source mesh get zero, as in :py:meth:`dolfinx.fem.Function.interpolate_nonmatching`.

    The points are the owned nodes of ``points_space``, in its dof order. The evaluation is
    collective: every rank has to call the same methods in the same order.

    Args:
        mesh_from: The source mesh, at its current geometry.
        points_space: A target space whose interpolation points are point evaluations.
        padding: Tolerance for locating points on the source mesh.
        batch_size: Points per Expression; every point in a batch is evaluated in every cell of
            the batch, so this trades JIT compilations against evaluation cost.
    """

    def __init__(
        self,
        mesh_from: dolfinx.mesh.Mesh,
        points_space: dolfinx.fem.FunctionSpace,
        padding: float,
        batch_size: int = 200,
    ):
        self._mesh = mesh_from
        self._batch_size = batch_size
        self.num_points = points_space.dofmap.index_map.size_local
        points = points_space.tabulate_dof_coordinates()[: self.num_points]
        ownership = dolfinx.geometry.determine_point_ownership(mesh_from, points, padding)
        src_owner = np.asarray(ownership.src_owner)
        dest_owner = np.asarray(ownership.dest_owner)
        assert np.all(np.diff(dest_owner) >= 0), "Evaluated points are expected grouped by requesting rank"
        self._cells = np.asarray(ownership.dest_cells, dtype=np.int32)
        self._reference = self._pull_back(np.asarray(ownership.dest_points).reshape(-1, 3))

        # Received values arrive grouped by evaluating rank, each group in the local point order.
        found = np.flatnonzero(src_owner >= 0)
        self._received = found[np.argsort(src_owner[found], stable=True)]
        self._evaluators, self._recv_counts = np.unique(src_owner[found], return_counts=True)
        self._requesters, self._send_counts = np.unique(dest_owner, return_counts=True)

    def _pull_back(self, points: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Reference coordinates of ``points`` in ``self._cells``, on the current geometry."""
        geometry = self._mesh.geometry
        cmaps = getattr(geometry, "cmaps", None)
        cmap = cmaps[0] if cmaps is not None else geometry.cmap
        dofmaps = getattr(geometry, "dofmaps", None)
        dofmap = dofmaps[0] if dofmaps is not None else geometry.dofmap
        gdim = geometry.dim
        reference = np.zeros((len(self._cells), self._mesh.topology.dim), dtype=geometry.x.dtype)
        for cell in np.unique(self._cells):
            rows = np.flatnonzero(self._cells == cell)
            reference[rows] = cmap.pull_back(points[rows, :gdim], geometry.x[dofmap[cell], :gdim])
        return reference

    def _exchange(self, data: np.ndarray, reverse: bool) -> np.ndarray:
        """Send per-point rows between the evaluating ranks and the ranks owning the points.

        Forward: rows at the evaluated points, in evaluation order, go to the owners and come back
        in received order (``self._received``). Reverse: rows in received order go to the
        evaluating ranks and arrive in evaluation order.
        """
        width = int(np.prod(data.shape[1:], dtype=int))
        sources, destinations = self._evaluators, self._requesters
        recv_counts, send_counts = self._recv_counts, self._send_counts
        num_recv = len(self._received)
        if reverse:
            sources, destinations = destinations, sources
            recv_counts, send_counts = send_counts, recv_counts
            num_recv = len(self._cells)
        received = np.zeros((num_recv, width), dtype=data.dtype)
        datatype = dtlib.from_numpy_dtype(data.dtype)
        comm = self._mesh.comm.Create_dist_graph_adjacent(sources.tolist(), destinations.tolist(), reorder=False)
        try:
            comm.Neighbor_alltoallv(
                [np.ascontiguousarray(data).reshape(-1), send_counts * width, datatype],
                [received, recv_counts * width, datatype],
            )
        finally:
            comm.Free()
        return received

    def _tabulate(self, expression: ufl.core.expr.Expr, dtype) -> np.ndarray:
        """``expression`` at each evaluated point, in its own cell: ``(points, ...)`` as ``Expression.eval``."""
        values = []
        for start in range(0, len(self._cells), self._batch_size):
            cells = self._cells[start : start + self._batch_size]
            compiled = dolfinx.fem.Expression(
                expression, self._reference[start : start + self._batch_size], comm=MPI.COMM_SELF, dtype=dtype
            )
            batch = compiled.eval(self._mesh, cells)
            # Every point was evaluated in every cell of the batch; keep each point's own cell.
            values.append(batch[np.arange(len(cells)), np.arange(len(cells))])
        return np.concatenate(values) if values else np.zeros((0,))

    def evaluate(self, expression: ufl.core.expr.Expr, dtype=dolfinx.default_scalar_type) -> np.ndarray:
        """A coefficient expression at the points: ``(num_points, *expression.ufl_shape)``."""
        shape = expression.ufl_shape
        local = self._tabulate(expression, dtype).reshape(len(self._cells), -1)
        values = np.zeros((self.num_points, int(np.prod(shape, dtype=int))), dtype=dtype)
        values[self._received] = self._exchange(local.astype(dtype, copy=False), reverse=False)
        return values.reshape(self.num_points, *shape)

    def apply_transpose(self, expression: ufl.core.expr.Expr, weights: np.ndarray, out: dolfinx.fem.Function) -> None:
        r"""Add the transpose of ``f -> expression[f](x_i)``, applied to ``weights``, into ``out``.

        Computes ``out_j += sum_i weights_i : expression[phi_j](x_i)`` over the basis ``phi_j`` of
        ``out``'s space, and reduces the ghost contributions onto their owners.

        Args:
            expression: Linear in a single {py:class}`ufl.Argument` on the source mesh's
                ``out.function_space``.
            weights: ``(num_points, *expression.ufl_shape)``.
            out: Accumulated into; ghost entries stay consistent with their owners.
        """
        (argument,) = ufl.algorithms.extract_arguments(expression)
        space = out.function_space
        assert argument.ufl_function_space() == space
        dtype = out.x.array.dtype
        width = int(np.prod(expression.ufl_shape, dtype=int))
        ordered = np.ascontiguousarray(weights.reshape(self.num_points, width)[self._received], dtype=dtype)
        local_weights = self._exchange(ordered, reverse=True)
        contribution = np.zeros(out.x.array.size, dtype=dtype)
        if len(self._cells) > 0:
            basis = self._tabulate(expression, dtype).reshape(len(self._cells), width, -1)
            local = np.einsum("pv,pvd->pd", local_weights, basis)
            bs = space.dofmap.bs
            dofs = space.dofmap.list[self._cells]
            blocked = (dofs[:, :, None] * bs + np.arange(bs)).reshape(len(self._cells), -1)
            np.add.at(contribution, blocked, local)
        scratch = dolfinx.fem.Function(space, dtype=dtype)
        scratch.x.array[:] = contribution
        scratch.x.scatter_reverse(dolfinx.la.InsertMode.add)
        scratch.x.scatter_forward()
        out.x.array[:] += scratch.x.array
