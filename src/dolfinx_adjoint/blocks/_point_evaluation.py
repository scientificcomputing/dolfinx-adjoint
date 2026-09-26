"""Exact evaluation of source-mesh expressions at another mesh's points, through fenicsx_ii."""

import dolfinx
import numpy as np
import ufl


class SourcePointEvaluator:
    """Expressions on a source mesh, evaluated at the interpolation points of a target space.

    Each point is pulled back into the source cell holding it and evaluated there
    (:py:func:`fenicsx_ii.evaluate_expression`), which is exact on curved and non-affine cells, unlike
    staging the expression in a discontinuous space. The values move between the ranks owning the
    points and the ranks holding their cells with :py:class:`fenicsx_ii.PointExchange`.
    Points outside the source mesh get zero, as in :py:meth:`dolfinx.fem.Function.interpolate_nonmatching`.

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
        # fenicsx_ii is an optional dependency, so it is imported only when this is used.
        from .nonmatching_interpolation import _import_fenicsx_ii

        fenicsx_ii = _import_fenicsx_ii()
        self._evaluate_expression = fenicsx_ii.evaluate_expression
        self._mesh = mesh_from
        self._batch_size = batch_size
        self.num_points = points_space.dofmap.index_map.size_local
        points = points_space.tabulate_dof_coordinates()[: self.num_points]
        ownership = dolfinx.geometry.determine_point_ownership(mesh_from, points, padding)
        self._exchange = fenicsx_ii.PointExchange(mesh_from.comm, ownership)

    def _tabulate(self, expression: ufl.core.expr.Expr, dtype) -> np.ndarray:
        """``expression`` at each point this rank evaluates, in the cell holding it."""
        return self._evaluate_expression(
            expression,
            self._mesh,
            self._exchange.evaluated_points,
            self._exchange.evaluated_cells,
            batch_size=self._batch_size,
            dtype=dtype,
        )

    def evaluate(self, expression: ufl.core.expr.Expr, dtype=dolfinx.default_scalar_type) -> np.ndarray:
        """A coefficient expression at the points: ``(num_points, *expression.ufl_shape)``."""
        return self._exchange.forward(self._tabulate(expression, dtype))

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
        local_weights = self._exchange.reverse(np.asarray(weights, dtype=dtype).reshape(self.num_points, width))
        contribution = np.zeros(out.x.array.size, dtype=dtype)
        cells = self._exchange.evaluated_cells
        if len(cells) > 0:
            basis = self._tabulate(expression, dtype).reshape(len(cells), width, -1)
            local = np.einsum("pv,pvd->pd", local_weights, basis)
            bs = space.dofmap.bs
            dofs = space.dofmap.list[cells]
            blocked = (dofs[:, :, None] * bs + np.arange(bs)).reshape(len(cells), -1)
            np.add.at(contribution, blocked, local)
        scratch = dolfinx.fem.Function(space, dtype=dtype)
        scratch.x.array[:] = contribution
        scratch.x.scatter_reverse(dolfinx.la.InsertMode.add)
        scratch.x.scatter_forward()
        out.x.array[:] += scratch.x.array
