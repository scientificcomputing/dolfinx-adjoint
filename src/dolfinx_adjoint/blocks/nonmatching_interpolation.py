import functools
import typing

import basix.ufl
import dolfinx
import numpy as np
import ufl
from pyadjoint import Block
from pyadjoint.tape import stop_annotating

from ..compat import get_interpolation_points
from ..types.function import _create_function
from ..types.mesh import Mesh, get_overloaded_mesh_if_annotated
from ._point_evaluation import SourcePointEvaluator
from .interpolation import MatrixFreeInterpolationOperator, _MatrixCSRWorkspace, coordinate_derivative, get_mult


def _import_fenicsx_ii():
    """Lazy import of fenicsx_ii to avoid strict dependencies."""
    try:
        import fenicsx_ii
    except ImportError as e:
        raise ImportError(
            "The 'fenicsx_ii' package is required for non-matching interpolation. "
            "Please install it using 'pip install fenicsx_ii'."
        ) from e
    return fenicsx_ii


def _base_element(space: dolfinx.fem.FunctionSpace):
    """The scalar element ``space`` is built from, blocked or not."""
    element = space.ufl_element()
    return element.sub_elements[0] if element.num_sub_elements > 0 else element


def _reject_unsupported_shape_control(space_to, red_op) -> None:
    """Refuse what the shape terms below do not cover.

    They assume each target dof is the source's value at a target point. A Piola-mapped target
    breaks that; :py:func:`dolfinx_adjoint.interpolate_nonmatching` splits it into blocks that
    do satisfy it. A custom reduction operator is a different calculation.

    Raises:
        NotImplementedError: For a custom ``red_op`` or a Piola-mapped target.
    """
    if red_op is not None:
        raise NotImplementedError(
            "Shape derivatives of a non-matching interpolation are only implemented for point "
            "evaluation; a custom red_op reduces the source differently and has its own derivative."
        )
    pullback = space_to.ufl_element().pullback
    if not pullback.is_identity:
        raise NotImplementedError(
            "This block's shape terms need an identity-pullback target, but it uses a "
            f"{type(pullback).__name__} pullback. dolfinx_adjoint.interpolate_nonmatching "
            "handles that case by splitting it into supported steps; use it instead of "
            "constructing the block directly."
        )


class NonmatchingInterpolationBlock(Block):
    """
    Block for interpolating a dolfinx.fem.Function between non-matching meshes.

    Uses `fenicsx_ii` to explicitly build the transfer matrix $J$ across non-matching
    grids, ensuring exact parallel mathematical transposes for the Adjoint and Hessian passes.
    """

    def __init__(
        self,
        func_from: dolfinx.fem.Function,
        func_to: dolfinx.fem.Function,
        cells,
        interpolation_data,
        tol: float = 1e-6,
        maxit: int = 15,
        red_op=None,  # Optional fenicsx_ii ReductionOperator
        ad_block_tag: str | None = None,
        use_petsc: bool = False,
    ):
        super().__init__(ad_block_tag=ad_block_tag)
        self.space_from = func_from.function_space
        self.space_to = func_to.function_space
        self.cells = cells
        self.interpolation_data = interpolation_data
        self.tol = tol
        self.maxit = maxit
        self._red_op = red_op
        self._use_petsc = use_petsc

        self.add_dependency(func_from)

        # Where the two meshes sit relative to each other is geometry, so a tracked mesh on
        # either side is a dependency -- and everything cached about that relative position has
        # to be rebuilt whenever it changes. `_mesh_from` and `_mesh_to` may be the same object,
        # in which case the two derivative terms below cancel, as they should.
        self._mesh_from = get_overloaded_mesh_if_annotated(self.space_from.mesh.ufl_domain())
        self._mesh_to = get_overloaded_mesh_if_annotated(self.space_to.mesh.ufl_domain())
        self._tracks_geometry = self._mesh_from is not None or self._mesh_to is not None
        if self._tracks_geometry:
            _reject_unsupported_shape_control(self.space_to, red_op)
            for mesh in (self._mesh_from, self._mesh_to):
                if mesh is not None:
                    self.add_dependency(mesh, no_duplicates=True)

        # Output caches
        self._adj_output: dolfinx.fem.Function | None = None
        self._tlm_output: dolfinx.fem.Function | None = None
        self._hessian_output: dolfinx.fem.Function | None = None
        self._geometry_output: dict[int, dolfinx.fem.Function] = {}

        # Matrix cache
        self._matrix_workspace = None
        self._shape_workspace: dict[str, typing.Any] = {}

    # --- Geometry ---

    def _refresh_geometry(self) -> None:
        """Drop everything that describes where the two meshes sit relative to each other.

        ``interpolation_data`` locates every target interpolation point inside a cell of the
        source mesh and stores its reference coordinates there; the transfer matrix is built from
        the same locations. Both are correct only for the geometry they were computed on, so a
        tape replay at a different displacement has to rebuild them -- otherwise even the
        *forward* value is wrong, before any derivative is involved.

        A no-op unless a mesh is tracked, so an ordinary non-matching interpolation still pays
        for the point location exactly once.
        """
        if not self._tracks_geometry:
            return
        self._matrix_workspace = None
        self._shape_workspace.clear()
        self.interpolation_data = dolfinx.fem.create_interpolation_data(
            self.space_to, self.space_from, self.cells, padding=self.tol
        )

    def _points(self) -> SourcePointEvaluator:
        """The target's interpolation points, located on the source mesh at its current geometry."""
        if "points" not in self._shape_workspace:
            self._shape_workspace["points"] = SourcePointEvaluator(
                self.space_from.mesh, self._displacement_space, self.tol
            )
        return self._shape_workspace["points"]

    @functools.cached_property
    def _displacement_space(self) -> dolfinx.fem.FunctionSpace:
        """The target's base element blocked to ``(gdim,)``: one node per target point."""
        shape = (self.space_to.mesh.geometry.dim,)
        element = basix.ufl.blocked_element(_base_element(self.space_to), shape=shape)
        return dolfinx.fem.functionspace(self.space_to.mesh, element)

    @property
    def _value_shape(self) -> tuple[int, ...]:
        return tuple(self.space_to.ufl_element().reference_value_shape)

    def _eulerian(self, expression, direction):
        r"""Derivative of the source-mesh field ``expression`` at a fixed physical point, as the
        source mesh moves by ``direction``: ``D_sigma f - grad(f) sigma``.

        ``D_sigma f`` is the derivative at a fixed reference point, nonzero for a Piola-mapped
        ``f`` (its push-forward depends on the Jacobian) and zero for an identity-pullback one.
        """
        material = coordinate_derivative(expression, self.space_from.mesh, None, direction)
        return material - ufl.dot(ufl.grad(expression), direction)

    def _as_target_fields(self, values: np.ndarray) -> list[dolfinx.fem.Function]:
        """``(points, *value, gdim)`` values as one displacement-space Function per value component."""
        gdim = self.space_to.mesh.geometry.dim
        values = values.reshape(self._points().num_points, -1, gdim)
        fields = []
        for component in range(values.shape[1]):
            field = dolfinx.fem.Function(self._displacement_space)
            field.x.array[: values.shape[0] * gdim] = values[:, component].reshape(-1)
            field.x.scatter_forward()
            fields.append(field)
        return fields

    def _target_motion_operator(self, fields: list[dolfinx.fem.Function]) -> MatrixFreeInterpolationOperator:
        r"""The map ``dx_B -> g(x_i) . dx_B(x_i)`` from the target mesh's geometry space, one
        ``g`` per target value component.

        The target's interpolation points ride along with its own mesh, so this is an ordinary
        same-mesh interpolation of ``dot(g, w)``.
        """
        assert self._mesh_to is not None
        w = ufl.TrialFunction(self._mesh_to._ad_function_space)
        dots = np.array([ufl.dot(field, w) for field in fields], dtype=object)
        expression = dots[0] if not self._value_shape else ufl.as_tensor(dots.reshape(self._value_shape).tolist())
        return MatrixFreeInterpolationOperator(expression, self.space_to)

    def _weights(self, vector) -> np.ndarray:
        """The owned part of a target-space vector, ``(points, *value)``."""
        num_points = self._points().num_points
        size = int(np.prod(self._value_shape, dtype=int))
        return vector.array[: num_points * size].reshape(num_points, *self._value_shape)

    def __str__(self):
        return f"interpolate_nonmatching_{self.space_from.mesh.name}_to_{self.space_to.mesh.name}"

    def _get_interpolation_matrix(self):
        if self._matrix_workspace is None:
            # We import fenicsx_ii lazily to avoid strict dependencies
            fenicsx_ii = _import_fenicsx_ii()

            # Use provided reduction operator or default to PointEvaluationOperator
            red_op = self._red_op
            if red_op is None:
                red_op = fenicsx_ii.PointwiseTrace(self.space_to.mesh)

            # Assemble the explicit global transfer matrix
            mat, _, _ = fenicsx_ii.create_interpolation_matrix(
                self.space_from,
                self.space_to,
                red_op=red_op,
                tol=self.tol,
                use_petsc=self._use_petsc,
            )

            if self._use_petsc:
                self._matrix_workspace = mat
            else:
                self._matrix_workspace = _MatrixCSRWorkspace(mat)

        return self._matrix_workspace

    # --- Recompute (Forward Pass) ---

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return None

    def recompute_component(self, inputs, block_variable, idx, prepared):
        func_from = inputs[0]
        output = block_variable.saved_output

        # We use the built-in FEniCSx C++ nonmatching interpolation for the forward evaluation
        # when no custom reduction operator is supplied, because it doesn't need the matrix
        # and is highly optimized. fenicsx_ii is only imported (and required) once a custom
        # red_op is used, since that's the only case that needs the explicit transfer matrix.
        with stop_annotating():
            self._refresh_geometry()
            if self._red_op is None:
                output.interpolate_nonmatching(
                    func_from, self.cells, self.interpolation_data, tol=self.tol, maxit=self.maxit
                )
            else:
                mat = self._get_interpolation_matrix()
                mult = get_mult(mat, transpose=False, accumulate=False)
                mult(func_from.x, output.x)
            output.x.scatter_forward()

        return output

    # --- Tangent Linear Model (TLM) ---

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        # We prepare the matrix here so we can guarantee the exact discrete
        # algebraic pathway as the Adjoint
        self._refresh_geometry()
        return self._prepared(inputs, any(t is not None for t in tlm_inputs[1:]))

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        if all(tlm_input is None for tlm_input in tlm_inputs):
            return None

        if self._tlm_output is None:
            self._tlm_output = _create_function(self.space_to)

        out_func = self._tlm_output
        out_func.x.array[:] = 0.0

        if tlm_inputs[0] is not None:
            mult = get_mult(prepared["matrix"], transpose=False, accumulate=True)
            mult(tlm_inputs[0].x, out_func.x)

        for dependency, tlm_input in zip(self.get_dependencies(), tlm_inputs):
            if tlm_input is None or not isinstance(dependency.output, Mesh):
                continue
            out_func.x.array[: self._owned(out_func)] += self._tangent_geometry_component(
                tlm_input, dependency.output, prepared
            )[: self._owned(out_func)]
        out_func.x.scatter_forward()

        return out_func

    # --- Shape terms ---

    @staticmethod
    def _owned(function: dolfinx.fem.Function) -> int:
        return function.function_space.dofmap.index_map.size_local * function.function_space.dofmap.index_map_bs

    def _involves_geometry(self, relevant_dependencies) -> bool:
        """Whether a sweep reaches one of the meshes.

        ``relevant_dependencies`` may be ``None`` when a block is driven directly rather than by
        the tape; that is treated as "possibly", so a tracked mesh is never silently skipped.
        """
        if not self._tracks_geometry:
            return False
        if relevant_dependencies is None:
            return True
        return any(isinstance(dependency.output, Mesh) for _, dependency in relevant_dependencies)

    def _prepared(self, inputs, needs_geometry: bool) -> dict:
        """The transfer matrix, plus the source function and ``grad(u)`` at the target points
        when a mesh is involved."""
        prepared = {"matrix": self._get_interpolation_matrix(), "u": inputs[0], "grad": None}
        if self._tracks_geometry and needs_geometry:
            prepared["grad"] = self._as_target_fields(self._points().evaluate(ufl.grad(inputs[0])))
        return prepared

    def _tangent_geometry_component(self, direction, mesh: Mesh, prepared: dict) -> np.ndarray:
        r"""The target dofs' response to displacing ``mesh`` by ``direction``.

        With ``c_i = u(x_i)``: ``grad(u)(x_i) . dx(x_i)`` for the target mesh, whose points move,
        and the Eulerian derivative of ``u`` at ``x_i`` for the source mesh. On one mesh the two
        cancel for an identity-pullback source, as interpolating onto your own mesh should.
        """
        scratch = _create_function(self.space_to)
        scratch.x.array[:] = 0.0
        if mesh is self._mesh_to:
            self._target_motion_operator(prepared["grad"]).mult(direction.x, scratch.x, accumulate=True)
        if mesh is self._mesh_from:
            values = self._points().evaluate(self._eulerian(prepared["u"], direction))
            scratch.x.array[: values.size] += values.reshape(-1)
        return scratch.x.array

    @staticmethod
    def _reduced_transpose(operator: MatrixFreeInterpolationOperator, vector, space) -> np.ndarray:
        """``operator^T vector`` in ``space``, with ghost contributions reduced.

        :py:meth:`MatrixFreeInterpolationOperator.mult_transpose` adds into ghost entries too and
        leaves them there, so the result has to be sent to the owners before it is used. It is
        reduced *on its own* rather than after being summed with the source-mesh term, whose
        transfer (``get_mult``) already reduces: reducing the sum would count that one twice.

        Invisible with a P1 target on a P1 geometry, whose interpolation points are the geometry
        vertices themselves, so every owned row touches only an owned vertex. A P2 target's
        edge-midpoint rows touch two vertices, which may sit on different ranks -- 2.3% wrong on
        three ranks before this was added.
        """
        scratch = _create_function(space)
        scratch.x.array[:] = 0.0
        operator.mult_transpose(vector, scratch.x, accumulate=True)
        scratch.x.scatter_reverse(dolfinx.la.InsertMode.add)
        scratch.x.scatter_forward()
        return scratch.x.array

    def _adjoint_geometry_component(self, adj_vector, mesh: Mesh, prepared: dict) -> dolfinx.fem.Function:
        """The transpose of :py:meth:`_tangent_geometry_component`, into ``mesh``'s geometry space."""
        out_func = self._geometry_output.get(id(mesh))
        if out_func is None:
            out_func = _create_function(mesh._ad_function_space)
            self._geometry_output[id(mesh)] = out_func
        out_func.x.array[:] = 0.0

        if mesh is self._mesh_to:
            out_func.x.array[:] += self._reduced_transpose(
                self._target_motion_operator(prepared["grad"]), adj_vector, mesh._ad_function_space
            )
        if mesh is self._mesh_from:
            test = ufl.TestFunction(mesh._ad_function_space)
            self._points().apply_transpose(self._eulerian(prepared["u"], test), self._weights(adj_vector), out_func)
        return out_func

    # --- Adjoint ---

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        self._refresh_geometry()
        return self._prepared(inputs, self._involves_geometry(relevant_dependencies))

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        # The source function is always dependency 0; any tracked mesh was registered after it.
        if idx > 0:
            mesh = self.get_dependencies()[idx].output
            return self._adjoint_geometry_component(adj_inputs[0], mesh, prepared).x

        if self._adj_output is None:
            self._adj_output = _create_function(self.space_from)

        out_func = self._adj_output
        out_func.x.array[:] = 0.0

        mult = get_mult(prepared["matrix"], transpose=True, accumulate=True)
        mult(adj_inputs[0], out_func.x)
        return out_func.x

    # --- Hessian ---

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        r"""The transfer matrix, ``grad(u)`` at the target points, and the tangent-linear direction.

        With ``c_i = u(x_i)``, a direction ``p = (du, sigma, dx)`` for the coefficient, the source
        mesh and the target mesh, and ``E(f, sigma)`` the Eulerian derivative of a source field
        (:py:meth:`_eulerian`), the first derivative is

            dc[p] = du + E(u, sigma) + grad(u) dx,

        at ``x_i``. Differentiating again, with ``A = du1 + E(u, sigma1)``,

            d2c[p1, p2] = E(du2, sigma1) + grad(du2) dx1
                          + E(A, sigma2) + grad(E(u, sigma2)) dx1
                          + (grad(A) + dx1^T H_u) dx2,

        grouped by the ``p2`` component each line is linear in. ``p1`` is the tangent-linear
        direction; :py:meth:`evaluate_hessian_component` applies the transpose in ``p2``. On a
        curved or Piola-mapped source, ``E`` carries the Jacobian's derivatives, which is why every
        term is evaluated at the points rather than through a discontinuous intermediate space.
        """
        self._refresh_geometry()
        prepared = self._prepared(inputs, self._tracks_geometry)
        prepared["second"] = self._second_order_direction(inputs[0]) if self._tracks_geometry else None
        return prepared

    def _tlm_of(self, target) -> typing.Any:
        """The tangent-linear value of the dependency whose output is ``target``, if any."""
        for dependency in self.get_dependencies():
            if dependency.output is target:
                return dependency.tlm_value
        return None

    def _second_order_direction(self, func_from: dolfinx.fem.Function) -> dict | None:
        """``du1``, ``sigma1``, ``dx1`` at the target points, and ``A``, or ``None`` with no direction."""
        u_dot = self.get_dependencies()[0].tlm_value
        sigma = self._tlm_of(self._mesh_from) if self._mesh_from is not None else None
        target_motion = self._tlm_of(self._mesh_to) if self._mesh_to is not None else None
        if u_dot is None and sigma is None and target_motion is None:
            return None

        moved = None
        if target_motion is not None:
            sampled = dolfinx.fem.Function(self._displacement_space)
            # Same mesh: the target points' own motion. An Expression, since DOLFINx cannot
            # interpolate a Function into a quadrature space (the split Piola case uses one).
            with stop_annotating():
                sampled.interpolate(
                    dolfinx.fem.Expression(target_motion, get_interpolation_points(sampled.function_space))
                )
            gdim = self.space_to.mesh.geometry.dim
            moved = sampled.x.array[: self._points().num_points * gdim].reshape(-1, gdim)
        terms = [term for term in (u_dot, sigma and self._eulerian(func_from, sigma)) if term is not None]
        return {"u_dot": u_dot, "sigma": sigma, "dx": moved, "A": sum(terms[1:], terms[0]) if terms else None}

    def _along_motion(self, weights: np.ndarray, moved: np.ndarray) -> np.ndarray:
        """``weights`` ``(points, *value)`` times the target motion ``(points, gdim)``, for a ``grad`` term."""
        return weights[..., None] * moved.reshape(moved.shape[0], *(1,) * (weights.ndim - 1), -1)

    def evaluate_hessian_component(
        self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None
    ):
        hessian_input = getattr(hessian_inputs[0], "x", hessian_inputs[0])
        adj_input = getattr(adj_inputs[0], "x", adj_inputs[0])
        second = prepared["second"]
        points = self._points() if second is not None else None
        u = prepared["u"]

        if idx > 0:
            mesh = self.get_dependencies()[idx].output
            out = self._adjoint_geometry_component(hessian_input, mesh, prepared)
            if second is None:
                return out.x
            lam = self._weights(adj_input)
            if mesh is self._mesh_to:
                gdim = self.space_to.mesh.geometry.dim
                q = np.zeros((points.num_points, *self._value_shape, gdim), dtype=lam.dtype)
                if second["A"] is not None:
                    q += points.evaluate(ufl.grad(second["A"]))
                if second["dx"] is not None:
                    hessian = points.evaluate(ufl.grad(ufl.grad(u)))
                    q += np.einsum("i...kl,ik->i...l", hessian, second["dx"])
                operator = self._target_motion_operator(self._as_target_fields(q))
                out.x.array[:] += self._reduced_transpose(operator, adj_input, mesh._ad_function_space)
            if mesh is self._mesh_from:
                test = ufl.TestFunction(mesh._ad_function_space)
                if second["A"] is not None:
                    points.apply_transpose(self._eulerian(second["A"], test), lam, out)
                if second["dx"] is not None:
                    points.apply_transpose(
                        ufl.grad(self._eulerian(u, test)), self._along_motion(lam, second["dx"]), out
                    )
            out.x.scatter_forward()
            return out.x

        if self._hessian_output is None:
            self._hessian_output = _create_function(self.space_from)

        out_func = self._hessian_output
        out_func.x.array[:] = 0.0

        mult = get_mult(prepared["matrix"], transpose=True, accumulate=True)
        mult(hessian_input, out_func.x)
        if second is not None:
            lam = self._weights(adj_input)
            test = ufl.TestFunction(self.space_from)
            if second["sigma"] is not None:
                points.apply_transpose(self._eulerian(test, second["sigma"]), lam, out_func)
            if second["dx"] is not None:
                points.apply_transpose(ufl.grad(test), self._along_motion(lam, second["dx"]), out_func)
        out_func.x.scatter_forward()
        return out_func.x
