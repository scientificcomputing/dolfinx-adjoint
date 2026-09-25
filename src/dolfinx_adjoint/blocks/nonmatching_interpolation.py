import typing

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import ufl
from pyadjoint import Block
from pyadjoint.tape import stop_annotating

from ..compat import get_interpolation_points
from ..types.function import _create_function
from ..types.mesh import Mesh, overloaded_mesh
from .interpolation import MatrixFreeInterpolationOperator, _MatrixCSRWorkspace, get_mult


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


def _blocked_space(mesh: dolfinx.mesh.Mesh, base, shape: tuple[int, ...]) -> dolfinx.fem.FunctionSpace:
    """``base`` blocked to ``shape`` on ``mesh``, so its points match the unblocked space's."""
    return dolfinx.fem.functionspace(mesh, basix.ufl.blocked_element(base, shape=shape))


def _coordinate_degree(mesh: dolfinx.mesh.Mesh) -> int:
    """Degree of the coordinate element; ``geometry.cmap`` is deprecated in favour of ``cmaps[0]``."""
    cmaps = getattr(mesh.geometry, "cmaps", None)
    return (cmaps[0] if cmaps is not None else mesh.geometry.cmap).degree


def _reject_unsupported_shape_control(space_from, space_to, red_op) -> None:
    """Refuse the cases whose shape derivative the terms below do not cover.

    The derivative assumes the target dof is a point evaluation, ``c_i = u(x_i)``, and that the
    source function is carried along by its mesh with its dofs fixed. Both break for an element
    whose pullback is not the identity: the dof is then a Piola-mapped moment, and the geometry
    enters the interpolation operator as well as the points. The same goes for a custom reduction
    operator, which replaces point evaluation with something else entirely (an average over a
    circle, say), and whose derivative is a different calculation.

    Args:
        space_from: The source function's space.
        space_to: The space being interpolated into.
        red_op: The reduction operator, or ``None`` for point evaluation.

    Raises:
        NotImplementedError: If either space is not identity-pullback, or a custom ``red_op``
            is in use.
    """
    if red_op is not None:
        raise NotImplementedError(
            "Shape derivatives of a non-matching interpolation are only implemented for point "
            "evaluation; a custom red_op reduces the source differently and has its own derivative."
        )
    for label, space in (("source", space_from), ("target", space_to)):
        pullback = space.ufl_element().pullback
        if not isinstance(pullback, ufl.pullback.IdentityPullback):
            raise NotImplementedError(
                f"Shape derivatives of a non-matching interpolation need point-evaluation dofs, but "
                f"the {label} space uses a {type(pullback).__name__} pullback, whose dofs are "
                "Piola-mapped moments rather than values at points."
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
        self._mesh_from = overloaded_mesh(self.space_from.mesh.ufl_domain())
        self._mesh_to = overloaded_mesh(self.space_to.mesh.ufl_domain())
        self._tracks_geometry = self._mesh_from is not None or self._mesh_to is not None
        if self._tracks_geometry:
            _reject_unsupported_shape_control(self.space_from, self.space_to, red_op)
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

    def _transfer_matrix(self, space_from: dolfinx.fem.FunctionSpace, space_to: dolfinx.fem.FunctionSpace):
        """A cached point-evaluation transfer matrix from ``space_from`` to ``space_to``."""
        key = f"T{space_from.ufl_element()}->{space_to.ufl_element()}"
        if key not in self._shape_workspace:
            fenicsx_ii = _import_fenicsx_ii()
            mat, _, _ = fenicsx_ii.create_interpolation_matrix(
                space_from,
                space_to,
                red_op=fenicsx_ii.PointwiseTrace(space_to.mesh),
                tol=self.tol,
                use_petsc=self._use_petsc,
            )
            self._shape_workspace[key] = mat if self._use_petsc else _MatrixCSRWorkspace(mat)
        return self._shape_workspace[key]

    def _gradient_at_target_points(self, func_from: dolfinx.fem.Function) -> list[dolfinx.fem.Function]:
        r"""``grad(u)`` of the source function, evaluated at the target's interpolation points.

        Both shape terms are built from this one quantity. It is obtained in two exact steps
        rather than one: ``grad(u)`` is interpolated into a discontinuous space on the *source*
        mesh, which represents it without error, and that is then transferred to the target by
        the same point evaluation the forward interpolation uses. Going through a continuous
        space instead would average across cells and quietly smooth the derivative.

        The discontinuous space has the source element's own degree rather than one less. One
        less is exact on a simplex, where ``grad(P_k) = [DG_{k-1}]^d``, but not on a
        quadrilateral, where ``d/dx Q_k`` is still degree ``k`` in ``y``; the higher degree is
        exact on both.

        Returned one component at a time -- entry ``m`` holds ``grad(u_m)`` -- rather than as a
        single tensor-valued field, because ``fenicsx_ii.create_interpolation_matrix`` cannot
        build a transfer for a tensor-shaped space. Splitting it also means every component
        shares one cached matrix, since they all live in the same vector-valued space.

        Args:
            func_from: The source function, at its checkpointed values.

        Returns:
            One Function per target value component, on the target mesh, holding that
            component's gradient at the target points.
        """
        value_size = self._value_size()
        if "grad" not in self._shape_workspace:
            gdim = self.space_from.mesh.geometry.dim
            # Also high enough for the gradient of a geometry-space field, which the second-order
            # terms transfer through the same space.
            degree = max(self.space_from.ufl_element().embedded_superdegree, _coordinate_degree(self.space_from.mesh))
            self._shape_workspace["grad_from"] = dolfinx.fem.Function(
                dolfinx.fem.functionspace(self.space_from.mesh, ("DG", degree, (gdim,)))
            )
            self._shape_workspace["grad"] = [
                dolfinx.fem.Function(self._displacement_space()) for _ in range(value_size)
            ]
        grad_from = self._shape_workspace["grad_from"]
        grads = self._shape_workspace["grad"]
        mult = get_mult(
            self._transfer_matrix(grad_from.function_space, self._displacement_space()),
            transpose=False,
            accumulate=False,
        )
        for component, grad_to in enumerate(grads):
            expression = ufl.grad(func_from if value_size == 1 else func_from[component])
            grad_from.interpolate(
                dolfinx.fem.Expression(expression, get_interpolation_points(grad_from.function_space))
            )
            grad_from.x.scatter_forward()
            grad_to.x.array[:] = 0.0
            mult(grad_from.x, grad_to.x)
        return grads

    def _value_size(self) -> int:
        """How many components the target function has at each point."""
        return int(np.prod(self.space_to.ufl_element().reference_value_shape, dtype=int))

    def _displacement_space(self) -> dolfinx.fem.FunctionSpace:
        """The target-side space a displacement is evaluated into, ``(gdim,)``-valued."""
        if "disp_space" not in self._shape_workspace:
            self._shape_workspace["disp_space"] = _blocked_space(
                self.space_to.mesh, _base_element(self.space_to), (self.space_to.mesh.geometry.dim,)
            )
        return self._shape_workspace["disp_space"]

    def _target_motion_operator(self, grads: list[dolfinx.fem.Function]) -> MatrixFreeInterpolationOperator:
        r"""The map ``dx_B -> grad(u)(x_i) . dx_B(x_i)``, from the target mesh's geometry space.

        The target's interpolation points ride along with its own mesh, so a displacement of it
        moves every evaluation point. Since the result is again a point evaluation on the target
        mesh, this is an ordinary same-mesh interpolation of ``dot(grad(u), w)`` -- the operator
        {py:class}`~dolfinx_adjoint.blocks.interpolation.ExprInterpolationBlock` already uses.
        """
        assert self._mesh_to is not None
        if "target_op" not in self._shape_workspace:
            w = ufl.TrialFunction(self._mesh_to._ad_function_space())
            expression = (
                ufl.dot(grads[0], w) if self._value_size() == 1 else ufl.as_vector([ufl.dot(grad, w) for grad in grads])
            )
            self._shape_workspace["target_op"] = MatrixFreeInterpolationOperator(expression, self.space_to)
        return self._shape_workspace["target_op"]

    def _contract(self, values: np.ndarray, grads: list[dolfinx.fem.Function], transpose: bool) -> np.ndarray:
        r"""Contract with ``grad(u)(x_i)`` over the target's value components.

        Forward (``transpose=False``): a displacement sampled at the target points, shape
        ``(nodes, gdim)``, becomes ``dc_{i,m} = sum_k grad_m(x_i)_k s_{i,k}``. Reverse: an
        adjoint value on the target, shape ``(nodes, value_size)``, becomes
        ``sum_m lambda_{i,m} grad_m(x_i)_k``.
        """
        gdim = self.space_to.mesh.geometry.dim
        stacked = np.stack([grad.x.array.reshape(-1, gdim) for grad in grads], axis=1)
        if transpose:
            return np.einsum("im,imk->ik", values.reshape(-1, len(grads)), stacked).reshape(-1)
        return np.einsum("imk,ik->im", stacked, values.reshape(-1, gdim)).reshape(-1)

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
                tlm_input, dependency.output, prepared["grad"]
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
        """The transfer matrix, plus ``grad(u)`` at the target points when a mesh is involved."""
        prepared = {"matrix": self._get_interpolation_matrix(), "grad": None}
        if self._tracks_geometry and needs_geometry:
            prepared["grad"] = self._gradient_at_target_points(inputs[0])
        return prepared

    def _tangent_geometry_component(self, direction, mesh: Mesh, grad_to: dolfinx.fem.Function) -> np.ndarray:
        r"""The target dofs' response to displacing ``mesh`` by ``direction``.

        Two terms, which are the same expression with opposite signs when one mesh plays both
        roles -- interpolating a function onto its own mesh has no shape derivative, and this
        reproduces that by cancellation rather than by a special case.
        """
        scratch = _create_function(self.space_to)
        scratch.x.array[:] = 0.0
        if mesh is self._mesh_to:
            self._target_motion_operator(grad_to).mult(direction.x, scratch.x, accumulate=True)
        if mesh is self._mesh_from:
            sampled = dolfinx.fem.Function(self._displacement_space())
            sampled.x.array[:] = 0.0
            mult = get_mult(
                self._transfer_matrix(mesh._ad_function_space(), self._displacement_space()),
                transpose=False,
                accumulate=False,
            )
            mult(direction.x, sampled.x)
            scratch.x.array[:] -= self._contract(sampled.x.array, grad_to, transpose=False)
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

    def _adjoint_geometry_component(self, adj_vector, mesh: Mesh, grad_to: dolfinx.fem.Function):
        """The transpose of :py:meth:`_tangent_geometry_component`, into ``mesh``'s geometry space."""
        out_func = self._geometry_output.get(id(mesh))
        if out_func is None:
            out_func = _create_function(mesh._ad_function_space())
            self._geometry_output[id(mesh)] = out_func
        out_func.x.array[:] = 0.0

        if mesh is self._mesh_to:
            out_func.x.array[:] += self._reduced_transpose(
                self._target_motion_operator(grad_to), adj_vector, mesh._ad_function_space()
            )
        if mesh is self._mesh_from:
            sampled = dolfinx.fem.Function(self._displacement_space())
            sampled.x.array[:] = -self._contract(adj_vector.array, grad_to, transpose=True)
            mult = get_mult(
                self._transfer_matrix(mesh._ad_function_space(), self._displacement_space()),
                transpose=True,
                accumulate=True,
            )
            mult(sampled.x, out_func.x)
        return out_func.x

    # --- Adjoint ---

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        self._refresh_geometry()
        return self._prepared(inputs, self._involves_geometry(relevant_dependencies))

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        # The source function is always dependency 0; any tracked mesh was registered after it.
        if idx > 0:
            mesh = self.get_dependencies()[idx].output
            return self._adjoint_geometry_component(adj_inputs[0], mesh, prepared["grad"])

        if self._adj_output is None:
            self._adj_output = _create_function(self.space_from)

        out_func = self._adj_output
        out_func.x.array[:] = 0.0

        mult = get_mult(prepared["matrix"], transpose=True, accumulate=True)
        mult(adj_inputs[0], out_func.x)
        return out_func.x

    # --- Hessian ---

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        """The transfer matrix, ``grad(u)`` at the target points, and the second-order fields.

        With ``c_i = u(x_i)``, directions ``p_k = (u_k, sigma_k, dx_k)`` for the coefficient, the
        source mesh and the target mesh, and ``d_k = dx_k(x_i) - sigma_k(x_i)`` the motion of each
        target point *relative to* the source mesh, the second derivative is

            d2c[p1, p2] = d1^T H_u d2 - grad(u).(grad(sigma2) d1) - grad(u).(grad(sigma1) d2)
                          + grad(u1).d2 + grad(u2).d1,

        everything evaluated at ``x_i``. It is derived in the source reference cell, where the
        point is fixed by ``F_A(xi) = x_i``; the second derivatives of the source mapping cancel
        between the two terms that carry them, so this holds on curved source cells too. Which
        cell holds ``x_i`` contributes nothing: it is piecewise constant in the geometry. On one
        mesh ``d_k = 0`` and the whole thing vanishes, as interpolating onto your own mesh should.

        Here ``p1`` is the tangent-linear direction, fixed by the time this runs, so the terms
        that do not involve ``p2`` are gathered into one field per value component,
        ``q_m = H_m d1 - grad(sigma1)^T grad(u_m) + grad(u1_m)``, and
        :py:meth:`evaluate_hessian_component` applies the transpose with respect to ``p2``.
        """
        self._refresh_geometry()
        prepared = self._prepared(inputs, self._tracks_geometry)
        prepared["second"] = self._second_order_fields(inputs[0], prepared["grad"]) if self._tracks_geometry else None
        return prepared

    def _tlm_of(self, target) -> typing.Any:
        """The tangent-linear value of the dependency whose output is ``target``, if any."""
        for dependency in self.get_dependencies():
            if dependency.output is target:
                return dependency.tlm_value
        return None

    def _at_target_points(self, expression) -> np.ndarray:
        """A ``(gdim,)``-valued expression on the source mesh, at the target points: ``(nodes, gdim)``.

        Interpolated exactly into the discontinuous space on the source first, as in
        :py:meth:`_gradient_at_target_points`, then transferred by point evaluation.
        """
        gdim = self.space_to.mesh.geometry.dim
        grad_from = self._shape_workspace["grad_from"]
        grad_from.interpolate(dolfinx.fem.Expression(expression, get_interpolation_points(grad_from.function_space)))
        grad_from.x.scatter_forward()
        values = dolfinx.fem.Function(self._displacement_space())
        values.x.array[:] = 0.0
        mult = get_mult(
            self._transfer_matrix(grad_from.function_space, self._displacement_space()),
            transpose=False,
            accumulate=False,
        )
        mult(grad_from.x, values.x)
        return values.x.array.reshape(-1, gdim).copy()

    def _second_order_fields(self, func_from: dolfinx.fem.Function, grads: list) -> dict | None:
        """``d1`` and the ``q_m`` of :py:meth:`prepare_evaluate_hessian`, or ``None`` with no direction."""
        gdim = self.space_to.mesh.geometry.dim
        value_size = self._value_size()
        u_dot = self.get_dependencies()[0].tlm_value
        sigma = self._tlm_of(self._mesh_from) if self._mesh_from is not None else None
        target_motion = self._tlm_of(self._mesh_to) if self._mesh_to is not None else None
        if u_dot is None and sigma is None and target_motion is None:
            return None

        with stop_annotating():
            d1 = np.zeros_like(grads[0].x.array).reshape(-1, gdim)
            if target_motion is not None:
                sampled = dolfinx.fem.Function(self._displacement_space())
                sampled.interpolate(target_motion)  # same mesh: the target points' own motion
                d1 += sampled.x.array.reshape(-1, gdim)
            if sigma is not None:
                sampled = dolfinx.fem.Function(self._displacement_space())
                sampled.x.array[:] = 0.0
                mult = get_mult(
                    self._transfer_matrix(self._mesh_from._ad_function_space(), self._displacement_space()),
                    transpose=False,
                    accumulate=False,
                )
                mult(sigma.x, sampled.x)
                d1 -= sampled.x.array.reshape(-1, gdim)
            # Collective: which branches run below involves scatters, so every rank must agree --
            # a rank whose points happen not to move must still take the same path.
            moves = self.space_to.mesh.comm.allreduce(bool(np.any(d1)), op=MPI.LOR)

            grad_sigma = None
            if sigma is not None:  # grad_sigma[i, r, l] = d(sigma_r)/dx_l at x_i
                grad_sigma = np.stack([self._at_target_points(ufl.grad(sigma[r])) for r in range(gdim)], axis=1)
            q = []
            for m, grad_to in enumerate(grads):
                u_m = func_from if value_size == 1 else func_from[m]
                g_m = grad_to.x.array.reshape(-1, gdim)
                q_m = np.zeros_like(g_m)
                if moves:
                    hessian = np.stack(
                        [self._at_target_points(ufl.grad(ufl.grad(u_m)[k])) for k in range(gdim)], axis=1
                    )
                    q_m += np.einsum("ikl,il->ik", hessian, d1)
                if grad_sigma is not None:
                    q_m -= np.einsum("irl,ir->il", grad_sigma, g_m)
                if u_dot is not None:
                    q_m += self._at_target_points(ufl.grad(u_dot if value_size == 1 else u_dot[m]))
                field = dolfinx.fem.Function(self._displacement_space())
                field.x.array[:] = q_m.reshape(-1)
                q.append(field)
        return {"d1": d1, "moves": moves, "q": q}

    def _gradient_transpose(self, weights: np.ndarray, space, component: int | None) -> np.ndarray:
        """Transpose of ``f -> [grad(f_c)(x_i) . w_i]``, applied to ``weights`` ``(nodes, gdim)``, into ``space``.

        The forward map is the exact interpolation of ``grad(f_c)`` into the discontinuous space on
        the source followed by the point-evaluation transfer, so its transpose is the transfer's
        transpose followed by the matrix-free gradient operator's, with ghosts reduced.
        """
        grad_space = self._shape_workspace["grad_from"].function_space
        weighted = dolfinx.fem.Function(self._displacement_space())
        weighted.x.array[:] = weights.reshape(-1)
        pulled = dolfinx.fem.Function(grad_space)
        pulled.x.array[:] = 0.0
        get_mult(self._transfer_matrix(grad_space, self._displacement_space()), transpose=True, accumulate=False)(
            weighted.x, pulled.x
        )
        key = f"R{space.ufl_element()}:{component}"
        if key not in self._shape_workspace:
            trial = ufl.TrialFunction(space)
            self._shape_workspace[key] = MatrixFreeInterpolationOperator(
                ufl.grad(trial if component is None else trial[component]), grad_space
            )
        return self._reduced_transpose(self._shape_workspace[key], pulled.x, space)

    def evaluate_hessian_component(
        self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None
    ):
        hessian_input = getattr(hessian_inputs[0], "x", hessian_inputs[0])
        adj_input = getattr(adj_inputs[0], "x", adj_inputs[0])
        second = prepared["second"]
        gdim = self.space_to.mesh.geometry.dim
        value_size = self._value_size()

        if idx > 0:
            mesh = self.get_dependencies()[idx].output
            out = self._adjoint_geometry_component(hessian_input, mesh, prepared["grad"])
            if second is None:
                return out
            lam = adj_input.array.reshape(-1, value_size)
            if mesh is self._mesh_to:
                w = ufl.TrialFunction(mesh._ad_function_space())
                q = second["q"]
                operator = MatrixFreeInterpolationOperator(
                    ufl.dot(q[0], w) if value_size == 1 else ufl.as_vector([ufl.dot(q_m, w) for q_m in q]),
                    self.space_to,
                )
                out.array[:] += self._reduced_transpose(operator, adj_input, mesh._ad_function_space())
            if mesh is self._mesh_from:
                weighted = dolfinx.fem.Function(self._displacement_space())
                weighted.x.array[:] = -sum(
                    lam[:, m, None] * q_m.x.array.reshape(-1, gdim) for m, q_m in enumerate(second["q"])
                ).reshape(-1)
                get_mult(
                    self._transfer_matrix(mesh._ad_function_space(), self._displacement_space()),
                    transpose=True,
                    accumulate=True,
                )(weighted.x, out)
                if second["moves"]:
                    grads = [g.x.array.reshape(-1, gdim) for g in prepared["grad"]]
                    for k in range(gdim):
                        weights = sum(lam[:, m, None] * g[:, k, None] * second["d1"] for m, g in enumerate(grads))
                        out.array[:] -= self._gradient_transpose(weights, mesh._ad_function_space(), k)
            out.scatter_forward()
            return out

        if self._hessian_output is None:
            self._hessian_output = _create_function(self.space_from)

        out_func = self._hessian_output
        out_func.x.array[:] = 0.0

        mult = get_mult(prepared["matrix"], transpose=True, accumulate=True)
        mult(hessian_input, out_func.x)
        if second is not None and second["moves"]:
            lam = adj_input.array.reshape(-1, value_size)
            for m in range(value_size):
                out_func.x.array[:] += self._gradient_transpose(
                    lam[:, m, None] * second["d1"], self.space_from, None if value_size == 1 else m
                )
        out_func.x.scatter_forward()
        return out_func.x
