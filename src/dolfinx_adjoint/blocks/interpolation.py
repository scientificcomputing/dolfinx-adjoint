from __future__ import annotations

import typing
import weakref
from typing import Callable

import dolfinx
import dolfinx.fem.petsc
import scifem
import ufl
from pyadjoint import Block, OverloadedType
from pyadjoint.tape import stop_annotating
from ufl.algorithms.analysis import traverse_unique_terminals

from ..compat import get_interpolation_points

if typing.TYPE_CHECKING:
    from petsc4py import PETSc

# Cache of assembled interpolation matrices, keyed by (id(space_from), id(space_to),
# use_petsc), shared across all InterpolationBlocks to avoid redundant matrix assembly
# (and, for PETSc matrices, MPI communicator exhaustion) when the same pair of spaces is
# interpolated between many times (e.g. across optimization iterations).
#
# Keying on id() would normally risk a stale/wrong entry once a FunctionSpace is garbage
# collected and its id is reused by an unrelated object (ids are only unique among
# simultaneously-alive objects). We close that hole with weakref.finalize: each space
# that contributes to a cache key gets a finalizer that purges every entry mentioning
# its id. Finalizers run at the point the object is actually deallocated, which is
# necessarily before CPython can hand that id out again, so a cache hit always
# corresponds to spaces that are still alive.
_INTERPOLATION_MATRIX_CACHE: dict[tuple[int, int, bool], "_MatrixCSRWorkspace | PETSc.Mat"] = {}
_CACHE_KEYS_BY_SPACE_ID: dict[int, set[tuple[int, int, bool]]] = {}


def _purge_cache_entries_for(space_id: int) -> None:
    for key in _CACHE_KEYS_BY_SPACE_ID.pop(space_id, ()):
        _INTERPOLATION_MATRIX_CACHE.pop(key, None)


def _register_cache_key(space: dolfinx.fem.FunctionSpace, key: tuple[int, int, bool]) -> None:
    space_id = id(space)
    if space_id not in _CACHE_KEYS_BY_SPACE_ID:
        weakref.finalize(space, _purge_cache_entries_for, space_id)
    _CACHE_KEYS_BY_SPACE_ID.setdefault(space_id, set()).add(key)


def get_dependency_index(dependencies, dep_or_bv) -> int:
    """Safely finds the index of a dependency using object identity instead of UFL equality."""

    # Unpack PyAdjoint's (input_index, BlockVariable) tuple if present
    if isinstance(dep_or_bv, tuple) and len(dep_or_bv) == 2 and isinstance(dep_or_bv[0], int):
        dep_or_bv = dep_or_bv[1]

    target = getattr(dep_or_bv, "output", dep_or_bv)
    target_bv = getattr(dep_or_bv, "block_variable", None) or getattr(target, "block_variable", None)

    for i, d in enumerate(dependencies):
        if d is target or id(d) == id(target):
            return i
        d_bv = getattr(d, "block_variable", None)
        if d_bv is not None and target_bv is not None and d_bv is target_bv:
            return i
        if getattr(d, "_cpp_object", None) is not None and getattr(d, "_cpp_object") is getattr(
            target, "_cpp_object", None
        ):
            return i

    raise ValueError(f"Could not locate dependency index for {dep_or_bv}")


class _MatrixCSRWorkspace:
    """A dolfinx.la.MatrixCSR paired with pre-allocated working vectors.

    dolfinx.la.MatrixCSR.mult requires vectors built from its own index maps
    as scratch space. Wrapping them here (built once, alongside the matrix)
    means they can be reused across repeated matrix-vector multiplications
    without allocating on every adjoint/TLM/Hessian evaluation, and without
    monkey-patching dynamic attributes onto the matrix object itself.
    """

    def __init__(self, mat: dolfinx.la.MatrixCSR):
        self.mat = mat
        self.row_vec = dolfinx.la.vector(mat.index_map(0), mat.block_size[0], dtype=mat.data.dtype)
        self.col_vec = dolfinx.la.vector(mat.index_map(1), mat.block_size[1], dtype=mat.data.dtype)


def get_mult(
    mat: "PETSc.Mat" | _MatrixCSRWorkspace,
    transpose: bool = False,
    accumulate: bool = False,
) -> Callable[[dolfinx.la.Vector, dolfinx.la.Vector], None]:
    """Return a function that performs matrix-vector multiplication, optionally accumulating results."""
    if isinstance(mat, _MatrixCSRWorkspace):
        workspace = mat

        def mult(v_in: dolfinx.la.Vector, v_out: dolfinx.la.Vector):
            in_size_local = v_in.index_map.size_local * v_in.block_size
            out_size_local = v_out.index_map.size_local * v_out.block_size
            # Zero the full working arrays (including ghosts) before each use
            # to prevent double-counting in parallel.
            workspace.row_vec.array[:] = 0.0
            workspace.col_vec.array[:] = 0.0
            if transpose:
                workspace.row_vec.array[:in_size_local] = v_in.array[:in_size_local]
                workspace.row_vec.scatter_forward()  # Ensure ghost values are updated before multiplication
                workspace.col_vec.array[:out_size_local] = 0.0
                workspace.mat.mult(workspace.row_vec, workspace.col_vec, transpose=True)
                if accumulate:
                    v_out.array[:out_size_local] += workspace.col_vec.array[:out_size_local]
                else:
                    v_out.array[:out_size_local] = workspace.col_vec.array[:out_size_local]
            else:
                workspace.row_vec.array[:out_size_local] = 0.0
                workspace.col_vec.array[:in_size_local] = v_in.array[:in_size_local]
                workspace.col_vec.scatter_forward()  # Ensure ghost values are updated before multiplication
                workspace.mat.mult(workspace.col_vec, workspace.row_vec)
                if accumulate:
                    v_out.array[:out_size_local] += workspace.row_vec.array[:out_size_local]
                else:
                    v_out.array[:out_size_local] = workspace.row_vec.array[:out_size_local]
            v_out.scatter_forward()

        return mult
    elif dolfinx.has_petsc4py and dolfinx.has_petsc:
        from petsc4py import PETSc

        if isinstance(mat, PETSc.Mat):

            def mult(v_in: dolfinx.la.Vector, v_out: dolfinx.la.Vector):
                if transpose:
                    if accumulate:
                        mat.multTransposeAdd(v_in.petsc_vec, v_out.petsc_vec, v_out.petsc_vec)
                    else:
                        mat.multTranspose(v_in.petsc_vec, v_out.petsc_vec)
                else:
                    if accumulate:
                        mat.multAdd(v_in.petsc_vec, v_out.petsc_vec, v_out.petsc_vec)
                    else:
                        mat.mult(v_in.petsc_vec, v_out.petsc_vec)
                v_out.scatter_forward()

            return mult
        else:
            raise TypeError("Expected a PETSc.Mat when PETSc is available.")
    else:
        raise TypeError(f"Matrix type not supported. Expected dolfinx.la.MatrixCSR or PETSc.Mat, got {type(mat)=}.")


def _build_interpolation_matrix(
    space_from: dolfinx.fem.FunctionSpace, space_to: dolfinx.fem.FunctionSpace, use_petsc: bool = False
) -> _MatrixCSRWorkspace | "PETSc.Mat":
    """Assemble the interpolation matrix for a pair of spaces."""
    if use_petsc:
        petsc_mat = dolfinx.fem.petsc.interpolation_matrix(space_from, space_to)
        petsc_mat.assemble()
        return petsc_mat

    mat = dolfinx.fem.interpolation_matrix(space_from, space_to)
    mat.scatter_reverse()
    return _MatrixCSRWorkspace(mat)


def _get_interpolation_matrix(
    space_from: dolfinx.fem.FunctionSpace, space_to: dolfinx.fem.FunctionSpace, use_petsc: bool = False
) -> _MatrixCSRWorkspace | "PETSc.Mat":
    """Retrieve or assemble the interpolation matrix for a pair of spaces, cached for reuse."""
    key = (id(space_from), id(space_to), use_petsc)
    if key not in _INTERPOLATION_MATRIX_CACHE:
        _INTERPOLATION_MATRIX_CACHE[key] = _build_interpolation_matrix(space_from, space_to, use_petsc=use_petsc)
        _register_cache_key(space_from, key)
        _register_cache_key(space_to, key)
    return _INTERPOLATION_MATRIX_CACHE[key]


class InterpolationBlock(Block):
    """Block for interpolating a dolfinx.fem.Function into another space."""

    def __init__(
        self,
        func_from: dolfinx.fem.Function,
        func_to: dolfinx.fem.Function,
        ad_block_tag: str | None = None,
        petsc_mat: bool = False,
    ):
        super().__init__(ad_block_tag=ad_block_tag)
        self.space_from = func_from.function_space
        self.space_to = func_to.function_space
        self._use_petsc = petsc_mat

        self.add_dependency(func_from)

        # Initialize internal caches for outputs to avoid MPI communicator exhaustion
        self._adj_output: dolfinx.fem.Function | None = None
        self._tlm_output: dolfinx.fem.Function | None = None
        self._hessian_output: dolfinx.fem.Function | None = None

    def __str__(self):
        return "interpolate_function"

    def _get_interpolation_matrix(self) -> _MatrixCSRWorkspace | "PETSc.Mat":
        return _get_interpolation_matrix(self.space_from, self.space_to, use_petsc=self._use_petsc)

    # --- Adjoint ---

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        return self._get_interpolation_matrix()

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = adj_inputs[0]
        mat = prepared

        if self._adj_output is None:
            self._adj_output = dolfinx.fem.Function(self.space_from)

        # Action of the adjoint: A^T * adj_input
        self._adj_output.x.array[:] = 0.0  # Reset the output vector before accumulation
        adj_input.x.scatter_forward()
        mult = get_mult(mat, transpose=True)
        mult(adj_input.x, self._adj_output.x)
        return self._adj_output

    # --- Tangent Linear Model (TLM) ---

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return self._get_interpolation_matrix()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        tlm_input = tlm_inputs[0]
        if tlm_input is None:
            return None

        mat = prepared

        if self._tlm_output is None:
            self._tlm_output = dolfinx.fem.Function(self.space_to)

        # Forward Jacobian action: A * tlm_input
        tlm_input.x.scatter_forward()
        self._tlm_output.x.array[:] = 0.0  # Reset the output vector before accumulation
        mult = get_mult(mat, transpose=False)
        mult(tlm_input.x, self._tlm_output.x)
        return self._tlm_output

    # --- Hessian ---

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        return self._get_interpolation_matrix()

    def evaluate_hessian_component(
        self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None
    ):
        hessian_input = hessian_inputs[0]
        mat = prepared

        if self._hessian_output is None:
            self._hessian_output = dolfinx.fem.Function(self.space_from)

        # Action of the adjoint on the incoming Hessian sensitivity
        hessian_input.x.scatter_forward()
        self._hessian_output.x.array[:] = 0.0  # Reset the output vector before accumulation
        mult = get_mult(mat, transpose=True)
        mult(hessian_input.x, self._hessian_output.x)
        return self._hessian_output

    # --- Recompute (Forward Pass) ---

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return None

    def recompute_component(self, inputs, block_variable, idx, prepared):
        func_from = inputs[0]

        # Update the tape's actual output object in-place (rather than a
        # separately cached Function) so that any Python reference the user
        # is holding to the interpolated Function stays in sync after the
        # tape is replayed, matching FunctionAssignBlock's convention. Note
        # this only holds until the output is first used as a dependency of
        # another block: pyadjoint then freezes a private checkpoint copy
        # (see OverloadedType._ad_will_add_as_dependency), so the live
        # Python object can still go stale after that point. This is an
        # existing pyadjoint/dolfinx_adjoint-wide characteristic (identical
        # behavior in FunctionAssignBlock), not specific to interpolation.
        output = block_variable.saved_output
        output.interpolate(func_from)
        output.x.scatter_forward()
        return output


class ExprInterpolationBlock(Block):
    """Block for interpolating a UFL expression with runtime-evaluated Jacobians via scifem."""

    def __init__(
        self,
        expr: ufl.core.expr.Expr,
        func_to: dolfinx.fem.Function,
        ad_block_tag: str | None = None,
        petsc_mat: bool = False,
    ):
        super().__init__(ad_block_tag=ad_block_tag)
        self.expr = expr
        self.space_to = func_to.function_space
        self._use_petsc = petsc_mat

        self._deps = []
        for op in traverse_unique_terminals(self.expr):
            if isinstance(op, OverloadedType):
                self.add_dependency(op, no_duplicates=True)
                self._deps.append(op)

        self._adj_output: dict[int, dolfinx.fem.Function] = {}
        self._tlm_output: dolfinx.fem.Function | None = None
        self._hessian_output: dict[int, dolfinx.fem.Function] = {}

    def __str__(self):
        return f"interpolate_expression_{str(self.expr)}_to_{str(self.space_to)}"

    def _assemble_jacobian(self, idx: int, inputs: list | None = None):
        dep_func = self._deps[idx]
        V_in = dep_func.function_space

        if inputs is not None:
            replace_map = {self._deps[i]: inputs[i] for i in range(len(self._deps))}
            current_expr = ufl.replace(self.expr, replace_map)
            target_dep = replace_map[dep_func]
        else:
            current_expr = self.expr
            target_dep = dep_func

        du = ufl.TrialFunction(V_in)
        dE = ufl.derivative(current_expr, target_dep, du)
        dE = ufl.algorithms.apply_derivatives.apply_derivatives(dE)
        if self._use_petsc:
            mat = scifem.petsc_interpolation_matrix(dE, self.space_to)
        else:
            mat = _MatrixCSRWorkspace(scifem.interpolation_matrix(dE, self.space_to))

        return mat

    # --- Adjoint ---

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        matrices = {}
        for dep in relevant_dependencies:
            idx = get_dependency_index(self._deps, dep)
            matrices[idx] = self._assemble_jacobian(idx, inputs)
        return matrices

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = adj_inputs[0]
        dep_idx = get_dependency_index(self._deps, block_variable) if block_variable is not None else idx
        mat = prepared[dep_idx]

        if dep_idx not in self._adj_output:
            self._adj_output[dep_idx] = dolfinx.fem.Function(self._deps[dep_idx].function_space)
        out_func = self._adj_output[dep_idx]

        out_func.x.array[:] = 0.0
        mult = get_mult(mat, transpose=True, accumulate=False)
        mult(adj_input.x, out_func.x)

        return out_func

    # --- Tangent Linear Model (TLM) ---

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        matrices = {}
        for dep in self.get_dependencies():
            idx = get_dependency_index(self._deps, dep)
            matrices[idx] = self._assemble_jacobian(idx, inputs)
        return matrices

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        if self._tlm_output is None:
            self._tlm_output = dolfinx.fem.Function(self.space_to)
        out_func = self._tlm_output

        # Reset output vector to prepare for accumulation
        out_func.x.array[:] = 0.0

        # The TLM is the sum of the Jacobians applied to each perturbation. tlm_inputs is
        # aligned with self.get_dependencies(), not necessarily with self._deps, so the
        # dependency index into `prepared` is re-derived explicitly rather than assumed
        # positional, matching every other evaluate_*_component method in this class.
        for i, dep_bv in enumerate(self.get_dependencies()):
            tlm_input = tlm_inputs[i]
            if tlm_input is None:
                continue

            dep_idx = get_dependency_index(self._deps, dep_bv)
            mat = prepared[dep_idx]
            mult = get_mult(mat, transpose=False, accumulate=True)
            mult(tlm_input.x, out_func.x)

        return out_func

    # --- Hessian ---
    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        matrices = {}

        # 1. Substitute current optimization step's inputs into the expression
        if inputs is not None:
            replace_map = {self._deps[i]: inputs[i] for i in range(len(self._deps))}
            current_expr = ufl.replace(self.expr, replace_map)
        else:
            inputs = self._deps
            replace_map = {d: d for d in self._deps}
            current_expr = self.expr

        # 2. Build the total first-order directional derivative: dE_total = sum( dE/dx_j * \delta x_j )
        dE_total = None
        deps_bv = self.get_dependencies()

        for i, dep_bv in enumerate(deps_bv):
            tlm_val = dep_bv.tlm_value

            # Fallback for controls or unrecorded raw inputs
            if tlm_val is None and hasattr(inputs[i], "block_variable") and inputs[i].block_variable is not None:
                tlm_val = inputs[i].block_variable.tlm_value

            if tlm_val is not None:
                target_dep = inputs[i]
                term = ufl.derivative(current_expr, target_dep, tlm_val)
                dE_total = term if dE_total is None else dE_total + term

        # 3. Assemble both the Jacobian and the Hessian matrix for each dependency
        for dep in relevant_dependencies:
            idx = get_dependency_index(self._deps, dep)

            # Matrix 1: The standard Jacobian (J)
            J_mat = self._assemble_jacobian(idx, inputs)

            # Matrix 2: The non-linear Curvature Matrix (H)
            H_mat = None
            if dE_total is not None:
                target_dep_i = inputs[idx]

                # Check if it exposes a function space (Constants do not need TrialFunctions)
                if hasattr(target_dep_i, "function_space"):
                    V_in = target_dep_i.function_space
                    du = ufl.TrialFunction(V_in)

                    # Differentiate the directional derivative again to get the 2nd derivative
                    d2E = ufl.derivative(dE_total, target_dep_i, du)
                    d2E = ufl.algorithms.apply_derivatives.apply_derivatives(d2E)
                    # Scifem will crash if the expression is purely linear (d2E == 0)
                    if not isinstance(d2E, (int, float)):
                        args = ufl.algorithms.extract_arguments(d2E)
                        if len(args) > 0:
                            if self._use_petsc:
                                H_mat = scifem.petsc_interpolation_matrix(d2E, self.space_to)
                            else:
                                H_mat = _MatrixCSRWorkspace(scifem.interpolation_matrix(d2E, self.space_to))

            matrices[idx] = (J_mat, H_mat)

        return matrices

    def evaluate_hessian_component(
        self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None
    ):
        hessian_input = hessian_inputs[0]
        adj_input = adj_inputs[0]  # The incoming adjoint sensitivity (y*)

        dep_idx = get_dependency_index(self._deps, block_variable) if block_variable is not None else idx
        J_mat, H_mat = prepared[dep_idx]

        if dep_idx not in self._hessian_output:
            self._hessian_output[dep_idx] = dolfinx.fem.Function(self._deps[dep_idx].function_space)
        out_func = self._hessian_output[dep_idx]

        # Reset output vector to 0.0 before accumulation
        out_func.x.array[:] = 0.0

        # Term 1: J^T * \delta y^*
        mult_J = get_mult(J_mat, transpose=True, accumulate=True)
        mult_J(hessian_input.x, out_func.x)

        # Term 2: H^T * y^* (Only applied if the expression was non-linear)
        if H_mat is not None:
            mult_H = get_mult(H_mat, transpose=True, accumulate=True)
            mult_H(adj_input.x, out_func.x)

        return out_func

    # --- Recompute (Forward Pass) ---

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return None

    def recompute_component(self, inputs, block_variable, idx, prepared):
        replace_map = {self._deps[i]: inputs[i] for i in range(len(self._deps))}
        updated_expr = ufl.replace(self.expr, replace_map)

        # Update the tape's actual output object in-place, matching
        # InterpolationBlock.recompute_component and FunctionAssignBlock's convention.
        output = block_variable.saved_output
        with stop_annotating():
            compiled_expr = dolfinx.fem.Expression(updated_expr, get_interpolation_points(self.space_to))
            output.interpolate(compiled_expr)
            output.x.scatter_forward()

        return output
