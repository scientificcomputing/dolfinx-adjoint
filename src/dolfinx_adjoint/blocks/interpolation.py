from __future__ import annotations

import typing
import weakref
from typing import Callable

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from pyadjoint import Block, OverloadedType
from pyadjoint.tape import stop_annotating
from ufl.algorithms.analysis import traverse_unique_terminals

from ..compat import get_interpolation_points
from ..types.function import Function, _create_function
from ..utils import unroll_dofmap

if typing.TYPE_CHECKING:
    from petsc4py import PETSc


def _import_scifem():
    """Import scifem lazily: only ExprInterpolationBlock needs it, so importing
    dolfinx_adjoint (or using InterpolationBlock) must not require it to be installed.
    Replace with lazy import once we are at 3.15:
    https://peps.python.org/pep-0810/
    """
    try:
        import scifem
    except ImportError as e:
        raise ImportError("scifem is required to interpolate a UFL expression: pip install scifem") from e
    return scifem


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

            workspace.row_vec.array[:] = 0.0
            workspace.col_vec.array[:] = 0.0

            if transpose:
                workspace.row_vec.array[:in_size_local] = v_in.array[:in_size_local]
                workspace.row_vec.scatter_forward()

                workspace.mat.mult(workspace.row_vec, workspace.col_vec, transpose=True)
                if accumulate:
                    v_out.array[:out_size_local] += workspace.col_vec.array[:out_size_local]
                else:
                    v_out.array[:out_size_local] = workspace.col_vec.array[:out_size_local]
            else:
                workspace.col_vec.array[:in_size_local] = v_in.array[:in_size_local]
                workspace.col_vec.scatter_forward()

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
                # PETSc handles parallel MPI ghost fetching entirely internally.
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
        self._adj_output: Function | None = None
        self._tlm_output: Function | None = None
        self._hessian_output: Function | None = None

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
            self._adj_output = _create_function(self.space_from)

        # Action of the adjoint: A^T * adj_input
        self._adj_output.x.array[:] = 0.0  # Reset the output vector before accumulation
        adj_input.x.scatter_forward()
        mult = get_mult(mat, transpose=True)
        mult(adj_input.x, self._adj_output.x)
        return self._adj_output.x

    # --- Tangent Linear Model (TLM) ---

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return self._get_interpolation_matrix()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        tlm_input = tlm_inputs[0]
        if tlm_input is None:
            return None

        mat = prepared

        if self._tlm_output is None:
            self._tlm_output = _create_function(self.space_to)

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
            self._hessian_output = _create_function(self.space_from)

        # Action of the adjoint on the incoming Hessian sensitivity
        hessian_input.x.scatter_forward()
        self._hessian_output.x.array[:] = 0.0  # Reset the output vector before accumulation
        mult = get_mult(mat, transpose=True)
        mult(hessian_input.x, self._hessian_output.x)
        return self._hessian_output.x

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

    def _assemble_operator(self, idx: int, inputs: list | None = None):
        """
        Assemble the interpolation operator (but not into a sparse matrix).
        """
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
        return MatrixFreeInterpolationOperator(dE, self.space_to)

    # --- Adjoint ---

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        operators = {}
        for idx, _dep in relevant_dependencies:
            operators[idx] = self._assemble_operator(idx, inputs)
        return operators

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = adj_inputs[0]
        operator = prepared[idx]

        if idx not in self._adj_output:
            self._adj_output[idx] = _create_function(self._deps[idx].function_space)
        out_func = self._adj_output[idx]

        out_func.x.array[:] = 0.0
        operator.mult_transpose(adj_input.x, out_func.x)
        out_func.x.scatter_reverse(dolfinx.la.InsertMode.add)
        out_func.x.scatter_forward()
        return out_func.x

    # --- Tangent Linear Model (TLM) ---

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        # 1. Substitute current optimization step's inputs into the expression
        if inputs is not None:
            replace_map = {self._deps[i]: inputs[i] for i in range(len(self._deps))}
            current_expr = ufl.replace(self.expr, replace_map)
        else:
            inputs = self._deps
            current_expr = self.expr

        # 2. Build the total directional derivative matrix-free: dE_total = sum( dE/dx_j * \delta x_j )
        dE_total = None
        for i, tlm_val in enumerate(tlm_inputs):
            if tlm_val is not None:
                target_dep = inputs[i]
                term = ufl.derivative(current_expr, target_dep, tlm_val)
                dE_total = term if dE_total is None else dE_total + term

        # 3. Force UFL to evaluate the calculus before compiling the Expression
        if dE_total is None:
            return None
        else:
            return dolfinx.fem.Expression(dE_total, get_interpolation_points(self.space_to))

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        if self._tlm_output is None:
            self._tlm_output = _create_function(self.space_to)

        out_func = self._tlm_output
        out_func.x.array[:] = 0.0
        if prepared is None:
            return out_func  # No TLM contribution if the directional derivative is zero

        # Matrix-free evaluation of the TLM
        with stop_annotating():
            out_func.interpolate(prepared)
        out_func.x.scatter_forward()
        return out_func

    # --- Hessian ---
    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        operators = {}

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
        for idx, _dep in relevant_dependencies:
            # The standard Jacobian (J)
            J_op = self._assemble_operator(idx, inputs)

            # Matrix-free Curvature (H)
            H_op = None
            if dE_total is not None:
                target_dep_i = inputs[idx]
                if hasattr(target_dep_i, "function_space"):
                    V_in = target_dep_i.function_space
                    du = ufl.TrialFunction(V_in)
                    d2E = ufl.derivative(dE_total, target_dep_i, du)
                    # ufl.derivative returns a lazy, unexpanded CoefficientDerivative node
                    # that formally references `du` regardless of whether the expanded
                    # expression actually depends on it (e.g. `dE_total` linear in
                    # target_dep_i, as for a bare-coefficient expr -- its second
                    # derivative is identically zero, but the *unexpanded* node still
                    # reports one argument, previously causing a spurious H_op to be
                    # compiled from a mesh-less zero expression). Expand derivatives
                    # first so the argument count (and isinstance-zero check) reflect
                    # the true, simplified expression.
                    d2E = ufl.algorithms.apply_derivatives.apply_derivatives(d2E)

                    if not isinstance(d2E, (int, float)):
                        args = ufl.algorithms.extract_arguments(d2E)
                        if len(args) == 1:
                            H_op = MatrixFreeInterpolationOperator(d2E, self.space_to)
                        elif len(args) > 1:
                            raise ValueError(
                                f"Second derivative of expression with respect to {target_dep_i}"
                                + f" has more than one argument: {args}"
                            )

            operators[idx] = (J_op, H_op)

        return operators

    def evaluate_hessian_component(
        self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None
    ):
        hessian_input = hessian_inputs[0]
        adj_input = adj_inputs[0]

        J_op, H_op = prepared[idx]

        if idx not in self._hessian_output:
            self._hessian_output[idx] = _create_function(self._deps[idx].function_space)
        out_func = self._hessian_output[idx]
        out_func.x.array[:] = 0.0

        # Term 1: J^T * \delta y^*
        if J_op is not None:
            J_op.mult_transpose(hessian_input.x, out_func.x, accumulate=True)

        # Term 2: H^T * y^*
        if H_op is not None:
            H_op.mult_transpose(adj_input.x, out_func.x, accumulate=True)
        out_func.x.scatter_reverse(dolfinx.la.InsertMode.add)
        out_func.x.scatter_forward()
        return out_func.x

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


class MatrixFreeInterpolationOperator:
    """A matrix-free operator that applies cell-local UFL expression interpolations."""

    def __init__(self, expr: ufl.core.expr.Expr, space_to: dolfinx.fem.FunctionSpace):

        args = ufl.algorithms.extract_arguments(expr)
        if len(args) != 1:
            raise ValueError("MatrixFreeInterpolationOperator only supports expressions with a single argument.")
        space_from = args[0].ufl_function_space()

        # Unroll DOF maps using your optimized function
        Q_dofmap = unroll_dofmap(space_to.dofmap.list, space_to.dofmap.bs)
        V_dofmap = unroll_dofmap(space_from.dofmap.list, space_from.dofmap.bs)

        # Get raw cell-local interpolation data from scifem
        scifem = _import_scifem()
        raw_data = scifem.prepare_interpolation_data(expr, space_to)

        num_cells = space_to.mesh.topology.index_map(space_to.mesh.topology.dim).size_local
        self.num_rows_local = space_to.dofmap.index_map.size_local * space_to.dofmap.bs

        # Reshape the 1D permuted matrix from scifem back into the cell-wise tensor
        A_local = np.copy(raw_data).reshape(num_cells, Q_dofmap.shape[1], V_dofmap.shape[1])

        # 1. Identify the unique, first appearance of each local Q DOF
        flat_Q = Q_dofmap.ravel()
        unique_Q, first_indices = np.unique(flat_Q, return_index=True)

        # 2. Keep only the locally owned DOFs (which are guaranteed to be 0 ... num_rows_local - 1)
        owned_mask = unique_Q < self.num_rows_local
        owned_first_indices = first_indices[owned_mask]

        # Convert flat indices back to (cell, local_q) coordinates
        valid_cells = owned_first_indices // Q_dofmap.shape[1]
        valid_q_local = owned_first_indices % Q_dofmap.shape[1]

        # 3. Store the PRUNED data: Shape is now exactly (num_local_Q, num_v_per_cell)
        # Because unique_Q is sorted 0 to num_rows_local-1, row `i` matches owned DOF `i` exactly!
        self.V_indices = V_dofmap[valid_cells, :]
        self.A_reduced = A_local[valid_cells, valid_q_local, :]

    def mult(self, v_in: dolfinx.la.Vector, v_out: dolfinx.la.Vector, accumulate: bool = False):
        """Forward action (TLM): A * v_in -> v_out"""
        v_in.scatter_forward()

        # x_local shape: (num_local_Q, num_v_per_cell)
        x_local = v_in.array[self.V_indices]

        # Row-wise dot product
        y_local = np.sum(self.A_reduced * x_local, axis=1)

        if not accumulate:
            v_out.array[:] = 0.0

        # Direct slice assignment! No advanced indexing array needed.
        v_out.array[: self.num_rows_local] += y_local

    def mult_transpose(self, v_in: dolfinx.la.Vector, v_out: dolfinx.la.Vector, accumulate: bool = False):
        """Adjoint action: A^T * v_in -> v_out (Pullback)"""
        v_in.scatter_forward()

        # Extract corresponding target DOFs. Direct slice!
        y_in = v_in.array[: self.num_rows_local]

        # Scale every coefficient row by the target DOF value
        # scaled_A shape: (num_local_Q, num_v_per_cell)
        scaled_A = self.A_reduced * y_in[:, None]

        if not accumulate:
            v_out.array[:] = 0.0

        # V_indices has overlapping/shared DOFs across cells, so np.add.at is required
        np.add.at(v_out.array, self.V_indices, scaled_A)
