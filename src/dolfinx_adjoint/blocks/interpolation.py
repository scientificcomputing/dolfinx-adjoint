from __future__ import annotations

import typing
from typing import Callable

import dolfinx
from pyadjoint import Block
from pyadjoint.overloaded_type import create_overloaded_object

if typing.TYPE_CHECKING:
    from petsc4py import PETSc

# Global cache to prevent redundant matrix assembly across multiple blocks/iterations
_INTERPOLATION_MATRIX_CACHE: dict[tuple[int, int], dolfinx.la.MatrixCSR] = {}


def attach_working_array(mat: dolfinx.la.MatrixCSR):
    """Attach working arrays to a dolfinx.la.MatrixCSR for efficient matrix-vector multiplication."""
    if not hasattr(mat, "_row_vec"):
        mat._row_vec = dolfinx.la.vector(mat.index_map(0), mat.block_size[0], dtype=mat.data.dtype)
    if not hasattr(mat, "_col_vec"):
        mat._col_vec = dolfinx.la.vector(mat.index_map(1), mat.block_size[1], dtype=mat.data.dtype)
    mat._row_vec.array[:] = 0.0
    mat._col_vec.array[:] = 0.0


def get_mult(
    mat: "PETSc.Mat" | dolfinx.la.MatrixCSR,
    transpose: bool = False,  # type: ignore
) -> Callable[[dolfinx.la.Vector, dolfinx.la.Vector], None]:
    """Return a function that performs matrix-vector multiplication with the given matrix."""
    if isinstance(mat, dolfinx.la.MatrixCSR):

        def mult(v_in: dolfinx.la.Vector, v_out: dolfinx.la.Vector):
            # Need to use vectors from
            in_size_local = v_in.index_map.size_local * v_in.block_size
            out_size_local = v_out.index_map.size_local * v_out.block_size
            attach_working_array(mat)  # Ensure working arrays are attached
            if transpose:
                # Prevent double-counting in parallel by zeroing ghosts of input vector
                # Calculate the exact number of local degrees of freedom
                mat._row_vec.array[:in_size_local] = v_in.array[:in_size_local]
                mat._row_vec.scatter_forward()  # Ensure ghost values are updated before multiplication
                mat._col_vec.array[:out_size_local] = 0.0
                mat.mult(mat._row_vec, mat._col_vec, transpose=True)
                v_out.array[:out_size_local] = mat._col_vec.array[:out_size_local]
            else:
                # Prevent double-counting in parallel by zeroing ghosts of input vector
                # Calculate the exact number of local degrees of freedom
                mat._row_vec.array[:out_size_local] = 0
                mat._col_vec.array[:in_size_local] = v_in.array[:in_size_local]
                mat._col_vec.scatter_forward()  # Ensure ghost values are updated before multiplication
                mat.mult(mat._col_vec, mat._row_vec)
                v_out.array[:out_size_local] = mat._row_vec.array[:out_size_local]
            v_out.scatter_forward()

        return mult
    elif dolfinx.has_petsc4py and dolfinx.has_petsc:
        from petsc4py import PETSc

        if isinstance(mat, PETSc.Mat):

            def mult(v_in: dolfinx.la.Vector, v_out: dolfinx.la.Vector):
                if transpose:
                    mat.multTranspose(v_in.petsc_vec, v_out.petsc_vec)
                else:
                    mat.mult(v_in.petsc_vec, v_out.petsc_vec)
                v_out.scatter_forward()

            return mult
        else:
            raise TypeError("Expected a PETSc.Mat when PETSc is available.")
    else:
        raise TypeError("Matrix type not supported. Expected dolfinx.la.MatrixCSR or PETSc.Mat, got {type(mat)=}.")


def _get_interpolation_matrix(
    space_from: dolfinx.fem.FunctionSpace, space_to: dolfinx.fem.FunctionSpace, use_petsc: bool = False
) -> dolfinx.la.MatrixCSR | "PETSc.Mat":
    """Retrieve or compute the interpolation matrix for a pair of spaces."""
    key = (id(space_from), id(space_to))

    if key not in _INTERPOLATION_MATRIX_CACHE:
        if use_petsc:
            mat = dolfinx.fem.petsc.interpolation_matrix(space_from, space_to)
            mat.assemble()
        else:
            mat = dolfinx.fem.interpolation_matrix(space_from, space_to)
            mat.scatter_reverse()
            # The built in interpolation matrix requires two working arrays
            attach_working_array(mat)

        _INTERPOLATION_MATRIX_CACHE[key] = mat

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
        self._recompute_output: dolfinx.fem.Function | None = None

    def __str__(self):
        return "interpolate_function"

    # --- Adjoint ---

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        return _get_interpolation_matrix(self.space_from, self.space_to, use_petsc=self._use_petsc)

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
        return _get_interpolation_matrix(self.space_from, self.space_to, use_petsc=self._use_petsc)

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
        return _get_interpolation_matrix(self.space_from, self.space_to, use_petsc=self._use_petsc)

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

        if self._recompute_output is None:
            self._recompute_output = dolfinx.fem.Function(self.space_to)

        self._recompute_output.interpolate(func_from)
        self._recompute_output.x.scatter_forward()

        # Overload the object to ensure PyAdjoint tracks it properly
        output = create_overloaded_object(self._recompute_output)
        return output
