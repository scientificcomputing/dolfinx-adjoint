from __future__ import annotations

import typing
import weakref
from typing import Callable

import dolfinx
import dolfinx.fem.petsc
from pyadjoint import Block

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
_INTERPOLATION_MATRIX_CACHE: dict[tuple[int, int, bool], "dolfinx.la.MatrixCSR | PETSc.Mat"] = {}
_CACHE_KEYS_BY_SPACE_ID: dict[int, set[tuple[int, int, bool]]] = {}


def _purge_cache_entries_for(space_id: int) -> None:
    for key in _CACHE_KEYS_BY_SPACE_ID.pop(space_id, ()):
        _INTERPOLATION_MATRIX_CACHE.pop(key, None)


def _register_cache_key(space: dolfinx.fem.FunctionSpace, key: tuple[int, int, bool]) -> None:
    space_id = id(space)
    if space_id not in _CACHE_KEYS_BY_SPACE_ID:
        weakref.finalize(space, _purge_cache_entries_for, space_id)
    _CACHE_KEYS_BY_SPACE_ID.setdefault(space_id, set()).add(key)


def attach_working_array(mat: dolfinx.la.MatrixCSR):
    """Attach working arrays to a dolfinx.la.MatrixCSR for efficient matrix-vector multiplication."""
    # mat._row_vec/_col_vec are monkey-patched on; cast to Any so mypy doesn't
    # flag the dynamic attributes.
    m = typing.cast(typing.Any, mat)
    if not hasattr(m, "_row_vec"):
        m._row_vec = dolfinx.la.vector(mat.index_map(0), mat.block_size[0], dtype=mat.data.dtype)
    if not hasattr(m, "_col_vec"):
        m._col_vec = dolfinx.la.vector(mat.index_map(1), mat.block_size[1], dtype=mat.data.dtype)
    m._row_vec.array[:] = 0.0
    m._col_vec.array[:] = 0.0


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
            m = typing.cast(typing.Any, mat)
            if transpose:
                # Prevent double-counting in parallel by zeroing ghosts of input vector
                # Calculate the exact number of local degrees of freedom
                m._row_vec.array[:in_size_local] = v_in.array[:in_size_local]
                m._row_vec.scatter_forward()  # Ensure ghost values are updated before multiplication
                m._col_vec.array[:out_size_local] = 0.0
                m.mult(m._row_vec, m._col_vec, transpose=True)
                v_out.array[:out_size_local] = m._col_vec.array[:out_size_local]
            else:
                # Prevent double-counting in parallel by zeroing ghosts of input vector
                # Calculate the exact number of local degrees of freedom
                m._row_vec.array[:out_size_local] = 0
                m._col_vec.array[:in_size_local] = v_in.array[:in_size_local]
                m._col_vec.scatter_forward()  # Ensure ghost values are updated before multiplication
                m.mult(m._col_vec, m._row_vec)
                v_out.array[:out_size_local] = m._row_vec.array[:out_size_local]
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


def _build_interpolation_matrix(
    space_from: dolfinx.fem.FunctionSpace, space_to: dolfinx.fem.FunctionSpace, use_petsc: bool = False
) -> dolfinx.la.MatrixCSR | "PETSc.Mat":
    """Assemble the interpolation matrix for a pair of spaces."""
    if use_petsc:
        petsc_mat = dolfinx.fem.petsc.interpolation_matrix(space_from, space_to)
        petsc_mat.assemble()
        return petsc_mat

    mat = dolfinx.fem.interpolation_matrix(space_from, space_to)
    mat.scatter_reverse()
    # The built in interpolation matrix requires two working arrays
    attach_working_array(mat)
    return mat


def _get_interpolation_matrix(
    space_from: dolfinx.fem.FunctionSpace, space_to: dolfinx.fem.FunctionSpace, use_petsc: bool = False
) -> dolfinx.la.MatrixCSR | "PETSc.Mat":
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

    def _get_interpolation_matrix(self) -> dolfinx.la.MatrixCSR | "PETSc.Mat":
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
