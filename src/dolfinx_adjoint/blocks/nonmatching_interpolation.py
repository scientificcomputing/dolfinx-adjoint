import dolfinx
from pyadjoint import Block
from pyadjoint.tape import stop_annotating
from ..types.function import _create_function
from .interpolation import _MatrixCSRWorkspace, get_mult


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

        # Output caches
        self._adj_output: dolfinx.fem.Function | None = None
        self._tlm_output: dolfinx.fem.Function | None = None
        self._hessian_output: dolfinx.fem.Function | None = None

        # Matrix cache
        self._matrix_workspace = None

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
        # because it doesn't need the matrix and is highly optimized.
        with stop_annotating():
            fenicsx_ii = _import_fenicsx_ii()
            if self._red_op is fenicsx_ii.PointwiseTrace:
                # If using the default PointwiseTrace, we can use the built-in method
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
        return self._get_interpolation_matrix()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        if tlm_inputs[0] is None:
            return None

        if self._tlm_output is None:
            self._tlm_output = _create_function(self.space_to)

        out_func = self._tlm_output
        out_func.x.array[:] = 0.0

        mult = get_mult(prepared, transpose=False, accumulate=True)
        mult(tlm_inputs[0].x, out_func.x)

        return out_func

    # --- Adjoint ---

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        return self._get_interpolation_matrix()

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if self._adj_output is None:
            self._adj_output = _create_function(self.space_from)

        out_func = self._adj_output
        out_func.x.array[:] = 0.0

        mult = get_mult(prepared, transpose=True, accumulate=True)
        mult(adj_inputs[0], out_func.x)
        return out_func.x

    # --- Hessian ---

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        return self._get_interpolation_matrix()

    def evaluate_hessian_component(
        self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None
    ):
        if self._hessian_output is None:
            self._hessian_output = _create_function(self.space_from)

        out_func = self._hessian_output
        out_func.x.array[:] = 0.0

        mult = get_mult(prepared, transpose=True, accumulate=True)
        mult(hessian_inputs[0].x, out_func.x)

        return out_func.x
