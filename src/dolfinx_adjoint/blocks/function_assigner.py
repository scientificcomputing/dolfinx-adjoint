import typing

import dolfinx
import numpy as np
import numpy.typing as npt
import ufl
from pyadjoint import AdjFloat, Block, OverloadedType
from ufl.corealg.traversal import traverse_unique_terminals
from ufl.formatting.ufl2unicode import ufl2unicode

from ..types.function import Function as _Function
from ..utils import assign_linear_combination, extract_linear_combination, function_from_vector
from ._vector import _vector


class FunctionAssignBlock(Block):
    """Block for assigning data directly to a `Function` on the tape.

    This block handles the assignment of a linear combination of `Function`s
    or constants to a target `Function`.
    """

    def __init__(
        self,
        other: np.inexact | int | float | _Function | ufl.core.expr.Expr,
        ad_block_tag: typing.Optional[str] = None,
    ):
        super().__init__(ad_block_tag=ad_block_tag)
        self.other = None
        self.expr = None
        if isinstance(other, OverloadedType):
            self.add_dependency(other, no_duplicates=True)
        elif isinstance(other, float) or isinstance(other, int):
            other = AdjFloat(other)
            self.add_dependency(other, no_duplicates=True)
        else:
            # Extract linear combination
            assert isinstance(other, ufl.core.expr.Expr), f"Expected UFL expression, got {type(other)}"
            lin_comb = extract_linear_combination(other)
            if len(lin_comb) == 0:
                raise ValueError("No linear combination found in the expression.")
            for op in traverse_unique_terminals(other):
                if isinstance(op, OverloadedType):
                    self.add_dependency(op, no_duplicates=True)
            self.expr = other

    def _replace_with_saved_output(self):
        if self.expr is None:
            return None
        replace_map = {}
        for dep in self.get_dependencies():
            replace_map[dep.output] = dep.saved_output
        return ufl.replace(self.expr, replace_map)

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        V = self.get_outputs()[0].output.function_space
        adj_input_func = function_from_vector(V, adj_inputs[0])

        if self.expr is None:
            return adj_input_func

        expr = self._replace_with_saved_output()
        return expr, adj_input_func

    def _compute_adjoint_of_broadcast(self, input: dolfinx.la.Vector | npt.NDArray | float | int) -> float | int:
        # Adjoint of a broadcast is just a sum
        if isinstance(input, dolfinx.la.Vector):
            one = dolfinx.la.vector(input.index_map, input.block_size, input.array.dtype)
            one.array[:] = 1
            return dolfinx.cpp.la.inner_product(input._cpp_object, one._cpp_object)  # type: ignore[arg-type]
        else:
            if hasattr(input, "sum"):
                return input.sum()
            else:
                # Catch the case where input is just a float
                return input

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        bo = block_variable.output
        if self.expr is None:
            assert len(adj_inputs) == 1
            if isinstance(func := bo, AdjFloat):
                return self._compute_adjoint_of_broadcast(adj_inputs[0])
            elif isinstance(func, dolfinx.fem.Function):
                assert func.function_space == prepared.function_space
                vec = _vector(
                    prepared.x.index_map, prepared.x.block_size, func.function_space, dtype=prepared.x.array.dtype
                )
                vec.array[:] = prepared.x.array[:]
                return vec
            elif isinstance(bo, dolfinx.fem.Constant):
                raise NotImplementedError(
                    "Adjoint for Constant assignment not implemented, use dolfinx_adjoint.Constant instead."
                )
            else:
                raise NotImplementedError(f"Adjoint for {block_variable=} not implemented.")
        else:
            # Linear combination
            expr, adj_input_func = prepared
            vec = _vector(
                bo.x.index_map,
                bo.x.block_size,
                bo.function_space,
                dtype=bo.x.array.dtype,
            )
            if isinstance(bo, dolfinx.fem.Function) and bo.function_space == adj_input_func.function_space:
                # Differentiate with respect to one of the input functions
                diff_expr = ufl.algorithms.expand_derivatives(
                    ufl.derivative(expr, block_variable.saved_output, adj_input_func)
                )
                temp_func = dolfinx.fem.Function(bo.function_space)
                assign_linear_combination(diff_expr, temp_func)
                vec.array[:] = temp_func.x.array[:]
                return vec
            elif isinstance(bo, dolfinx.fem.Function) and bo.ufl_element().is_real:
                # Differentiate with respect to a real function (constant stored as Function)
                # Create a perturbation direction in the Real space (value = 1.0)
                direction = dolfinx.fem.Function(bo.function_space)
                direction.x.array[0] = 1.0

                # Differentiate expr w.r.t 'bo' in that direction
                diff_expr = ufl.algorithms.expand_derivatives(
                    ufl.derivative(expr, block_variable.saved_output, direction)
                )

                # Evaluate the derivative at the DOFs of the target space V
                diff_eval = dolfinx.fem.Function(adj_input_func.function_space)
                assign_linear_combination(diff_expr, diff_eval)

                # Chain rule: dot product of (dz/dr) and adjoint inputs (bar_u)
                vec.array[0] = dolfinx.cpp.la.inner_product(diff_eval.x._cpp_object, adj_input_func.x._cpp_object)
                return vec
            else:
                raise NotImplementedError(f"Adjoint for {block_variable=} not implemented.")

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        if self.expr is None:
            return None

        return self._replace_with_saved_output()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        if self.expr is None:
            return tlm_inputs[0]
        expr = prepared
        dudm = dolfinx.fem.Function(block_variable.output.function_space)
        dudm.x.array[:] = 0.0
        dudmi = dolfinx.fem.Function(block_variable.output.function_space)
        for dep in self.get_dependencies():
            if dep.tlm_value:
                diff_expr = ufl.algorithms.expand_derivatives(ufl.derivative(expr, dep.saved_output, dep.tlm_value))
                assign_linear_combination(diff_expr, dudmi)
                dudm.x.array[:] += dudmi.x.array[:]

        return dudm

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        return self.prepare_evaluate_adj(inputs, hessian_inputs, relevant_dependencies)

    def evaluate_hessian_component(
        self,
        inputs,
        hessian_inputs,
        adj_inputs,
        block_variable,
        idx,
        relevant_dependencies,
        prepared=None,
    ):
        # Current implementation assumes lincom in hessian,
        # otherwise we need second-order derivatives here.
        return self.evaluate_adj_component(inputs, hessian_inputs, block_variable, idx, prepared)

    def prepare_recompute_component(self, inputs, relevant_outputs):
        if self.expr is None:
            return None
        return self._replace_with_saved_output()

    def recompute_component(self, inputs, block_variable, idx, prepared):
        if self.expr is None:
            prepared = inputs[0]

        # We should return the exact object instance to maintain C++ memory bindings
        # (especially for DirichletBCs), updating it in-place.
        output = block_variable.saved_output
        if isinstance(prepared, dolfinx.fem.Function):
            output.x.array[:] = prepared.x.array[:]
        elif isinstance(prepared, (float, int)):
            output.x.array[:] = prepared
        else:
            assign_linear_combination(prepared, output)
        return output

    def __str__(self):
        rhs = self.expr or self.other or self.get_dependencies()[0].output
        if isinstance(rhs, ufl.core.expr.Expr):
            rhs_str = ufl2unicode(rhs)
        else:
            rhs_str = str(rhs)
        return f"assign({rhs_str})"
