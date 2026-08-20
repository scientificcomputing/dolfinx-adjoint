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


def is_real_scalar_func(func: dolfinx.fem.Function) -> bool:
    """Check if a function is a real scalar function.

    Args:
        func: The function to check.
    """
    return func.ufl_element().is_real and func.ufl_shape == ()


def create_function_with_special_vector(func: _Function, name: str | None = None) -> _Function:
    """Create a new function with the same function space as `func` but with a special vector for adjoint computations.

    Args:
        func: The original function from which to derive the new function.

    Returns:
        A new function with the same function space as `func` but with a special vector for adjoint computations.
    """
    name = name or f"{func.name}_special_vector"
    vec = _vector(func.x.index_map, func.x.block_size, func.function_space, dtype=func.x.array.dtype)
    return _Function(func.function_space, x=vec, annotate=False, name=name)


class FunctionAssignBlock(Block):
    """Block for assigning data directly to a :py:class:`dolfinx_adjoint.Function` on the tape.

    This block handles the assignment of a linear combination of ":py:class:`dolfinx_adjoint.Function` objects
    or constants to a target :py:class:`dolfinx_adjoint.Function`.

    Args:
        other: The right-hand side of the assignment, which can be a :py:class:`dolfinx_adjoint.Function`,
            a :py:class:`dolfinx_adjoint.Constant`, or a linear combination of :py:class:`dolfinx_adjoint.Function`
            objects.
        func: The target :py:class:`dolfinx_adjoint.Function` to which the assignment is made.
        ad_block_tag: Optional tag for identifying the block in the adjoint tape.
            If not provided, a default tag will be generated.
    """

    _working_memory: list[_Function]
    _one: _Function  # Array for storing the value 1.0 for adjoint of broadcast operations

    def __init__(
        self,
        other: np.inexact | int | float | _Function | ufl.core.expr.Expr,
        func: _Function,
        ad_block_tag: typing.Optional[str] = None,
    ):
        super().__init__(ad_block_tag=ad_block_tag)

        # Allocate working memory for adjoint computations
        self._working_memory = []
        for i in range(2):
            self._working_memory.append(create_function_with_special_vector(func, name=f"working_memory_{i}"))

        # Extract dependencies
        self.other = None
        self.expr = None

        if isinstance(other, (float, int)) and not isinstance(other, OverloadedType):
            other = AdjFloat(other)

        if isinstance(other, OverloadedType):
            self.add_dependency(other, no_duplicates=True)
            # If the dependency is a scalar broadcast, allocate the ones vector
            if isinstance(other, AdjFloat):
                self._one = _Function(func.function_space, name="one", annotate=False)
                self._one.x.array[:] = 1.0
            elif isinstance(other, _Function) and is_real_scalar_func(other):
                self._working_memory.append(create_function_with_special_vector(other, name="working_memory_2"))
                self._one = _Function(func.function_space, name="one", annotate=False)
                self._one.x.array[:] = 1.0
        else:
            self.expr = other

            # Extract linear combination
            assert isinstance(other, ufl.core.expr.Expr), f"Expected UFL expression, got {type(other)}"
            lin_comb = extract_linear_combination(other)
            if len(lin_comb) == 0:
                raise ValueError("No linear combination found in the expression.")
            for op in traverse_unique_terminals(other):
                if isinstance(op, OverloadedType):
                    self.add_dependency(op, no_duplicates=True)

            # Allocate extra memory for adjoint computations if any of the operands are real functions
            for op in traverse_unique_terminals(other):
                if isinstance(op, _Function) and is_real_scalar_func(op):
                    self._working_memory.append(create_function_with_special_vector(op, name="working_memory_2"))
                    break

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

    @classmethod
    def _compute_adjoint_of_broadcast(
        cls, input: dolfinx.la.Vector | npt.NDArray | float | int, one: _Function
    ) -> float | int:
        """
        Computes the adjoint of a broadcast operation into an R^N vector, which is simply the sum of the input values.
        """
        # Adjoint of a broadcast is just a sum
        if isinstance(input, dolfinx.la.Vector):
            return dolfinx.cpp.la.inner_product(input._cpp_object, one.x._cpp_object)  # type: ignore[arg-type]
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
            if isinstance(bo, AdjFloat):
                return self._compute_adjoint_of_broadcast(adj_inputs[0], self._one)
            elif isinstance(bo, dolfinx.fem.Function):
                if is_real_scalar_func(bo):
                    # Adjoint of a broadcast into a real function (constant stored as Function)
                    self._working_memory[2].x.array[0] = self._compute_adjoint_of_broadcast(adj_inputs[0], self._one)
                    return self._working_memory[2].x
                if bo.function_space != prepared.function_space:
                    raise ValueError(
                        "Function spaces of the block variable and prepared function must match for adjoint evaluation."
                    )
                self._working_memory[0].x.array[:] = adj_inputs[0].array[:]
                return self._working_memory[0].x
            elif isinstance(bo, dolfinx.fem.Constant):
                raise NotImplementedError(
                    "Adjoint for Constant assignment not implemented, use dolfinx_adjoint.Constant instead."
                )
            else:
                raise NotImplementedError(f"Adjoint for {block_variable=} not implemented.")
        else:
            # Linear combination
            expr, adj_input_func = prepared
            if isinstance(bo, dolfinx.fem.Function) and bo.function_space == adj_input_func.function_space:
                # Differentiate with respect to one of the input functions
                diff_expr = ufl.algorithms.expand_derivatives(
                    ufl.derivative(expr, block_variable.saved_output, adj_input_func)
                )
                assign_linear_combination(diff_expr, self._working_memory[0])
                return self._working_memory[0].x
            elif isinstance(bo, dolfinx.fem.Function) and is_real_scalar_func(bo):
                # Differentiate with respect to a real function (constant stored as Function)
                # Create a perturbation direction in the Real space (value = 1.0)
                assert len(self._working_memory) == 3, "Working memory not allocated for real function adjoint."
                direction = self._working_memory[2]
                direction.x.array[0] = 1.0

                # Differentiate expr w.r.t 'bo' in that direction
                diff_expr = ufl.algorithms.expand_derivatives(
                    ufl.derivative(expr, block_variable.saved_output, direction)
                )

                # Evaluate the derivative at the DOFs of the target space V
                diff_eval = self._working_memory[1]
                assign_linear_combination(diff_expr, diff_eval)

                # Chain rule: dot product of (dz/dr) and adjoint inputs (bar_u)
                self._working_memory[2].x.array[0] = dolfinx.cpp.la.inner_product(
                    diff_eval.x._cpp_object, adj_input_func.x._cpp_object
                )
                return self._working_memory[2].x
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
        dudm = self._working_memory[0]
        dudm.x.array[:] = 0.0
        dudmi = self._working_memory[1]
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
