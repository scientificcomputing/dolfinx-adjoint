import typing

import dolfinx
import numpy as np
import ufl
from pyadjoint import AdjFloat, Block, OverloadedType
from ufl.formatting.ufl2unicode import ufl2unicode

from dolfinx_adjoint.utils import function_from_vector

from ._vector import _vector


class FunctionAssignBlock(Block):
    def __init__(
        self,
        other: typing.Union[np.inexact, int, float],
        func: dolfinx.fem.Function,
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
        elif not (isinstance(other, float) or isinstance(other, int)):
            raise NotImplementedError("This should eventually be supported")
            # # Assume that this is a point-wise evaluated UFL expression (firedrake only)
            # for op in traverse_unique_terminals(other):
            #     if isinstance(op, OverloadedType):
            #         self.add_dependency(op, no_duplicates=True)
            # self.expr = other
        else:
            raise NotImplementedError("We should not get here!")

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

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        if self.expr is None:
            if isinstance(block_variable.output, AdjFloat):
                # Adjoint of a broadcast is just a sum
                if isinstance(adj_inputs[0], dolfinx.la.Vector):
                    vec = adj_inputs[0]
                    one = dolfinx.la.vector(
                        adj_inputs[0].index_map, adj_inputs[0].block_size, adj_inputs[0].array.dtype
                    )
                    one.array[:] = 1
                    return dolfinx.cpp.la.inner_product(vec._cpp_object, one._cpp_object)
                else:
                    try:
                        return adj_inputs[0].sum()
                    except AttributeError:
                        # Catch the case where adj_inputs[0] is just a float
                        return adj_inputs[0]
            elif isinstance(func := block_variable.output, dolfinx.fem.Function):
                assert func.function_space == prepared.function_space
                vec = _vector(
                    prepared.x.index_map, prepared.x.block_size, func.function_space, dtype=prepared.x.array.dtype
                )
                vec.array[:] = prepared.x.array[:]
                return vec

            else:
                raise NotImplementedError(f"Adjoint for {block_variable=} not implemented.")
            # elif isinstance(block_variable.output, dolfinx.fem.Constant):
        #         R = block_variable.output._ad_function_space(prepared.function_space.mesh)
        #         return self._adj_assign_constant(prepared, R)
        #     else:
        #         adj_output = dolfinx.fem.Function(
        #             block_variable.output.function_space())
        #         adj_output.assign(prepared)
        #         return adj_output.vector()
        # else:
        #     # Linear combination
        #     expr, adj_input_func = prepared
        #     adj_output = dolfinx.fem.Function(adj_input_func.function_space)
        #     if not isinstance(block_variable.output, dolfinx.fem.Constant):
        #         diff_expr = ufl.algorithms.expand_derivatives(
        #             ufl.derivative(expr, block_variable.saved_output, adj_input_func)
        #         )
        #         adj_output.assign(diff_expr)
        #     else:
        #         mesh = adj_output.function_space().mesh()
        #         diff_expr = ufl.algorithms.expand_derivatives(
        #             ufl.derivative(
        #                 expr,
        #                 block_variable.saved_output,
        #                 create_constant(1., domain=mesh)
        #             )
        #         )
        #         adj_output.assign(diff_expr)
        #         return adj_output.vector().inner(adj_input_func.vector())

        #     if isinstance(block_variable.output, dolfin.Constant):
        #         R = block_variable.output._ad_function_space(adj_output.function_space().mesh())
        #         return self._adj_assign_constant(adj_output, R)
        #     else:
        #         return adj_output.vector()

    def _adj_assign_constant(self, adj_output, constant_fs):
        r = dolfinx.fem.Function(constant_fs)
        shape = r.ufl_shape
        raise NotImplementedError("Not implemented for constants.")

        if shape == () or shape[0] == 1:
            # Scalar Constant
            raise NotImplementedError("Not implemented for scalar constants yet.")
            # r.vector()[:] = adj_output.vector().sum()
        # else:
        #     # We assume the shape of the constant == shape of the output function if not scalar.
        #     # This assumption is due to FEniCS not supporting products with non-scalar constants in assign.
        #     values = []
        #     for i in range(shape[0]):
        #         values.append(adj_output.sub(i, deepcopy=True).vector().sum())
        #     r.assign(dolfin.Constant(values))
        return r.vector()

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        if self.expr is None:
            return None

        return self._replace_with_saved_output()

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        if self.expr is None:
            return tlm_inputs[0]
        expr = prepared
        dudm = dolfinx.fem.Function(block_variable.output.function_space)
        dudmi = dolfinx.fem.Function(block_variable.output.function_space)
        for dep in self.get_dependencies():
            if dep.tlm_value:
                dudmi.assign(ufl.algorithms.expand_derivatives(ufl.derivative(expr, dep.saved_output, dep.tlm_value)))
                dudm.vector().axpy(1.0, dudmi.vector())

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

    # def recompute_component(self, inputs, block_variable, idx, prepared):
    #     if self.expr is None:
    #         prepared = inputs[0]
    #     output = dolfinx.fem.Function(
    #         block_variable.output.function_space, name="f{block_variable.output.name}_AssignBlockRecompute"
    #     )
    #     try:
    #         if output.function_space == prepared.function_space:
    #             output.x.array[:] = prepared.x.array[:]
    #     except AttributeError:
    #         # Handling float value
    #         output.x.array[:] = prepared
    #     return output
    def recompute_component(self, inputs, block_variable, idx, prepared):
        if self.expr is None:
            prepared = inputs[0]

        # We must return the exact same object instance to maintain C++ memory bindings
        # (especially for DirichletBCs), updating it in-place.
        output = block_variable.saved_output

        try:
            if output.function_space == prepared.function_space:
                output.x.array[:] = prepared.x.array[:]
        except AttributeError:
            # Handling float value
            output.x.array[:] = prepared

        return output

    def __str__(self):
        rhs = self.expr or self.other or self.get_dependencies()[0].output
        if isinstance(rhs, ufl.core.expr.Expr):
            rhs_str = ufl2unicode(rhs)
        else:
            rhs_str = str(rhs)
        return f"assign({rhs_str})"
