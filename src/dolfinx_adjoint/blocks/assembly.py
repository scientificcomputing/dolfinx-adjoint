import typing

from mpi4py import MPI

import dolfinx
import ufl
from pyadjoint import Block, create_overloaded_object
from ufl.formatting.ufl2unicode import ufl2unicode

from ._vector import _create_vector, _SpecialVector, _vector  # noqa: F401


def assemble_compiled_form(form: dolfinx.fem.Form, tensor: typing.Optional[typing.Union[dolfinx.la.Vector, _SpecialVector | float]] = None) -> typing.Union[dolfinx.la.Vector, _SpecialVector, float]:
    """Assemble a compiled form and optionally apply Dirichlet boundary condition.

    Args:
        form: Compiled form to assemble.
        tensor: Optional vector to which the assembled form will be added.
    Returns:
        tensor: The assembled vector, which is either the input tensor or a new vector
            created from the form's function space(s).
    Raises:
        NotImplementedError: If the form's rank is not 0 or 1.
    """

    if form.rank == 1:
        assert isinstance(tensor, dolfinx.la.Vector)
        dolfinx.fem.assemble._assemble_vector_array(tensor.array, form)
        tensor.scatter_reverse(dolfinx.la.InsertMode.add)
        tensor.scatter_forward()
    elif form.rank == 0:
        local_val = dolfinx.fem.assemble_scalar(form)
        comm = form.mesh.comm
        tensor = comm.allreduce(local_val, op=MPI.SUM)

    else:
        raise NotImplementedError("Only 1-form assembly is currently supported.")
    return tensor


class AssembleBlock(Block):
    """Block for assembling a symbolic UFL form into a tensor.

    Args:
        form: The UFL form to assemble.
        ad_block_tag: Tag for the block in the adjoint tape.
        jit_options: Dictionary of options for JIT compilation.
        form_compiler_options: Dictionary of options for the form compiler.
        entity_maps: Dictionary mapping meshes to entity maps for assembly.
    """

    def __init__(
        self,
        form: ufl.Form,
        ad_block_tag: typing.Optional[str] = None,
        jit_options: typing.Optional[dict] = None,
        form_compiler_options: typing.Optional[dict] = None,
        entity_maps: typing.Optional[typing.Sequence[dolfinx.mesh.EntityMap]] = None,
    ):
        super(AssembleBlock, self).__init__(ad_block_tag=ad_block_tag)

        # Store the options for code generation
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps

        # Store compiled and original form
        self.form = form
        self.compiled_form = dolfinx.fem.form(form, jit_options=jit_options, form_compiler_options=form_compiler_options, entity_maps=entity_maps)

        # NOTE: Add when we want to do shape optimization
        # mesh = self.form.ufl_domain().ufl_cargo()
        # self.add_dependency(mesh)
        for coefficient in self.form.coefficients():
            self.add_dependency(coefficient, no_duplicates=True)
        for constant in self.form.constants():
            self.add_dependency(constant, no_duplicates=True)
        # Set up cache for vectors that can be reused in adjoint action
        # self._cached_vectors: dict[int, _SpecialVector] = {}

    def __str__(self):
        return f"assemble({ufl2unicode(self.form)})"

    def compute_action_adjoint(
        self,
        adj_input: typing.Union[float, dolfinx.la.Vector],
        arity_form: int,
        form: typing.Optional[ufl.Form] = None,
        c_rep: typing.Optional[typing.Union[ufl.Coefficient, ufl.Constant]] = None,
        space: typing.Optional[dolfinx.fem.FunctionSpace] = None,
        dform: typing.Optional[dolfinx.fem.Form] = None,
    ):
        """This computes the action of the adjoint of the derivative of `form` wrt `c_rep` on `adj_input`.

        In other words, it returns:

        .. math::

            \\left\\langle\\left(\\frac{\\partial form}{\\partial c_{rep}}\\right)^*, adj_{input} \\right\\rangle

        - If `form` has arity 0, then :math:`\\frac{\\partial form}{\\partial c_{rep}}` is a 1-form
          and `adj_input` a float, we can simply use the `*` operator.

        - If `form` has arity 1 then :math:`\\frac{\\partial form}{\\partial c_{rep}}` is a 2-form
          and we can symbolically take its adjoint and then apply the action on `adj_input`, to finally
          assemble the result.

        Args:
            adj_input: The input to the adjoint operation, typically a scalar or vector.
            arity_form: The arity of the form, i.e., 0 for scalar, 1 for vector, 2 for matrix etc.
            form: The UFL form to differentiate if `dform` is not provided.
            c_rep: The coefficient or constant with respect to which the derivative is taken.
            space: The function space associated with the `c_rep` to form an `ufl.Argument` in.
            dform: Pre-computed derivative form, :math:`\\frac{\\partial form}{\\partial c_{rep}}`.
        """

        if arity_form == 0:
            assert arity_form == self.compiled_form.rank, "Inconsistent arity of input form and block form."
            if dform is None:
                assert space is not None
                if isinstance(c_rep, dolfinx.fem.Constant):
                    # Substitute Constant with Real space Function
                    c_func = dolfinx.fem.Function(space)
                    c_func.x.array[:] = c_rep.value
                    form_v = ufl.replace(form, {c_rep: c_func})
                    dc = ufl.TestFunction(space)
                    dform = ufl.derivative(form_v, c_func, dc)
                else:
                    dc = ufl.TestFunction(space)
                    dform = ufl.derivative(form, c_rep, dc)

            assert isinstance(dform, ufl.Form), "dform must be a UFL form."
            compiled_adjoint = dolfinx.fem.form(
                dform,
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )
            if space is None:
                # If space is not supplied infer it from the form
                assert len(dform.arguments()) == 1
                space = dform.arguments()[0].ufl_function_space()
                # self._cached_vectors[id(space)] = _create_vector(compiled_adjoint)
            vector = _create_vector(compiled_adjoint, space)
            vector.array[:] = 0.0
            # elif self._cached_vectors.get(id(space)) is None:
            # Create a new vector for this space
            # self._cached_vectors[id(space)] = _create_vector(compiled_adjoint)
            # self._cached_vectors[id(space)].array[:] = 0.0
            # assemble_compiled_form(compiled_adjoint, self._cached_vectors[id(space)])
            assemble_compiled_form(compiled_adjoint, vector)
            vector.array[:] *= vector.x.array.dtype.type(adj_input)
            vector.scatter_forward()

            # Return a scalar for Constant adjoint output
            if isinstance(c_rep, dolfinx.fem.Constant):
                return float(vector.array[0]) if len(vector.array) > 0 else 0.0, dform

            return vector, dform
            # Return a Vector scaled by the scalar `adj_input`
            # self._cached_vectors[id(space)].array[:] *= adj_input
            # self._cached_vectors[id(space)].scatter_forward()
            # return self._cached_vectors[id(space)], dform
            # elif arity_form == 1:
            #     if dform is None:
            #         dc = dolfin.TrialFunction(space)
            #         dform = dolfin.derivative(form, c_rep, dc)
            #     # Get the Function
            #     adj_input = adj_input.function
            #     # Symbolic operators such as action/adjoint require derivatives to have been expanded beforehand.
            #     # However, UFL doesn't support expanding coordinate derivatives of Coefficients in physical space,
            #     # implying that we can't symbolically take the action/adjoint of the Jacobian for SpatialCoordinates.
            #     # -> Workaround: Apply action/adjoint numerically (using PETSc).
            #     if not isinstance(c_rep, dolfin.SpatialCoordinate):
            #         # Symbolically compute: (dform/dc_rep)^* * adj_input
            #         adj_output = dolfin.action(dolfin.adjoint(dform), adj_input)
            #         adj_output = assemble_adjoint_value(adj_output)
            #     else:
            #         # Get PETSc matrix
            #         dform_mat = assemble_adjoint_value(dform).petscmat
            #         # Action of the adjoint (Hermitian transpose)
            #         adj_output = dolfin.Function(space)
            #         with adj_input.dat.vec_ro as v_vec:
            #             with adj_output.dat.vec as res_vec:
            #                 dform_mat.multHermitian(v_vec, res_vec)
            #     return adj_output, dform
        else:
            raise ValueError("Forms with arity > 1 are not handled yet!")

    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in self.form.coefficients():
                replaced_coeffs[coeff] = c_rep
            if coeff in self.form.coefficients() or coeff in self.form.constants():
                replaced_coeffs[coeff] = c_rep

        form = ufl.replace(self.form, replaced_coeffs)
        return form

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        form = prepared
        adj_input = adj_inputs[0]
        c = block_variable.output
        c_rep = block_variable.saved_output

        from ufl.algorithms.analysis import extract_arguments

        arity_form = len(extract_arguments(form))

        if isinstance(c, dolfinx.fem.Function):
            space = c.function_space
        elif isinstance(c, dolfinx.fem.Constant):
            domain = form.ufl_domain()
            mesh = dolfinx.mesh.Mesh(domain.ufl_cargo(), domain)
            space = c._ad_function_space(mesh)

        else:
            raise NotImplementedError(f"Adjoint evaluation for {type(c)} not implemented.")

        return self.compute_action_adjoint(adj_input, arity_form, form, c_rep, space)[0]

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, tlm_inputs, self.get_dependencies())

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        form = prepared
        dform = 0.0

        from ufl.algorithms.analysis import extract_arguments

        arity_form = len(extract_arguments(form))
        for bv in self.get_dependencies():
            c_rep = bv.saved_output
            tlm_value = bv.tlm_value
            if tlm_value is None:
                continue
            if isinstance(c_rep, dolfinx.mesh.Mesh):
                X = ufl.SpatialCoordinate(c_rep)
                dform += ufl.derivative(form, X, tlm_value)
            else:
                dform += ufl.derivative(form, c_rep, tlm_value)
        if not isinstance(dform, float):
            dform = ufl.algorithms.expand_derivatives(dform)
            compiled_form = dolfinx.fem.form(
                dform,
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )
            dform = assemble_compiled_form(compiled_form)
            if arity_form == 1 and dform != 0:
                # Then dform is a Vector
                dform = dform.function
        return dform

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        return self.prepare_evaluate_adj(inputs, adj_inputs, relevant_dependencies)

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
        form = prepared
        hessian_input = hessian_inputs[0]
        adj_input = adj_inputs[0]

        from ufl.algorithms.analysis import extract_arguments

        arity_form = len(extract_arguments(form))

        c1 = block_variable.output
        c1_rep = block_variable.saved_output

        if isinstance(c1, dolfinx.fem.Constant):
            mesh = form.ufl_domain()
            space = c1._ad_function_space(mesh)
        elif isinstance(c1, dolfinx.fem.Function):
            space = c1.function_space
        elif isinstance(c1, dolfinx.mesh.Mesh):
            c1_rep = ufl.SpatialCoordinate(c1)
            space = c1._ad_function_space()
        else:
            return None
        hessian_outputs, dform = self.compute_action_adjoint(hessian_input, arity_form, form, c1_rep, space)
        ddform = 0
        for other_idx, bv in relevant_dependencies:
            c2_rep = bv.saved_output
            tlm_input = bv.tlm_value

            if tlm_input is None:
                continue

            if isinstance(c2_rep, dolfinx.mesh.Mesh):
                X = ufl.SpatialCoordinate(c2_rep)
                ddform += ufl.derivative(dform, X, tlm_input)
            else:
                ddform += ufl.derivative(dform, c2_rep, tlm_input)
        if not isinstance(ddform, float):
            ddform = ufl.algorithms.expand_derivatives(ddform)

        if not ddform.empty():
            # FIXME: COmpare ddform with legacy dolfin_adjoitn here, as this is DG-0, while hessian is in DG-0
            adj_action = self.compute_action_adjoint(adj_input, arity_form, dform=ddform)[0]
            try:
                hessian_outputs += adj_action
            except TypeError:
                hessian_outputs.array[:] += adj_action.array[:]
        return hessian_outputs

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, None, None)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        form = prepared

        compiled_form = dolfinx.fem.form(
            form,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        local_output = dolfinx.fem.assemble_scalar(compiled_form)
        comm = compiled_form.mesh.comm
        output = comm.allreduce(local_output, op=MPI.SUM)
        output = create_overloaded_object(output)
        return output
