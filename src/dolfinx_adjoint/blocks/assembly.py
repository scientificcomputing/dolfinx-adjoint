import typing

from mpi4py import MPI

import dolfinx
import ufl
from pyadjoint import Block, OverloadedType, create_overloaded_object
from ufl.algorithms.analysis import extract_arguments
from ufl.formatting.ufl2unicode import ufl2unicode

from ..types.mesh import Mesh, get_overloaded_mesh_if_annotated
from ..ufl_utils import reject_form_with_unsupported_shape_derivative
from ._assemble import assemble_compiled_form  # noqa: F401
from ._vector import _create_vector, _SpecialVector, _vector  # noqa: F401


class AssembleBlock(Block):
    """Block for assembling a symbolic UFL form into a tensor.

    Args:
        form: The UFL form to assemble.
        ad_block_tag: Tag for the block in the adjoint tape.
        jit_options: Dictionary of options for JIT compilation.
        form_compiler_options: Dictionary of options for the form compiler.
        entity_maps: Dictionary mapping meshes to entity maps for assembly.
    """

    # If domain in input form was annotated or not at time of initialization. If annoated
    # this stores the `ufl_id()` of the domain. If not annoated store `None`.
    _unannotated_domain: int | None

    def __init__(
        self,
        form: ufl.Form,
        ad_block_tag: str | None = None,
        jit_options: dict | None = None,
        form_compiler_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
    ):
        super(AssembleBlock, self).__init__(ad_block_tag=ad_block_tag)

        # Store the options for code generation
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps

        # Store compiled and original form
        self.form = form
        self.compiled_form = dolfinx.fem.form(
            form, jit_options=jit_options, form_compiler_options=form_compiler_options, entity_maps=entity_maps
        )

        # If the mesh is annotated, it means that it potentially needs shape derivatives, and we store
        # it as a dependency of the block.
        mesh = get_overloaded_mesh_if_annotated(self.form.ufl_domain())
        if mesh is not None:
            reject_form_with_unsupported_shape_derivative(self.form)
            self.add_dependency(mesh, no_duplicates=True)
        else:
            # See _ProblemBlockBase._register_mesh_dependency: a block built before the mesh
            # was overloaded cannot be rewound, so record the domain for move() to refuse on.
            domain = self.form.ufl_domain()
            self._unannotated_domain = None if domain is None else domain.ufl_id()
        for coefficient in self.form.coefficients():
            if isinstance(coefficient, OverloadedType):
                self.add_dependency(coefficient, no_duplicates=True)
        # Set up cache for vectors that can be reused in adjoint action
        # self._cached_vectors: dict[int, _SpecialVector] = {}

    def __str__(self):
        return f"assemble({ufl2unicode(self.form)})"

    def compute_action_adjoint(
        self,
        adj_input: typing.Union[float, dolfinx.la.Vector],
        arity_form: int,
        form: ufl.Form | None = None,
        c_rep: typing.Union[ufl.Coefficient, ufl.Constant] | None = None,
        space: dolfinx.fem.FunctionSpace | None = None,
        dform: dolfinx.fem.Form | None = None,
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
            # return a vector scaled by the scalar `adj_input`
            # Safegaurd against None seeds from PyAdjoint
            if adj_input is None:
                adj_input = 1.0
            vector.array[:] *= vector.x.array.dtype.type(adj_input)
            vector.scatter_forward()

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
        """``self.form`` with every coefficient dependency replaced by its checkpointed value.

        Shared by the recompute, TLM and Hessian sweeps. A mesh dependency needs no replacement:
        the form references its coordinates, which the tape has already restored.

        Returns:
            The replaced UFL form.
        """
        replaced_coeffs = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            c_rep = block_variable.saved_output
            if coeff in self.form.coefficients():
                replaced_coeffs[coeff] = c_rep
        form = ufl.replace(self.form, replaced_coeffs)
        return form

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        r"""The adjoint action :math:`(\partial F/\partial c)^*[\lambda]` for dependency ``c``.

        For a functional (arity 0) this is the assembled vector :math:`\lambda\,\partial F/\partial c`
        in the dual of ``c``'s space; a mesh dependency is differentiated through its
        ``SpatialCoordinate``, into its coordinate space. Arity 1 is not implemented yet.

        Args:
            prepared: ``self.form`` with the dependencies replaced by their checkpointed values.
            adj_inputs: ``[lambda]``, the adjoint seed of the output, a scalar for arity 0.
            block_variable: The dependency ``c`` differentiated with respect to.

        Returns:
            The adjoint contribution to ``c``, a :py:class:`dolfinx.la.Vector`.

        Raises:
            ValueError: For a form of arity 1 or higher.
            NotImplementedError: For a dependency that is neither a Function nor a Mesh.
        """
        form = prepared
        adj_input = adj_inputs[0]
        c = block_variable.output
        c_rep = block_variable.saved_output

        arity_form = len(extract_arguments(form))

        if isinstance(c, Mesh):
            # Differentiate w.r.t. the coordinate field rather than the mesh object: that
            # is what the form actually references. c_rep is the checkpointed coordinate
            # array, which is not a UFL object, so the mesh itself supplies both.
            c_rep = ufl.SpatialCoordinate(c)
            space = c._ad_function_space
        elif isinstance(c, dolfinx.fem.Function):
            space = c.function_space
        else:
            raise NotImplementedError(f"Unsupported control {type(c)}")

        return self.compute_action_adjoint(adj_input, arity_form, form, c_rep, space)[0]

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs):
        """The replaced form, as for :py:meth:`prepare_evaluate_adj`."""
        return self.prepare_evaluate_adj(inputs, tlm_inputs, self.get_dependencies())

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        r"""The directional derivative :math:`\sum_c (\partial F/\partial c)[\dot c]`.

        Summed over the dependencies with a tangent-linear value :math:`\dot c`; a mesh
        dependency is differentiated through its ``SpatialCoordinate``.

        Args:
            prepared: The replaced form from :py:meth:`prepare_evaluate_tlm`.

        Returns:
            A scalar for a functional (arity 0), a Function for a vector (arity 1), or ``0.0``
            when no dependency has a tangent-linear value.
        """
        form = prepared
        dform = 0.0

        arity_form = len(extract_arguments(form))
        for bv in self.get_dependencies():
            c_rep = bv.saved_output
            tlm_value = bv.tlm_value
            if tlm_value is None:
                continue
            if isinstance(bv.output, Mesh):
                X = ufl.SpatialCoordinate(bv.output)
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
        """The replaced form, as for :py:meth:`prepare_evaluate_adj`."""
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
        r"""The second-order adjoint contribution to dependency ``c1``.

        With :math:`\hat\lambda` the Hessian seed of the output and :math:`\lambda` its adjoint
        seed, this is

        .. math::

            (\partial F/\partial c_1)^*[\hat\lambda]
            + \Big(\sum_{c_2} \partial^2 F/\partial c_1 \partial c_2 [\dot c_2]\Big)^*[\lambda],

        the sum running over the dependencies with a tangent-linear value :math:`\dot c_2`.
        Implemented for a functional (arity 0), like :py:meth:`evaluate_adj_component`.

        Args:
            prepared: The replaced form from :py:meth:`prepare_evaluate_hessian`.

        Returns:
            The contribution as a :py:class:`dolfinx.la.Vector`, or ``None`` for a dependency
            that is neither a Function nor a Mesh.

        Raises:
            RuntimeError: For a :py:class:`dolfinx.fem.Constant` dependency, which should have
                been replaced by a real-space coefficient earlier.
        """
        form = prepared
        hessian_input = hessian_inputs[0]
        adj_input = adj_inputs[0]

        arity_form = len(extract_arguments(form))

        c1 = block_variable.output
        c1_rep = block_variable.saved_output

        if isinstance(c1, dolfinx.fem.Constant):
            raise RuntimeError(
                "All constants should have been replaced with real space coefficients before this point."
            )
        if isinstance(c1, Mesh):
            # Differentiate w.r.t. the coordinate field rather than the mesh object: that is
            # what the form actually references. The mesh's checkpoint is a coordinate array,
            # not a UFL object, so the mesh itself supplies both.
            c1_rep = ufl.SpatialCoordinate(c1)
            space = c1._ad_function_space
        elif isinstance(c1, dolfinx.fem.Function):
            space = c1.function_space
        else:
            return None
        hessian_outputs, dform = self.compute_action_adjoint(hessian_input, arity_form, form, c1_rep, space)
        ddform = 0.0
        for other_idx, bv in relevant_dependencies:
            c2_rep = bv.saved_output
            tlm_input = bv.tlm_value

            if tlm_input is None:
                continue

            if isinstance(bv.output, Mesh):
                X = ufl.SpatialCoordinate(bv.output)
                ddform += ufl.derivative(dform, X, tlm_input)
            else:
                ddform += ufl.derivative(dform, c2_rep, tlm_input)
        if not isinstance(ddform, float):
            ddform = ufl.algorithms.expand_derivatives(ddform)

        if not ddform.empty():
            adj_action = self.compute_action_adjoint(adj_input, arity_form, dform=ddform)[0]
            try:
                hessian_outputs += adj_action
            except TypeError:
                hessian_outputs.array[:] += adj_action.array[:]
        return hessian_outputs

    def prepare_recompute_component(self, inputs, relevant_outputs):
        """The replaced form, as for :py:meth:`prepare_evaluate_adj`."""
        return self.prepare_evaluate_adj(inputs, None, None)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        """Reassemble the functional at the checkpointed dependency values.

        Args:
            prepared: The replaced form from :py:meth:`prepare_recompute_component`.

        Returns:
            The value, summed over all ranks, as an overloaded float.
        """
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
