import typing

import dolfinx
import ufl
from pyadjoint import Block, OverloadedType, create_overloaded_object
from ufl.formatting.ufl2unicode import ufl2unicode

from ..types.function import _create_function
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
        # Reused across sweeps: DOLFINx vectors and functions are costly to create. Keyed by use and
        # dependency index, never shared between dependencies or between the adjoint and Hessian
        # sweeps, since pyadjoint keeps the returned vectors as adjoint values by reference.
        self._vectors: dict[tuple[str, int], _SpecialVector] = {}
        self._functions: dict[str, dolfinx.fem.Function] = {}

    def __str__(self):
        return f"assemble({ufl2unicode(self.form)})"

    def _function(self, key: str, space: dolfinx.fem.FunctionSpace) -> dolfinx.fem.Function:
        """A cached Function on ``space``, created once per ``key``."""
        function = self._functions.get(key)
        if function is None:
            function = _create_function(space)
            self._functions[key] = function
        return function

    def _functional(self, form: ufl.Form, seed, key: str) -> tuple[ufl.Form, float]:
        """``form`` as a functional to differentiate, and the factor to scale its derivative by.

        A functional (arity 0) is returned as it stands, scaled by the scalar ``seed``. For a
        vector (arity 1) the test function is replaced by the Function whose dofs are ``seed``,
        so that ``d/dc F(u; seed) = (dF/dc)^* seed``: the adjoint action becomes the derivative of
        a functional, which is also what a shape derivative needs, since UFL cannot take the
        adjoint of an unexpanded coordinate derivative.
        """
        arguments = form.arguments()
        if not arguments:
            return form, 1.0 if seed is None else float(seed)
        (test,) = arguments
        contracted = self._function(key, test.ufl_function_space())
        contracted.x.array[:] = getattr(seed, "array", getattr(seed, "x", seed).array)
        contracted.x.scatter_forward()
        return ufl.replace(form, {test: contracted}), 1.0

    def _assemble_dual(self, dform: ufl.Form, space: dolfinx.fem.FunctionSpace, scale: float, key) -> _SpecialVector:
        """Assemble the one-form ``dform`` on ``space`` into the cached vector for ``key``, scaled."""
        compiled = dolfinx.fem.form(
            dform,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        vector = self._vectors.get(key)
        if vector is None:
            vector = _create_vector(compiled, space)
            self._vectors[key] = vector
        vector.array[:] = 0.0
        assemble_compiled_form(compiled, vector)
        if scale != 1.0:
            vector.array[:] *= vector.array.dtype.type(scale)
        vector.scatter_forward()
        return vector

    @staticmethod
    def _target(c) -> tuple[typing.Any, dolfinx.fem.FunctionSpace] | None:
        """What to differentiate with respect to for dependency ``c``, and the space of the result.

        A mesh is differentiated through its ``SpatialCoordinate``: that is what the form
        references, and its checkpoint is a coordinate array, not a UFL object.
        """
        if isinstance(c, Mesh):
            return ufl.SpatialCoordinate(c), c._ad_function_space
        if isinstance(c, dolfinx.fem.Function):
            return None, c.function_space
        return None

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

        For a functional (arity 0) this is :math:`\lambda\,\partial F/\partial c`, for a vector
        (arity 1) the derivative of :math:`F` with its test function replaced by :math:`\lambda`
        (see :py:meth:`_functional`). Either way it is a vector in the dual of ``c``'s space; a
        mesh is differentiated through its ``SpatialCoordinate``, into its coordinate space.

        Args:
            prepared: ``self.form`` with the dependencies replaced by their checkpointed values.
            adj_inputs: ``[lambda]``, the adjoint seed of the output: a scalar for arity 0, a
                vector on the test space for arity 1.
            block_variable: The dependency ``c`` differentiated with respect to.

        Returns:
            The adjoint contribution to ``c``, a vector cached per dependency.

        Raises:
            NotImplementedError: For a dependency that is neither a Function nor a Mesh.
        """
        target = self._target(block_variable.output)
        if target is None:
            raise NotImplementedError(f"Unsupported control {type(block_variable.output)}")
        c_rep, space = target
        c_rep = block_variable.saved_output if c_rep is None else c_rep
        functional, scale = self._functional(prepared, adj_inputs[0], "adjoint seed")
        dform = ufl.derivative(functional, c_rep, ufl.TestFunction(space))
        return self._assemble_dual(dform, space, scale, ("adjoint", idx))

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
            A scalar for a functional (arity 0), a cached Function on the test space for a vector
            (arity 1), or ``0.0`` when no dependency has a tangent-linear value.
        """
        form = prepared
        dform = 0.0
        for bv in self.get_dependencies():
            if bv.tlm_value is None:
                continue
            c_rep = ufl.SpatialCoordinate(bv.output) if isinstance(bv.output, Mesh) else bv.saved_output
            dform += ufl.derivative(form, c_rep, bv.tlm_value)
        if isinstance(dform, float):
            return dform
        compiled = dolfinx.fem.form(
            ufl.algorithms.expand_derivatives(dform),
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        if not form.arguments():
            return assemble_compiled_form(compiled)
        output = self._function("tlm", form.arguments()[0].ufl_function_space())
        output.x.array[:] = 0.0
        assemble_compiled_form(compiled, output.x)
        return output

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
        For a vector (arity 1) both seeds replace the test function, as in
        :py:meth:`evaluate_adj_component`.

        Args:
            prepared: The replaced form from :py:meth:`prepare_evaluate_hessian`.

        Returns:
            The contribution, a vector cached per dependency, or ``None`` for a dependency that
            is neither a Function nor a Mesh.

        Raises:
            RuntimeError: For a :py:class:`dolfinx.fem.Constant` dependency, which should have
                been replaced by a real-space coefficient earlier.
        """
        c1 = block_variable.output
        if isinstance(c1, dolfinx.fem.Constant):
            raise RuntimeError(
                "All constants should have been replaced with real space coefficients before this point."
            )
        target = self._target(c1)
        if target is None:
            return None
        c1_rep, space = target
        c1_rep = block_variable.saved_output if c1_rep is None else c1_rep
        dc = ufl.TestFunction(space)

        functional, scale = self._functional(prepared, hessian_inputs[0], "hessian seed")
        hessian_output = self._assemble_dual(ufl.derivative(functional, c1_rep, dc), space, scale, ("hessian", idx))

        functional, scale = self._functional(prepared, adj_inputs[0], "adjoint seed")
        dform = ufl.derivative(functional, c1_rep, dc)
        ddform = 0.0
        for _, bv in relevant_dependencies:
            if bv.tlm_value is None:
                continue
            c2_rep = ufl.SpatialCoordinate(bv.output) if isinstance(bv.output, Mesh) else bv.saved_output
            ddform += ufl.derivative(dform, c2_rep, bv.tlm_value)
        if not isinstance(ddform, float):
            ddform = ufl.algorithms.expand_derivatives(ddform)
            if not ddform.empty():
                second = self._assemble_dual(ddform, space, scale, ("hessian second order", idx))
                hessian_output.array[:] += second.array
        return hessian_output

    def prepare_recompute_component(self, inputs, relevant_outputs):
        """The replaced form, as for :py:meth:`prepare_evaluate_adj`."""
        return self.prepare_evaluate_adj(inputs, None, None)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        """Reassemble the form at the checkpointed dependency values.

        Args:
            prepared: The replaced form from :py:meth:`prepare_recompute_component`.

        Returns:
            For a functional, its value summed over all ranks, as an overloaded float; for a
            vector, the output Function, reassembled in place.
        """
        compiled = dolfinx.fem.form(
            prepared,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        if prepared.arguments():
            output = block_variable.saved_output
            output.x.array[:] = 0.0
            assemble_compiled_form(compiled, output.x)
            return output
        return create_overloaded_object(assemble_compiled_form(compiled))
