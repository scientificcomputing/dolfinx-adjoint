import typing

from mpi4py import MPI

import dolfinx
import numpy
import ufl
from pyadjoint import Block, OverloadedType, create_overloaded_object
from ufl.formatting.ufl2unicode import ufl2unicode

from ..ufl_utils import _wirtinger_derivative_forms
from ..utils import _compile_form
from ._vector import _create_vector, _SpecialVector, _vector  # noqa: F401


def _assemble_scalar_value(
    form: typing.Union[ufl.Form, dolfinx.fem.Form],
    jit_options: dict | None = None,
    form_compiler_options: dict | None = None,
    entity_maps: typing.Any = None,
) -> float:
    """Assemble a rank-0 form into a real scalar, reducing across the communicator.

    This is the only rank-0 assembly in the package, and so the only place that fixes what an
    assembled scalar *means*: under a complex-scalar build the real part is taken
    unconditionally, defining the assembled quantity as

    .. math::

        J := \\mathrm{Re}\\left(\\mathrm{assemble}(form)\\right)

    That is a definition rather than error-correction. A Functional must be real-valued even
    when the state is complex-valued, and :math:`\\mathrm{Re}(f(u))` is a perfectly
    well-defined real functional of a complex state -- the adjoint path differentiates *it*,
    consistently, which is exactly why
    {py:meth}`~dolfinx_adjoint.blocks.assembly.AssembleBlock.compute_action_adjoint` carries a
    factor of one half (:math:`\\mathrm{Re}(f(u))`'s Wirtinger derivative is
    :math:`\\tfrac{1}{2}f'(u)`, not :math:`f'(u)`). Taking the real part is also not optional
    plumbing: {py:func}`dolfinx.fem.assemble_scalar` returns a complex value under a
    complex-PETSc build whatever the form's structure.

    The definition lives here, beside the only other assembly in the package, rather than in
    {py:func}`dolfinx_adjoint.assemble_scalar`: what an assembled rank-0 form means is a fact
    about assembly, not about tape annotation, and the annotating entry point is a caller of
    this like any other.

    Args:
        form: The rank-0 form, symbolic (UFL) or already compiled. A compiled form ignores
            the compilation options below.
        jit_options: JIT compilation options.
        form_compiler_options: Form compiler options.
        entity_maps: Relations between the meshes of the form's arguments and coefficients.

    Returns:
        The reduced, real-valued scalar.
    """
    if isinstance(form, ufl.Form):
        form = _compile_form(
            form,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )
    local_output = dolfinx.fem.assemble_scalar(form)
    output = form.mesh.comm.allreduce(local_output, op=MPI.SUM)
    # See this function's docstring: J := Re(assemble(form)), unconditionally.
    return float(numpy.real(output))


def _assemble_rank1_form(
    form: ufl.Form,
    space: dolfinx.fem.FunctionSpace | None = None,
    jit_options: dict | None = None,
    form_compiler_options: dict | None = None,
    entity_maps: typing.Any = None,
) -> tuple[_SpecialVector, dolfinx.fem.FunctionSpace]:
    """Compile a rank-1 form and assemble it into a vector of its own.

    Args:
        form: The rank-1 form.
        space: The space its argument lives on. Inferred from the form when not given.
        jit_options: JIT compilation options.
        form_compiler_options: Form compiler options.
        entity_maps: Relations between the meshes of the form's arguments and coefficients.

    Returns:
        The assembled vector, and the space it lives on -- which a caller assembling a second
        form into the same space needs, having possibly left it to be inferred here.
    """
    compiled_form = _compile_form(
        form,
        jit_options=jit_options,
        form_compiler_options=form_compiler_options,
        entity_maps=entity_maps,
    )
    if space is None:
        (argument,) = form.arguments()
        space = argument.ufl_function_space()
    vector = _create_vector(compiled_form, space)
    vector.array[:] = 0.0
    assemble_compiled_form(compiled_form, vector)
    return vector, space


def _assemble_wirtinger_seed(
    form: ufl.Form,
    coefficient: typing.Union[ufl.Coefficient, ufl.Constant],
    argument: ufl.Argument,
    space: dolfinx.fem.FunctionSpace | None = None,
    jit_options: dict | None = None,
    form_compiler_options: dict | None = None,
    entity_maps: typing.Any = None,
) -> _SpecialVector:
    r"""Assemble the adjoint seed of a rank-0 ``form`` with respect to ``coefficient``.

    The one place a seed is derived and assembled, so that the difference between the two
    scalar types stays here rather than at each call site. Under a real-scalar build the seed
    is one assembled derivative. Under a complex-scalar build it takes two, combined vector by
    vector as :math:`\mathrm{Re}(v_1) + i\,\mathrm{Re}(v_2)`; see
    {py:func}`dolfinx_adjoint.ufl_utils._wirtinger_derivative_forms` for why one derivative is
    not enough and what the two are.

    Args:
        form: The rank-0 form to differentiate.
        coefficient: The coefficient to differentiate with respect to.
        argument: The direction to differentiate in, an argument over a real-valued basis.
        space: The space ``argument`` lives on. Inferred from the derivative when not given.
        jit_options: JIT compilation options.
        form_compiler_options: Form compiler options.
        entity_maps: Relations between the meshes of the form's arguments and coefficients.

    Returns:
        The assembled seed.
    """
    dform, dform_imaginary_direction = _wirtinger_derivative_forms(form, coefficient, argument)
    assert isinstance(dform, ufl.Form), "dform must be a UFL form."
    if dform.empty():
        # The form does not depend on this coefficient at all -- a second-order seed of a
        # Functional that is linear in the state, say, where the first derivative has already
        # differentiated the state away. The seed is zero, but it still has to be a vector on
        # the right space, so it is assembled from an explicit zero rather than short-circuited
        # here: an empty Form is not compilable, while a ZeroBaseForm over the same argument is.
        dform = ufl.ZeroBaseForm((argument,))
        dform_imaginary_direction = None
    vector, space = _assemble_rank1_form(
        dform,
        space,
        jit_options=jit_options,
        form_compiler_options=form_compiler_options,
        entity_maps=entity_maps,
    )
    if dform_imaginary_direction is not None:
        imaginary_direction_vector, _ = _assemble_rank1_form(
            dform_imaginary_direction,
            space,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )
        vector.array[:] = vector.array.real + 1j * imaginary_direction_vector.array.real
    return vector


def assemble_compiled_form(
    form: dolfinx.fem.Form, tensor: typing.Union[dolfinx.la.Vector, _SpecialVector | float] | None = None
) -> typing.Union[dolfinx.la.Vector, _SpecialVector, float]:
    """Assemble a compiled form into ``tensor`` (or return a new scalar).

    Args:
        form: Compiled form to assemble.
        tensor: For a rank-1 form, the vector to accumulate the assembled contribution
            into, while it is unused for a rank-0 form.
    Returns:
        For a rank-1 form, ``tensor`` itself (mutated in place). For a rank-0 form, the
        assembled scalar as a Python ``float`` -- delegated to
        {py:func}`_assemble_scalar_value` so that the definition
        :math:`J := \\mathrm{Re}(\\mathrm{assemble}(form))` is stated in exactly one place.
    Raises:
        NotImplementedError: If the form's rank is not 0 or 1.
    """

    if form.rank == 1:
        if tensor is None:
            raise ValueError("tensor must be provided for rank-1 forms.")
        assert isinstance(tensor, dolfinx.la.Vector)
        dolfinx.fem.assemble._assemble_vector_array(tensor.array, form)
        tensor.scatter_reverse(dolfinx.la.InsertMode.add)
        tensor.scatter_forward()
    elif form.rank == 0:
        tensor = _assemble_scalar_value(form)
    else:
        raise NotImplementedError("Only 1-form assembly is currently supported.")
    assert tensor is not None
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
        self.compiled_form = _compile_form(
            form, jit_options=jit_options, form_compiler_options=form_compiler_options, entity_maps=entity_maps
        )

        # NOTE: Add when we want to do shape optimization
        # mesh = self.form.ufl_domain().ufl_cargo()
        # self.add_dependency(mesh)
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
        dform: ufl.Form | None = None,
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
            computing_own_output_derivative = dform is None
            if dform is None:
                assert space is not None
                assert form is not None and c_rep is not None
                # Not ufl.derivative directly: under a complex build a single derivative does
                # not carry enough information to seed the adjoint, and the one UFL produces
                # breaks its own complex-mode arity rules. See
                # {py:func}`_assemble_wirtinger_seed`.
                vector = _assemble_wirtinger_seed(
                    form,
                    c_rep,
                    ufl.TestFunction(space),
                    space,
                    jit_options=self._jit_options,
                    form_compiler_options=self._form_compiler_options,
                    entity_maps=self._entity_maps,
                )
            else:
                # An already-derived form (the second-order path's second directional
                # derivative): the seed it stands for was decided by whoever derived it, so
                # there is nothing to split here and it is assembled as it comes.
                assert isinstance(dform, ufl.Form), "dform must be a UFL form."
                vector, space = _assemble_rank1_form(
                    dform,
                    space,
                    jit_options=self._jit_options,
                    form_compiler_options=self._form_compiler_options,
                    entity_maps=self._entity_maps,
                )

            if computing_own_output_derivative and numpy.iscomplexobj(vector.array):
                # This block's own forward output is Re(assemble(form)) (`assemble_scalar`
                # takes the real part unconditionally under a complex-PETSc build, since a
                # Functional must be real-valued even when the state is complex-valued), while
                # the seed above differentiates the underlying *complex* form. A real
                # parameter's gradient is recovered from an accumulated seed as `2*Re[.]`, in
                # `Function._ad_convert_riesz`; the derivative of Re(f) wanted here is one
                # half of that, so the one-half is applied at this end. Only applies when the
                # seed is freshly derived from `form` here (this block's own output); a
                # caller-supplied `dform` (e.g. the Hessian path's second directional
                # derivative) is a different quantity and is left untouched -- complex-mode
                # Hessians are not yet supported.
                vector.array[:] *= 0.5

            # return a vector scaled by the scalar `adj_input`
            # Safegaurd against None seeds from PyAdjoint
            if adj_input is None:
                adj_input = 1.0
            vector.array[:] *= vector.x.array.dtype.type(adj_input)
            vector.scatter_forward()

            # Returns the raw complex result under a complex build: this may be an adjoint
            # seed bound for a Block further upstream, so its real part must not be taken
            # here. The `2*Re[.]` that turns an accumulated seed into a real parameter's
            # gradient happens once, at the Control, in `Function._ad_convert_riesz`.
            return vector
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
        form = ufl.replace(self.form, replaced_coeffs)
        return form

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        form = prepared
        adj_input = adj_inputs[0]
        c = block_variable.output
        c_rep = block_variable.saved_output

        from ufl.algorithms.analysis import extract_arguments

        arity_form = len(extract_arguments(form))

        # if isinstance(c, dolfin.Constant):
        #     mesh = extract_mesh_from_form(self.form)
        #     space = c._ad_function_space(mesh)
        if isinstance(c, dolfinx.fem.Function):
            space = c.function_space
        # elif isinstance(c, dolfin.Mesh):
        #     c_rep = dolfin.SpatialCoordinate(c_rep)
        #     space = c._ad_function_space()

        return self.compute_action_adjoint(adj_input, arity_form, form, c_rep, space)

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
            compiled_form = _compile_form(
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
            raise RuntimeError(
                "All constants should have been replaced with real space coefficients before this point."
            )
        if isinstance(c1, dolfinx.fem.Function):
            space = c1.function_space
        # TODO: Add support for shape optimization
        # elif isinstance(c1, dolfinx.mesh.Mesh):
        #     c1_rep = ufl.SpatialCoordinate(c1)
        #     space = c1._ad_function_space()
        else:
            return None
        hessian_outputs = self.compute_action_adjoint(hessian_input, arity_form, form, c1_rep, space)

        # The remaining term seeds `c1` from this block's output *derivative* in the
        # tangent-linear direction, rather than from the output itself. That derivative is
        # `tlm_form` below -- the very form `evaluate_tlm_component` assembles -- and the
        # second-order seed of it is the first-order seed machinery applied to it unchanged.
        #
        # Deriving it that way rather than by differentiating the already-derived `dform` a
        # second time is what makes the complex-scalar case come out: this block's output is
        # Re(assemble(form)), so its directional derivative is Re(assemble(tlm_form)), and
        # seeding a real part correctly needs the two Wirtinger derivatives and the one-half
        # that `compute_action_adjoint` already applies to a form it is handed as `form`. A
        # form handed in as `dform` gets neither, since by then the seed it stands for has
        # already been decided. The two constructions agree term by term under a real build,
        # where differentiation simply commutes.
        # ZeroBaseForm rather than 0.0 as the identity to sum onto: a dependency with no
        # tangent-linear value contributes nothing, and starting from a float would leave the
        # sum a float in the case where *every* dependency does, which then needs testing for
        # separately from an empty form.
        tlm_form = ufl.ZeroBaseForm(())
        for other_idx, bv in relevant_dependencies:
            c2_rep = bv.saved_output
            tlm_input = bv.tlm_value

            if tlm_input is None:
                continue

            if isinstance(c2_rep, dolfinx.mesh.Mesh):
                X = ufl.SpatialCoordinate(c2_rep)
                tlm_form += ufl.derivative(form, X, tlm_input)
            else:
                tlm_form += ufl.derivative(form, c2_rep, tlm_input)
        tlm_form = ufl.algorithms.expand_derivatives(tlm_form)

        if not tlm_form.empty():
            adj_action = self.compute_action_adjoint(adj_input, arity_form, tlm_form, c1_rep, space)
            try:
                hessian_outputs += adj_action
            except TypeError:
                hessian_outputs.array[:] += adj_action.array[:]
        return hessian_outputs

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return self.prepare_evaluate_adj(inputs, None, None)

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return create_overloaded_object(
            _assemble_scalar_value(
                prepared,
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )
        )
