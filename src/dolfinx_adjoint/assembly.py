import typing

import dolfinx
import numpy
import numpy.typing as npt
import ufl
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

from .blocks.assembly import AssembleBlock, _assemble_scalar_value


def assemble_scalar(form: typing.Union[ufl.Form, dolfinx.fem.Form], **kwargs):
    """Assemble a rank-0 form into a real scalar, annotating it onto the tape.

    The annotating entry point for
    {py:func}`dolfinx_adjoint.blocks.assembly._assemble_scalar_value`, which does the
    assembly and states what the assembled quantity *is* -- in particular that under a
    complex-scalar build the real part is taken unconditionally,
    :math:`J := \\mathrm{Re}(\\mathrm{assemble}(form))`. See it for why that is a definition
    rather than error-correction. This function adds only the tape block.

    Args:
        form: Symbolic form (UFL) to assemble, or an already-compiled rank-0
            {py:class}`dolfinx.fem.Form`. A compiled form cannot be annotated onto the tape
            (recording a block needs the symbolic form), so it requires ``annotate=False``.
        kwargs: Keyword arguments to pass to the assembly routine.
            Includes ``"ad_block_tag"`` to tag the block in the adjoint tape,
            ``"annotate"`` to control whether the assembly is annotated in the adjoint tape,
            ``"jit_options"`` for JIT compilation options,
            ``"form_compiler_options"`` for form compiler options, and ``"entity_maps"`` for
            assembling with Arguments and coefficients form meshes that has some relation.

    Raises:
        ValueError: If an already-compiled form is passed with annotation enabled.
    """
    ad_block_tag = kwargs.pop("ad_block_tag", None)

    annotate = annotate_tape(kwargs)
    already_compiled = not isinstance(form, ufl.Form)
    if already_compiled and annotate:
        raise ValueError(
            "assemble_scalar cannot annotate an already-compiled form -- recording a block "
            "requires the symbolic UFL form. Pass the UFL form, or annotate=False."
        )

    with stop_annotating():
        output = _assemble_scalar_value(
            form,
            jit_options=kwargs.pop("jit_options", None),
            form_compiler_options=kwargs.pop("form_compiler_options", None),
            entity_maps=kwargs.pop("entity_maps", None),
        )

    output = create_overloaded_object(output)

    if annotate:
        block = AssembleBlock(form, ad_block_tag=ad_block_tag)

        tape = get_working_tape()
        tape.add_block(block)

        block.add_output(output.block_variable)

    return output


def error_norm(
    u_ex: ufl.core.expr.Expr,
    u: ufl.core.expr.Expr,
    norm_type: typing.Literal["L2", "H1"] = "L2",
    jit_options: dict | None = None,
    form_compiler_options: dict | None = None,
    entity_map: dict[dolfinx.mesh.Mesh, npt.NDArray[numpy.int32]] | None = None,
    ad_block_tag: str | None = None,
    annotate: bool = True,
) -> float:
    """Compute the error norm between the exact solution and the computed solution.

    Args:
        u_ex: The exact solution as a UFL expression.
        u: The computed solution as a UFL expression.
        norm_type: The type of norm to compute, either "L2" or "H1".
        jit_options: Optional JIT compilation options.
        form_compiler_options: Optional form compiler options.
        entity_map: Optional mapping from mesh entities to submesh entities.
        ad_block_tag: Optional tag for the block in the adjoint tape.
        annotate: Whether to annotate the assignment in the adjoint tape.
    Returns:
        The computed error norm as a float.
    """
    diff = u_ex - u
    norm = ufl.inner(diff, diff) * ufl.dx
    if norm_type == "H1":
        norm += ufl.inner(ufl.grad(diff), ufl.grad(diff)) * ufl.dx
    return numpy.sqrt(
        assemble_scalar(
            norm,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_map,
            ad_block_tag=ad_block_tag,
            annotate=annotate,
        )
    )
