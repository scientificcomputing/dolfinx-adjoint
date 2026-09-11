import typing

from mpi4py import MPI

import dolfinx
import numpy
import numpy.typing as npt
import ufl
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

from .blocks.assembly import AssembleBlock


def assemble_scalar(form: ufl.Form, **kwargs):
    """Assemble as scalar value from a form.

    Args:
        form: Symbolic form (UFL) to assemble.
        kwargs: Keyword arguments to pass to the assembly routine.
            Includes ``"ad_block_tag"`` to tag the block in the adjoint tape,
            ``"annotate"`` to control whether the assembly is annotated in the adjoint tape,
            ``"jit_options"`` for JIT compilation options,
            ``"form_compiler_options"`` for form compiler options, and ``"entity_maps"`` for
            assembling with Arguments and coefficients form meshes that has some relation.
    """
    ad_block_tag = kwargs.pop("ad_block_tag", None)

    annotate = annotate_tape(kwargs)
    with stop_annotating():
        compiled_form = dolfinx.fem.form(
            form,
            jit_options=kwargs.pop("jit_options", None),
            form_compiler_options=kwargs.pop("form_compiler_options", None),
            entity_maps=kwargs.pop("entity_maps", None),
        )

        local_output = dolfinx.fem.assemble_scalar(compiled_form)
        comm = compiled_form.mesh.comm
        output = comm.allreduce(local_output, op=MPI.SUM)
        assert isinstance(output, float)

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
