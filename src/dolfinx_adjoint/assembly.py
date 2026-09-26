import typing

from mpi4py import MPI

import dolfinx
import numpy
import numpy.typing as npt
import ufl
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

from .blocks._assemble import assemble_compiled_form
from .blocks.assembly import AssembleBlock
from .types.function import Function
from .ufl_utils import pin_quadrature_degrees


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
    # So every derivative of the functional uses the quadrature it was assembled with.
    form = pin_quadrature_degrees(form)
    options = _compile_options(kwargs)

    annotate = annotate_tape(kwargs)
    with stop_annotating():
        compiled_form = dolfinx.fem.form(form, **options)

        local_output = dolfinx.fem.assemble_scalar(compiled_form)
        comm = compiled_form.mesh.comm
        output = comm.allreduce(local_output, op=MPI.SUM)
        assert isinstance(output, float)

    output = create_overloaded_object(output)

    if annotate:
        block = AssembleBlock(form, ad_block_tag=ad_block_tag, **options)

        tape = get_working_tape()
        tape.add_block(block)

        block.add_output(output.block_variable)

    return output


def _compile_options(kwargs: dict) -> dict:
    """The form compilation options in ``kwargs``, removed from it."""
    return {name: kwargs.pop(name, None) for name in ("jit_options", "form_compiler_options", "entity_maps")}


def assemble_vector(form: ufl.Form, **kwargs) -> Function:
    """Assemble a linear form, returning its entries as the dofs of a Function on the test space.

    The result is a dual vector stored in a primal Function, so it can be used as a coefficient
    downstream; its derivatives treat it as the vector of entries.

    Args:
        form: Linear form (UFL) to assemble, with a single test function.
        kwargs: As for :py:func:`assemble_scalar`.

    Returns:
        The assembled vector as a Function, ghosts updated.

    Raises:
        ValueError: If ``form`` is not linear.
    """
    ad_block_tag = kwargs.pop("ad_block_tag", None)
    form = pin_quadrature_degrees(form)
    options = _compile_options(kwargs)
    arguments = form.arguments()
    if len(arguments) != 1:
        raise ValueError(f"assemble_vector needs a linear form, got one with {len(arguments)} arguments.")

    annotate = annotate_tape(kwargs)
    with stop_annotating():
        output = Function(arguments[0].ufl_function_space())
        output.x.array[:] = 0.0
        assemble_compiled_form(dolfinx.fem.form(form, **options), output.x)

    if annotate:
        block = AssembleBlock(form, ad_block_tag=ad_block_tag, **options)
        get_working_tape().add_block(block)
        block.add_output(output.create_block_variable())
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
