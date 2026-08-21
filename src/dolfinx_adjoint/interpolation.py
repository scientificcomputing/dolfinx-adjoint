import dolfinx
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

from .blocks.interpolation import InterpolationBlock


def interpolate(u: dolfinx.fem.Function, V: dolfinx.fem.FunctionSpace, **kwargs):
    """Interpolate a function to a different function space.

    Args:
        u: The function to interpolate.
        V: The function space to interpolate to.
        kwargs: Keyword arguments to pass to the interpolation routine.
            Includes ``"ad_block_tag"`` to tag the block in the adjoint tape,
            ``"annotate"`` to control whether the assembly is annotated in the adjoint tape.
            If you want to use PETSc based interpolation matrices, you can passe `petsc_mat=True` in the kwargs.
    """
    ad_block_tag = kwargs.pop("ad_block_tag", None)
    petsc_mat = kwargs.pop("petsc_mat", False)
    annotate = annotate_tape(kwargs)
    with stop_annotating():
        v = dolfinx.fem.Function(V)
        v.interpolate(u)
    output = create_overloaded_object(v)

    if annotate:
        block = InterpolationBlock(u, output, ad_block_tag=ad_block_tag, petsc_mat=petsc_mat)

        tape = get_working_tape()
        tape.add_block(block)

        block.add_output(output.block_variable)

    return output
