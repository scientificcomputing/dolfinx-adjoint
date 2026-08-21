import dolfinx
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating
import ufl
from .blocks.interpolation import InterpolationBlock, ExprInterpolationBlock
from .compat import get_interpolation_points


def interpolate(u_or_expr, V: dolfinx.fem.FunctionSpace, **kwargs):
    """Interpolate a Function or UFL Expression into a different function space."""
    ad_block_tag = kwargs.pop("ad_block_tag", None)
    petsc_mat = kwargs.pop("petsc_mat", False)

    annotate = annotate_tape(kwargs)

    # Evaluate the forward interpolation without recording to tape yet
    with stop_annotating():
        v = dolfinx.fem.Function(V)
        if isinstance(u_or_expr, dolfinx.fem.Function):
            v.interpolate(u_or_expr)
        elif isinstance(u_or_expr, ufl.core.expr.Expr):
            ip = get_interpolation_points(V)
            compiled_expr = dolfinx.fem.Expression(u_or_expr, ip)
            v.interpolate(compiled_expr)
        else:
            raise TypeError("Input must be a dolfinx.fem.Function or ufl.core.expr.Expr")

        v.x.scatter_forward()

    output = create_overloaded_object(v)

    if annotate:
        tape = get_working_tape()

        if isinstance(u_or_expr, dolfinx.fem.Function):
            # Assume InterpolationBlock is imported
            block = InterpolationBlock(u_or_expr, output, ad_block_tag=ad_block_tag, petsc_mat=petsc_mat)
        elif isinstance(u_or_expr, ufl.core.expr.Expr):
            block = ExprInterpolationBlock(u_or_expr, output, ad_block_tag=ad_block_tag, petsc_mat=petsc_mat)

        tape.add_block(block)
        block.add_output(output.block_variable)

    return output
