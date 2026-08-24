import warnings

import dolfinx
import numpy as np
import ufl
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

from .blocks.interpolation import ExprInterpolationBlock, InterpolationBlock
from .blocks.nonmatching_interpolation import NonmatchingInterpolationBlock
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


def interpolate_nonmatching(
    u_from: dolfinx.fem.Function,
    V_to: dolfinx.fem.FunctionSpace,
    cells=None,
    interpolation_data=None,
    tol: float = 1e-6,
    maxit: int = 15,
    **kwargs,
):
    """Interpolate a Function into a different function space on a non-matching mesh."""
    ad_block_tag = kwargs.pop("ad_block_tag", None)
    petsc_mat = kwargs.pop("petsc_mat", False)
    red_op = kwargs.pop("red_op", None)

    if red_op is not None and (cells is not None or interpolation_data is not None):
        warnings.warn(
            "A custom `red_op` was supplied together with explicit `cells`/`interpolation_data`. "
            "The transfer matrix used for the adjoint, TLM, Hessian, and all recomputes with a "
            "custom `red_op` is built by `fenicsx_ii.create_interpolation_matrix`, which does not "
            "accept `cells`/`interpolation_data` — those are only honored by the initial, "
            "tape-external forward evaluation done here.",
            stacklevel=2,
        )

    annotate = annotate_tape(kwargs)

    with stop_annotating():
        v = dolfinx.fem.Function(V_to)

        # 1. Provide defaults for cells and interpolation_data if not supplied
        if cells is None:
            mesh_to = V_to.mesh
            cells = np.arange(mesh_to.topology.index_map(mesh_to.topology.dim).size_local, dtype=np.int32)

        if interpolation_data is None:
            # Note: create_interpolation_data takes C++ objects for the function spaces
            interpolation_data = dolfinx.fem.create_interpolation_data(V_to, u_from.function_space, cells, padding=tol)

        # 2. Evaluate the forward non-matching interpolation natively
        v.interpolate_nonmatching(u_from, cells, interpolation_data, tol=tol, maxit=maxit)
        v.x.scatter_forward()

    # 3. Create the PyAdjoint wrapper
    output = create_overloaded_object(v)

    if annotate:
        tape = get_working_tape()

        # 4. Construct the block with all non-matching metadata
        block = NonmatchingInterpolationBlock(
            u_from,
            output,
            cells=cells,
            interpolation_data=interpolation_data,
            tol=tol,
            maxit=maxit,
            red_op=red_op,
            ad_block_tag=ad_block_tag,
            use_petsc=petsc_mat,
        )

        tape.add_block(block)
        block.add_output(output.block_variable)

    return output
