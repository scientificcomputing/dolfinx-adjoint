import warnings

import basix.ufl
import dolfinx
import numpy as np
import ufl
from pyadjoint.overloaded_type import create_overloaded_object
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

from .blocks.interpolation import ExprInterpolationBlock, InterpolationBlock, _reads_geometry
from .blocks.nonmatching_interpolation import NonmatchingInterpolationBlock
from .compat import get_interpolation_points
from .types.mesh import get_overloaded_mesh_if_annotated


def interpolate(u_or_expr, V: dolfinx.fem.FunctionSpace, **kwargs):
    """Interpolate a Function or UFL Expression into a different function space."""
    ad_block_tag = kwargs.pop("ad_block_tag", None)
    petsc_mat = kwargs.pop("petsc_mat", False)

    annotate = annotate_tape(kwargs)

    if not isinstance(u_or_expr, dolfinx.fem.Function):
        if not isinstance(u_or_expr, ufl.core.expr.Expr):
            raise TypeError("Input must be a dolfinx.fem.Function or ufl.core.expr.Expr")
        return _interpolate_expression(u_or_expr, V, annotate, ad_block_tag, petsc_mat)

    reads_geometry = _reads_geometry(u_or_expr, V)
    domain = V.mesh.ufl_domain()
    if annotate and reads_geometry and get_overloaded_mesh_if_annotated(domain) is not None:
        # A Piola map on either side makes the interpolation operator a function of the
        # geometry, which only the expression block differentiates.
        return _interpolate_expression(u_or_expr, V, annotate, ad_block_tag, petsc_mat)

    with stop_annotating():
        v = dolfinx.fem.Function(V)
        v.interpolate(u_or_expr)
        v.x.scatter_forward()
    output = create_overloaded_object(v)

    if annotate:
        block = InterpolationBlock(u_or_expr, output, ad_block_tag=ad_block_tag, petsc_mat=petsc_mat)
        if reads_geometry:
            # See _ProblemBlockBase._register_mesh_dependency: move() refuses on this.
            block._unannotated_domain = None if domain is None else domain.ufl_id()
        get_working_tape().add_block(block)
        block.add_output(output.block_variable)
    return output


def _interpolate_expression(expr, V: dolfinx.fem.FunctionSpace, annotate: bool, ad_block_tag, petsc_mat: bool):
    """Interpolate a UFL expression (a Function included) into ``V``, recorded as an expression block."""
    with stop_annotating():
        v = dolfinx.fem.Function(V)
        v.interpolate(dolfinx.fem.Expression(expr, get_interpolation_points(V)))
        v.x.scatter_forward()
    output = create_overloaded_object(v)
    if annotate:
        block = ExprInterpolationBlock(expr, output, ad_block_tag=ad_block_tag, petsc_mat=petsc_mat)
        get_working_tape().add_block(block)
        block.add_output(output.block_variable)
    return output


def _needs_split(space_from: dolfinx.fem.FunctionSpace, space_to: dolfinx.fem.FunctionSpace) -> bool:
    """Whether a shape-tracked non-matching interpolation has a Piola-mapped target."""
    tracked = any(
        get_overloaded_mesh_if_annotated(space.mesh.ufl_domain()) is not None for space in (space_from, space_to)
    )
    return tracked and not space_to.ufl_element().pullback.is_identity


def _interpolate_nonmatching_split(u_from, V_to, cells, interpolation_data, tol, maxit, ad_block_tag, petsc_mat):
    """Non-matching interpolation into a Piola-mapped space on a tracked mesh, as supported steps.

    The source is evaluated into a quadrature space at the target's interpolation points, an
    identity-pullback non-matching step, and the Piola map is then applied by an expression
    interpolation on the target mesh, whose shape derivative includes it. Both steps are exact.
    """
    if interpolation_data is not None:
        warnings.warn(
            "interpolation_data is recomputed for the intermediate quadrature space of a "
            "shape-tracked non-matching interpolation into a Piola-mapped space.",
            stacklevel=3,
        )
    X = V_to.element.interpolation_points
    points = basix.ufl.quadrature_element(V_to.mesh.basix_cell(), points=X, weights=np.ones(X.shape[0]))
    shape = tuple(V_to.value_shape)
    point_space = dolfinx.fem.functionspace(
        V_to.mesh, basix.ufl.blocked_element(points, shape=shape) if shape else points
    )
    at_points = interpolate_nonmatching(
        u_from, point_space, cells=cells, tol=tol, maxit=maxit, ad_block_tag=ad_block_tag, petsc_mat=petsc_mat
    )
    return _interpolate_expression(at_points, V_to, True, ad_block_tag, petsc_mat)


def interpolate_nonmatching(
    u_from: dolfinx.fem.Function,
    V_to: dolfinx.fem.FunctionSpace,
    cells=None,
    interpolation_data=None,
    tol: float = 1e-6,
    maxit: int = 15,
    **kwargs,
):
    """Interpolate a Function into a different function space on a non-matching mesh.

    On a mesh tracked for shape differentiation, a Piola-mapped target is recorded as two blocks
    (see :py:func:`_interpolate_nonmatching_split`).
    """
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
    if annotate and red_op is None and _needs_split(u_from.function_space, V_to):
        return _interpolate_nonmatching_split(
            u_from, V_to, cells, interpolation_data, tol, maxit, ad_block_tag, petsc_mat
        )

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
