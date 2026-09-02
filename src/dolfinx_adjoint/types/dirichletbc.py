import dolfinx
import numpy as np
import numpy.typing as npt
import pyadjoint
import ufl
from pyadjoint.overloaded_type import FloatingType, create_overloaded_object
from pyadjoint.tape import get_working_tape, stop_annotating

from ..blocks.dirichletbc import DirichletBCBlock, build_cpp_bc_and_kwargs
from ..blocks.interpolation import ExprInterpolationBlock
from ..compat import get_interpolation_points
from .function import Function


def _pack_bc_value(g, V: dolfinx.fem.FunctionSpace, annotate: bool) -> Function:
    """Interpolate a Dirichlet bc value into the constrained space `V`, always via
    :py:class:`~dolfinx_adjoint.blocks.interpolation.ExprInterpolationBlock` -- even when
    `g` is already a bare :py:class:`~dolfinx_adjoint.Function`/:py:class:`~dolfinx_adjoint.Constant`,
    via ``ufl.as_ufl(g)`` (a no-op wrap for either: both are already valid UFL terminals).

    This is what makes :py:class:`~dolfinx_adjoint.blocks.dirichletbc.DirichletBCBlock`'s
    own adjoint/Hessian trivial: its single dependency is always this packed Function,
    living on `V` regardless of what `g` was -- a plain value, a
    :py:class:`~dolfinx_adjoint.Constant` broadcasting from its own private real space, or
    a genuinely nonlinear multi-coefficient expression.
    :py:class:`~dolfinx_adjoint.blocks.interpolation.ExprInterpolationBlock`'s existing,
    general adjoint/Hessian machinery (matrix-free interpolation-operator transpose
    action) already computes the correct sensitivity for all three cases -- including a
    `Constant`'s broadcast-sum, which is exactly the transpose of interpolating a single
    real-space dof across every dof of `V`.
    """
    expr = ufl.as_ufl(g)
    with stop_annotating():
        v = dolfinx.fem.Function(V)
        v.interpolate(dolfinx.fem.Expression(expr, get_interpolation_points(V)))
        v.x.scatter_forward()

    output = create_overloaded_object(v)
    if annotate:
        tape = get_working_tape()
        block = ExprInterpolationBlock(expr, output)
        tape.add_block(block)
        block.add_output(output.block_variable)
    return output


class DirichletBC(dolfinx.fem.DirichletBC, FloatingType):
    """A class overloading :py:class:`dolfinx.fem.DirichletBC` to support
    it being used as a control variable in the adjoint framework.

    Args:
        g: The value of the Dirichlet BC. May be a :py:class:`dolfinx_adjoint.Function`,
            a :py:class:`dolfinx_adjoint.Constant`, or an arbitrary UFL expression built
            from tracked coefficients (e.g. ``m**3`` for a :py:class:`dolfinx_adjoint.Constant`
            `m`) -- it is always packed into a fresh :py:class:`dolfinx_adjoint.Function`
            on `V` first, see :py:func:`_pack_bc_value`. Pass the *original* `g` (not
            ``bc.g``, which is the packed Function) to :py:class:`pyadjoint.Control`.
        dofs: An array of degree-of-freedom indices in `V` where the BC should be applied.
        V: The function space on which the boundary condition is defined (the space being
            constrained). Defaults to ``g.function_space`` when `g` has one (a `Function`
            or `Constant`); required when `g` is a general expression with no natural
            space of its own.
        **kwargs: Additional keyword arguments to pass to the
            :py:func:`pyadjoint.overloaded_type.FloatingType` constructor.

    """

    def __init__(
        self,
        g,
        dofs: npt.NDArray[np.int32],
        V: dolfinx.fem.FunctionSpace | None = None,
        **kwargs,
    ):
        V_used = V if V is not None else getattr(g, "function_space", None)
        if V_used is None:
            raise ValueError(
                "V is required: g has no function_space of its own to default to "
                "(it is a general UFL expression, not a Function/Constant)."
            )

        annotate = kwargs.pop("annotate", True)
        annotate = annotate and pyadjoint.annotate_tape()

        g_packed = _pack_bc_value(g, V_used, annotate)
        cpp_bc, bc_kwargs = build_cpp_bc_and_kwargs(g_packed, dofs, V_used)
        super().__init__(cpp_bc, **bc_kwargs)

        FloatingType.__init__(
            self,
            g_packed,
            dtype=g_packed.dtype,
            block_class=kwargs.pop("block_class", DirichletBCBlock),
            _ad_floating_active=False,
            _ad_args=kwargs.pop("_ad_args", (g_packed, dofs, V_used)),
            annotate=annotate,
            **kwargs,
        )

        if annotate:
            self._ad_annotate_block()

    def _ad_create_checkpoint(self):
        return self

    def _ad_restore_at_checkpoint(self, checkpoint):
        return self


def dirichletbc(
    value,
    dofs: npt.NDArray[np.int32],
    V: dolfinx.fem.FunctionSpace | None = None,
    **kwargs,
) -> DirichletBC:
    """Overloaded DirichletBC constructor that creates an adjoint-aware DirichletBC.

    Args:
        value: The value of the Dirichlet BC: a :py:class:`dolfinx_adjoint.Function`, a
            :py:class:`dolfinx_adjoint.Constant`, or an arbitrary UFL expression built
            from tracked coefficients. Always packed into a fresh Function on `V` --
            use `value` itself (not ``bc.g``) as the :py:class:`pyadjoint.Control`.
        dofs: An array of degree-of-freedom indices in `V` where the BC should be applied.
        V: The function space being constrained. Defaults to ``value.function_space`` when
            `value` has one; required otherwise (a general expression has no space of its
            own to default to).
        **kwargs: Additional keyword arguments to pass to the
            :py:class:`dolfinx_adjoint.types.dirichletbc.DirichletBC` constructor.


    """
    return DirichletBC(value, dofs, V=V, **kwargs)
