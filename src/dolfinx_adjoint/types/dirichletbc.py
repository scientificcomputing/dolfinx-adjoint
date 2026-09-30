import dolfinx
import numpy as np
import numpy.typing as npt
import pyadjoint
import ufl
from pyadjoint.overloaded_type import FloatingType, OverloadedType, create_overloaded_object
from pyadjoint.tape import get_working_tape, stop_annotating
from ufl.corealg.traversal import traverse_unique_terminals

from ..blocks.dirichletbc import DirichletBCBlock, build_cpp_bc_and_kwargs
from ..blocks.interpolation import ExprInterpolationBlock
from ..compat import get_interpolation_points
from .function import Function


def _tracked_terminals(expr: ufl.core.expr.Expr) -> list[OverloadedType]:
    """The tape-tracked coefficients a bc value expression depends on -- exactly the
    dependencies :py:class:`~dolfinx_adjoint.blocks.interpolation.ExprInterpolationBlock`
    records for it."""
    return [op for op in traverse_unique_terminals(expr) if isinstance(op, OverloadedType)]


def _pack_bc_value(g, V: dolfinx.fem.FunctionSpace, annotate: bool) -> Function:
    """Interpolate a Dirichlet bc value into the constrained space `V`, always via
    :py:class:`~dolfinx_adjoint.blocks.interpolation.ExprInterpolationBlock` -- even when
    `g` is already a bare :py:class:`~dolfinx_adjoint.Function`/:py:class:`~dolfinx_adjoint.Constant`,
    via ``ufl.as_ufl(g)`` (a no-op wrap for either).

    This is what makes :py:class:`~dolfinx_adjoint.blocks.dirichletbc.DirichletBCBlock`'s
    own adjoint/Hessian trivial: its single dependency is always this packed Function,
    living on `V` regardless of what `g` was, and
    :py:class:`~dolfinx_adjoint.blocks.interpolation.ExprInterpolationBlock`'s existing,
    general adjoint/Hessian machinery already computes the correct sensitivity for every
    case. See dolfinx-adjoint-knowledge's scratch/boundary-control/spec.md ("Every bc
    value is packed through ExprInterpolationBlock") for the full rationale.
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

    Like a plain dolfinx bc, the bc follows later updates to `g`: every
    :py:class:`~dolfinx_adjoint.LinearProblem`/:py:class:`~dolfinx_adjoint.NonlinearProblem`
    solve applies (and, when annotating, records a dependency on) `g`'s value at the time
    of that solve, see :py:meth:`_ad_refresh`. Only those overloaded solves do this: a
    plain ``dolfinx.fem.petsc`` solve reads the packed ``bc.g`` as last refreshed.

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

        # Pin the Python-level packed value. dolfinx 0.12 keeps it on the wrapper itself
        # (FEniCS/dolfinx#4342) and its own ``g`` would return it, but 0.11's unwraps to
        # the cpp Function -- and ``bc.g`` is documented above as the *packed Function*,
        # which callers index, interpolate from and build UFL expressions out of. The
        # ``g`` override below keeps that promise identical on every supported dolfinx.
        # ``function_space`` is deliberately *not* overridden the same way: 0.11's own
        # bcs_by_block and assemble_matrix read ``bc.function_space`` expecting the cpp
        # space, so forcing the Python wrapper there would break dolfinx's internals.
        # dolfinx_adjoint.compat.bcs_by_block normalises both flavours instead.
        self._packed_g = g_packed

        # Keep the *source* value (not just its packed snapshot) so the bc follows later
        # updates to it -- see _ad_refresh, which every Problem solve calls.
        self._source_expr = ufl.as_ufl(g)
        self._pack_expr = dolfinx.fem.Expression(self._source_expr, get_interpolation_points(V_used))
        self._ad_annotated = annotate
        self._packed_from = self._source_block_variables() if annotate else []

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

    @property
    def g(self) -> Function:  # type: ignore[override]
        """The packed bc value: always the Python-level :py:class:`dolfinx_adjoint.Function`
        on `V`, never dolfinx 0.11's cpp ``Function``. See :py:func:`_pack_bc_value`."""
        return self._packed_g

    def _source_block_variables(self) -> list:
        return [op.block_variable for op in _tracked_terminals(self._source_expr)]

    def _ad_refresh(self, annotate: bool) -> None:
        """Bring the bc up to date with its source value `g` before a solve reads it.

        The packed ``bc.g`` (see :py:func:`_pack_bc_value`) is a separate Function from
        `g`, so it is re-interpolated from `g` here; otherwise an update to `g` after the bc
        was built -- ``assign(new_value, g)`` each time step -- would never reach the solve.

        When annotating, and any tracked coefficient of `g` has a newer block variable
        than the one the bc's current tape entry was recorded from, a fresh
        :py:class:`~dolfinx_adjoint.blocks.interpolation.ExprInterpolationBlock` and
        :py:class:`~dolfinx_adjoint.blocks.dirichletbc.DirichletBCBlock` are recorded, so
        the solve that follows depends on the bc value at this point of the tape rather
        than on the value at construction time.

        Args:
            annotate: Whether the solve about to consume this bc is being annotated.
        """
        with stop_annotating():
            self._packed_g.interpolate(self._pack_expr)
            self._packed_g.x.scatter_forward()

        if not (annotate and self._ad_annotated):
            return
        current = self._source_block_variables()
        if len(current) == len(self._packed_from) and all(a is b for a, b in zip(current, self._packed_from)):
            return

        tape = get_working_tape()
        block = ExprInterpolationBlock(self._source_expr, self._packed_g)
        tape.add_block(block)
        block.add_output(self._packed_g.create_block_variable())
        self._ad_annotate_block()
        self._packed_from = current

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
