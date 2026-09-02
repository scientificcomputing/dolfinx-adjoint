from __future__ import annotations

import abc
import itertools
import typing

import dolfinx.fem.petsc
import pyadjoint
import ufl
from dolfinx.fem.function import Function as _Function

from .blocks.solvers import LinearProblemBlock, NonlinearProblemBlock, _ProblemBlockBase, collect_coefficients
from .petsc_utils import HomogeneousBCLinearProblem
from .types import Function
from .typing_utils import NestedSequence
from .ufl_utils import (
    assign_mixed_parts,
    compute_adjoint,
    get_sorted_arguments,
    recursive_replace,
    sum_form,
)

# A counter incremented once per Problem construction is deterministic
# and identical on every rank, since construction happens in lock-step
# in a well-formed SPMD program.
_PROBLEM_PREFIX_COUNTER = itertools.count()


@typing.overload
def find_or_create_then_overload(u: _Function | None, L: ufl.Form) -> _Function: ...
@typing.overload
def find_or_create_then_overload(
    u: typing.Sequence[_Function] | None, L: typing.Sequence[ufl.Form]
) -> typing.Sequence[_Function]: ...


def find_or_create_then_overload(
    u: _Function | typing.Sequence[_Function] | None, L: ufl.Form | typing.Sequence[ufl.Form]
) -> _Function | typing.Sequence[_Function]:
    """Find or create, then overload the unknown
    {py:class}`~dolfinx_adjoint.Function` ``u`` for a `*Problem`.

    If ``u`` was not supplied by the caller, a fresh {py:class}`~dolfinx_adjoint.Function`
    is created per block, using the function space of the corresponding
    {py:class}`ufl.Argument` in ``L``. If ``u`` was supplied, it is wrapped with
    {py:func}`pyadjoint.create_overloaded_object` so the tape can record operations on it,
    regardless of whether it arrived as a plain {py:class}`~dolfinx_adjoint.Function` or
    already-overloaded one.

    Args:
        u: The unknown Function (or, for a blocked problem, a sequence of them), or
            ``None`` to have one created per block.
        L: The right-hand-side form (or, for a blocked problem, a sequence of forms) whose
            test-function space determines the space of a newly created ``u``. Only
            consulted when ``u`` is ``None``.

    Returns:
        A single {py:class}`~dolfinx_adjoint.Function`, or a list of them for a blocked
        problem, matching the shape of ``L``.
    """
    if u is None:
        try:
            # Extract function space for unknown from the right hand
            # side of the equation.
            assert isinstance(L, ufl.Form)
            return Function(L.arguments()[0].ufl_function_space())
        except AttributeError:
            assert isinstance(L, typing.Iterable)
            return [Function(Li.arguments()[0].ufl_function_space()) for Li in L]
    else:
        if isinstance(u, dolfinx.fem.Function):
            return pyadjoint.create_overloaded_object(u)
        else:
            return [pyadjoint.create_overloaded_object(ui) for ui in u]


class HessianTemplates(typing.NamedTuple):
    """Per-dependency compiled templates used to assemble a Hessian action.

    Shared shape for {py:class}`~dolfinx_adjoint.LinearProblem`/
    {py:class}`~dolfinx_adjoint.NonlinearProblem`, so callers in
    ``blocks/solvers.py`` unpack by name rather than by position -- the two
    classes used to return differently-shaped tuples here, which was a latent
    footgun for any code touching both.

    Attributes:
        soa_self: The SOA right-hand-side's contribution from ``dF/du``'s own
            second derivative w.r.t. ``u`` (``d2Fdu2``) -- a
            {py:class}`ufl.ZeroBaseForm` if that term is structurally zero, so callers
            can always assemble it unconditionally. Built by the same code
            ({py:func}`~dolfinx_adjoint.solvers._build_soa_self_template`) for both classes; it simply always
            compiles to zero for a linear problem, since ``dF/du`` doesn't
            reference ``u`` there -- not a hardcoded special case. A list of
            one compiled form per output row (not a single form) for a
            blocked problem, since the SOA right-hand side is itself block
            structured then.
        soa_cross: The SOA right-hand-side's contribution, per dependency,
            from that dependency's tangent-linear direction. Also a list of
            one form per output row, per dependency, for a blocked problem.
        fixed: The part of each dependency's own Hessian-action output that
            does not depend on any *other* dependency's tangent-linear value.
            One form per dependency regardless of blocking -- it lives on the
            *control's* own test space, not the (possibly blocked) state's.
        cross: Each dependency's Hessian-action contribution from *another*
            dependency's tangent-linear direction, keyed by ``(c, c2)``. Same
            per-dependency (not per-row) shape as ``fixed``.
    """

    soa_self: NestedSequence[dolfinx.fem.Form]
    soa_cross: dict
    fixed: dict
    cross: dict


def _pad_blocks_by_part(form: ufl.Form, test_funcs: typing.Sequence[ufl.Argument]) -> list[ufl.Form | ufl.ZeroBaseForm]:
    """Split a blocked one-form into one entry per ``test_funcs`` part, in part order.

    {py:func}`ufl.extract_blocks` only returns an entry for a part that actually appears in
    ``form`` -- a part a differentiation happened to eliminate entirely (e.g. a
    dependency that only appears in one output block's equation) is simply absent from
    its result, not returned as an explicit zero. Padding with {py:class}`ufl.ZeroBaseForm` for
    every missing part keeps every caller's per-row list the same length and order as
    ``test_funcs``, so it can always be assembled/indexed positionally.

    ``form.empty()`` is checked before calling {py:func}`ufl.extract_blocks`: a form that is
    structurally empty (no arguments at all, e.g. ``d2Fdu2`` for a linear residual) has
    no parts for {py:func}`ufl.extract_blocks` to find, so every row is padded to zero directly
    instead.
    """
    padded: list[ufl.Form | ufl.ZeroBaseForm] = [ufl.ZeroBaseForm((test,)) for test in test_funcs]
    if form.empty():
        return padded
    assert isinstance(form, ufl.Form)
    for block in ufl.extract_blocks(form):
        args = block.arguments()
        assert len(args) == 1, "Expected a single test function in the block."
        padded[args[0].part()] = block
    return padded


def _build_soa_self_template(
    dFdu_template: ufl.Form,
    state_placeholder: dolfinx.fem.Function,
    hessian_u_seed: dolfinx.fem.Function,
    adjoint_solution_placeholder: dolfinx.fem.Function,
    *,
    jit_options: dict | None,
    form_compiler_options: dict | None,
    entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None,
) -> NestedSequence[dolfinx.fem.Form]:
    """Build the SOA self-term ``adjoint(d2F/du2) . adjoint_solution``.

    The same computation for both {py:class}`~dolfinx_adjoint.LinearProblem` and
    {py:class}`~dolfinx_adjoint.NonlinearProblem`: ``d2F/du2`` is structurally zero
    for a linear residual (``dF/du`` doesn't reference ``u``), so this compiles to
    a {py:class}`ufl.ZeroBaseForm` for {py:class}`~dolfinx_adjoint.LinearProblem`
    as a *result* of running the same code, not as a special case -- callers can
    always assemble the returned form unconditionally, mirroring how an inactive
    TLM/fixed template is represented elsewhere in this module (e.g.
    {py:meth}`~dolfinx_adjoint.solvers._ProblemBase._get_or_build_tlm_rhs_templates`).
    """
    d2Fdu2 = ufl.algorithms.expand_derivatives(ufl.derivative(dFdu_template, state_placeholder, hessian_u_seed))
    if d2Fdu2.empty():
        soa_self_form = ufl.ZeroBaseForm((dFdu_template.arguments()[0],))
    else:
        soa_self_form = ufl.action(ufl.adjoint(d2Fdu2), adjoint_solution_placeholder)
    return dolfinx.fem.form(  # type: ignore[call-overload]
        soa_self_form,
        jit_options=jit_options,
        form_compiler_options=form_compiler_options,
        entity_maps=entity_maps,
    )


class _ProblemBase(abc.ABC):
    """Shared lazy adjoint/TLM solver machinery for
    {py:class}`~dolfinx_adjoint.LinearProblem`/{py:class}`~dolfinx_adjoint.NonlinearProblem`.
    """

    # A plain mixin -- not a subclass of any dolfinx.fem.petsc class -- so
    # LinearProblem(_ProblemBase, dolfinx.fem.petsc.LinearProblem) (and the
    # NonlinearProblem equivalent) gets an unambiguous MRO for solve(): each
    # subclass keeps defining its own solve(), delegating the shared middle to
    # _record_and_solve() below via the _make_block()/_dolfinx_solve() hooks it
    # implements.

    # Every attribute below is set by the concrete subclass's own __init__
    # (either directly, or inherited from the dolfinx.fem.petsc base it also
    # derives from) before any method on this class runs.
    ad_block_tag: str | None
    bcs: typing.Sequence[dolfinx.fem.DirichletBC]
    _u: _Function | typing.Sequence[_Function]
    _rhs: typing.Any
    _preconditioner: typing.Any
    _value_placeholders: dict[dolfinx.fem.Function, dolfinx.fem.Function]
    _jit_options: dict | None
    _form_compiler_options: dict | None
    _entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None
    _adj_options: dict | None
    _tlm_options: dict | None
    _petsc_options_prefix: str
    _kind: typing.Any

    @property
    def value_placeholders(self) -> dict[dolfinx.fem.Function, dolfinx.fem.Function]:
        """Map from each non-``u`` dependency to its dedicated placeholder coefficient."""
        # Public so *ProblemBlock (blocks/solvers.py) can refresh a dependency's
        # placeholder value ahead of a solve/replay without reaching into this
        # Problem's private state.
        return self._value_placeholders

    @property
    def residual_state_placeholder(self) -> dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]:
        """The dedicated "state" placeholder(s) standing in for ``u`` in every
        compiled template built from the residual (see
        {py:meth}`~dolfinx_adjoint.solvers._ProblemBase._get_or_build_residual_template`).
        A single Function for a scalar problem, or one per output block for a
        blocked problem.
        """
        assert self._residual_state_placeholder is not None
        return self._residual_state_placeholder

    @property
    def adjoint_solution_placeholder(self) -> dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]:
        """The placeholder(s) for the first-order adjoint solution in the cached
        Hessian templates (see
        {py:meth}`~dolfinx_adjoint.solvers._ProblemBase._get_or_build_hessian_templates`).
        """
        assert self._adjoint_solution_placeholder is not None
        return self._adjoint_solution_placeholder

    @property
    def second_adjoint_solution_placeholder(self) -> dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]:
        """The placeholder(s) for the second-order adjoint (SOA) solution in the
        cached Hessian templates (see
        {py:meth}`~dolfinx_adjoint.solvers._ProblemBase._get_or_build_hessian_templates`).
        """
        assert self._second_adjoint_solution_placeholder is not None
        return self._second_adjoint_solution_placeholder

    @property
    def hessian_u_seed(self) -> dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]:
        """The placeholder(s) for the state's own tangent-linear direction in the
        cached Hessian self-term (see
        {py:meth}`~dolfinx_adjoint.solvers._ProblemBase._get_or_build_hessian_templates`).
        """
        assert self._hessian_u_seed is not None
        return self._hessian_u_seed

    def _init_adjoint_state(self) -> None:
        """Initialize the lazily-built adjoint/TLM solver state."""
        # Called once, at the end of __init__, after the base dolfinx.fem.petsc
        # solver is constructed. Everything below is built lazily, on first use
        # (see _get_or_build_adjoint_solver/_get_or_build_tlm_solver/
        # _get_or_build_hessian_templates), so pure forward (non-annotated) use
        # never pays for a symbolic adjoint form it doesn't need.
        #
        # Shared by every block this Problem records rather than rebuilt per
        # block/solve() call. Each block holds only a weakref back here, not a
        # strong reference -- see *ProblemBlock.get_reference_problem
        # (blocks/solvers.py) for why, and how a block copes if this Problem is
        # collected before it's done needing it.
        self._adjoint_solver: HomogeneousBCLinearProblem | None = None
        self._tlm_solver: HomogeneousBCLinearProblem | None = None
        self._residual_template: ufl.Form | None = None
        self._residual_state_placeholder: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function] | None = None
        self._dFdu_template: ufl.Form | None = None
        self._dFdu_adj_template: ufl.Form | typing.Sequence | None = None
        self._tlm_rhs_templates: dict | None = None
        self._tlm_seed_placeholders: dict[dolfinx.fem.Function, dolfinx.fem.Function] = {}
        self._hessian_templates: HessianTemplates | None = None
        self._adjoint_solution_placeholder: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function] | None = None
        self._second_adjoint_solution_placeholder: (
            dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function] | None
        ) = None
        self._hessian_u_seed: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function] | None = None
        self._adjoint_reaction_template: dolfinx.fem.Form | NestedSequence[dolfinx.fem.Form] | None = None
        self._second_order_adjoint_reaction_template: dolfinx.fem.Form | NestedSequence[dolfinx.fem.Form] | None = None

    @abc.abstractmethod
    def _get_or_build_residual_template(
        self,
    ) -> tuple[ufl.Form, dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]]:
        """Build (once) and return F with every coefficient replaced by its placeholder,
        and u replaced by a dedicated "state" placeholder for "u at this evaluation point".

        Each subclass overrides this -- the one genuinely irreducible difference
        between the two Problem kinds (LinearProblem builds it from ``a``/``L``
        via {py:func}`ufl.action`; NonlinearProblem already has ``F`` directly).
        Everything derived below (``dF/du``, TLM right-hand side, Hessian
        templates) shares this one template, via the same symbolic
        differentiation for both kinds.

        Returns:
            A ``(F_template, state_placeholder)`` pair: the residual with every
            coefficient (including ``u``) substituted by its placeholder, and
            that state placeholder itself (single, or one per output block for
            a blocked problem).
        """

    def _get_or_build_dFdu_template(self) -> ufl.Form:
        """Build (once) and return dF/du, evaluated at the residual template's state placeholder."""
        # Shared by both classes: derived from _get_or_build_residual_template by
        # symbolic differentiation (free at compile time) rather than
        # special-cased per class. Basis for the adjoint operator
        # (_get_or_build_adjoint_solver, which just adjoints it) and the TLM
        # operator (_get_or_build_tlm_solver, used as-is). Built once for the
        # life of this Problem -- callers only ever refresh the placeholders'
        # values afterwards (see *ProblemBlock.prepare_evaluate_adj/
        # prepare_evaluate_hessian/prepare_evaluate_tlm in blocks/solvers.py).
        if self._dFdu_template is None:
            F_template, state_placeholder = self._get_or_build_residual_template()
            if isinstance(self._u, list):
                assert isinstance(state_placeholder, typing.Sequence)
                test_functions = get_sorted_arguments(F_template.arguments(), 0)
                state_list = list(state_placeholder)
                trial_functions = [
                    ufl.TrialFunction(state.function_space, part=arg.part())
                    for arg, state in zip(test_functions, state_list, strict=True)
                ]
                dFdu = ufl.derivative(F_template, state_list, trial_functions)
            else:
                assert isinstance(state_placeholder, dolfinx.fem.Function)
                trial_function = ufl.TrialFunction(state_placeholder.function_space)
                dFdu = ufl.derivative(F_template, state_placeholder, trial_function)
            self._dFdu_template = ufl.algorithms.expand_derivatives(dFdu)
        return self._dFdu_template

    def _get_or_build_dFdu_adj_template(self) -> ufl.Form | typing.Sequence:
        """Build (once) and return adjoint(dF/du).

        Shared by the adjoint solver (`_get_or_build_adjoint_solver`) and, for
        scalar problems, the Hessian SOA right-hand side's cross-dependency
        templates (`_get_or_build_hessian_templates`).
        """
        if self._dFdu_adj_template is None:
            # compute_adjoint() (ufl_utils.py) swaps argument numbers while
            # preserving mixed-space ufl.Argument.part tags, then decomposes back
            # into blocks via ufl.extract_blocks -- a no-op for a scalar form.
            # Kept exactly as that returns it (a nested list of forms for a
            # blocked problem), since that's the shape HomogeneousBCLinearProblem/
            # dolfinx.fem.petsc.LinearProblem need for block matrix assembly;
            # callers wanting a single summed form (Hessian templating,
            # scalar-only) apply ufl_utils.sum_form() themselves.
            self._dFdu_adj_template = compute_adjoint(
                self._get_or_build_dFdu_template()  # type: ignore[arg-type]
            )
        return self._dFdu_adj_template

    def _get_or_build_tlm_rhs_templates(
        self,
    ) -> tuple[
        dict[dolfinx.fem.Function, typing.Any],
        dict[dolfinx.fem.Function, dolfinx.fem.Function],
        dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function],
    ]:
        """Build (once) and return the per-dependency TLM right-hand-side templates."""
        # Shared by both classes: built purely from the residual template
        # (_get_or_build_residual_template), since dF/dm genuinely depends on the
        # state for either a linear or nonlinear residual -- differentiating
        # w.r.t. a coefficient embedded in the residual while holding u fixed
        # leaves u in the result even when the residual is linear in u itself.
        F_template, state_placeholder = self._get_or_build_residual_template()
        if self._tlm_rhs_templates is None:
            if isinstance(self._u, list):
                test_funcs = list(get_sorted_arguments(F_template.arguments(), 0))
            else:
                test_funcs = [F_template.arguments()[0]]

            templates: dict[dolfinx.fem.Function, typing.Any] = {}
            # One compiled one-form per dependency, using a dedicated "direction"
            # placeholder (below, cached in _tlm_seed_placeholders) rather than a
            # single form summed over every dependency: summing symbolically
            # would require deciding, once and for all, which dependencies
            # contribute, but that varies from call to call. Zeroing an unused
            # dependency's seed and evaluating its term anyway isn't a safe
            # substitute for skipping it -- if that dependency appears somewhere
            # singular at its current value (e.g. a 1/c term with c legitimately
            # zero somewhere), the assembled contribution is 0 * inf = NaN there
            # even though the seed is zero, silently corrupting the sum. Keeping
            # each dependency as its own compiled form, only ever assembled when
            # it actually has a tangent-linear value (see
            # *ProblemBlock.prepare_evaluate_tlm in blocks/solvers.py), avoids
            # that entirely.
            for c, c_placeholder in self._value_placeholders.items():
                seed = dolfinx.fem.Function(c.function_space)
                dFdm_c = ufl.algorithms.expand_derivatives(-ufl.derivative(F_template, c_placeholder, seed))
                if isinstance(self._u, list):
                    dFdm_c = _pad_blocks_by_part(dFdm_c, test_funcs)
                else:
                    if dFdm_c == 0 or dFdm_c.empty():
                        dFdm_c = ufl.ZeroBaseForm((test_funcs[0],))
                templates[c] = dolfinx.fem.form(
                    dFdm_c,
                    jit_options=self._jit_options,
                    form_compiler_options=self._form_compiler_options,
                    entity_maps=self._entity_maps,
                )
                self._tlm_seed_placeholders[c] = seed
            self._tlm_rhs_templates = templates
        return self._tlm_rhs_templates, self._tlm_seed_placeholders, state_placeholder

    def _ensure_hessian_placeholders(
        self,
    ) -> dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]:
        """Build (once) the adjoint_solution_placeholder/second_adjoint_solution_placeholder/
        hessian_u_seed placeholders, independently of the rest of the (more expensive) Hessian
        templates.

        Factored out of `_get_or_build_hessian_templates` so that
        `_get_or_build_adjoint_reaction_template` (a boundary-control gradient, needed even
        when no Hessian is ever requested) can share the same placeholder without pulling in
        the full Hessian machinery -- see `dolfinx-adjoint-knowledge`'s
        `scratch/boundary-control/spec.md` for the design rationale.

        Returns:
            The (possibly newly built) adjoint_solution_placeholder.
        """
        if self._adjoint_solution_placeholder is None:
            _, state_placeholder = self._get_or_build_residual_template()
            if isinstance(self._u, list):
                assert isinstance(state_placeholder, typing.Sequence)
                state_list = list(state_placeholder)
                self._adjoint_solution_placeholder = [dolfinx.fem.Function(s.function_space) for s in state_list]
                self._second_adjoint_solution_placeholder = [dolfinx.fem.Function(s.function_space) for s in state_list]
                self._hessian_u_seed = [dolfinx.fem.Function(s.function_space) for s in state_list]
            else:
                assert isinstance(state_placeholder, dolfinx.fem.Function)
                self._adjoint_solution_placeholder = dolfinx.fem.Function(state_placeholder.function_space)
                self._second_adjoint_solution_placeholder = dolfinx.fem.Function(state_placeholder.function_space)
                self._hessian_u_seed = dolfinx.fem.Function(state_placeholder.function_space)
        assert self._adjoint_solution_placeholder is not None
        return self._adjoint_solution_placeholder

    def _get_or_build_adjoint_reaction_template(
        self,
    ) -> dolfinx.fem.Form | NestedSequence[dolfinx.fem.Form]:
        """Build (once) and return ``action(adjoint(dF/du), adjoint_solution_placeholder)``,
        compiled with **no bcs applied at all** -- the boundary-control gradient recipe. See
        `dolfinx-adjoint-knowledge`'s `scratch/boundary-control/spec.md` for the full
        derivation and why no bcs are applied here.

        Deliberately a separate, lighter-weight method from `_get_or_build_hessian_templates`
        (which needs the *symbolic* form to keep differentiating further for soa_cross) --
        this one only ever needs the *compiled*, directly assemblable form, and must not
        force building the rest of the (more expensive) Hessian machinery for problems that
        never request a Hessian.
        """
        if self._adjoint_reaction_template is None:
            placeholder = self._ensure_hessian_placeholders()
            self._adjoint_reaction_template = self._build_adjoint_reaction_template(placeholder)
        return self._adjoint_reaction_template

    def _get_or_build_second_order_adjoint_reaction_template(
        self,
    ) -> dolfinx.fem.Form | NestedSequence[dolfinx.fem.Form]:
        """Build (once) and return ``action(adjoint(dF/du), second_adjoint_solution_placeholder)``
        -- the Hessian-side counterpart of `_get_or_build_adjoint_reaction_template`, sharing
        the same ``adjoint(dF/du)`` operator (the SOA equation's LHS is verbatim the
        first-order adjoint equation's, see
        `blocks/solvers.py::_ProblemBlockBase.prepare_evaluate_hessian`) but evaluated at the
        second-order adjoint solution instead of the first-order one. This is the *entire*
        Hessian-action contribution for a Dirichlet bc control -- see
        `dolfinx-adjoint-knowledge`'s `scratch/boundary-control/spec.md` for why.
        """
        if self._second_order_adjoint_reaction_template is None:
            self._ensure_hessian_placeholders()
            assert self._second_adjoint_solution_placeholder is not None
            self._second_order_adjoint_reaction_template = self._build_adjoint_reaction_template(
                self._second_adjoint_solution_placeholder
            )
        return self._second_order_adjoint_reaction_template

    def _build_adjoint_reaction_template(
        self, adjoint_placeholder: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]
    ) -> dolfinx.fem.Form | NestedSequence[dolfinx.fem.Form]:
        """Compile ``action(adjoint(dF/du), adjoint_placeholder)``, with no bcs applied.

        Shared builder for `_get_or_build_adjoint_reaction_template` (first-order) and
        `_get_or_build_second_order_adjoint_reaction_template` (SOA) -- identical except for
        which adjoint-solution placeholder is applied.
        """
        dFdu_adj_template = sum_form(self._get_or_build_dFdu_adj_template())  # type: ignore[arg-type]
        assert isinstance(dFdu_adj_template, ufl.Form)
        reaction_form = ufl.action(dFdu_adj_template, adjoint_placeholder)
        if isinstance(self._u, list):
            F_template, _ = self._get_or_build_residual_template()
            test_funcs = list(get_sorted_arguments(F_template.arguments(), 0))
            return [
                dolfinx.fem.form(  # type: ignore[return-value]
                    form_i,  # type: ignore[arg-type]
                    jit_options=self._jit_options,
                    form_compiler_options=self._form_compiler_options,
                    entity_maps=self._entity_maps,
                )
                for form_i in _pad_blocks_by_part(reaction_form, test_funcs)
            ]
        else:
            return dolfinx.fem.form(
                reaction_form,
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )

    def _get_or_build_hessian_templates(self) -> HessianTemplates:
        """Build (once) and return the per-dependency Hessian templates.

        Feeds *ProblemBlock.prepare_evaluate_hessian's SOA right-hand side and
        evaluate_hessian_component's Hessian-action output (blocks/solvers.py).
        Shared by both classes and by scalar and blocked problems, built
        entirely from the other cached templates (`_get_or_build_residual_template`,
        `_get_or_build_dFdu_template`, `_get_or_build_dFdu_adj_template`,
        `_get_or_build_tlm_rhs_templates`).
        """
        if self._hessian_templates is None:
            _, seed_placeholders, state_placeholder = self._get_or_build_tlm_rhs_templates()
            F_template, _ = self._get_or_build_residual_template()
            dFdu_template = self._get_or_build_dFdu_template()
            dFdu_adj_template = sum_form(self._get_or_build_dFdu_adj_template())  # type: ignore[arg-type]
            assert isinstance(dFdu_template, ufl.Form)
            assert isinstance(dFdu_adj_template, ufl.Form)

            self._ensure_hessian_placeholders()
            assert self._hessian_u_seed is not None
            assert self._adjoint_solution_placeholder is not None
            blocked = isinstance(self._u, list)
            soa_self: NestedSequence[dolfinx.fem.Form]
            if blocked:
                assert isinstance(state_placeholder, typing.Sequence)
                state_list = list(state_placeholder)
                test_funcs = list(get_sorted_arguments(F_template.arguments(), 0))
                state_arg: typing.Any = state_list

                # soa_self = adjoint(d2F/du2) . adjoint_solution -- the SOA
                # right-hand side's contribution from dF/du's own second
                # derivative w.r.t. u. d2Fdu2 is structurally zero whenever the
                # residual is linear in u (dF/du doesn't reference u), so this
                # comes out a ufl.ZeroBaseForm for LinearProblem as a *result*
                # of running the same code, not a per-class branch (see
                # _build_soa_self_template, used below for the scalar case).
                d2Fdu2 = ufl.algorithms.expand_derivatives(
                    ufl.derivative(dFdu_template, state_list, self._hessian_u_seed)
                )
                if d2Fdu2.empty():
                    soa_self_form = d2Fdu2
                else:
                    soa_self_form = ufl.action(ufl.adjoint(d2Fdu2), self._adjoint_solution_placeholder)
                # soa_self feeds the (blocked) SOA right-hand-side vector, so it
                # becomes a list of one compiled form per output row here,
                # padded via _pad_blocks_by_part for any row a differentiation
                # happened to eliminate entirely.
                soa_self = [
                    dolfinx.fem.form(
                        form_i,  # type: ignore[arg-type]
                        jit_options=self._jit_options,
                        form_compiler_options=self._form_compiler_options,
                        entity_maps=self._entity_maps,
                    )
                    for form_i in _pad_blocks_by_part(soa_self_form, test_funcs)
                ]
            else:
                assert isinstance(state_placeholder, dolfinx.fem.Function)
                assert isinstance(self._hessian_u_seed, dolfinx.fem.Function)
                assert isinstance(self._adjoint_solution_placeholder, dolfinx.fem.Function)
                state_arg = state_placeholder

                soa_self = _build_soa_self_template(
                    dFdu_template,
                    state_placeholder,
                    self._hessian_u_seed,
                    self._adjoint_solution_placeholder,
                    jit_options=self._jit_options,
                    form_compiler_options=self._form_compiler_options,
                    entity_maps=self._entity_maps,
                )

            # dFdu_adj_applied/L1/L2 are shared building blocks for every
            # dependency's soa_cross/fixed/cross templates below: dFdu_adj_applied
            # is dF/du^T applied to the first-order adjoint solution (the base
            # for each dependency's soa_cross term), L1/L2 are the residual
            # applied to the first/second-order adjoint solutions respectively
            # (the base for each dependency's fixed/cross terms).
            dFdu_adj_applied = ufl.action(dFdu_adj_template, self._adjoint_solution_placeholder)
            L1 = ufl.action(F_template, self._adjoint_solution_placeholder)
            L2 = ufl.action(F_template, self._second_adjoint_solution_placeholder)

            soa_cross_templates: dict = {}
            fixed_templates: dict = {}
            cross_templates: dict = {}
            # fixed/cross live on each control's own test space, so their shape
            # is unaffected by blocking. soa_self/soa_cross do feed the
            # (possibly blocked) SOA right-hand side, so for a blocked problem
            # each becomes a list of one compiled form per output row (padded
            # via _pad_blocks_by_part for any row a differentiation eliminated).
            #
            # Each cross-term below uses its own dedicated seed_placeholders
            # direction placeholder rather than one combined form, for the same
            # 0 * inf = NaN reason as _get_or_build_tlm_rhs_templates.
            for c, c_placeholder in self._value_placeholders.items():
                seed = seed_placeholders[c]

                # soa_cross[c]: the SOA right-hand side's contribution from c's
                # own tangent-linear direction, via dFdu_adj_applied.
                soa_form = ufl.algorithms.expand_derivatives(ufl.derivative(dFdu_adj_applied, c_placeholder, seed))
                if not (soa_form == 0 or soa_form.empty()):
                    if blocked:
                        soa_cross_templates[c] = [
                            dolfinx.fem.form(
                                form_i,  # type: ignore[arg-type]
                                jit_options=self._jit_options,
                                form_compiler_options=self._form_compiler_options,
                                entity_maps=self._entity_maps,
                            )
                            for form_i in _pad_blocks_by_part(soa_form, test_funcs)
                        ]
                    else:
                        soa_cross_templates[c] = dolfinx.fem.form(
                            soa_form,
                            jit_options=self._jit_options,
                            form_compiler_options=self._form_compiler_options,
                            entity_maps=self._entity_maps,
                        )

                # fixed[c]: c's own Hessian-action contribution that does not
                # depend on any *other* dependency's tangent-linear value --
                # the second-order-adjoint term (dL2dm, from L2) plus the
                # mixed state/control second derivative (d2Fdudm, from L1).
                dc = ufl.TestFunction(c.function_space)
                dL1dm = ufl.derivative(L1, c_placeholder, dc)
                dL2dm = ufl.derivative(L2, c_placeholder, dc)
                d2Fdudm = ufl.algorithms.expand_derivatives(ufl.derivative(dL1dm, state_arg, self._hessian_u_seed))
                fixed_form = ufl.algorithms.expand_derivatives(dL2dm + d2Fdudm)
                if fixed_form == 0 or fixed_form.empty():
                    fixed_form = ufl.ZeroBaseForm((dc,))
                fixed_templates[c] = dolfinx.fem.form(
                    fixed_form,
                    jit_options=self._jit_options,
                    form_compiler_options=self._form_compiler_options,
                    entity_maps=self._entity_maps,
                )

                # cross[(c, c2)]: c's Hessian-action contribution from another
                # dependency c2's tangent-linear direction, reusing dL1dm.
                for c2, c2_placeholder in self._value_placeholders.items():
                    seed2 = seed_placeholders[c2]
                    cross_form = ufl.algorithms.expand_derivatives(ufl.derivative(dL1dm, c2_placeholder, seed2))
                    if cross_form == 0 or cross_form.empty():
                        continue
                    cross_templates[(c, c2)] = dolfinx.fem.form(
                        cross_form,
                        jit_options=self._jit_options,
                        form_compiler_options=self._form_compiler_options,
                        entity_maps=self._entity_maps,
                    )
            self._hessian_templates = HessianTemplates(soa_self, soa_cross_templates, fixed_templates, cross_templates)
        return self._hessian_templates

    def _get_or_build_adjoint_solver(self) -> HomogeneousBCLinearProblem:
        """Build (once) and return the adjoint solver shared by every block this Problem records."""
        if self._adjoint_solver is None:
            self._adjoint_solver = HomogeneousBCLinearProblem(
                self._get_or_build_dFdu_adj_template(),  # type: ignore[arg-type]
                self._rhs,
                bcs=self.bcs,
                P=self._preconditioner,
                form_compiler_options=self._form_compiler_options,
                jit_options=self._jit_options,
                petsc_options=self._adj_options,
                petsc_options_prefix=f"{self._petsc_options_prefix}adjoint_",
                kind=self._kind,
                entity_maps=self._entity_maps,
            )  # type: ignore[misc]
        return self._adjoint_solver

    def _get_or_build_tlm_solver(self) -> HomogeneousBCLinearProblem:
        """Build (once) and return the TLM solver shared by every block this Problem records."""
        # No explicit u= is passed: like the adjoint solver, this gets its own
        # scratch solution Function from the base class, and callers copy the
        # result out (see *ProblemBlock.prepare_evaluate_tlm, blocks/solvers.py)
        # rather than relying on solver-owned storage identity, since that
        # storage is now shared across every block instead of private to one.
        if self._tlm_solver is None:
            dFdu_template = self._get_or_build_dFdu_template()  # type: ignore[attr-defined]
            if isinstance(self._u, list):
                # Unlike the adjoint operator (which decomposes dF/du back into
                # blocks itself, inside compat.compute_form_adjoint/
                # ufl_utils.compute_adjoint), dF/du is used here as-is, so for a
                # blocked problem it must be decomposed with ufl.extract_blocks
                # before compiling: a summed multi-part form is a perfectly good
                # UFL object to keep substituting into and differentiating, but
                # not, on its own, a compilable one -- the parts must be split
                # apart first.
                dFdu_template = ufl.extract_blocks(dFdu_template)
            self._tlm_solver = HomogeneousBCLinearProblem(
                dFdu_template,
                self._rhs,
                bcs=self.bcs,
                P=self._preconditioner,
                form_compiler_options=self._form_compiler_options,
                jit_options=self._jit_options,
                petsc_options=self._tlm_options,
                petsc_options_prefix=f"{self._petsc_options_prefix}tlm_",
                kind=self._kind,
                entity_maps=self._entity_maps,
            )  # type: ignore[misc]
        return self._tlm_solver

    @abc.abstractmethod
    def _make_block(self) -> _ProblemBlockBase:
        """Construct the tape block this Problem records for its forward solve.

        Each subclass overrides this to instantiate its own Block kind
        (LinearProblemBlock/NonlinearProblemBlock, blocks/solvers.py; constructor
        kwargs differ the same way the two Problem kinds' own constructors do),
        passing ``self`` so the Block can reach back into this Problem's shared
        solvers (see *ProblemBlock.get_reference_problem).

        Returns:
            A newly constructed, not-yet-recorded Block for this solve.
        """

    @abc.abstractmethod
    def _dolfinx_solve(self) -> _Function | typing.Sequence[_Function]:
        """Perform the actual forward solve, via the base ``dolfinx.fem.petsc`` class.

        Each subclass overrides this to call its own base class's ``solve()``
        directly (``dolfinx.fem.petsc.LinearProblem.solve``/
        ``NonlinearProblem.solve``) rather than ``self.solve()``, which would
        recurse back into this Problem's own overridden, tape-recording
        ``solve()``.

        Returns:
            The solution Function, or one per output block for a blocked problem.
        """

    def _record_and_solve(self, annotate: bool) -> _Function | typing.Sequence[_Function]:
        """Shared ``solve()`` skeleton for both classes.

        Args:
            annotate: Whether to record this solve as a block on the working tape.

        Returns:
            The solution Function, or one per output block for a blocked problem.
        """
        annotate = pyadjoint.annotate_tape({"annotate": annotate})
        block = self._make_block() if annotate else None
        if annotate:
            assert block is not None
            tape = pyadjoint.get_working_tape()
            tape.add_block(block)

        # Refresh the forward solver's placeholders from the user's own current
        # values: a prior recompute (see *ProblemBlock.prepare_recompute_component,
        # blocks/solvers.py) may have left them holding a checkpointed/candidate
        # value instead.
        for original, placeholder in self._value_placeholders.items():
            placeholder.x.array[:] = original.x.array[:]
            placeholder.x.scatter_forward()

        out = self._dolfinx_solve()
        if annotate:
            assert block is not None
            if isinstance(out, Function):
                block.add_output(out.create_block_variable())
            else:
                for ui in out:
                    assert isinstance(ui, Function)
                    block.add_output(ui.create_block_variable())
        return out


class LinearProblem(_ProblemBase, dolfinx.fem.petsc.LinearProblem):
    """A linear problem that can be used with adjoint methods.

    This class extends the `dolfinx.fem.petsc.LinearProblem` to support adjoint methods.

    Args:
        a: The bilinear form representing the left-hand side of the equation.
        L: The linear form representing the right-hand side of the equation.
        bcs: Boundary conditions to apply to the problem.
        u: Solution vector.
        P: Preconditioner for the linear problem.
        kind: Kind of PETSc Matrix to assemble the system into.
        petsc_options: Options dictionary for the PETSc krylov supspace solver.
        petsc_options_prefix: Options prefix for the PETSc solver -- auto-generated,
            unique per `LinearProblem`, if not supplied.
        form_compiler_options: Form compiler options for generating assembly kernels.
        jit_options: Options for just-in-time compilation of the forms.
        entity_maps: Mapping from meshes that coefficients and arguments are defined on to the
            integration domain of the forms.
        ad_block_tag: Tag for adjoint blocks in the tape.
        adjoint_petsc_options: PETSc options for adjoint problems.
        tlm_petsc_options: Optional PETSc options for TLM problems.
    """

    @typing.overload
    def __init__(
        self,
        a: ufl.Form,
        L: ufl.Form,
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: _Function | None = None,
        P: ufl.Form | None = None,
        kind: str | None = None,
        petsc_options: dict | None = None,
        petsc_options_prefix: str | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        a: typing.Sequence[typing.Sequence[ufl.Form]],
        L: typing.Sequence[ufl.Form],
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: typing.Sequence[_Function] | None = None,
        P: typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        kind: str | typing.Sequence[typing.Sequence[str]] | None = None,
        petsc_options: dict | None = None,
        petsc_options_prefix: str | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
    ) -> None: ...
    def __init__(
        self,
        a: ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]],
        L: ufl.Form | typing.Sequence[ufl.Form],
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: _Function | typing.Sequence[_Function] | None = None,
        P: ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        kind: str | typing.Sequence[typing.Sequence[str]] | None = None,
        petsc_options: dict | None = None,
        petsc_options_prefix: str | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
    ) -> None:
        self.ad_block_tag = ad_block_tag
        self._adj_options = adjoint_petsc_options
        self._tlm_options = tlm_petsc_options

        # If a form is blocked from the user-side, it can be made without
        # a {py:class}`ufl.MixedFunctionSpace`. Therefore we modify the
        # form to use a {py:class}`ufl.MixedFunctionSpace` and split the form into its
        # components.
        if not isinstance(a, ufl.Form):
            a, L = assign_mixed_parts(a, L)  # type: ignore[arg-type]
            if P is not None:
                P, _ = assign_mixed_parts(P, L)  # type: ignore[arg-type]

        self._u = find_or_create_then_overload(u, L)  # type: ignore[arg-type]

        # Unique, synchronized prefix for every solver instance (as SNES requires sync in prefix
        # across processes).
        if petsc_options_prefix is None:
            petsc_options_prefix = f"dxa_linear_problem_{next(_PROBLEM_PREFIX_COUNTER)}_"

        # Cache some objects
        self._lhs = a
        self._rhs = L
        self._petsc_options = petsc_options
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._petsc_options_prefix = petsc_options_prefix
        self._kind = kind

        # We replace all input coefficients with local coefficients,
        # so that we don't disturb the input data during re-computations
        # adjoints, etc. The local coefficients are stored in self._value_placeholders.
        # The unknown `u` is not replaced, as it shouldn't be part of a linear problem's coefficients.
        u_list = self._u if isinstance(self._u, list) else [self._u]
        coefficients = collect_coefficients(a) | collect_coefficients(L)
        if set(u_list).issubset(coefficients):
            raise ValueError("The unknown `u` should not be part of the coefficients of a linear problem.")
        if P is not None:
            coefficients |= collect_coefficients(P)
            if set(u_list).issubset(coefficients):
                raise ValueError("The unknown `u` should not be part of the coefficients of a linear problem.")
        # Has to be sorted when creating placeholders, as function creation is a collective operation
        sorted_coefficients = sorted(coefficients, key=lambda c: c.ufl_id())
        self._value_placeholders: dict[dolfinx.fem.Function, dolfinx.fem.Function] = {
            c: dolfinx.fem.Function(c.function_space) for c in sorted_coefficients
        }
        a_R, L_R, P_R = recursive_replace((a, L, P), self._value_placeholders)  # type: ignore[misc]
        super().__init__(
            a=a_R,  # type: ignore[arg-type]
            L=L_R,  # type: ignore[arg-type]
            bcs=bcs,
            u=self._u,  # type: ignore[arg-type]
            P=P_R,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            petsc_options_prefix=petsc_options_prefix,
            petsc_options=petsc_options,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )  # type: ignore[misc]

        # Match the adjoint/TLM solvers' matrix layout to whatever `kind` the
        # forward solver actually resolved to (kind=None can auto-resolve to
        # "nest" for blocked problems).
        self._kind = "nest" if self.A.getType() == "nest" else kind

        # Adjoint and tangent-linear solver state: shared lazy-init machinery
        # lives in _ProblemBase._init_adjoint_state -- see its docstring for
        # why laziness and Problem-owned (not block-owned) solvers matter.
        self._init_adjoint_state()

    def _get_or_build_residual_template(
        self,
    ) -> tuple[ufl.Form, dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]]:
        """Build (once) and return F = action(a, state) - L, placeholder-substituted, with a
        dedicated "state" placeholder standing in for "u at this evaluation point", distinct
        from the live ``self._u`` the forward solve owns.

        The shared basis for ``dF/du``
        ({py:meth}`_ProblemBase._get_or_build_dFdu_template<dolfinx_adjoint.solvers._ProblemBase._get_or_build_dFdu_template>`)
        and the TLM right-hand side
        ({py:meth}`~dolfinx_adjoint.solvers._ProblemBase._get_or_build_tlm_rhs_templates`):
        ``dF/dm`` genuinely depends on the state even for a linear problem
        (``a`` is bilinear, so differentiating w.r.t. a coefficient embedded in
        ``a`` while holding ``u`` fixed leaves ``u`` in the result), and
        ``dF/du`` itself is now derived from this template by the same symbolic
        differentiation {py:class}`~dolfinx_adjoint.NonlinearProblem` uses,
        rather than a shortcut that skips differentiating altogether.
        """
        if self._residual_template is None:
            u_list = self._u if isinstance(self._u, list) else [self._u]
            if isinstance(self._u, list):
                self._residual_state_placeholder = [
                    dolfinx.fem.Function(ui.function_space)  # type: ignore[union-attr]
                    for ui in u_list
                ]
                state_arg: typing.Any = self._residual_state_placeholder
            else:
                self._residual_state_placeholder = dolfinx.fem.Function(self._u.function_space)  # type: ignore[union-attr]
                state_arg = self._residual_state_placeholder

            a_template = ufl.replace(sum_form(self._lhs), self._value_placeholders)  # type: ignore[arg-type]
            L_template = ufl.replace(sum_form(self._rhs), self._value_placeholders)  # type: ignore[arg-type]
            self._residual_template = ufl.action(a_template, state_arg) - L_template  # type: ignore[arg-type]
        return self._residual_template, self._residual_state_placeholder  # type: ignore[return-value]

    def _make_block(self) -> LinearProblemBlock:
        return LinearProblemBlock(
            self._lhs,  # type: ignore[arg-type]
            self._rhs,  # type: ignore[arg-type]
            bcs=self.bcs,
            u=self.u,  # type: ignore[arg-type]
            P=self._preconditioner,  # type: ignore[arg-type]
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            entity_maps=self._entity_maps,
            kind=self._kind,
            petsc_options=self._petsc_options,
            petsc_options_prefix=self._petsc_options_prefix,
            adjoint_petsc_options=self._adj_options,
            tlm_petsc_options=self._tlm_options,
            ad_block_tag=self.ad_block_tag,
            problem=self,
        )  # type: ignore[misc]

    def _dolfinx_solve(self) -> _Function | typing.Sequence[_Function]:
        return dolfinx.fem.petsc.LinearProblem.solve(self)

    def solve(self, annotate: bool = True) -> dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]:
        """
        Solve the linear problem and return the solution.
        """
        return self._record_and_solve(annotate)


class NonlinearProblem(_ProblemBase, dolfinx.fem.petsc.NonlinearProblem):
    """A nonlinear problem that can be used with adjoint methods.

    This class extends the `dolfinx.fem.petsc.NonlinearProblem` to support adjoint methods.

    Args:
        F: The residual form.
        u: Solution vector.
        bcs: Boundary conditions to apply to the problem.
        J: The Jacobian form. Computed from ``F`` if not supplied.
        P: Preconditioner for the nonlinear problem.
        kind: Kind of PETSc Matrix to assemble the system into.
        petsc_options: Options dictionary for the PETSc SNES solver.
        petsc_options_prefix: Options prefix for the PETSc solver -- auto-generated,
            unique per `NonlinearProblem`, if not supplied.
        form_compiler_options: Form compiler options for generating assembly kernels.
        jit_options: Options for just-in-time compilation of the forms.
        entity_maps: Mapping from meshes that coefficients and arguments are defined on to the
            integration domain of the forms.
        ad_block_tag: Tag for adjoint blocks in the tape.
        adjoint_petsc_options: PETSc options for adjoint problems.
        tlm_petsc_options: Optional PETSc options for TLM problems.
    """

    @typing.overload
    def __init__(
        self,
        F: ufl.form.Form,
        u: _Function,
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        J: ufl.form.Form | None = None,
        P: ufl.form.Form | None = None,
        kind: str | None = None,
        petsc_options: dict | None = None,
        petsc_options_prefix: str | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        F: typing.Sequence[ufl.form.Form],
        u: typing.Sequence[_Function],
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        J: typing.Sequence[typing.Sequence[ufl.form.Form]] | None = None,
        P: typing.Sequence[typing.Sequence[ufl.form.Form]] | None = None,
        kind: str | typing.Sequence[typing.Sequence[str]] | None = None,
        petsc_options: dict | None = None,
        petsc_options_prefix: str | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
    ) -> None: ...
    def __init__(
        self,
        F: ufl.form.Form | typing.Sequence[ufl.form.Form],
        u: _Function | typing.Sequence[_Function],
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        J: ufl.form.Form | typing.Sequence[typing.Sequence[ufl.form.Form]] | None = None,
        P: ufl.form.Form | typing.Sequence[typing.Sequence[ufl.form.Form]] | None = None,
        kind: str | typing.Sequence[typing.Sequence[str]] | None = None,
        petsc_options: dict | None = None,
        petsc_options_prefix: str | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
    ) -> None:

        self.ad_block_tag = ad_block_tag
        self._adj_options = adjoint_petsc_options
        self._tlm_options = tlm_petsc_options

        # If a form is blocked from the user-side, it can be made without
        # a {py:class}`ufl.MixedFunctionSpace`. Therefore we modify the
        # form to use a {py:class}`ufl.MixedFunctionSpace` and split the form into its
        # components.
        if not isinstance(F, ufl.Form):
            F = assign_mixed_parts(F)  # type: ignore[arg-type]

        self._u = find_or_create_then_overload(u, F)  # type: ignore[arg-type]
        self._bcs = [] if bcs is None else bcs

        # Unique, synchronized prefix for every solver instance (as SNES requires sync in prefix
        # across processes).
        if petsc_options_prefix is None:
            petsc_options_prefix = f"dxa_nonlinear_problem_{next(_PROBLEM_PREFIX_COUNTER)}_"

        # The user's own J, kept only to scan for dependency coefficients that
        # might appear in a hand-supplied Jacobian but not in F itself (e.g. a
        # stabilization term); the Jacobian _get_or_build_dFdu_template uses
        # for adjoint/TLM/Hessian purposes is always derived symbolically from
        # F, never from this.
        self._user_J = J
        self._rhs = F
        self._petsc_options = petsc_options
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._petsc_options_prefix = petsc_options_prefix
        self._kind = kind

        # We replace all input coefficients with local coefficients,
        # so that we don't disturb the input data during re-computations
        # adjoints, etc. The local coefficients are stored in self._value_placeholders.
        # The unknown is not replaced in the residual, but when deriving the
        # adjoint and TLM solutions, through `_residual_state_placeholder`
        u_list = self._u if isinstance(self._u, list) else [self._u]
        coefficients = collect_coefficients(F) - set(u_list)
        if J is not None:
            coefficients |= collect_coefficients(J) - set(u_list)
        self._value_placeholders: dict[dolfinx.fem.Function, dolfinx.fem.Function] = {
            c: dolfinx.fem.Function(c.function_space) for c in coefficients
        }

        # Initialize nonlinear solver
        F_R, J_R, P_R = recursive_replace((F, J, P), self._value_placeholders)  # type: ignore[misc]
        super().__init__(
            F=F_R,  # type: ignore[arg-type]
            J=J_R,  # type: ignore[arg-type]
            P=P_R,  # type: ignore[arg-type]
            bcs=self._bcs,
            u=self._u,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            petsc_options_prefix=petsc_options_prefix,
            petsc_options=petsc_options,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )  # type: ignore[misc]

        # Adjoint and tangent-linear solver state: shared lazy-init machinery
        # lives in _ProblemBase._init_adjoint_state -- see LinearProblem's use
        # of it, and its docstring, for the rationale (same for both classes).
        self._init_adjoint_state()

    @property
    def bcs(self) -> typing.Sequence[dolfinx.fem.DirichletBC]:
        """Dirichlet boundary conditions applied to the residual and Jacobian.

        {py:class}`dolfinx.fem.petsc.NonlinearProblem` has no ``bcs`` attribute of its
        own (its SNES callbacks close over a fixed ``bcs`` list at construction); this
        property exposes ``self._bcs`` under the same name {py:class}`~dolfinx_adjoint.LinearProblem` uses (there, it is
        the base class's own attribute), so {py:class}`~dolfinx_adjoint.solvers._ProblemBase`'s shared methods
        can read/write ``self.bcs`` uniformly across both classes.
        """
        return self._bcs

    @bcs.setter
    def bcs(self, value: typing.Sequence[dolfinx.fem.DirichletBC]) -> None:
        self._bcs = value

    def _get_or_build_residual_template(
        self,
    ) -> tuple[ufl.Form, dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]]:
        """Build (once) and return F with every non-u coefficient replaced by its placeholder,
        and u itself replaced by a dedicated "state" placeholder standing in for "u at this
        evaluation point", distinct from the live ``self._u`` the forward SNES path owns.

        The shared basis for ``dF/du``
        ({py:meth}`_ProblemBase._get_or_build_dFdu_template<dolfinx_adjoint.solvers._ProblemBase._get_or_build_dFdu_template>`)
        and the TLM right-hand side
        ({py:meth}`~dolfinx_adjoint.solvers._ProblemBase._get_or_build_tlm_rhs_templates`):
        refreshed from a block's own checkpointed output before each
        adjoint/TLM/Hessian solve (see
        {py:meth}`NonlinearProblemBlock._refresh_dFdu_state<dolfinx_adjoint.blocks.solvers.NonlinearProblemBlock._refresh_dFdu_state>`/
        {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.prepare_evaluate_tlm`),
        keeping this template fixed for the life of the Problem, exactly like
        the non-u dependencies already routed through ``self._value_placeholders``.
        """
        if self._residual_template is None:
            u_list = self._u if isinstance(self._u, list) else [self._u]
            if isinstance(self._u, list):
                self._residual_state_placeholder = [
                    dolfinx.fem.Function(ui.function_space)  # type: ignore[union-attr]
                    for ui in u_list
                ]
                state_list = self._residual_state_placeholder
            else:
                self._residual_state_placeholder = dolfinx.fem.Function(self._u.function_space)  # type: ignore[union-attr]
                state_list = [self._residual_state_placeholder]
            replace_map: dict = dict(self._value_placeholders)
            replace_map.update(zip(u_list, state_list))

            if isinstance(self._rhs, ufl.Form):
                self._residual_template = ufl.replace(self._rhs, replace_map)
            else:
                self._residual_template = sum_form([ufl.replace(Fi, replace_map) for Fi in self._rhs])
        return self._residual_template, self._residual_state_placeholder  # type: ignore[return-value]

    def _make_block(self) -> NonlinearProblemBlock:
        return NonlinearProblemBlock(
            J=self._user_J,  # type: ignore[arg-type]
            F=self._rhs,  # type: ignore[arg-type]
            bcs=self.bcs,
            u=self.u,  # type: ignore[arg-type]
            P=self._preconditioner,  # type: ignore[arg-type]
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            entity_maps=self._entity_maps,
            kind=self._kind,
            petsc_options=self._petsc_options,
            petsc_options_prefix=self._petsc_options_prefix,
            adjoint_petsc_options=self._adj_options,
            tlm_petsc_options=self._tlm_options,
            ad_block_tag=self.ad_block_tag,
            problem=self,
        )  # type: ignore[misc]

    def _dolfinx_solve(self) -> _Function | typing.Sequence[_Function]:
        return dolfinx.fem.petsc.NonlinearProblem.solve(self)

    def solve(self, annotate: bool = True) -> dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]:
        """
        Solve the nonlinear problem and return the solution.
        """
        return self._record_and_solve(annotate)
