from __future__ import annotations

import typing

import dolfinx.fem.petsc
import pyadjoint
import ufl
from dolfinx.fem.function import Function as _Function

from .blocks.solvers import (
    LinearProblemBlock,
    NonlinearProblemBlock,
    _collect_coefficients,
    _ProblemBlockBase,
    assign_mixed_parts,
    get_sorted_arguments,
    sum_form,
)
from .petsc_utils import HomogeneousBCLinearProblem
from .types import Function


def _replace_with_placeholders(
    form: ufl.Form | typing.Sequence | None, placeholders: dict
) -> ufl.Form | typing.Sequence | None:
    """Recursively apply ``ufl.replace(form, placeholders)`` to a (possibly nested) form
    structure.

    A module-level function, not a nested closure: a nested function that
    recurses by calling itself by name captures *itself* as a free variable,
    which makes the function object (and, via ``self`` if the closure also
    needs it) part of a reference cycle -- collected only by the cyclic
    garbage collector, at a moment that differs between MPI ranks, not by
    ordinary refcounting. That is exactly the hazard ``Problem`` owning its
    solvers (rather than each ``Block``) exists to avoid: a self-referential
    ``_replace`` closure inside ``LinearProblem``/``NonlinearProblem.__init__``
    would keep the ``Problem`` itself -- and its PETSc solvers -- alive as
    cyclic garbage. Taking ``placeholders`` as a plain argument instead of
    capturing ``self`` sidesteps this entirely: a module-level function
    referring to itself by name is looked up through the module's namespace,
    not a closure cell, so no cycle is created.
    """
    if form is None:
        return None
    if isinstance(form, ufl.Form):
        return ufl.replace(form, placeholders)
    return [_replace_with_placeholders(f, placeholders) for f in form]


@typing.overload
def resolve_u(u: _Function | None, L: ufl.Form) -> _Function: ...
@typing.overload
def resolve_u(u: typing.Sequence[_Function] | None, L: typing.Sequence[ufl.Form]) -> typing.Sequence[_Function]: ...


def resolve_u(
    u: _Function | typing.Sequence[_Function] | None, L: ufl.Form | typing.Sequence[ufl.Form]
) -> _Function | typing.Sequence[_Function]:
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

    Shared shape for ``LinearProblem``/``NonlinearProblem``, so callers in
    ``blocks/solvers.py`` unpack by name rather than by position -- the two
    classes used to return differently-shaped tuples here, which was a latent
    footgun for any code touching both.

    Attributes:
        soa_self: The SOA right-hand-side's contribution from ``dF/du``'s own
            second derivative w.r.t. ``u`` (``d2Fdu2``) -- a
            ``ufl.ZeroBaseForm`` if that term is structurally zero, so callers
            can always assemble it unconditionally. Built by the same code
            (``_build_soa_self_template``) for both classes; it simply always
            compiles to zero for a linear problem, since ``dF/du`` doesn't
            reference ``u`` there -- not a hardcoded special case.
        soa_cross: The SOA right-hand-side's contribution, per dependency,
            from that dependency's tangent-linear direction.
        fixed: The part of each dependency's own Hessian-action output that
            does not depend on any *other* dependency's tangent-linear value.
        cross: Each dependency's Hessian-action contribution from *another*
            dependency's tangent-linear direction, keyed by ``(c, c2)``.
    """

    soa_self: dolfinx.fem.Form
    soa_cross: dict
    fixed: dict
    cross: dict


def _build_soa_self_template(
    dFdu_template: ufl.Form,
    state_placeholder: dolfinx.fem.Function,
    hessian_u_seed: dolfinx.fem.Function,
    adjoint_solution_placeholder: dolfinx.fem.Function,
    *,
    jit_options: dict | None,
    form_compiler_options: dict | None,
    entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None,
) -> dolfinx.fem.Form:
    """Build the SOA self-term ``adjoint(d2F/du2) . adjoint_solution``.

    The same computation for both ``LinearProblem`` and ``NonlinearProblem``:
    ``d2F/du2`` is structurally zero for a linear residual (``dF/du`` doesn't
    reference ``u``), so this compiles to a ``ufl.ZeroBaseForm`` for
    ``LinearProblem`` as a *result* of running the same code, not as a special
    case -- callers can always assemble the returned form unconditionally,
    mirroring how an inactive TLM/fixed template is represented elsewhere in
    this module (e.g. ``_get_or_build_tlm_rhs_templates``).
    """
    d2Fdu2 = ufl.algorithms.expand_derivatives(ufl.derivative(dFdu_template, state_placeholder, hessian_u_seed))
    if d2Fdu2.empty():
        soa_self_form = ufl.ZeroBaseForm((dFdu_template.arguments()[0],))
    else:
        soa_self_form = ufl.action(ufl.adjoint(d2Fdu2), adjoint_solution_placeholder)
    return dolfinx.fem.form(
        soa_self_form,
        jit_options=jit_options,
        form_compiler_options=form_compiler_options,
        entity_maps=entity_maps,
    )


class _ProblemBase:
    """Shared lazy adjoint/TLM solver machinery for ``LinearProblem``/``NonlinearProblem``.

    A plain mixin -- it does not inherit from any ``dolfinx.fem.petsc`` class,
    so ``LinearProblem(_ProblemBase, dolfinx.fem.petsc.LinearProblem)`` (and
    the ``NonlinearProblem`` equivalent) has an unambiguous MRO for
    ``solve()``: each subclass keeps defining its own ``solve()``, delegating
    the shared middle to ``_record_and_solve`` below via the ``_make_block``/
    ``_dolfinx_solve`` hooks it implements.

    Every attribute referenced here is set by the concrete subclass's own
    ``__init__`` (either directly, or inherited from the ``dolfinx.fem.petsc``
    base it also derives from) before any of these methods run.
    """

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

    def _init_adjoint_state(self) -> None:
        """Initialize the lazily-built adjoint/TLM solver state.

        Called once, at the end of ``__init__``, after the base
        ``dolfinx.fem.petsc`` solver has been constructed. Built lazily (on
        first use, see ``_get_or_build_adjoint_solver``/
        ``_get_or_build_tlm_solver``/``_get_or_build_hessian_templates``) and
        shared by every block this Problem records, rather than one per
        block/solve() call. Each block holds a plain (strong) reference back
        to this Problem (see ``*ProblemBlock._problem``/``__init__`` for why a
        strong reference is safe here and doesn't reintroduce the MPI
        collective-destruction hazard documented in
        dolfinx-adjoint-knowledge's mpi-collective-destruction-hazard note),
        so this Problem -- and hence its solvers' PETSc objects -- is released
        deterministically via ordinary refcounting once every block
        referencing it is unreachable, rather than being left to pyadjoint's
        tape/cyclic-GC schedule. Laziness keeps pure forward (non-annotated)
        use from paying for a symbolic adjoint form it never needs.
        """
        self._adjoint_solver: HomogeneousBCLinearProblem | None = None
        self._tlm_solver: HomogeneousBCLinearProblem | None = None
        self._residual_template: ufl.Form | None = None
        self._residual_state_placeholder: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function] | None = None
        self._dFdu_template: ufl.Form | typing.Sequence | None = None
        self._dFdu_adj_template: ufl.Form | typing.Sequence | None = None
        self._tlm_rhs_templates: dict | None = None
        self._tlm_seed_placeholders: dict[dolfinx.fem.Function, dolfinx.fem.Function] = {}
        self._hessian_templates: HessianTemplates | None = None
        self._adjoint_solution_placeholder: dolfinx.fem.Function | None = None
        self._second_adjoint_solution_placeholder: dolfinx.fem.Function | None = None
        self._hessian_u_seed: dolfinx.fem.Function | None = None

    def _get_or_build_residual_template(
        self,
    ) -> tuple[ufl.Form, dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]]:
        """Build (once) and return F with every coefficient replaced by its placeholder, and u
        replaced by a dedicated "state" placeholder standing in for "u at this evaluation point".

        The one genuinely irreducible difference between the two Problem kinds -- LinearProblem
        builds it from ``a``/``L`` via ``ufl.action``, NonlinearProblem already has ``F``
        directly -- everything built on top of it below (``dF/du``, the TLM right-hand side, the
        Hessian templates) is derived from this one template by the same shared symbolic
        differentiation, since that costs nothing extra at compile time: there is no reason to
        keep a separate "dF/du is just a, never differentiated" shortcut for LinearProblem.
        Not implemented on the base; each subclass overrides it.
        """
        raise NotImplementedError

    def _get_or_build_dFdu_template(self) -> ufl.Form | typing.Sequence:
        """Build (once) and return dF/du, evaluated at the residual template's state placeholder.

        Shared by both classes: derived from ``_get_or_build_residual_template`` by symbolic
        differentiation, which is free at compile time, rather than special-cased per class.
        This is the shared basis for the adjoint operator (``_get_or_build_adjoint_solver``,
        which just adjoints it) and the TLM operator (``_get_or_build_tlm_solver``, used as-is):
        built once, for the life of this Problem, so neither ever needs to rebuild or recompile
        it -- only refresh the placeholders' values (see
        ``*ProblemBlock.prepare_evaluate_adj``/``prepare_evaluate_hessian``/``prepare_evaluate_tlm``).
        """
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
        """Build (once) and return adjoint(dF/du), shared by the adjoint solver
        (``_get_or_build_adjoint_solver``) and, for scalar problems, the Hessian
        SOA right-hand-side's cross-dependency templates
        (``_get_or_build_hessian_templates``).

        The same computation for both classes: ``_ProblemBlockBase._compute_adjoint``
        swaps argument numbers while preserving mixed-space ``part()`` tags (via
        ``compute_form_adjoint``) and decomposes the result back into blocks (via
        ``ufl.extract_blocks``) -- a no-op decomposition for a scalar, non-blocked
        form. Kept exactly as that returns it (a nested list of forms for a blocked
        problem) since that structure is what ``HomogeneousBCLinearProblem``/
        ``dolfinx.fem.petsc.LinearProblem`` needs for block matrix assembly; callers
        that need a single summed form (Hessian templating, scalar-only) apply
        ``sum_form`` themselves.
        """
        if self._dFdu_adj_template is None:
            self._dFdu_adj_template = _ProblemBlockBase._compute_adjoint(
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
        """Build (once) and return the per-dependency TLM right-hand-side templates.

        Shared by both classes: built purely from the residual template
        (``_get_or_build_residual_template``), since ``dF/dm`` genuinely depends on the state
        for either a linear or nonlinear residual -- differentiating w.r.t. a coefficient
        embedded in the residual while holding ``u`` fixed leaves ``u`` in the result even when
        the residual is linear in ``u`` itself.

        One compiled one-form is built per dependency, using a dedicated "direction"
        placeholder for that dependency (``_tlm_seed_placeholders``) rather than a single
        combined form summed over every dependency: summing symbolically would require
        deciding, once and for all, which dependencies contribute, but which ones actually have
        a tangent-linear value varies from call to call. Refreshing an unused dependency's seed
        to zero and evaluating its term anyway is not a safe substitute for skipping it: if that
        dependency appears in a way that is singular at its current value (e.g. a `1/c` term,
        with `c` legitimately zero somewhere in the domain), the assembled contribution would be
        `0 * inf = NaN` there even though the *seed* is zero, silently corrupting the sum.
        Keeping every dependency's contribution as its own compiled form, only ever assembled
        when that dependency actually has a tangent-linear value (see
        ``*ProblemBlock.prepare_evaluate_tlm``), avoids that entirely.
        """
        F_template, state_placeholder = self._get_or_build_residual_template()
        if self._tlm_rhs_templates is None:
            if isinstance(self._u, list):
                test_funcs = list(get_sorted_arguments(F_template.arguments(), 0))
            else:
                test_funcs = [F_template.arguments()[0]]

            templates: dict[dolfinx.fem.Function, typing.Any] = {}
            for c, c_placeholder in self._value_placeholders.items():
                seed = dolfinx.fem.Function(c.function_space)
                dFdm_c = ufl.algorithms.expand_derivatives(-ufl.derivative(F_template, c_placeholder, seed))
                if isinstance(self._u, list):
                    blocks = ufl.extract_blocks(dFdm_c)
                    padded = [ufl.ZeroBaseForm((test,)) for test in test_funcs]
                    for block in blocks:
                        args = block.arguments()
                        assert len(args) == 1, "Expected a single test function in the block."
                        padded[args[0].part()] = block
                    dFdm_c = padded
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

    def _get_or_build_hessian_templates(self) -> HessianTemplates:
        """Build (once) and return the per-dependency Hessian templates used by
        ``*ProblemBlock.prepare_evaluate_hessian``'s SOA right-hand side and
        ``evaluate_hessian_component``'s own Hessian-action output.

        Shared by both classes -- built purely from ``_get_or_build_residual_template``/
        ``_get_or_build_dFdu_template``/``_get_or_build_dFdu_adj_template``/
        ``_get_or_build_tlm_rhs_templates``, all themselves shared. ``soa_self`` (see
        ``_build_soa_self_template``) comes out a ``ufl.ZeroBaseForm`` for ``LinearProblem``
        (``dF/du`` doesn't depend on ``u``) and generally nonzero for ``NonlinearProblem``, as a
        *result* of running the same code, not a per-class branch.

        Scalar (non-blocked) problems only -- the blocked Hessian cross-term stays on the
        pre-templating, per-call ``ufl.replace`` + recompile path (see
        ``_ProblemBlockBase._evaluate_hessian_blocked_rhs``/``_evaluate_hessian_component_blocked``).

        Each of ``soa_cross``/``fixed``/``cross`` is kept as its own compiled one-form, using a
        dedicated "direction" placeholder (the same ``_tlm_seed_placeholders`` the TLM
        right-hand side already uses -- safe to share, since the TLM forward sweep has always
        finished computing every tangent-linear value before the reverse (adjoint/Hessian)
        sweep that needs these runs), for the same reason as ``_get_or_build_tlm_rhs_templates``:
        summing every dependency's cross-term contribution into one combined form and zeroing an
        inactive dependency's seed has the same ``0 * inf = NaN`` hazard there does.
        """
        if self._hessian_templates is None:
            assert not isinstance(self._u, list), "Hessian templating is only implemented for scalar problems."
            _, seed_placeholders, state_placeholder = self._get_or_build_tlm_rhs_templates()
            F_template, _ = self._get_or_build_residual_template()
            dFdu_template = self._get_or_build_dFdu_template()
            dFdu_adj_template = sum_form(self._get_or_build_dFdu_adj_template())  # type: ignore[arg-type]
            assert isinstance(dFdu_template, ufl.Form)
            assert isinstance(dFdu_adj_template, ufl.Form)
            assert isinstance(state_placeholder, dolfinx.fem.Function)

            self._adjoint_solution_placeholder = dolfinx.fem.Function(state_placeholder.function_space)
            self._second_adjoint_solution_placeholder = dolfinx.fem.Function(state_placeholder.function_space)
            self._hessian_u_seed = dolfinx.fem.Function(state_placeholder.function_space)

            soa_self = _build_soa_self_template(
                dFdu_template,
                state_placeholder,
                self._hessian_u_seed,
                self._adjoint_solution_placeholder,
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )

            dFdu_adj_applied = ufl.action(dFdu_adj_template, self._adjoint_solution_placeholder)
            L1 = ufl.action(F_template, self._adjoint_solution_placeholder)
            L2 = ufl.action(F_template, self._second_adjoint_solution_placeholder)

            soa_cross_templates: dict = {}
            fixed_templates: dict = {}
            cross_templates: dict = {}
            for c, c_placeholder in self._value_placeholders.items():
                seed = seed_placeholders[c]

                soa_form = ufl.algorithms.expand_derivatives(ufl.derivative(dFdu_adj_applied, c_placeholder, seed))
                if not (soa_form == 0 or soa_form.empty()):
                    soa_cross_templates[c] = dolfinx.fem.form(
                        soa_form,
                        jit_options=self._jit_options,
                        form_compiler_options=self._form_compiler_options,
                        entity_maps=self._entity_maps,
                    )

                dc = ufl.TestFunction(c.function_space)
                dL1dm = ufl.derivative(L1, c_placeholder, dc)
                dL2dm = ufl.derivative(L2, c_placeholder, dc)
                d2Fdudm = ufl.algorithms.expand_derivatives(
                    ufl.derivative(dL1dm, state_placeholder, self._hessian_u_seed)
                )
                fixed_form = ufl.algorithms.expand_derivatives(dL2dm + d2Fdudm)
                if fixed_form == 0 or fixed_form.empty():
                    fixed_form = ufl.ZeroBaseForm((dc,))
                fixed_templates[c] = dolfinx.fem.form(
                    fixed_form,
                    jit_options=self._jit_options,
                    form_compiler_options=self._form_compiler_options,
                    entity_maps=self._entity_maps,
                )

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
        """Build (once) and return the TLM solver shared by every block this Problem records.

        No explicit ``u=`` is passed: like the adjoint solver, this gets its own
        scratch solution Function from the base class, and callers copy the
        result out (see ``*ProblemBlock.prepare_evaluate_tlm``) rather than
        relying on solver-owned storage identity, since that storage is now
        shared across every block instead of private to one.

        Unlike the adjoint operator (which decomposes dF/du back into blocks
        itself, inside ``compute_form_adjoint``/``_compute_adjoint``), dF/du
        is used here as-is, so for a blocked problem it must be decomposed
        with ``ufl.extract_blocks`` before compiling: a summed multi-part
        form is a perfectly good UFL object to keep substituting into and
        differentiating, but it is not, on its own, a compilable one -- the
        parts must be split apart first.
        """
        if self._tlm_solver is None:
            dFdu_template = self._get_or_build_dFdu_template()  # type: ignore[attr-defined]
            if isinstance(self._u, list):
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

    def _record_and_solve(self, annotate: bool) -> _Function | typing.Sequence[_Function]:
        """Shared ``solve()`` skeleton for both classes.

        Records a tape block (via the subclass's ``_make_block`` hook) when
        annotating, refreshes the forward solver's placeholder coefficients
        from the user's own current values (a prior recompute -- see
        ``*ProblemBlock.prepare_recompute_component`` -- may have left them
        holding a checkpointed/candidate value instead), solves (via the
        subclass's ``_dolfinx_solve`` hook), and records the block's outputs.
        """
        annotate = pyadjoint.annotate_tape({"annotate": annotate})
        block = self._make_block() if annotate else None  # type: ignore[attr-defined]
        if annotate:
            assert block is not None
            tape = pyadjoint.get_working_tape()
            tape.add_block(block)

        for original, placeholder in self._value_placeholders.items():
            placeholder.x.array[:] = original.x.array[:]
            placeholder.x.scatter_forward()

        out = self._dolfinx_solve()  # type: ignore[attr-defined]
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
        petsc_options_prefix: str = "dxa_linear_problem_",
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
        petsc_options_prefix: str = "dxa_linear_problem_",
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
        petsc_options_prefix: str = "dxa_linear_problem_",
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

        # Assign mixed-space `part` indices to Test/Trial arguments once,
        # here, for blocked systems (mirroring what LinearProblemBlock used to
        # redo per block): needed so a blocked bilinear/linear form can be
        # safely combined into one whole-system form (via sum_form) when
        # building the adjoint solver below.
        if not isinstance(a, ufl.Form):
            a, L = assign_mixed_parts(a, L)  # type: ignore[arg-type]
            if P is not None:
                P, _ = assign_mixed_parts(P, L)  # type: ignore[arg-type]

        self._u = resolve_u(u, L)  # type: ignore[arg-type]

        # Cache some objects
        self._lhs = a
        self._rhs = L
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._petsc_options = petsc_options
        self._petsc_options_prefix = petsc_options_prefix
        self._kind = kind

        # The forward solver's compiled forms reference dedicated placeholder
        # coefficients rather than the user's own dependency objects --
        # exactly like NonlinearProblem, so both classes share the same
        # data-handling story: a solve always means "refresh the
        # placeholders' values, then call the solver", never "recompile a
        # form" or "mutate the user's own coefficient in place". solve()
        # (below) refreshes them from the user's own current values;
        # LinearProblemBlock.prepare_recompute_component refreshes them from
        # a block's checkpointed/candidate values instead. Neither ever
        # writes into the user's own coefficient objects, so a Taylor test
        # that perturbs the original control directly
        # (`pyadjoint.taylor_test(Jh, m, dm)`) always sees a pristine `m`.
        u_list = self._u if isinstance(self._u, list) else [self._u]
        coefficients = _collect_coefficients(a) | _collect_coefficients(L)
        if P is not None:
            coefficients |= _collect_coefficients(P)
        coefficients -= set(u_list)
        self._value_placeholders: dict[dolfinx.fem.Function, dolfinx.fem.Function] = {
            c: dolfinx.fem.Function(c.function_space) for c in coefficients
        }

        # Initialize linear solver
        super().__init__(
            a=_replace_with_placeholders(a, self._value_placeholders),  # type: ignore[arg-type]
            L=_replace_with_placeholders(L, self._value_placeholders),  # type: ignore[arg-type]
            bcs=bcs,
            u=self._u,  # type: ignore[arg-type]
            P=_replace_with_placeholders(P, self._value_placeholders),  # type: ignore[arg-type]
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

        The shared basis for ``dF/du`` (``_ProblemBase._get_or_build_dFdu_template``) and the
        TLM right-hand side (``_get_or_build_tlm_rhs_templates``): ``dF/dm`` genuinely depends
        on the state even for a linear problem (``a`` is bilinear, so differentiating w.r.t. a
        coefficient embedded in ``a`` while holding ``u`` fixed leaves ``u`` in the result), and
        ``dF/du`` itself is now derived from this template by the same symbolic differentiation
        ``NonlinearProblem`` uses, rather than a shortcut that skips differentiating altogether.
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
            ad_block_tag=self.ad_block_tag,
            problem=self,
        )  # type: ignore[misc]

    def _dolfinx_solve(self) -> _Function | typing.Sequence[_Function]:
        return dolfinx.fem.petsc.LinearProblem.solve(self)

    def solve(self, annotate: bool = True) -> typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]]:
        """
        Solve the linear problem and return the solution.
        """
        return self._record_and_solve(annotate)


class NonlinearProblem(_ProblemBase, dolfinx.fem.petsc.NonlinearProblem):
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
        petsc_options_prefix: str = "dxa_nonlinear_problem_",
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
        petsc_options_prefix: str = "dxa_nonlinear_problem_",
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
        petsc_options_prefix: str = "dxa_nonlinear_problem_",
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

        # Assign mixed-space `part` indices to the test functions in a blocked
        # residual once, here, mirroring LinearProblem: needed so a blocked
        # residual's per-block forms can be safely combined into one
        # whole-system form (via sum_form) inside _get_or_build_residual_template.
        if not isinstance(F, ufl.Form):
            F = assign_mixed_parts(F)  # type: ignore[arg-type]

        self._u = resolve_u(u, F)  # type: ignore[arg-type]
        self._bcs = [] if bcs is None else bcs
        # The user's own J, kept only to scan for dependency coefficients that
        # might appear in a hand-supplied Jacobian but not in F itself (e.g. a
        # stabilization term); the Jacobian _get_or_build_dFdu_template uses
        # for adjoint/TLM/Hessian purposes is always derived symbolically from
        # F, never from this. Named distinctly from dolfinx.fem.petsc.NonlinearProblem's
        # own `_J` (its compiled Jacobian, set by super().__init__() below) to avoid
        # colliding with it.
        self._user_J = J
        self._rhs = F
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._petsc_options = petsc_options
        self._petsc_options_prefix = petsc_options_prefix
        self._kind = kind

        # The SNES built by super().__init__() below binds to the exact
        # compiled F/J/P Form objects passed to it, forever: its residual and
        # Jacobian callbacks close over those objects in a context dict set up
        # once (see dolfinx.fem.petsc.NonlinearProblem.__init__'s
        # jacobian_ctx/function_ctx), so reassigning self._F/self._J later --
        # the trick LinearProblem.solve() uses to switch between "live" and
        # "recompute" forms -- would have no effect on what the SNES actually
        # assembles. The only way to make the SNES see a different value for a
        # coefficient is to mutate the exact Function object its compiled
        # forms reference.
        #
        # To keep that mutation from ever touching an object the user (or a
        # Taylor test perturbing a control directly) holds a live reference
        # to, every non-u coefficient is routed through a dedicated
        # placeholder Function from the very start: the SNES is built against
        # F/J/P with every such coefficient replaced by its placeholder, and
        # the placeholders are (re)populated -- from the user's own current
        # values for an ordinary solve() (see solve() below), or from a
        # block's checkpointed/candidate values for a recompute (see
        # NonlinearProblemBlock.prepare_recompute_component) -- before every
        # solve, never the other way around.
        u_list = self._u if isinstance(self._u, list) else [self._u]
        coefficients = _collect_coefficients(F) - set(u_list)
        if J is not None:
            coefficients |= _collect_coefficients(J) - set(u_list)
        self._value_placeholders: dict[dolfinx.fem.Function, dolfinx.fem.Function] = {
            c: dolfinx.fem.Function(c.function_space) for c in coefficients
        }

        # Initialize nonlinear solver
        super().__init__(
            F=_replace_with_placeholders(F, self._value_placeholders),  # type: ignore[arg-type]
            J=_replace_with_placeholders(J, self._value_placeholders),  # type: ignore[arg-type]
            P=_replace_with_placeholders(P, self._value_placeholders),  # type: ignore[arg-type]
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

        ``dolfinx.fem.petsc.NonlinearProblem`` has no ``bcs`` attribute of its
        own (its SNES callbacks close over a fixed ``bcs`` list at
        construction, see the note in ``__init__``); this property exposes
        ``self._bcs`` under the same name ``LinearProblem`` uses (there, it is
        the base class's own attribute), so ``_ProblemBase``'s shared methods
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

        The shared basis for ``dF/du`` (``_ProblemBase._get_or_build_dFdu_template``) and the
        TLM right-hand side (``_get_or_build_tlm_rhs_templates``): refreshed from a block's own
        checkpointed output before each adjoint/TLM/Hessian solve (see
        ``NonlinearProblemBlock._refresh_dFdu_state``/``prepare_evaluate_tlm``), keeping this
        template fixed for the life of the Problem, exactly like the non-u dependencies already
        routed through ``self._value_placeholders``.
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
            ad_block_tag=self.ad_block_tag,
            problem=self,
        )  # type: ignore[misc]

    def _dolfinx_solve(self) -> _Function | typing.Sequence[_Function]:
        return dolfinx.fem.petsc.NonlinearProblem.solve(self)

    def solve(self, annotate: bool = True) -> typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]]:
        """
        Solve the nonlinear problem and return the solution.
        """
        return self._record_and_solve(annotate)
