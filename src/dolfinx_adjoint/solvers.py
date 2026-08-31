from __future__ import annotations

import typing

import dolfinx.fem.petsc
import pyadjoint
import ufl
from dolfinx.fem.function import Function as _Function

from .blocks.solvers import (
    LinearProblemBlock,
    NonlinearProblemBlock,
    assign_mixed_parts,
    get_sorted_arguments,
    sum_form,
)
from .petsc_utils import LinearAdjointProblem
from .types import Function


def _collect_coefficients(form: ufl.Form | typing.Sequence | None) -> set:
    """Return the set of UFL coefficients appearing anywhere in ``form``.

    ``form`` may be a single form or an arbitrarily nested sequence of forms
    (entries may be ``None``, e.g. a zero block in a blocked system). Plain set
    union rather than ``sum_form``: unlike summing, this never requires the
    sub-forms' arguments to be mutually compatible (e.g. carry matching
    ``part()`` tags), which blocked NonlinearProblem forms are not.
    """
    if form is None:
        return set()
    if isinstance(form, ufl.Form):
        return set(form.coefficients())
    coefficients: set = set()
    for f in form:
        coefficients |= _collect_coefficients(f)
    return coefficients


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


class LinearProblem(dolfinx.fem.petsc.LinearProblem):
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
        ad_block_tag: typing.Optional[str] = None,
        adjoint_petsc_options: typing.Optional[dict] = None,
        tlm_petsc_options: typing.Optional[dict] = None,
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
        ad_block_tag: typing.Optional[str] = None,
        adjoint_petsc_options: typing.Optional[dict] = None,
        tlm_petsc_options: typing.Optional[dict] = None,
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
        ad_block_tag: typing.Optional[str] = None,
        adjoint_petsc_options: typing.Optional[dict] = None,
        tlm_petsc_options: typing.Optional[dict] = None,
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

        # Adjoint and tangent-linear solvers: built lazily (on first use, see
        # _get_or_build_adjoint_solver/_get_or_build_tlm_solver below) and shared
        # by every LinearProblemBlock this Problem records, rather than one per
        # block/solve() call. Each block holds a plain (strong) reference back
        # to this Problem (see LinearProblemBlock._problem/__init__ for why a
        # strong reference is safe here and doesn't reintroduce the MPI
        # collective-destruction hazard documented in
        # dolfinx-adjoint-knowledge's mpi-collective-destruction-hazard note),
        # so this Problem -- and hence its solvers' PETSc objects -- is
        # released deterministically via ordinary refcounting once every block
        # referencing it is unreachable (e.g. after clearing the tape), rather
        # than being left to pyadjoint's tape/cyclic-GC schedule. Laziness
        # keeps pure forward (non-annotated) use from paying for a symbolic
        # adjoint form it never needs.
        self._adjoint_solver: typing.Optional[LinearAdjointProblem] = None
        self._tlm_solver: typing.Optional[LinearAdjointProblem] = None
        self._dFdu_template: typing.Optional[ufl.Form | typing.Sequence] = None
        self._dFdu_adj_template: typing.Optional[ufl.Form | typing.Sequence] = None
        self._tlm_rhs_templates: typing.Optional[dict] = None
        self._tlm_seed_placeholders: dict[dolfinx.fem.Function, dolfinx.fem.Function] = {}
        self._residual_state_placeholder: typing.Union[
            dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function], None
        ] = None
        self._hessian_templates: typing.Optional[tuple[dict, dict, dict]] = None
        self._adjoint_solution_placeholder: typing.Optional[dolfinx.fem.Function] = None
        self._second_adjoint_solution_placeholder: typing.Optional[dolfinx.fem.Function] = None
        self._hessian_u_seed: typing.Optional[dolfinx.fem.Function] = None

    def _get_or_build_dFdu_template(self) -> ufl.Form | typing.Sequence:
        """Build (once) and return dF/du with every non-u coefficient replaced by its
        placeholder.

        dF/du does not actually depend on the state u for a linear problem:
        F(u, v) = a(u, v) - L(v) is linear in u, so its derivative doesn't
        reference u's value at all, only whatever *other* coefficients a
        itself depends on. This is exactly a (placeholder-substituted), and
        is the shared basis for both the adjoint operator
        (``_get_or_build_adjoint_solver``, which just adjoints it) and the
        TLM operator (``_get_or_build_tlm_solver``, used as-is): built once,
        for the life of this Problem, so neither ever needs to rebuild or
        recompile it -- only refresh the placeholders' values (see
        ``LinearProblemBlock.prepare_evaluate_adj``/``prepare_evaluate_hessian``/``prepare_evaluate_tlm``).
        """
        if self._dFdu_template is None:
            self._dFdu_template = ufl.replace(sum_form(self._lhs), self._value_placeholders)  # type: ignore[arg-type]
        return self._dFdu_template

    def _get_or_build_dFdu_adj_template(self) -> ufl.Form | typing.Sequence:
        """Build (once) and return adjoint(dF/du), shared by the adjoint solver
        (``_get_or_build_adjoint_solver``) and, for scalar problems, the Hessian
        SOA right-hand-side's cross-dependency templates
        (``_get_or_build_hessian_templates``).

        Kept exactly as ``_compute_adjoint`` returns it -- a nested list of
        forms for a blocked problem -- since that structure is what
        ``LinearAdjointProblem``/``dolfinx.fem.petsc.LinearProblem`` needs for
        block matrix assembly; callers that need a single summed form (Hessian
        templating, scalar-only) apply ``sum_form`` themselves.
        """
        if self._dFdu_adj_template is None:
            self._dFdu_adj_template = LinearProblemBlock._compute_adjoint(
                self._get_or_build_dFdu_template()  # type: ignore[arg-type]
            )
        return self._dFdu_adj_template

    def _get_or_build_adjoint_solver(self) -> LinearAdjointProblem:
        """Build (once) and return the adjoint solver shared by every block this Problem records."""
        if self._adjoint_solver is None:
            self._adjoint_solver = LinearAdjointProblem(
                self._get_or_build_dFdu_adj_template(),  # type: ignore[arg-type]
                self._rhs,  # type: ignore[arg-type]
                bcs=self.bcs,
                P=self._preconditioner,  # type: ignore[arg-type]
                form_compiler_options=self._form_compiler_options,
                jit_options=self._jit_options,
                petsc_options=self._adj_options,
                petsc_options_prefix=f"{self._petsc_options_prefix}adjoint_",
                kind=self._kind,  # type: ignore[arg-type]
                entity_maps=self._entity_maps,
            )  # type: ignore[misc]
        return self._adjoint_solver

    def _get_or_build_tlm_solver(self) -> LinearAdjointProblem:
        """Build (once) and return the TLM solver shared by every block this Problem records.

        No explicit ``u=`` is passed: like the adjoint solver, this gets its own
        scratch solution Function from the base class, and callers copy the
        result out (see ``LinearProblemBlock.prepare_evaluate_tlm``) rather than
        relying on solver-owned storage identity, since that storage is now
        shared across every block instead of private to one.

        Unlike the adjoint operator (which decomposes dF/du back into blocks
        itself, inside ``compute_form_adjoint``/``_compute_adjoint``), dF/du
        is used here as-is, so for a blocked problem it must be decomposed
        with ``ufl.extract_blocks`` before compiling: a summed multi-part
        form is a perfectly good UFL object to keep substituting into and
        differentiating, but it is not, on its own, a compilable one -- the
        parts must be split apart first (mirroring
        ``_compute_residual_derivative``'s ``ufl.extract_blocks(dFdu)`` in the
        pre-templating code this replaces).
        """
        if self._tlm_solver is None:
            dFdu_template = self._get_or_build_dFdu_template()
            if isinstance(self._u, list):
                dFdu_template = ufl.extract_blocks(dFdu_template)  # type: ignore[arg-type]
            self._tlm_solver = LinearAdjointProblem(
                dFdu_template,  # type: ignore[arg-type]
                self._rhs,  # type: ignore[arg-type]
                bcs=self.bcs,
                P=self._preconditioner,  # type: ignore[arg-type]
                form_compiler_options=self._form_compiler_options,
                jit_options=self._jit_options,
                petsc_options=self._tlm_options,
                petsc_options_prefix=f"{self._petsc_options_prefix}tlm_",
                kind=self._kind,  # type: ignore[arg-type]
                entity_maps=self._entity_maps,
            )  # type: ignore[misc]
        return self._tlm_solver

    def _get_or_build_tlm_rhs_templates(
        self,
    ) -> tuple[
        dict[dolfinx.fem.Function, typing.Any],
        dict[dolfinx.fem.Function, dolfinx.fem.Function],
        typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]],
    ]:
        """Build (once) and return the per-dependency TLM right-hand-side templates.

        Unlike dF/du, dF/dm genuinely depends on the state u even for a
        linear problem (a is bilinear, so differentiating w.r.t. a
        coefficient embedded in a while holding u fixed leaves u in the
        result), so this needs its own "current state" placeholder,
        ``_residual_state_placeholder``, distinct from the live self._u the
        forward solve owns.

        One compiled one-form is built per dependency, using a dedicated
        "direction" placeholder for that dependency (``_tlm_seed_placeholders``)
        rather than a single combined form summed over every dependency:
        summing symbolically would require deciding, once and for all, which
        dependencies contribute, but which ones actually have a tangent-linear
        value varies from call to call. Refreshing an unused dependency's
        seed to zero and evaluating its term anyway is not a safe substitute
        for skipping it: if that dependency appears in a way that is singular
        at its current value (e.g. a `1/c` term, with `c` legitimately zero
        somewhere in the domain), the assembled contribution would be `0 *
        inf = NaN` there even though the *seed* is zero, silently corrupting
        the sum. Keeping every dependency's contribution as its own compiled
        form, only ever assembled when that dependency actually has a
        tangent-linear value (see ``LinearProblemBlock.prepare_evaluate_tlm``),
        avoids that entirely by never evaluating an inactive dependency's term
        at all -- exactly matching what skipping it symbolically did before.
        """
        if self._tlm_rhs_templates is None:
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

            a_template = self._get_or_build_dFdu_template()
            L_template = ufl.replace(sum_form(self._rhs), self._value_placeholders)  # type: ignore[arg-type]
            F_template = ufl.action(a_template, state_arg) - L_template  # type: ignore[arg-type]

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
        return self._tlm_rhs_templates, self._tlm_seed_placeholders, self._residual_state_placeholder  # type: ignore[return-value]

    def _get_or_build_hessian_templates(self) -> tuple[dict, dict, dict]:
        """Build (once) and return the per-dependency Hessian templates used by
        ``LinearProblemBlock.prepare_evaluate_hessian``'s SOA right-hand side and
        ``evaluate_hessian_component``'s own Hessian-action output.

        Scalar (non-blocked) problems only -- the blocked Hessian cross-term
        stays on the pre-templating, per-call ``ufl.replace`` + recompile path
        (see the module-level plan notes: this is deferred, ownership-move
        only, for the blocked case).

        Three dicts are returned:

        - ``soa_templates[c]``: the SOA right-hand-side's contribution from
          dependency ``c``'s tangent-linear direction, a 1-form in the state's
          own test function.
        - ``fixed_templates[c]``: the part of dependency ``c``'s own
          Hessian-action output that does not depend on any *other*
          dependency's tangent-linear value (``dL2dm + d2Fdudm``) -- always
          assembled.
        - ``cross_templates[(c, c2)]``: dependency ``c``'s Hessian-action
          contribution from *another* dependency ``c2``'s tangent-linear
          direction (``d2Fdm2``).

        Each is kept as its own compiled one-form, using a dedicated
        "direction" placeholder (the same ``_tlm_seed_placeholders`` the TLM
        right-hand side already uses -- safe to share, since the TLM forward
        sweep has always finished computing every tangent-linear value before
        the reverse (adjoint/Hessian) sweep that needs these runs), for the
        same reason as ``_get_or_build_tlm_rhs_templates``: summing every
        dependency's cross-term contribution into one combined form and
        zeroing an inactive dependency's seed has the same ``0 * inf = NaN``
        hazard there does.
        """
        if self._hessian_templates is None:
            assert not isinstance(self._u, list), "Hessian templating is only implemented for scalar problems."
            _, seed_placeholders, state_placeholder = self._get_or_build_tlm_rhs_templates()
            dFdu_template = self._get_or_build_dFdu_template()
            dFdu_adj_template = sum_form(self._get_or_build_dFdu_adj_template())  # type: ignore[arg-type]
            assert isinstance(dFdu_template, ufl.Form)
            assert isinstance(dFdu_adj_template, ufl.Form)
            assert isinstance(state_placeholder, dolfinx.fem.Function)

            self._adjoint_solution_placeholder = dolfinx.fem.Function(self._u.function_space)  # type: ignore[union-attr]
            self._second_adjoint_solution_placeholder = dolfinx.fem.Function(
                self._u.function_space  # type: ignore[union-attr]
            )
            self._hessian_u_seed = dolfinx.fem.Function(self._u.function_space)  # type: ignore[union-attr]

            L_template = ufl.replace(sum_form(self._rhs), self._value_placeholders)  # type: ignore[arg-type]
            F_template = ufl.action(dFdu_template, state_placeholder) - L_template  # type: ignore[arg-type]

            # dF/du does not depend on u for a linear problem, so its second
            # derivative w.r.t. u is always exactly zero: there is no SOA
            # "self" term to template here, only the cross-dependency terms
            # below (contrast NonlinearProblem, where F is nonlinear in u).
            # Verify this invariant once, here, rather than on every call.
            d2Fdu2_check = ufl.algorithms.expand_derivatives(
                ufl.derivative(dFdu_template, state_placeholder, self._hessian_u_seed)
            )
            if not d2Fdu2_check.empty():
                raise RuntimeError(f"This term {d2Fdu2_check} should be zero for linear problems.")

            dFdu_adj_applied = ufl.action(dFdu_adj_template, self._adjoint_solution_placeholder)
            L1 = ufl.action(F_template, self._adjoint_solution_placeholder)
            L2 = ufl.action(F_template, self._second_adjoint_solution_placeholder)

            soa_templates: dict = {}
            fixed_templates: dict = {}
            cross_templates: dict = {}
            for c, c_placeholder in self._value_placeholders.items():
                seed = seed_placeholders[c]

                soa_form = ufl.algorithms.expand_derivatives(ufl.derivative(dFdu_adj_applied, c_placeholder, seed))
                if not (soa_form == 0 or soa_form.empty()):
                    soa_templates[c] = dolfinx.fem.form(
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
            self._hessian_templates = (soa_templates, fixed_templates, cross_templates)
        return self._hessian_templates

    def solve(self, annotate: bool = True) -> typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]]:
        """
        Solve the linear problem and return the solution.
        """
        annotate = pyadjoint.annotate_tape({"annotate": annotate})
        if annotate:
            block = LinearProblemBlock(
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
            tape = pyadjoint.get_working_tape()
            tape.add_block(block)

        # Refresh the forward solver's placeholder coefficients from the
        # user's own, current values before an ordinary solve: a prior
        # recompute (see LinearProblemBlock.prepare_recompute_component) may
        # have left them holding a checkpointed/candidate value instead.
        for original, placeholder in self._value_placeholders.items():
            placeholder.x.array[:] = original.x.array[:]
            placeholder.x.scatter_forward()

        out = dolfinx.fem.petsc.LinearProblem.solve(self)
        if annotate:
            if isinstance(out, Function):
                block.add_output(out.create_block_variable())
            else:
                for ui in out:
                    assert isinstance(ui, Function)
                    block.add_output(ui.create_block_variable())
        return out


class NonlinearProblem(dolfinx.fem.petsc.NonlinearProblem):
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
        ad_block_tag: typing.Optional[str] = None,
        adjoint_petsc_options: typing.Optional[dict] = None,
        tlm_petsc_options: typing.Optional[dict] = None,
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
        ad_block_tag: typing.Optional[str] = None,
        adjoint_petsc_options: typing.Optional[dict] = None,
        tlm_petsc_options: typing.Optional[dict] = None,
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
        ad_block_tag: typing.Optional[str] = None,
        adjoint_petsc_options: typing.Optional[dict] = None,
        tlm_petsc_options: typing.Optional[dict] = None,
    ) -> None:

        self.ad_block_tag = ad_block_tag
        self._adj_options = adjoint_petsc_options
        self._tlm_options = tlm_petsc_options
        self._u = resolve_u(u, F)  # type: ignore[arg-type]
        self._bcs = [] if bcs is None else bcs
        self._lhs = dolfinx.fem.forms.derivative_block(F, self._u)  # type: ignore[arg-type]
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

        # Adjoint and tangent-linear solvers: built lazily (see
        # _get_or_build_adjoint_solver/_get_or_build_tlm_solver) and shared by
        # every NonlinearProblemBlock this Problem records, rather than one
        # per block/solve() call -- same rationale as LinearProblem.
        self._adjoint_solver: typing.Optional[LinearAdjointProblem] = None
        self._tlm_solver: typing.Optional[LinearAdjointProblem] = None
        self._dFdu_template: typing.Optional[ufl.Form] = None
        self._state_placeholder: typing.Optional[dolfinx.fem.Function] = None
        self._tlm_rhs_templates: typing.Optional[dict] = None
        self._tlm_seed_placeholders: dict[dolfinx.fem.Function, dolfinx.fem.Function] = {}
        self._hessian_templates: typing.Optional[tuple] = None
        self._adjoint_solution_placeholder: typing.Optional[dolfinx.fem.Function] = None
        self._second_adjoint_solution_placeholder: typing.Optional[dolfinx.fem.Function] = None
        self._hessian_u_seed: typing.Optional[dolfinx.fem.Function] = None

    def _get_or_build_dFdu_template(self) -> ufl.Form:
        """Build (once) and return dF/du with every non-u coefficient replaced by its
        placeholder, and u itself replaced by a dedicated "state" placeholder
        standing in for "u at this evaluation point".

        Unlike the linear case, dF/du genuinely depends on u's current value
        here (F is nonlinear in u), so it needs a coefficient slot for that --
        but that slot need not be ``self._u`` itself (which the live
        SNES/forward path owns): a dedicated placeholder, refreshed from a
        block's own checkpointed output before each adjoint/TLM/Hessian solve
        (see ``NonlinearProblemBlock.prepare_evaluate_adj``/
        ``prepare_evaluate_tlm``/``prepare_evaluate_hessian``), keeps this
        operator's compiled form fixed for the life of the Problem, exactly
        like the non-u dependencies already routed through
        ``self._value_placeholders``. Shared, verbatim, by the adjoint
        operator (which just adjoints it) and the TLM operator (used as-is),
        mirroring ``LinearProblem._get_or_build_dFdu_template``.
        """
        if self._dFdu_template is None:
            if not isinstance(self._rhs, ufl.Form):
                raise NotImplementedError("Blocked systems not implemented yet.")
            self._state_placeholder = dolfinx.fem.Function(self._u.function_space)  # type: ignore[union-attr]
            replace_map: dict = {**self._value_placeholders, self._u: self._state_placeholder}
            self._dFdu_template = ufl.replace(self._lhs, replace_map)  # type: ignore[arg-type]
        return self._dFdu_template

    def _get_or_build_adjoint_solver(self) -> LinearAdjointProblem:
        """Build (once) and return the adjoint solver shared by every block this Problem records."""
        if self._adjoint_solver is None:
            dFdu_adj = ufl.adjoint(self._get_or_build_dFdu_template())
            self._adjoint_solver = LinearAdjointProblem(
                dFdu_adj,  # type: ignore[arg-type]
                self._rhs,  # type: ignore[arg-type]
                bcs=self._bcs,
                P=self._preconditioner,  # type: ignore[arg-type]
                form_compiler_options=self._form_compiler_options,
                jit_options=self._jit_options,
                petsc_options=self._adj_options,
                petsc_options_prefix=f"{self._petsc_options_prefix}adjoint_",
                kind=self._kind,  # type: ignore[arg-type]
                entity_maps=self._entity_maps,
            )  # type: ignore[misc]
        return self._adjoint_solver

    def _get_or_build_tlm_solver(self) -> LinearAdjointProblem:
        """Build (once) and return the TLM solver shared by every block this Problem records.

        dF/du is used here as-is (unlike the adjoint operator, which adjoints
        it), mirroring ``LinearProblem._get_or_build_tlm_solver``.
        """
        if self._tlm_solver is None:
            self._tlm_solver = LinearAdjointProblem(
                self._get_or_build_dFdu_template(),  # type: ignore[arg-type]
                self._rhs,  # type: ignore[arg-type]
                bcs=self._bcs,
                P=self._preconditioner,  # type: ignore[arg-type]
                form_compiler_options=self._form_compiler_options,
                jit_options=self._jit_options,
                petsc_options=self._tlm_options,
                petsc_options_prefix=f"{self._petsc_options_prefix}tlm_",
                kind=self._kind,  # type: ignore[arg-type]
                entity_maps=self._entity_maps,
            )  # type: ignore[misc]
        return self._tlm_solver

    def _get_or_build_tlm_rhs_templates(
        self,
    ) -> tuple[
        dict[dolfinx.fem.Function, typing.Any],
        dict[dolfinx.fem.Function, dolfinx.fem.Function],
        dolfinx.fem.Function,
    ]:
        """Build (once) and return the per-dependency TLM right-hand-side templates.

        One compiled one-form is built per dependency, using a dedicated
        "direction" placeholder for that dependency
        (``_tlm_seed_placeholders``) rather than a single combined form summed
        over every dependency, for the same reason as
        ``LinearProblem._get_or_build_tlm_rhs_templates``: which dependencies
        actually have a tangent-linear value varies from call to call, and
        evaluating an inactive dependency's term with a zeroed seed instead of
        skipping it outright risks ``0 * inf = NaN`` if that dependency's
        derivative is singular where it is currently valued (e.g. a `1/c`
        term with `c` legitimately zero somewhere in the domain).
        """
        if self._tlm_rhs_templates is None:
            self._get_or_build_dFdu_template()  # ensures self._state_placeholder exists
            assert self._state_placeholder is not None
            assert isinstance(self._rhs, ufl.Form)
            replace_map: dict = {**self._value_placeholders, self._u: self._state_placeholder}
            F_template = ufl.replace(self._rhs, replace_map)
            test_func = F_template.arguments()[0]

            templates: dict[dolfinx.fem.Function, typing.Any] = {}
            for c, c_placeholder in self._value_placeholders.items():
                seed = dolfinx.fem.Function(c.function_space)
                dFdm_c = ufl.algorithms.expand_derivatives(-ufl.derivative(F_template, c_placeholder, seed))
                if dFdm_c == 0 or dFdm_c.empty():
                    dFdm_c = ufl.ZeroBaseForm((test_func,))
                templates[c] = dolfinx.fem.form(
                    dFdm_c,
                    jit_options=self._jit_options,
                    form_compiler_options=self._form_compiler_options,
                    entity_maps=self._entity_maps,
                )
                self._tlm_seed_placeholders[c] = seed
            self._tlm_rhs_templates = templates
        return self._tlm_rhs_templates, self._tlm_seed_placeholders, self._state_placeholder  # type: ignore[return-value]

    def _get_or_build_hessian_templates(
        self,
    ) -> tuple[typing.Optional[dolfinx.fem.Form], dict, dict, dict]:
        """Build (once) and return the per-dependency Hessian templates used by
        ``NonlinearProblemBlock.prepare_evaluate_hessian``'s SOA right-hand side
        and ``evaluate_hessian_component``'s own Hessian-action output.

        Four values are returned:

        - ``soa_self_template``: the SOA right-hand-side's contribution from
          dF/du's own second derivative w.r.t. u (``d2Fdu2``) -- genuinely
          nonzero here since F is nonlinear in u (contrast
          ``LinearProblem._get_or_build_hessian_templates``, where this term
          is always zero and skipped entirely) -- or ``None`` if it happens
          to vanish structurally. Always assembled when not ``None``.
        - ``soa_cross_templates[c]``: the SOA right-hand-side's contribution
          from dependency ``c``'s tangent-linear direction, a 1-form in the
          state's own test function.
        - ``fixed_templates[c]``: the part of dependency ``c``'s own
          Hessian-action output that does not depend on any *other*
          dependency's tangent-linear value (``dL2dm + d2Fdudm``) -- always
          assembled.
        - ``cross_templates[(c, c2)]``: dependency ``c``'s Hessian-action
          contribution from *another* dependency ``c2``'s tangent-linear
          direction (``d2Fdm2``).

        Mirrors ``LinearProblem._get_or_build_hessian_templates`` exactly for
        the cross-dependency terms (same per-dependency-template rationale,
        including the ``0 * inf = NaN`` hazard of a combined, zeroed-seed
        form), with one addition: the SOA "self" term, which for a linear
        problem is always zero and so needs no template at all.
        """
        if self._hessian_templates is None:
            dFdu_template = self._get_or_build_dFdu_template()
            dFdu_adj_template = ufl.adjoint(dFdu_template)
            assert self._state_placeholder is not None
            state_placeholder = self._state_placeholder
            assert isinstance(self._rhs, ufl.Form)

            self._adjoint_solution_placeholder = dolfinx.fem.Function(self._u.function_space)  # type: ignore[union-attr]
            self._second_adjoint_solution_placeholder = dolfinx.fem.Function(
                self._u.function_space  # type: ignore[union-attr]
            )
            self._hessian_u_seed = dolfinx.fem.Function(self._u.function_space)  # type: ignore[union-attr]

            d2Fdu2_template = ufl.algorithms.expand_derivatives(
                ufl.derivative(dFdu_template, state_placeholder, self._hessian_u_seed)
            )
            soa_self_template = None
            if not d2Fdu2_template.empty():
                soa_self_form = ufl.action(ufl.adjoint(d2Fdu2_template), self._adjoint_solution_placeholder)
                soa_self_template = dolfinx.fem.form(
                    soa_self_form,
                    jit_options=self._jit_options,
                    form_compiler_options=self._form_compiler_options,
                    entity_maps=self._entity_maps,
                )

            dFdu_adj_applied = ufl.action(dFdu_adj_template, self._adjoint_solution_placeholder)

            _, seed_placeholders, _ = self._get_or_build_tlm_rhs_templates()
            replace_map: dict = {**self._value_placeholders, self._u: state_placeholder}
            F_template = ufl.replace(self._rhs, replace_map)
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
            self._hessian_templates = (soa_self_template, soa_cross_templates, fixed_templates, cross_templates)
        return self._hessian_templates  # type: ignore[return-value]

    def solve(self, annotate: bool = True) -> typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]]:
        """
        Solve the linear problem and return the solution.
        """
        annotate = pyadjoint.annotate_tape({"annotate": annotate})
        if annotate:
            block = NonlinearProblemBlock(
                J=self._lhs,  # type: ignore[arg-type]
                F=self._rhs,  # type: ignore[arg-type]
                bcs=self._bcs,
                u=self.u,  # type: ignore[arg-type]
                P=self._preconditioner,  # type: ignore[arg-type]
                form_compiler_options=self._form_compiler_options,
                jit_options=self._jit_options,
                entity_maps=self._entity_maps,
                ad_block_tag=self.ad_block_tag,
                problem=self,
            )  # type: ignore[misc]
            tape = pyadjoint.get_working_tape()
            tape.add_block(block)

        # Refresh the SNES-facing placeholder coefficients from the user's
        # own, current values before an ordinary solve: a prior recompute
        # (see NonlinearProblemBlock.prepare_recompute_component) may have
        # left them holding a checkpointed/candidate value instead.
        for original, placeholder in self._value_placeholders.items():
            placeholder.x.array[:] = original.x.array[:]
            placeholder.x.scatter_forward()

        out = dolfinx.fem.petsc.NonlinearProblem.solve(self)
        if annotate:
            if isinstance(out, Function):
                block.add_output(out.create_block_variable())
            else:
                for ui in out:
                    assert isinstance(ui, Function)
                    block.add_output(ui.create_block_variable())
        return out
