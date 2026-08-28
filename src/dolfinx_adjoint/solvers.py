from __future__ import annotations

import typing

import dolfinx.fem.petsc
import pyadjoint
import ufl
from dolfinx.fem.function import Function as _Function

from .blocks.solvers import LinearProblemBlock, NonlinearProblemBlock, assign_mixed_parts, sum_form
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

        def _replace(form):
            if form is None:
                return None
            if isinstance(form, ufl.Form):
                return ufl.replace(form, self._value_placeholders)
            return [_replace(f) for f in form]

        # Initialize linear solver
        super().__init__(
            a=_replace(a),  # type: ignore[arg-type]
            L=_replace(L),  # type: ignore[arg-type]
            bcs=bcs,
            u=self._u,  # type: ignore[arg-type]
            P=_replace(P),  # type: ignore[arg-type]
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
        # block/solve() call. Blocks only ever hold a weak reference back to this
        # Problem (see LinearProblemBlock._problem), so dropping this Problem
        # releases the forward, adjoint and TLM solvers' PETSc objects
        # deterministically instead of leaving that to pyadjoint's tape/cyclic-GC
        # schedule -- see the "mpi-collective-destruction-hazard" note in the
        # dolfinx-adjoint-knowledge repository for why that matters. Laziness
        # keeps pure forward (non-annotated) use from paying for a symbolic
        # adjoint form it never needs.
        self._adjoint_solver: typing.Optional[LinearAdjointProblem] = None
        self._tlm_solver: typing.Optional[LinearAdjointProblem] = None

    def _get_or_build_adjoint_solver(self) -> LinearAdjointProblem:
        """Build (once) and return the adjoint solver shared by every block this Problem records.

        adjoint(dF/du) does not actually depend on the state u for a linear
        problem: F(u, v) = a(u, v) - L(v) is linear in u, so its derivative
        doesn't reference u's value at all, only whatever *other*
        coefficients a itself depends on. Building it here from the
        placeholder-substituted ``a`` (the same placeholders the forward
        solver's compiled form already uses, see ``_value_placeholders``)
        rather than the original ``a`` means this operator is compiled
        exactly once, ever, for the life of this Problem: every block only
        ever needs to refresh the placeholders' values (see
        ``LinearProblemBlock.prepare_evaluate_adj``/``prepare_evaluate_hessian``),
        never rebuild or recompile this form.
        """
        if self._adjoint_solver is None:
            lhs_for_adjoint = ufl.replace(sum_form(self._lhs), self._value_placeholders)  # type: ignore[arg-type]
            self._adjoint_solver = LinearAdjointProblem(
                LinearProblemBlock._compute_adjoint(lhs_for_adjoint),  # type: ignore[arg-type]
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

    def _get_or_build_tlm_solver(self, dFdu_form: ufl.Form | typing.Sequence) -> LinearAdjointProblem:
        """Build (once) and return the TLM solver shared by every block this Problem records.

        No explicit ``u=`` is passed: like the adjoint solver, this gets its own
        scratch solution Function from the base class, and callers copy the
        result out (see ``LinearProblemBlock.prepare_evaluate_tlm``) rather than
        relying on solver-owned storage identity, since that storage is now
        shared across every block instead of private to one.
        """
        if self._tlm_solver is None:
            self._tlm_solver = LinearAdjointProblem(
                dFdu_form,  # type: ignore[arg-type]
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

        def _replace(form):
            if form is None:
                return None
            if isinstance(form, ufl.Form):
                return ufl.replace(form, self._value_placeholders)
            return [_replace(f) for f in form]

        # Initialize nonlinear solver
        super().__init__(
            F=_replace(F),  # type: ignore[arg-type]
            J=_replace(J),  # type: ignore[arg-type]
            P=_replace(P),  # type: ignore[arg-type]
            bcs=self._bcs,
            u=self._u,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            petsc_options_prefix=petsc_options_prefix,
            petsc_options=petsc_options,
            form_compiler_options=form_compiler_options,
            jit_options=jit_options,
            entity_maps=entity_maps,
        )  # type: ignore[misc]

        # Adjoint solver: built lazily (see _get_or_build_adjoint_solver) and
        # shared by every NonlinearProblemBlock this Problem records, rather
        # than one per block/solve() call -- same rationale as LinearProblem.
        self._adjoint_solver: typing.Optional[LinearAdjointProblem] = None

    def _get_or_build_adjoint_solver(self) -> LinearAdjointProblem:
        """Build (once) and return the adjoint solver shared by every block this Problem records.

        Unlike the linear case, adjoint(dF/du) genuinely depends on u's
        current value here (F is nonlinear in u), so it needs a coefficient
        slot for "u at this evaluation point" -- but that slot need not be
        ``self._u`` itself (which the live SNES/forward path owns): a
        dedicated placeholder, refreshed from a block's own checkpointed
        output before each adjoint/Hessian solve (see
        ``NonlinearProblemBlock.prepare_evaluate_adj``/``prepare_evaluate_hessian``),
        keeps this operator's compiled form fixed for the life of the
        Problem, exactly like the non-u dependencies already routed through
        ``self._value_placeholders``.
        """
        if self._adjoint_solver is None:
            if not isinstance(self._rhs, ufl.Form):
                raise NotImplementedError("Blocked systems not implemented yet.")
            self._adjoint_u_placeholder = dolfinx.fem.Function(self._u.function_space)  # type: ignore[union-attr]
            replace_map: dict = {**self._value_placeholders, self._u: self._adjoint_u_placeholder}
            rhs_for_adjoint = ufl.replace(self._rhs, replace_map)
            dFdu_adj = ufl.adjoint(ufl.derivative(rhs_for_adjoint, self._adjoint_u_placeholder))
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
                tlm_petsc_options=self._tlm_options,
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
