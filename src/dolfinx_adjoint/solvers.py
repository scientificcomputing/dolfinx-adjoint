from __future__ import annotations

import typing

import dolfinx.fem.petsc
import pyadjoint
import ufl
from dolfinx.fem.function import Function as _Function

from .blocks.solvers import LinearProblemBlock, NonlinearProblemBlock, assign_mixed_parts, sum_form
from .petsc_utils import LinearAdjointProblem
from .types import Function


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

        # Initialize linear solver
        super().__init__(
            a=a,  # type: ignore[arg-type]
            L=L,  # type: ignore[arg-type]
            bcs=bcs,
            u=self._u,  # type: ignore[arg-type]
            P=P,  # type: ignore[arg-type]
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

        # The "live" compiled forms, referencing the user's own coefficient
        # objects directly -- used for every ordinary .solve() call. Snapshot
        # them once so solve() can always restore them, even after a
        # recompute (see _get_or_build_recompute_forms) has pointed self._a/
        # self._L/self._preconditioner at the placeholder-based recompute
        # forms in between.
        self._live_a = self._a
        self._live_L = self._L
        self._live_preconditioner = self._preconditioner

        # Recompute forms: built lazily (see _get_or_build_recompute_forms),
        # compiled once against dedicated placeholder coefficients rather than
        # the user's own dependency objects. Blocks replaying this Problem
        # with a checkpointed/candidate value (e.g. during a Taylor test or an
        # optimisation iteration) write into these placeholders instead of the
        # user's own Functions: mutating the user's own control object in
        # place would corrupt it for any later use of that same object (e.g.
        # a Taylor test that perturbs the original control directly, as
        # `pyadjoint.taylor_test(Jh, m, dm)` does), silently invalidating
        # every subsequent perturbation computed relative to it.
        self._recompute_value_placeholders: typing.Optional[dict] = None
        self._recompute_a = None
        self._recompute_L = None
        self._recompute_preconditioner = None

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
        """Build (once) and return the adjoint solver shared by every block this Problem records."""
        if self._adjoint_solver is None:
            self._adjoint_solver = LinearAdjointProblem(
                LinearProblemBlock._compute_adjoint(sum_form(self._lhs)),  # type: ignore[arg-type]
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

    def _get_or_build_recompute_forms(self) -> tuple[dict, typing.Any, typing.Any, typing.Any]:
        """Build (once) and return the placeholder-based forms used to replay this
        Problem's forward solve with different dependency values.

        Returns a ``(value_placeholders, compiled_a, compiled_L, compiled_preconditioner)``
        tuple. ``value_placeholders`` maps each of this Problem's dependency
        coefficients (as they appear in ``a``/``L``) to a dedicated, plain
        ``dolfinx.fem.Function`` that the compiled forms reference instead.
        Blocks write a checkpointed/candidate value into the placeholder
        before replaying (see ``LinearProblemBlock.prepare_recompute_component``),
        leaving the user's own coefficient objects untouched.
        """
        if self._recompute_value_placeholders is None:
            coefficients: set = set()
            lhs_summed = sum_form(self._lhs)  # type: ignore[arg-type]
            rhs_summed = sum_form(self._rhs)  # type: ignore[arg-type]
            if lhs_summed is not None:
                coefficients.update(lhs_summed.coefficients())
            if rhs_summed is not None:
                coefficients.update(rhs_summed.coefficients())
            placeholders = {c: dolfinx.fem.Function(c.function_space) for c in coefficients}

            def _replace(form):
                if form is None:
                    return None
                if isinstance(form, ufl.Form):
                    return ufl.replace(form, placeholders)
                return [_replace(f) for f in form]

            recompute_lhs = _replace(self._lhs)
            recompute_rhs = _replace(self._rhs)
            recompute_preconditioner = _replace(self._preconditioner) if self._preconditioner is not None else None
            self._recompute_value_placeholders = placeholders
            self._recompute_a = dolfinx.fem.form(
                recompute_lhs,
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )
            self._recompute_L = dolfinx.fem.form(
                recompute_rhs,
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )
            self._recompute_preconditioner = (
                dolfinx.fem.form(
                    recompute_preconditioner,
                    jit_options=self._jit_options,
                    form_compiler_options=self._form_compiler_options,
                    entity_maps=self._entity_maps,
                )
                if recompute_preconditioner is not None
                else None
            )
        return (
            self._recompute_value_placeholders,
            self._recompute_a,
            self._recompute_L,
            self._recompute_preconditioner,
        )

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

        # A prior recompute (see LinearProblemBlock.prepare_recompute_component)
        # may have pointed self._a/self._L/self._preconditioner at the
        # placeholder-based recompute forms; always restore the "live" forms
        # -- which reference the user's own, current coefficient values
        # directly -- before an ordinary solve.
        self._a = self._live_a  # type: ignore[assignment]
        self._L = self._live_L  # type: ignore[assignment]
        self._preconditioner = self._live_preconditioner
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

        # Initialize linear solver
        super().__init__(
            F=F,  # type: ignore[arg-type]
            J=J,  # type: ignore[arg-type]
            P=P,  # type: ignore[arg-type]
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
        """Build (once) and return the adjoint solver shared by every block this Problem records."""
        if self._adjoint_solver is None:
            if not isinstance(self._rhs, ufl.Form):
                raise NotImplementedError("Blocked systems not implemented yet.")
            dFdu_adj = ufl.adjoint(ufl.derivative(self._rhs, self._u))
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

        out = dolfinx.fem.petsc.NonlinearProblem.solve(self)
        if annotate:
            if isinstance(out, Function):
                block.add_output(out.create_block_variable())
            else:
                for ui in out:
                    assert isinstance(ui, Function)
                    block.add_output(ui.create_block_variable())
        return out
