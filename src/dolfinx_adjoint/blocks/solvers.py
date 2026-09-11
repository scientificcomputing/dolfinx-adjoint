from __future__ import annotations

import abc
import gc
import typing
import warnings
import weakref

from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import pyadjoint
import ufl

from ..compat import bcs_by_block
from ..types import Function
from ..types.mesh import Mesh, overloaded_mesh
from ..typing_utils import MaybeBlocked, MaybeBlockedMatrix, NestedSequence
from ..ufl_utils import (
    assign_mixed_parts,
    get_sorted_arguments,
    reject_geometry_without_shape_derivative,
    sum_form,
)
from .assembly import _create_vector, _SpecialVector, _vector, assemble_compiled_form

if typing.TYPE_CHECKING:
    from ..solvers import LinearProblem, NonlinearProblem


def collect_coefficients(form: ufl.BaseForm | typing.Sequence | None) -> set[Function]:
    """Return the set of UFL coefficients appearing anywhere in ``form``.

    ``form`` may be a single form or an arbitrarily nested sequence of forms
    (entries may be ``None``, e.g. a zero block in a blocked system). Plain set
    union rather than ``sum_form``: unlike summing, this never requires the
    sub-forms' arguments to be mutually compatible (e.g. carry matching
    ``part()`` tags), which a blocked ``NonlinearProblem``'s forms are not
    required to be before ``assign_mixed_parts`` runs.
    """
    if form is None:
        return set()
    if isinstance(form, ufl.BaseForm):
        return set(form.coefficients())
    coefficients: set = set()
    for f in form:
        coefficients |= collect_coefficients(f)
    return coefficients


def _map_block_variables_to_form(
    form: NestedSequence[ufl.Form | None],
    block_variables: typing.Iterable[pyadjoint.block_variable.BlockVariable],
) -> dict[Function, Function]:
    """Map each ``block_variable``'s output coefficient, where it appears in ``form``, to its
    checkpointed value.

    ``form`` may be a single form or an arbitrarily nested sequence of forms (entries may be
    ``None``, e.g. a zero block in a blocked system); recurses once per nesting level,
    independently of how many ``block_variables`` are given.
    """
    if form is None:
        return {}
    if isinstance(form, ufl.Form):
        coefficients = form.coefficients()
        return {
            block_variable.output: block_variable.saved_output
            for block_variable in block_variables
            if block_variable.output in coefficients
        }
    replace_map: dict = {}
    for f in form:
        replace_map.update(_map_block_variables_to_form(f, block_variables))
    return replace_map


class _ProblemBlockBase(pyadjoint.Block, abc.ABC):
    """Shared tape-block machinery for
    {py:class}`~dolfinx_adjoint.blocks.solvers.LinearProblemBlock`/{py:class}`~dolfinx_adjoint.blocks.solvers.NonlinearProblemBlock`.

    Holds what is unconditionally identical or shares one implementation
    between the two Problem kinds: fetching the owning Problem, recovering
    Dirichlet BC dependencies, detecting a boundary-condition-only adjoint,
    transposing a (possibly blocked) bilinear form, the shared warm-started
    recompute flow, and -- since
    {py:meth}`LinearProblemBlock._compute_residual<dolfinx_adjoint.blocks.solvers.LinearProblemBlock._compute_residual>`
    and
    {py:meth}`NonlinearProblemBlock._compute_residual<dolfinx_adjoint.blocks.solvers.NonlinearProblemBlock._compute_residual>`
    both settle on the same output shape (a single summed {py:class}`ufl.Form`
    plus its dependency replacement map, see each subclass's docstring) --
    every first-order adjoint, TLM and
    Hessian method built *on top of* that residual, both scalar and blocked.
    Each concrete subclass still implements its own ``__init__`` (constructor
    kwargs differ: ``a``/``L`` vs ``J``/``F``) and
    {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase._compute_residual`
    itself, since building the residual is the one place a shared algorithm
    isn't possible -- the two classes start from different user-supplied
    data.
    """

    _problem_ref: weakref.ReferenceType["LinearProblem | NonlinearProblem"]
    _rebuilt_problem: "LinearProblem | NonlinearProblem | None" = None
    _bcs: typing.Sequence[dolfinx.fem.DirichletBC]
    _u: MaybeBlocked[Function]
    _adjoint_solutions: MaybeBlocked[Function]
    _second_adjoint_solutions: MaybeBlocked[Function]
    _tlm_solutions: MaybeBlocked[Function]
    _jit_options: dict | None
    _form_compiler_options: dict | None
    _entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None
    _adj_sol_bdy: MaybeBlocked[_SpecialVector] | None = None
    _adj_sol2_bdy: MaybeBlocked[_SpecialVector] | None = None

    def get_reference_problem(self) -> "LinearProblem | NonlinearProblem":
        """Return this block's owning Problem, which owns the shared solvers.

        Held via a weakref, not a strong reference: a throwaway Problem (solve it, then
        only touch the resulting {py:class}`~pyadjoint.ReducedFunctional`) must not be
        kept alive for as long as this Block stays reachable from the tape. If the
        Problem has already been collected, rebuild an equivalent one from the
        fields this Block stored at construction time (see
        {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase._rebuild_problem`)
        and keep a strong reference to the rebuilt Problem on this Block from then
        on -- this is a rare, costly fallback
        (a fresh Problem means fresh, uncompiled forms and solvers), not the common
        path, so it warns; the strong reference amortizes that cost across every later
        call this Block makes rather than rebuilding again on every call.

        Returns:
            This block's owning Problem (rebuilt, if necessary).
        """
        problem = self._problem_ref()
        if problem is None:
            problem = self._rebuild_problem()
            self._problem_ref = weakref.ref(problem)
            self._rebuilt_problem = problem
        return problem

    @abc.abstractmethod
    def _rebuild_problem(self) -> "LinearProblem | NonlinearProblem":
        """Reconstruct an equivalent Problem from the fields this Block stored at
        construction time, after the original Problem has been garbage collected.

        Each subclass overrides this to call its own Problem constructor
        ({py:class}`~dolfinx_adjoint.LinearProblem`/{py:class}`~dolfinx_adjoint.NonlinearProblem`
        take different arguments -- ``a``/``L`` vs ``F``/``J``).

        Returns:
            A freshly constructed Problem, equivalent to the one this Block was
            originally given.
        """

    @abc.abstractmethod
    def _compute_residual(self) -> tuple[ufl.Form, dict[Function, Function]]:
        """Build this block's residual ``F(u, v) = 0`` at its checkpointed dependency values.

        The one genuinely irreducible difference between the two Problem kinds:
        {py:class}`~dolfinx_adjoint.LinearProblem` derives it from ``a``/``L`` via
        {py:func}`ufl.action`, {py:class}`~dolfinx_adjoint.NonlinearProblem`
        already has ``F`` directly. Both settle on the same output shape -- a
        single summed {py:class}`ufl.Form` plus its dependency replacement map --
        which is what lets every method built on top of this one (below) be
        shared. Each subclass overrides this.

        Returns:
            A ``(F_form, replacement_map)`` pair: a single summed {py:class}`ufl.Form` for the
            residual, and the dependency-to-checkpoint replacement map used to build it.
        """

    def _should_compute_boundary_adjoint(
        self, dependencies: typing.Iterable[pyadjoint.block_variable.BlockVariable]
    ) -> bool:
        """Whether any of ``dependencies`` is a Dirichlet BC -- i.e. whether the boundary-
        control reaction term (see
        {py:meth}`*Problem._get_or_build_adjoint_reaction_template<dolfinx_adjoint.solvers._ProblemBase._get_or_build_adjoint_reaction_template>`)
        is worth computing this call. A bc dependency is never a form coefficient, so it
        cannot flow through the ordinary ``dF/dm`` sensitivity path
        {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.evaluate_adj_component`
        otherwise uses.
        """
        return any(isinstance(dep.output, dolfinx.fem.DirichletBC) for dep in dependencies)

    def _snapshot_rhs(self, rhs_vec: PETSc.Vec) -> MaybeBlocked[np.ndarray]:  # type: ignore[name-defined]
        """Take a local, per-output-block numpy snapshot of ``rhs_vec``'s current values.

        Robust to whether the shared solver's PETSc layout is ``nest`` or monolithic:
        {py:func}`dolfinx.la.petsc.assign` dispatches on argument type, and its
        ``(PETSc.Vec, array(s))`` overload is exactly the inverse of the
        ``(array(s), PETSc.Vec)`` overload this same code already uses to *build*
        ``rhs_vec`` in {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.prepare_evaluate_adj`/
        {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.prepare_evaluate_hessian` --
        reused here in reverse rather than assuming a flat/monolithic layout.
        """
        # mypy infers ui: MaybeBlocked[Function] here despite the isinstance
        # narrowing above (the same narrowing pattern used, unannotated, throughout this
        # module) -- an apparent quirk of this base class's attribute-type inference;
        # narrow explicitly rather than chase it further.
        u_list = self._u if isinstance(self._u, list) else [self._u]
        arrs = [
            np.zeros(
                ui.function_space.dofmap.index_map.size_local * ui.function_space.dofmap.index_map_bs,  # type: ignore[union-attr]
                dtype=dolfinx.default_scalar_type,
            )
            for ui in u_list
        ]
        dolfinx.la.petsc.assign(rhs_vec, arrs)  # type: ignore[arg-type]
        return arrs if isinstance(self._u, list) else arrs[0]

    def _compute_boundary_reaction(
        self,
        rhs_snapshot: MaybeBlocked[np.ndarray],
        reaction_template: MaybeBlocked[dolfinx.fem.Form],
    ) -> MaybeBlocked[_SpecialVector]:
        r"""Compute ``adj_sol_bdy = rhs_snapshot - action(adjoint(dF/du), adjoint_solution)``,
        per output block, given a pre-homogenization snapshot of the adjoint/SOA equation's
        right-hand side (``rhs_snapshot``, from
        {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase._snapshot_rhs`) and the
        compiled ``reaction_template`` (from
        {py:meth}`*Problem._get_or_build_adjoint_reaction_template<dolfinx_adjoint.solvers._ProblemBase._get_or_build_adjoint_reaction_template>`).

        This is ~0 on interior dofs (where the homogeneous adjoint/SOA equation holds) and
        equals the sensitivity of J w.r.t. a Dirichlet bc's value on that bc's own
        constrained dofs -- see
        {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase._mask_reaction_to_bc`
        for how a specific bc's contribution is extracted from this.
        """

        def _one(ui: Function, snap: np.ndarray, template: dolfinx.fem.Form) -> _SpecialVector:
            reaction = _create_vector(template, ui.function_space)
            reaction.array[:] = 0.0
            assemble_compiled_form(template, reaction)
            local_size = ui.function_space.dofmap.index_map.size_local * ui.function_space.dofmap.index_map_bs
            out = _vector(
                ui.function_space.dofmap.index_map,
                ui.function_space.dofmap.index_map_bs,
                ui.function_space,
                dtype=reaction.array.dtype,
            )
            out.array[:local_size] = snap - reaction.array[:local_size]
            out.scatter_forward()
            return out

        if isinstance(self._u, list):
            assert isinstance(rhs_snapshot, typing.Sequence) and isinstance(reaction_template, typing.Sequence)
            return [
                _one(ui, snap, template)
                for ui, snap, template in zip(self._u, rhs_snapshot, reaction_template, strict=True)
            ]
        else:
            assert isinstance(rhs_snapshot, np.ndarray)
            return _one(self._u, rhs_snapshot, reaction_template)  # type: ignore[arg-type]

    def _mask_reaction_to_bc(
        self,
        bc: dolfinx.fem.DirichletBC,
        reaction: MaybeBlocked[_SpecialVector],
    ) -> _SpecialVector:
        """Mask a (possibly per-block) boundary reaction vector onto ``bc``'s own
        constrained dofs, zero elsewhere, returned on ``bc.function_space``.

        Both owned and ghost dofs are copied (`dolfinx.fem.DirichletBC.dof_indices()`
        returns both, unrolled): ``reaction``'s ghost entries are already correctly
        populated (its own construction ends in ``scatter_forward()``), so this stays a
        purely local operation with no further communication needed.
        """
        if isinstance(self._u, list):
            assert isinstance(reaction, typing.Sequence)
            reaction_i = reaction[self._bc_block_index[bc]]
        else:
            reaction_i = reaction
        assert isinstance(reaction_i, _SpecialVector)
        dofs, _ = bc.dof_indices()
        result = _vector(
            bc.function_space.dofmap.index_map,
            bc.function_space.dofmap.index_map_bs,
            bc.function_space,
            dtype=reaction_i.array.dtype,
        )
        result.array[:] = 0.0
        result.array[dofs] = reaction_i.array[dofs]
        return result

    def _register_mesh_dependency(self) -> None:
        """Record the mesh as a dependency, if it has been moved.

        A residual posed on a moved mesh depends on that mesh's geometry through its
        {py:class}`ufl.SpatialCoordinate`, exactly as it depends on any coefficient
        appearing in it. {py:func}`~dolfinx_adjoint.types.mesh.overloaded_mesh` returns
        ``None`` for a mesh that was never passed to {py:func}`~dolfinx_adjoint.move`,
        which is every problem that is not a shape optimization, and then this is a no-op.
        """
        u = self._u[0] if isinstance(self._u, list) else self._u
        assert isinstance(u, dolfinx.fem.Function)
        mesh = overloaded_mesh(u.function_space.mesh.ufl_domain())
        if mesh is not None:
            reject_geometry_without_shape_derivative(self._rhs)
            self.add_dependency(mesh, no_duplicates=True)

    def _assert_shape_dependencies_are_checkpointed(self) -> None:
        """Refuse a shape derivative whose residual would be built at the wrong state.

        A backstop. {py:func}`~dolfinx_adjoint.move` already refuses any checkpoint schedule
        outside its allowlist of ones verified to retain every step, so in normal use this
        never fires. It stays because that allowlist is a claim about *other* packages'
        behaviour: if a schedule on it ever starts releasing checkpoints, this catches it
        instead of letting the gradient go quietly wrong.

        Unlike ``dF/dm`` for a coefficient ``m`` of a *linear* residual, ``dF/dX`` depends on
        the values of every coefficient in the residual, the state included -- the geometry
        enters through the measure and through where the basis functions are evaluated, so
        differentiating it drags the whole integrand along. Those values are read from each
        dependency's ``saved_output``.

        Under a checkpoint schedule that recomputes (a ``Revolve``, say, as opposed to one
        that stores every step), pyadjoint releases a dependency's checkpoint once it
        believes nothing still needs it -- it keeps only those marked as adjoint
        dependencies. ``saved_output`` then quietly returns the *live* value of that
        function, which during a reverse sweep is whatever the last recomputed step left in
        it, not the value this block saw. The resulting shape gradient is wrong by a wide
        margin (16% on a four-step heat equation) and, because the tape is self-consistent,
        a Taylor test still converges at rate 2 and reports nothing.

        A coefficient control is unaffected, which is why this shows up only here: for a
        linear residual ``dF/dm`` does not reference the released values at all.
        """
        released = [
            block_variable
            for block_variable in self.get_dependencies()
            if isinstance(block_variable.output, dolfinx.fem.Function) and block_variable.checkpoint is None
        ]
        if released:
            raise NotImplementedError(
                "A shape derivative cannot be taken under a checkpoint schedule that "
                "recomputes: this block's dependencies "
                f"{sorted(bv.output.name for bv in released)} have had their checkpoints "
                "released, so the residual would be differentiated at the wrong state and "
                "the gradient would be silently wrong. Use a schedule that retains every "
                "step -- checkpoint_schedules.SingleMemoryStorageSchedule or "
                "SingleDiskStorageSchedule -- or no schedule at all."
            )

    def _shape_sensitivity(self, residual: ufl.Form, mesh: Mesh) -> _SpecialVector:
        r"""Return :math:`-\left(\partial F/\partial X\right)^{*}\lambda`, the residual's
        sensitivity to the mesh geometry.

        This dependency cannot take the symbolic route the rest of
        {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.evaluate_adj_component`
        uses: ``ufl.action(ufl.adjoint(dFdX), lambda)`` raises ``Derivatives should be
        applied before executing replace``, because UFL cannot expand a coordinate
        derivative of a coefficient in physical space and so cannot form that Jacobian's
        adjoint symbolically.

        The way around it is to contract *before* differentiating rather than after.
        Substituting the adjoint solution for the residual's test function turns ``F`` into
        a functional, whose coordinate derivative is a one-form on the geometry space and
        assembles straight into a vector. That is the same quantity: ``F`` is linear in its
        test function, so with :math:`F_i := F(u, v_i)`,

        .. math::
            \frac{\partial}{\partial X_j} F(u, \lambda)
            = \sum_i \lambda_i \frac{\partial F_i}{\partial X_j}
            = \left[\left(\frac{\partial F}{\partial X}\right)^{*}\lambda\right]_j ,

        with :math:`\lambda`'s dof values held fixed, which is exactly how UFL differentiates
        a coefficient with respect to the coordinate. Verified against assembling
        ``dF/dX`` as a rectangular matrix and applying its transpose with PETSc -- the two
        agree to 1e-15 -- and that route is what dolfin-adjoint uses. This one needs no
        matrix, so the block never touches a PETSc object whose destructor is collective,
        and it extends to a blocked problem for free: substitute each test function part by
        its own adjoint solution component.

        Args:
            residual: The block's residual ``F``, at its checkpointed dependency values.
            mesh: The moved mesh to differentiate with respect to.

        Returns:
            The assembled sensitivity, in the mesh's geometry function space.
        """
        adjoint_solutions = (
            self._adjoint_solutions if isinstance(self._adjoint_solutions, list) else [self._adjoint_solutions]
        )
        test_functions = list(get_sorted_arguments(residual.arguments(), 0))
        contracted = ufl.replace(residual, dict(zip(test_functions, adjoint_solutions, strict=True)))

        V_geom = mesh._ad_function_space()
        dFdX = ufl.algorithms.expand_derivatives(
            ufl.derivative(contracted, ufl.SpatialCoordinate(mesh), ufl.TestFunction(V_geom))
        )

        vec = _vector(V_geom.dofmap.index_map, V_geom.dofmap.index_map_bs, function_space=V_geom)
        vec.array[:] = 0.0
        if dFdX.empty():
            return vec

        compiled = dolfinx.fem.form(
            dFdX,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        assemble_compiled_form(compiled, tensor=vec)
        # The sensitivity is -dF/dX, matching the sign evaluate_adj_component uses for every
        # other dependency. Negating the assembled vector rather than the form is deliberate:
        # scaling a form that still holds an unexpanded CoordinateDerivative puts a node
        # outside it, and UFL rejects that with "CoordinateDerivative(s) must be outermost".
        # Ghost entries negate with the owned ones, so no further scatter is needed.
        vec.array[:] *= -1.0
        return vec

    def _reject_higher_order_shape_derivative(self) -> None:
        """Refuse a tangent-linear or Hessian evaluation that would need a shape term.

        The tangent-linear right-hand side is assembled from templates compiled once per
        *coefficient* of the residual, and a mesh is not one, so a mesh direction is
        skipped by the loop that builds it -- yielding not an error but a quietly wrong
        tangent-linear solution, and a quietly wrong Hessian on top of it. Failing here
        keeps that from happening. The first-order adjoint, which takes its own route
        through
        {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase._shape_sensitivity`,
        is unaffected.
        """
        for block_variable in self.get_dependencies():
            if isinstance(block_variable.output, Mesh) and block_variable.tlm_value is not None:
                raise NotImplementedError(
                    "Tangent-linear and Hessian evaluations through a moved mesh are not "
                    "supported yet; only the first-order shape derivative "
                    "(ReducedFunctional.derivative) is. The tangent-linear model with "
                    "respect to a shape control needs dF/dX in its right-hand side, which "
                    "is not assembled."
                )

    def _refresh_dFdu_state(self, problem: "LinearProblem | NonlinearProblem") -> None:
        """Refresh whichever coefficient stands in for "the state" in ``dF/du``, if any.

        A no-op by default: {py:class}`~dolfinx_adjoint.LinearProblem`'s ``dF/du``
        (``a``) is bilinear and never references the state at all, so there is
        nothing to refresh before using the shared adjoint solver.
        {py:class}`~dolfinx_adjoint.blocks.solvers.NonlinearProblemBlock`
        overrides this, since ``dF/du`` genuinely depends on ``u``'s current
        value there -- and, unlike the TLM path (which already loops over
        ``problem.residual_state_placeholder`` generically), neither
        {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.prepare_evaluate_adj`
        nor
        {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.prepare_evaluate_hessian`
        otherwise touches it.

        Args:
            problem: This block's owning Problem (see ``self.get_reference_problem()``).
        """
        pass

    def _create_replace_map(self, form: NestedSequence[ufl.Form | None]) -> dict[Function, Function]:
        """Map each dependency and output to its checkpointed value, wherever it appears in ``form``.

        Args:
            form: This block's residual (or nested block structure thereof).

        Returns:
            A dict from each coefficient (dependency or output) appearing in ``form``
            to the {py:class}`~dolfinx_adjoint.Function` it should be replaced by (its
            checkpointed value).
        """
        replace_map: dict = {}
        replace_map.update(_map_block_variables_to_form(form, self.get_dependencies()))
        replace_map.update(_map_block_variables_to_form(form, self.get_outputs()))
        return replace_map

    def prepare_evaluate_tlm(self, inputs, tlm_inputs, relevant_outputs) -> MaybeBlocked[Function]:
        """Assemble and solve the tangent-linear (TLM) system for this block.

        The TLM solver -- and the compiled LHS it solves with, shared verbatim with
        dF/du (see
        {py:meth}`*Problem._get_or_build_dFdu_template<dolfinx_adjoint.solvers._ProblemBase._get_or_build_dFdu_template>`)
        -- are shared across every block this Problem records; likewise the
        per-dependency TLM right-hand-side templates (see
        {py:meth}`*Problem._get_or_build_tlm_rhs_templates<dolfinx_adjoint.solvers._ProblemBase._get_or_build_tlm_rhs_templates>`)
        are each compiled once. Refresh this block's own checkpointed values into the
        placeholders and re-establish this block's own bcs on every call, since
        another block may have used the same solver in between -- but never rebuild
        or recompile any of these forms. Only dependencies that actually have a
        tangent-linear value this call are assembled into the right-hand side (see
        {py:meth}`*Problem._get_or_build_tlm_rhs_templates<dolfinx_adjoint.solvers._ProblemBase._get_or_build_tlm_rhs_templates>`
        for why an inactive dependency's term must be skipped entirely rather than
        evaluated with a zeroed direction).

        Args:
            inputs: The dependencies' current values. Unused: each dependency is
                visited via ``self.get_dependencies()`` instead.
            tlm_inputs: The dependencies' tangent-linear values, parallel to
                ``inputs``. Unused for the same reason.
            relevant_outputs: The output block variables relevant to
                {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.evaluate_tlm_component`.
                Unused: this block always solves for every output at once (see
                ``self.get_outputs()``).

        Returns:
            ``self._tlm_solutions``: this block's own tangent-linear solution(s),
            already solved for. Passed through unchanged as ``prepared`` to every
            subsequent {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.evaluate_tlm_component` call.
        """
        self._reject_higher_order_shape_derivative()
        problem = self.get_reference_problem()
        tlm_solver = problem._get_or_build_tlm_solver()
        tlm_solver.bcs = self._bcs
        # A perturbed bc enters as an inhomogeneous condition on the TLM solve (see
        # HomogeneousBCLinearProblem.tlm_bcs/solve()), not as an ordinary RHS term -- only
        # tracked bcs with an actual tangent-linear value this call contribute one; an
        # untracked bc, or a tracked one with no perturbation this call, correctly keeps
        # u_dot=0 there via the solver's own unconditional alpha=0.0 pass.
        tlm_solver.tlm_bcs = [
            perturbed_bc
            for bc in self._bcs
            if hasattr(bc, "block_variable") and (perturbed_bc := bc.block_variable.tlm_value) is not None
        ]
        templates, seed_placeholders, state_placeholder = problem._get_or_build_tlm_rhs_templates()

        for block_variable in self.get_dependencies():
            placeholder = problem.value_placeholders.get(block_variable.output)
            if placeholder is not None:
                placeholder.x.array[:] = block_variable.saved_output.x.array[:]
                placeholder.x.scatter_forward()
        state_list = state_placeholder if isinstance(state_placeholder, list) else [state_placeholder]
        for placeholder, out_bv in zip(state_list, self.get_outputs(), strict=True):
            placeholder.x.array[:] = out_bv.saved_output.x.array[:]
            placeholder.x.scatter_forward()

        # 3. Assemble RHS Vector utilizing the shared solver's cached vector,
        # accumulating only the dependencies that actually have a
        # tangent-linear value this call -- see
        # *Problem._get_or_build_tlm_rhs_templates for why an inactive
        # dependency's term must be skipped entirely rather than evaluated
        # with a zeroed direction.
        b_petsc = tlm_solver.b
        with b_petsc.localForm() as b_loc:
            b_loc.set(0.0)
        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            if tlm_value is None:
                continue
            template = templates.get(block_variable.output)
            if template is None:
                continue
            seed = seed_placeholders[block_variable.output]
            seed.x.array[:] = tlm_value.x.array[:]
            seed.x.scatter_forward()
            dolfinx.fem.petsc.assemble_vector(b_petsc, template)
        dolfinx.la.petsc._ghost_update(b_petsc, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]

        # 4. Solve the full monolithic TLM system. The solver's own solution
        # storage is shared across every block, so copy the result out into
        # this block's own buffer immediately (mirroring how the adjoint path
        # copies out of the shared adjoint solver in prepare_evaluate_adj)
        # rather than relying on solver-owned storage identity.
        tlm_solver.solve()
        if isinstance(self._tlm_solutions, list):
            for tlm_sol, sol in zip(self._tlm_solutions, tlm_solver.u):
                tlm_sol.x.array[:] = sol.x.array[:]
        else:
            assert isinstance(self._tlm_solutions, dolfinx.fem.Function)
            self._tlm_solutions.x.array[:] = tlm_solver.u.x.array[:]

        return self._tlm_solutions

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx: int, prepared=None) -> Function:
        """Return this output's share of the tangent-linear solution already computed
        by {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.prepare_evaluate_tlm`.

        Args:
            inputs: The dependencies' current values. Unused.
            tlm_inputs: The dependencies' tangent-linear values. Unused.
            block_variable: The output block variable corresponding to ``idx``.
                Unused: the result is read from ``prepared`` instead.
            idx: Index of the output to return, into ``self._tlm_solutions`` if it is
                a list (a blocked problem); ignored for a scalar problem.
            prepared: ``self._tlm_solutions``, as returned by
                {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.prepare_evaluate_tlm`.

        Returns:
            This output's tangent-linear solution.
        """
        if isinstance(self._tlm_solutions, list):
            return self._tlm_solutions[idx]
        else:
            assert isinstance(self._tlm_solutions, Function)
            return self._tlm_solutions

    def prepare_evaluate_adj(
        self,
        inputs: typing.Sequence[Function],
        adj_inputs: typing.Sequence[dolfinx.la.Vector],
        relevant_dependencies: typing.List[tuple[int, pyadjoint.block_variable.BlockVariable]],
    ) -> tuple[ufl.Form, dict[Function, Function]]:
        """Assemble and solve the first-order adjoint equation for this block.

        The adjoint solver -- and the compiled LHS it solves with -- are shared
        across every block this Problem records. Refresh this block's own
        checkpointed values into the placeholders, refresh whatever "state" dF/du is
        evaluated at if it depends on one (a no-op for
        {py:class}`~dolfinx_adjoint.LinearProblem`, see
        {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase._refresh_dFdu_state`),
        and re-establish this block's own bcs on every call, since another block may
        have used the same solver in between -- but never rebuild or recompile the
        form itself.

        Args:
            inputs: The dependencies' current values. Unused: each dependency's
                checkpointed value is read directly from ``self.get_dependencies()``.
            adj_inputs: The adjoint seed(s) received from this block's output(s) --
                one entry per output for a blocked problem -- assembled into the
                adjoint equation's right-hand side.
            relevant_dependencies: The dependency block variables relevant to
                {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.evaluate_adj_component`.
                Unused: every dependency is visited via ``self.get_dependencies()``
                instead.

        Returns:
            A ``(F_form, replacement_map)`` pair -- this block's residual (from
            ``self._compute_residual()``) and its dependency replacement map --
            passed through unchanged as ``prepared`` to every subsequent
            {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.evaluate_adj_component` call.
        """
        problem = self.get_reference_problem()
        adjoint_solver = problem._get_or_build_adjoint_solver()
        adjoint_solver.bcs = self._bcs
        for block_variable in self.get_dependencies():
            placeholder = problem.value_placeholders.get(block_variable.output)
            if placeholder is not None:
                placeholder.x.array[:] = block_variable.saved_output.x.array[:]
                placeholder.x.scatter_forward()
        self._refresh_dFdu_state(problem)

        # Extract dJ/du[v] from the adjoint inputs.
        if len(adj_inputs) == 1:
            adj_rhs = adj_inputs[0]
            dJdu = adjoint_solver.b
            with dJdu.localForm() as dJdu_loc, adj_rhs.petsc_vec.localForm() as adj_rhs_loc:
                dJdu_loc.array[:] = adj_rhs_loc.array[:]
        else:
            assert len(adj_inputs) == len(self.get_outputs()), (
                f"Expected {len(self.get_outputs())} adjoint inputs, got {len(adj_inputs)})"
            )
            dJdu = adjoint_solver.b
            with dJdu.localForm() as dJdu_loc:
                dJdu_loc.set(0.0)

            arrs = []
            for adj_rhs, output in zip(adj_inputs, self.get_outputs()):
                local_size = output.output.index_map.size_local * output.output.function_space.dofmap.index_map_bs
                if adj_rhs is None:
                    arrs.append(np.zeros(local_size, dtype=dolfinx.default_scalar_type))
                else:
                    arrs.append(adj_rhs.array[:local_size])
            dolfinx.la.petsc.assign(arrs, dJdu)
            dJdu.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore[arg-type]

        # A Dirichlet bc is never a form coefficient, so its sensitivity can't flow
        # through evaluate_adj_component's ordinary dF/dm path below -- snapshot the
        # adjoint right-hand side now, before HomogeneousBCLinearProblem.solve() zeros
        # every bc dof, so that dJdu - action(adjoint(dF/du), adj_sol) (computed after
        # the solve, once adj_sol is known) is available as this bc's reaction. See
        # *Problem._get_or_build_adjoint_reaction_template for the full recipe.
        compute_bdy = self._should_compute_boundary_adjoint(self.get_dependencies())
        dJdu_snapshot = self._snapshot_rhs(dJdu) if compute_bdy else None

        adjoint_solver.solve()
        if isinstance(self._adjoint_solutions, list):
            for adj_sol, sol in zip(self._adjoint_solutions, adjoint_solver.u):
                adj_sol.x.array[:] = sol.x.array[:]
        else:
            assert isinstance(self._adjoint_solutions, dolfinx.fem.Function)
            self._adjoint_solutions.x.array[:] = adjoint_solver.u.x.array[:]

        if compute_bdy:
            problem._ensure_hessian_placeholders()
            adj_sol_placeholder = problem.adjoint_solution_placeholder
            placeholder_list = adj_sol_placeholder if isinstance(adj_sol_placeholder, list) else [adj_sol_placeholder]
            adj_sol_list = (
                self._adjoint_solutions if isinstance(self._adjoint_solutions, list) else [self._adjoint_solutions]
            )
            for placeholder, sol in zip(placeholder_list, adj_sol_list, strict=True):
                placeholder.x.array[:] = sol.x.array[:]
                placeholder.x.scatter_forward()
            reaction_template = problem._get_or_build_adjoint_reaction_template()
            self._adj_sol_bdy = self._compute_boundary_reaction(dJdu_snapshot, reaction_template)  # type: ignore[arg-type]
        else:
            self._adj_sol_bdy = None

        # F_form/replacement_map are still needed by evaluate_adj_component
        # (to build each dependency's own sensitivity form), but the adjoint
        # LHS itself is already correct on adjoint_solver -- no rebuild, no
        # recompile.
        F_form, replacement_map = self._compute_residual()
        return F_form, replacement_map

    def evaluate_adj_component(
        self,
        inputs: typing.Iterable[Function],
        adj_inputs: typing.Iterable[dolfinx.la.Vector],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        prepared: tuple[ufl.Form, dict[Function, Function]],
    ) -> _SpecialVector:
        r"""Return this dependency's contribution to the adjoint action,
        :math:`\left(\partial F/\partial m\right)^{*}\lambda`.

        Args:
            inputs: The dependencies' current values. Unused.
            adj_inputs: The adjoint seed(s) already consumed by
                {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.prepare_evaluate_adj`. Unused here.
            block_variable: The dependency block variable corresponding to ``idx``.
            idx: Index of the dependency to compute the contribution for, used to
                select the correct mixed-space {py:meth}`ufl.Argument.part` for a blocked problem's
                trial function.
            prepared: The ``(F_form, replacement_map)`` pair returned by
                {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.prepare_evaluate_adj`.

        Returns:
            The assembled sensitivity vector for this dependency, where
            :math:`\lambda` is the first-order adjoint solution
            (``self._adjoint_solutions``).
        """
        residual, replacement_map = prepared
        c = block_variable.output
        c_rep = block_variable.saved_output

        if isinstance(c, Mesh):
            self._assert_shape_dependencies_are_checkpointed()
            return self._shape_sensitivity(sum_form(residual), c)

        if isinstance(c, dolfinx.fem.DirichletBC):
            # A bc is never a form coefficient, so it is never in replacement_map and
            # there is no dF/dm to differentiate -- prepare_evaluate_adj already
            # computed the boundary reaction this bc's contribution is masked from.
            assert self._adj_sol_bdy is not None
            return self._mask_reaction_to_bc(c, self._adj_sol_bdy)

        if isinstance(c, Function):
            # Need some clever construction of the TrialFunction to get a part of the mixed space
            part = idx if isinstance(self._u, list) else None
            dc = ufl.TrialFunction(c_rep.function_space, part=part)
        else:
            raise NotImplementedError(f"Unsupported control {type(c)}")

        # Compute the sensitivity of the residual with respect to the parameter
        sum_res = sum_form(residual)
        assert c in replacement_map.keys()
        assert c_rep == replacement_map[c]
        dFdm = -ufl.derivative(sum_res, c_rep, dc)
        if dFdm.empty():
            # Generate a dummy form to safely extract the correct Vector wrapper type
            dFdm = dolfinx.fem.form(ufl.ZeroBaseForm((dc,)))  # type: ignore[call-overload]

        dFdm_adj = ufl.adjoint(dFdm)
        sensitivity = ufl.action(dFdm_adj, self._adjoint_solutions)

        compiled_sensitivity = dolfinx.fem.form(
            sensitivity,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        vec = _create_vector(compiled_sensitivity, sensitivity.arguments()[0].ufl_function_space())
        vec.array[:] = 0.0
        assemble_compiled_form(compiled_sensitivity, tensor=vec)
        return vec

    def prepare_recompute_component(
        self, inputs: typing.Sequence[typing.Any], relevant_outputs: typing.Sequence[typing.Any]
    ) -> MaybeBlocked[Function]:
        """Recompute the block's own forward solution(s) from its checkpointed dependencies and outputs.

        Each problem has replaced its own forms' coefficients with placeholders, which are populated
        from the block's saved outputs and dependencies here, then the shared forward solver is called
        once to recompute the solution(s).

        The forward solver (``self.get_reference_problem()``) is bound, forever, to
        compiled forms referencing dedicated placeholder coefficients rather
        than the user's own dependency objects (see
        {py:class}`~dolfinx_adjoint.solvers._ProblemBase`'s ``value_placeholders``): writing this call's
        candidate/checkpointed values into the placeholders -- never into
        ``block_variable.output`` itself -- is what the next solve sees,
        without ever mutating an object the user (or a Taylor test perturbing
        a control directly) holds a live reference to.

        The Problem's own unknown(s) are warm-started from this block's own
        saved outputs rather than zeroed: required for SNES/Newton
        convergence, and applied to a KSP-based {py:class}`~dolfinx_adjoint.LinearProblem` the same
        way, so an iterative solver configured with a nonzero initial guess
        benefits from it too -- both classes now solve by refreshing
        placeholder/unknown values and calling an unchanging, already-built
        solver, never by mutating the user's own coefficient or recompiling a
        form.

        Solving happens once, here -- not once per output in
        {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.recompute_component`
        -- which matters for a multi-output (blocked)
        problem. The base ``dolfinx`` ``solve()`` is called directly (via
        ``problem._dolfinx_solve()``), not ``problem.solve()``, which would
        record another block onto the tape.

        Args:
            inputs: The dependencies' current (candidate/checkpointed) values.
                Unused: each dependency is visited via ``self.get_dependencies()``
                instead.
            relevant_outputs: The ``(idx, block_variable)`` pairs identifying which
                of this block's own outputs the Problem's unknown(s) should be
                warm-started from.

        Returns:
            The Problem's own (just-recomputed) unknown(s), ``problem.u`` -- shared
            storage, not yet isolated per block;
            {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.recompute_component`
            below copies out this block's own share.
        """
        problem = self.get_reference_problem()
        for block_variable in self.get_dependencies():
            placeholder = problem.value_placeholders.get(block_variable.output)
            if placeholder is not None:
                placeholder.x.array[:] = block_variable.saved_output.x.array[:]
                placeholder.x.scatter_forward()

        # Re-establish this block's own bcs on the shared forward solver (the
        # Problem itself -- see _problem), since another block may have
        # used it with different bcs in between.
        problem.bcs = self._bcs

        # Warm-start the Problem's own unknown(s) from this block's own saved
        # outputs.
        u_list = problem.u if isinstance(problem.u, list) else [problem.u]
        for idx, out_bv in relevant_outputs:
            u_list[idx].x.array[:] = out_bv.saved_output.x.array[:]
            u_list[idx].x.scatter_forward()

        with pyadjoint.stop_annotating():
            problem._dolfinx_solve()
        return problem.u

    def recompute_component(
        self,
        inputs: typing.Iterable[Function],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        prepared: MaybeBlocked[Function],
    ) -> Function:
        """Return an isolated copy of this block's own share of the already-recomputed state.

        Args:
            inputs: The dependencies' current values. Unused.
            block_variable: The output block variable corresponding to ``idx``.
                Unused: the result is read from ``prepared`` instead.
            idx: Index of the output to return, into ``prepared`` if it is a
                sequence (a blocked problem); ignored for a scalar problem.
            prepared: The Problem's own unknown(s), as returned by
                {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.prepare_recompute_component`
                -- shared storage the Problem itself still owns.

        Returns:
            An isolated copy of this output, so this tape block's own checkpoint
            stays stable even if the shared Problem's unknown is later overwritten
            by another block's recompute. Always goes through
            {py:meth}`~dolfinx_adjoint.types.function.Function._ad_create_checkpoint`
            -- not a bare ``.copy()``, which always returns a plain, non-overloaded
            ``dolfinx.fem.Function`` regardless of the source's concrete type, and not
            a reuse of ``block_variable.checkpoint`` in place, which would silently
            bypass the disk-checkpointing seam that ``_ad_create_checkpoint`` provides
            (see ``maybe_disk_checkpoint`` in
            {py:mod}`~dolfinx_adjoint.checkpointing`) on every recompute after the
            first, and would corrupt any earlier checkpoint that a schedule such as
            {py:class}`~checkpoint_schedules.Revolve` still holds a live reference to.

        Note:
            This allocates a function on every recompute, where reusing the existing
            checkpoint in place would not. Reinstating that reuse behind a "no schedule is
            active" test was considered and rejected on measurement: across an evaluation
            and its adjoint of a 40-step heat equation, all
            ``_ad_create_checkpoint`` calls together account for 2.9 ms of 263 ms, and
            end-to-end the two are indistinguishable. That is not worth a second code path
            whose safety rests on an invariant about pyadjoint's internals -- that
            ``TimeStep.checkpoint`` aliases ``BlockVariable._checkpoint`` for global
            dependencies, and ``restore_from_checkpoint`` hands the same object back.
        """
        if isinstance(prepared, Function):
            assert idx == 0
            source = prepared
        else:
            assert isinstance(prepared, typing.Sequence)
            source = prepared[idx]
        return source._ad_create_checkpoint()

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        """Assemble and solve the second-order-adjoint (SOA) equation.

        Shared by both Problem kinds and both scalar and blocked problems, built
        entirely from the Problem's cached Hessian/TLM templates (see
        {py:class}`~dolfinx_adjoint.solvers.HessianTemplates`) -- no
        block-specific residual construction needed, since the SOA self-term is
        already correctly zero/nonzero per class (see
        {py:func}`~dolfinx_adjoint.solvers._build_soa_self_template`) and, for a
        blocked problem, already split into one compiled form per output row (see
        {py:meth}`~dolfinx_adjoint.solvers._ProblemBase._get_or_build_hessian_templates`).

        Args:
            inputs: The dependencies' current values. Unused.
            hessian_inputs: The Hessian seed(s) received from this block's output(s)
                -- one entry per output for a blocked problem -- added into the
                second-order-adjoint equation's right-hand side.
            adj_inputs: The first-order adjoint seed(s). Unused: the first-order
                adjoint solution needed here was already solved for and cached in
                ``self._adjoint_solutions`` by
                {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.prepare_evaluate_adj`.
            relevant_dependencies: The dependency block variables relevant to
                {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.evaluate_hessian_component`.
                Unused: every dependency is visited via ``self.get_dependencies()``
                instead.

        Returns:
            A ``(residual, adjoint_solution, second_adjoint_solution)`` tuple -- this
            block's residual (from ``self._compute_residual()``) and both adjoint
            solutions -- passed through unchanged as ``prepared`` to every
            subsequent
            {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.evaluate_hessian_component`
            call. ``None`` if there is nothing to do (no dependency has a
            tangent-linear value).
        """
        outputs = self.get_outputs()
        tlm_output = [output.tlm_value for output in outputs if output is not None]
        if len(tlm_output) == 0:
            return

        # The adjoint solver -- and the compiled LHS it solves with, shared
        # verbatim with the first-order adjoint equation -- is shared across
        # every block this Problem records. Refresh this block's own
        # checkpointed values into the placeholders and re-establish this
        # block's own bcs on every call, since another block may have used
        # the same solver in between -- but never rebuild or recompile the
        # LHS itself.
        self._reject_higher_order_shape_derivative()
        problem = self.get_reference_problem()
        adjoint_solver = problem._get_or_build_adjoint_solver()
        adjoint_solver.bcs = self._bcs
        for block_variable in self.get_dependencies():
            placeholder = problem.value_placeholders.get(block_variable.output)
            if placeholder is not None:
                placeholder.x.array[:] = block_variable.saved_output.x.array[:]
                placeholder.x.scatter_forward()

        # Use the cached per-dependency Hessian templates (see
        # *Problem._get_or_build_hessian_templates) instead of rebuilding and
        # recompiling the SOA right-hand side symbolically on every call.
        # Refreshing the state placeholder here subsumes _refresh_dFdu_state's
        # no-op-for-Linear/state-only-for-Nonlinear distinction: the Hessian's
        # own F_template/L1 depend on the state's value for *both* classes
        # (Linear's F_template is action(a, state) - L, genuinely state-valued),
        # unlike dF/du itself, which only Nonlinear's depends on.
        _, seed_placeholders, state_placeholder = problem._get_or_build_tlm_rhs_templates()
        hessian_templates = problem._get_or_build_hessian_templates()

        state_list = state_placeholder if isinstance(state_placeholder, list) else [state_placeholder]
        adj_sol_placeholders = problem.adjoint_solution_placeholder
        adj_sol_placeholder_list = (
            adj_sol_placeholders if isinstance(adj_sol_placeholders, list) else [adj_sol_placeholders]
        )
        adjoint_solutions_list = (
            self._adjoint_solutions if isinstance(self._adjoint_solutions, list) else [self._adjoint_solutions]
        )
        hessian_u_seed = problem.hessian_u_seed
        hessian_u_seed_list = hessian_u_seed if isinstance(hessian_u_seed, list) else [hessian_u_seed]

        for placeholder, out_bv in zip(state_list, outputs, strict=True):
            placeholder.x.array[:] = out_bv.saved_output.x.array[:]
            placeholder.x.scatter_forward()
        for placeholder, adj_sol in zip(adj_sol_placeholder_list, adjoint_solutions_list, strict=True):
            placeholder.x.array[:] = adj_sol.x.array[:]
            placeholder.x.scatter_forward()
        for placeholder, tlm_val in zip(hessian_u_seed_list, tlm_output, strict=True):
            placeholder.x.array[:] = tlm_val.x.array[:]
            placeholder.x.scatter_forward()

        if len(outputs) == 1:
            b = adjoint_solver.b
            with b.localForm() as b_loc:
                b_loc.set(0.0)
            dolfinx.fem.petsc.assemble_vector(b, hessian_templates.soa_self)
            for block_variable in self.get_dependencies():
                tlm_input = block_variable.tlm_value
                if tlm_input is None:
                    continue
                c = block_variable.output
                if isinstance(c, dolfinx.mesh.Mesh):
                    raise NotImplementedError(f"Hessian computation for {type(c)} control not implemented yet.")
                if isinstance(c, dolfinx.fem.DirichletBC):
                    # A bc's SOA-rhs contribution is handled entirely via the boundary
                    # reaction computed after adjoint_solver.solve() below (d2F/dm2 =
                    # d2F/dudm = 0 for a bc control), not via soa_cross here.
                    continue
                template = hessian_templates.soa_cross.get(c)
                if template is None:
                    continue
                seed = seed_placeholders[c]
                seed.x.array[:] = tlm_input.x.array[:]
                seed.x.scatter_forward()
                dolfinx.fem.petsc.assemble_vector(b, template)
            dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
            b.scale(-1)

            with b.localForm() as b_loc:
                b_loc.array[:] += hessian_inputs[0].array[:]
            b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        else:
            assert isinstance(hessian_templates.soa_self, list)
            bs = []
            for i, soa_self_i in enumerate(hessian_templates.soa_self):
                out_i = outputs[i].saved_output
                bi = dolfinx.la.vector(out_i.function_space.dofmap.index_map, out_i.function_space.dofmap.index_map_bs)
                bi.array[:] = 0.0
                dolfinx.fem.assemble_vector(bi.array, soa_self_i)
                bs.append(bi)

            for block_variable in self.get_dependencies():
                tlm_input = block_variable.tlm_value
                if tlm_input is None:
                    continue
                c = block_variable.output
                if isinstance(c, dolfinx.mesh.Mesh):
                    raise NotImplementedError(f"Hessian computation for {type(c)} control not implemented yet.")
                if isinstance(c, dolfinx.fem.DirichletBC):
                    # A bc's SOA-rhs contribution is handled entirely via the boundary
                    # reaction computed after adjoint_solver.solve() below (d2F/dm2 =
                    # d2F/dudm = 0 for a bc control), not via soa_cross here.
                    continue
                templates = hessian_templates.soa_cross.get(c)
                if templates is None:
                    continue
                seed = seed_placeholders[c]
                seed.x.array[:] = tlm_input.x.array[:]
                seed.x.scatter_forward()
                for bi, template_i in zip(bs, templates, strict=True):
                    dolfinx.fem.assemble_vector(bi.array, template_i)

            for i, bi in enumerate(bs):
                bi.scatter_reverse(dolfinx.la.InsertMode.add)
                bi.scatter_forward()
                bi.array[:] *= -1
                hess_input = hessian_inputs[i]
                if hess_input is not None:
                    bi.array[:] += hess_input.array
                bi.scatter_forward()

            b = adjoint_solver.b
            local_arrays = [bi.array[: bi.index_map.size_local * bi.block_size] for bi in bs]
            dolfinx.la.petsc.assign(local_arrays, b)
            b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        # Snapshot the SOA right-hand side now, before HomogeneousBCLinearProblem.solve()
        # zeros every bc dof -- see prepare_evaluate_adj's identical comment; the SOA
        # equation's b, built above, plays the same role dJdu does for the first-order
        # adjoint.
        compute_bdy = self._should_compute_boundary_adjoint(self.get_dependencies())
        b_snapshot = self._snapshot_rhs(b) if compute_bdy else None

        # The SOA (second-order-adjoint) equation shares its LHS verbatim with
        # the first-order adjoint equation (both are adjoint(dF/du)) --
        # already correct and permanent on adjoint_solver, so no rebuild or
        # recompile needed here either.
        adjoint_solver.solve()
        if isinstance(self._second_adjoint_solutions, list):
            for adj_sol, sol in zip(self._second_adjoint_solutions, adjoint_solver.u):
                adj_sol.x.array[:] = sol.x.array[:]
        else:
            self._second_adjoint_solutions.x.array[:] = adjoint_solver.u.x.array[:]

        # fixed/cross (used by evaluate_hessian_component below) depend on
        # problem.second_adjoint_solution_placeholder's current value for
        # both scalar and blocked problems -- refresh it here regardless.
        second_adjoint_solutions_list = (
            self._second_adjoint_solutions
            if isinstance(self._second_adjoint_solutions, list)
            else [self._second_adjoint_solutions]
        )
        second_adj_placeholders = problem.second_adjoint_solution_placeholder
        second_adj_placeholder_list = (
            second_adj_placeholders if isinstance(second_adj_placeholders, list) else [second_adj_placeholders]
        )
        for placeholder, sol in zip(second_adj_placeholder_list, second_adjoint_solutions_list, strict=True):
            placeholder.x.array[:] = sol.x.array[:]
            placeholder.x.scatter_forward()

        if compute_bdy:
            reaction_template = problem._get_or_build_second_order_adjoint_reaction_template()
            self._adj_sol2_bdy = self._compute_boundary_reaction(b_snapshot, reaction_template)  # type: ignore[arg-type]
        else:
            self._adj_sol2_bdy = None

        return self._compute_residual(), self._adjoint_solutions, self._second_adjoint_solutions

    def evaluate_hessian_component(
        self,
        inputs,
        hessian_inputs,
        adj_inputs,
        block_variable,
        idx,
        relevant_dependencies,
        prepared=None,
    ):
        """Return this dependency's contribution to the Hessian action.

        Shared by both Problem kinds and both scalar and blocked problems, built
        entirely from the Problem's cached Hessian templates -- ``fixed``/``cross``
        have the same per-dependency (not per-output-row) shape regardless of
        blocking, since they live on the *control's* own test space, not the
        (possibly blocked) state's (see {py:class}`~dolfinx_adjoint.solvers.HessianTemplates`).

        Args:
            inputs: The dependencies' current values. Unused.
            hessian_inputs: The Hessian seed(s) already consumed by
                {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.prepare_evaluate_hessian`. Unused here.
            adj_inputs: The first-order adjoint seed(s). Unused.
            block_variable: The dependency block variable corresponding to ``idx``.
            idx: Index of the dependency to compute the contribution for. Unused
                directly: ``block_variable.output``/``.saved_output`` identify the
                dependency instead.
            relevant_dependencies: The ``(idx, block_variable)`` pairs for every
                other dependency that may contribute a cross term.
            prepared: The ``(residual, adjoint_solution, second_adjoint_solution)``
                tuple returned by
                {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase.prepare_evaluate_hessian`.
                Unused here: every quantity it carries was already used to
                refresh the cached templates' placeholders there.

        Returns:
            The assembled Hessian-action contribution for this dependency.
        """
        c = block_variable.output
        c_rep = block_variable.saved_output

        # If m = DirichletBC then d^2F(u,m)/dm^2 = 0 and d^2F(u,m)/dudm = 0,
        # so we only have the term dF(u,m)/dm * adj_sol2 -- i.e. the boundary reaction
        # computed against the *second-order* adjoint solution in prepare_evaluate_hessian,
        # masked onto this bc's own dofs exactly like the first-order case.
        if isinstance(c, dolfinx.fem.DirichletBC):
            return self._mask_reaction_to_bc(c, self._adj_sol2_bdy)
        if isinstance(c_rep, dolfinx.fem.Constant):
            raise NotImplementedError("Hessian computation for Constant control not implemented yet.")
            # mesh = extract_mesh_from_form(F_form)
            # W = c._ad_function_space(mesh)
        elif isinstance(c, dolfinx.mesh.Mesh):
            raise NotImplementedError("Hessian computation for Mesh control not implemented yet.")
            # X = dolfin.SpatialCoordinate(c)
            # W = c._ad_function_space()
        else:
            assert isinstance(c, dolfinx.fem.Function)
            W = c.function_space

        # Use the cached per-dependency Hessian templates instead of
        # rebuilding and recompiling the Hessian-action output symbolically
        # on every call. All the placeholders these templates reference (the
        # dependency values, the state, and both adjoint solutions) were
        # already refreshed by prepare_evaluate_hessian above; only the
        # per-dependency "direction" seeds need setting here, and only for
        # dependencies that actually have a tangent-linear value this call --
        # see *Problem._get_or_build_hessian_templates for why an inactive
        # dependency's cross term must be skipped entirely rather than
        # evaluated with a zeroed direction.
        problem = self.get_reference_problem()
        hessian_templates = problem._get_or_build_hessian_templates()
        _, seed_placeholders, _ = problem._get_or_build_tlm_rhs_templates()

        fixed_template = hessian_templates.fixed[c]
        hessian_output = _create_vector(fixed_template, W)
        hessian_output.array[:] = 0.0
        assemble_compiled_form(fixed_template, hessian_output)

        for _, bv in relevant_dependencies:
            c2 = bv.output
            if isinstance(c2, dolfinx.fem.DirichletBC):
                continue
            tlm_input = bv.tlm_value
            if tlm_input is None:
                continue
            template = hessian_templates.cross.get((c, c2))
            if template is None:
                continue
            seed2 = seed_placeholders[c2]
            seed2.x.array[:] = tlm_input.x.array[:]
            seed2.x.scatter_forward()
            assemble_compiled_form(template, hessian_output)

        hessian_output.array[:] *= -1.0
        return hessian_output


class LinearProblemBlock(_ProblemBlockBase):
    """The pyadjoint tape block recorded by a {py:class}`~dolfinx_adjoint.LinearProblem` solve.

    See `_ProblemBlockBase`'s own docstring for the shared adjoint/TLM/Hessian machinery;
    this subclass supplies the pieces genuinely specific to a linear ``a``/``L`` residual.
    """

    _adjoint_solutions: MaybeBlocked[Function]
    _tlm_solutions: MaybeBlocked[Function]
    _second_adjoint_solutions: MaybeBlocked[Function]

    # 2. Overload for the SCALAR case
    @typing.overload
    def __init__(
        self,
        a: ufl.Form,
        L: ufl.Form,
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: Function | None = None,
        P: ufl.Form | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        kind: typing.Any = None,
        petsc_options: dict | None = None,
        petsc_options_prefix: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
        ad_block_tag: str | None = None,
        problem: "LinearProblem" = ...,
    ) -> None: ...

    @typing.overload
    def __init__(
        self,
        a: typing.Sequence[typing.Sequence[ufl.Form]],
        L: typing.Sequence[ufl.Form],
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: typing.Sequence[Function] | None = None,
        P: typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        kind: typing.Any = None,
        petsc_options: dict | None = None,
        petsc_options_prefix: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
        ad_block_tag: str | None = None,
        problem: "LinearProblem" = ...,
    ) -> None: ...

    def __init__(
        self,
        a: MaybeBlockedMatrix[ufl.Form],
        L: MaybeBlocked[ufl.Form],
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: MaybeBlocked[Function] | None = None,
        P: MaybeBlockedMatrix[ufl.Form] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        kind: typing.Any = None,
        petsc_options: dict | None = None,
        petsc_options_prefix: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
        ad_block_tag: str | None = None,
        problem: "LinearProblem" = None,  # type: ignore[assignment]
    ) -> None:

        assert problem is not None, "problem must be provided."
        # Held via a weakref, not a strong reference: a throwaway LinearProblem
        # (solve it, then only touch the ReducedFunctional) must not be kept
        # alive for as long as this Block is reachable from the tape. The
        # remaining constructor arguments cached below (kind/petsc_options/...)
        # are not needed for this Block's own forward/adjoint/TLM/Hessian math
        # -- that all lives on `problem` -- they exist solely to let
        # _rebuild_problem() reconstruct an equivalent LinearProblem if the
        # original one is ever collected while this Block still needs it.
        self._problem_ref = weakref.ref(problem)
        self._kind = kind
        self._petsc_options = petsc_options
        self._petsc_options_prefix = petsc_options_prefix
        self._adjoint_petsc_options = adjoint_petsc_options
        self._tlm_petsc_options = tlm_petsc_options
        super().__init__(ad_block_tag=ad_block_tag)

        # Collect all arguments in variational forms and replace them with similar
        # once that is based on a mixed functionspace.
        if not isinstance(a, ufl.Form):
            a, L = assign_mixed_parts(a, L)
            if P is not None:
                P, _ = assign_mixed_parts(P, L)

        self._lhs = a
        self._rhs = L
        self._preconditioner = P

        # Create overloaded functions
        self._u: MaybeBlocked[Function]
        if isinstance(u, dolfinx.fem.Function):
            self._u = pyadjoint.create_overloaded_object(u)
        elif u is None:
            try:
                # Extract function space for unknown from the right hand
                # side of the equation.
                self._u = Function(L.arguments()[0].ufl_function_space())  # type: ignore
            except AttributeError:
                self._u = [Function(Li.arguments()[0].ufl_function_space()) for Li in L]  # type: ignore[union-attr]
        else:
            self._u = [pyadjoint.create_overloaded_object(ui) for ui in u]

        # NOTE: Add mesh and constants as dependencies later on

        # To ensure that the solver can be recycled in time dependent loops, the unknown is also added as a dependency
        # if present in the form.
        coeffs: set[Function]
        if isinstance(self._u, dolfinx.fem.Function):
            assert isinstance(self._lhs, ufl.Form)
            assert isinstance(self._rhs, ufl.Form)
            if self._u in self._lhs.coefficients() or self._u in self._rhs.coefficients():
                raise RuntimeError("The unknown function u should not be present in the variational forms a or L.")
            coeffs = set(self._lhs.coefficients() + self._rhs.coefficients())
        elif isinstance(self._u, typing.Iterable):
            coeffs = set()
            for Ai in self._lhs:  # type: ignore
                for Aij in Ai:
                    if Aij is not None:
                        assert isinstance(Aij, ufl.Form)
                        A_ij_coeffs = set(Aij.coefficients())
                        coeffs |= A_ij_coeffs
                        for c in A_ij_coeffs:
                            if c in self._u:
                                raise RuntimeError(
                                    "The unknown function u should not be present in the variational forms a or L."
                                )
            for part in self._rhs:  # type: ignore
                bi_coeffs = set(part.coefficients())
                coeffs |= bi_coeffs
                for c in bi_coeffs:
                    if c in self._u:
                        raise RuntimeError(
                            "The unknown function u should not be present in the variational forms a or L."
                        )
        else:
            raise RuntimeError(f"Unknown type for unknown function u={type(self._u)}.")

        sorted_coefficients = sorted(coeffs, key=lambda c: c.ufl_id())
        for c in sorted_coefficients:
            self.add_dependency(c, no_duplicates=True)
        self._register_mesh_dependency()

        # Cache form parameters for later
        # NOTE: Should probably be in a struct
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._bcs = bcs if bcs is not None else []

        # Add dependencies from the boundary conditions
        for bc in self._bcs:
            if hasattr(bc, "block_variable"):
                self.add_dependency(bc, no_duplicates=True)

        # Which output block each bc constrains, for evaluate_adj_component/
        # evaluate_hessian_component's DirichletBC branch to index into the
        # (possibly per-block) boundary reaction with -- computed once here, since a
        # bc's block assignment is static for this Block's lifetime. Reuses dolfinx's
        # own bcs_by_block rather than a hand-rolled containment check, matching the
        # grouping HomogeneousBCLinearProblem.solve() already relies on.
        self._bc_block_index: dict[dolfinx.fem.DirichletBC, int] = {}
        if isinstance(self._u, list) and self._bcs:
            spaces = [ui.function_space for ui in self._u]
            grouped = bcs_by_block(spaces, self._bcs)
            for block_idx, bcs_in_block in enumerate(grouped):
                for bc in bcs_in_block:
                    self._bc_block_index[bc] = block_idx
        self._adj_sol_bdy = None
        self._adj_sol2_bdy = None

        # No forward/adjoint/TLM solver is built here: this block shares the
        # ones owned by self.get_reference_problem() (see LinearProblem in ../solvers.py),
        # built once and reused across every block that Problem records
        # instead of once per solve() call.

        # Private, isolated scratch storage for this block's own adjoint/TLM
        # solutions -- never shared with problem.u or with any other block.
        # Built via _ad_create_checkpoint(), not a bare .copy(): the latter
        # always returns a plain, non-overloaded dolfinx.fem.Function
        # regardless of the source's concrete type (see the same note on
        # Function._ad_create_checkpoint in types/function.py).
        if isinstance(self._u, dolfinx.fem.Function):
            self._adjoint_solutions = self._u._ad_create_checkpoint()
            self._second_adjoint_solutions = self._u._ad_create_checkpoint()
            self._tlm_solutions = self._u._ad_create_checkpoint()
        else:
            assert isinstance(self._u, typing.Iterable)
            self._adjoint_solutions = [u._ad_create_checkpoint() for u in self._u]
            self._second_adjoint_solutions = [u._ad_create_checkpoint() for u in self._u]
            self._tlm_solutions = [u._ad_create_checkpoint() for u in self._u]

    def _compute_residual(self) -> tuple[ufl.Form, dict[Function, Function]]:
        """Convert the formulation :math:`a(u, v)=L(v)` into a residual :math:`F(u_b, v) = 0` where
        :math:`u_b` is the solution of the forward problem at the current time and all coefficients are updated.

        Returns:
            A ``(F_form, replacement_map)`` pair, per
            {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase._compute_residual`.
        """
        # NOTE: Should probably be possible to compile this form once.
        replacement_functions = self.get_outputs()
        r_funcs = (
            [r.saved_output for r in replacement_functions]
            if len(replacement_functions) > 1
            else replacement_functions[0].saved_output
        )
        summed_form = sum_form(self._lhs)
        F_form = ufl.action(summed_form, r_funcs) - sum_form(self._rhs)
        replacement_map = self._create_replace_map(F_form)
        F_form = ufl.replace(F_form, replacement_map)
        return F_form, replacement_map

    def _rebuild_problem(self) -> "LinearProblem":
        """See {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase._rebuild_problem`.

        Returns:
            A freshly constructed {py:class}`~dolfinx_adjoint.LinearProblem`, built from the ``a``/``L``/bcs/
            options this block itself stored at construction time.
        """
        from ..solvers import _PROBLEM_PREFIX_COUNTER, LinearProblem

        warnings.warn(
            "This block's LinearProblem was garbage collected before being "
            "recomputed/differentiated; rebuilding an equivalent one. Keep the "
            "LinearProblem object alive for as long as its blocks may need "
            "replay to avoid this cost.",
            stacklevel=4,
        )
        prefix = f"RebuiltLinearProblem_{next(_PROBLEM_PREFIX_COUNTER)}_"

        # Collect before building the replacement, which allocates fresh Functions.
        # Every ufl.replace() leaves a cyclic Replacer behind -- UFL builds its handler
        # table out of bound methods that point back at it -- and that cycle pins the
        # Functions in the Replacer's own mapping, so a discarded Problem's placeholders
        # stay alive until the cyclic collector runs. That matters because a live dolfinx
        # Function currently costs two neighbourhood communicators: la::Vector builds its
        # own Scatterer, which calls MPI_Dist_graph_create_adjacent twice. Measured, ~1024
        # retained Functions exhaust MPICH's 2048-context table, so repeated rebuilds
        # abort with "Too many communicators" without this.
        #
        # Deliberately not PETSc.garbage_cleanup(): the PETSc objects here are released by
        # refcounting at the same point on every rank, so PETSc's garbage stash stays empty
        # (measured, both ranks) and there is nothing for it to drain. The communicators at
        # stake belong to dolfinx, not PETSc. Revisit once FEniCS/dolfinx#4484 -- one
        # cached scatter pattern per IndexMap, rather than one Scatterer per Function --
        # reaches a release we depend on.
        gc.collect()

        return LinearProblem(
            self._lhs,  # type: ignore[arg-type]
            self._rhs,  # type: ignore[arg-type]
            bcs=self._bcs,
            u=self._u,  # type: ignore[arg-type]
            P=self._preconditioner,  # type: ignore[arg-type]
            kind=self._kind,
            petsc_options=self._petsc_options,
            petsc_options_prefix=prefix,
            adjoint_petsc_options=self._adjoint_petsc_options,
            tlm_petsc_options=self._tlm_petsc_options,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            entity_maps=self._entity_maps,
        )  # type: ignore[misc]


class NonlinearProblemBlock(_ProblemBlockBase):
    """The pyadjoint tape block recorded by a {py:class}`~dolfinx_adjoint.NonlinearProblem` solve.

    See `_ProblemBlockBase`'s own docstring for the shared adjoint/TLM/Hessian machinery;
    this subclass supplies the pieces genuinely specific to a nonlinear ``F`` residual.
    """

    _adjoint_solutions: MaybeBlocked[Function]
    _second_adjoint_solutions: MaybeBlocked[Function]
    _tlm_solutions: MaybeBlocked[Function]
    _rhs: MaybeBlocked[ufl.Form]

    @typing.overload
    def __init__(
        self,
        F: ufl.Form,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: Function | None = None,
        J: ufl.Form | None = None,
        P: ufl.Form | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        kind: typing.Any = None,
        petsc_options: dict | None = None,
        petsc_options_prefix: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
        ad_block_tag: str | None = None,
        problem: "NonlinearProblem" = ...,
    ) -> None: ...

    @typing.overload
    def __init__(
        self,
        F: typing.Sequence[ufl.Form],
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: typing.Sequence[Function] | None = None,
        J: typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        P: typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        kind: typing.Any = None,
        petsc_options: dict | None = None,
        petsc_options_prefix: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
        ad_block_tag: str | None = None,
        problem: "NonlinearProblem" = ...,
    ) -> None: ...

    def __init__(
        self,
        F: MaybeBlocked[ufl.Form],
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: MaybeBlocked[Function] | None = None,
        J: MaybeBlockedMatrix[ufl.Form] | None = None,
        P: MaybeBlockedMatrix[ufl.Form] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        kind: typing.Any = None,
        petsc_options: dict | None = None,
        petsc_options_prefix: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
        ad_block_tag: str | None = None,
        problem: "NonlinearProblem" = None,  # type: ignore[assignment]
    ) -> None:

        assert problem is not None, "problem must be provided."
        # Held via a weakref -- see LinearProblemBlock.__init__ for the
        # rationale. The remaining constructor arguments cached below are not
        # needed for this Block's own math (that all lives on `problem`);
        # they exist solely for _rebuild_problem() to reconstruct an
        # equivalent NonlinearProblem if the original one is ever collected
        # while this Block still needs it.
        self._problem_ref = weakref.ref(problem)
        self._kind = kind
        self._petsc_options = petsc_options
        self._petsc_options_prefix = petsc_options_prefix
        self._adjoint_petsc_options = adjoint_petsc_options
        self._tlm_petsc_options = tlm_petsc_options
        super().__init__(ad_block_tag=ad_block_tag)
        self._preconditioner = P

        # Create overloaded functions
        assert u is not None, "Control variable(s) must be provided."
        self._u: MaybeBlocked[Function]
        if isinstance(u, dolfinx.fem.Function):
            self._u = pyadjoint.create_overloaded_object(u)
            replace_dict = {u: self._u}
            self._rhs = ufl.replace(F, replace_dict)
            J = ufl.replace(J, replace_dict) if J is not None else None
            self._preconditioner = ufl.replace(P, replace_dict) if P is not None else None
        else:
            self._u = [pyadjoint.create_overloaded_object(ui) for ui in u]
            assert isinstance(F, typing.Iterable)
            replace_dict = {ui: _ui for ui, _ui in zip(u, self._u)}
            self._rhs = [ufl.replace(Fi, replace_dict) for Fi in F]
        # Kept only for _rebuild_problem() -- see NonlinearProblem.__init__'s
        # own self._user_J for why this must never be read for anything else.
        self._user_J = J

        # NOTE: Add constants as dependencies later on
        u_list = self._u if isinstance(self._u, list) else [self._u]
        coeffs = collect_coefficients(J) | collect_coefficients(self._rhs)
        coeffs -= set(u_list)
        sorted_coeffs = sorted(coeffs, key=lambda c: c.ufl_id())
        for c in sorted_coeffs:
            self.add_dependency(c, no_duplicates=True)
        self._register_mesh_dependency()

        # Cache form parameters for later
        # NOTE: Should probably be in a struct
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._bcs = bcs if bcs is not None else []

        # Add dependencies from the boundary conditions
        for bc in self._bcs:
            if hasattr(bc, "block_variable"):
                self.add_dependency(bc, no_duplicates=True)

        # Which output block each bc constrains, for evaluate_adj_component/
        # evaluate_hessian_component's DirichletBC branch to index into the
        # (possibly per-block) boundary reaction with -- computed once here, since a
        # bc's block assignment is static for this Block's lifetime. Reuses dolfinx's
        # own bcs_by_block rather than a hand-rolled containment check, matching the
        # grouping HomogeneousBCLinearProblem.solve() already relies on.
        self._bc_block_index: dict[dolfinx.fem.DirichletBC, int] = {}
        if isinstance(self._u, list) and self._bcs:
            spaces = [ui.function_space for ui in self._u]
            grouped = bcs_by_block(spaces, self._bcs)
            for block_idx, bcs_in_block in enumerate(grouped):
                for bc in bcs_in_block:
                    self._bc_block_index[bc] = block_idx

        # No forward/adjoint solver is built here: this block shares the ones
        # owned by self.get_reference_problem() (see NonlinearProblem in ../solvers.py),
        # built once and reused across every block that Problem records
        # instead of once per solve() call.

        # Private, isolated scratch storage for this block's own adjoint/TLM
        # solutions -- never shared with problem.u or with any other block.
        # Built via _ad_create_checkpoint(), not a bare .copy(): the latter
        # always returns a plain, non-overloaded dolfinx.fem.Function
        # regardless of the source's concrete type (see the same note on
        # Function._ad_create_checkpoint in types/function.py).
        if isinstance(self._u, dolfinx.fem.Function):
            self._adjoint_solutions = self._u._ad_create_checkpoint()
            self._second_adjoint_solutions = self._u._ad_create_checkpoint()
            self._tlm_solutions = self._u._ad_create_checkpoint()
        else:
            assert isinstance(self._u, typing.Iterable)
            self._adjoint_solutions = [u._ad_create_checkpoint() for u in self._u]
            self._second_adjoint_solutions = [u._ad_create_checkpoint() for u in self._u]
            self._tlm_solutions = [u._ad_create_checkpoint() for u in self._u]

    def _compute_residual(self) -> tuple[ufl.Form, dict[Function, Function]]:
        """Build the residual :math:`F(u_b, v) = 0` at the current checkpointed dependency values.

        Unlike {py:class}`~dolfinx_adjoint.blocks.solvers.LinearProblemBlock` (which
        derives ``F`` from ``a``/``L`` via {py:func}`ufl.action`), ``F`` is already
        the residual here -- this only needs to substitute in the checkpointed
        dependency and output values. Settles on the same output shape as
        {py:meth}`~dolfinx_adjoint.blocks.solvers.LinearProblemBlock._compute_residual`
        (a single summed form plus its replacement map) so every method built on
        top of it (adjoint, TLM, Hessian) is shared on the base class.

        Returns:
            A ``(F_form, replacement_map)`` pair, per
            {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase._compute_residual`.
        """
        replacement_functions = self.get_outputs()
        replacement_map = self._create_replace_map(self._rhs)

        u_list = self._u if isinstance(self._u, list) else [self._u]
        for u, block in zip(u_list, replacement_functions):
            replacement_map[u] = block.saved_output

        F_form = ufl.replace(sum_form(self._rhs), replacement_map)
        return F_form, replacement_map

    def _rebuild_problem(self) -> "NonlinearProblem":
        """See {py:meth}`~dolfinx_adjoint.blocks.solvers._ProblemBlockBase._rebuild_problem`.

        Returns:
            A freshly constructed {py:class}`~dolfinx_adjoint.NonlinearProblem`, built from the ``F``/``J``/
            bcs/options this block itself stored at construction time.
        """
        from ..solvers import _PROBLEM_PREFIX_COUNTER, NonlinearProblem

        warnings.warn(
            "This block's NonlinearProblem was garbage collected before being "
            "recomputed/differentiated; rebuilding an equivalent one. Keep the "
            "NonlinearProblem object alive for as long as its blocks may need "
            "replay to avoid this cost.",
            stacklevel=4,
        )
        # See LinearProblemBlock._rebuild_problem for why this is here, and why it is
        # not PETSc.garbage_cleanup().
        gc.collect()

        prefix = f"RebuiltNonlinearProblem_{next(_PROBLEM_PREFIX_COUNTER)}_"
        return NonlinearProblem(
            self._rhs,  # type: ignore[arg-type]
            u=self._u,  # type: ignore[arg-type]
            bcs=self._bcs,
            J=self._user_J,  # type: ignore[arg-type]
            P=self._preconditioner,  # type: ignore[arg-type]
            kind=self._kind,
            petsc_options=self._petsc_options,
            petsc_options_prefix=prefix,
            adjoint_petsc_options=self._adjoint_petsc_options,
            tlm_petsc_options=self._tlm_petsc_options,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            entity_maps=self._entity_maps,
        )  # type: ignore[misc]

    def _refresh_dFdu_state(self, problem: "LinearProblem | NonlinearProblem") -> None:
        """Refresh ``problem``'s "state" placeholder(s) from this block's own saved outputs.

        Unlike {py:class}`~dolfinx_adjoint.LinearProblem` (where ``dF/du`` never references the state), ``dF/du`` here
        genuinely depends on ``u``'s current value, so the shared adjoint solver's compiled LHS
        must be evaluated at *this* block's checkpointed output before it is used -- another
        block sharing the same solver may have left a different value there.

        Args:
            problem: This block's owning Problem (see ``self.get_reference_problem()``).
        """
        state_placeholder = problem.residual_state_placeholder
        state_list = state_placeholder if isinstance(state_placeholder, list) else [state_placeholder]
        for placeholder, out_bv in zip(state_list, self.get_outputs(), strict=True):
            placeholder.x.array[:] = out_bv.saved_output.x.array[:]
            placeholder.x.scatter_forward()
