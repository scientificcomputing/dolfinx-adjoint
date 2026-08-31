from __future__ import annotations

import typing

from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import pyadjoint
import ufl
from dolfinx.fem.function import Function as _Function

from ..compat import compute_form_adjoint
from ..types import Function
from .assembly import _create_vector, _SpecialVector, assemble_compiled_form

if typing.TYPE_CHECKING:
    from ..solvers import LinearProblem, NonlinearProblem

type NestedMutableSequence[T] = T | typing.MutableSequence["NestedMutableSequence[T]"]
type NestedSequence[T] = T | typing.Sequence["NestedSequence[T]"]


@typing.overload
def assign_mixed_parts[T: NestedSequence[ufl.Form]](form1: T, /) -> T: ...
@typing.overload
def assign_mixed_parts[T: NestedSequence[ufl.Form], S: NestedSequence[ufl.Form]](
    form1: T, form2: S, /
) -> tuple[T, S]: ...
def assign_mixed_parts(
    *form_structs: NestedSequence[ufl.Form],
) -> NestedSequence[ufl.Form] | tuple[NestedSequence[ufl.Form], ...]:
    """
    Recursively assigns mixed-space `part` indices to {py:class}`ufl.Argument`
    (test and trial functions), within nested iterables of forms.

    When solving monolithic block systems in FEniCSx, the UFL arguments must have the
    method {py:meth}`ufl.Argument.part` return the index corresponding to their block position.
    For a block matrix (list of lists), the TestFunction corresponds to the row index, and the
    TrialFunction corresponds to the column index.

    This utility traverses arbitrary nested structures (e.g., a 2D list for the LHS
    matrix `a` and a 1D list for the RHS vector `L` simultaneously), extracts arguments
    that lack a part index, builds a unified replacement map, and applies it.

    Args:
        *form_structs: One or more UFL forms, or nested iterables (lists/tuples) of
            UFL forms. Passing multiple structures (like `a` and `L`) ensures they
            share the same replacement map, preventing mismatched compilation.

    Returns:
        The modified form structures with identical nesting and sequence types, where
        all unassigned TestFunction and TrialFunction arguments have been mapped.
        Returns a single structure if one was passed, otherwise returns a tuple.

    Note:
        The replacement arguments are drawn from {py:func}`ufl.TestFunctions`
        and {py:func}`ufl.TrialFunctions` of a single
        {py:class}`ufl.MixedFunctionSpace` built from the row/column function spaces
        discovered while walking the structure.
    """
    spaces: dict[int, ufl.functionspace.AbstractFunctionSpace] = {}

    def _discover_spaces(obj: NestedSequence[ufl.Form], indices: tuple[int, ...]) -> None:
        """Recursively discover, for each row/column index, the function space of the
        (as yet unassigned) argument occupying that position.

        `indices` will be `(row,)` for vectors and `(row, col)` for matrices.
        """
        if isinstance(obj, ufl.Form):
            for arg in obj.arguments():
                if arg.part() is None:
                    # The argument number corresponds to the index of the row/column
                    # in the nested structure
                    num = arg.number()
                    if num < len(indices):
                        spaces.setdefault(indices[num], arg.ufl_function_space())
        elif isinstance(obj, typing.Iterable):
            for i, item in enumerate(obj):
                if item is not None:
                    _discover_spaces(item, indices + (i,))
        else:
            raise TypeError(f"Expected ufl.Form or iterable, got {type(obj)}")

    for struct in form_structs:
        _discover_spaces(struct, ())

    # If no replacements are needed, exit early to save computation
    if not spaces:
        return form_structs if len(form_structs) > 1 else form_structs[0]

    num_parts = max(spaces) + 1
    mixed_space = ufl.MixedFunctionSpace(*(spaces[i] for i in range(num_parts)))
    test_functions = ufl.TestFunctions(mixed_space)
    trial_functions = ufl.TrialFunctions(mixed_space)

    replace_map = {}

    def _build_map(obj: NestedSequence[ufl.Form], indices: tuple[int, ...]) -> None:
        if isinstance(obj, ufl.Form):
            for arg in obj.arguments():
                if arg.part() is None and arg not in replace_map:
                    num = arg.number()
                    if num < len(indices):
                        replace_map[arg] = (test_functions if num == 0 else trial_functions)[indices[num]]
        elif isinstance(obj, typing.Iterable):
            for i, item in enumerate(obj):
                if item is not None:
                    _build_map(item, indices + (i,))

    for struct in form_structs:
        _build_map(struct, ())

    def _replace(obj: typing.Any) -> typing.Any:
        """
        Recursively rebuild the structure using the populated replace_map,
        strictly preserving original sequence types (lists vs. tuples).
        """
        if isinstance(obj, ufl.Form):
            return ufl.replace(obj, replace_map)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_replace(item) for item in obj)
        return obj

    # Apply the replacements and unpack if necessary
    replaced = tuple(_replace(struct) for struct in form_structs)
    return replaced if len(replaced) > 1 else replaced[0]


def get_sorted_arguments(arguments: typing.Iterable[ufl.Argument], number: int) -> typing.Iterable[ufl.Argument]:
    """Extract all arguments of a given number, sorted by part."""
    return sorted(filter(lambda x: x.number() == number, arguments), key=lambda a: a.part())


def _collect_coefficients(form: ufl.Form | typing.Sequence | None) -> set:
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
    if isinstance(form, ufl.Form):
        return set(form.coefficients())
    coefficients: set = set()
    for f in form:
        coefficients |= _collect_coefficients(f)
    return coefficients


def _map_block_variables_to_form(
    form: ufl.Form | NestedMutableSequence[ufl.Form] | None,
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


def sum_form(form: NestedSequence[ufl.Form | None]) -> ufl.Form | None:
    """Sum a blocked form into a single form."""
    # Handle top-level None
    if form is None:
        return None

    if isinstance(form, ufl.Form):
        return form

    elif isinstance(form, typing.Iterable):
        # Recursively sum items, filtering out Nones
        valid_forms: list[ufl.Form] = []
        for fi in form:
            summed_fi = sum_form(fi)
            if summed_fi is not None:
                valid_forms.append(summed_fi)

        # Handle empty case safely
        if not valid_forms:
            return None

        # Safely sum without defaulting to integer 0, removing the need for type: ignore
        return sum(valid_forms[1:], start=valid_forms[0])

    else:
        raise TypeError(f"Cannot sum form of type {type(form)}")


class _ProblemBlockBase(pyadjoint.Block):
    """Shared tape-block machinery for ``LinearProblemBlock``/``NonlinearProblemBlock``.

    Holds what is unconditionally identical or shares one implementation
    between the two Problem kinds: fetching the owning Problem, recovering
    Dirichlet BC dependencies, detecting a boundary-condition-only adjoint,
    transposing a (possibly blocked) bilinear form, the shared warm-started
    recompute flow, and -- since ``LinearProblem._compute_residual`` and
    ``NonlinearProblem._compute_residual`` both settle on the same output
    shape (a single summed ``ufl.Form`` plus its dependency replacement map,
    see each subclass's docstring) -- every first-order adjoint, TLM and
    Hessian method built *on top of* that residual, both scalar and blocked.
    Each concrete subclass still implements its own ``__init__`` (constructor
    kwargs differ: ``a``/``L`` vs ``J``/``F``) and ``_compute_residual``
    itself, since building the residual is the one place a shared algorithm
    isn't possible -- the two classes start from different user-supplied
    data.
    """

    _problem_obj: typing.Any
    _bcs: typing.Sequence[dolfinx.fem.DirichletBC]
    _u: _Function | typing.Sequence[_Function]
    _adjoint_solutions: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]
    _second_adjoint_solutions: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]
    _tlm_solutions: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]
    _jit_options: dict | None
    _form_compiler_options: dict | None
    _entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None

    @property
    def _problem(self) -> typing.Any:
        """Return this block's owning Problem, which owns the shared solvers."""
        return self._problem_obj

    def _compute_residual(self) -> tuple[ufl.Form, dict[Function, Function]]:
        """Build this block's residual ``F(u, v) = 0`` at its checkpointed dependency values.

        The one genuinely irreducible difference between the two Problem kinds: ``LinearProblem``
        derives it from ``a``/``L`` via ``ufl.action``, ``NonlinearProblem`` already has ``F``
        directly. Both settle on the same output shape -- a single summed ``ufl.Form`` plus its
        dependency replacement map -- which is what lets every method built on top of this one
        (below) be shared. Not implemented on the base; each subclass overrides it.
        """
        raise NotImplementedError

    def _refresh_dFdu_state(self, problem: typing.Any) -> None:
        """Refresh whichever coefficient stands in for "the state" in ``dF/du``, if any.

        A no-op by default: ``LinearProblem``'s ``dF/du`` (``a``) is bilinear and never
        references the state at all, so there is nothing to refresh before using the shared
        adjoint solver. ``NonlinearProblemBlock`` overrides this, since ``dF/du`` genuinely
        depends on ``u``'s current value there -- and, unlike the TLM path (which already loops
        over ``problem._get_or_build_tlm_rhs_templates()``'s state placeholder(s) generically),
        neither ``prepare_evaluate_adj`` nor ``prepare_evaluate_hessian`` otherwise touches it.
        """

    def _recover_bcs(self):
        bcs = []
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, dolfinx.fem.DirichletBC):
                bcs.append(c_rep)
        return bcs

    def _should_compute_boundary_adjoint(
        self, relevant_dependencies: typing.List[tuple[int, pyadjoint.block_variable.BlockVariable]]
    ) -> bool:
        """Determine if the adjoint should be computed with respect to the boundary conditions."""
        bdy = False
        for _, dep in relevant_dependencies:
            if isinstance(dep.output, dolfinx.fem.DirichletBC):
                bdy = True
                break
        return bdy

    @classmethod
    def _compute_adjoint(cls, form: ufl.Form) -> typing.Sequence[typing.Sequence[ufl.Form]] | ufl.Form:
        """
        Compute adjoint of a bilinear form :math:`a(u, v)`, which could be written as a blocked system.
        """
        return ufl.extract_blocks(compute_form_adjoint(form))

    def _create_replace_map(self, form: ufl.Form | NestedMutableSequence[ufl.Form] | None) -> dict[Function, Function]:
        """Replace dependencies with latest checkpoint."""
        replace_map: dict = {}
        replace_map.update(_map_block_variables_to_form(form, self.get_dependencies()))
        replace_map.update(_map_block_variables_to_form(form, self.get_outputs()))
        return replace_map

    def _compute_residual_derivative(self) -> ufl.Form | list[list[ufl.Form]]:
        """Compute the derivative of the residual with respect to the outputs.

        Shared by both Problem kinds: built purely from ``self._compute_residual()``'s output
        (a single summed form, the same shape for both classes), so no per-class override is
        needed even though the two classes construct that residual very differently.
        """
        F_form, _ = self._compute_residual()

        outputs = self.get_outputs()
        # Use r.saved_output directly; no lookup in replacement_map needed!
        r_funcs = [r.saved_output for r in outputs]

        test_functions = get_sorted_arguments(F_form.arguments(), 0)
        trial_functions = [
            ufl.TrialFunction(output.function_space, part=arg.part())
            for arg, output in zip(test_functions, r_funcs, strict=True)
        ]

        dFdu = ufl.derivative(F_form, r_funcs, trial_functions)

        if isinstance(self._u, list):
            return ufl.extract_blocks(dFdu)
        return dFdu

    def prepare_evaluate_tlm(
        self, inputs, tlm_inputs, relevant_outputs
    ) -> typing.Sequence[Function] | dolfinx.fem.Function:

        # The TLM solver -- and the compiled LHS it solves with, shared
        # verbatim with dF/du (see *Problem._get_or_build_dFdu_template) --
        # are shared across every block this Problem records; likewise the
        # per-dependency TLM right-hand-side templates (see
        # *Problem._get_or_build_tlm_rhs_templates) are each compiled once.
        # Refresh this block's own checkpointed values into the placeholders
        # and re-establish this block's own bcs on every call, since another
        # block may have used the same solver in between -- but never rebuild
        # or recompile any of these forms.
        problem = self._problem
        tlm_solver = problem._get_or_build_tlm_solver()
        tlm_solver.bcs = self._bcs
        templates, seed_placeholders, state_placeholder = problem._get_or_build_tlm_rhs_templates()

        for block_variable in self.get_dependencies():
            placeholder = problem._value_placeholders.get(block_variable.output)
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

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None) -> dolfinx.fem.Function:
        # The system was solved natively in prepare_evaluate_tlm.
        # Return the corresponding requested sub-function.
        if isinstance(self._tlm_solutions, list):
            return self._tlm_solutions[idx]
        else:
            assert isinstance(self._tlm_solutions, dolfinx.fem.Function)
            return self._tlm_solutions

    def prepare_evaluate_adj(
        self,
        inputs: typing.Sequence[Function],
        adj_inputs: typing.Sequence[dolfinx.la.Vector],
        relevant_dependencies: typing.List[tuple[int, pyadjoint.block_variable.BlockVariable]],
    ) -> tuple[ufl.Form, dict[Function, Function]]:
        """Prepare the block for evaluating the adjoint."""

        # The adjoint solver -- and the compiled LHS it solves with -- are
        # shared across every block this Problem records. Refresh this
        # block's own checkpointed values into the placeholders, refresh
        # whatever "state" dF/du is evaluated at if it depends on one (a
        # no-op for LinearProblem, see _refresh_dFdu_state), and re-establish
        # this block's own bcs on every call, since another block may have
        # used the same solver in between -- but never rebuild or recompile
        # the form itself.
        problem = self._problem
        adjoint_solver = problem._get_or_build_adjoint_solver()
        adjoint_solver.bcs = self._bcs
        for block_variable in self.get_dependencies():
            placeholder = problem._value_placeholders.get(block_variable.output)
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

        # F_form/replacement_map are still needed by evaluate_adj_component
        # (to build each dependency's own sensitivity form), but the adjoint
        # LHS itself is already correct on adjoint_solver -- no rebuild, no
        # recompile.
        F_form, replacement_map = self._compute_residual()
        adjoint_solver.solve()
        if isinstance(self._adjoint_solutions, list):
            for adj_sol, sol in zip(self._adjoint_solutions, adjoint_solver.u):
                adj_sol.x.array[:] = sol.x.array[:]
        else:
            assert isinstance(self._adjoint_solutions, dolfinx.fem.Function)
            self._adjoint_solutions.x.array[:] = adjoint_solver.u.x.array[:]
        return F_form, replacement_map

    def evaluate_adj_component(
        self,
        inputs: typing.Iterable[Function],
        adj_inputs: typing.Iterable[dolfinx.la.Vector],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        prepared: tuple[ufl.Form, dict[Function, Function]],
    ) -> _SpecialVector:
        """Evaluate the adjoint component, i.e. :math:`\\frac{\\partial F}{\\partial m}`."""

        residual, replacement_map = prepared
        c = block_variable.output
        c_rep = block_variable.saved_output
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
    ) -> _Function | typing.Sequence[_Function]:
        """Prepare for recomputing the block with different control inputs, and solve.

        The forward solver (``self._problem``) is bound, forever, to
        compiled forms referencing dedicated placeholder coefficients rather
        than the user's own dependency objects (see ``LinearProblem``/
        ``NonlinearProblem._value_placeholders``): writing this call's
        candidate/checkpointed values into the placeholders -- never into
        ``block_variable.output`` itself -- is what the next solve sees,
        without ever mutating an object the user (or a Taylor test perturbing
        a control directly) holds a live reference to.

        The Problem's own unknown(s) are warm-started from this block's own
        saved outputs rather than zeroed: required for SNES/Newton
        convergence, and applied to a KSP-based ``LinearProblem`` the same
        way, so an iterative solver configured with a nonzero initial guess
        benefits from it too -- both classes now solve by refreshing
        placeholder/unknown values and calling an unchanging, already-built
        solver, never by mutating the user's own coefficient or recompiling a
        form.

        Solving happens once, here -- not once per output in
        ``recompute_component`` -- which matters for a multi-output (blocked)
        problem. The base ``dolfinx`` ``solve()`` is called directly (via
        ``problem._dolfinx_solve()``), not ``problem.solve()``, which would
        record another block onto the tape.
        """
        problem = self._problem
        for block_variable in self.get_dependencies():
            placeholder = problem._value_placeholders.get(block_variable.output)
            if placeholder is not None:
                placeholder.x.array[:] = block_variable.saved_output.x.array[:]
                placeholder.x.scatter_forward()

        # Re-establish this block's own bcs on the shared forward solver (the
        # Problem itself -- see _problem()), since another block may have
        # used it with different bcs in between.
        problem.bcs = self._bcs

        # Warm-start the Problem's own unknown(s) from this block's own saved
        # outputs.
        u_list = problem._u if isinstance(problem._u, list) else [problem._u]
        for idx, out_bv in relevant_outputs:
            u_list[idx].x.array[:] = out_bv.saved_output.x.array[:]
            u_list[idx].x.scatter_forward()

        with pyadjoint.stop_annotating():
            problem._dolfinx_solve()
        return problem._u

    def recompute_component(
        self,
        inputs: typing.Iterable[Function],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        prepared: _Function | typing.Sequence[_Function],
    ) -> Function:
        """Return an isolated copy of this block's own share of the already-recomputed state."""
        if isinstance(prepared, dolfinx.fem.Function):
            assert idx == 0
            # Return an explicit copy so each tape block gets an isolated state snapshot
            return prepared.copy()
        else:
            assert isinstance(prepared, typing.Sequence)
            return prepared[idx].copy()

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        """Assemble and solve the second-order-adjoint (SOA) equation.

        Scalar (single-output) problems share one implementation for both
        Problem kinds, built entirely from the Problem's cached Hessian/TLM
        templates (see ``HessianTemplates``) -- no block-specific residual
        construction needed, since the SOA self-term is already correctly
        zero/nonzero per class (see ``_build_soa_self_template``). A blocked
        (multi-output) problem has no templated fast path; its right-hand
        side is assembled by ``_evaluate_hessian_blocked_rhs``, also shared
        by both Problem kinds, built from ``_compute_residual_derivative``/
        ``_compute_adjoint`` -- themselves built only from the per-class
        ``_compute_residual`` hook (the two Problem kinds start from
        different user-supplied data there).
        """
        outputs = self.get_outputs()
        tlm_output = [output.tlm_value for output in outputs if output is not None]
        if (hessian_inputs is None) or (len(tlm_output) == 0):
            return

        # The adjoint solver -- and the compiled LHS it solves with, shared
        # verbatim with the first-order adjoint equation -- is shared across
        # every block this Problem records. Refresh this block's own
        # checkpointed values into the placeholders and re-establish this
        # block's own bcs on every call, since another block may have used
        # the same solver in between -- but never rebuild or recompile the
        # LHS itself.
        problem = self._problem
        adjoint_solver = problem._get_or_build_adjoint_solver()
        adjoint_solver.bcs = self._bcs
        for block_variable in self.get_dependencies():
            placeholder = problem._value_placeholders.get(block_variable.output)
            if placeholder is not None:
                placeholder.x.array[:] = block_variable.saved_output.x.array[:]
                placeholder.x.scatter_forward()
        self._refresh_dFdu_state(problem)

        if len(outputs) == 1:
            # Use the cached per-dependency Hessian templates (see
            # *Problem._get_or_build_hessian_templates) instead of rebuilding
            # and recompiling the SOA right-hand side symbolically on every
            # call.
            _, seed_placeholders, state_placeholder = problem._get_or_build_tlm_rhs_templates()
            hessian_templates = problem._get_or_build_hessian_templates()

            state_placeholder.x.array[:] = outputs[0].saved_output.x.array[:]  # type: ignore[union-attr]
            state_placeholder.x.scatter_forward()  # type: ignore[union-attr]
            problem._adjoint_solution_placeholder.x.array[:] = self._adjoint_solutions.x.array[:]  # type: ignore[union-attr]
            problem._adjoint_solution_placeholder.x.scatter_forward()  # type: ignore[union-attr]
            problem._hessian_u_seed.x.array[:] = tlm_output[0].x.array[:]  # type: ignore[union-attr]
            problem._hessian_u_seed.x.scatter_forward()  # type: ignore[union-attr]

            b = adjoint_solver.b
            with b.localForm() as b_loc:
                b_loc.set(0.0)
            dolfinx.fem.petsc.assemble_vector(b, hessian_templates.soa_self)
            for block_variable in self.get_dependencies():
                tlm_input = block_variable.tlm_value
                if tlm_input is None:
                    continue
                c = block_variable.output
                if isinstance(c, (dolfinx.mesh.Mesh, dolfinx.fem.DirichletBC)):
                    raise NotImplementedError(f"Hessian computation for {type(c)} control not implemented yet.")
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
            self._evaluate_hessian_blocked_rhs(adjoint_solver, hessian_inputs, tlm_output, relevant_dependencies)

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
            if len(outputs) == 1:
                problem._second_adjoint_solution_placeholder.x.array[:] = (  # type: ignore[union-attr]
                    self._second_adjoint_solutions.x.array[:]
                )
                problem._second_adjoint_solution_placeholder.x.scatter_forward()  # type: ignore[union-attr]

        return self._compute_residual(), self._adjoint_solutions, self._second_adjoint_solutions

    def _evaluate_hessian_blocked_rhs(self, adjoint_solver, hessian_inputs, tlm_output, relevant_dependencies):
        """Assemble the SOA right-hand side for a blocked (multi-output) problem into ``adjoint_solver.b``.

        No templated fast path exists for a blocked problem (see
        ``*Problem._get_or_build_hessian_templates``): the right-hand side is built symbolically,
        per call, from ``_compute_residual_derivative``/``_compute_adjoint`` -- the same two
        hooks the scalar path's templates are themselves built from -- so this one implementation
        is shared by both Problem kinds. ``d2Fdu2`` (the SOA self-term) is always included
        unconditionally rather than asserted zero: it comes out zero for a linear residual as a
        result of the maths, exactly like ``_build_soa_self_template``'s scalar-path self-term --
        and, mirroring that helper, is reduced from bilinear (test and trial) to linear (test
        only) by adjointing and acting on the first-order adjoint solution before use, since
        ``ufl.derivative`` w.r.t. a *coefficient* (``unknowns``) leaves ``dF/du``'s own trial
        argument untouched.
        """
        dFdu_form = self._compute_residual_derivative()
        unknowns = [output.saved_output for output in self.get_outputs()]
        summed_form = sum_form(dFdu_form)
        d2Fdu2 = ufl.algorithms.expand_derivatives(ufl.derivative(summed_form, unknowns, tlm_output))

        if d2Fdu2.empty():
            b_form = d2Fdu2
        else:
            b_form = ufl.action(ufl.adjoint(d2Fdu2), self._adjoint_solutions)
        dFdu_adj = self._compute_adjoint(sum_form(dFdu_form))
        for bo in self.get_dependencies():
            c = bo.output
            c_rep = bo.saved_output
            tlm_input = bo.tlm_value
            if tlm_input is None:
                continue
            if isinstance(c, (dolfinx.mesh.Mesh, dolfinx.fem.DirichletBC)):
                raise NotImplementedError(f"Hessian computation for {type(c)} control not implemented yet.")
            else:
                summed_form = sum_form(dFdu_adj)
                dFdu_adj_applied = ufl.action(summed_form, self._adjoint_solutions)
                b_form += ufl.derivative(dFdu_adj_applied, c_rep, tlm_input)

        bs = []
        b_form = ufl.extract_blocks(b_form)
        for i, hess_input in enumerate(hessian_inputs):
            if hess_input is not None:
                bi = dolfinx.la.vector(hess_input.index_map, hess_input.block_size)
            else:
                out_i = self.get_outputs()[i].saved_output
                bi = dolfinx.la.vector(out_i.function_space.dofmap.index_map, out_i.function_space.dofmap.index_map_bs)
            bs.append(bi)
            bi.array[:] = 0.0
            form_i = ufl.algorithms.apply_derivatives.apply_derivatives(b_form[i])
            if not form_i.empty():
                compiled_soa_rhs = dolfinx.fem.form(
                    form_i,
                    jit_options=self._jit_options,
                    form_compiler_options=self._form_compiler_options,
                    entity_maps=self._entity_maps,
                )
                dolfinx.fem.assemble_vector(bi.array, compiled_soa_rhs)
                bi.scatter_reverse(dolfinx.la.InsertMode.add)
                bi.scatter_forward()
                bi.array[:] *= -1

            if hess_input is not None:
                bi.array[:] += hess_input.array

            bi.scatter_forward()
        b = adjoint_solver.b
        local_arrays = [bi.array[: bi.index_map.size_local * bi.block_size] for bi in bs]
        dolfinx.la.petsc.assign(local_arrays, b)
        b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

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

        Scalar (single-output) problems share one implementation for both
        Problem kinds (see ``prepare_evaluate_hessian``); a blocked
        (multi-output) problem falls back to
        ``_evaluate_hessian_component_blocked``.
        """
        c = block_variable.output
        residual_prepared, adj_sol, adj_sol2 = prepared
        outputs = self.get_outputs()
        tlm_output = [output.tlm_value for output in outputs]
        c_rep = block_variable.saved_output

        # If m = DirichletBC then d^2F(u,m)/dm^2 = 0 and d^2F(u,m)/dudm = 0,
        # so we only have the term dF(u,m)/dm * adj_sol2
        if isinstance(c, dolfinx.fem.DirichletBC):
            raise NotImplementedError("Hessian computation for DirichletBC control not implemented yet.")
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

        if len(outputs) == 1:
            # Use the cached per-dependency Hessian templates instead of
            # rebuilding and recompiling the Hessian-action output
            # symbolically on every call. All the placeholders these
            # templates reference (the dependency values, the state, and both
            # adjoint solutions) were already refreshed by
            # prepare_evaluate_hessian above; only the per-dependency
            # "direction" seeds need setting here, and only for dependencies
            # that actually have a tangent-linear value this call -- see
            # *Problem._get_or_build_hessian_templates for why an inactive
            # dependency's cross term must be skipped entirely rather than
            # evaluated with a zeroed direction.
            problem = self._problem
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

        return self._evaluate_hessian_component_blocked(
            residual_prepared, adj_sol, adj_sol2, c, c_rep, W, tlm_output, relevant_dependencies
        )

    def _evaluate_hessian_component_blocked(
        self, residual_prepared, adj_sol, adj_sol2, c, c_rep, W, tlm_output, relevant_dependencies
    ):
        """Return this dependency's Hessian-action contribution for a blocked (multi-output) problem.

        No templated fast path exists for a blocked problem; this builds it symbolically, per
        call, from the residual ``prepare_evaluate_hessian`` already recorded
        (``residual_prepared``) -- shared by both Problem kinds, since both settle on the same
        ``_compute_residual`` output shape.

        We are trying to compute (dF/dm)^T lambda_1
        and (dF_dm)^T lambda_ 2. However, standard approach of UFL
        does not work for MixedFunctionSpaces, as the control space is not
        mixed. Therefore, we instead we compute it as dL/dm = d(lambda_i^T F(m))/dm,
        which is equivalent.
        """
        F_form, replacement_map = residual_prepared
        outputs = self.get_outputs()

        F_summed = sum_form(F_form)
        L1 = ufl.action(F_summed, adj_sol)
        L2 = ufl.action(F_summed, adj_sol2)

        # Compute first derivatives (1-forms tested exactly against the single 'dc' object)
        dc = ufl.TestFunction(W)
        assert c_rep in replacement_map.values()
        dL1dm = ufl.derivative(L1, c_rep, dc)
        dL2dm = ufl.derivative(L2, c_rep, dc)

        sa = [output.saved_output for output in outputs]
        d2Fdudm = ufl.algorithms.expand_derivatives(ufl.derivative(dL1dm, sa, tlm_output))

        d2Fdm2 = ufl.ZeroBaseForm((dc,))  # Initialize the second derivative form
        # We need to add terms from every other dependency
        # i.e. the terms d^2F/dm_1dm_2
        for _, bv in relevant_dependencies:
            c2 = bv.output
            c2_rep = bv.saved_output

            if isinstance(c2, dolfinx.fem.DirichletBC):
                continue
            tlm_input = bv.tlm_value
            if tlm_input is None:
                continue

            if isinstance(c2_rep, dolfinx.mesh.Mesh):
                X = ufl.SpatialCoordinate(c2_rep)
                d2Fdm2 += ufl.algorithms.expand_derivatives(ufl.derivative(dL1dm, X, tlm_input))
            else:
                d2Fdm2 += ufl.algorithms.expand_derivatives(ufl.derivative(dL1dm, c2_rep, tlm_input))

        hessian_form = ufl.algorithms.expand_derivatives(d2Fdm2 + dL2dm + d2Fdudm)

        compiled_hessian = dolfinx.fem.form(
            hessian_form,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )

        test_functions = get_sorted_arguments(hessian_form.arguments(), 0)
        assert len(test_functions) == 1
        hessian_output = _create_vector(compiled_hessian, test_functions[0].ufl_function_space())

        hessian_output.array[:] = 0.0
        assemble_compiled_form(compiled_hessian, hessian_output)
        hessian_output.array[:] *= -1.0
        return hessian_output


class LinearProblemBlock(_ProblemBlockBase):
    """A linear problem that can be used with adjoint methods.

    This class extends the `dolfinx.fem.petsc.LinearProblem` to support adjoint methods.
    """

    _adjoint_solutions: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]
    _tlm_solutions: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]
    _second_adjoint_solutions: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]

    # 2. Overload for the SCALAR case
    @typing.overload
    def __init__(
        self,
        a: ufl.Form,
        L: ufl.Form,
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: _Function | None = None,
        P: ufl.Form | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
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
        u: typing.Sequence[_Function] | None = None,
        P: typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        problem: "LinearProblem" = ...,
    ) -> None: ...

    def __init__(
        self,
        a: ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]],
        L: ufl.Form | typing.Sequence[ufl.Form],
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: _Function | typing.Sequence[_Function] | None = None,
        P: ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        problem: "LinearProblem" = None,  # type: ignore[assignment]
    ) -> None:

        assert problem is not None, "problem must be provided."
        # Strong reference, deliberately: a throwaway LinearProblem (solve it,
        # then only touch the ReducedFunctional) is a common pattern, so the
        # block must keep the Problem -- and its shared solvers -- alive for
        # as long as the block itself is reachable. Not cyclic garbage on its
        # own (verified), so this doesn't reintroduce the MPI
        # collective-destruction hazard from dolfinx-adjoint-knowledge's
        # mpi-collective-destruction-hazard note -- that needs an unmerged
        # checkpoint schedule making the *tape* cyclic.
        self._problem_obj = problem
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
        self._u: _Function | typing.Sequence[_Function]
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
        if isinstance(self._u, dolfinx.fem.Function):
            assert isinstance(self._lhs, ufl.Form)
            assert isinstance(self._rhs, ufl.Form)
            if self._u in self._lhs.coefficients() or self._u in self._rhs.coefficients():
                raise RuntimeError("The unknown function u should not be present in the variational forms a or L.")
            for c in self._lhs.coefficients():
                self.add_dependency(c, no_duplicates=True)
            for c in self._rhs.coefficients():
                self.add_dependency(c, no_duplicates=True)
        elif isinstance(self._u, typing.Iterable):
            for Ai in self._lhs:  # type: ignore
                for Aij in Ai:
                    if Aij is not None:
                        assert isinstance(Aij, ufl.Form)
                        for c in Aij.coefficients():
                            if c in self._u:
                                raise RuntimeError(
                                    "The unknown function u should not be present in the variational forms a or L."
                                )
                            self.add_dependency(c, no_duplicates=True)
            for part in self._rhs:  # type: ignore
                for c in part.coefficients():
                    if c in self._u:
                        raise RuntimeError(
                            "The unknown function u should not be present in the variational forms a or L."
                        )
                    self.add_dependency(c, no_duplicates=True)
        else:
            raise RuntimeError(f"Unknown type for unknown function u={type(self._u)}.")
        # Cache form parameters for later
        # NOTE: Should probably be in a struct
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._bcs = bcs if bcs is not None else []

        # Add dependencies from the boundary conditions
        if self._bcs is not None:
            for bc in self._bcs:
                if hasattr(bc, "block_variable"):
                    self.add_dependency(bc, no_duplicates=True)

        # No forward/adjoint/TLM solver is built here: this block shares the
        # ones owned by self._problem (see LinearProblem in ../solvers.py),
        # built once and reused across every block that Problem records
        # instead of once per solve() call.

        if isinstance(self._u, dolfinx.fem.Function):
            self._adjoint_solutions = self._u.copy()
            self._second_adjoint_solutions = self._u.copy()
            self._tlm_solutions = self._u.copy()
        else:
            assert isinstance(self._u, typing.Iterable)
            self._adjoint_solutions = [u.copy() for u in self._u]
            self._second_adjoint_solutions = [u.copy() for u in self._u]
            self._tlm_solutions = [u.copy() for u in self._u]

    def _compute_residual(self) -> tuple[ufl.Form, dict[Function, Function]]:
        """Convert the formulation :math:`a(u, v)=L(v)` into a residual :math:`F(u_b, v) = 0` where
        :math:`u_b` is the solution of the forward problem at the current time and all coefficients are updated.
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


class NonlinearProblemBlock(_ProblemBlockBase):
    """A linear problem that can be used with adjoint methods.

    This class extends the `dolfinx.fem.petsc.LinearProblem` to support adjoint methods.
    """

    _adjoint_solutions: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]
    _second_adjoint_solutions: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]
    _tlm_solutions: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]
    _rhs: ufl.Form | typing.Sequence[ufl.Form]

    @typing.overload
    def __init__(
        self,
        F: ufl.Form,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: dolfinx.fem.Function | None = None,
        J: ufl.Form | None = None,
        P: ufl.Form | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        problem: "NonlinearProblem" = ...,
    ) -> None: ...

    @typing.overload
    def __init__(
        self,
        F: typing.Sequence[ufl.Form],
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: typing.Sequence[dolfinx.fem.Function] | None = None,
        J: typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        P: typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        problem: "NonlinearProblem" = ...,
    ) -> None: ...

    def __init__(
        self,
        F: ufl.Form | typing.Sequence[ufl.Form],
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function] | None = None,
        J: ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        P: ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        problem: "NonlinearProblem" = None,  # type: ignore[assignment]
    ) -> None:

        assert problem is not None, "problem must be provided."
        # See LinearProblemBlock.__init__ for the rationale for holding a
        # plain (strong) reference here.
        self._problem_obj = problem
        super().__init__(ad_block_tag=ad_block_tag)
        self._preconditioner = P

        # Create overloaded functions
        assert u is not None, "Control variable(s) must be provided."
        self._u: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]
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

        # NOTE: Add mesh and constants as dependencies later on
        u_list = self._u if isinstance(self._u, list) else [self._u]
        for c in _collect_coefficients(J) - set(u_list):
            self.add_dependency(c, no_duplicates=True)
        for c in _collect_coefficients(self._rhs) - set(u_list):
            self.add_dependency(c, no_duplicates=True)

        # Cache form parameters for later
        # NOTE: Should probably be in a struct
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._bcs = bcs if bcs is not None else []

        # No forward/adjoint solver is built here: this block shares the ones
        # owned by self._problem (see NonlinearProblem in ../solvers.py),
        # built once and reused across every block that Problem records
        # instead of once per solve() call.

        if isinstance(self._u, dolfinx.fem.Function):
            self._adjoint_solutions = self._u.copy()  # type: ignore[assignment]
            self._second_adjoint_solutions = self._u.copy()  # type: ignore[assignment]
            self._tlm_solutions = self._u.copy()  # type: ignore[assignment]
        else:
            assert isinstance(self._u, typing.Iterable)
            self._adjoint_solutions = [u.copy() for u in self._u]
            self._second_adjoint_solutions = [u.copy() for u in self._u]
            self._tlm_solutions = [u.copy() for u in self._u]

    def _compute_residual(self) -> tuple[ufl.Form, dict[Function, Function]]:
        """Build the residual :math:`F(u_b, v) = 0` at the current checkpointed dependency values.

        Unlike ``LinearProblemBlock`` (which derives ``F`` from ``a``/``L`` via ``ufl.action``),
        ``F`` is already the residual here -- this only needs to substitute in the checkpointed
        dependency and output values. Settles on the same output shape as
        ``LinearProblemBlock._compute_residual`` (a single summed form plus its replacement map)
        so every method built on top of it (adjoint, TLM, Hessian) is shared on the base class.
        """
        replacement_functions = self.get_outputs()
        replacement_map = self._create_replace_map(self._rhs)

        u_list = self._u if isinstance(self._u, list) else [self._u]
        for u, block in zip(u_list, replacement_functions):
            replacement_map[u] = block.saved_output

        F_form = ufl.replace(sum_form(self._rhs), replacement_map)
        return F_form, replacement_map

    def _refresh_dFdu_state(self, problem: typing.Any) -> None:
        """Refresh ``problem``'s "state" placeholder(s) from this block's own saved outputs.

        Unlike ``LinearProblem`` (where ``dF/du`` never references the state), ``dF/du`` here
        genuinely depends on ``u``'s current value, so the shared adjoint solver's compiled LHS
        must be evaluated at *this* block's checkpointed output before it is used -- another
        block sharing the same solver may have left a different value there.
        """
        state_placeholder = problem._residual_state_placeholder
        state_list = state_placeholder if isinstance(state_placeholder, list) else [state_placeholder]
        for placeholder, out_bv in zip(state_list, self.get_outputs(), strict=True):
            placeholder.x.array[:] = out_bv.saved_output.x.array[:]
            placeholder.x.scatter_forward()
