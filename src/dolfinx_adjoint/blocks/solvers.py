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
    Recursively assigns mixed-space `part` indices to UFL Test and Trial functions
    within nested iterables of forms.

    When solving monolithic block systems in FEniCSx, the UFL arguments must be tagged
    with a `.part()` index corresponding to their block position. For a block matrix
    (list of lists), the TestFunction corresponds to the row index, and the
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
        The replacement arguments are drawn from ``ufl.TestFunctions``/``ufl.TrialFunctions``
        of a single ``ufl.MixedFunctionSpace`` built from the row/column function spaces
        discovered while walking the structure, rather than hand-constructed
        via ``ufl.Argument(..., part=...)``: this is exactly what a user who
        builds the block system directly on a ``ufl.MixedFunctionSpace`` (and
        then calls ``ufl.extract_blocks``) already gets, so a form assembled
        this way and one assembled from a plain ``[[a00, ...], ...]`` nested
        list end up as the same UFL objects -- one code path handles both,
        rather than two subtly different ones.
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
                    num = arg.number()
                    # Because num is 0 for TestFunctions and 1 for TrialFunctions,
                    # it maps perfectly to our nested dimension indices!
                    # If num < len(indices), we have traversed deep enough to assign it.
                    if num < len(indices):
                        spaces.setdefault(indices[num], arg.ufl_function_space())
        elif isinstance(obj, typing.Iterable):
            for i, item in enumerate(obj):
                if item is not None:
                    _discover_spaces(item, indices + (i,))

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


def to_list(data):
    if isinstance(data, (tuple, list)):
        return [to_list(item) for item in data]
    return data


def get_sorted_arguments(arguments: typing.Iterable[ufl.Argument], number: int) -> typing.Iterable[ufl.Argument]:
    """Extract all arguments of a given number, sorted by part."""
    return sorted(filter(lambda x: x.number() == number, arguments), key=lambda a: a.part())


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


class LinearProblemBlock(pyadjoint.Block):
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
        # A plain (strong) reference is deliberate: constructing a LinearProblem
        # as a throwaway local (solve it, then only ever touch the resulting
        # ReducedFunctional) is a common, already-tested pattern, and this
        # block -- kept alive by the tape -- must keep the Problem (and hence
        # its shared solvers) alive for exactly as long as the block itself is
        # reachable, mirroring the lifetime the old per-block-owned solver had.
        # A plain LinearProblemBlock is not itself cyclic garbage (verified:
        # dropping the Problem and the tape releases it via ordinary
        # refcounting, no gc.collect() required), so this does not reintroduce
        # the MPI collective-destruction hazard documented in
        # dolfinx-adjoint-knowledge's mpi-collective-destruction-hazard note --
        # that hazard is specifically about pyadjoint's checkpoint-schedule
        # bookkeeping (an unmerged, not-yet-present feature) making the *tape*
        # cyclic, and would need revisiting if/when that lands.
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
        # ones owned by self._problem() (see LinearProblem in ../solvers.py),
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

    def _problem(self) -> "LinearProblem":
        """Return this block's owning Problem, which owns the shared solvers."""
        return self._problem_obj

    def _recover_bcs(self):
        bcs = []
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, dolfinx.fem.DirichletBC):
                bcs.append(c_rep)
        return bcs

    def _create_replace_map(self, form: ufl.Form | NestedMutableSequence[ufl.Form] | None) -> dict[Function, Function]:
        """Replace dependencies with latest checkpoint."""
        replace_map = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if isinstance(form, ufl.Form):
                if coeff in form.coefficients():
                    replace_map[coeff] = block_variable.saved_output
            elif form is None:
                return {}
            else:
                for f in form:
                    replace_map.update(self._create_replace_map(f))
        for block_variable in self.get_outputs():
            coeff = block_variable.output
            if isinstance(form, ufl.Form):
                if coeff in form.coefficients():
                    replace_map[coeff] = block_variable.saved_output
            elif form is None:
                return {}
            else:
                for f in form:
                    replace_map.update(self._create_replace_map(f))
        return replace_map

    def prepare_recompute_component(
        self, inputs: typing.Sequence[typing.Any], relevant_outputs: typing.Sequence[typing.Any]
    ) -> dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]:
        """Prepare for recomputing the block with different control inputs."""

        # The forward solver (self._problem()) is bound, forever, to compiled
        # forms referencing dedicated placeholder coefficients rather than the
        # user's own dependency objects (see LinearProblem._value_placeholders,
        # which mirrors NonlinearProblem's): writing this call's
        # candidate/checkpointed values into the placeholders -- never into
        # block_variable.output itself -- is what the next solve sees,
        # without ever mutating an object the user (or a Taylor test
        # perturbing a control directly) holds a live reference to.
        problem = self._problem()
        for block_variable in self.get_dependencies():
            placeholder = problem._value_placeholders.get(block_variable.output)
            if placeholder is not None:
                placeholder.x.array[:] = block_variable.saved_output.x.array[:]
                placeholder.x.scatter_forward()

        # Re-establish this block's own bcs on the shared forward solver (the
        # Problem itself -- see _problem()), since another block may have
        # used it with different bcs in between.
        problem.bcs = self._bcs
        problem._u = self._u

        # Clear solution vector
        if isinstance(self._u, dolfinx.fem.Function):
            self._u.x.array[:] = 0.0
        else:
            for ui in self._u:
                ui.x.array[:] = 0.0
        # Solve forward state while halting annotation. Call the base-class
        # solve() directly (not problem.solve()), which would record another
        # block onto the tape.
        with pyadjoint.stop_annotating():
            dolfinx.fem.petsc.LinearProblem.solve(problem)
        return self._u

    def recompute_component(
        self,
        inputs: typing.Iterable[Function],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        prepared: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function],
    ) -> dolfinx.fem.Function:
        """Recompute and return an isolated copy of the solution state."""
        if isinstance(prepared, dolfinx.fem.Function):
            assert idx == 0
            # Return an explicit copy so each tape block gets an isolated state snapshot
            return prepared.copy()
        else:
            assert isinstance(prepared, typing.Iterable)
            return prepared[idx].copy()

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

    def _compute_residual_derivative(self) -> typing.Union[ufl.Form, list[list[ufl.Form]]]:
        """Compute the derivative of the residual with respect to the outputs."""

        res = self._compute_residual()
        F_form = res[0] if isinstance(res, tuple) else res
        assert isinstance(F_form, ufl.Form), "Residual form must be a single UFL form."

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
    ) -> tuple[typing.Union[list[ufl.Form], ufl.Form], dolfinx.fem.Form]:

        # The TLM solver -- and the compiled LHS it solves with, shared
        # verbatim with dF/du (see LinearProblem._get_or_build_dFdu_template)
        # -- are shared across every block this Problem records; likewise the
        # per-dependency TLM right-hand-side templates (see
        # LinearProblem._get_or_build_tlm_rhs_templates) are each compiled
        # once. Refresh this block's own checkpointed values into the
        # placeholders and re-establish this block's own bcs on every call,
        # since another block may have used the same solver in between -- but
        # never rebuild or recompile any of these forms.
        problem = self._problem()
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
        # LinearProblem._get_or_build_tlm_rhs_templates for why an inactive
        # dependency's term must be skipped entirely rather than evaluated
        # with a zeroed direction.
        b_petsc = tlm_solver._b
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
        dolfinx.la.petsc._ghost_update(b_petsc, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)

        # Homogeneous boundary conditions are applied to b_petsc by
        # tlm_solver.solve() itself (it homogenizes using tlm_solver.bcs,
        # already re-established above), so there is no need to duplicate
        # that here.

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
        # shared across every block this Problem records: adjoint(dF/du) does
        # not actually depend on the state u for a linear problem (F is
        # linear in u, so its derivative doesn't reference u's value at all),
        # so once every *other* dependency is routed through a placeholder
        # (see LinearProblem._value_placeholders), that operator is fixed for
        # the life of the Problem and was compiled once, in
        # LinearProblem._get_or_build_adjoint_solver. Refresh this block's own
        # checkpointed values into the placeholders and re-establish this
        # block's own bcs on every call, since another block may have used
        # the same solver in between -- but never rebuild or recompile the
        # form itself.
        problem = self._problem()
        adjoint_solver = problem._get_or_build_adjoint_solver()
        adjoint_solver.bcs = self._bcs
        for block_variable in self.get_dependencies():
            placeholder = problem._value_placeholders.get(block_variable.output)
            if placeholder is not None:
                placeholder.x.array[:] = block_variable.saved_output.x.array[:]
                placeholder.x.scatter_forward()

        # Extract dJ/du[v] from the adjoint inputs.
        if len(adj_inputs) == 1:
            adj_rhs = adj_inputs[0]
            dJdu = adjoint_solver._b
            with dJdu.localForm() as dJdu_loc, adj_rhs.petsc_vec.localForm() as adj_rhs_loc:
                dJdu_loc.array[:] = adj_rhs_loc.array[:]
        else:
            assert len(adj_inputs) == len(self.get_outputs()), (
                f"Expected {len(self.get_outputs())} adjoint inputs, got {len(adj_inputs)})"
            )
            dJdu = adjoint_solver._b
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

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        # First fetch all relevant values
        outputs = self.get_outputs()
        tlm_output = [output.tlm_value for output in outputs if output is not None]
        if (hessian_inputs is None) or (len(tlm_output) == 0):
            return

        # The adjoint solver -- and the compiled LHS it solves with, shared
        # verbatim with the first-order adjoint equation (see
        # LinearProblem._get_or_build_adjoint_solver) -- are shared across
        # every block this Problem records. Refresh this block's own
        # checkpointed values into the placeholders and re-establish this
        # block's own bcs on every call, since another block may have used
        # the same solver in between -- but never rebuild or recompile the
        # LHS itself.
        problem = self._problem()
        adjoint_solver = problem._get_or_build_adjoint_solver()
        adjoint_solver.bcs = self._bcs
        for block_variable in self.get_dependencies():
            placeholder = problem._value_placeholders.get(block_variable.output)
            if placeholder is not None:
                placeholder.x.array[:] = block_variable.saved_output.x.array[:]
                placeholder.x.scatter_forward()

        if len(outputs) == 1:
            # Scalar problem: use the cached per-dependency Hessian templates
            # (see LinearProblem._get_or_build_hessian_templates) instead of
            # rebuilding and recompiling the SOA right-hand side symbolically
            # on every call.
            _, seed_placeholders, state_placeholder = problem._get_or_build_tlm_rhs_templates()
            soa_templates, _, _ = problem._get_or_build_hessian_templates()

            state_placeholder.x.array[:] = outputs[0].saved_output.x.array[:]  # type: ignore[union-attr]
            state_placeholder.x.scatter_forward()  # type: ignore[union-attr]
            problem._adjoint_solution_placeholder.x.array[:] = self._adjoint_solutions.x.array[:]  # type: ignore[union-attr]
            problem._adjoint_solution_placeholder.x.scatter_forward()  # type: ignore[union-attr]
            problem._hessian_u_seed.x.array[:] = tlm_output[0].x.array[:]  # type: ignore[union-attr]
            problem._hessian_u_seed.x.scatter_forward()  # type: ignore[union-attr]

            b = adjoint_solver._b
            with b.localForm() as b_loc:
                b_loc.set(0.0)
            for block_variable in self.get_dependencies():
                tlm_input = block_variable.tlm_value
                if tlm_input is None:
                    continue
                c = block_variable.output
                if isinstance(c, (dolfinx.mesh.Mesh, dolfinx.fem.DirichletBC)):
                    raise NotImplementedError(f"Hessian computation for {type(c)} control not implemented yet.")
                template = soa_templates.get(c)
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
            adjoint_solver._b = b

        else:
            # Blocked problem: Hessian cross-term templating is not
            # implemented for this case (see the module-level plan notes);
            # this keeps the pre-templating, per-call ufl.replace + recompile
            # path, with the shared adjoint solver applying only the
            # ownership-move.
            dFdu_form = self._compute_residual_derivative()
            unknowns = [output.saved_output for output in self.get_outputs()]
            summed_form = sum_form(dFdu_form)
            d2Fdu2 = ufl.algorithms.expand_derivatives(ufl.derivative(summed_form, unknowns, tlm_output))

            # Assemble right hand side of second order adjoint equation
            # Note this term should always be zero for linear problems, but we include it for completeness.
            if not d2Fdu2.empty():
                raise RuntimeError(f"This term {d2Fdu2:s} should be zero for linear problems.")
            b_form = d2Fdu2
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
                    bi = dolfinx.la.vector(
                        out_i.function_space.dofmap.index_map, out_i.function_space.dofmap.index_map_bs
                    )
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
            b = adjoint_solver._b
            local_arrays = [bi.array[: bi.index_map.size_local * bi.block_size] for bi in bs]
            dolfinx.la.petsc.assign(local_arrays, b)
            b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        # The SOA (second-order-adjoint) equation shares its LHS verbatim with
        # the first-order adjoint equation (both are adjoint(dF/du), computed
        # from dFdu_form above) -- already correct and permanent on
        # adjoint_solver, so no rebuild or recompile needed here either.
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
        c = block_variable.output

        (F_form, replacement_map), adj_sol, adj_sol2 = prepared

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
            # Scalar problem: use the cached per-dependency Hessian templates
            # (see LinearProblem._get_or_build_hessian_templates) instead of
            # rebuilding and recompiling the Hessian-action output
            # symbolically on every call. All the placeholders these
            # templates reference (the dependency values, the state, and both
            # adjoint solutions) were already refreshed by
            # prepare_evaluate_hessian above; only the per-dependency
            # "direction" seeds need setting here, and only for dependencies
            # that actually have a tangent-linear value this call -- see
            # LinearProblem._get_or_build_hessian_templates for why an
            # inactive dependency's cross term must be skipped entirely
            # rather than evaluated with a zeroed direction.
            problem = self._problem()
            _, fixed_templates, cross_templates = problem._get_or_build_hessian_templates()
            _, seed_placeholders, _ = problem._get_or_build_tlm_rhs_templates()

            fixed_template = fixed_templates[c]
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
                template = cross_templates.get((c, c2))
                if template is None:
                    continue
                seed2 = seed_placeholders[c2]
                seed2.x.array[:] = tlm_input.x.array[:]
                seed2.x.scatter_forward()
                assemble_compiled_form(template, hessian_output)

            hessian_output.array[:] *= -1.0
            return hessian_output

        # Blocked problem: Hessian cross-term templating is not implemented
        # for this case (see the module-level plan notes); this keeps the
        # pre-templating, per-call ufl.replace + recompile path.
        #
        # We are trying to compute (dF/dm)^T lambda_1
        # and (dF_dm)^T lambda_ 2. However, standard approach of UFL
        # does not work for MixedFunctionSpaces, as the control space is not
        # mixed. Therefore, we instead we compute it as dL/dm = d(lambda_i^T F(m))/dm,
        # which is equivalent.
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


class NonlinearProblemBlock(pyadjoint.Block):
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
        if J is not None:
            assert isinstance(J, ufl.Form)
            for c in J.coefficients():
                if c not in u_list:  # Exclude unknown
                    self.add_dependency(c, no_duplicates=True)
        if self._rhs is not None:
            assert isinstance(self._rhs, ufl.Form)
            for c in self._rhs.coefficients():
                if c not in u_list:  # Exclude unknown
                    self.add_dependency(c, no_duplicates=True)

        # Cache form parameters for later
        # NOTE: Should probably be in a struct
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._bcs = bcs if bcs is not None else []

        # No forward/adjoint solver is built here: this block shares the ones
        # owned by self._problem() (see NonlinearProblem in ../solvers.py),
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

    def _problem(self) -> "NonlinearProblem":
        """Return this block's owning Problem, which owns the shared solvers."""
        return self._problem_obj

    def _recover_bcs(self):
        bcs = []
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, dolfinx.fem.DirichletBC):
                bcs.append(c_rep)
        return bcs

    def _create_replace_map(self, form: ufl.Form) -> dict[Function, Function]:
        """Replace dependencies with latest checkpoint."""
        replace_map = {}
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if coeff in form.coefficients():
                replace_map[coeff] = block_variable.saved_output
        return replace_map

    def _replace_coefficients_in_form(self, form: ufl.Form) -> ufl.Form:
        """Replace coefficients in the form with saved outputs.

        Args:
            form: The UFL form to replace coefficients in.
        """
        replace_map = self._create_replace_map(form)
        return ufl.replace(form, replace_map)

    def prepare_recompute_component(self, inputs, relevant_outputs):
        """Prepare for recomputing the block with different control inputs."""

        # The SNES the shared forward solver (self._problem()) owns is bound,
        # forever, to a fixed set of placeholder coefficients rather than the
        # user's own dependency objects (see
        # NonlinearProblem._value_placeholders): writing this call's
        # candidate/checkpointed values into the placeholders -- never into
        # block_variable.output itself -- is what makes the SNES see them,
        # without ever mutating an object the user (or a Taylor test
        # perturbing a control directly) holds a live reference to.
        problem = self._problem()
        for block_variable in self.get_dependencies():
            placeholder = problem._value_placeholders.get(block_variable.output)
            if placeholder is not None:
                placeholder.x.array[:] = block_variable.saved_output.x.array[:]
                placeholder.x.scatter_forward()

        # Re-establish this block's own bcs on the shared forward solver (the
        # Problem itself -- see _problem()), since another block may have used
        # it with different bcs in between.
        problem.bcs = self._bcs

        # Warm-start original unknown objects in place
        u_list = problem._u if isinstance(problem._u, list) else [problem._u]
        for idx, out_bv in relevant_outputs:
            u_list[idx].x.array[:] = out_bv.saved_output.x.array[:]
            u_list[idx].x.scatter_forward()

        return None

    def recompute_component(
        self, inputs: typing.Iterable[Function], block_variable, idx: int, prepared: None
    ) -> Function:
        """Recompute the block with the prepared linear problem."""
        problem = self._problem()
        # Call the base-class solve() directly (not problem.solve()), which
        # would record another block onto the tape.
        with pyadjoint.tape.stop_annotating():
            dolfinx.fem.petsc.NonlinearProblem.solve(problem)
        if isinstance(problem._u, list):
            output = problem._u[idx]
        else:
            output = problem._u
        assert isinstance(output, Function)
        return output

    def _should_compute_boundary_adjoint(
        self, relevant_dependencies: typing.List[tuple[int, pyadjoint.block_variable.BlockVariable]]
    ) -> bool:
        bdy = False
        for _, dep in relevant_dependencies:
            if isinstance(dep.output, dolfinx.fem.DirichletBC):
                bdy = True
                break
        return bdy

    @classmethod
    @typing.overload
    def _compute_adjoint(
        cls, form: typing.Sequence[typing.Sequence[ufl.Form]]
    ) -> typing.Sequence[typing.Sequence[ufl.Form]]: ...

    @classmethod
    @typing.overload
    def _compute_adjoint(cls, form: ufl.Form) -> ufl.Form: ...

    @classmethod
    def _compute_adjoint(
        cls, form: ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]]
    ) -> ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]]:
        """
        Compute adjoint of a bilinear form :math:`a(u, v)`, which could be written as a blocked system.
        """
        if isinstance(form, ufl.Form):
            return ufl.adjoint(form)
        else:
            assert isinstance(form, typing.Iterable)
            adj_form: list[list[ufl.Form]] = []
            tmp_form: list[list[ufl.Form]] = []
            for i, f_i in enumerate(form):
                tmp_form.append([])
                adj_form.append([])
                for j, form_ij in enumerate(f_i):
                    tmp_form[i].append(ufl.adjoint(form_ij))
                    adj_form[i].append(ufl.adjoint(form_ij))
            for i, f_i in enumerate(tmp_form):
                for j, form_ij in enumerate(f_i):
                    adj_form[j][i] = form_ij
            return adj_form

    def _compute_residual(self) -> typing.Union[ufl.Form, list[ufl.Form]]:
        """Convert the formulation :math:`a(u, v)=L(v)` into a residual :math:`F(u_b, v) = 0` where
        :math:`u_b` is the solution of the forward problem at the current time and all coefficients are updated.
        """
        # NOTE: Should probably be possible to compile this form once.
        replacement_functions = self.get_outputs()
        assert isinstance(self._rhs, (ufl.Form, typing.Sequence))
        assert isinstance(self._rhs, ufl.Form)
        replacement_map = self._create_replace_map(self._rhs)

        u_list = self._u if isinstance(self._u, list) else [self._u]
        for u, block in zip(u_list, replacement_functions):
            replacement_map[u] = block.saved_output

        if isinstance(self._u, dolfinx.fem.Function):
            F_form = ufl.replace(self._rhs, replacement_map)
        else:
            assert isinstance(self._rhs, typing.Iterable)
            F_form = [ufl.replace(rhs_j, replacement_map) for rhs_j in self._rhs]
        return F_form

    def _compute_residual_derivative(self) -> typing.Union[ufl.Form, list[list[ufl.Form]]]:
        """Compute the derivative of the residual with respect to the outputs."""

        F_form = self._compute_residual()
        outputs = [output.saved_output for output in self.get_outputs()]
        if len(outputs) == 1:
            assert isinstance(F_form, ufl.Form)
            dFdu = ufl.derivative(F_form, outputs[0], ufl.TrialFunction(outputs[0].function_space))
        else:
            assert isinstance(F_form, list)
            dFdu = []
            for i in range(len(outputs)):
                dFdu.append([])
                for j in range(len(outputs)):
                    dFdu[-1].append(ufl.derivative(F_form[i], outputs[j], ufl.TrialFunction(outputs[j].function_space)))
        return dFdu

    def prepare_evaluate_tlm(
        self, inputs, tlm_inputs, relevant_outputs
    ) -> typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]]:
        # The TLM solver -- and the compiled LHS it solves with, shared
        # verbatim with dF/du (see NonlinearProblem._get_or_build_dFdu_template)
        # -- are shared across every block this Problem records; likewise the
        # per-dependency TLM right-hand-side templates (see
        # NonlinearProblem._get_or_build_tlm_rhs_templates) are each compiled
        # once. Refresh this block's own checkpointed values into the
        # placeholders and re-establish this block's own bcs on every call,
        # since another block may have used the same solver in between -- but
        # never rebuild or recompile any of these forms.
        problem = self._problem()
        tlm_solver = problem._get_or_build_tlm_solver()
        tlm_solver.bcs = self._bcs
        templates, seed_placeholders, state_placeholder = problem._get_or_build_tlm_rhs_templates()

        for block_variable in self.get_dependencies():
            placeholder = problem._value_placeholders.get(block_variable.output)
            if placeholder is not None:
                placeholder.x.array[:] = block_variable.saved_output.x.array[:]
                placeholder.x.scatter_forward()
        out_bv = self.get_outputs()[0]
        state_placeholder.x.array[:] = out_bv.saved_output.x.array[:]
        state_placeholder.x.scatter_forward()

        # Assemble RHS vector using the shared solver's cached vector,
        # accumulating only the dependencies that actually have a
        # tangent-linear value this call -- see
        # NonlinearProblem._get_or_build_tlm_rhs_templates for why an
        # inactive dependency's term must be skipped entirely rather than
        # evaluated with a zeroed direction.
        b_petsc = tlm_solver._b
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
        dolfinx.la.petsc._ghost_update(b_petsc, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)

        # Homogeneous boundary conditions are applied to b_petsc by
        # tlm_solver.solve() itself (it homogenizes using tlm_solver.bcs,
        # already re-established above), so there is no need to duplicate
        # that here.
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
    ) -> typing.Union[ufl.Form, typing.Iterable[ufl.Form]]:
        """Prepare the block for evaluating the adjoint."""

        # The adjoint solver -- and the compiled LHS it solves with -- are
        # shared across every block this Problem records: once every non-u
        # dependency is routed through its placeholder and "u at this
        # evaluation point" through its own dedicated placeholder (see
        # NonlinearProblem._get_or_build_adjoint_solver), that operator is
        # fixed for the life of the Problem and was compiled once. Refresh
        # this block's own checkpointed values into the placeholders and
        # re-establish this block's own bcs on every call, since another
        # block may have used the same solver in between -- but never
        # rebuild or recompile the LHS itself.
        problem = self._problem()
        adjoint_solver = problem._get_or_build_adjoint_solver()
        adjoint_solver.bcs = self._bcs
        for block_variable in self.get_dependencies():
            placeholder = problem._value_placeholders.get(block_variable.output)
            if placeholder is not None:
                placeholder.x.array[:] = block_variable.saved_output.x.array[:]
                placeholder.x.scatter_forward()
        out_bv = self.get_outputs()[0]
        problem._state_placeholder.x.array[:] = out_bv.saved_output.x.array[:]  # type: ignore[union-attr]
        problem._state_placeholder.x.scatter_forward()  # type: ignore[union-attr]

        # F_form is still needed by evaluate_adj_component (to build each
        # dependency's own sensitivity form), but the adjoint LHS itself is
        # already correct on adjoint_solver -- no rebuild, no recompile.
        F_form = self._compute_residual()
        # Extract dJ/du[v] from the adjoint inputs.
        assert len(adj_inputs) == 1
        adj_rhs = adj_inputs[0]
        dJdu = adjoint_solver._b
        with dJdu.localForm() as dJdu_loc, adj_rhs.petsc_vec.localForm() as adj_rhs_loc:
            dJdu_loc.array[:] = adj_rhs_loc.array[:]

        adjoint_solver.solve()
        if isinstance(self._adjoint_solutions, list):
            for adj_sol, sol in zip(self._adjoint_solutions, adjoint_solver.u):
                adj_sol.x.array[:] = sol.x.array[:]
        else:
            assert isinstance(self._adjoint_solutions, dolfinx.fem.Function)
            self._adjoint_solutions.x.array[:] = adjoint_solver.u.x.array[:]
        return F_form

    def evaluate_adj_component(
        self,
        inputs: typing.Iterable[Function],
        adj_inputs: typing.Iterable[dolfinx.la.Vector],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        prepared: typing.Union[ufl.Form, typing.Iterable[ufl.Form]],
    ) -> typing.Union[_SpecialVector, typing.Iterable[_SpecialVector]]:
        """Evaluate the adjoint component, i.e. :math:`\frac{\\partial Au - b}{\\partial c}`."""

        residual = prepared

        c = block_variable.output

        c_rep = block_variable.saved_output
        if isinstance(c, dolfinx.fem.Function):
            dc = ufl.TrialFunction(c.function_space)
        else:
            raise NotImplementedError(f"Unsupported control {type(c)}")
        dFdm = -ufl.derivative(residual, c_rep, dc)

        # Safe return for empty sensitivities
        if dFdm.empty():
            dFdm = ufl.ZeroBaseForm((dc,))

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

    def prepare_evaluate_hessian(self, inputs, hessian_inputs, adj_inputs, relevant_dependencies):
        # First fetch all relevant values
        outputs = self.get_outputs()
        tlm_output = [output.tlm_value for output in outputs if output is not None]
        if (hessian_inputs is None) or (len(tlm_output) == 0):
            return

        # The adjoint solver -- and the compiled LHS it solves with, shared
        # verbatim with the first-order adjoint equation (both are
        # adjoint(dF/du), see NonlinearProblem._get_or_build_adjoint_solver)
        # -- are shared across every block this Problem records. Refresh this
        # block's own checkpointed values into the placeholders and
        # re-establish this block's own bcs on every call, since another
        # block may have used the same solver in between -- but never
        # rebuild or recompile the LHS itself.
        problem = self._problem()
        adjoint_solver = problem._get_or_build_adjoint_solver()
        adjoint_solver.bcs = self._bcs
        for block_variable in self.get_dependencies():
            placeholder = problem._value_placeholders.get(block_variable.output)
            if placeholder is not None:
                placeholder.x.array[:] = block_variable.saved_output.x.array[:]
                placeholder.x.scatter_forward()
        problem._state_placeholder.x.array[:] = outputs[0].saved_output.x.array[:]  # type: ignore[union-attr]
        problem._state_placeholder.x.scatter_forward()  # type: ignore[union-attr]

        assert len(outputs) == 1, "Hessian computation only implemented for single output blocks."
        assert len(tlm_output) == 1, "Hessian computation only implemented for single TLM output blocks."
        assert len(hessian_inputs) == 1, "Hessian computation only implemented for single hessian input blocks."

        # The SOA solver -- and the compiled LHS it solves with, shared
        # verbatim with the first-order adjoint equation -- is shared across
        # every block this Problem records; likewise the SOA right-hand-side
        # templates (see NonlinearProblem._get_or_build_hessian_templates)
        # are each compiled once. Refresh the placeholders they reference and
        # never rebuild or recompile any of these forms.
        soa_self_template, soa_cross_templates, _, _ = problem._get_or_build_hessian_templates()
        _, seed_placeholders, _ = problem._get_or_build_tlm_rhs_templates()

        problem._adjoint_solution_placeholder.x.array[:] = self._adjoint_solutions.x.array[:]  # type: ignore[union-attr]
        problem._adjoint_solution_placeholder.x.scatter_forward()  # type: ignore[union-attr]
        problem._hessian_u_seed.x.array[:] = tlm_output[0].x.array[:]  # type: ignore[union-attr]
        problem._hessian_u_seed.x.scatter_forward()  # type: ignore[union-attr]

        b = adjoint_solver._b
        with b.localForm() as b_loc:
            b_loc.set(0.0)
        if soa_self_template is not None:
            dolfinx.fem.petsc.assemble_vector(b, soa_self_template)
        for block_variable in self.get_dependencies():
            tlm_input = block_variable.tlm_value
            if tlm_input is None:
                continue
            c = block_variable.output
            if isinstance(c, (dolfinx.mesh.Mesh, dolfinx.fem.DirichletBC)):
                raise NotImplementedError(f"Hessian computation for {type(c)} control not implemented yet.")
            template = soa_cross_templates.get(c)
            if template is None:
                continue
            seed = seed_placeholders[c]
            seed.x.array[:] = tlm_input.x.array[:]
            seed.x.scatter_forward()
            dolfinx.fem.petsc.assemble_vector(b, template)
        dolfinx.la.petsc._ghost_update(b, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        b.scale(-1)
        with b.localForm() as b_loc, hessian_inputs[0].petsc_vec.localForm() as hess_loc:
            b_loc.array[:] += hess_loc.array[:]

        # The SOA (second-order-adjoint) equation shares its LHS verbatim
        # with the first-order adjoint equation (both are
        # adjoint(dF/du) -- already correct and permanent on adjoint_solver,
        # so no rebuild or recompile needed here either.
        #
        # Redirect the shared solver's scratch solution storage at this
        # block's own second-adjoint buffer for the duration of this solve.
        adjoint_solver._u = self._second_adjoint_solutions
        adjoint_solver.solve()
        if isinstance(self._second_adjoint_solutions, list):
            for adj_sol, sol in zip(self._second_adjoint_solutions, adjoint_solver.u):
                adj_sol.x.array[:] = sol.x.array[:]
        else:
            self._second_adjoint_solutions.x.array[:] = adjoint_solver.u.x.array[:]
            problem._second_adjoint_solution_placeholder.x.array[:] = (  # type: ignore[union-attr]
                self._second_adjoint_solutions.x.array[:]
            )
            problem._second_adjoint_solution_placeholder.x.scatter_forward()  # type: ignore[union-attr]
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
        c = block_variable.output

        # prepared (F_form, adj_sol, adj_sol2) is unused here: every quantity
        # this method needs is already baked into the cached Hessian
        # templates below, refreshed from those same values by
        # prepare_evaluate_hessian.
        outputs = self.get_outputs()
        assert len(outputs) == 1, "Hessian computation only implemented for single output blocks."

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

        # Use the cached per-dependency Hessian templates (see
        # NonlinearProblem._get_or_build_hessian_templates) instead of
        # rebuilding and recompiling the Hessian-action output symbolically
        # on every call. All the placeholders these templates reference (the
        # dependency values, the state, and both adjoint solutions) were
        # already refreshed by prepare_evaluate_hessian above; only the
        # per-dependency "direction" seeds need setting here, and only for
        # dependencies that actually have a tangent-linear value this call --
        # see NonlinearProblem._get_or_build_hessian_templates for why an
        # inactive dependency's cross term must be skipped entirely rather
        # than evaluated with a zeroed direction.
        problem = self._problem()
        _, _, fixed_templates, cross_templates = problem._get_or_build_hessian_templates()
        _, seed_placeholders, _ = problem._get_or_build_tlm_rhs_templates()

        fixed_template = fixed_templates[c]
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
            template = cross_templates.get((c, c2))
            if template is None:
                continue
            seed2 = seed_placeholders[c2]
            seed2.x.array[:] = tlm_input.x.array[:]
            seed2.x.scatter_forward()
            assemble_compiled_form(template, hessian_output)

        hessian_output.array[:] *= -1.0
        return hessian_output
