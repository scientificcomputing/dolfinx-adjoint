from __future__ import annotations

import typing

from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import pyadjoint
import ufl
from dolfinx.fem.function import Function as _Function

from ..compat import compute_form_adjoint
from ..petsc_utils import LinearAdjointProblem, solve_linear_problem
from ..types import Function
from .assembly import _create_vector, _SpecialVector, assemble_compiled_form

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
    """
    replace_map = {}

    def _build_map(obj: NestedSequence[ufl.Form], indices: tuple[int, ...]) -> None:
        """
        Recursively discover forms, tracking the depth and index of the nesting.
        `indices` will be `(row,)` for vectors and `(row, col)` for matrices.
        """
        if isinstance(obj, ufl.Form):
            for arg in obj.arguments():
                # Only map arguments that haven't been assigned a part yet
                if arg.part() is None and arg not in replace_map:
                    num = arg.number()

                    # Because num is 0 for TestFunctions and 1 for TrialFunctions,
                    # it maps perfectly to our nested dimension indices!
                    # If num < len(indices), we have traversed deep enough to assign it.
                    if num < len(indices):
                        replace_map[arg] = ufl.Argument(arg.ufl_function_space(), number=num, part=indices[num])

        elif isinstance(obj, typing.Iterable):
            for i, item in enumerate(obj):
                if item is not None:
                    # Append current topological index to the path and recurse
                    _build_map(item, indices + (i,))

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

    # 1. Build a shared map across all inputs (e.g., ensuring RHS TestFunctions
    # perfectly match LHS TestFunctions)
    for struct in form_structs:
        _build_map(struct, ())

    # 2. If no replacements are needed, exit early to save computation
    if not replace_map:
        return form_structs if len(form_structs) > 1 else form_structs[0]

    # 3. Apply the replacements and unpack if necessary
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
        kind: str | None = None,
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
        petsc_options_prefix: str = "dxa_linear_problem_block_",
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
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
        petsc_options_prefix: str = "dxa_linear_problem_block_",
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
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
        petsc_options_prefix: str = "dxa_linear_problem_block_",
    ) -> None:

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
        self._compiled_lhs = dolfinx.fem.form(
            self._lhs,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )
        self._compiled_rhs = dolfinx.fem.form(
            self._rhs,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )
        # Cache form parameters for later
        # NOTE: Should probably be in a struct
        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._petsc_options = petsc_options if petsc_options is not None else {}
        self._petsc_options_prefix = petsc_options_prefix
        self._bcs = bcs if bcs is not None else []

        # Add dependencies from the boundary conditions
        if self._bcs is not None:
            for bc in self._bcs:
                if hasattr(bc, "block_variable"):
                    self.add_dependency(bc, no_duplicates=True)

        # Solver for recomputing the linear problem
        self._forward_solver = dolfinx.fem.petsc.LinearProblem(
            a=self._lhs,  # type: ignore[arg-type]
            L=self._rhs,  # type: ignore[arg-type]
            bcs=self._bcs,
            u=self._u,  # type: ignore[arg-type]
            P=self._preconditioner,  # type: ignore[arg-type]
            petsc_options=self._petsc_options,
            petsc_options_prefix=petsc_options_prefix,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            kind=kind,  # type: ignore[arg-type]
            entity_maps=self._entity_maps,
        )  # type: ignore[misc]

        self._kind = "nest" if self._forward_solver.A.getType() == "nest" else kind

        if isinstance(self._u, dolfinx.fem.Function):
            self._adjoint_solutions = self._u.copy()
            self._second_adjoint_solutions = self._u.copy()
            self._tlm_solutions = self._u.copy()
        else:
            assert isinstance(self._u, typing.Iterable)
            self._adjoint_solutions = [u.copy() for u in self._u]
            self._second_adjoint_solutions = [u.copy() for u in self._u]
            self._tlm_solutions = [u.copy() for u in self._u]

        self._adjoint_solver = LinearAdjointProblem(
            self._compute_adjoint(sum_form(self._lhs)),  # type: ignore[arg-type]
            self._rhs,  # type: ignore[arg-type]
            bcs=self._bcs,
            P=self._preconditioner,  # type: ignore[arg-type]
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            petsc_options=self._adjoint_petsc_options,
            petsc_options_prefix=self._petsc_options_prefix,
            kind=kind,  # type: ignore[arg-type]
            entity_maps=self._entity_maps,
        )  # type: ignore[misc]
        self._tlm_solver = None

    def _recover_bcs(self):
        bcs = []
        for block_variable in self.get_dependencies():
            c = block_variable.output
            c_rep = block_variable.saved_output

            if isinstance(c, dolfinx.fem.DirichletBC):
                bcs.append(c_rep)
        return bcs

    def construct_tlm_solver(self):
        dFdu_form = self._compute_residual_derivative()
        tlm_solver = LinearAdjointProblem(
            dFdu_form,  # type: ignore[arg-type]
            self._rhs,  # type: ignore[arg-type]
            bcs=self._bcs,
            u=self._tlm_solutions,  # type: ignore[arg-type]
            P=self._preconditioner,  # type: ignore[arg-type]
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            petsc_options=self._tlm_petsc_options,
            petsc_options_prefix=self._petsc_options_prefix,
            kind=self._kind,  # type: ignore[arg-type]
            entity_maps=self._entity_maps,
        )  # type: ignore[misc]
        return tlm_solver

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

    def _create_recompute_replace_map(self, inputs: typing.Sequence[typing.Any]) -> dict:
        """Map original dependency coefficients to active recomputation inputs."""
        replace_map = {}
        for block_variable, input_val in zip(self.get_dependencies(), inputs, strict=True):
            coeff = block_variable.output
            val = input_val.output if isinstance(input_val, pyadjoint.block_variable.BlockVariable) else input_val
            replace_map[coeff] = val
        return replace_map

    @typing.overload
    def _replace_form_coefficients_recompute(
        self, form: ufl.Form | NestedSequence[ufl.Form], replace_map: dict
    ) -> ufl.Form | NestedMutableSequence[ufl.Form | None]: ...
    @typing.overload
    def _replace_form_coefficients_recompute(self, form: None, replace_map: dict) -> None: ...
    def _replace_form_coefficients_recompute(
        self, form: ufl.Form | NestedSequence[ufl.Form] | None, replace_map: dict
    ) -> ufl.Form | NestedMutableSequence[ufl.Form | None] | None:
        """Recursively replace form coefficients for scalar or blocked form structures."""
        if form is None:
            return None
        if isinstance(form, ufl.Form):
            coeffs_in_form = form.coefficients()
            sub_map = {k: v for k, v in replace_map.items() if k in coeffs_in_form}
            return ufl.replace(form, sub_map) if sub_map else form
        elif isinstance(form, typing.Sequence):
            return [self._replace_form_coefficients_recompute(f, replace_map) for f in form]
        else:
            raise TypeError(f"Cannot replace coefficients in form of type {type(form)}")

    def prepare_recompute_component(
        self, inputs: typing.Sequence[typing.Any], relevant_outputs: typing.Sequence[typing.Any]
    ) -> dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]:
        """Prepare for recomputing the block with different control inputs."""

        replace_map = self._create_recompute_replace_map(inputs)

        # 1. Substitute forms using active candidate inputs instead of static tape checkpoints
        lhs = self._replace_form_coefficients_recompute(self._lhs, replace_map)
        rhs = self._replace_form_coefficients_recompute(self._rhs, replace_map)
        preconditioner = (
            self._replace_form_coefficients_recompute(self._preconditioner, replace_map)
            if self._preconditioner is not None
            else None
        )

        # 2. Recompile UFL forms with candidate inputs
        compiled_lhs = dolfinx.fem.form(
            lhs,  # type: ignore[arg-type]
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        compiled_rhs = dolfinx.fem.form(
            rhs,  # type: ignore[arg-type]
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        compiled_preconditioner = (
            dolfinx.fem.form(
                preconditioner,  # type: ignore[arg-type]
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )
            if preconditioner is not None
            else None
        )

        # 3. Hot-swap solver forms
        self._forward_solver._a = compiled_lhs  # type: ignore[assignment]
        self._forward_solver._L = compiled_rhs  # type: ignore[assignment]
        self._forward_solver._preconditioner = compiled_preconditioner
        self._forward_solver.bcs = self._bcs
        self._forward_solver._u = self._u

        # Clear solution vector
        if isinstance(self._u, dolfinx.fem.Function):
            self._u.x.array[:] = 0.0
        else:
            for ui in self._u:
                ui.x.array[:] = 0.0
        # 4. Solve forward state while halting annotation
        with pyadjoint.stop_annotating():
            self._forward_solver.solve()
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

        F_form, replacement_map = self._compute_residual()
        if self._tlm_solver is None:
            self._tlm_solver = self.construct_tlm_solver()
        # Even if the solver is cached, we need to replace the form, as the output from pyadjoint
        # is stored in a new function.
        assert isinstance(self._tlm_solver, LinearAdjointProblem)
        self._tlm_solver._a = dolfinx.fem.form(
            self._compute_residual_derivative(),
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        #  Build RHS (dFdm) for the monolithic system
        if isinstance(self._u, list):
            test_funcs = get_sorted_arguments(F_form.arguments(), 0)
            dFdm = sum([ufl.ZeroBaseForm((test,)) for test in test_funcs])
        else:
            test_funcs = [F_form.arguments()[0]]
            dFdm = ufl.ZeroBaseForm((test_funcs[0],))

        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            c_rep = block_variable.saved_output
            if tlm_value is None:
                continue
            assert c_rep in replacement_map.values()
            # Accumulate sensitivities across all block components
            dFdm += ufl.derivative(-F_form, c_rep, tlm_value)

        # Safely wrap zero forms to prevent compilation crashes
        dFdm = ufl.algorithms.expand_derivatives(dFdm)
        if isinstance(self._u, list):
            blocks = ufl.extract_blocks(dFdm)
            if len(blocks) != len(self._u):
                # Some zero blocks, manually pad with zero forms
                _dFdm = [ufl.ZeroBaseForm((test,)) for test in test_funcs]
                for block in blocks:
                    args = block.arguments()
                    assert len(args) == 1, "Expected a single test function in the block."
                    _dFdm[args[0].part()] = block
                dFdm = _dFdm
        else:
            if dFdm == 0 or dFdm.empty():
                dFdm = ufl.ZeroBaseForm((test_funcs[0],))

        dFdm_compiled = dolfinx.fem.form(
            dFdm,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )

        # 3. Assemble RHS Vector utilizing the internal block-allocated vector
        b_petsc = self._tlm_solver._b
        with b_petsc.localForm() as b_loc:
            b_loc.set(0.0)

        dolfinx.fem.petsc.assemble_vector(b_petsc, dFdm_compiled)
        dolfinx.la.petsc._ghost_update(b_petsc, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)

        # 4. Apply Homogeneous Boundary Conditions safely directly to the block vector
        if self._bcs:
            try:
                for bc in self._bcs:
                    bc.set(b_petsc.array_w, alpha=0.0)
            except RuntimeError:
                # FEniCSx throws RuntimeError for flat .set() on a blocked array.
                # We use the bcs_by_block utility to handle the nested extraction.
                from dolfinx.fem.bcs import bcs_by_block

                V_ext = dolfinx.fem.extract_function_spaces(dFdm_compiled)
                bcs_lift = bcs_by_block(V_ext, self._bcs)
                dolfinx.fem.petsc.set_bc(b_petsc, bcs_lift, alpha=0.0)

        # 5. Solve the full monolithic TLM system
        self._tlm_solver.solve()

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

        # Extract dJ/du[v] from the adjoint inputs.
        if len(adj_inputs) == 1:
            adj_rhs = adj_inputs[0]
            dJdu = self._adjoint_solver._b
            with dJdu.localForm() as dJdu_loc, adj_rhs.petsc_vec.localForm() as adj_rhs_loc:
                dJdu_loc.array[:] = adj_rhs_loc.array[:]
        else:
            assert len(adj_inputs) == len(self.get_outputs()), (
                f"Expected {len(self.get_outputs())} adjoint inputs, got {len(adj_inputs)})"
            )
            dJdu = self._adjoint_solver._b
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

        # Compute (dF/du[v])* for the linear problem.
        F_form, replacement_map = self._compute_residual()
        dFdu = self._compute_residual_derivative()
        summed = sum_form(dFdu)
        dFdu_adj = compute_form_adjoint(summed)
        dFdu_adj = ufl.algorithms.apply_derivatives.apply_derivatives(ufl.algorithms.expand_derivatives(dFdu_adj))
        assert dFdu_adj.empty() is False, "Adjoint of dF/du[v] is empty. Check if the problem is linear."
        # Solve adjoint problem
        compiled_dFdu = dolfinx.fem.form(
            ufl.extract_blocks(dFdu_adj),  # type: ignore[arg-type]
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        self._adjoint_solver._a = compiled_dFdu
        self._adjoint_solver.solve()
        if isinstance(self._adjoint_solutions, list):
            for adj_sol, sol in zip(self._adjoint_solutions, self._adjoint_solver.u):
                adj_sol.x.array[:] = sol.x.array[:]
        else:
            assert isinstance(self._adjoint_solutions, dolfinx.fem.Function)
            self._adjoint_solutions.x.array[:] = self._adjoint_solver.u.x.array[:]
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

        # Using the equation Form we derive dF/du, d^2F/du^2 * du/dm * direction.
        dFdu_form = self._compute_residual_derivative()

        # For linear forms d2Fdu2 is zero, but we include it for completeness.

        unknowns = [output.saved_output for output in self.get_outputs()]
        summed_form = sum_form(dFdu_form)
        d2Fdu2 = ufl.algorithms.expand_derivatives(ufl.derivative(summed_form, unknowns, tlm_output))

        # bdy = self._should_compute_boundary_adjoint(relevant_dependencies)

        # Assemble right hand side of second order adjoint equation
        # Note this term should always be zero for linear problems, but we include it for completeness.
        if not d2Fdu2.empty():
            raise RuntimeError(f"This term {d2Fdu2:s} should be zero for linear problems.")
        b_form = d2Fdu2 if d2Fdu2.empty() else ufl.action(ufl.adjoint(d2Fdu2), self._adjoint_solutions)
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

        if len(outputs) == 1:
            b = self._adjoint_solver._b
            with b.localForm() as b_loc:
                b_loc.set(0.0)
            form_i = ufl.algorithms.apply_derivatives.apply_derivatives(b_form)
            if not form_i.empty():
                compiled_soa_rhs = dolfinx.fem.form(
                    form_i,
                    jit_options=self._jit_options,
                    form_compiler_options=self._form_compiler_options,
                    entity_maps=self._entity_maps,
                )
                dolfinx.fem.petsc.assemble_vector(b, compiled_soa_rhs)
                b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)

                b.scale(-1)

            with b.localForm() as b_loc:
                b_loc.array[:] += hessian_inputs[0].array[:]
            b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
            self._adjoint_solver._b = b

        else:
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
            b = self._adjoint_solver._b
            local_arrays = [bi.array[: bi.index_map.size_local * bi.block_size] for bi in bs]
            dolfinx.la.petsc.assign(local_arrays, b)
            b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        # Compile SOA LHS
        dFdu_adj = dolfinx.fem.form(
            self._compute_adjoint(sum_form(dFdu_form)),
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )

        # Solve adjoint problem
        self._adjoint_solver._a = dFdu_adj
        self._adjoint_solver.solve()
        if isinstance(self._second_adjoint_solutions, list):
            for adj_sol, sol in zip(self._second_adjoint_solutions, self._adjoint_solver.u):
                adj_sol.x.array[:] = sol.x.array[:]
        else:
            self._second_adjoint_solutions.x.array[:] = self._adjoint_solver.u.x.array[:]

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
        kind: str | None = None,
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
        petsc_options_prefix: str = "dxa_nonlinear_block_",
    ) -> None: ...

    @typing.overload
    def __init__(
        self,
        F: typing.Sequence[ufl.Form],
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: typing.Sequence[dolfinx.fem.Function] | None = None,
        J: typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        P: typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        kind: str | typing.Sequence[typing.Sequence[str]] | None = None,
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
        petsc_options_prefix: str = "dxa_nonlinear_block_",
    ) -> None: ...

    def __init__(
        self,
        F: ufl.Form | typing.Sequence[ufl.Form],
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        u: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function] | None = None,
        J: ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        P: ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]] | None = None,
        kind: str | typing.Sequence[typing.Sequence[str]] | None = None,
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        ad_block_tag: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
        petsc_options_prefix: str = "dxa_nonlinear_block_",
    ) -> None:

        self._adjoint_petsc_options = adjoint_petsc_options
        self._tlm_petsc_options = tlm_petsc_options
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
        self._petsc_options = petsc_options if petsc_options is not None else {}
        self._petsc_options_prefix = petsc_options_prefix
        self._bcs = bcs if bcs is not None else []
        # Solver for recomputing the linear problem
        self._forward_solver = dolfinx.fem.petsc.NonlinearProblem(
            J=J,  # type: ignore[arg-type]
            F=self._rhs,  # type: ignore[arg-type]
            bcs=self._bcs,
            u=self._u,  # type: ignore[arg-type]
            P=self._preconditioner,  # type: ignore[arg-type]
            petsc_options=self._petsc_options,
            petsc_options_prefix=petsc_options_prefix,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            kind=kind,  # type: ignore[arg-type]
            entity_maps=self._entity_maps,
        )  # type: ignore[misc]

        self._kind = "nest" if self._forward_solver.A.getType() == "nest" else kind

        if isinstance(self._u, dolfinx.fem.Function):
            self._adjoint_solutions = self._u.copy()  # type: ignore[assignment]
            self._second_adjoint_solutions = self._u.copy()  # type: ignore[assignment]
            self._tlm_solutions = self._u.copy()  # type: ignore[assignment]
        else:
            assert isinstance(self._u, typing.Iterable)
            self._adjoint_solutions = [u.copy() for u in self._u]
            self._second_adjoint_solutions = [u.copy() for u in self._u]
            self._tlm_solutions = [u.copy() for u in self._u]

        if isinstance(F, ufl.Form):
            dFdu_adj = ufl.adjoint(ufl.derivative(F, u))
        else:
            raise NotImplementedError("Blocked systems not implemented yet.")
        self._adjoint_solver = LinearAdjointProblem(
            dFdu_adj,  # type: ignore[arg-type]
            self._rhs,  # type: ignore[arg-type]
            bcs=self._bcs,
            P=self._preconditioner,  # type: ignore[arg-type]
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            petsc_options=self._adjoint_petsc_options,
            petsc_options_prefix=self._petsc_options_prefix,
            kind=kind,  # type: ignore[arg-type]
            entity_maps=self._entity_maps,
        )  # type: ignore[misc]

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

        # As opposed to the linear problem, we need to update the coefficients in place,
        # as the nonlinear problem snes.setContext doesn't reflect in place updates on the solver.
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if isinstance(coeff, dolfinx.fem.Function):
                coeff.x.array[:] = block_variable.saved_output.x.array[:]
                coeff.x.scatter_forward()

        # Warm-start original unknown objects in place
        u_list = self._forward_solver._u if isinstance(self._forward_solver._u, list) else [self._forward_solver._u]
        for idx, out_bv in relevant_outputs:
            u_list[idx].x.array[:] = out_bv.saved_output.x.array[:]
            u_list[idx].x.scatter_forward()

        return None

    def recompute_component(
        self, inputs: typing.Iterable[Function], block_variable, idx: int, prepared: None
    ) -> Function:
        """Recompute the block with the prepared linear problem."""
        with pyadjoint.tape.stop_annotating():
            self._forward_solver.solve()
        if isinstance(self._forward_solver._u, list):
            output = self._forward_solver._u[idx]
        else:
            output = self._forward_solver._u
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
    ) -> tuple[typing.Union[list[ufl.Form], ufl.Form], dolfinx.fem.Form]:
        F_form = self._compute_residual()

        dFdu_compiled = dolfinx.fem.form(
            self._compute_residual_derivative(),  # type: ignore[arg-type]
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        return F_form, dFdu_compiled  # type: ignore[return-value]

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None) -> dolfinx.fem.Function:
        """Solve the TLM equation for the block variable.

        .. math::

            \frac{\\partial F}{\\partial u} \frac{\\partial u}{\\partial m} = \frac{\\partial F}{\\partial m}

        """
        F, dFdu = prepared

        V = self.get_outputs()[idx].output.function_space

        # FIXME: DirichletBC not block variable yet. Required later on. Currently all bcs should be homogenized
        bcs = []
        for bc in self._bcs:
            bcs.append(bc)

        dFdm = ufl.ZeroBaseForm((ufl.TestFunction(V),))
        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            c_rep = block_variable.saved_output
            if tlm_value is None:
                continue
            dFdm += ufl.derivative(-F, c_rep, tlm_value)

        if isinstance(dFdm, float):
            v = dFdu.arguments()[0]
            dFdm = ufl.ZeroBaseForm((v,))

        dFdm = ufl.algorithms.expand_derivatives(dFdm)
        dFdm_compiled = dolfinx.fem.form(
            dFdm,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        dudm = dolfinx.fem.Function(V, name="du_dm_tlm_linearblock")
        A_tlm = dolfinx.fem.petsc.assemble_matrix(dFdu, bcs=bcs)
        A_tlm.assemble()
        b_tlm = dolfinx.fem.create_vector(dolfinx.fem.extract_function_spaces(dFdm_compiled))  # type: ignore[arg-type]
        b_tlm.array[:] = 0.0
        dolfinx.fem.petsc.assemble_vector(b_tlm.petsc_vec, dFdm_compiled)

        if bcs is not None:
            # This system should never be "blocked"
            dolfinx.fem.petsc.apply_lifting(b_tlm.petsc_vec, [dFdu], bcs=[bcs], alpha=0)
            dolfinx.la.petsc._ghost_update(b_tlm.petsc_vec, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore [arg-type]
            for bc in bcs:
                bc.set(b_tlm.array, alpha=0)
        else:
            dolfinx.la.petsc._ghost_update(b_tlm, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore [arg-type]
        solve_linear_problem(A_tlm, dudm.x, b_tlm, petsc_options=self._tlm_petsc_options)
        return dudm

    def prepare_evaluate_adj(
        self,
        inputs: typing.Sequence[Function],
        adj_inputs: typing.Sequence[dolfinx.la.Vector],
        relevant_dependencies: typing.List[tuple[int, pyadjoint.block_variable.BlockVariable]],
    ) -> typing.Union[ufl.Form, typing.Iterable[ufl.Form]]:
        """Prepare the block for evaluating the adjoint."""

        # Compute (dF/du[v])* for the linear problem.
        F_form = self._compute_residual()
        dFdu = self._compute_residual_derivative()
        dFdu_adj = self._compute_adjoint(dFdu)
        # Extract dJ/du[v] from the adjoint inputs.
        assert len(adj_inputs) == 1
        adj_rhs = adj_inputs[0]
        dJdu = self._adjoint_solver._b
        with dJdu.localForm() as dJdu_loc, adj_rhs.petsc_vec.localForm() as adj_rhs_loc:
            dJdu_loc.array[:] = adj_rhs_loc.array[:]

        # Solve adjoint problem
        compiled_dFdu = dolfinx.fem.form(
            dFdu_adj,  # type: ignore[arg-type]
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        self._adjoint_solver._a = compiled_dFdu
        self._adjoint_solver.solve()
        if isinstance(self._adjoint_solutions, list):
            for adj_sol, sol in zip(self._adjoint_solutions, self._adjoint_solver.u):
                adj_sol.x.array[:] = sol.x.array[:]
        else:
            assert isinstance(self._adjoint_solutions, dolfinx.fem.Function)
            self._adjoint_solutions.x.array[:] = self._adjoint_solver.u.x.array[:]
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

        # Using the equation Form we derive dF/du, d^2F/du^2 * du/dm * direction.
        dFdu_form = self._compute_residual_derivative()
        assert len(outputs) == 1, "Hessian computation only implemented for single output blocks."
        assert len(tlm_output) == 1, "Hessian computation only implemented for single TLM output blocks."
        d2Fdu2 = ufl.algorithms.expand_derivatives(ufl.derivative(dFdu_form, outputs[0].saved_output, tlm_output[0]))

        # bdy = self._should_compute_boundary_adjoint(relevant_dependencies)
        assert len(hessian_inputs) == 1, "Hessian computation only implemented for single hessian input blocks."

        # Assemble right hand side of second order adjoint equation
        b_form = d2Fdu2 if d2Fdu2.empty() else ufl.action(ufl.adjoint(d2Fdu2), self._adjoint_solutions)
        dFdu_adj = ufl.action(ufl.adjoint(dFdu_form), self._adjoint_solutions)
        for bo in self.get_dependencies():
            c = bo.output
            c_rep = bo.saved_output
            tlm_input = bo.tlm_value
            if tlm_input is None:
                continue
            if isinstance(c, (dolfinx.mesh.Mesh, dolfinx.fem.DirichletBC)):
                raise NotImplementedError(f"Hessian computation for {type(c)} control not implemented yet.")
            else:
                b_form += ufl.derivative(dFdu_adj, c_rep, tlm_input)
        b = self._adjoint_solver._b
        with b.localForm() as b_loc:
            b_loc.set(0.0)
        if not ufl.algorithms.apply_derivatives.apply_derivatives(b_form).empty():
            compiled_soa_rhs = dolfinx.fem.form(
                b_form,
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )
            dolfinx.fem.petsc.assemble_vector(b, compiled_soa_rhs)
            b.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore [arg-type]
            b.scale(-1)
        with b.localForm() as b_loc, hessian_inputs[0].petsc_vec.localForm() as hess_loc:
            b_loc.array[:] += hess_loc.array[:]

        # Compile SOA LHS
        dFdu_adj = dolfinx.fem.form(
            ufl.adjoint(dFdu_form),
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )

        self._adjoint_solver._a = dFdu_adj
        self._adjoint_solver._u = self._second_adjoint_solutions
        self._adjoint_solver.solve()
        if isinstance(self._second_adjoint_solutions, list):
            for adj_sol, sol in zip(self._second_adjoint_solutions, self._adjoint_solver.u):
                adj_sol.x.array[:] = sol.x.array[:]
        else:
            self._second_adjoint_solutions.x.array[:] = self._adjoint_solver.u.x.array[:]
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

        F_form, adj_sol, adj_sol2 = prepared

        outputs = self.get_outputs()
        assert len(outputs) == 1, "Hessian computation only implemented for single output blocks."
        tlm_output = outputs[0].tlm_value

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

        dc = ufl.TestFunction(W)
        form_adj = ufl.action(F_form, adj_sol)
        form_adj2 = ufl.action(F_form, adj_sol2)
        if isinstance(c, dolfinx.mesh.Mesh):
            raise NotImplementedError("Hessian computation for Mesh control not implemented yet.")
            # dFdm_adj = ufl.derivative(form_adj, X, dc)
            # dFdm_adj2 = ufl.derivative(form_adj2, X, dc)
        else:
            # Assume Function
            dFdm_adj = ufl.derivative(form_adj, c_rep, dc)
            dFdm_adj2 = ufl.derivative(form_adj2, c_rep, dc)

        # TODO: Old comment claims this might break on split. Confirm if true or not.
        d2Fdudm = ufl.algorithms.expand_derivatives(ufl.derivative(dFdm_adj, outputs[0].saved_output, tlm_output))

        d2Fdm2 = 0
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

            if c2 == self._u and not self.linear:
                continue

            # TODO: If tlm_input is a Sum, this crashes in some instances?
            if isinstance(c2_rep, dolfinx.mesh.Mesh):
                X = ufl.SpatialCoordinate(c2_rep)
                d2Fdm2 += ufl.algorithms.expand_derivatives(ufl.derivative(dFdm_adj, X, tlm_input))
            else:
                d2Fdm2 += ufl.algorithms.expand_derivatives(ufl.derivative(dFdm_adj, c2_rep, tlm_input))

        hessian_form = ufl.algorithms.expand_derivatives(d2Fdm2 + dFdm_adj2 + d2Fdudm)

        compiled_hessian = dolfinx.fem.form(
            hessian_form,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        hessian_output = _create_vector(compiled_hessian, hessian_form.arguments()[0].ufl_function_space())
        hessian_output.array[:] = 0.0
        assemble_compiled_form(compiled_hessian, hessian_output)
        hessian_output.array[:] *= -1.0
        return hessian_output
