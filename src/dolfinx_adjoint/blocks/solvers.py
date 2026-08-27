from __future__ import annotations

import typing

from petsc4py import PETSc

import dolfinx.fem.petsc
import numpy as np
import pyadjoint
import ufl
from dolfinx.fem.function import Function as _Function

from ..compat import compute_form_adjoint
from ..petsc_utils import LinearAdjointProblem
from ..types import Function
from .assembly import _create_vector, _SpecialVector, assemble_compiled_form


def get_sorted_arguments(arguments: typing.Iterable[ufl.Argument], number: int) -> typing.Iterable[ufl.Argument]:
    """Extract all arguments of a given number, sorted by part."""
    return sorted(filter(lambda x: x.number() == number, arguments), key=lambda a: a.part())


def sum_form(form: typing.Sequence[typing.Sequence[ufl.Form] | ufl.Form] | ufl.Form) -> ufl.Form:
    """Sum a blocked form into a single form."""
    if isinstance(form, ufl.Form):
        return form
    if isinstance(form, typing.Iterable):
        return sum(sum_form(fi) for fi in form if fi is not None)


class LinearProblemBlock(pyadjoint.Block):
    """A linear problem that can be used with adjoint methods.

    This class extends the `dolfinx.fem.petsc.LinearProblem` to support adjoint methods.
    """

    _adjoint_solutions: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]
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

        # Collect all arguments in variational forms and replace them with similar once that is based on a mixed functionspace.
        if not isinstance(a, ufl.Form):
            # Get all arguments from the RHS and LHS forms
            trial_functions = len(a) * [None]
            test_functions = len(a) * [None]
            for i, ai in enumerate(a):
                for j, aij in enumerate(ai):
                    if aij is not None:
                        test_functions[i] = aij.arguments()[0]
                        trial_functions[j] = aij.arguments()[1]
            assert all(tf is not None for tf in trial_functions), "Not all trial functions were found."
            assert all(tf is not None for tf in test_functions), "Not all test functions were found."
            trial_parts = [tf.part() for tf in trial_functions]
            test_parts = [tf.part() for tf in test_functions]
            if any(tp is None for tp in trial_parts) or any(tp is None for tp in test_parts):
                replace_map = {}
                for i, tf in enumerate(trial_functions):
                    new_tf = ufl.TrialFunction(tf.ufl_function_space(), i)
                    replace_map[tf] = new_tf
                for i, tf in enumerate(test_functions):
                    new_tf = ufl.TestFunction(tf.ufl_function_space(), i)
                    replace_map[tf] = new_tf
                for i, ai in enumerate(a):
                    for j, aij in enumerate(ai):
                        if aij is not None:
                            a[i][j] = ufl.replace(aij, replace_map)
                for i, Li in enumerate(L):
                    if Li is not None:
                        L[i] = ufl.replace(Li, replace_map)

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
        if isinstance(self._u, dolfinx.fem.Function):
            for c in self._lhs.coefficients():  # type: ignore
                if c != self._u:  # Exclude the unknown
                    self.add_dependency(c, no_duplicates=True)
            for c in self._rhs.coefficients():  # type: ignore
                if c != self._u:  # Exclude the unknown
                    self.add_dependency(c, no_duplicates=True)
        elif isinstance(self._u, typing.Iterable):
            for Ai in self._lhs:  # type: ignore
                for Aij in Ai:
                    if Aij is not None:
                        for c in Aij.coefficients():
                            if c not in self._u:
                                self.add_dependency(c, no_duplicates=True)
            for i, part in enumerate(self._rhs):  # type: ignore
                for c in part.coefficients():
                    if c not in self._u:
                        self.add_dependency(c, no_duplicates=True)
        else:
            raise NotImplementedError("Blocked systems not implemented yet.")
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
            self._compute_adjoint(self._lhs),  # type: ignore[arg-type]
            self._rhs,  # type: ignore[arg-type]
            bcs=self._bcs,
            u=self._adjoint_solutions,  # type: ignore[arg-type]
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

    def _create_replace_map(self, form: ufl.Form | typing.Iterable[ufl.Form] | None) -> dict[Function, Function]:
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
        return replace_map

    def _replace_coefficients_in_form(
        self, form: ufl.Form | typing.Iterable[ufl.Form]
    ) -> ufl.Form | typing.Iterable[ufl.Form]:
        """Replace coefficients in the form with saved outputs.

        Args:
            form: The UFL form to replace coefficients in.
        """
        replace_map = self._create_replace_map(form)
        if isinstance(form, ufl.Form):
            return ufl.replace(form, replace_map)
        elif isinstance(form, typing.Iterable):
            replaced_forms = []
            for f in form:
                if f is None:
                    replaced_forms.append(None)
                elif isinstance(f, typing.Iterable):
                    replaced_forms.append(self._replace_coefficients_in_form(f))
                else:
                    replaced_forms.append(ufl.replace(f, replace_map))
            return replaced_forms

    def prepare_recompute_component(self, inputs, relevant_outputs):
        """Prepare for recomputing the block with different control inputs."""

        # Replace form coefficients with checkpointed values.
        # Loop through the dependencies of the lhs and rhs, check if they are in the respective form
        lhs = self._replace_coefficients_in_form(self._lhs)
        rhs = self._replace_coefficients_in_form(self._rhs)
        preconditioner = (
            self._replace_coefficients_in_form(self._preconditioner) if self._preconditioner is not None else None
        )
        compiled_lhs = dolfinx.fem.form(
            lhs,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        compiled_rhs = dolfinx.fem.form(
            rhs,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        compiled_preconditioner = (
            dolfinx.fem.form(
                preconditioner,
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )
            if preconditioner is not None
            else None
        )

        # Replace the compiled forms with those with new coefficients.
        self._forward_solver._a = compiled_lhs
        self._forward_solver._L = compiled_rhs
        self._forward_solver._P = compiled_preconditioner
        self._forward_solver.bcs = self._bcs
        self._forward_solver._u = self._u
        with pyadjoint.stop_annotating():
            solution = self._forward_solver.solve()
        return solution

    def recompute_component(
        self, inputs: typing.Iterable[Function], block_variable, idx: int, prepared: None
    ) -> typing.Union[dolfinx.fem.Function, typing.Iterable[dolfinx.fem.Function]]:
        """Recompute the block with the prepared linear problem."""
        if isinstance(prepared, dolfinx.fem.Function):
            assert idx == 0
            return prepared
        else:
            return prepared[idx]

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
    @typing.overload
    def _compute_adjoint(
        cls, form: typing.Sequence[typing.Sequence[ufl.Form]]
    ) -> typing.Sequence[typing.Sequence[ufl.Form]]: ...

    @classmethod
    @typing.overload
    def _compute_adjoint(cls, form: ufl.Form) -> ufl.Form: ...

    @classmethod
    def _compute_adjoint(
        cls, form: typing.Union[ufl.Form, typing.Sequence[typing.Sequence[ufl.Form]]]
    ) -> typing.Union[ufl.Form, typing.Sequence[typing.Sequence[ufl.Form]]]:
        """
        Compute adjoint of a bilinear form :math:`a(u, v)`, which could be written as a blocked system.
        """
        if isinstance(form, ufl.Form):
            return ufl.adjoint(form)
        else:
            assert isinstance(form, typing.Iterable)
            sum_form = sum([fij for fi in form for fij in fi if fij is not None])
            return ufl.extract_blocks(compute_form_adjoint(sum_form))

    def _compute_residual(self) -> typing.Union[ufl.Form, list[ufl.Form]]:
        """Convert the formulation :math:`a(u, v)=L(v)` into a residual :math:`F(u_b, v) = 0` where
        :math:`u_b` is the solution of the forward problem at the current time and all coefficients are updated.
        """
        # NOTE: Should probably be possible to compile this form once.
        replacement_functions = self.get_outputs()
        r_funcs = [r.saved_output for r in replacement_functions]
        if isinstance(self._u, Function):
            assert len(replacement_functions) == 1, (
                f"Expected a single output function, got {len(replacement_functions)}"
            )
            F_form = ufl.action(self._lhs, r_funcs[0]) - self._rhs
        else:
            # Blocked formulation (assuming no mixed function-space)
            assert len(self._u) == len(replacement_functions), (
                f"Expected {len(self._u)} output functions, got {len(replacement_functions)}"
            )
            summed_form = sum_form(self._lhs)
            F_form = ufl.action(summed_form, r_funcs) - sum_form(self._rhs)
        replacement_map = self._create_replace_map(F_form)
        F_form = ufl.replace(F_form, replacement_map)
        return F_form

    def _compute_residual_derivative(self) -> typing.Union[ufl.Form, list[list[ufl.Form]]]:
        """Compute the derivative of the residual with respect to the outputs."""

        F_form = self._compute_residual()
        outputs = [output.saved_output for output in self.get_outputs()]
        if len(outputs) == 1:
            assert isinstance(F_form, ufl.Form)
            dFdu = ufl.derivative(F_form, outputs[0], ufl.TrialFunction(outputs[0].function_space))
        else:
            # Replacement trial function needs to be in mixed space if initial form is created
            # with a mixed function space.
            # This means re-using the trialfunctions from the lhs
            trial_functions = get_sorted_arguments(sum_form(self._lhs).arguments(), 1)
            dFdu = ufl.derivative(F_form, outputs, trial_functions)
        return ufl.extract_blocks(dFdu)

    def prepare_evaluate_tlm(
        self, inputs, tlm_inputs, relevant_outputs
    ) -> tuple[typing.Union[list[ufl.Form], ufl.Form], dolfinx.fem.Form]:

        F_form = self._compute_residual()
        if self._tlm_solver is None:
            self._tlm_solver = self.construct_tlm_solver()
        # Even if the solver is cached, we need to replace the form, as the output from pyadjoint
        # is stored in a new function.
        self._tlm_solver._a = dolfinx.fem.form(
            self._compute_residual_derivative(),
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        #  Build RHS (dFdm) for the monolithic system
        if isinstance(self._u, list):
            test_funcs = get_sorted_arguments(sum_form(self._rhs).arguments(), 0)
            dFdm = sum([ufl.ZeroBaseForm((test,)) for test in test_funcs])
        else:
            test_funcs = [self._rhs.arguments()[0]]
            dFdm = ufl.ZeroBaseForm((test_funcs[0],))

        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            c_rep = block_variable.saved_output
            if tlm_value is None:
                continue

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
            return self._tlm_solutions

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
        dFdu_adj = compute_form_adjoint(sum_form(dFdu))
        # Extract dJ/du[v] from the adjoint inputs.
        if len(adj_inputs) == 1:
            adj_rhs = adj_inputs[0]
            dJdu = self._adjoint_solver._b
            with dJdu.localForm() as dJdu_loc, adj_rhs.petsc_vec.localForm() as adj_rhs_loc:
                dJdu_loc.array[:] = adj_rhs_loc.array[:]
        else:
            dFdu_adj = ufl.extract_blocks(dFdu_adj)
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

        # Solve adjoint problem
        compiled_dFdu = dolfinx.fem.form(
            dFdu_adj,  # type: ignore[arg-type]
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        self._adjoint_solver._a = compiled_dFdu
        self._adjoint_solver._u = self._adjoint_solutions  # type: ignore[assignment]
        self._adjoint_solver.solve()
        return F_form

    def evaluate_adj_component(
        self,
        inputs: typing.Iterable[Function],
        adj_inputs: typing.Iterable[dolfinx.la.Vector],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        prepared: typing.Union[ufl.Form, typing.Iterable[ufl.Form]],
    ) -> typing.Union[_SpecialVector, typing.Iterable[_SpecialVector]]:
        """Evaluate the adjoint component, i.e. :math:`\\frac{\\partial F}{\\partial m}`."""

        residual = prepared
        c = block_variable.output
        c_rep = block_variable.saved_output
        if isinstance(c, dolfinx.fem.Function):
            # Need some clever construction of the TrialFunction to get a part of the mixed space
            part = idx if isinstance(self._u, list) else None
            dc = ufl.TrialFunction(c_rep.function_space, part=part)
        else:
            raise NotImplementedError(f"Unsupported control {type(c)}")

        # Compute the sensitivity of the residual with respect to the parameter
        sum_res = sum_form(residual)
        dFdm = -ufl.derivative(sum_res, c_rep, dc)
        if dFdm.empty():
            # Generate a dummy form to safely extract the correct Vector wrapper type
            dFdm = dolfinx.fem.form(ufl.ZeroBaseForm((dc,)))

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
        if isinstance(dFdu_form, tuple):
            unknowns = [output.saved_output for output in self.get_outputs()]
            summed_form = sum_form(dFdu_form)
            d2Fdu2 = ufl.algorithms.expand_derivatives(ufl.derivative(summed_form, unknowns, tlm_output))
        else:
            d2Fdu2 = ufl.algorithms.expand_derivatives(
                ufl.derivative(dFdu_form, outputs[0].saved_output, tlm_output[0])
            )

        # bdy = self._should_compute_boundary_adjoint(relevant_dependencies)

        # Assemble right hand side of second order adjoint equation
        # Note this term should always be zero for linear problems, but we include it for completeness.
        if not d2Fdu2.empty():
            raise RuntimeError(f"This term {d2Fdu2:s} should be zero for linear problems.")
        b_form = d2Fdu2 if d2Fdu2.empty() else ufl.action(ufl.adjoint(d2Fdu2), self._adjoint_solutions)
        b_form = len(outputs) * [b_form]
        for bo in self.get_dependencies():
            c = bo.output
            c_rep = bo.saved_output
            tlm_input = bo.tlm_value
            if tlm_input is None:
                continue
            if isinstance(c, (dolfinx.mesh.Mesh, dolfinx.fem.DirichletBC)):
                raise NotImplementedError(f"Hessian computation for {type(c)} control not implemented yet.")
            else:
                dFdu_adj = self._compute_adjoint(dFdu_form)
                if isinstance(dFdu_form, tuple):
                    summed_form = sum_form(dFdu_adj)
                    dFdu_adj_applied = ufl.action(summed_form, self._adjoint_solutions)
                    b_form = ufl.extract_blocks(ufl.derivative(dFdu_adj_applied, c_rep, tlm_input))
                else:
                    dFdu_adj_applied = ufl.action(dFdu_adj, self._adjoint_solutions)
                    b_form[0] += ufl.derivative(dFdu_adj_applied, c_rep, tlm_input)

        if len(outputs) == 1:
            b = self._adjoint_solver._b
            with b.localForm() as b_loc:
                b_loc.set(0.0)
            form_i = ufl.algorithms.apply_derivatives.apply_derivatives(b_form[0])
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

        # Compile SOA LHS
        dFdu_adj = dolfinx.fem.form(
            self._compute_adjoint(dFdu_form),
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )

        # Solve adjoint problem
        self._adjoint_solver._a = dFdu_adj
        self._adjoint_solver._u = self._second_adjoint_solutions
        self._adjoint_solver.solve()
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

        dc = ufl.TestFunction(W)
        F_summed = sum_form(F_form)
        form_adj = ufl.action(F_summed, adj_sol)
        form_adj2 = ufl.action(F_summed, adj_sol2)
        if isinstance(c, dolfinx.mesh.Mesh):
            raise NotImplementedError("Hessian computation for Mesh control not implemented yet.")
            # dFdm_adj = ufl.derivative(form_adj, X, dc)
            # dFdm_adj2 = ufl.derivative(form_adj2, X, dc)
        else:
            # Assume Function
            dFdm_adj = ufl.derivative(form_adj, c_rep, dc)
            dFdm_adj2 = ufl.derivative(form_adj2, c_rep, dc)

        # TODO: Old comment claims this might break on split. Confirm if true or not.
        sa = [output.saved_output for output in outputs]
        d2Fdudm = ufl.algorithms.expand_derivatives(ufl.derivative(dFdm_adj, sa, tlm_output))

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

            # If problem is non-linear we need to skip the output variable as a control, as we can't differentiate with
            # respect to the initial guess
            # if c2 == self._u and not self.linear:
            #     continue

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

        test_functions = get_sorted_arguments(hessian_form.arguments(), 0)
        assert len(test_functions) == 1
        hessian_output = _create_vector(compiled_hessian, test_functions[0].ufl_function_space())

        hessian_output.array[:] = 0.0
        assemble_compiled_form(compiled_hessian, hessian_output)
        hessian_output.array[:] *= -1.0
        return hessian_output


class NonlinearProblemBlock(pyadjoint.Block):
    """A nonlinear problem that can be used with adjoint methods.

    This class extends the `dolfinx.fem.petsc.NonlinearProblem` to support adjoint methods.
    """

    _adjoint_solutions: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]
    _second_adjoint_solutions: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function]

    @typing.overload
    def __init__(
        self,
        F: ufl.Form,
        u: _Function,
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
        J: ufl.Form | None = None,
        P: ufl.Form | None = None,
        petsc_options: dict | None = None,
        form_compiler_options: dict | None = None,
        jit_options: dict | None = None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None = None,
        kind: str | None = None,
        ad_block_tag: str | None = None,
        adjoint_petsc_options: dict | None = None,
        tlm_petsc_options: dict | None = None,
        petsc_options_prefix: str = "dxa_nonlinear_problem_block_",
    ) -> None: ...

    @typing.overload
    def __init__(
        self,
        F: typing.Sequence[ufl.Form],
        u: typing.Sequence[_Function],
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
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
        petsc_options_prefix: str = "dxa_nonlinear_problem_block_",
    ) -> None: ...

    def __init__(
        self,
        F: ufl.Form | typing.Sequence[ufl.Form],
        u: _Function | typing.Sequence[_Function],
        *,
        bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None,
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
        petsc_options_prefix: str = "dxa_nonlinear_problem_block_",
    ) -> None:

        self._adjoint_petsc_options = adjoint_petsc_options
        self._tlm_petsc_options = tlm_petsc_options
        super().__init__(ad_block_tag=ad_block_tag)

        # Replace mixed space parts if necessary to construct blocked systems correctly
        if not isinstance(F, ufl.Form):
            test_functions = len(F) * [None]
            for i, Fi in enumerate(F):
                if Fi is not None:
                    test_functions[i] = Fi.arguments()[0]
            assert all(tf is not None for tf in test_functions), "Not all test functions were found."
            test_parts = [tf.part() for tf in test_functions]

            if any(tp is None for tp in test_parts):
                replace_map = {}
                for i, tf in enumerate(test_functions):
                    new_tf = ufl.TestFunction(tf.ufl_function_space(), i)
                    replace_map[tf] = new_tf
                for i, Fi in enumerate(F):
                    if Fi is not None:
                        F[i] = ufl.replace(Fi, replace_map)

                if J is not None:
                    for i, Ji in enumerate(J):
                        for j, Jij in enumerate(Ji):
                            if Jij is not None:
                                J[i][j] = ufl.replace(Jij, replace_map)
        self._F = F

        # Auto-derive Jacobian if not provided
        if J is None:
            if isinstance(F, list):
                J = ufl.extract_blocks(ufl.derivative(sum_form(F), u))
            else:
                J = ufl.derivative(F, u)

        self._J = J
        self._P = P

        if isinstance(u, dolfinx.fem.Function):
            self._u = pyadjoint.create_overloaded_object(u)
        else:
            self._u = [pyadjoint.create_overloaded_object(ui) for ui in u]

        # Add Dependencies: The coefficients of F and J, plus the unknown u (initial guess)
        u_list = [self._u] if isinstance(self._u, dolfinx.fem.Function) else self._u
        for ui in u_list:
            self.add_dependency(ui, no_duplicates=True)

        if isinstance(self._F, ufl.Form):
            for c in self._F.coefficients():
                if c not in u_list:
                    self.add_dependency(c, no_duplicates=True)
            for c in self._J.coefficients():
                if c not in u_list:
                    self.add_dependency(c, no_duplicates=True)
            if self._P is not None:
                for c in self._P.coefficients():
                    if c not in u_list:
                        self.add_dependency(c, no_duplicates=True)
        else:
            for Fi in self._F:
                if Fi is not None:
                    for c in Fi.coefficients():
                        if c not in u_list:
                            self.add_dependency(c, no_duplicates=True)
            for Ji in self._J:
                for Jij in Ji:
                    if Jij is not None:
                        for c in Jij.coefficients():
                            if c not in u_list:
                                self.add_dependency(c, no_duplicates=True)
            if self._P is not None:
                for Pi in self._P:
                    for Pij in Pi:
                        if Pij is not None:
                            for c in Pij.coefficients():
                                if c not in u_list:
                                    self.add_dependency(c, no_duplicates=True)

        self._jit_options = jit_options
        self._form_compiler_options = form_compiler_options
        self._entity_maps = entity_maps
        self._petsc_options = petsc_options if petsc_options is not None else {}
        self._petsc_options_prefix = petsc_options_prefix
        self._bcs = bcs if bcs is not None else []

        if self._bcs is not None:
            for bc in self._bcs:
                if hasattr(bc, "block_variable"):
                    self.add_dependency(bc, no_duplicates=True)

        # Build Nonlinear Forward Solver using native SNES context
        self._forward_solver = dolfinx.fem.petsc.NonlinearProblem(
            F=self._F,
            u=self._u,
            bcs=self._bcs,
            J=self._J,
            P=self._P,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            entity_maps=self._entity_maps,
            kind=kind,
            petsc_options_prefix=self._petsc_options_prefix,
            petsc_options=self._petsc_options,
        )

        self._kind = "nest" if self._forward_solver.A.getType() == "nest" else kind

        if isinstance(self._u, dolfinx.fem.Function):
            self._adjoint_solutions = self._u.copy()
            self._second_adjoint_solutions = self._u.copy()
            self._tlm_solutions = self._u.copy()
        else:
            self._adjoint_solutions = [u.copy() for u in self._u]
            self._second_adjoint_solutions = [u.copy() for u in self._u]
            self._tlm_solutions = [u.copy() for u in self._u]

        # Initialize Adjoint Solver matching the architecture of the Linear problem
        self._adjoint_solver = LinearAdjointProblem(
            self._compute_adjoint(self._J),
            self._F,  # Dummy RHS for initialization formatting
            bcs=self._bcs,
            u=self._adjoint_solutions,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            petsc_options=self._adjoint_petsc_options,
            petsc_options_prefix=self._petsc_options_prefix,
            kind=self._kind,
            entity_maps=self._entity_maps,
        )
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
            dFdu_form,
            self._F,  # Dummy RHS
            bcs=self._bcs,
            u=self._tlm_solutions,
            P=self._P,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            petsc_options=self._tlm_petsc_options,
            petsc_options_prefix=self._petsc_options_prefix,
            kind=self._kind,
            entity_maps=self._entity_maps,
        )
        return tlm_solver

    def _create_replace_map(self, form: ufl.Form | typing.Iterable[ufl.Form] | None) -> dict[Function, Function]:
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
        return replace_map

    def _replace_coefficients_in_form(
        self, form: ufl.Form | typing.Iterable[ufl.Form]
    ) -> ufl.Form | typing.Iterable[ufl.Form]:
        replace_map = self._create_replace_map(form)
        if isinstance(form, ufl.Form):
            return ufl.replace(form, replace_map)
        elif isinstance(form, typing.Iterable):
            replaced_forms = []
            for f in form:
                if f is None:
                    replaced_forms.append(None)
                elif isinstance(f, typing.Iterable):
                    replaced_forms.append(self._replace_coefficients_in_form(f))
                else:
                    replaced_forms.append(ufl.replace(f, replace_map))
            return replaced_forms

    def prepare_recompute_component(self, inputs, relevant_outputs):
        """Prepare for recomputing the block with different control inputs."""

        # 1. Update coefficients IN-PLACE.
        # This dynamically updates the evaluation of F and J without triggering FFCx recompilation
        for block_variable in self.get_dependencies():
            coeff = block_variable.output
            if isinstance(coeff, dolfinx.fem.Function):
                coeff.x.array[:] = block_variable.saved_output.x.array[:]
                coeff.x.scatter_forward()

        # 2. Warm-start the initial guess identically to the original forward pass
        u_list = self._u if isinstance(self._u, list) else [self._u]
        for idx, out_bv in relevant_outputs:
            u_list[idx].x.array[:] = out_bv.saved_output.x.array[:]
            u_list[idx].x.scatter_forward()

        # 3. Solve the nonlinear system seamlessly utilizing the updated array memory
        with pyadjoint.stop_annotating():
            self._forward_solver.solve()

        return self._u

    def recompute_component(
        self, inputs: typing.Iterable[Function], block_variable, idx: int, prepared: None
    ) -> typing.Union[dolfinx.fem.Function, typing.Iterable[dolfinx.fem.Function]]:
        if isinstance(prepared, dolfinx.fem.Function):
            assert idx == 0
            return prepared
        else:
            return prepared[idx]

    @classmethod
    def _compute_adjoint(
        cls, form: typing.Union[ufl.Form, typing.Sequence[typing.Sequence[ufl.Form]]]
    ) -> typing.Union[ufl.Form, typing.Sequence[typing.Sequence[ufl.Form]]]:
        if isinstance(form, ufl.Form):
            return ufl.adjoint(form)
        else:
            assert isinstance(form, typing.Iterable)
            summed_form = sum([fij for fi in form for fij in fi if fij is not None])
            return ufl.extract_blocks(compute_form_adjoint(summed_form))

    def _compute_residual(self) -> typing.Union[ufl.Form, list[ufl.Form]]:
        """Map coefficients to current saved checkpoint memory."""
        replacement_map = self._create_replace_map(self._F)
        return ufl.replace(self._F, replacement_map)

    def _compute_residual_derivative(self) -> typing.Union[ufl.Form, list[list[ufl.Form]]]:
        """J is the residual derivative. Map coefficients to saved memory."""
        replacement_map = self._create_replace_map(self._J)
        return ufl.replace(self._J, replacement_map)

    def prepare_evaluate_tlm(
        self, inputs, tlm_inputs, relevant_outputs
    ) -> tuple[typing.Union[list[ufl.Form], ufl.Form], dolfinx.fem.Form]:

        F_form = self._compute_residual()
        if self._tlm_solver is None:
            self._tlm_solver = self.construct_tlm_solver()

        self._tlm_solver._a = dolfinx.fem.form(
            self._compute_residual_derivative(),
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )

        if isinstance(self._u, list):
            test_funcs = get_sorted_arguments(sum_form(self._F).arguments(), 0)
            dFdm = sum([ufl.ZeroBaseForm((test,)) for test in test_funcs])
        else:
            test_funcs = [self._F.arguments()[0]]
            dFdm = ufl.ZeroBaseForm((test_funcs[0],))

        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            c_rep = block_variable.saved_output
            if tlm_value is None:
                continue

            dFdm += ufl.derivative(-F_form, c_rep, tlm_value)

        dFdm = ufl.algorithms.expand_derivatives(dFdm)
        if isinstance(self._u, list):
            blocks = ufl.extract_blocks(dFdm)
            if len(blocks) != len(self._u):
                _dFdm = [ufl.ZeroBaseForm((test,)) for test in test_funcs]
                for block in blocks:
                    args = block.arguments()
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

        b_petsc = self._tlm_solver._b
        with b_petsc.localForm() as b_loc:
            b_loc.set(0.0)

        dolfinx.fem.petsc.assemble_vector(b_petsc, dFdm_compiled)
        dolfinx.la.petsc._ghost_update(b_petsc, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)

        if self._bcs:
            try:
                for bc in self._bcs:
                    bc.set(b_petsc.array_w, alpha=0.0)
            except RuntimeError:
                from dolfinx.fem.bcs import bcs_by_block

                V_ext = dolfinx.fem.extract_function_spaces(dFdm_compiled)
                bcs_lift = bcs_by_block(V_ext, self._bcs)
                dolfinx.fem.petsc.set_bc(b_petsc, bcs_lift, alpha=0.0)

        self._tlm_solver.solve()
        return self._tlm_solutions

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None) -> dolfinx.fem.Function:
        if isinstance(self._tlm_solutions, list):
            return self._tlm_solutions[idx]
        else:
            return self._tlm_solutions

    def prepare_evaluate_adj(
        self,
        inputs: typing.Sequence[Function],
        adj_inputs: typing.Sequence[dolfinx.la.Vector],
        relevant_dependencies: typing.List[tuple[int, pyadjoint.block_variable.BlockVariable]],
    ) -> typing.Union[ufl.Form, typing.Iterable[ufl.Form]]:

        F_form = self._compute_residual()
        dFdu = self._compute_residual_derivative()
        dFdu_adj = compute_form_adjoint(sum_form(dFdu))

        if len(adj_inputs) == 1:
            adj_rhs = adj_inputs[0]
            dJdu = self._adjoint_solver._b
            with dJdu.localForm() as dJdu_loc, adj_rhs.petsc_vec.localForm() as adj_rhs_loc:
                dJdu_loc.array[:] = adj_rhs_loc.array[:]
        else:
            dFdu_adj = ufl.extract_blocks(dFdu_adj)
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

        compiled_dFdu = dolfinx.fem.form(
            dFdu_adj,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        self._adjoint_solver._a = compiled_dFdu
        self._adjoint_solver._u = self._adjoint_solutions
        self._adjoint_solver.solve()
        return F_form

    def evaluate_adj_component(
        self,
        inputs: typing.Iterable[Function],
        adj_inputs: typing.Iterable[dolfinx.la.Vector],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        prepared: typing.Union[ufl.Form, typing.Iterable[ufl.Form]],
    ) -> typing.Union[_SpecialVector, typing.Iterable[_SpecialVector]]:

        residual = prepared
        c = block_variable.output
        c_rep = block_variable.saved_output

        if not isinstance(c, dolfinx.fem.Function):
            raise NotImplementedError(f"Unsupported control {type(c)}")

        dc = ufl.TrialFunction(c_rep.function_space)

        sum_res = sum_form(residual)
        dFdm = ufl.algorithms.apply_derivatives.apply_derivatives((ufl.derivative(sum_res, c_rep, dc)))

        if dFdm.empty():
            dFdm = dolfinx.fem.form(ufl.ZeroBaseForm((dc,)))

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
        outputs = self.get_outputs()
        tlm_output = [output.tlm_value for output in outputs if output is not None]
        if (hessian_inputs is None) or (len(tlm_output) == 0):
            return

        dFdu_form = self._compute_residual_derivative()
        dFdu_adj = self._compute_adjoint(dFdu_form)

        # 1. Base adjoint action: lambda^T * dF/du
        # This yields a linear form whose TestFunction is exactly the original TrialFunction.
        if isinstance(self._u, list):
            dFdu_adj_applied = ufl.action(sum_form(dFdu_adj), self._adjoint_solutions)
            # CRITICAL FIX: Extract TrialFunctions (index 1) to use as the new TestFunctions
            test_funcs = get_sorted_arguments(sum_form(dFdu_form).arguments(), 1)
            b_form = [ufl.ZeroBaseForm((test,)) for test in test_funcs]
        else:
            dFdu_adj_applied = ufl.action(dFdu_adj, self._adjoint_solutions)
            test_funcs = [dFdu_form.arguments()[1]]
            b_form = ufl.ZeroBaseForm((test_funcs[0],))

        # 2. Add State Hessian Term: d/du (lambda^T * dF/du) * delta_u
        # Differentiating the applied linear form is mathematically exact and avoids
        # brittle UFL tensor transpositions.
        unknowns = [output.saved_output for output in outputs]
        if isinstance(self._u, list):
            d2Fdu2_term = ufl.algorithms.expand_derivatives(ufl.derivative(dFdu_adj_applied, unknowns, tlm_output))
        else:
            d2Fdu2_term = ufl.algorithms.expand_derivatives(
                ufl.derivative(dFdu_adj_applied, unknowns[0], tlm_output[0])
            )

        if not (isinstance(d2Fdu2_term, (int, float)) or (hasattr(d2Fdu2_term, "empty") and d2Fdu2_term.empty())):
            if isinstance(self._u, list):
                for block in ufl.extract_blocks(d2Fdu2_term):
                    if block is not None and not (hasattr(block, "empty") and block.empty()):
                        b_form[block.arguments()[0].part()] += block
            else:
                b_form += d2Fdu2_term

        # 3. Add Control Hessian Term: d/dm (lambda^T * dF/du) * delta_m
        for bo in self.get_dependencies():
            c = bo.output
            c_rep = bo.saved_output
            tlm_input = bo.tlm_value

            # Skip non-differentiable dependencies
            if tlm_input is None or isinstance(bo.output, dolfinx.fem.DirichletBC):
                continue

            term = ufl.algorithms.expand_derivatives(ufl.derivative(dFdu_adj_applied, c_rep, tlm_input))
            if isinstance(term, (int, float)) or (hasattr(term, "empty") and term.empty()):
                continue

            if isinstance(self._u, list):
                for block in ufl.extract_blocks(term):
                    if block is not None and not (hasattr(block, "empty") and block.empty()):
                        b_form[block.arguments()[0].part()] += block
            else:
                b_form += term

        # 4. Clean up forms to prevent FEniCSx compilation crashes
        if isinstance(self._u, list):
            b_form = [ufl.algorithms.expand_derivatives(bf) for bf in b_form]
            for i, bf in enumerate(b_form):
                if bf == 0 or bf.empty():
                    b_form[i] = ufl.ZeroBaseForm((test_funcs[i],))
        else:
            b_form = ufl.algorithms.expand_derivatives(b_form)
            if b_form == 0 or b_form.empty():
                b_form = ufl.ZeroBaseForm((test_funcs[0],))

        compiled_soa_rhs = dolfinx.fem.form(
            b_form,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )

        b_petsc = self._adjoint_solver._b
        with b_petsc.localForm() as b_loc:
            b_loc.set(0.0)

        # 5. Native assembly into blocks
        dolfinx.fem.petsc.assemble_vector(b_petsc, compiled_soa_rhs)
        try:
            dolfinx.la.petsc._ghost_update(b_petsc, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)
        except AttributeError:
            b_petsc.ghostUpdate(PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)

        b_petsc.scale(-1.0)

        # 6. Accumulate incoming Hessian DOFs
        import numpy as np

        tmp_b = b_petsc.duplicate()
        arrs = []
        for i, hess_in in enumerate(hessian_inputs):
            local_size = (
                outputs[i].saved_output.function_space.dofmap.index_map.size_local
                * outputs[i].saved_output.function_space.dofmap.index_map_bs
            )
            if hess_in is not None:
                arrs.append(hess_in.array[:local_size])
            else:
                arrs.append(np.zeros(local_size, dtype=dolfinx.default_scalar_type))

        dolfinx.la.petsc.assign(arrs, tmp_b)

        try:
            dolfinx.la.petsc._ghost_update(tmp_b, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)
        except AttributeError:
            tmp_b.ghostUpdate(PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)

        b_petsc.axpy(1.0, tmp_b)
        tmp_b.destroy()

        if self._bcs:
            try:
                for bc in self._bcs:
                    bc.set(b_petsc.array_w, alpha=0.0)
            except RuntimeError:
                from dolfinx.fem.bcs import bcs_by_block

                V_ext = dolfinx.fem.extract_function_spaces(compiled_soa_rhs)
                bcs_lift = bcs_by_block(V_ext, self._bcs)
                dolfinx.fem.petsc.set_bc(b_petsc, bcs_lift, alpha=0.0)

        compiled_dFdu_adj = dolfinx.fem.form(
            self._compute_adjoint(dFdu_form),
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )

        self._adjoint_solver._a = compiled_dFdu_adj
        self._adjoint_solver._u = self._second_adjoint_solutions
        self._adjoint_solver.solve()

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
        tlm_output = [output.tlm_value for output in outputs]
        c_rep = block_variable.saved_output

        if not isinstance(c, dolfinx.fem.Function):
            raise NotImplementedError(f"Hessian computation for {type(c)} not implemented yet.")

        dc = ufl.TestFunction(c.function_space)
        F_summed = sum_form(F_form)
        form_adj = ufl.action(F_summed, adj_sol)
        form_adj2 = ufl.action(F_summed, adj_sol2)

        dFdm_adj = ufl.derivative(form_adj, c_rep, dc)
        dFdm_adj2 = ufl.derivative(form_adj2, c_rep, dc)

        sa = [output.saved_output for output in outputs]
        d2Fdudm = ufl.algorithms.expand_derivatives(ufl.derivative(dFdm_adj, sa, tlm_output))

        d2Fdm2 = 0
        for _, bv in relevant_dependencies:
            c2_rep = bv.saved_output
            tlm_input = bv.tlm_value
            if tlm_input is None or isinstance(bv.output, dolfinx.fem.DirichletBC):
                continue

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

        test_functions = get_sorted_arguments(hessian_form.arguments(), 0)
        hessian_output = _create_vector(compiled_hessian, test_functions[0].ufl_function_space())
        hessian_output.array[:] = 0.0
        assemble_compiled_form(compiled_hessian, hessian_output)
        hessian_output.array[:] *= -1.0

        return hessian_output
