import typing

from petsc4py import PETSc

import dolfinx.fem.petsc
import pyadjoint
import ufl

from dolfinx_adjoint.petsc_utils import LinearAdjointProblem, solve_linear_problem
from dolfinx_adjoint.types import Function

from .assembly import _create_vector, _SpecialVector, assemble_compiled_form


class LinearProblemBlock(pyadjoint.Block):
    """A linear problem that can be used with adjoint methods.

    This class extends the `dolfinx.fem.petsc.LinearProblem` to support adjoint methods.
    """

    _adjoint_solutions: typing.Union[Function, typing.Sequence[Function]]
    _second_adjoint_solutions: typing.Union[Function, typing.Sequence[Function]]

    def __init__(
        self,
        a: ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]],
        L: ufl.Form | typing.Sequence[ufl.Form],
        bcs: typing.Sequence[dolfinx.fem.DirichletBC],
        u: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function],
        P: ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]] | None,
        kind: str | typing.Sequence[typing.Sequence[str]] | None,
        petsc_options: dict | None,
        form_compiler_options: dict | None,
        jit_options: dict | None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None,
        ad_block_tag: str | None,
        adjoint_petsc_options: dict | None,
        tlm_petsc_options: dict | None,
        petsc_options_prefix: str = "dxa_linear_problem_block_",
    ) -> None:
        self._adjoint_petsc_options = adjoint_petsc_options
        self._tlm_petsc_options = tlm_petsc_options
        super().__init__(ad_block_tag=ad_block_tag)
        self._lhs = a
        self._rhs = L
        self._preconditioner = P

        # Create overloaded functions
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
        try:
            for c in self._lhs.coefficients():  # type: ignore
                self.add_dependency(c, no_duplicates=True)
            for c in self._rhs.coefficients():  # type: ignore
                self.add_dependency(c, no_duplicates=True)
        except AttributeError:
            raise NotImplementedError("Blocked systems not implemented yet.")
        self._compiled_lhs = dolfinx.fem.form(
            self._lhs,  # type: ignore
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
        # Solver for recomputing the linear problem
        self._forward_solver = dolfinx.fem.petsc.LinearProblem(
            a=self._lhs,
            L=self._rhs,
            bcs=self._bcs,
            u=self._u,
            P=self._preconditioner,
            petsc_options=self._petsc_options,
            petsc_options_prefix=petsc_options_prefix,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            kind=kind,
            entity_maps=self._entity_maps,
        )

        self._kind = "nest" if self._forward_solver.A.getType() == "nest" else kind

        if isinstance(self._u, dolfinx.fem.Function):
            self._adjoint_solutions = self._u.copy()  # type: ignore[assignment]
            self._second_adjoint_solutions = self._u.copy()  # type: ignore[assignment]
        else:
            assert isinstance(self._u, typing.Iterable)
            self._adjoint_solutions = [u.copy() for u in self._u]
            self._second_adjoint_solutions = [u.copy() for u in self._u]
        self._adjoint_solver = LinearAdjointProblem(
            self._compute_adjoint(self._lhs),
            self._rhs,
            bcs=self._bcs,
            u=self._adjoint_solutions,
            P=self._preconditioner,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            petsc_options=self._adjoint_petsc_options,
            petsc_options_prefix=self._petsc_options_prefix,
            kind=kind,
            entity_maps=self._entity_maps,
        )

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

        # Create initial guess for the KSP solver
        # Form independnet compilation would make it possible to use the same KSP for all re-evaluations.
        if isinstance(self._u, Function):
            initial_guess = dolfinx.fem.Function(self._u.function_space, name=self._u.name + "_initial_guess")
        else:
            initial_guess = [dolfinx.fem.Function(u.function_space, name=u.name + "_initial_guess") for u in self._u]

        # Replace values in the DirichletBC if it is dependent on a control
        # NOTE: Currently assume that BCS are control independent.
        bcs = self._bcs
        # for block_variable in self.get_dependencies():
        #     c = block_variable.output
        #     c_rep = block_variable.saved_output

        #     if isinstance(c, dolfinx.fem.DirichletBC):
        #         bcs.append(c_rep)

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
        self._forward_solver.bcs = bcs
        self._forward_solver._u = initial_guess

    def recompute_component(
        self, inputs: typing.Iterable[Function], block_variable, idx: int, prepared: None
    ) -> typing.Union[dolfinx.fem.Function, typing.Iterable[dolfinx.fem.Function]]:
        """Recompute the block with the prepared linear problem."""
        solution = self._forward_solver.solve()
        return solution

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
        F_form: typing.Union[ufl.Form, list[ufl.Form]] = []
        if isinstance(self._u, Function):
            assert len(replacement_functions) == 1, (
                f"Expected a single output function, got {len(replacement_functions)}"
            )
            F_form = ufl.action(self._lhs, replacement_functions[0].saved_output) - self._rhs
        else:
            # Blocked formulation (assuming no mixed function-space)
            assert len(self._u) == len(replacement_functions), (
                f"Expected {len(self._u)} output functions, got {len(replacement_functions)}"
            )
            for i in range(len(self._u)):
                assert isinstance(F_form, list)
                res_i = ufl.ZeroBaseForm((self._u[i],))
                for j in range(len(self._u)):
                    res_i += ufl.action(self._lhs[i][j], replacement_functions[j].saved_output)  # type: ignore[index]
                res_i -= self._rhs[i]  # type: ignore[index]
                F_form.append(res_i)
        # NOTE: Will fail for blocked systems atm
        assert isinstance(F_form, ufl.Form)
        replacement_map = self._create_replace_map(F_form)
        if isinstance(self._u, Function):
            F_form = ufl.replace(F_form, replacement_map)
        else:
            assert isinstance(F_form, list)
            for j in range(len(F_form)):
                F_form[j] = ufl.replace(F_form[j], replacement_map)
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
        return F_form, dFdu_compiled

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None) -> dolfinx.fem.Function:
        """Solve the TLM equation for the block variable.

        .. math::

            \frac{\\partial F}{\\partial u} \frac{\\partial u}{\\partial m} = \frac{\\partial F}{\\partial m}

        """
        # FIXME: Think about blocks later
        F, dFdu = prepared

        V = self.get_outputs()[idx].output.function_space

        # FIXME: DirichletBC not block variable yet. Required later on. Currently all bcs should be homogenized
        bcs = []
        for bc in self._bcs:
            bcs.append(bc)
        dFdm = ufl.ZeroBaseForm((ufl.TestFunction(V),))
        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            # c = block_variable.output
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
            dolfinx.la.petsc._ghost_update(b_tlm.petsc_vec, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
            for bc in bcs:
                bc.set(b_tlm.array, alpha=0)
        else:
            dolfinx.la.petsc._ghost_update(b_tlm, PETSc.InsertMode.ADD, PETSc.ScatterMode.REVERSE)  # type: ignore[arg-type]
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
        dJdu = dolfinx.la.vector(adj_rhs.index_map, adj_rhs.block_size)
        dJdu.array[:] = adj_rhs.array[:].copy()

        # Solve adjoint problem
        compiled_dFdu = dolfinx.fem.form(
            dFdu_adj,  # type: ignore[arg-type]
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        self._adjoint_solver._a = compiled_dFdu
        self._adjoint_solver._b = dJdu.petsc_vec
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
        """Evaluate the adjoint component, i.e. :math:`\frac{\\partial Au - b}{\\partial c}`."""

        residual = prepared

        c = block_variable.output

        c_rep = block_variable.saved_output
        if isinstance(c, dolfinx.fem.Function):
            dc = ufl.TrialFunction(c.function_space)
        else:
            raise NotImplementedError(f"Unsupported control {type(c)}")

        dFdm = -ufl.derivative(residual, c_rep, dc)
        dFdm_adj = ufl.adjoint(dFdm)
        sensitivity = ufl.action(dFdm_adj, self._adjoint_solutions)
        compiled_sensitivity = dolfinx.fem.form(
            sensitivity,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        vec = _create_vector(compiled_sensitivity, sensitivity.arguments()[0].ufl_function_space())
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
        for bo in self.get_dependencies():
            c = bo.output
            c_rep = bo.saved_output
            tlm_input = bo.tlm_value
            if tlm_input is None:
                continue
            if isinstance(c, (dolfinx.mesh.Mesh, dolfinx.fem.DirichletBC)):
                raise NotImplementedError(f"Hessian computation for {type(c)} control not implemented yet.")
            else:
                dFdu_adj = ufl.action(ufl.adjoint(dFdu_form), self._adjoint_solutions)
                b_form += ufl.derivative(dFdu_adj, c_rep, tlm_input)

        b = dolfinx.la.vector(hessian_inputs[0].index_map, hessian_inputs[0].block_size)
        b.array[:] = 0.0
        if not ufl.algorithms.apply_derivatives.apply_derivatives(b_form).empty():
            compiled_soa_rhs = dolfinx.fem.form(
                b_form,
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )
            dolfinx.fem.petsc.assemble_vector(b.petsc_vec, compiled_soa_rhs)
            b.scatter_reverse(dolfinx.la.InsertMode.add)
            b.array[:] *= -1

        b.array[:] += hessian_inputs[0].array

        # Compile SOA LHS
        dFdu_adj = dolfinx.fem.form(
            ufl.adjoint(dFdu_form),
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )

        # Solve adjoint problem
        self._adjoint_solver._a = dFdu_adj
        self._adjoint_solver._b = b.petsc_vec
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
        hessian_output = _create_vector(compiled_hessian, hessian_form.arguments()[0].ufl_function_space())
        assemble_compiled_form(compiled_hessian, hessian_output)
        hessian_output.array[:] *= -1.0
        return hessian_output


class NonlinearProblemBlock(pyadjoint.Block):
    """A linear problem that can be used with adjoint methods.

    This class extends the `dolfinx.fem.petsc.LinearProblem` to support adjoint methods.
    """

    _adjoint_solutions: typing.Union[Function, typing.Sequence[Function]]
    _second_adjoint_solutions: typing.Union[Function, typing.Sequence[Function]]
    _rhs: ufl.Form | typing.Sequence[ufl.Form]

    def __init__(
        self,
        F: ufl.Form | typing.Sequence[ufl.Form],
        bcs: typing.Sequence[dolfinx.fem.DirichletBC],
        u: dolfinx.fem.Function | typing.Sequence[dolfinx.fem.Function],
        J: ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]],
        P: ufl.Form | typing.Sequence[typing.Sequence[ufl.Form]],
        kind: str | typing.Sequence[typing.Sequence[str]] | None,
        petsc_options: dict | None,
        form_compiler_options: dict | None,
        jit_options: dict | None,
        entity_maps: typing.Sequence[dolfinx.mesh.EntityMap] | None,
        ad_block_tag: str | None,
        adjoint_petsc_options: dict | None,
        tlm_petsc_options: dict | None,
        petsc_options_prefix: str = "dxa_nonlinear_block_",
    ) -> None:
        self._adjoint_petsc_options = adjoint_petsc_options
        self._tlm_petsc_options = tlm_petsc_options
        super().__init__(ad_block_tag=ad_block_tag)
        self._lhs = J
        self._preconditioner = P

        # Create overloaded functions
        if isinstance(u, dolfinx.fem.Function):
            self._u = pyadjoint.create_overloaded_object(u)
            self._rhs = F
        else:
            self._u = [pyadjoint.create_overloaded_object(ui) for ui in u]
            assert isinstance(F, typing.Iterable)
            replace_dict = {ui: _ui for ui, _ui in zip(u, self._u)}
            self._rhs = [ufl.replace(Fi, replace_dict) for Fi in F]

        # NOTE: Add mesh and constants as dependencies later on
        try:
            for c in self._lhs.coefficients():  # type: ignore
                self.add_dependency(c, no_duplicates=True)
            for c in self._rhs.coefficients():  # type: ignore
                self.add_dependency(c, no_duplicates=True)
        except AttributeError:
            raise NotImplementedError("Blocked systems not implemented yet.")
        self._compiled_lhs = dolfinx.fem.form(
            self._lhs,  # type: ignore
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
        # Solver for recomputing the linear problem
        self._forward_solver = dolfinx.fem.petsc.NonlinearProblem(
            J=self._lhs,
            F=self._rhs,
            bcs=self._bcs,
            u=self._u,
            P=self._preconditioner,
            petsc_options=self._petsc_options,
            petsc_options_prefix=petsc_options_prefix,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            kind=kind,
            entity_maps=self._entity_maps,
        )

        self._kind = "nest" if self._forward_solver.A.getType() == "nest" else kind

        if isinstance(self._u, dolfinx.fem.Function):
            self._adjoint_solutions = self._u.copy()  # type: ignore[assignment]
            self._second_adjoint_solutions = self._u.copy()  # type: ignore[assignment]
        else:
            assert isinstance(self._u, typing.Iterable)
            self._adjoint_solutions = [u.copy() for u in self._u]
            self._second_adjoint_solutions = [u.copy() for u in self._u]

        if isinstance(F, ufl.Form):
            dFdu_adj = ufl.adjoint(ufl.derivative(F, u))
        else:
            raise NotImplementedError("Blocked systems not implemented yet.")
        self._adjoint_solver = LinearAdjointProblem(
            dFdu_adj,
            self._rhs,
            bcs=self._bcs,
            u=self._adjoint_solutions,
            P=self._preconditioner,
            form_compiler_options=self._form_compiler_options,
            jit_options=self._jit_options,
            petsc_options=self._adjoint_petsc_options,
            petsc_options_prefix=self._petsc_options_prefix,
            kind=kind,
            entity_maps=self._entity_maps,
        )

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

        # Create initial guess for the KSP solver
        # Form independnet compilation would make it possible to use the same KSP for all re-evaluations.
        if isinstance(self._u, Function):
            initial_guess = dolfinx.fem.Function(self._u.function_space, name=self._u.name + "_initial_guess")
        else:
            initial_guess = [dolfinx.fem.Function(u.function_space, name=u.name + "_initial_guess") for u in self._u]

        # Replace values in the DirichletBC if it is dependent on a control
        # NOTE: Currently assume that BCS are control independent.
        bcs = self._bcs
        # for block_variable in self.get_dependencies():
        #     c = block_variable.output
        #     c_rep = block_variable.saved_output

        #     if isinstance(c, dolfinx.fem.DirichletBC):
        #         bcs.append(c_rep)

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
        self._forward_solver.bcs = bcs
        self._forward_solver._u = initial_guess

    def recompute_component(
        self, inputs: typing.Iterable[Function], block_variable, idx: int, prepared: None
    ) -> typing.Union[dolfinx.fem.Function, typing.Iterable[dolfinx.fem.Function]]:
        """Recompute the block with the prepared linear problem."""
        solution = self._forward_solver.solve()
        return solution

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
    def _compute_adjoint(
        cls, form: typing.Union[ufl.Form, typing.Iterable[typing.Iterable[ufl.Form]]]
    ) -> typing.Union[ufl.Form, typing.Sequence[typing.Iterable[ufl.Form]]]:
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
        F_form: typing.Union[ufl.Form, list[ufl.Form]] = []
        assert isinstance(self._rhs, ufl.Form)
        replacement_map = self._create_replace_map(self._rhs)
        u_list = self._u if isinstance(self._u, list) else [self._u]
        for u, block in zip(u_list, replacement_functions):
            replacement_map[u] = block.saved_output

        if isinstance(self._u, Function):
            F_form = ufl.replace(self._rhs, replacement_map)
        else:
            assert isinstance(F_form, list)
            assert isinstance(self._rhs, typing.Sequence)
            assert len(F_form) == len(self._rhs)
            for j, rhs_j in enumerate(self._rhs):
                F_form[j] = ufl.replace(rhs_j, replacement_map)
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
        return F_form, dFdu_compiled

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None) -> dolfinx.fem.Function:
        """Solve the TLM equation for the block variable.

        .. math::

            \frac{\\partial F}{\\partial u} \frac{\\partial u}{\\partial m} = \frac{\\partial F}{\\partial m}

        """
        # FIXME: Think about blocks later
        F, dFdu = prepared

        V = self.get_outputs()[idx].output.function_space

        # FIXME: DirichletBC not block variable yet. Required later on. Currently all bcs should be homogenized
        bcs = []
        for bc in self._bcs:
            bcs.append(bc)
        dFdm = ufl.ZeroBaseForm((ufl.TestFunction(V),))
        for block_variable in self.get_dependencies():
            tlm_value = block_variable.tlm_value
            # c = block_variable.output
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
        dJdu = dolfinx.la.vector(adj_rhs.index_map, adj_rhs.block_size)
        dJdu.array[:] = adj_rhs.array[:].copy()

        # Solve adjoint problem
        compiled_dFdu = dolfinx.fem.form(
            dFdu_adj,  # type: ignore[arg-type]
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        self._adjoint_solver._a = compiled_dFdu
        self._adjoint_solver._b = dJdu.petsc_vec
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
        """Evaluate the adjoint component, i.e. :math:`\frac{\\partial Au - b}{\\partial c}`."""

        residual = prepared

        c = block_variable.output

        c_rep = block_variable.saved_output
        if isinstance(c, dolfinx.fem.Function):
            dc = ufl.TrialFunction(c.function_space)
        else:
            raise NotImplementedError(f"Unsupported control {type(c)}")
        dFdm = -ufl.derivative(residual, c_rep, dc)
        dFdm_adj = ufl.adjoint(dFdm)
        sensitivity = ufl.action(dFdm_adj, self._adjoint_solutions)
        compiled_sensitivity = dolfinx.fem.form(
            sensitivity,
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )
        vec = _create_vector(compiled_sensitivity, sensitivity.arguments()[0].ufl_function_space())
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
        for bo in self.get_dependencies():
            c = bo.output
            c_rep = bo.saved_output
            tlm_input = bo.tlm_value
            if tlm_input is None:
                continue
            if isinstance(c, (dolfinx.mesh.Mesh, dolfinx.fem.DirichletBC)):
                raise NotImplementedError(f"Hessian computation for {type(c)} control not implemented yet.")
            else:
                dFdu_adj = ufl.action(ufl.adjoint(dFdu_form), self._adjoint_solutions)
                b_form += ufl.derivative(dFdu_adj, c_rep, tlm_input)

        b = dolfinx.la.vector(hessian_inputs[0].index_map, hessian_inputs[0].block_size)
        b.array[:] = 0.0
        if not ufl.algorithms.apply_derivatives.apply_derivatives(b_form).empty():
            compiled_soa_rhs = dolfinx.fem.form(
                b_form,
                jit_options=self._jit_options,
                form_compiler_options=self._form_compiler_options,
                entity_maps=self._entity_maps,
            )
            dolfinx.fem.petsc.assemble_vector(b.petsc_vec, compiled_soa_rhs)
            b.scatter_reverse(dolfinx.la.InsertMode.add)
            b.array[:] *= -1

        b.array[:] += hessian_inputs[0].array

        # Compile SOA LHS
        dFdu_adj = dolfinx.fem.form(
            ufl.adjoint(dFdu_form),
            jit_options=self._jit_options,
            form_compiler_options=self._form_compiler_options,
            entity_maps=self._entity_maps,
        )

        # Solve adjoint problem
        self._adjoint_solver._a = dFdu_adj
        self._adjoint_solver._b = b.petsc_vec
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
        assemble_compiled_form(compiled_hessian, hessian_output)
        hessian_output.array[:] *= -1.0
        return hessian_output
