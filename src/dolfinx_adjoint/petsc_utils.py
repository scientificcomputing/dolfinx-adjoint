from __future__ import annotations

import typing

from petsc4py import PETSc

import dolfinx.fem.petsc


def solve_linear_problem(
    A: PETSc.Mat,  # type: ignore [name-defined]
    x: dolfinx.la.Vector,
    b: dolfinx.la.Vector,
    petsc_options: typing.Optional[dict] = None,
    P: typing.Optional[PETSc.Mat] = None,  # type: ignore [name-defined]
):
    """Solve a linear problem :math:`Ax = b`.

    Args:
        A: The matrix
        x: The solution vector
        b: The right-hand side vector
        petsc_options: Optional dictionary of PETSc options for the solver.
        P: Optional preconditioner matrix. If not provided, no preconditioner is used.
    """

    petsc_options = {} if petsc_options is None else petsc_options
    error_if_not_converged = petsc_options.pop("ksp_error_if_not_converged", True)
    petsc_options["ksp_error_if_not_converged"] = error_if_not_converged
    ksp = PETSc.KSP().create(A.comm)  # type: ignore [attr-defined]

    ksp.setOperators(A, P)

    # Give PETSc solver options a unique prefix
    problem_prefix = f"dolfinx_adjoint_linear_problem_{id(ksp)}"
    ksp.setOptionsPrefix(problem_prefix)

    # Set PETSc options
    opts = PETSc.Options()  # type: ignore [attr-defined]
    opts.prefixPush(problem_prefix)
    for k, v in petsc_options.items():
        opts.setValue(k, v)
    opts.prefixPop()
    ksp.setFromOptions()

    # Set matrix and vector PETSc options
    A.setOptionsPrefix(problem_prefix)
    A.setFromOptions()
    b.petsc_vec.setOptionsPrefix(problem_prefix)
    b.petsc_vec.setFromOptions()

    # Free option space post setting
    for k in petsc_options.keys():
        opts.delValue(k)
    ksp.solve(b.petsc_vec, x.petsc_vec)
    ksp.destroy()
    x.scatter_forward()


class LinearAdjointProblem(dolfinx.fem.petsc.LinearProblem):
    """Linear problem helper class that homogenizes the boundary conditions, meaning that no lifting is applied."""

    def solve(
        self,
    ) -> typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]]:
        """Solve the problem."""

        # Assemble lhs
        self._A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self._A, self._a, bcs=self.bcs)  # type: ignore
        self._A.assemble()

        # Assemble preconditioner
        if self._P_mat is not None:
            self._P_mat.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(self._P_mat, self._preconditioner, bcs=self.bcs)  # type: ignore
            self._P_mat.assemble()

        if self.bcs is not None:
            try:
                for bc in self.bcs:
                    bc.set(self._b.array_w, alpha=0.0)
            except RuntimeError:
                bcs0 = dolfinx.fem.bcs.bcs_by_block(dolfinx.fem.forms.extract_spaces(self._L), self.bcs)  # type: ignore
                dolfinx.fem.petsc.set_bc(self._b, bcs0, alpha=0.0)

        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        dolfinx.la.petsc._ghost_update(self._x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore
        dolfinx.fem.petsc.assign(self._x, self._u)  # type: ignore
        return self._u
