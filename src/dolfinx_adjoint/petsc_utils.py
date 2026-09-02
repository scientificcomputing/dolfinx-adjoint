from __future__ import annotations

import typing

from petsc4py import PETSc

import dolfinx.fem.petsc


def solve_linear_problem(
    A: PETSc.Mat,  # type: ignore [name-defined]
    x: dolfinx.la.Vector,
    b: dolfinx.la.Vector,
    petsc_options: dict | None = None,
    P: PETSc.Mat | None = None,  # type: ignore [name-defined]
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


class HomogeneousBCLinearProblem(dolfinx.fem.petsc.LinearProblem):
    """Linear problem helper class that homogenizes the boundary conditions, meaning that no lifting is applied.

    Used for both the adjoint and the tangent-linear solve -- neither is "the adjoint problem"
    specifically, they are both just a linear solve against a zero-lifted right-hand side, so the
    name no longer singles out one of the two callers.
    """

    #: Optional bcs whose *value* (not just dof pattern) should be written into ``self.b``
    #: after the usual ``alpha=0.0`` homogenization -- the tangent-linear solve's mechanism
    #: for a boundary-control perturbation (see ``solve()`` and
    #: ``blocks/solvers.py::_ProblemBlockBase.prepare_evaluate_tlm``). ``None`` for the
    #: adjoint solver, which never perturbs a bc's value, only its dofs.
    tlm_bcs: typing.Sequence[dolfinx.fem.DirichletBC] | None = None

    def solve(
        self,
    ) -> typing.Union[dolfinx.fem.Function, typing.Sequence[dolfinx.fem.Function]]:
        """Solve the problem.

        Unlike the base class, ``self.b`` is never (re-)assembled from the compiled
        right-hand-side form here. This solver is shared, and reused verbatim, across every
        block a Problem records (see dolfinx-adjoint-knowledge's solver-reuse note), so the
        caller (see ``_ProblemBlockBase.prepare_evaluate_adj``/``prepare_evaluate_hessian``/
        ``prepare_evaluate_tlm``) has already assembled its own right-hand side directly into
        ``self.b`` before calling ``solve()``. The only thing this method does to ``self.b`` is
        modify it in place -- zeroing every Dirichlet-BC dof (``alpha=0.0``) -- rather than
        lifting, since a caller's right-hand side is never the original problem's own Dirichlet
        data.
        """

        # Assemble lhs
        self._A.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(self._A, self._a, bcs=self.bcs)  # type: ignore
        self._A.assemble()

        # Assemble preconditioner
        if self._P_mat is not None:
            self._P_mat.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix(self._P_mat, self._preconditioner, bcs=self.bcs)  # type: ignore
            self._P_mat.assemble()

        # Tangent-linear boundary control: self._A's bc columns are already eliminated, so
        # simply setting self._b's boundary dofs to the perturbation direction would leave
        # u_dot's interior dofs at whatever the caller's own RHS already put there,
        # discarding the perturbation's propagation through the PDE. apply_lifting
        # recomputes that propagation from the (unmodified) form self._a -- it must run
        # *before* the bc dofs are set to their final values (its own alpha=1.0 default
        # expects x0=0, matching the zeroed state prepare_evaluate_tlm leaves this vector
        # in). Dofs not covered by self.tlm_bcs (untracked, or no tangent-linear value this
        # call) correctly get no lifting, matching u_dot=0 there. See
        # dolfinx-adjoint-knowledge's scratch/boundary-control/spec.md for the full
        # derivation.
        if self.tlm_bcs:
            if isinstance(self._u, list):
                bcs_lift = dolfinx.fem.bcs.bcs_by_block(dolfinx.fem.extract_function_spaces(self._L), self.tlm_bcs)  # type: ignore
                dolfinx.fem.petsc.apply_lifting(self._b, self._a, bcs=bcs_lift)  # type: ignore
            else:
                dolfinx.fem.petsc.apply_lifting(self._b, [self._a], bcs=[self.tlm_bcs])  # type: ignore
            self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)  # type: ignore

        if self.bcs is not None:
            if isinstance(self._u, list):
                # `bc.set()` on the monolithic blocked vector has no block-offset
                # translation: a bc constraining any block other than the first would land
                # its raw (block-local) dof indices in the wrong block. Route through
                # bcs_by_block/set_bc unconditionally for blocked problems, mirroring the
                # base LinearProblem.solve()'s own isinstance(self.u, Sequence) branch (see
                # dolfinx-adjoint-knowledge's scratch/boundary-control/issues/01 for the
                # bug this fixes).
                bcs0 = dolfinx.fem.bcs.bcs_by_block(dolfinx.fem.extract_function_spaces(self._L), self.bcs)  # type: ignore
                dolfinx.fem.petsc.set_bc(self._b, bcs0, alpha=0.0)
            else:
                for bc in self.bcs:
                    bc.set(self._b.array_w, alpha=0.0)

        # Overwrite (alpha=1.0, x0=None -> x[dof]=g) the state's tlm value at exactly the
        # bcs in self.tlm_bcs' own dofs with g -- their perturbation direction -- rather
        # than the homogeneous 0 the pass above just wrote everywhere.
        if self.tlm_bcs:
            if isinstance(self._u, list):
                bcs0 = dolfinx.fem.bcs.bcs_by_block(dolfinx.fem.extract_function_spaces(self._L), self.tlm_bcs)  # type: ignore
                dolfinx.fem.petsc.set_bc(self._b, bcs0, alpha=1.0)
            else:
                for bc in self.tlm_bcs:
                    bc.set(self._b.array_w, alpha=1.0)

        self._b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)  # type: ignore
        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        dolfinx.la.petsc._ghost_update(self._x, PETSc.InsertMode.INSERT, PETSc.ScatterMode.FORWARD)  # type: ignore
        dolfinx.fem.petsc.assign(self._x, self._u)  # type: ignore
        return self._u
