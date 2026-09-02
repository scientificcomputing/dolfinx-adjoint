from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, assemble_scalar, dirichletbc
from dolfinx_adjoint.solvers import NonlinearProblem


def test_sequential_nonlinear_problems():
    """
    Test two cascaded non-linear PDEs.
    PDE 1: -div(u1 * grad(u1)) = f
    PDE 2: -div(u2 * grad(u2)) = u1
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 7)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    f = Function(V, name="control")
    # Keep the control positive to avoid degeneracy in the diffusion tensor
    f.interpolate(lambda x: 2.0 + np.sin(x[0]))

    u1 = Function(V, name="state_1")
    u1.interpolate(lambda x: np.ones_like(x[0]))  # Non-zero initial guess
    v1 = ufl.TestFunction(V)

    F1 = (1 + u1**2) * ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx(domain=mesh) - f * v1 * ufl.dx(domain=mesh)

    # Setup PDE 2
    u2 = Function(V, name="state_2")
    u2.interpolate(lambda x: np.ones_like(x[0]))  # Non-zero initial guess
    v2 = ufl.TestFunction(V)
    F2 = (2 + u2**2) * ufl.inner(ufl.grad(u2), ufl.grad(v2)) * ufl.dx(domain=mesh) - u1 * v2 * ufl.dx(domain=mesh)

    # 4. Boundary Conditions (u = 1.0 on boundary)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc_val = dolfinx.fem.Constant(mesh, np.dtype(dolfinx.default_scalar_type).type(1.0))
    bc = dolfinx.fem.dirichletbc(bc_val, boundary_dofs, V)

    # Use SNES options for the nonlinear solver
    direct_options = {
        "ksp_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "mat_mumps_icntl_24": 1,
        "pc_factor_mat_ordering_type": "rcm",
    }
    options = {
        "snes_monitor": None,
        "snes_error_if_not_converged": True,
        "snes_type": "newtonls",
    }
    options.update(direct_options)

    # 5. Solve the Cascade
    problem1 = NonlinearProblem(F1, u=u1, bcs=[bc], petsc_options=options, adjoint_petsc_options=direct_options)
    problem1.solve()

    problem2 = NonlinearProblem(F2, u=u2, bcs=[bc], petsc_options=options, adjoint_petsc_options=direct_options)
    problem2.solve()

    # 6. Objective (using the cubed error to ensure a 3.0 Hessian rate)
    d = pyadjoint.AdjFloat(0.2)
    error = (u2 - d) ** 3 * ufl.dx(domain=mesh)
    J = assemble_scalar(error)

    # 7. Taylor Tests
    control = pyadjoint.Control(f)
    Jh = pyadjoint.ReducedFunctional(J, control)

    ctrl_eval = Function(V)
    ctrl_eval.interpolate(lambda x: 4.0 + np.sin(x[0]))

    pert = Function(V)
    pert.interpolate(lambda x: 15.1 * np.cos(x[1]))

    Jh(ctrl_eval)
    min_rate_grad = pyadjoint.taylor_test(Jh, ctrl_eval, pert, dJdm=0)
    print("\n--- 1st-order Taylor test ---")
    Jh(ctrl_eval)
    min_rate_grad = pyadjoint.taylor_test(Jh, ctrl_eval, pert)
    assert np.isclose(min_rate_grad, 2.0, rtol=1e-2, atol=1e-2), f"Expected 2.0, got {min_rate_grad}"

    print("\n--- 2nd-order Taylor test ---")
    Jh(ctrl_eval)
    dJdm = Jh.derivative()._ad_dot(pert)
    dHddu = Jh.hessian(pert)._ad_dot(pert)
    min_rate_hess = pyadjoint.taylor_test(Jh, ctrl_eval, pert, dJdm=dJdm, Hm=dHddu)
    assert np.isclose(min_rate_hess, 3.0, rtol=1e-2, atol=1e-2), f"Expected 3.0, got {min_rate_hess}"


@pytest.mark.xfail(
    reason="Boundary control (a tracked Dirichlet bc value) is not implemented for "
    "NonlinearProblem: NonlinearProblemBlock never adds a bc as a tape dependency, so "
    "it would otherwise give pyadjoint a silent zero gradient rather than erroring, and "
    "an explicit guard in NonlinearProblemBlock.__init__ (blocks/solvers.py) raises "
    "NotImplementedError instead. Use LinearProblem if the bc's value needs to be a "
    "control. Marked xfail(strict=True) rather than asserted via pytest.raises so that "
    "implementing this feature later trips an XPASS here as a reminder to upgrade this "
    "test into a real one, instead of the guard's removal going unnoticed.",
    raises=NotImplementedError,
    strict=True,
)
def test_tracked_bc_rejected():
    """A tracked (dolfinx_adjoint) Dirichlet bc value is not supported as a control for
    NonlinearProblem -- see the xfail reason above.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    u = Function(V, name="state")
    u.interpolate(lambda x: np.ones_like(x[0]))
    v = ufl.TestFunction(V)
    F = ufl.inner((1 + u**2) * ufl.grad(u), ufl.grad(v)) * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)

    g = Function(V, name="bc_value")
    g.interpolate(lambda x: np.ones_like(x[0]))
    bc = dirichletbc(g, boundary_dofs, V=V)

    # The tape-recording NonlinearProblemBlock (where the guard lives) is only built
    # lazily, on the first solve -- see _ProblemBase._make_block.
    problem = NonlinearProblem(F, u=u, bcs=[bc])
    problem.solve()


if __name__ == "__main__":
    test_sequential_nonlinear_problems()
