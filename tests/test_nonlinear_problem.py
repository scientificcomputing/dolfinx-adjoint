from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import ufl

from dolfinx_adjoint import Constant, Function, assemble_scalar, dirichletbc
from dolfinx_adjoint.solvers import NonlinearProblem

# A direct linear solve, shared by every bc-control test below's *adjoint/TLM* solver
# (always linear, even for NonlinearProblem -- see _get_or_build_adjoint_solver/
# _get_or_build_tlm_solver in solvers.py) -- no snes_* options here, since SNES never
# runs for that solve.
_bc_control_linear_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "mat_mumps_icntl_24": 1,
}
# Explicit, tight SNES tolerances (rather than the defaults) for every bc-control test
# below's *forward* solve: a Taylor test replays the same NonlinearProblem at many nearby
# control values in a row, which is exactly the SNES warm-start pattern that can produce a
# false DIVERGED_LINE_SEARCH without them (see dolfinx-adjoint-knowledge's solver-reuse
# notes).
_bc_control_snes_options = {
    "snes_type": "newtonls",
    "snes_error_if_not_converged": True,
    "snes_atol": 1e-11,
    "snes_rtol": 1e-11,
    "snes_stol": 1e-11,
    **_bc_control_linear_options,
}


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


def test_scalar_dirichletbc_control():
    """A tracked (dolfinx_adjoint) Dirichlet bc value IS supported as a control for
    NonlinearProblem: NonlinearProblemBlock registers it as a tape dependency exactly like
    LinearProblemBlock does, and the shared boundary-reaction machinery in
    _ProblemBlockBase (blocks/solvers.py) does not distinguish Problem kind.

    Unlike every bc-control test in test_dirichlet_bc.py (a *linear* PDE, where the
    objective is exactly quadratic in the bc value and the standard rate-3 Hessian
    Taylor test degenerates into floating-point noise), F here is genuinely nonlinear in
    u, so J is not exactly quadratic in g and the standard Taylor ladder is a real,
    meaningful signal at every order.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    u = Function(V, name="state")
    u.interpolate(lambda x: np.ones_like(x[0]))
    v = ufl.TestFunction(V)
    F = ufl.inner((1 + u**2) * ufl.grad(u), ufl.grad(v)) * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)

    g = Function(V, name="bc_value")
    g.interpolate(lambda x: 1.0 + 0.5 * np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))
    bc = dirichletbc(g, boundary_dofs, V=V)

    # The tape-recording NonlinearProblemBlock (where the bc-dependency registration
    # lives) is only built lazily, on the first solve -- see _ProblemBase._make_block.
    problem = NonlinearProblem(
        F,
        u=u,
        bcs=[bc],
        petsc_options=_bc_control_snes_options,
        adjoint_petsc_options=_bc_control_linear_options,
    )
    problem.solve()

    J = assemble_scalar(ufl.inner(u, u) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(g))

    g0 = Function(V)
    g0.x.array[:] = g.x.array
    # An asymmetric perturbation: a direction with an accidental symmetry (e.g. a bare
    # cos(pi*x[1]) against a gradient that happens to be independent of x[1] for this
    # particular g) can integrate to a deceptively clean-looking near-zero gradient that
    # is a coincidence of the L2 inner product, not a real signal.
    h = Function(V)
    h.interpolate(lambda x: 0.7 * np.cos(2 * np.pi * x[0]) + 0.9 * np.sin(3 * np.pi * x[1]))

    Jhat(g0)
    min_rate0 = pyadjoint.taylor_test(Jhat, g0, h, dJdm=0)
    assert np.isclose(min_rate0, 1.0, rtol=1e-1, atol=1e-1), f"Expected rate 1.0, got {min_rate0}"

    Jhat(g0)
    min_rate1 = pyadjoint.taylor_test(Jhat, g0, h)
    assert np.isclose(min_rate1, 2.0, rtol=1e-1, atol=1e-1), f"Expected rate 2.0, got {min_rate1}"

    Jhat(g0)
    dJdm = Jhat.derivative()._ad_dot(h)
    Hm = Jhat.hessian(h)._ad_dot(h)
    min_rate2 = pyadjoint.taylor_test(Jhat, g0, h, dJdm=dJdm, Hm=Hm)
    assert np.isclose(min_rate2, 3.0, rtol=1e-1, atol=1e-1), f"Expected rate 3.0, got {min_rate2}"


def test_scalar_dirichletbc_general_expression():
    """As test_scalar_dirichletbc_control, but the bc value is a genuine nonlinear UFL
    expression of a control (``m**3``, mirroring
    test_dirichlet_bc.py::test_scalar_dirichletbc_general_expression), composed with a
    forward problem that is itself nonlinear in the state -- exercises
    ExprInterpolationBlock feeding into the NonlinearProblem boundary-control path.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    u = Function(V, name="state")
    u.interpolate(lambda x: np.ones_like(x[0]))
    v = ufl.TestFunction(V)
    F = ufl.inner((1 + u**2) * ufl.grad(u), ufl.grad(v)) * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)

    m = Constant(mesh, 1.2)
    bc = dirichletbc(m**3, boundary_dofs, V=V)

    problem = NonlinearProblem(
        F,
        u=u,
        bcs=[bc],
        petsc_options=_bc_control_snes_options,
        adjoint_petsc_options=_bc_control_linear_options,
    )
    problem.solve()

    J = assemble_scalar(ufl.inner(u, u) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(m))

    m0 = Constant(mesh, 1.2)
    h = Constant(mesh, 1.0)

    Jhat(m0)
    min_rate0 = pyadjoint.taylor_test(Jhat, m0, h, dJdm=0)
    assert np.isclose(min_rate0, 1.0, rtol=1e-1, atol=1e-1), f"Expected rate 1.0, got {min_rate0}"

    Jhat(m0)
    min_rate1 = pyadjoint.taylor_test(Jhat, m0, h)
    assert np.isclose(min_rate1, 2.0, rtol=1e-1, atol=1e-1), f"Expected rate 2.0, got {min_rate1}"

    Jhat(m0)
    dJdm = Jhat.derivative()._ad_dot(h)
    Hm = Jhat.hessian(h)._ad_dot(h)
    min_rate2 = pyadjoint.taylor_test(Jhat, m0, h, dJdm=dJdm, Hm=Hm)
    assert np.isclose(min_rate2, 3.0, rtol=1e-1, atol=1e-1), f"Expected rate 3.0, got {min_rate2}"


if __name__ == "__main__":
    test_sequential_nonlinear_problems()
