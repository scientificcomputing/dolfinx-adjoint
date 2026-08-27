import typing
from mpi4py import MPI
import basix.ufl
import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, assemble_scalar
from dolfinx_adjoint.solvers import NonlinearProblem


@pytest.fixture(scope="module")
def mesh_2D():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)


def test_unblocked_nonlinear(mesh_2D):
    """Test standard scalar (unblocked) nonlinear solver (Nonlinear Poisson)."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = mesh_2D

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    uh = Function(V, name="state")
    v = ufl.TestFunction(V)

    f = Function(V, name="control")
    f.interpolate(lambda x: np.sin(x[0] * np.pi) * np.cos(x[1] * np.pi))

    # Nonlinear Poisson: -div((1+u^2)*grad(u)) = f
    F = (1 + uh**2) * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V)

    problem = NonlinearProblem(F, u=uh, bcs=[bc])
    problem.solve()

    # Define Objective
    d = Function(V)
    d.interpolate(lambda x: x[0] + x[1])
    J = assemble_scalar(0.5 * ufl.inner(uh - d, uh - d) * ufl.dx)

    control = pyadjoint.Control(f)
    Jh = pyadjoint.ReducedFunctional(J, control)

    # Initial control state
    m0 = Function(V)
    m0.interpolate(lambda x: x[0] ** 2)

    # Perturbation (scaled to avoid machine precision float cancellation)
    dm = Function(V)
    dm.interpolate(lambda x: 100 * np.sin(x[0] * np.pi))

    # 1. Gradient Taylor Test
    min_rate_grad = pyadjoint.taylor_test(Jh, m0, dm, dJdm=0)
    assert np.isclose(min_rate_grad, 2.0, rtol=1e-1, atol=1e-1), f"Grad rate failed: {min_rate_grad}"

    # 2. Hessian Taylor Test
    Jh(m0)
    dJ = Jh.derivative()._ad_dot(dm)
    H = Jh.hessian(dm)._ad_dot(dm)
    min_rate_hess = pyadjoint.taylor_test(Jh, m0, dm, dJdm=dJ, Hm=H)
    assert np.isclose(min_rate_hess, 3.0, rtol=1e-1, atol=1e-1), f"Hessian rate failed: {min_rate_hess}"


def test_blocked_nonlinear(mesh_2D):
    """Test blocked nonlinear solver using Steady Navier-Stokes."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = mesh_2D

    el_u = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    el_p = basix.ufl.element("P", mesh.basix_cell(), 1)
    V = dolfinx.fem.functionspace(mesh, el_u)
    Q = dolfinx.fem.functionspace(mesh, el_p)

    uh = Function(V, name="velocity")
    ph = Function(Q, name="pressure")
    v = ufl.TestFunction(V)
    q = ufl.TestFunction(Q)

    f = Function(V, name="control")
    f.interpolate(lambda x: (np.sin(x[0]), np.cos(x[1])))

    nu = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.1))

    # Epsilon penalty to remove the pressure nullspace gracefully across MPI ranks
    eps = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1e-6))

    # Steady Navier Stokes
    F = (
        nu * ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx
        + ufl.inner(ufl.grad(uh) * uh, v) * ufl.dx
        - ph * ufl.div(v) * ufl.dx
        + q * ufl.div(uh) * ufl.dx
        + eps * ph * q * ufl.dx
        - ufl.inner(f, v) * ufl.dx
    )

    # Extract blocked forms
    F_blocks = ufl.extract_blocks(F)

    # No-slip boundary conditions
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs_V = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc_u = dolfinx.fem.dirichletbc(np.zeros(2, dtype=dolfinx.default_scalar_type), boundary_dofs_V, V)

    options = {
        "snes_type": "newtonls",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-9,
    }

    problem = NonlinearProblem(F_blocks, u=[uh, ph], bcs=[bc_u], petsc_options=options)
    problem.solve()

    # Target velocity field
    d = Function(V)
    d.interpolate(lambda x: (x[1], -x[0]))
    J = assemble_scalar(0.5 * ufl.inner(uh - d, uh - d) * ufl.dx)

    control = pyadjoint.Control(f)
    Jh = pyadjoint.ReducedFunctional(J, control)

    # Control evaluation point
    m0 = Function(V)
    m0.interpolate(lambda x: (x[0] ** 2, x[1] ** 2))

    # Perturbation (scaled to pull the 3rd order remainder out of the float noise floor)
    dm = Function(V)
    dm.interpolate(lambda x: (400 * np.sin(x[1] * np.pi), 400 * np.cos(x[0] * np.pi)))

    # 1. Gradient Taylor Test
    min_rate_grad = pyadjoint.taylor_test(Jh, m0, dm, dJdm=0)
    assert np.isclose(min_rate_grad, 2.0, rtol=1e-1, atol=1e-1), f"Grad rate failed: {min_rate_grad}"

    # 2. Hessian Taylor Test
    Jh(m0)
    dJ = Jh.derivative()._ad_dot(dm)
    H = Jh.hessian(dm)._ad_dot(dm)
    min_rate_hess = pyadjoint.taylor_test(Jh, m0, dm, dJdm=dJ, Hm=H)
    assert np.isclose(min_rate_hess, 3.0, rtol=1e-1, atol=1e-1), f"Hessian rate failed: {min_rate_hess}"
