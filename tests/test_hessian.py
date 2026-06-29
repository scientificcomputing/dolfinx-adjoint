from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import ufl

import dolfinx_adjoint


def test_constant_hessian():
    pyadjoint.get_working_tape().clear_tape()

    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
    V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

    # ==========================================
    # 1. SETUP THE FORWARD PROBLEM
    # PDE: -div(grad(u)) + m * u = f
    # Where 'm' is our scalar control parameter
    # ==========================================
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = dolfinx_adjoint.Constant(domain, 1.0)

    # The true parameter value
    m_val = 2.0
    m_control = dolfinx_adjoint.Constant(domain, m_val)

    # Weak form
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + m_control * ufl.inner(u, v) * ufl.dx
    L = ufl.inner(f, v) * ufl.dx

    # Zero Dirichlet Boundary Conditions
    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
    uD = dolfinx_adjoint.Function(V)
    uD.x.array[:] = 0.0
    bc = dolfinx_adjoint.dirichletbc(uD, boundary_dofs)

    # Solve and tape the PDE
    u_sol = dolfinx_adjoint.Function(V, name="State")
    problem = dolfinx_adjoint.LinearProblem(a, L, bcs=[bc], u=u_sol)
    problem.solve()

    # ==========================================
    # 2. SETUP DATA MISFIT
    # J_data = 1/(2*var) * \int (u - u_obs)^2 dx
    # ==========================================
    u_obs = dolfinx_adjoint.Function(V)
    u_obs.x.array[:] = 0.0  # Dummy observation

    J_form = 0.5 * ufl.inner(u_sol - u_obs, u_sol - u_obs) * ufl.dx
    J_data = dolfinx_adjoint.assemble_scalar(J_form)

    # ==========================================
    # 3. EXTRACT THE EXACT TRUE HESSIAN
    # ==========================================
    control = pyadjoint.Control(m_control)
    Jhat = pyadjoint.ReducedFunctional(J_data, control)

    # To get the dense 1x1 Hessian matrix, we compute the Hessian Action
    # in the standard basis direction (which for a scalar is simply 1.0)
    direction = dolfinx_adjoint.Constant(domain, 1.0)
    hessian_action = Jhat.hessian(direction)

    # Cast the action result to a standard Python float
    H_misfit = hessian_action.x.array[0]

    # Ensure PyAdjoint actually computed a non-zero curvature!
    assert H_misfit > 0.0, "Hessian computation failed or is zero!"


def test_constant_hessian_assemble_only():
    """
    Test 1: Does AssembleBlock support Hessians for Constants?
    J(m) = 0.5 * (m - 5)**2 * vol
    d2J/dm2 = 1.0 * vol
    """
    pyadjoint.get_working_tape().clear_tape()
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)

    m = dolfinx_adjoint.Constant(domain, 3.0)

    # J = 0.5 * (m - 5)^2 * dx
    J_form = 0.5 * (m - 5.0) ** 2 * ufl.dx(domain)
    J = dolfinx_adjoint.assemble_scalar(J_form)

    control = pyadjoint.Control(m)
    Jhat = pyadjoint.ReducedFunctional(J, control)

    # Direction m_t = 1.0
    direction = dolfinx_adjoint.Constant(domain, 1.0)
    hessian_action = Jhat.hessian(direction)

    # Expected Hessian is simply the volume of the domain (1.0 for a unit square)
    H_val = hessian_action.x.array[0]

    assert H_val > 0.0, f"AssembleBlock Hessian failed! Value is {H_val}"
    assert np.isclose(H_val, 1.0), f"Expected 1.0, got {H_val}"


def test_constant_hessian_linear_source():
    """
    Test 2: Does LinearProblemBlock support TLM and SOA for linear parameters?
    PDE: -div(grad(u)) = m
    J(u) = 0.5 * u**2 * dx
    Here, d2F/dudm = 0, so the Hessian is purely the pullback of the objective Hessian.
    """
    pyadjoint.get_working_tape().clear_tape()
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

    m = dolfinx_adjoint.Constant(domain, 2.0)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # m is just a source term (linear dependence)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = m * v * ufl.dx

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
    u_bc = dolfinx_adjoint.Function(V)
    u_bc.x.array[:] = 0.0
    bc = dolfinx_adjoint.dirichletbc(u_bc, boundary_dofs)

    u_sol = dolfinx_adjoint.Function(V)
    problem = dolfinx_adjoint.LinearProblem(a, L, bcs=[bc], u=u_sol)
    problem.solve()

    J_form = 0.5 * ufl.inner(u_sol, u_sol) * ufl.dx
    J = dolfinx_adjoint.assemble_scalar(J_form)

    control = pyadjoint.Control(m)
    Jhat = pyadjoint.ReducedFunctional(J, control)

    direction = dolfinx_adjoint.Constant(domain, 1.0)
    hessian_action = Jhat.hessian(direction)
    H_val = hessian_action.x.array[0]

    assert H_val > 0.0, f"Linear source Hessian failed! Value is {H_val}"


def test_constant_hessian_linear_operator():
    """
    Test 3: Does LinearProblemBlock support cross-derivatives (d2F/dudm)?
    PDE: -div(grad(u)) + m * u = f
    """
    pyadjoint.get_working_tape().clear_tape()
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
    V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

    m = dolfinx_adjoint.Constant(domain, 2.0)
    f = dolfinx_adjoint.Constant(domain, 1.0)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # m multiplies u (non-linear dependence on the parameter)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + m * ufl.inner(u, v) * ufl.dx
    L = f * v * ufl.dx

    domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)
    u_bc = dolfinx_adjoint.Function(V)
    u_bc.x.array[:] = 0.0
    bc = dolfinx_adjoint.dirichletbc(u_bc, boundary_dofs)

    u_sol = dolfinx_adjoint.Function(V)
    problem = dolfinx_adjoint.LinearProblem(a, L, bcs=[bc], u=u_sol)
    problem.solve()

    J_form = 0.5 * ufl.inner(u_sol, u_sol) * ufl.dx
    J = dolfinx_adjoint.assemble_scalar(J_form)

    control = pyadjoint.Control(m)
    Jhat = pyadjoint.ReducedFunctional(J, control)

    direction = dolfinx_adjoint.Constant(domain, 1.0)
    hessian_action = Jhat.hessian(direction)
    H_val = hessian_action.x.array[0]

    assert H_val > 0.0, f"Operator Hessian failed! Value is {H_val}"
