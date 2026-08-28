import typing

from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, assemble_scalar
from dolfinx_adjoint.solvers import LinearProblem


@pytest.fixture(scope="module")
def mesh_1D():
    return dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


@pytest.fixture(scope="module")
def mesh_2D():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 7)


@pytest.fixture(scope="module")
def mesh_3D():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 11, 13, 12, cell_type=dolfinx.mesh.CellType.hexahedron)


@pytest.mark.parametrize("constant", [np.float64(0.2), float(-0.13), int(3)])
@pytest.mark.parametrize("mesh_var_name", ["mesh_1D", "mesh_2D", "mesh_3D"])
def test_solver(mesh_var_name: str, request, constant: typing.Union[float, int, np.floating]):
    pyadjoint.get_working_tape().clear_tape()
    mesh = request.getfixturevalue(mesh_var_name)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # type: ignore[arg-type]
    uh = Function(V, name="u_output")

    f = Function(V, name="control")
    f.interpolate(lambda x: np.sin(x[0]))
    k = Function(V, name="kappa")
    k.x.array[:] = 1.0
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx(domain=mesh)
    L = ufl.inner(f, v) * ufl.dx(domain=mesh)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc_val = dolfinx.fem.Constant(mesh, np.dtype(dolfinx.default_scalar_type).type(1.0))
    bc = dolfinx.fem.dirichletbc(bc_val, boundary_dofs, V)

    options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = LinearProblem(
        a, L, u=uh, bcs=[bc], petsc_options=options, adjoint_petsc_options=options, tlm_petsc_options=options
    )
    problem.solve()

    d = pyadjoint.AdjFloat(constant)
    error = (uh - d) ** 3 * ufl.dx
    J = assemble_scalar(error)

    control = pyadjoint.Control(f)
    Jh = pyadjoint.ReducedFunctional(J, control)
    d = Function(V)
    d.interpolate(lambda x: 10 * x[0])

    e = Function(V)
    e.interpolate(lambda x: 10 * np.sin(x[0]))

    Jh.derivative()

    min_rate = pyadjoint.taylor_test(Jh, d, e, dJdm=0)
    assert np.isclose(min_rate, 1.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 1.0, got {min_rate}"

    min_rate = pyadjoint.taylor_test(Jh, d, e)
    assert np.isclose(min_rate, 2.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 2.0, got {min_rate}"

    Jh(d)
    dJdm = Jh.derivative()._ad_dot(e)
    hessian = Jh.hessian(e)
    dHddu = hessian._ad_dot(e)
    min_rate = pyadjoint.taylor_test(Jh, d, e, dJdm=dJdm, Hm=dHddu)
    assert np.isclose(min_rate, 3.0, rtol=5e-3, atol=5e-3), f"Expected convergence rate close to 3.0, got {min_rate}"


def test_linear_mixed_derivative_hessian(mesh_2D):
    """Test LinearProblem with a control in the bilinear form (d2F/dudm != 0)."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = mesh_2D

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    uh = Function(V, name="state")
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)

    # Control variable (conductivity)
    m = Function(V, name="control")
    m.interpolate(lambda x: 1.0 + x[0] ** 2 + x[1] ** 2)

    # Constant Source term
    f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))

    # Linear problem where the bilinear form explicitly depends on the control 'm'
    a = m * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V)

    # Forward Solve
    petsc_options = {
        "ksp_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        u=uh,
        petsc_options=petsc_options,
        adjoint_petsc_options=petsc_options,
        tlm_petsc_options=petsc_options,
    )
    problem.solve()

    # Define Objective
    d = Function(V)
    d.interpolate(lambda x: np.sin(np.pi * x[0]))
    J = assemble_scalar(0.5 * ufl.inner(uh - d, uh - d) * ufl.dx)

    control = pyadjoint.Control(m)
    Jh = pyadjoint.ReducedFunctional(J, control)

    # Perturbation
    dm = Function(V)
    dm.interpolate(lambda x: 2 * np.sin(x[0] * np.pi) * np.cos(x[1] * np.pi))

    # Perturbation test
    min_rate = pyadjoint.taylor_test(Jh, m, dm, dJdm=0)
    assert np.isclose(min_rate, 1.0, rtol=1e-1, atol=1e-1), f"Expected convergence rate close to 1.0, got {min_rate}"
    Jh(m)

    # Gradient Taylor Test
    min_rate_grad = pyadjoint.taylor_test(Jh, m, dm)
    assert np.isclose(min_rate_grad, 2.0, rtol=1e-1, atol=1e-1), f"Grad rate failed: {min_rate_grad}"

    # Hessian Taylor Test
    Jh(m)
    dJ = Jh.derivative()._ad_dot(dm)
    H = Jh.hessian(dm)._ad_dot(dm)
    min_rate_hess = pyadjoint.taylor_test(Jh, m, dm, dJdm=dJ, Hm=H)
    assert np.isclose(min_rate_hess, 3.0, rtol=1e-1, atol=1e-1), f"Hessian rate failed: {min_rate_hess}"
