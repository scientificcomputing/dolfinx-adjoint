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


def test_create_many_solvers():
    tape = pyadjoint.get_working_tape()
    tape.clear_tape()
    mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, cell_type=dolfinx.mesh.CellType.hexahedron)

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

    J = 0.0
    # problems = []
    for i in tape.timestepper(iter(range(500))):
        problem = LinearProblem(
            a,
            L,
            u=uh,
            bcs=[bc],
            petsc_options=options,
            adjoint_petsc_options=options,
            tlm_petsc_options=options,
            petsc_options_prefix=f"dxa_linear_problem_block_{i}",
        )
        problem.solve()

        d = pyadjoint.AdjFloat(0.2)
        error = (uh - d) ** 3 * ufl.dx
        J += assemble_scalar(error)

    control = pyadjoint.Control(f)
    Jh = pyadjoint.ReducedFunctional(J, control)

    Jh(f)
