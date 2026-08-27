import typing

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, assemble_scalar
from dolfinx_adjoint.solvers import LinearProblem


@pytest.fixture(scope="module")
def mesh_2D():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 7)


@pytest.mark.parametrize("use_mixed_space", [True, False])
def test_solver(use_mixed_space: bool, mesh_2D):
    pyadjoint.get_working_tape().clear_tape()
    mesh = mesh_2D
    el_u = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    el_p = basix.ufl.element("P", mesh.basix_cell(), 1)
    V = dolfinx.fem.functionspace(mesh, el_u)
    Q = dolfinx.fem.functionspace(mesh, el_p)
    dx = ufl.Measure("dx", domain=mesh)
    a00 = lambda u, v: ufl.inner(ufl.grad(u), ufl.grad(v)) * dx
    a01 = lambda p, v: ufl.inner(p, ufl.div(v)) * dx
    a10 = lambda q, u: ufl.inner(q, ufl.div(u)) * dx
    L0 = lambda f, v: ufl.inner(f, v) * dx
    L1 = lambda mesh, q: dolfinx.fem.Constant(mesh, 0.0) * q * dx

    Z = dolfinx.fem.functionspace(mesh, ("DG", 0, (mesh.geometry.dim,)))
    f = Function(Z, name="control")
    f.interpolate(lambda x: (np.sin(x[0]), x[1]))

    if use_mixed_space:
        W = ufl.MixedFunctionSpace(*[V, Q])
        u, p = ufl.TrialFunctions(W)
        v, q = ufl.TestFunctions(W)
        a = ufl.extract_blocks(a00(u, v) + a01(p, v) + a10(q, u))
        L = ufl.extract_blocks(L0(f, v) + L1(mesh, q))
    else:
        u = ufl.TrialFunction(V)
        p = ufl.TrialFunction(Q)
        v = ufl.TestFunction(V)
        q = ufl.TestFunction(Q)
        a = [[a00(u, v), a01(p, v)], [a10(q, u), None]]
        L = [L0(f, v), L1(mesh, q)]

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc_val = dolfinx.fem.Constant(mesh, np.zeros((mesh.geometry.dim,), dtype=dolfinx.default_scalar_type))
    bc = dolfinx.fem.dirichletbc(bc_val, boundary_dofs, V)

    options = {
        "ksp_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    uh, ph = (Function(V, name="state"), Function(Q, name="pressure"))
    problem = LinearProblem(
        a,
        L,
        u=[uh, ph],
        bcs=[bc],
        petsc_options=options,
        adjoint_petsc_options=options,
        tlm_petsc_options=options,
    )
    problem.solve()

    x = ufl.SpatialCoordinate(mesh)
    c = ufl.as_vector((2 * ufl.sin(x[0]), 3 * ufl.cos(x[1])))
    error = ufl.inner(uh - c, uh - c) * ufl.inner(uh - c, uh - c) * ufl.dx
    J = assemble_scalar(error)

    control = pyadjoint.Control(f)
    Jh = pyadjoint.ReducedFunctional(J, control)
    d = Function(Z)
    d.interpolate(lambda x: (10 * x[0], x[1]))

    e = Function(Z)
    e.interpolate(lambda x: (1e3 * np.sin(x[1]), 1e3 * x[0]))
    min_rate = pyadjoint.taylor_test(Jh, d, e, dJdm=0)
    assert np.isclose(min_rate, 1.0, rtol=1e-1, atol=1e-1), f"Expected convergence rate close to 1.0, got {min_rate}"

    Jh.derivative()
    min_rate = pyadjoint.taylor_test(Jh, d, e)
    assert np.isclose(min_rate, 2.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 2.0, got {min_rate}"

    Jh(d)
    dJdm = Jh.derivative()._ad_dot(e)
    hessian = Jh.hessian(e)
    dHddu = hessian._ad_dot(e)
    min_rate = pyadjoint.taylor_test(Jh, d, e, dJdm=dJdm, Hm=dHddu)
    assert np.isclose(min_rate, 3.0, rtol=0.1, atol=0.1), f"Expected convergence rate close to 3.0, got {min_rate}"

    z = Function(Z)
    z.interpolate(lambda x: (np.sin(x[1]), -(x[0] ** 2)))
    f = Function(Z)
    f.interpolate(lambda x: (1e4 * x[0], 1e5 * np.sin(x[1])))
    Jh(z)
    dJdm = Jh.derivative()._ad_dot(f)
    hessian = Jh.hessian(f)
    dHddu = hessian._ad_dot(f)
    min_rate = pyadjoint.taylor_test(Jh, z, f, dJdm=dJdm, Hm=dHddu)
    assert np.isclose(min_rate, 3.0, rtol=0.1, atol=0.1), f"Expected convergence rate close to 3.0, got {min_rate}"
