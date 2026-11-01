from mpi4py import MPI

import dolfinx
import numpy
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, assemble_scalar

# The L2 Riesz map solves a mass-matrix system with PETSc.
riesz_representation = "L2" if dolfinx.has_petsc4py else "l2"


@pytest.fixture(scope="module")
def mesh_1D():
    return dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


@pytest.fixture(scope="module")
def mesh_2D():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 7, 7)


@pytest.fixture(scope="module")
def mesh_3D():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)


@pytest.mark.parametrize("mesh_var_name", ["mesh_1D", "mesh_2D", "mesh_3D"])
def test_function_control(mesh_var_name: str, request):
    pyadjoint.get_working_tape().clear_tape()
    mesh = request.getfixturevalue(mesh_var_name)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # type: ignore[arg-type]

    v = Function(V)
    v.name = "v"
    v.x.array[:] = 0.2

    def u_ex(mod, x):
        return x[0]

    x = ufl.SpatialCoordinate(mesh)
    c = u_ex(ufl, x)
    error = ufl.inner(v - c, v - c) * ufl.inner(v - c, v - c) * ufl.dx(domain=mesh)

    J = assemble_scalar(error)

    derivative_options = {
        "riesz_representation": riesz_representation,
        "petsc_options": {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        "jit_options": {"cffi_extra_compile_args": ["-Ofast", "-march=native"], "cffi_libraries": ["m"]},
    }
    control = pyadjoint.Control(v, riesz_map=derivative_options)
    Jh = pyadjoint.ReducedFunctional(J, control)
    assert Jh(v) > 0

    # Perform taylor test
    du = Function(V)
    du.interpolate(lambda x: numpy.sin(x[0]))

    # Without gradient
    Jh(v)
    min_rate = pyadjoint.taylor_test(Jh, v, du, dJdm=0)
    assert numpy.isclose(min_rate, 1.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 1.0, got {min_rate}"

    # With gradient
    Jh(v)
    min_rate = pyadjoint.taylor_test(Jh, v, du)
    assert numpy.isclose(min_rate, 2.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 2.0, got {min_rate}"

    # Perform taylor test
    Jh(v)
    dJdm = Jh.derivative()._ad_dot(du)
    hessian = Jh.hessian(du)
    dHddu = hessian._ad_dot(du)
    min_rate = pyadjoint.taylor_test(Jh, v, du, dJdm=dJdm, Hm=dHddu)
    assert numpy.isclose(min_rate, 3.0, rtol=1e-3, atol=1e-3), f"Expected convergence rate close to 3.0, got {min_rate}"
    Jh(v)

    tol = 1e-9
    opt = pyadjoint.minimize(
        Jh,
        method="Newton-CG",
        tol=tol,
        scale=1e9,
        options={"maxiter": 200, "disp": True},
    )

    u_opt = dolfinx.fem.Function(V)
    u_opt.interpolate(lambda x: u_ex(numpy, x))
    assert numpy.isclose(Jh(opt), 0.0, atol=1e-5)
    numpy.testing.assert_allclose(opt.x.array, u_opt.x.array, rtol=1e-3, atol=1e-3)
