import typing

from mpi4py import MPI

import dolfinx
import numpy
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Constant, Function, assemble_scalar, assign


@pytest.fixture(scope="module")
def mesh_1D():
    return dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


@pytest.fixture(scope="module")
def mesh_2D():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 7)


@pytest.fixture(scope="module")
def mesh_3D():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 50, 50, 50)


def test_multiple_assign_adjoint_accumulation(mesh_1D):
    """
    Test that assigning a Function multiple times and computing the adjoint
    successfully accumulates the adjoints. This tests the IAddableVector patch.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = mesh_1D

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    # 1. Create a control function
    f = Function(V, name="f_control")
    f.interpolate(lambda x: 2.0 * x[0])

    # 2. Fork the computational graph by assigning f to two different functions
    u1 = Function(V, name="u1")
    assign(f, u1)

    u2 = Function(V, name="u2")
    assign(f, u2)

    # 3. Assemble a functional depending on both branches
    J_form = (u1**2 + u2**2) * ufl.dx(domain=mesh)
    J = assemble_scalar(J_form)

    control = pyadjoint.Control(f)
    Jhat = pyadjoint.ReducedFunctional(J, control)

    # 4. Compute the derivative
    # Before the patch, this would crash because pyadjoint attempts to sum
    # the adjoint contributions from u1 and u2 using `+=` on dolfinx.la.Vector.
    # After the patch, IAddableVector handles this gracefully.
    grad = Jhat.derivative()
    assert grad is not None

    # 5. Verify the math with a Taylor test
    df = Function(V)
    df.interpolate(lambda x: np.sin(x[0]))

    # Without gradient
    min_rate_0 = pyadjoint.taylor_test(Jhat, f, df, dJdm=0)
    assert np.isclose(min_rate_0, 1.0, rtol=1e-2, atol=1e-2)

    # With gradient
    min_rate_1 = pyadjoint.taylor_test(Jhat, f, df)
    assert np.isclose(min_rate_1, 2.0, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("constant", [np.float64(0.2), float(-0.13), int(3)])
@pytest.mark.parametrize("mesh_var_name", ["mesh_1D", "mesh_2D", "mesh_3D"])
def test_assign_constant(mesh_var_name: str, request, constant: typing.Union[float, int, np.floating]):
    pyadjoint.get_working_tape().clear_tape()
    mesh = request.getfixturevalue(mesh_var_name)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # type: ignore[arg-type]
    u = Function(V, name="u_output")

    # Control variable
    d = pyadjoint.AdjFloat(constant)

    assign(d, u)

    c = 0.3
    error = ufl.inner(u - c, u - c) * ufl.inner(u - c, u - c) * ufl.dx(domain=mesh)

    J = assemble_scalar(error)

    control = pyadjoint.Control(d)
    Jh = pyadjoint.ReducedFunctional(J, control)

    assert np.isclose(Jh(d), (float(d) - c) ** 4)

    # Check derivative
    dJ = Jh.derivative()
    assert np.isclose(dJ, 4 * (float(d) - c) ** 3)

    # Perform taylor test
    du = pyadjoint.AdjFloat(0.1)

    # Without gradient
    Jh(d)
    min_rate = pyadjoint.taylor_test(Jh, d, du, dJdm=0)
    assert numpy.isclose(min_rate, 1.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 1.0, got {min_rate}"

    # With gradient
    Jh(d)
    min_rate = pyadjoint.taylor_test(Jh, d, du)
    assert numpy.isclose(min_rate, 2.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 2.0, got {min_rate}"

    Jh(d)
    tol = 1e-9
    opt = pyadjoint.minimize(
        Jh,
        method="BFGS",
        tol=tol,
        scale=1e10,
        options={"maxiter": 200, "disp": True},
    )
    np.testing.assert_allclose(float(opt), float(c), atol=1e-5)


def test_assign_constant_derivative():
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)

    # Target constant
    c = Constant(mesh, dolfinx.default_scalar_type(0.5))

    # Control variable
    d = pyadjoint.AdjFloat(0.2)

    # Tape the assignment
    assign(d, c)

    # Assemble a functional: J = c^2 * dx
    J_form = c**2 * ufl.dx(domain=mesh)
    J = assemble_scalar(J_form)

    control = pyadjoint.Control(d)
    Jhat = pyadjoint.ReducedFunctional(J, control)

    # Forward check: J(0.2) = 0.2^2 * 1.0 = 0.04
    assert np.isclose(Jhat(d), 0.04)

    # Gradient check: dJ/dd = 2*d * 1.0 = 2(0.2) = 0.4
    dJ = Jhat.derivative()
    assert np.isclose(dJ, 0.4)

    # Taylor test validation
    dd = pyadjoint.AdjFloat(0.1)
    min_rate = pyadjoint.taylor_test(Jhat, d, dd)
    assert np.isclose(min_rate, 2.0, rtol=1e-2, atol=1e-2)
