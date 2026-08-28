import typing

from mpi4py import MPI

import basix
import dolfinx
import numpy
import numpy as np
import pyadjoint
import pytest
import ufl
from checkpoint_schedules import Revolve

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


def test_assign_linear_combination(mesh_1D):
    V = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 1))
    f = Function(V, name="f")
    f.interpolate(lambda x: 2.0 * x[0])
    g = Function(V, name="g")
    g.interpolate(lambda x: 3.0 * x[0] ** 2)
    u = Function(V, name="u")

    assign(3 * f - g, u)

    J = assemble_scalar(u**2 * ufl.dx)
    rf = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))
    h = Function(V)
    rng = np.random.default_rng(seed=42)
    num_dofs_local = (V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts) * V.dofmap.index_map_bs
    rand = rng.random(size=num_dofs_local, dtype=h.dtype)
    h.x.array[:] = rand
    h.x.scatter_forward()
    assert pyadjoint.taylor_test(rf, f, h) > 1.9

    rf2 = pyadjoint.ReducedFunctional(J, pyadjoint.Control(g))
    assert pyadjoint.taylor_test(rf2, g, h) > 1.9


def test_assign_lincomb_real_space(mesh_1D):
    r_el = basix.ufl.real_element(mesh_1D.basix_cell(), value_shape=())
    R = dolfinx.fem.functionspace(mesh_1D, r_el)
    r = Function(R, name="r")
    r.x.array[0] = 0.2

    V = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 2))
    v = Function(V, name="u")
    v.interpolate(lambda x: 3.0 * x[0] ** 2)

    z = -2 * v + 4 * r * v
    u = Function(V, name="u_output")
    assign(z, u)

    J = assemble_scalar(u**2 * ufl.dx)
    rf = pyadjoint.ReducedFunctional(J, pyadjoint.Control(r))
    h = Function(R)
    rng = np.random.default_rng(seed=42)
    num_dofs_local = (R.dofmap.index_map.size_local + R.dofmap.index_map.num_ghosts) * R.dofmap.index_map_bs
    rand = rng.random(size=num_dofs_local, dtype=h.dtype)
    h.x.array[:] = rand
    h.x.scatter_forward()
    assert pyadjoint.taylor_test(rf, r, h) > 1.9

    rf2 = pyadjoint.ReducedFunctional(J, pyadjoint.Control(v))
    hv = Function(V)
    num_dofs_local = (V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts) * V.dofmap.index_map_bs
    rand = rng.random(size=num_dofs_local, dtype=hv.dtype)
    hv.x.array[:] = rand
    hv.x.scatter_forward()
    assert pyadjoint.taylor_test(rf2, v, hv) > 1.9


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


def test_assign_wrong_function_space(mesh_1D):
    """Test that assigning a single function from a different space fails."""
    # Create two different function spaces (different DoF counts)
    V1 = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 1))
    V2 = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 2))

    f_wrong = Function(V2, name="f_wrong")
    u = Function(V1, name="u")

    # The assignment should fail due to mismatched array lengths/spaces
    with pytest.raises(ValueError):
        assign(f_wrong, u)


def test_assign_linear_combination_wrong_function_space(mesh_1D):
    """Test that assigning a linear combination with an incompatible term fails."""
    V1 = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 1))
    V2 = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 2))

    f = Function(V1, name="f")
    g_wrong = Function(V2, name="g_wrong")
    u = Function(V1, name="u")

    # 3 * f is valid for u, but subtracting g_wrong should trigger an error
    with pytest.raises(ValueError):
        assign(3 * f - g_wrong, u)


@pytest.mark.xfail(strict=True, reason="Non-linear UFL expressions cannot be assigned directly via array operations.")
def test_assign_non_linear_expression(mesh_1D):
    """Test that assigning a non-linear expression fails."""
    V = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 1))

    f = Function(V, name="f")
    u = Function(V, name="u")

    # Create a non-linear expression (e.g., f squared)
    non_linear_expr = f**2

    # This should raise an error during the `extract_linear_combination` phase
    # since `f**2` is a ufl.Power or ufl.Product, not a linear combination.
    assign(non_linear_expr, u)


def test_assign_real_function_equals_constant(mesh_1D):
    """Test that assigning a Real space function behaves identically to a scalar constant,
    both in the forward pass and the adjoint pass."""

    # Standard spatial space and Real (global scalar) space
    V = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 1))

    # Note: Depending on your exact Basix/DOLFINx version, this might be ("R", 0)
    R = dolfinx.fem.functionspace(mesh_1D, basix.ufl.real_element(mesh_1D.basix_cell(), value_shape=()))

    target_real = Function(V, name="target_real")
    target_const = Function(V, name="target_const")

    val = 4.2
    # Assign using a Function from a Real space
    r_func = Function(R, name="r_func")
    r_func.x.array[:] = val
    assign(r_func, target_real)

    # Assign using a raw float (which your block converts to AdjFloat)
    assign(val, target_const)

    # The resulting degrees of freedom should be exactly identical
    np.testing.assert_allclose(
        target_real.x.array,
        target_const.x.array,
        err_msg="Forward assignment of Real function and constant do not match.",
    )

    J = assemble_scalar(target_real**2 * ufl.dx)

    # Test the sensitivity with respect to the Real function
    rf = pyadjoint.ReducedFunctional(J, pyadjoint.Control(r_func))

    # Create a perturbation direction in the Real space
    h = Function(R, name="h")
    h.x.array[:] = 0.75

    # Verify the adjoint derivative is correct (should converge at rate ~ 2.0)
    convergence_rate = pyadjoint.taylor_test(rf, r_func, h)
    assert convergence_rate > 1.9, f"Taylor test failed with rate {convergence_rate}"


def test_recompute_does_not_alias_state_across_timesteps(mesh_1D):
    V = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 1))

    def run(schedule):
        pyadjoint.get_working_tape().clear_tape()
        tape = pyadjoint.Tape()
        pyadjoint.set_working_tape(tape)
        if schedule is not None:
            tape.enable_checkpointing(schedule)

        m = Function(V, name="control")
        m.interpolate(lambda x: 1.0 + x[0])

        controls = []
        for i in range(5):
            c = Function(V, name=f"control_{i}")
            c.interpolate(lambda x, i=i: 1.0 + 0.1 * (i + 1) * x[0])
            controls.append(c)

        prev = Function(V, name="prev")
        assign(0.0, prev)

        J = 0.0
        for i in tape.timestepper(iter(range(5))):
            state = Function(V, name="state")
            assign(prev + controls[i], state)
            J = J + assemble_scalar(state * state * ufl.dx)
            assign(state, prev)

        rf = pyadjoint.ReducedFunctional(J, [pyadjoint.Control(c) for c in controls])
        return rf, controls

    # A tape that has had checkpointing enabled keeps eagerly checkpointing outputs even after
    # clear_tape() (see tests/test_checkpointing.py's isolated_tape fixture docstring), so the
    # Revolve-enabled tape built by run() below must not leak out as the working tape once this
    # test returns -- restore whatever was active beforehand.
    previous_tape = pyadjoint.get_working_tape()
    try:
        rf_plain, controls_plain = run(None)
        rf_plain(controls_plain)
        grad_plain = [np.copy(g.x.array) for g in rf_plain.derivative()]

        rf_ckpt, controls_ckpt = run(Revolve(5, 2))
        rf_ckpt(controls_ckpt)
        grad_ckpt = [np.copy(g.x.array) for g in rf_ckpt.derivative()]

        for i, (a, e) in enumerate(zip(grad_ckpt, grad_plain, strict=True)):
            np.testing.assert_allclose(a, e, rtol=1e-12, atol=1e-14, err_msg=f"control {i}")
    finally:
        pyadjoint.set_working_tape(previous_tape)
