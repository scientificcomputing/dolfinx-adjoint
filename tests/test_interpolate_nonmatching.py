from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl
from dolfinx.fem import functionspace
from dolfinx.mesh import create_unit_cube, create_unit_square

from dolfinx_adjoint import Function, assemble_scalar, interpolate_nonmatching


def _run_adjoint_and_taylor_test(mesh_from, mesh_to, use_petsc):
    """Helper function to test Adjoint symmetry and Taylor convergence."""
    pyadjoint.get_working_tape().clear_tape()

    V_from = functionspace(mesh_from, ("Lagrange", 1))
    V_to = functionspace(mesh_to, ("Lagrange", 1))

    rng = np.random.default_rng(seed=42)

    # 1. Setup Control
    u = Function(V_from, name="u_control")
    u.x.array[:] = rng.random(len(u.x.array))
    u.x.scatter_forward()

    # 2. Evaluate forward interpolation and automatically record to tape!
    v = interpolate_nonmatching(u, V_to, petsc_mat=use_petsc)

    # ==========================================
    # TEST 1: Exact Algebraic Adjoint Property
    # ==========================================
    # Extract the automatically created block from the tape
    tape = pyadjoint.get_working_tape()
    block = tape.get_blocks()[-1]

    # Forward Tangent pass (J * u)
    mat_tlm = block.prepare_evaluate_tlm([u], [u], None)
    tlm_output = block.evaluate_tlm_component(inputs=[u], tlm_inputs=[u], block_variable=None, idx=0, prepared=mat_tlm)

    # Reverse Adjoint pass (J^T * v). A real upstream block passes adj_inputs as raw
    # vectors during tape.evaluate_adj(), not Functions, so mirror that with v.x here.
    mat_adj = block.prepare_evaluate_adj([u], [v.x], None)
    adj_output = block.evaluate_adj_component(
        inputs=[u], adj_inputs=[v.x], block_variable=None, idx=0, prepared=mat_adj
    )

    # inner(Ju, v) == inner(u, J^T v)
    inner_forward = dolfinx.cpp.la.inner_product(tlm_output.x._cpp_object, v.x._cpp_object)
    inner_adjoint = dolfinx.cpp.la.inner_product(u.x._cpp_object, adj_output._cpp_object)

    np.testing.assert_allclose(
        inner_forward,
        inner_adjoint,
        rtol=1e-10,
        atol=1e-10,
        err_msg=f"Adjoint property failed for Nonmatching Interpolation (PETSc={use_petsc})",
    )

    # ==========================================
    # TEST 2: PyAdjoint Taylor Test
    # ==========================================
    target = Function(V_to)
    target.interpolate(lambda x: x[0] + x[1])

    # Objective J(u) = \int (v - target)^2 dx
    error = ufl.inner(ufl.exp(v) - target, ufl.exp(v) - target) * ufl.dx(domain=mesh_to)
    J = assemble_scalar(error)

    control = pyadjoint.Control(u)
    Jh = pyadjoint.ReducedFunctional(J, control)

    du = Function(V_from)
    du.interpolate(lambda x: np.sin(x[0]) * np.cos(x[1]))

    # Gradient check: expect rate = 2.0
    Jh(u)
    min_rate = pyadjoint.taylor_test(Jh, u, du)
    assert np.isclose(min_rate, 2.0, rtol=1e-2, atol=1e-2)

    # Hessian check: expect rate = 3.0 (Linear operator Hessian is 0, so Taylor series is exact)
    Jh(u)
    dJdm = Jh.derivative()._ad_dot(du)
    dHddu = Jh.hessian(du)._ad_dot(du)
    min_rate_hess = pyadjoint.taylor_test(Jh, u, du, dJdm=dJdm, Hm=dHddu)
    assert np.isclose(min_rate_hess, 3.0, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("use_petsc", [False, True])
def test_nonmatching_3D_to_2D(use_petsc):
    """Tests interpolating from a 3D volume to a 2D surface."""
    mesh_from = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4)
    mesh_to = create_unit_square(MPI.COMM_WORLD, 8, 8)

    _run_adjoint_and_taylor_test(mesh_from, mesh_to, use_petsc)


@pytest.mark.parametrize("use_petsc", [False, True])
def test_nonmatching_2D_to_2D(use_petsc):
    """Tests interpolating from a 2D surface to a 2D surface."""
    mesh_from = create_unit_square(MPI.COMM_WORLD, 8, 8)
    mesh_to = create_unit_square(MPI.COMM_WORLD, 4, 4)

    _run_adjoint_and_taylor_test(mesh_from, mesh_to, use_petsc)
