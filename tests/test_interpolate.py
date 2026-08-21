from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, assemble_scalar, interpolate
from dolfinx_adjoint.blocks.interpolation import InterpolationBlock

# Dynamically determine available matrix backends
petsc_options = [False]
if getattr(dolfinx, "has_petsc", False) and getattr(dolfinx, "has_petsc4py", False):
    petsc_options.append(True)


@pytest.fixture(scope="module")
def mesh_1D():
    return dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)


@pytest.fixture(scope="module")
def mesh_2D():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 7, 7)


@pytest.fixture(scope="module")
def mesh_3D():
    return dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 5, 5, 5)


# ==============================================================================
# Test 1: Algebraic Adjoint Property (<Au, v> == <u, A*v>)
# ==============================================================================


@pytest.mark.parametrize(
    "family, degree_from, degree_to",
    [
        ("Lagrange", 1, 2),
        ("DG", 0, 1),
        ("N1curl", 1, 2),
    ],
)
@pytest.mark.parametrize("use_petsc", petsc_options)
def test_interpolation_block_adjoint_property(mesh_3D, family, degree_from, degree_to, use_petsc):
    """Verifies that the TLM and Adjoint exactly satisfy the linear adjoint identity."""
    mesh = mesh_3D

    V_from = dolfinx.fem.functionspace(mesh, (family, degree_from))
    V_to = dolfinx.fem.functionspace(mesh, (family, degree_to))

    # Use modern NumPy random generator with a fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)

    u = Function(V_from)
    u.x.array[:] = rng.random(len(u.x.array))

    v = Function(V_to)
    v.x.array[:] = rng.random(len(v.x.array))

    # Initialize the block directly, passing the PETSc backend flag
    block = InterpolationBlock(u, v, petsc_mat=use_petsc)

    mat_tlm = block.prepare_evaluate_tlm([u], [u], None)
    mat_adj = block.prepare_evaluate_adj([u], [v], None)

    tlm_output = block.evaluate_tlm_component(inputs=[u], tlm_inputs=[u], block_variable=None, idx=0, prepared=mat_tlm)
    adj_output = block.evaluate_adj_component(inputs=[u], adj_inputs=[v], block_variable=None, idx=0, prepared=mat_adj)

    inner_forward = dolfinx.cpp.la.inner_product(tlm_output.x._cpp_object, v.x._cpp_object)
    inner_adjoint = dolfinx.cpp.la.inner_product(u.x._cpp_object, adj_output.x._cpp_object)

    comm = mesh.comm
    global_inner_forward = comm.allreduce(inner_forward, op=MPI.SUM)
    global_inner_adjoint = comm.allreduce(inner_adjoint, op=MPI.SUM)

    np.testing.assert_allclose(
        global_inner_forward,
        global_inner_adjoint,
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"Adjoint property failed (PETSc={use_petsc}): <Au, v> != <u, A*v>",
    )


# ==============================================================================
# Test 2: Taylor Remainder Convergence (Graph & Optimization Integration)
# ==============================================================================


@pytest.mark.parametrize("mesh_var_name", ["mesh_1D", "mesh_2D", "mesh_3D"])
@pytest.mark.parametrize("use_petsc", petsc_options)
def test_interpolation_taylor_test(mesh_var_name: str, request, use_petsc):
    """Verifies that the exposed interpolate function works with ReducedFunctional."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = request.getfixturevalue(mesh_var_name)

    V_from = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    V_to = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))

    u = Function(V_from)
    u.name = "u_control"
    u.x.array[:] = 0.2

    # Forward the backend flag to the exposed wrapper
    v = interpolate(u, V_to, petsc_mat=use_petsc)

    def u_ex(mod, x_coords):
        return x_coords[0]

    x = ufl.SpatialCoordinate(mesh)
    c = u_ex(ufl, x)
    error = ufl.inner(v - c, v - c) * ufl.inner(v - c, v - c) * ufl.dx(domain=mesh)

    J = assemble_scalar(error)

    derivative_options = {
        "riesz_representation": "L2",
        "petsc_options": {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
    }

    control = pyadjoint.Control(u, riesz_map=derivative_options)
    Jh = pyadjoint.ReducedFunctional(J, control)
    assert Jh(u) > 0

    du = Function(V_from)
    du.interpolate(lambda x_coords: np.sin(x_coords[0]))

    # --- 1. Zero-order Taylor test ---
    Jh(u)
    min_rate = pyadjoint.taylor_test(Jh, u, du, dJdm=0)
    assert np.isclose(min_rate, 1.0, rtol=1e-2, atol=1e-2)

    # --- 2. First-order Taylor test ---
    Jh(u)
    min_rate = pyadjoint.taylor_test(Jh, u, du)
    assert np.isclose(min_rate, 2.0, rtol=1e-2, atol=1e-2)

    # --- 3. Second-order Taylor test ---
    Jh(u)
    dJdm = Jh.derivative()._ad_dot(du)
    hessian = Jh.hessian(du)
    dHddu = hessian._ad_dot(du)

    min_rate = pyadjoint.taylor_test(Jh, u, du, dJdm=dJdm, Hm=dHddu)
    assert np.isclose(min_rate, 3.0, rtol=1e-3, atol=1e-3)

    pyadjoint.get_working_tape().clear_tape()
