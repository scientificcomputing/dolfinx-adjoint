import gc

from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, assemble_scalar, interpolate
from dolfinx_adjoint.blocks.interpolation import (
    _CACHE_KEYS_BY_SPACE_ID,
    _INTERPOLATION_MATRIX_CACHE,
    InterpolationBlock,
    _get_interpolation_matrix,
)

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


# ==============================================================================
# Test 3: Interpolation Matrix Cache
# ==============================================================================


def test_interpolation_matrix_cache_reused_for_same_spaces(mesh_1D):
    """The same (space_from, space_to) pair should reuse one assembled matrix."""
    V_from = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 1))
    V_to = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 2))

    mat1 = _get_interpolation_matrix(V_from, V_to)
    mat2 = _get_interpolation_matrix(V_from, V_to)
    assert mat1 is mat2

    # A different space pair must not share the cached matrix.
    V_other = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 3))
    mat3 = _get_interpolation_matrix(V_from, V_other)
    assert mat3 is not mat1


def test_interpolation_matrix_cache_purged_when_space_is_garbage_collected(mesh_1D):
    """Regression test for the id()-collision bug: a cache entry keyed on
    id(space) must be dropped once that space is garbage collected, otherwise
    a later, unrelated FunctionSpace object that happens to reuse the same id
    could silently be served a stale matrix built for a completely different
    pair of spaces.
    """
    V_to = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 2))

    def build_and_return_id():
        # V_from only lives inside this function, so it is eligible for
        # collection as soon as it returns.
        v_from = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 1))
        _get_interpolation_matrix(v_from, V_to)
        return id(v_from)

    space_id = build_and_return_id()
    gc.collect()

    assert space_id not in _CACHE_KEYS_BY_SPACE_ID
    assert all(key[0] != space_id for key in _INTERPOLATION_MATRIX_CACHE)


def test_interpolation_matrix_cache_does_not_leak_across_transient_spaces(mesh_1D):
    """The cache must not grow without bound as short-lived FunctionSpaces
    (e.g. created once per optimization iteration) are used and discarded.
    """
    baseline = len(_INTERPOLATION_MATRIX_CACHE)

    for _ in range(20):
        v_from = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 1))
        v_to = dolfinx.fem.functionspace(mesh_1D, ("Lagrange", 2))
        _get_interpolation_matrix(v_from, v_to)

    gc.collect()
    assert len(_INTERPOLATION_MATRIX_CACHE) <= baseline + 1
