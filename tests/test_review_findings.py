"""One test per review finding on ``dokken/shape-control``.

Each test asserts the *current* behaviour, so it turns red when the finding is
fixed -- read a failure here as "that finding is addressed, delete this test".

    python -m pytest tests/test_review_findings.py -q
    mpirun -n 2 python -m pytest tests/test_review_findings.py -q -m parallel

Finding A is that ``geometry_function_space`` does not build against DOLFINx
main at all (scifem passes the 0.12 ``IndexMap`` wrapper into a nanobind
constructor that wants the raw C++ object). Every other test needs a working
geometry space, so ``_shim`` below unwraps it for them; ``test_A`` is marked
``no_shim`` and runs against the real thing. Remove the shim once scifem is
fixed and the whole file still works, except that ``test_A`` then skips.

Grids are deliberately tiny: the file runs in ~5 s serially, ~3 s on two ranks.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

import dolfinx_adjoint as dxa
from dolfinx_adjoint.types.mesh import Mesh

COMM = MPI.COMM_WORLD
LU = {"ksp_type": "preonly", "pc_type": "lu"}

_REAL_DOFMAP = dolfinx.cpp.fem.DofMap


class _UnwrappingDofMap(_REAL_DOFMAP):
    """``cpp.fem.DofMap`` that accepts either the wrapped or the raw index map."""

    def __init__(self, layout, index_map, index_map_bs, dofmap, bs):
        super().__init__(layout, getattr(index_map, "_cpp_object", index_map), index_map_bs, dofmap, bs)


@pytest.fixture(autouse=True)
def _shim(request):
    """Work around finding A so the remaining findings can be measured."""
    import scifem.mesh

    wanted = _REAL_DOFMAP if request.node.get_closest_marker("no_shim") else _UnwrappingDofMap
    dolfinx.cpp.fem.DofMap = wanted
    scifem.mesh.dolfinx.cpp.fem.DofMap = wanted
    yield
    dolfinx.cpp.fem.DofMap = _REAL_DOFMAP
    scifem.mesh.dolfinx.cpp.fem.DofMap = _REAL_DOFMAP


def dilation(x):
    return np.vstack((x[0], x[1]))


def interior_bump(x):
    b = np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    return np.vstack((b, b))


def homogeneous_bc(V):
    mesh = V.mesh
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    return dolfinx.fem.dirichletbc(0.0, dolfinx.fem.locate_dofs_topological(V, tdim - 1, facets), V)


def directional(vec, direction, S):
    """<vec, direction>, summed over owned dofs only."""
    owned = S.dofmap.index_map.size_local * S.dofmap.index_map_bs
    return COMM.allreduce(float(np.dot(vec.x.array[:owned], direction.x.array[:owned])), op=MPI.SUM)


# --- A: the geometry function space does not build on DOLFINx 0.12 --------------

@pytest.mark.no_shim
def test_A_geometry_function_space_is_broken_on_dolfinx_main():
    """`geometry_function_space` -> scifem -> cpp.fem.DofMap, which now wants a raw IndexMap.

    `blocks/_vector.py` was updated for exactly this change (FEniCS/dolfinx#4496);
    `geometry_function_space` delegates to scifem, which was not.
    """
    mesh = dolfinx.mesh.create_unit_square(COMM, 2, 2)
    if dolfinx.__version__.startswith("0.11"):
        pytest.skip("only breaks from 0.12, where the index map became a Python wrapper")
    with pytest.raises(TypeError, match="incompatible function arguments"):
        dxa.geometry_function_space(mesh)


# --- B/C: the shape Hessian in parallel ----------------------------------------

def _hessian_and_fd(n=8, with_coefficient=False):
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(COMM, n, n)
    S = dxa.geometry_function_space(mesh)
    s = dxa.Function(S)
    dxa.move(mesh, s)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    source = ufl.sin(ufl.pi * X[0]) * ufl.cos(ufl.pi * X[1])
    problem = dxa.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        ufl.inner(source, v) * ufl.dx,
        bcs=[homogeneous_bc(V)], petsc_options=LU, petsc_options_prefix="rev_h_",
    )
    uh = problem.solve()
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx + ufl.inner(X, X) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    h = dxa.Function(S)
    h.interpolate(interior_bump)
    Jhat(s)
    Jhat.derivative()
    Hh = float(Jhat.hessian(h)._ad_dot(h))
    eps = 1e-3

    def gradient_dot(scale):
        Jhat(s._ad_add(h._ad_mul(scale)))
        return float(Jhat.derivative()._ad_dot(h))

    fd = (gradient_dot(eps) - gradient_dot(-eps)) / (2 * eps)
    Jhat(s)
    return Hh, fd, problem


@pytest.mark.parallel
def test_B_shape_hessian_is_rank_dependent():
    """The shape Hessian disagrees with a central difference of its own gradient in parallel.

    Serially the two agree to ~1e-10. On 2 ranks the Hessian comes out with the
    wrong sign and several times the magnitude. The coefficient-control Hessian
    on the same problem is rank-independent, so the shape path is the one at fault.
    """
    Hh, fd, _keep = _hessian_and_fd()
    if COMM.size == 1:
        assert abs(Hh - fd) < 1e-4 * abs(fd), f"serial disagreement {Hh} vs {fd}"
    else:
        assert abs(Hh - fd) > 1e-3, (
            f"expected a parallel discrepancy, got hessian {Hh} vs fd {fd} on {COMM.size} ranks"
        )


@pytest.mark.parallel
def test_C_the_hessian_checker_cannot_fail_here():
    """conftest's checker uses atol=1e-2 on quantities of size ~3e-4.

    So `np.isclose(Hh, fd, rtol=1e-2, atol=1e-2)` holds no matter how wrong the
    Hessian is, and the parallel breakage in test_B passes through it unseen.
    """
    Hh, fd, _keep = _hessian_and_fd()
    tolerance = 1e-2 + 1e-2 * abs(fd)
    assert max(abs(Hh), abs(fd)) < tolerance / 5, (
        f"quantities {Hh}, {fd} are no longer small next to the tolerance {tolerance}"
    )
    assert np.isclose(Hh, fd, rtol=1e-2, atol=1e-2), "the checker would now catch this"


# --- D: the guard never sees a LinearProblem's bilinear form --------------------

def _stabilised_gradient_error(weight, n=6):
    def forward(values=None, counter=[0]):
        counter[0] += 1
        pyadjoint.get_working_tape().clear_tape()
        mesh = dolfinx.mesh.create_unit_square(COMM, n, n)
        S = dxa.geometry_function_space(mesh)
        s = dxa.Function(S)
        if values is not None:
            s.x.array[:] = values
        dxa.move(mesh, s)
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        hK = ufl.CellDiameter(mesh)
        problem = dxa.LinearProblem(
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + weight * hK * ufl.inner(u, v) * ufl.dx,
            ufl.inner(dolfinx.fem.Constant(mesh, 1.0), v) * ufl.dx,
            bcs=[homogeneous_bc(V)], petsc_options=LU,
            petsc_options_prefix=f"rev_d{weight:g}_{counter[0]}_",
        )
        uh = problem.solve()
        return dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx), s, S, problem

    S0 = dxa.geometry_function_space(dolfinx.mesh.create_unit_square(COMM, n, n))
    d = dolfinx.fem.Function(S0)
    d.interpolate(dilation)
    h = d.x.array.copy()
    eps = 1e-6
    fd = (float(forward(eps * h)[0]) - float(forward(-eps * h)[0])) / (2 * eps)
    J, s, S, _keep = forward()
    hf = dxa.Function(S)
    hf.x.array[:] = h
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    adjoint = directional(Jhat.derivative(), hf, S)
    return adjoint, fd


@pytest.mark.parametrize("weight,floor", [(1.0, 1e-3), (50.0, 0.1), (500.0, 1.0)])
def test_D_celldiameter_in_the_bilinear_form_is_accepted_and_wrong(weight, floor):
    """`_register_mesh_dependency` scans `self._rhs` (= L) only; the residual is action(a,u) - L.

    The same CellDiameter in L, or in assemble_scalar, is refused.
    """
    adjoint, fd = _stabilised_gradient_error(weight)
    error = abs(adjoint - fd) / abs(fd)
    assert error > floor, f"weight {weight}: relative error {error:.2%} no longer exceeds {floor:.0%}"


def test_D_same_quantity_in_the_rhs_is_refused():
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(COMM, 4, 4)
    S = dxa.geometry_function_space(mesh)
    dxa.move(mesh, dxa.Function(S))
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    with pytest.raises(NotImplementedError, match="differentiates every geometric quantity"):
        dxa.LinearProblem(
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
            ufl.inner(ufl.CellDiameter(mesh), v) * ufl.dx,
            bcs=[homogeneous_bc(V)], petsc_options=LU, petsc_options_prefix="rev_d_rhs_",
        ).solve()


# --- E: the guard is never called from the interpolation block ------------------

def test_E_celldiameter_in_an_interpolated_expression_is_accepted_and_wrong():
    """`ExprInterpolationBlock` registers the mesh for any GeometricQuantity but never
    calls `reject_geometry_without_shape_derivative`, so the dropped term is silent."""
    def forward(values=None, mix=True):
        pyadjoint.get_working_tape().clear_tape()
        mesh = dolfinx.mesh.create_unit_square(COMM, 6, 6)
        S = dxa.geometry_function_space(mesh)
        s = dxa.Function(S)
        if values is not None:
            s.x.array[:] = values
        dxa.move(mesh, s)
        V = dolfinx.fem.functionspace(mesh, ("DG", 0))
        X = ufl.SpatialCoordinate(mesh)
        expr = ufl.CellDiameter(mesh) + X[0] if mix else X[0]
        uh = dxa.interpolate(expr, V)
        return dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx), s, S

    S0 = dxa.geometry_function_space(dolfinx.mesh.create_unit_square(COMM, 6, 6))
    d = dolfinx.fem.Function(S0)
    d.interpolate(dilation)
    h = d.x.array.copy()
    eps = 1e-6
    results = {}
    for mix in (True, False):
        fd = (float(forward(eps * h, mix)[0]) - float(forward(-eps * h, mix)[0])) / (2 * eps)
        J, s, S = forward(None, mix)
        hf = dxa.Function(S)
        hf.x.array[:] = h
        Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
        results[mix] = (directional(Jhat.derivative(), hf, S), fd)

    adjoint, fd = results[True]
    assert abs(adjoint - fd) / abs(fd) > 0.05, "the dropped CellDiameter term no longer shows"
    adjoint, fd = results[False]
    assert abs(adjoint - fd) / abs(fd) < 1e-6, "the control case (x[0] alone) should be exact"


# --- F: annotate-before-first-form is load-bearing and unchecked ----------------

def _replay_vs_rebuild(annotate_first, steps=(0.0, 0.05, 0.10, 0.05), n=6):
    def build(values, counter=[0]):
        counter[0] += 1
        pyadjoint.get_working_tape().clear_tape()
        mesh = dolfinx.mesh.create_unit_square(COMM, n, n)
        S = dxa.geometry_function_space(mesh)
        s = dxa.Function(S)
        s.x.array[:] = values
        if annotate_first:
            dxa.annotate_mesh(mesh)
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        X = ufl.SpatialCoordinate(mesh)
        problem = dxa.LinearProblem(
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
            ufl.inner(ufl.sin(ufl.pi * X[0]) * ufl.cos(ufl.pi * X[1]), v) * ufl.dx,
            bcs=[homogeneous_bc(V)], petsc_options=LU,
            petsc_options_prefix=f"rev_f{int(annotate_first)}_{counter[0]}_",
        )
        uh = problem.solve()          # posed on the mesh BEFORE move()
        dxa.move(mesh, s)
        return dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx), s, S, problem

    S0 = dxa.geometry_function_space(dolfinx.mesh.create_unit_square(COMM, n, n))
    d = dolfinx.fem.Function(S0)
    d.interpolate(dilation)
    h = d.x.array.copy()
    # every rebuild first: a clear_tape() under a live ReducedFunctional invalidates it
    rebuilt = {st: float(build(st * h)[0]) for st in steps}
    J, s, S, _keep = build(steps[0] * h)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    out = []
    for st in steps[1:]:
        f = dxa.Function(S)
        f.x.array[:] = st * h
        out.append((st, float(Jhat(f)), rebuilt[st]))
    return out


def test_F_annotate_mesh_first_is_required_and_unchecked():
    good = _replay_vs_rebuild(annotate_first=True)
    for st, replay, rebuilt in good:
        assert abs(replay - rebuilt) < 1e-11 * max(1.0, abs(rebuilt)), f"s={st}: {replay} vs {rebuilt}"

    bad = _replay_vs_rebuild(annotate_first=False)
    errors = [abs(replay / rebuilt - 1) for _, replay, rebuilt in bad]
    assert max(errors) > 0.2, f"no drift without annotate_mesh(): {errors}"


# --- G: move() under stop_annotating promotes permanently -----------------------

def test_G_move_under_stop_annotating_still_promotes():
    tape = pyadjoint.get_working_tape()
    tape.clear_tape()
    mesh = dolfinx.mesh.create_unit_square(COMM, 4, 4)
    S = dxa.geometry_function_space(mesh)
    with pyadjoint.stop_annotating():
        dxa.move(mesh, dxa.Function(S))
    assert len(tape.get_blocks()) == 0, "no block should be recorded"
    assert isinstance(mesh, Mesh), "...but the mesh was promoted anyway"

    tape.clear_tape()
    with pytest.raises(NotImplementedError, match="differentiates every geometric quantity"):
        dxa.assemble_scalar(ufl.CellDiameter(mesh) * ufl.dx)


# --- H: Problem-level mesh cache vs per-block lookup ----------------------------

def test_H_problem_mesh_cache_can_disagree_with_the_block():
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(COMM, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    problem = dxa.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        ufl.inner(dolfinx.fem.Constant(mesh, 1.0), v) * ufl.dx,
        bcs=[homogeneous_bc(V)], petsc_options=LU, petsc_options_prefix="rev_h_cache_",
    )
    problem.solve()
    templates, _seeds, _state = problem._get_or_build_tlm_rhs_templates()   # built while plain
    dxa.move(mesh, dxa.Function(dxa.geometry_function_space(mesh)))
    problem.solve()                                                          # mesh now moved
    block = pyadjoint.get_working_tape().get_blocks()[-1]

    assert any(isinstance(bv.output, Mesh) for bv in block.get_dependencies()), \
        "the block registers the mesh"
    assert not any(isinstance(k, Mesh) for k in templates), \
        "the Problem's TLM templates do not contain it"
    assert problem._get_shape_mesh() is None, "the None was cached"
    # prepare_evaluate_tlm then does templates.get(mesh) -> None and `continue`s


# --- I: MoveBlock hands the same buffer to both dependencies --------------------

def test_I_moveblock_aliases_one_adjoint_buffer():
    """Reachable from the second move() on, where both dependencies are control-relevant."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(COMM, 5, 5)
    S = dxa.geometry_function_space(mesh)
    s1, s2 = dxa.Function(S, name="s1"), dxa.Function(S, name="s2")
    dxa.move(mesh, s1)
    X = ufl.SpatialCoordinate(mesh)
    J = dxa.assemble_scalar(ufl.inner(X, X) * ufl.dx)
    dxa.move(mesh, s2)
    J = J + dxa.assemble_scalar(ufl.inner(X, X) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, [pyadjoint.Control(s1), pyadjoint.Control(s2)])
    Jhat.derivative()

    second = [b for b in pyadjoint.get_working_tape().get_blocks()
              if type(b).__name__ == "MoveBlock"][1]
    mesh_bv, displacement_bv = second.get_dependencies()
    assert mesh_bv.adj_value is displacement_bv.adj_value, "one buffer, two block variables"

    before = float(np.abs(displacement_bv.adj_value.array).max())
    mesh_bv.add_adj_output(displacement_bv.adj_value)
    after = float(np.abs(displacement_bv.adj_value.array).max())
    assert after != before, "accumulating into one wrote through into the other"


# --- J: apply_displacement never scatters ---------------------------------------

def _tear(mesh):
    index_map = mesh.geometry.index_map()
    n_owned = index_map.size_local
    x = mesh.geometry.x
    probe = dolfinx.la.vector(index_map, 3)
    probe.array[:] = 0.0
    probe.array[: n_owned * 3] = x[:n_owned, :].reshape(-1)
    probe.scatter_forward()
    local = np.abs(probe.array[n_owned * 3:] - x[n_owned:, :].reshape(-1)).max() \
        if x.shape[0] > n_owned else 0.0
    return COMM.allreduce(float(local), op=MPI.MAX)


@pytest.mark.parallel
def test_J_unscattered_displacement_tears_the_mesh():
    if COMM.size == 1:
        pytest.skip("needs more than one rank: there are no ghost nodes in serial")
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(COMM, 8, 8)
    S = dxa.geometry_function_space(mesh)
    s = dxa.Function(S)
    n_owned = S.dofmap.index_map.size_local * S.dofmap.index_map_bs
    s.x.array[:] = 0.0
    s.x.array[:n_owned] = 0.1          # owned dofs only, no scatter_forward
    assert _tear(mesh) == 0.0
    dxa.move(mesh, s)
    assert _tear(mesh) > 1e-12, "the mesh should be torn at the partition boundary"

    pyadjoint.get_working_tape().clear_tape()
    mesh2 = dolfinx.mesh.create_unit_square(COMM, 8, 8)
    S2 = dxa.geometry_function_space(mesh2)
    s2 = dxa.Function(S2)
    n2 = S2.dofmap.index_map.size_local * S2.dofmap.index_map_bs
    s2.x.array[:] = 0.0
    s2.x.array[:n2] = 0.1
    s2.x.scatter_forward()             # the one missing line
    dxa.move(mesh2, s2)
    assert _tear(mesh2) == 0.0, "with a scatter first, the mesh stays consistent"


# --- K: what move()'s space check does and does not distinguish -----------------

def test_K_move_accepts_a_displacement_from_a_different_mesh():
    pyadjoint.get_working_tape().clear_tape()
    mesh_a = dolfinx.mesh.create_unit_square(COMM, 5, 5)
    mesh_b = dolfinx.mesh.create_rectangle(
        COMM, [np.array([0.0, 0.0]), np.array([2.0, 3.0])], [5, 5]
    )
    S_b = dxa.geometry_function_space(mesh_b)
    s_b = dxa.Function(S_b)
    s_b.x.array[:] = 0.1
    before = mesh_a.geometry.x.copy()
    dxa.move(mesh_a, s_b)              # accepted: same element, same block size
    assert not np.allclose(mesh_a.geometry.x, before), "mesh A was moved by mesh B's field"

    X = ufl.SpatialCoordinate(mesh_a)
    J = dxa.assemble_scalar(ufl.inner(X, X) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s_b))
    gradient = Jhat.derivative()       # a gradient in mesh B's space, assembled over mesh A
    assert np.abs(gradient.x.array).max() > 0.0


def test_K_a_plain_P1_lookalike_is_accepted_but_happens_to_be_correct():
    """The dof-ordering hazard does not bite: the geometry dofmap and a fresh
    vector-P1 dofmap have identical dof coordinates, in order."""
    mesh = dolfinx.mesh.create_unit_square(COMM, 5, 5)
    S = dxa.geometry_function_space(mesh)
    W = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
    assert W.element == S.element and W.dofmap.index_map_bs == S.dofmap.index_map_bs, \
        "move() cannot tell them apart"
    assert np.allclose(S.tabulate_dof_coordinates(), W.tabulate_dof_coordinates()), \
        "but they order their dofs the same way, so the accepted look-alike is harmless"


# --- L: retracted -- CellVolume/FacetArea on non-affine cells ---------------------

@pytest.mark.parametrize("quantity,measure", [("CellVolume", "dx"), ("FacetArea", "ds")])
def test_L_ffcx_refuses_these_on_non_affine_cells_anyway(quantity, measure):
    """Retraction of an earlier finding.

    UFL lowers CellVolume/FacetArea only on an affine simplex domain, so the
    allowlist's stated justification does not hold elsewhere. But FFCx cannot
    generate code for the un-lowered terminal *in the forward form either*, so
    no wrong gradient is reachable: the form never compiles.
    """
    mesh = dolfinx.mesh.create_unit_square(COMM, 4, 4, dolfinx.mesh.CellType.quadrilateral)
    assert not mesh.ufl_domain().is_piecewise_linear_simplex_domain()
    form = getattr(ufl, quantity)(mesh) * getattr(ufl, measure)
    with pytest.raises(RuntimeError, match="Not handled"):
        dolfinx.fem.form(form)
