from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, assemble_scalar, dirichletbc
from dolfinx_adjoint.solvers import LinearProblem, NonlinearProblem

direct_solve = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "ksp_error_if_not_converged": True,
    "pc_factor_mat_solver_type": "mumps",
}


@pytest.fixture(scope="module")
def mesh_2D():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 27, 32)


@pytest.mark.parametrize("use_mixed_space", [True, False])
def test_solver(use_mixed_space: bool, mesh_2D):
    pyadjoint.get_working_tape().clear_tape()
    mesh = mesh_2D
    el_u = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    el_p = basix.ufl.element("P", mesh.basix_cell(), 1)
    V = dolfinx.fem.functionspace(mesh, el_u)
    Q = dolfinx.fem.functionspace(mesh, el_p)
    dx = ufl.Measure("dx", domain=mesh)

    def a00(u, v):
        return ufl.inner(ufl.grad(u), ufl.grad(v)) * dx

    def a01(p, v):
        return ufl.inner(p, ufl.div(v)) * dx

    def a10(q, u):
        return ufl.inner(q, ufl.div(u)) * dx

    def L0(f, v):
        return ufl.inner(f, v) * dx

    def L1(mesh, q):
        return dolfinx.fem.Constant(mesh, 0.0) * q * dx

    Z = dolfinx.fem.functionspace(mesh, ("DG", 0, (mesh.geometry.dim,)))
    f = Function(Z, name="control")
    f.interpolate(lambda x: (np.sin(x[0]), -2 * x[1]))

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
    with pyadjoint.stop_annotating():
        baseline = 100
        d = Function(Z)
        d.interpolate(lambda x: (baseline * x[0], 2 * baseline * x[1]))

        e = Function(Z)
        e.interpolate(lambda x: (10 * baseline * np.sin(x[1]), 3 * baseline * x[0] ** 2))
        min_rate = pyadjoint.taylor_test(Jh, d, e, dJdm=0)
        assert np.isclose(min_rate, 1.0, rtol=1e-1, atol=1e-1), (
            f"Expected convergence rate close to 1.0, got {min_rate}"
        )

        Jh.derivative()
        min_rate = pyadjoint.taylor_test(Jh, d, e)
        assert np.isclose(min_rate, 2.0, rtol=1e-2, atol=1e-2), (
            f"Expected convergence rate close to 2.0, got {min_rate}"
        )

        # This is a saddle-point (velocity/pressure) system -- the standard rate-3
        # taylor_test's cubic remainder is not a robust check on it (see
        # assert_hessian_matches_finite_difference's own docstring in conftest.py).
        assert_hessian_matches_finite_difference(Jh, d, e)

        z = Function(Z)
        z.interpolate(
            lambda x: (5 * baseline * np.sin(x[1]) + 0.3 * baseline * x[1] * x[0], -2 * baseline * (x[0]) + (1 - x[1]))
        )
        f = Function(Z)
        f.interpolate(
            lambda x: (0.8 * baseline * x[1] ** 2, 2 * baseline * x[0] ** 2)
        )  # NOTE: Has to be divergence free
        f.x.scatter_forward()
        assert_hessian_matches_finite_difference(Jh, z, f)


@pytest.mark.parametrize("use_mixed_space", [True, False])
def test_nonlinear_solver(use_mixed_space: bool, mesh_2D, assert_hessian_matches_finite_difference):
    """As ``test_solver``, but for a blocked ``NonlinearProblem``: a Navier-Stokes-like
    velocity/pressure system with a viscosity control, genuinely nonlinear in the state
    via the convective term, exercising the same forward/adjoint/TLM/Hessian paths as
    ``test_solver`` for the nonlinear (rather than linear) blocked residual.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = mesh_2D
    el_u = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    el_p = basix.ufl.element("P", mesh.basix_cell(), 1)
    V = dolfinx.fem.functionspace(mesh, el_u)
    Q = dolfinx.fem.functionspace(mesh, el_p)
    Z = dolfinx.fem.functionspace(mesh, ("DG", 0))
    dx = ufl.Measure("dx", domain=mesh)

    mu = Function(Z, name="viscosity")
    baseline = 0.08
    mu.interpolate(lambda x: baseline + 0.5 * baseline * np.sin(np.pi * x[0]))

    uh, ph = Function(V, name="velocity"), Function(Q, name="pressure")

    # A moderate, non-conservative body force: large enough that the Taylor remainders
    # stay clear of round-off, small enough that the Newton solve below converges
    # reliably (the convective term is quadratic in the state).
    x = ufl.SpatialCoordinate(mesh)
    f = 10.0 * ufl.as_vector((ufl.sin(ufl.pi * x[1]), ufl.cos(ufl.pi * x[0])))

    if use_mixed_space:
        W = ufl.MixedFunctionSpace(*[V, Q])
        v, q = ufl.TestFunctions(W)
        F = ufl.extract_blocks(
            ufl.inner(mu * ufl.grad(uh), ufl.grad(v)) * dx
            + ufl.inner(ufl.dot(ufl.grad(uh), uh), v) * dx
            + ufl.inner(ph, ufl.div(v)) * dx
            - ufl.inner(f, v) * dx
            + ufl.inner(q, ufl.div(uh)) * dx
        )
    else:
        v, q = ufl.TestFunction(V), ufl.TestFunction(Q)
        F = [
            ufl.inner(mu * ufl.grad(uh), ufl.grad(v)) * dx
            + ufl.inner(ufl.dot(ufl.grad(uh), uh), v) * dx
            + ufl.inner(ph, ufl.div(v)) * dx
            - ufl.inner(f, v) * dx,
            ufl.inner(q, ufl.div(uh)) * dx,
        ]

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc_val = dolfinx.fem.Constant(mesh, np.zeros((mesh.geometry.dim,), dtype=dolfinx.default_scalar_type))
    bc = dolfinx.fem.dirichletbc(bc_val, boundary_dofs, V)

    forward_options = {
        "snes_type": "newtonls",
        "snes_error_if_not_converged": True,
        "snes_atol": 1e-9,
        "snes_rtol": 1e-9,
        "snes_monitor": None,
    }
    forward_options.update(direct_solve)
    problem = NonlinearProblem(
        F,
        u=[uh, ph],
        bcs=[bc],
        # Distinct from the default prefix: this fixture's tight snes_atol/rtol/stol
        # must not leak into -- or collide with -- another NonlinearProblem elsewhere
        # in the suite that happens to use the default prefix.
        petsc_options_prefix=f"dxa_blocked_nonlinear_test_{use_mixed_space}_",
        petsc_options=forward_options,
        adjoint_petsc_options=direct_solve,
        tlm_petsc_options=direct_solve,
    )
    problem.solve()

    # Quartic in the state and with no constant offset, for the same round-off-avoidance
    # reason as test_solver's objective.
    J = assemble_scalar(ufl.inner(uh, uh) ** 2 * dx)

    control = pyadjoint.Control(mu)
    Jh = pyadjoint.ReducedFunctional(J, control)
    baseline = 0.08
    with pyadjoint.stop_annotating():
        d = Function(Z)
        d.interpolate(lambda x: 2 * baseline + 0.3 * baseline * np.cos(np.pi * x[1]))
        e = Function(Z)
        e.interpolate(lambda x: 0.5 * baseline * np.sin(3 * x[0]))
        assert np.min(d.x.array - e.x.array) > 0.0, "Taylor test perturbation must not violate positivity of viscosity"
        min_rate = pyadjoint.taylor_test(Jh, d, e, dJdm=0)
        assert np.isclose(min_rate, 1.0, rtol=1e-1, atol=1e-1), (
            f"Expected convergence rate close to 1.0, got {min_rate}"
        )

        Jh.derivative()
        min_rate = pyadjoint.taylor_test(Jh, d, e)
        assert np.isclose(min_rate, 2.0, rtol=1e-1, atol=1e-1), (
            f"Expected convergence rate close to 2.0, got {min_rate}"
        )

        # This is a saddle-point (velocity/pressure) system -- the standard rate-3
        # taylor_test's cubic remainder is not a robust check on it (see
        # assert_hessian_matches_finite_difference's own docstring in conftest.py).
        assert_hessian_matches_finite_difference(Jh, d, e)

        # A second, independent evaluation point/direction: a cached-but-unrefreshed
        # adjoint/TLM/Hessian operator (see tests/test_tlm_update.py) could pass the
        # check above yet still be silently wrong here.
        mu2 = Function(Z)
        mu2.interpolate(lambda x: 3 * baseline + 0.8 * baseline * np.sin(x[1]))
        h2 = Function(Z)
        h2.interpolate(lambda x: 0.4 * baseline + 0.9 * baseline * np.cos(x[0]))  # NOTE: min(mu2)-max(h2) > 0
        assert np.min(mu2.x.array - h2.x.array) > 0.0, (
            "Taylor test perturbation must not violate positivity of viscosity"
        )
        assert_hessian_matches_finite_difference(Jh, mu2, h2)


def test_blocked_dirichletbc_control_on_second_block(mesh_2D):
    """Regression test for the ``HomogeneousBCLinearProblem`` block-offset bug: `bc.set()`
    on the monolithic blocked vector has no block-offset translation, so a bc constraining
    any block other than the first previously landed on the wrong block's dofs (see
    petsc_utils.py). Also exercises ``DirichletBCBlock``'s adjoint/TLM/Hessian machinery
    for a bc on a *non-first* block.

    Two decoupled scalar Poisson problems share one blocked ``LinearProblem``: an ordinary
    RHS control `f` drives the first block, and a Dirichlet bc control `g` constrains the
    whole boundary of the *second*. Being decoupled, this also directly regresses the
    root-caused EMI-demo bug (a tracked-but-irrelevant bc on another block corrupting an
    unrelated control's gradient): `f`'s own Taylor rate must be unaffected by `g`'s bc
    merely being on the tape.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = mesh_2D
    V0 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    V1 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    dx = ufl.Measure("dx", domain=mesh)

    u0, u1 = ufl.TrialFunction(V0), ufl.TrialFunction(V1)
    v0, v1 = ufl.TestFunction(V0), ufl.TestFunction(V1)

    f = Function(V0, name="control")
    f.interpolate(lambda x: np.sin(np.pi * x[0]) * x[1])

    a = [
        [ufl.inner(ufl.grad(u0), ufl.grad(v0)) * dx, None],
        [None, ufl.inner(ufl.grad(u1), ufl.grad(v1)) * dx],
    ]
    L = [ufl.inner(f, v0) * dx, ufl.inner(dolfinx.fem.Constant(mesh, 0.0), v1) * dx]

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)

    # V0's own Poisson block has no coupling to V1 and no other boundary condition of its
    # own -- a plain, untracked, homogeneous bc pins it down (otherwise it is a singular
    # pure-Neumann problem, exactly the ill-posedness the boundary-control root-cause
    # investigation (see AGENTS.md-adjacent dxa notes) already found produces a spuriously
    # huge, non-smooth "solution" from a direct LU factorization).
    boundary_dofs_0 = dolfinx.fem.locate_dofs_topological(V0, mesh.topology.dim - 1, boundary_facets)
    bc0 = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh, 0.0), boundary_dofs_0, V0)

    boundary_dofs = dolfinx.fem.locate_dofs_topological(V1, mesh.topology.dim - 1, boundary_facets)

    g = Function(V1, name="bc_value")
    g.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = dirichletbc(g, boundary_dofs, V=V1)

    uh0, uh1 = Function(V0, name="state0"), Function(V1, name="state1")
    problem = LinearProblem(
        a,
        L,
        u=[uh0, uh1],
        bcs=[bc0, bc],
        petsc_options=direct_solve,
        adjoint_petsc_options=direct_solve,
        tlm_petsc_options=direct_solve,
    )
    problem.solve()

    J = assemble_scalar(ufl.inner(uh0, uh0) * dx + ufl.inner(uh1, uh1) * dx)

    # Both f (an ordinary RHS control) and g (the bc control) enter this *linear* PDE
    # linearly with a homogeneous forward problem, so J (quadratic in the state) is
    # *exactly* quadratic in each of them -- the standard 0th-order (dJdm=0) Taylor check
    # is confounded by the (non-negligible) quadratic term unless h is tiny enough to be
    # swamped by solver/round-off noise (the same phenomenon documented for every other
    # bc-control Taylor test in this session/suite). Only the gradient-corrected rate-2
    # check is a reliable gradient signal here; skip the rate-1 check for both.

    # f's Taylor rate must be unaffected by g's bc merely being tracked on the tape.
    Jh_f = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))
    f0 = Function(V0)
    f0.x.array[:] = f.x.array
    df = Function(V0)
    df.interpolate(lambda x: 50.0 * np.cos(np.pi * x[1]))
    Jh_f(f0)
    min_rate = pyadjoint.taylor_test(Jh_f, f0, df)
    assert np.isclose(min_rate, 2.0, rtol=1e-1, atol=1e-1), f"Expected convergence rate close to 2.0, got {min_rate}"

    # g's own bc-control gradient, block index 1 -- the block the offset bug corrupted.
    Jh_g = pyadjoint.ReducedFunctional(J, pyadjoint.Control(g))
    g0 = Function(V1)
    g0.x.array[:] = g.x.array
    dg = Function(V1)
    dg.interpolate(lambda x: 30.0 * np.cos(np.pi * x[0]) * x[1])
    Jh_g(g0)
    min_rate = pyadjoint.taylor_test(Jh_g, g0, dg)
    assert np.isclose(min_rate, 2.0, rtol=1e-1, atol=1e-1), f"Expected convergence rate close to 2.0, got {min_rate}"

    # Hessian: J is exactly quadratic in a Dirichlet bc control entering a linear PDE
    # (see tests/test_dirichlet_bc.py's own bc-control tests for the same phenomenon),
    # so the meaningful check is a direct exact-quadratic remainder, not the standard
    # rate-3 taylor_test (which only measures floating-point noise here).
    Jh_g(g0)
    J0 = float(Jh_g(g0))
    dJdm0 = Jh_g.derivative()._ad_dot(dg)
    Hm0 = Jh_g.hessian(dg)._ad_dot(dg)
    for scale in [1.0, 0.3, 0.1, 0.01]:
        gp = Function(V1)
        gp.x.array[:] = g0.x.array + scale * dg.x.array
        Jp = float(Jh_g(gp))
        predicted = J0 + scale * dJdm0 + 0.5 * scale**2 * Hm0
        assert np.isclose(Jp, predicted, rtol=0, atol=1e-2), (
            f"scale={scale}: exact 2nd-order remainder {Jp - predicted:.3e} too large"
        )


def test_blocked_dirichletbc_control_with_entity_maps(mesh_2D):
    """As ``test_blocked_dirichletbc_control_on_second_block``, but each block's state
    lives on its own submesh of a shared parent mesh (``dolfinx.mesh.create_submesh``),
    requiring ``entity_maps`` -- the EMI-style shape that motivated this feature.
    Regression test that ``entity_maps`` threads correctly through the *internal*
    boundary-reaction/TLM templates dxa builds for a bc control (see
    ``solvers.py::_build_adjoint_reaction_template``), not just the user-supplied a/L
    forms, which -- being purely single-mesh integrals here -- would compile even if
    that internal threading were broken.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = mesh_2D
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    midpoints = dolfinx.mesh.compute_midpoints(mesh, tdim, np.arange(num_cells, dtype=np.int32))
    left_cells = np.arange(num_cells, dtype=np.int32)[midpoints[:, 0] < 0.5]
    right_cells = np.arange(num_cells, dtype=np.int32)[midpoints[:, 0] >= 0.5]

    mesh0, cell_map0, _, _ = dolfinx.mesh.create_submesh(mesh, tdim, left_cells)
    mesh1, cell_map1, _, _ = dolfinx.mesh.create_submesh(mesh, tdim, right_cells)
    entity_maps = [cell_map0, cell_map1]

    V0 = dolfinx.fem.functionspace(mesh0, ("Lagrange", 1))
    V1 = dolfinx.fem.functionspace(mesh1, ("Lagrange", 1))
    dx0 = ufl.Measure("dx", domain=mesh0)
    dx1 = ufl.Measure("dx", domain=mesh1)

    u0, u1 = ufl.TrialFunction(V0), ufl.TrialFunction(V1)
    v0, v1 = ufl.TestFunction(V0), ufl.TestFunction(V1)

    f = Function(V0, name="control")
    f.interpolate(lambda x: np.sin(np.pi * x[0]) * x[1])

    a = [
        [ufl.inner(ufl.grad(u0), ufl.grad(v0)) * dx0, None],
        [None, ufl.inner(ufl.grad(u1), ufl.grad(v1)) * dx1],
    ]
    L = [ufl.inner(f, v0) * dx0, ufl.inner(dolfinx.fem.Constant(mesh1, 0.0), v1) * dx1]

    mesh0.topology.create_connectivity(mesh0.topology.dim - 1, mesh0.topology.dim)
    boundary_facets0 = dolfinx.mesh.exterior_facet_indices(mesh0.topology)
    boundary_dofs0 = dolfinx.fem.locate_dofs_topological(V0, mesh0.topology.dim - 1, boundary_facets0)
    bc0 = dolfinx.fem.dirichletbc(dolfinx.fem.Constant(mesh0, 0.0), boundary_dofs0, V0)

    mesh1.topology.create_connectivity(mesh1.topology.dim - 1, mesh1.topology.dim)
    boundary_facets1 = dolfinx.mesh.exterior_facet_indices(mesh1.topology)
    boundary_dofs1 = dolfinx.fem.locate_dofs_topological(V1, mesh1.topology.dim - 1, boundary_facets1)

    g = Function(V1, name="bc_value")
    g.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))
    bc = dirichletbc(g, boundary_dofs1, V=V1)

    uh0, uh1 = Function(V0, name="state0"), Function(V1, name="state1")
    problem = LinearProblem(
        a,
        L,
        u=[uh0, uh1],
        bcs=[bc0, bc],
        entity_maps=entity_maps,
        petsc_options=direct_solve,
        adjoint_petsc_options=direct_solve,
        tlm_petsc_options=direct_solve,
    )
    problem.solve()

    # dolfinx.fem.form compiles a single integration domain per Form even with
    # entity_maps (entity_maps relates *coefficient/argument* meshes to that one
    # domain, it does not let two top-level integration domains share one Form) -- sum
    # the two single-mesh objective terms as separate tracked scalar assembles instead.
    J = assemble_scalar(ufl.inner(uh0, uh0) * dx0) + assemble_scalar(ufl.inner(uh1, uh1) * dx1)

    Jh_g = pyadjoint.ReducedFunctional(J, pyadjoint.Control(g))
    g0 = Function(V1)
    g0.x.array[:] = g.x.array
    dg = Function(V1)
    dg.interpolate(lambda x: 30.0 * np.cos(np.pi * x[0]) * x[1])
    Jh_g(g0)
    min_rate = pyadjoint.taylor_test(Jh_g, g0, dg)
    assert np.isclose(min_rate, 2.0, rtol=1e-1, atol=1e-1), f"Expected convergence rate close to 2.0, got {min_rate}"

    Jh_g(g0)
    J0 = float(Jh_g(g0))
    dJdm0 = Jh_g.derivative()._ad_dot(dg)
    Hm0 = Jh_g.hessian(dg)._ad_dot(dg)
    for scale in [1.0, 0.3, 0.1, 0.01]:
        gp = Function(V1)
        gp.x.array[:] = g0.x.array + scale * dg.x.array
        Jp = float(Jh_g(gp))
        predicted = J0 + scale * dJdm0 + 0.5 * scale**2 * Hm0
        assert np.isclose(Jp, predicted, rtol=0, atol=1e-2), (
            f"scale={scale}: exact 2nd-order remainder {Jp - predicted:.3e} too large"
        )
