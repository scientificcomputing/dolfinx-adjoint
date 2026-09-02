"""Regression tests for cached adjoint/TLM/Hessian templates staying correct across evaluations.

``_ProblemBase`` (``solvers.py``) builds ``dF/du``, the TLM right-hand side, and the Hessian
templates once per ``LinearProblem``/``NonlinearProblem`` and reuses them -- refreshed at each
new evaluation point -- across every block that Problem records, rather than recompiling per
block/solve. These tests check that reuse never leaves an operator evaluated at a stale point:
when the control appears in a way that makes ``dF/du`` genuinely depend on it, a cached-but-
unrefreshed operator would make every Hessian computed after the first one silently wrong.
Covers scalar and blocked (multi-field) problems, for both ``LinearProblem`` and
``NonlinearProblem``, plus a warm-start pollution check (``test_hessian_mpi_breakdown`` --
not actually MPI-specific, despite the name: it checks that solving at one control value in
between two evaluations at another value doesn't perturb the second evaluation's result).
"""

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, assemble_scalar
from dolfinx_adjoint.solvers import LinearProblem, NonlinearProblem

direct_solve = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "ksp_error_if_not_converged": True,
    "pc_factor_mat_solver_type": "mumps",
}


@pytest.fixture(scope="module")
def mesh_2D():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 7)


def _viscous_stokes(mesh):
    """Blocked Stokes-like problem whose control ``mu`` sits inside ``a[0][0]``.

    The control is part of the bilinear form, so that dF/du!=0, and we get
    a genuinely control-dependent tangent-linear operator.
    """
    el_u = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    el_p = basix.ufl.element("P", mesh.basix_cell(), 1)
    V = dolfinx.fem.functionspace(mesh, el_u)
    Q = dolfinx.fem.functionspace(mesh, el_p)
    Z = dolfinx.fem.functionspace(mesh, ("DG", 0))
    dx = ufl.Measure("dx", domain=mesh)

    mu = Function(Z, name="viscosity")
    mu.interpolate(lambda x: 1.0 + 0.5 * np.sin(np.pi * x[0]))

    u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)
    # A rotational (non-conservative) body force.  Avoid using constant force to avoid
    # velocity being 0 always, independent of control. Scaled to ensure decent-sized Taylor
    # remainders.
    x = ufl.SpatialCoordinate(mesh)
    f = 1e3 * ufl.as_vector((ufl.sin(ufl.pi * x[1]), ufl.cos(ufl.pi * x[0])))

    a = [
        [ufl.inner(mu * ufl.grad(u), ufl.grad(v)) * dx, ufl.inner(p, ufl.div(v)) * dx],
        [ufl.inner(q, ufl.div(u)) * dx, None],
    ]
    L = [ufl.inner(f, v) * dx, dolfinx.fem.Constant(mesh, 0.0) * q * dx]

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    zero = dolfinx.fem.Constant(mesh, np.zeros(mesh.geometry.dim, dtype=dolfinx.default_scalar_type))
    bc = dolfinx.fem.dirichletbc(zero, dofs, V)

    uh, ph = Function(V, name="velocity"), Function(Q, name="pressure")
    problem = LinearProblem(
        a,
        L,
        u=[uh, ph],
        bcs=[bc],
        petsc_options=direct_solve,
        adjoint_petsc_options=direct_solve,
        tlm_petsc_options=direct_solve,
    )
    problem.solve()

    # Quartic in the state and with no constant offset, so that the Taylor remainders
    # stay well above round-off: a functional dominated by a control-independent term
    # bottoms out at machine precision before the third-order rate is visible.
    J = assemble_scalar(ufl.inner(uh, uh) ** 2 * dx)
    return pyadjoint.ReducedFunctional(J, pyadjoint.Control(mu)), Z


@pytest.mark.parametrize("warm_up_at_another_point", [False, True])
def test_hessian_is_independent_of_previous_evaluation_points(warm_up_at_another_point, mesh_2D):
    """The Hessian at ``m2`` must not depend on whether ``J`` was evaluated at ``m1`` first.

    Both parametrizations run the identical second-order Taylor test at ``m2``.  The only
    difference is a prior evaluate-and-differentiate sweep at a *different* control value,
    which must not change the answer.
    """
    pyadjoint.get_working_tape().clear_tape()
    Jh, Z = _viscous_stokes(mesh_2D)

    m1, m2 = Function(Z), Function(Z)
    m1.interpolate(lambda x: 1.0 + 0.5 * np.sin(np.pi * x[0]))
    m2.interpolate(lambda x: 2.0 + 0.5 * np.cos(np.pi * x[1]))
    h = Function(Z)
    h.interpolate(lambda x: 10.0 + 3.2 * np.sin(3 * x[0]))

    if warm_up_at_another_point:
        Jh(m1)
        Jh.derivative()
        Jh.hessian(h)

    Jh(m2)
    dJdm = Jh.derivative()._ad_dot(h)
    Hm = Jh.hessian(h)._ad_dot(h)

    min_rate = pyadjoint.taylor_test(Jh, m2, h, dJdm=dJdm, Hm=Hm)
    assert np.isclose(min_rate, 3.0, rtol=0.1, atol=0.1), f"Expected convergence rate close to 3.0, got {min_rate}"


def _navier_stokes(mesh):
    """``_viscous_stokes`` with a ``u . grad(u)`` convective term added, making the residual
    genuinely nonlinear in the state -- a ``NonlinearProblem`` sibling of ``_viscous_stokes``,
    exercising the blocked (multi-output) Hessian path for ``NonlinearProblem``, which used to
    raise ``NotImplementedError`` unconditionally and is now the same shared code
    ``LinearProblemBlock`` already used for its own blocked Hessian.
    """
    el_u = basix.ufl.element("P", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    el_p = basix.ufl.element("P", mesh.basix_cell(), 1)
    V = dolfinx.fem.functionspace(mesh, el_u)
    Q = dolfinx.fem.functionspace(mesh, el_p)
    Z = dolfinx.fem.functionspace(mesh, ("DG", 0))
    dx = ufl.Measure("dx", domain=mesh)

    mu = Function(Z, name="viscosity")
    mu.interpolate(lambda x: 1.0 + 0.5 * np.sin(np.pi * x[0]))

    uh, ph = Function(V, name="velocity"), Function(Q, name="pressure")
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)

    # A moderate, non-conservative body force: large enough that the Taylor remainders stay
    # clear of round-off, small enough that the Newton solve below converges reliably (the
    # convective term is quadratic in the state, so scaling it up the way _viscous_stokes
    # scales its linear counterpart would make the residual far stiffer).
    x = ufl.SpatialCoordinate(mesh)
    f = 10.0 * ufl.as_vector((ufl.sin(ufl.pi * x[1]), ufl.cos(ufl.pi * x[0])))

    F0 = (
        ufl.inner(mu * ufl.grad(uh), ufl.grad(v)) * dx
        + ufl.inner(ufl.dot(ufl.grad(uh), uh), v) * dx
        + ufl.inner(ph, ufl.div(v)) * dx
        - ufl.inner(f, v) * dx
    )
    F1 = ufl.inner(q, ufl.div(uh)) * dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    zero = dolfinx.fem.Constant(mesh, np.zeros(mesh.geometry.dim, dtype=dolfinx.default_scalar_type))
    bc = dolfinx.fem.dirichletbc(zero, dofs, V)

    forward_options = {
        "snes_type": "newtonls",
        "snes_error_if_not_converged": True,
        # A warm-started recompute (see NonlinearProblemBlock._refresh_dFdu_state /
        # *ProblemBlockBase.prepare_recompute_component) starts SNES already at (or
        # extremely close to) a converged point when the control hasn't moved.  With
        # only the default rtol, SNES keeps trying to shrink a residual that is
        # already at floating-point noise, and the resulting near-singular Newton
        # step can make backtracking line search report DIVERGED_LINE_SEARCH.  An
        # explicit absolute tolerance lets it recognize "already converged" and
        # exit immediately instead.
        "snes_atol": 1e-9,
        "snes_rtol": 1e-9,
        "snes_stol": 1e-12,
    }
    forward_options.update(direct_solve)
    problem = NonlinearProblem(
        [F0, F1],
        u=[uh, ph],
        bcs=[bc],
        # A prefix distinct from the default ("dxa_nonlinear_problem_"): PETSc's options
        # database is process-global and keyed by prefix, and this fixture's snes_atol/rtol/stol
        # (needed for the warm-started recompute above) must not leak into -- or collide with --
        # another NonlinearProblem elsewhere in the suite that happens to use the default prefix.
        petsc_options_prefix="dxa_navier_stokes_test_",
        petsc_options=forward_options,
        adjoint_petsc_options=direct_solve,
        tlm_petsc_options=direct_solve,
    )
    problem.solve()

    # Quartic in the state and with no constant offset, for the same round-off-avoidance
    # reason as _viscous_stokes's objective.
    J = assemble_scalar(ufl.inner(uh, uh) ** 2 * dx)
    return pyadjoint.ReducedFunctional(J, pyadjoint.Control(mu)), Z


def test_hessian_is_independent_of_previous_evaluation_points_navier_stokes(mesh_2D):
    """``test_hessian_is_independent_of_previous_evaluation_points``'s ``NonlinearProblem``
    sibling: the same second-order Taylor test, but on a genuinely nonlinear, blocked
    (multi-output) residual, exercising the blocked Hessian path in
    ``_ProblemBase._get_or_build_hessian_templates``/``_ProblemBlockBase.prepare_evaluate_hessian``
    for ``NonlinearProblem`` for the first time -- previously this path only ever ran for
    ``LinearProblem``.
    """
    pyadjoint.get_working_tape().clear_tape()
    Jh, Z = _navier_stokes(mesh_2D)

    m2 = Function(Z)
    m2.interpolate(lambda x: 2.0 + 0.5 * np.cos(np.pi * x[1]))
    h = Function(Z)
    h.interpolate(lambda x: 1.0 + 0.3 * np.sin(3 * x[0]))

    Jh(m2)
    dJdm = Jh.derivative()._ad_dot(h)
    Hm = Jh.hessian(h)._ad_dot(h)

    min_rate = pyadjoint.taylor_test(Jh, m2, h, dJdm=dJdm, Hm=Hm)
    assert np.isclose(min_rate, 3.0, rtol=0.1, atol=0.1), f"Expected convergence rate close to 3.0, got {min_rate}"


def _diffusive_poisson(mesh):
    """Scalar Poisson problem whose control ``m`` sits inside ``a`` (the diffusivity).

    A scalar-space sibling of ``_viscous_stokes``: the control has to enter the
    bilinear form for this test to have any teeth, for the same reason noted there.
    """
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    Z = dolfinx.fem.functionspace(mesh, ("DG", 0))
    dx = ufl.Measure("dx", domain=mesh)

    m = Function(Z, name="diffusivity")
    m.interpolate(lambda x: 1.0 + 0.5 * np.sin(np.pi * x[0]))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = 1e2 * ufl.sin(ufl.pi * x[0]) * ufl.cos(ufl.pi * x[1])

    a = ufl.inner(m * ufl.grad(u), ufl.grad(v)) * dx
    L = ufl.inner(f, v) * dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), dofs, V)

    uh = Function(V, name="state")
    problem = LinearProblem(
        a,
        L,
        u=uh,
        bcs=[bc],
        petsc_options=direct_solve,
        adjoint_petsc_options=direct_solve,
        tlm_petsc_options=direct_solve,
    )
    problem.solve()

    # Quartic in the state, for the same round-off-avoidance reason as
    # ``_viscous_stokes``'s objective.
    J = assemble_scalar(uh**4 * dx)
    return pyadjoint.ReducedFunctional(J, pyadjoint.Control(m)), Z


@pytest.mark.parametrize("warm_up_at_another_point", [False, True])
def test_hessian_is_independent_of_previous_evaluation_points_scalar(warm_up_at_another_point, mesh_2D):
    """Scalar-space sibling of ``test_hessian_is_independent_of_previous_evaluation_points``.

    That test only covers the blocked/Stokes path; a plain (non-blocked) ``LinearProblem``
    with the control inside the bilinear form is the common case the cached, compiled-once
    adjoint/TLM/Hessian templates (``LinearProblem._get_or_build_adjoint_solver``/
    ``_get_or_build_tlm_solver``/``_get_or_build_hessian_templates``) exist to serve, and
    deserves the identical regression coverage: the Hessian at ``m2`` must not depend on
    whether ``J`` was evaluated at ``m1`` first.
    """
    pyadjoint.get_working_tape().clear_tape()
    Jh, Z = _diffusive_poisson(mesh_2D)

    m1, m2 = Function(Z), Function(Z)
    m1.interpolate(lambda x: 1.0 + 0.5 * np.sin(np.pi * x[0]))
    m2.interpolate(lambda x: 2.0 + 0.5 * np.cos(np.pi * x[1]))
    h = Function(Z)
    h.interpolate(lambda x: 1.0 + 0.3 * np.sin(3 * x[0]))

    if warm_up_at_another_point:
        Jh(m1)
        Jh.derivative()
        Jh.hessian(h)

    Jh(m2)
    dJdm = Jh.derivative()._ad_dot(h)
    Hm = Jh.hessian(h)._ad_dot(h)

    min_rate = pyadjoint.taylor_test(Jh, m2, h, dJdm=dJdm, Hm=Hm)
    assert np.isclose(min_rate, 3.0, rtol=0.1, atol=0.1), f"Expected convergence rate close to 3.0, got {min_rate}"


def test_hessian_mpi_breakdown(mesh_2D):
    pyadjoint.get_working_tape().clear_tape()
    Jh, Z = _viscous_stokes(mesh_2D)

    m1, m2 = Function(Z), Function(Z)
    m1.interpolate(lambda x: 1.0 + 0.5 * np.sin(np.pi * x[0]))
    m2.interpolate(lambda x: 2.0 + 0.5 * np.cos(np.pi * x[1]))
    h = Function(Z)
    h.interpolate(lambda x: 3.0 + 2.0 * np.sin(3 * x[0]))

    # === COLD START ===
    J_cold = float(Jh(m2))
    dJ_cold = Jh.derivative()
    H_cold = Jh.hessian(h)

    # Extract underlying FEniCSx arrays securely
    dJ_cold_array = dJ_cold.x.array.copy() if hasattr(dJ_cold, "x") else dJ_cold.array.copy()
    H_cold_array = H_cold.x.array.copy() if hasattr(H_cold, "x") else H_cold.array.copy()

    # === WARM START (Pollution check) ===
    Jh(m1)
    Jh.derivative()
    Jh.hessian(h)

    J_warm = float(Jh(m2))
    dJ_warm = Jh.derivative()
    H_warm = Jh.hessian(h)

    dJ_warm_array = dJ_warm.x.array if hasattr(dJ_warm, "x") else dJ_warm.array
    H_warm_array = H_warm.x.array if hasattr(H_warm, "x") else H_warm.array

    # === ASSERTIONS ===

    comm = mesh_2D.comm
    rank = comm.rank

    # 1. Forward State Check
    assert np.isclose(J_cold, J_warm), f"Rank {rank}: Forward evaluation J(m2) differs!"

    # 2. Gradient Check
    grad_diff = np.linalg.norm(dJ_cold_array - dJ_warm_array)
    assert grad_diff < 1e-10, f"Rank {rank}: Gradient differs after warm up! Diff: {grad_diff}"

    # 3. Hessian Check
    hess_diff = np.linalg.norm(H_cold_array - H_warm_array)
    assert hess_diff < 1e-10, f"Rank {rank}: Hessian differs after warm up! Diff: {hess_diff}"

    # 4. If arrays match, run Taylor test
    dJdm = dJ_warm._ad_dot(h)
    Hm = H_warm._ad_dot(h)
    min_rate = pyadjoint.taylor_test(Jh, m2, h, dJdm=dJdm, Hm=Hm)
    assert np.isclose(min_rate, 3.0, rtol=0.1, atol=0.1), f"Expected 3.0, got {min_rate}"
