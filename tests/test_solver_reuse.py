"""Regression tests for solver/form reuse across repeated LinearProblem.solve() calls.

These guard the "problem owns its solvers, blocks share them" refactor: recomputing
a LinearProblem's forward solve (e.g. during a tape replay, Taylor test, or
optimisation iteration) must not recompile the underlying UFL forms via FFCx on
every call, and must never mutate the user's own dependency/control objects in
place (doing so would corrupt them for any later use of that same object, such as
a Taylor test that perturbs the original control directly).
"""

import gc
import weakref

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, LinearProblem, NonlinearProblem, assemble_scalar, assign


def _run_heat_steps(num_steps: int, monkeypatch) -> int:
    """Solve a small time-stepping heat problem for ``num_steps`` steps on a single,
    reused LinearProblem, then replay it once via a ReducedFunctional at a different
    control value, counting calls to ``dolfinx.fem.form`` during that replay only.
    """
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    dt = 0.1

    m = Function(V, name="control")
    m.interpolate(lambda x: np.sin(x[0] * np.pi))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    uh = Function(V, name="state")
    u_prev = Function(V, name="state_prev")

    F = (u - u_prev) / dt * v * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - m * v * ufl.dx
    a, L = ufl.system(F)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V)

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = LinearProblem(a, L, bcs=[bc], u=uh, petsc_options=petsc_options)

    for _ in range(num_steps):
        problem.solve()
        assign(uh, u_prev)

    # A single objective computed once, after the loop: its own AssembleBlock
    # recompiles independently of this test and independently of num_steps, so
    # it contributes a fixed, step-count-independent cost that does not
    # confound the assertion this test is making about LinearProblemBlock.
    J = assemble_scalar(0.5 * ufl.inner(uh, uh) * ufl.dx)
    control = pyadjoint.Control(m)
    Jhat = pyadjoint.ReducedFunctional(J, control)

    m2 = Function(V)
    m2.interpolate(lambda x: 1.0 + np.cos(x[0] * np.pi))

    real_form = dolfinx.fem.form
    calls = []

    def counting_form(*args, **kwargs):
        calls.append(1)
        return real_form(*args, **kwargs)

    with monkeypatch.context() as ctx:
        ctx.setattr(dolfinx.fem, "form", counting_form)
        Jhat(m2)

    return len(calls)


def test_recompute_form_count_independent_of_step_count(monkeypatch):
    """Replaying more timesteps on the same LinearProblem must not compile more forms.

    The recompute-time placeholder forms (see
    ``LinearProblem._value_placeholders``) are compiled exactly once, when the
    Problem is constructed -- not once per block/timestep, and not again on
    replay. So the number of ``dolfinx.fem.form`` calls triggered by replaying
    the whole tape must be the same whether the tape has 3 timesteps or 8.
    """
    count_short = _run_heat_steps(3, monkeypatch)
    count_long = _run_heat_steps(8, monkeypatch)
    assert count_short > 0
    assert count_short == count_long


def test_recompute_does_not_corrupt_original_control():
    """A Taylor test that perturbs the original control object directly must see
    that object's pristine value at every perturbation, not whatever a previous
    recompute happened to leave written into it.

    This is a direct regression test for a bug caught while implementing
    recompute-time placeholders: mutating a block's dependency Function in place
    during recompute (rather than into a dedicated placeholder) corrupts the
    control for any later computation relative to it -- exactly what
    ``pyadjoint.taylor_test(Jh, m, dm)`` does by calling ``m._ad_add(...)`` on the
    same, live ``m`` object between evaluations.
    """
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    uh = Function(V, name="state")
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)

    m = Function(V, name="control")
    m.interpolate(lambda x: 1.0 + x[0] ** 2 + x[1] ** 2)
    m_original = m.x.array.copy()

    f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
    a = m * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V)

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = LinearProblem(a, L, bcs=[bc], u=uh, petsc_options=petsc_options)
    problem.solve()

    d = Function(V)
    d.interpolate(lambda x: np.sin(np.pi * x[0]))
    J = assemble_scalar(0.5 * ufl.inner(uh - d, uh - d) * ufl.dx)

    control = pyadjoint.Control(m)
    Jh = pyadjoint.ReducedFunctional(J, control)

    dm = Function(V)
    dm.interpolate(lambda x: 2 * np.sin(x[0] * np.pi) * np.cos(x[1] * np.pi))

    # Passing `m` itself (not a copy) as the expansion point, matching a
    # perfectly ordinary way to write this call.
    min_rate = pyadjoint.taylor_test(Jh, m, dm, dJdm=0)

    assert np.allclose(m.x.array, m_original), (
        "the original control object was mutated by recompute; it must stay pristine"
    )
    assert np.isclose(min_rate, 1.0, rtol=1e-1, atol=1e-1), f"Expected convergence rate close to 1.0, got {min_rate}"


def test_nonlinear_recompute_does_not_corrupt_original_control():
    """As above, but for NonlinearProblem.

    NonlinearProblem's SNES binds its residual/Jacobian callbacks to fixed
    compiled Form objects at construction time, so there is no way to later
    swap in a differently-compiled form the way a KSP-based solve can; the
    only way to make the SNES see a different coefficient value is to mutate
    the exact object its compiled forms reference. Every non-``u`` coefficient
    is therefore routed through a dedicated placeholder from the moment the
    SNES is built (``NonlinearProblem._value_placeholders``), and
    ``LinearProblem`` now uses the identical mechanism -- both classes always
    solve by refreshing placeholder values and calling an unchanging,
    already-compiled solver, never by mutating the user's own coefficient or
    recompiling a form.
    """
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 7)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    f = Function(V, name="control")
    f.interpolate(lambda x: 2.0 + np.sin(x[0]))
    f_original = f.x.array.copy()

    u1 = Function(V, name="state")
    u1.interpolate(lambda x: np.ones_like(x[0]))
    v1 = ufl.TestFunction(V)
    F1 = (1 + u1**2) * ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx - f * v1 * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc_val = dolfinx.fem.Constant(mesh, np.dtype(dolfinx.default_scalar_type).type(1.0))
    bc = dolfinx.fem.dirichletbc(bc_val, boundary_dofs, V)

    direct_options = {
        "ksp_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    }
    options = {
        "snes_error_if_not_converged": True,
    }
    problem = NonlinearProblem(F1, u=u1, bcs=[bc], petsc_options=options, adjoint_petsc_options=direct_options)
    problem.solve()

    d = pyadjoint.AdjFloat(0.2)
    J = assemble_scalar((u1 - d) ** 3 * ufl.dx)
    control = pyadjoint.Control(f)
    Jh = pyadjoint.ReducedFunctional(J, control)

    dm = Function(V)
    dm.interpolate(lambda x: 2 * np.sin(x[0] * np.pi))

    # Passing `f` itself (not a copy) as the expansion point.
    min_rate = pyadjoint.taylor_test(Jh, f, dm, dJdm=0)

    assert np.allclose(f.x.array, f_original), (
        "the original control object was mutated by recompute; it must stay pristine"
    )
    assert np.isclose(min_rate, 1.0, rtol=1e-1, atol=1e-1), f"Expected convergence rate close to 1.0, got {min_rate}"


def test_linear_adjoint_lhs_compiled_once():
    """adjoint(dF/du) does not depend on the state for a linear problem, so it
    should be compiled exactly once per Problem (in
    ``LinearProblem._get_or_build_adjoint_solver``) and never rebuilt --
    including across repeated ``derivative()``/``hessian()`` calls at
    different control values, which is exactly the scenario
    (``test_tlm_update.py``'s "warm up at another point" tests) that would
    catch a stale operator being silently reused with the wrong coefficient
    values.

    The control ``m`` sits inside the bilinear form itself (not just the
    right-hand side), so the adjoint operator actually depends on it -- a
    control that only appeared in ``L`` wouldn't exercise this at all.
    """
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    uh = Function(V, name="state")
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)

    m = Function(V, name="control")
    m.interpolate(lambda x: 1.0 + x[0] ** 2 + x[1] ** 2)

    f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
    a = m * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V)

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = LinearProblem(a, L, bcs=[bc], u=uh, petsc_options=petsc_options)
    problem.solve()

    d = Function(V)
    d.interpolate(lambda x: np.sin(np.pi * x[0]))
    J = assemble_scalar(0.5 * ufl.inner(uh - d, uh - d) * ufl.dx)
    control = pyadjoint.Control(m)
    Jh = pyadjoint.ReducedFunctional(J, control)

    adjoint_solver = problem._get_or_build_adjoint_solver()
    compiled_lhs = adjoint_solver._a
    assert compiled_lhs is not None

    Jh.derivative()
    assert adjoint_solver._a is compiled_lhs, "adjoint LHS was rebuilt by derivative()"

    m2 = Function(V)
    m2.interpolate(lambda x: 2.0 + np.sin(x[0]))
    Jh(m2)
    Jh.derivative()
    assert adjoint_solver._a is compiled_lhs, "adjoint LHS was rebuilt after evaluating at a new point"

    dm = Function(V)
    dm.interpolate(lambda x: np.cos(x[0] * np.pi))
    Jh.hessian(dm)
    assert adjoint_solver._a is compiled_lhs, "the SOA (Hessian) solve rebuilt the shared adjoint LHS"


def test_nonlinear_adjoint_lhs_compiled_once():
    """As above, but for NonlinearProblem: adjoint(dF/du) does depend on u's
    current value here, but that dependency is routed through a dedicated
    placeholder (``NonlinearProblem._state_placeholder``), refreshed per
    call, so the compiled LHS itself is still built exactly once.
    """
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    f = Function(V, name="control")
    f.interpolate(lambda x: 2.0 + np.sin(x[0]))

    u1 = Function(V, name="state")
    u1.interpolate(lambda x: np.ones_like(x[0]))
    v1 = ufl.TestFunction(V)
    F1 = (1 + u1**2) * ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx - f * v1 * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc_val = dolfinx.fem.Constant(mesh, np.dtype(dolfinx.default_scalar_type).type(1.0))
    bc = dolfinx.fem.dirichletbc(bc_val, boundary_dofs, V)

    direct_options = {
        "ksp_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    }
    options = {
        "snes_error_if_not_converged": True,
    }
    options.update(direct_options)
    problem = NonlinearProblem(F1, u=u1, bcs=[bc], petsc_options=options, adjoint_petsc_options=direct_options)
    problem.solve()

    d = pyadjoint.AdjFloat(0.2)
    J = assemble_scalar((u1 - d) ** 3 * ufl.dx)
    control = pyadjoint.Control(f)
    Jh = pyadjoint.ReducedFunctional(J, control)

    adjoint_solver = problem._get_or_build_adjoint_solver()
    compiled_lhs = adjoint_solver._a
    assert compiled_lhs is not None

    Jh.derivative()
    assert adjoint_solver._a is compiled_lhs, "adjoint LHS was rebuilt by derivative()"

    f2 = Function(V)
    f2.interpolate(lambda x: 3.0 + np.cos(x[0]))
    Jh(f2)
    Jh.derivative()
    assert adjoint_solver._a is compiled_lhs, "adjoint LHS was rebuilt after evaluating at a new point"

    dm = Function(V)
    dm.interpolate(lambda x: np.sin(x[0] * np.pi))
    Jh.hessian(dm)
    assert adjoint_solver._a is compiled_lhs, "the SOA (Hessian) solve rebuilt the shared adjoint LHS"


def test_tlm_rhs_templates_compiled_once():
    """Each dependency's TLM right-hand-side template
    (``LinearProblem._get_or_build_tlm_rhs_templates``) must be compiled exactly
    once and never rebuilt, including across repeated ``hessian()`` calls at
    different control values (a Hessian evaluation always drives a TLM sweep
    first).
    """
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    uh = Function(V, name="state")
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)

    m = Function(V, name="control")
    m.interpolate(lambda x: 1.0 + x[0] ** 2 + x[1] ** 2)

    f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
    a = m * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V)

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = LinearProblem(a, L, bcs=[bc], u=uh, petsc_options=petsc_options)
    problem.solve()

    d = Function(V)
    d.interpolate(lambda x: np.sin(np.pi * x[0]))
    J = assemble_scalar(0.5 * ufl.inner(uh - d, uh - d) * ufl.dx)
    control = pyadjoint.Control(m)
    Jh = pyadjoint.ReducedFunctional(J, control)

    dm = Function(V)
    dm.interpolate(lambda x: np.cos(x[0] * np.pi))
    Jh.hessian(dm)

    templates, _, _ = problem._get_or_build_tlm_rhs_templates()
    compiled_ids = {c: id(form) for c, form in templates.items()}
    assert compiled_ids, "no TLM RHS templates were built"

    m2 = Function(V)
    m2.interpolate(lambda x: 2.0 + np.sin(x[0]))
    Jh(m2)
    Jh.hessian(dm)

    templates_after, _, _ = problem._get_or_build_tlm_rhs_templates()
    for c, form in templates_after.items():
        assert id(form) == compiled_ids[c], f"TLM RHS template for {c.name} was rebuilt"


def test_tlm_skips_inactive_dependency_with_singular_derivative():
    """An inactive dependency (no tangent-linear value) whose derivative would be
    singular at its current value must never be evaluated at all -- not even
    with a zeroed seed.

    ``c`` enters the bilinear form as ``sqrt(c)``, which is finite at ``c == 0``
    but whose derivative, ``1 / (2 * sqrt(c))``, is not. ``c`` is set to exactly
    zero over part of the domain and is never declared a control, so its
    tangent-linear value is always ``None``. If that dependency's contribution
    were assembled with a zeroed seed instead of skipped outright, the ``0 *
    inf`` there would poison the whole tangent-linear (and, downstream,
    Hessian) result with ``NaN``. Only ``m`` is seeded.
    """
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    Z = dolfinx.fem.functionspace(mesh, ("DG", 0))

    uh = Function(V, name="state")
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)

    c = Function(Z, name="not_a_control")
    c.interpolate(lambda x: np.maximum(x[0] - 0.5, 0.0))

    m = Function(V, name="control")
    m.interpolate(lambda x: 1.0 + x[0] ** 2 + x[1] ** 2)

    a = (1.0 + ufl.sqrt(c)) * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = m * v * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V)

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = LinearProblem(a, L, bcs=[bc], u=uh, petsc_options=petsc_options)
    problem.solve()
    assert np.isfinite(uh.x.array).all()

    d = Function(V)
    d.interpolate(lambda x: np.sin(np.pi * x[0]))
    J = assemble_scalar(0.5 * ufl.inner(uh - d, uh - d) * ufl.dx)
    control = pyadjoint.Control(m)
    Jh = pyadjoint.ReducedFunctional(J, control)

    dm = Function(V)
    dm.interpolate(lambda x: np.cos(x[0] * np.pi))

    grad = Jh.derivative()
    assert np.isfinite(grad.x.array).all(), "gradient contains NaN/Inf from the inactive singular dependency"

    hessian_action = Jh.hessian(dm)
    assert np.isfinite(hessian_action.x.array).all(), (
        "Hessian action contains NaN/Inf from the inactive singular dependency"
    )


def test_nonlinear_tlm_rhs_templates_compiled_once():
    """As ``test_tlm_rhs_templates_compiled_once``, but for NonlinearProblem:
    the TLM solver (``NonlinearProblem._get_or_build_tlm_solver``) and each
    dependency's TLM right-hand-side template
    (``NonlinearProblem._get_or_build_tlm_rhs_templates``) must be compiled
    exactly once and never rebuilt, including across repeated ``hessian()``
    calls at different control values (a Hessian evaluation always drives a
    TLM sweep first).
    """
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    f = Function(V, name="control")
    f.interpolate(lambda x: 2.0 + np.sin(x[0]))

    u1 = Function(V, name="state")
    u1.interpolate(lambda x: np.ones_like(x[0]))
    v1 = ufl.TestFunction(V)
    F1 = (1 + u1**2) * ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx - f * v1 * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc_val = dolfinx.fem.Constant(mesh, np.dtype(dolfinx.default_scalar_type).type(1.0))
    bc = dolfinx.fem.dirichletbc(bc_val, boundary_dofs, V)

    options = {
        "snes_error_if_not_converged": True,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = NonlinearProblem(F1, u=u1, bcs=[bc], petsc_options=options, adjoint_petsc_options=options)
    problem.solve()

    d = pyadjoint.AdjFloat(0.2)
    J = assemble_scalar((u1 - d) ** 3 * ufl.dx)
    control = pyadjoint.Control(f)
    Jh = pyadjoint.ReducedFunctional(J, control)

    dm = Function(V)
    dm.interpolate(lambda x: np.sin(x[0] * np.pi))
    Jh.hessian(dm)

    tlm_solver = problem._get_or_build_tlm_solver()
    compiled_tlm_lhs = tlm_solver._a
    assert compiled_tlm_lhs is not None

    templates, _, _ = problem._get_or_build_tlm_rhs_templates()
    compiled_ids = {c: id(form) for c, form in templates.items()}
    assert compiled_ids, "no TLM RHS templates were built"

    f2 = Function(V)
    f2.interpolate(lambda x: 3.0 + np.cos(x[0]))
    Jh(f2)
    Jh.hessian(dm)

    assert tlm_solver._a is compiled_tlm_lhs, "TLM LHS was rebuilt after evaluating at a new point"
    templates_after, _, _ = problem._get_or_build_tlm_rhs_templates()
    for c, form in templates_after.items():
        assert id(form) == compiled_ids[c], f"TLM RHS template for {c.name} was rebuilt"


def test_linear_problem_released_by_refcounting_not_gc():
    """Dropping a LinearProblem (and clearing the tape) must release it -- and its
    PETSc solvers -- via ordinary reference counting, never leaving it as cyclic
    garbage collected only by a later ``gc.collect()``.

    Regression test for a real bug found while building the placeholder-substitution
    machinery: a recursive nested ``_replace`` helper inside ``LinearProblem.__init__``
    captured *itself* (for the recursive call) and ``self`` in its closure, forming a
    reference cycle. That cycle kept the Problem -- and hence its PETSc Mat/KSP
    objects -- alive until whenever the cyclic garbage collector next ran, exactly
    the "freed at a rank-nondeterministic moment" hazard the whole
    Problem-owns-its-solvers refactor exists to eliminate: confirmed live under
    ``mpirun -n 2`` by observing the two ranks diverge into two different collective
    calls -- one inside a PETSc Mat's collective MUMPS-termination destructor, the
    other already building an unrelated dofmap for the next test -- while this bug
    was present. Fixed by making the recursive helper
    (``dolfinx_adjoint.ufl_utils.recursive_replace``) a plain module-level
    function taking the placeholder dict as an explicit argument, so it does not need
    to capture itself or ``self``.
    """
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    uh = Function(V, name="state")
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)
    m = Function(V, name="control")
    m.interpolate(lambda x: 1.0 + x[0] ** 2)

    a = m * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = ufl.inner(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), v) * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V)

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    gc.disable()
    try:
        problem = LinearProblem(a, L, bcs=[bc], u=uh, petsc_options=petsc_options)
        problem.solve()

        problem_ref = weakref.ref(problem)
        del problem
        pyadjoint.get_working_tape().clear_tape()

        assert problem_ref() is None, (
            "LinearProblem was not released by ordinary refcounting -- it is cyclic "
            "garbage, collected only by gc.collect() at a moment that can differ "
            "between MPI ranks"
        )
    finally:
        gc.enable()


def test_nonlinear_problem_released_by_refcounting_not_gc():
    """As above, but for NonlinearProblem."""
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    f = Function(V, name="control")
    f.interpolate(lambda x: 2.0 + np.sin(x[0]))

    u1 = Function(V, name="state")
    u1.interpolate(lambda x: np.ones_like(x[0]))
    v1 = ufl.TestFunction(V)
    F1 = (1 + u1**2) * ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx - f * v1 * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc_val = dolfinx.fem.Constant(mesh, np.dtype(dolfinx.default_scalar_type).type(1.0))
    bc = dolfinx.fem.dirichletbc(bc_val, boundary_dofs, V)
    direct_options = {
        "ksp_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    }
    options = {
        "snes_error_if_not_converged": True,
    }
    options.update(direct_options)
    gc.disable()
    try:
        problem = NonlinearProblem(F1, u=u1, bcs=[bc], petsc_options=options, adjoint_petsc_options=direct_options)
        problem.solve()

        problem_ref = weakref.ref(problem)
        del problem
        pyadjoint.get_working_tape().clear_tape()

        assert problem_ref() is None, (
            "NonlinearProblem was not released by ordinary refcounting -- it is cyclic "
            "garbage, collected only by gc.collect() at a moment that can differ "
            "between MPI ranks"
        )
    finally:
        gc.enable()


def test_linear_problem_rebuilt_after_garbage_collection():
    """A block only holds a ``weakref`` to its owning LinearProblem (see
    LinearProblemBlock._problem/_rebuild_problem), precisely so dropping every
    external reference to the Problem releases it immediately, even while blocks
    that reference it are still on the tape. Replaying the tape afterwards (e.g. to
    differentiate) must still work: the block rebuilds an equivalent LinearProblem on
    demand, warning since that is a costly fallback, and the rebuilt Problem must
    give the same answer the original would have.
    """
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    uh = Function(V, name="state")
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)
    m = Function(V, name="control")
    m.interpolate(lambda x: 1.0 + x[0] ** 2)

    a = m * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = ufl.inner(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), v) * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V)

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }

    problem = LinearProblem(a, L, bcs=[bc], u=uh, petsc_options=petsc_options, petsc_options_prefix="dxa_rebuild_test_")
    problem.solve()

    J = assemble_scalar(uh * uh * ufl.dx)
    control = pyadjoint.Control(m)
    Jh = pyadjoint.ReducedFunctional(J, control)

    problem_ref = weakref.ref(problem)
    del problem
    assert problem_ref() is None, (
        "LinearProblem should be released the instant its last external reference is "
        "dropped, since blocks only hold a weakref to it"
    )

    pert = Function(V)
    pert.interpolate(lambda x: np.cos(x[1]))

    with pytest.warns(UserWarning, match="LinearProblem was garbage collected"):
        min_rate = pyadjoint.taylor_test(Jh, m, pert)
    assert np.isclose(min_rate, 2.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 2.0, got {min_rate}"


def test_nonlinear_problem_rebuilt_after_garbage_collection():
    """As ``test_linear_problem_rebuilt_after_garbage_collection``, but for
    NonlinearProblem."""
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    f = Function(V, name="control")
    f.interpolate(lambda x: 2.0 + np.sin(x[0]))

    u1 = Function(V, name="state")
    u1.interpolate(lambda x: np.ones_like(x[0]))
    v1 = ufl.TestFunction(V)
    F1 = (1 + u1**2) * ufl.inner(ufl.grad(u1), ufl.grad(v1)) * ufl.dx - f * v1 * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc_val = dolfinx.fem.Constant(mesh, np.dtype(dolfinx.default_scalar_type).type(1.0))
    bc = dolfinx.fem.dirichletbc(bc_val, boundary_dofs, V)

    options = {
        "snes_error_if_not_converged": True,
    }
    direct_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    options.update(direct_options)

    problem = NonlinearProblem(
        F1,
        u=u1,
        bcs=[bc],
        petsc_options=options,
        adjoint_petsc_options=direct_options,
        petsc_options_prefix="dxa_nonlinear_rebuild_test_",
    )
    problem.solve()

    d = pyadjoint.AdjFloat(0.2)
    J = assemble_scalar((u1 - d) * (u1 - d) * ufl.dx)
    control = pyadjoint.Control(f)
    Jh = pyadjoint.ReducedFunctional(J, control)

    problem_ref = weakref.ref(problem)
    del problem
    assert problem_ref() is None, (
        "NonlinearProblem should be released the instant its last external reference "
        "is dropped, since blocks only hold a weakref to it"
    )

    pert = Function(V)
    pert.interpolate(lambda x: np.cos(x[1]))

    with pytest.warns(UserWarning, match="NonlinearProblem was garbage collected"):
        min_rate = pyadjoint.taylor_test(Jh, f, pert)
    assert np.isclose(min_rate, 2.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 2.0, got {min_rate}"


def test_default_petsc_options_prefix_is_unique_per_problem():
    """Two Problems of the same kind that don't pass ``petsc_options_prefix``
    explicitly must not resolve to the same default prefix.

    PETSc's options database is process-global and keyed by prefix: two Problems
    sharing one would let one instance's SNES/KSP options silently leak into (or get
    overwritten by) the other's, exactly the kind of collision this default-prefix
    scheme exists to prevent -- see ``dolfinx_adjoint.solvers._PROBLEM_PREFIX_COUNTER``.
    """
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(u, v) * ufl.dx
    L = ufl.inner(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), v) * ufl.dx

    linear_a = LinearProblem(a, L)
    linear_b = LinearProblem(a, L)
    assert linear_a._petsc_options_prefix != linear_b._petsc_options_prefix

    u1 = Function(V, name="state")
    F = ufl.inner((1 + u1**2) * u1, v) * ufl.dx - ufl.inner(dolfinx.fem.Constant(mesh, 1.0), v) * ufl.dx
    nonlinear_a = NonlinearProblem(F, u=u1)
    nonlinear_b = NonlinearProblem(F, u=u1)
    assert nonlinear_a._petsc_options_prefix != nonlinear_b._petsc_options_prefix

    # Also unique across the two classes, since they draw from one shared counter.
    assert linear_a._petsc_options_prefix != nonlinear_a._petsc_options_prefix

    # An explicitly-passed prefix must be honored unchanged.
    explicit = LinearProblem(a, L, petsc_options_prefix="my_custom_prefix_")
    assert explicit._petsc_options_prefix == "my_custom_prefix_"


def test_nonlinear_blocked_problem_templates_compiled_once():
    """As ``test_nonlinear_adjoint_lhs_compiled_once``/``test_nonlinear_tlm_rhs_templates_compiled_once``,
    but for a *blocked* ``NonlinearProblem`` (a Navier-Stokes-like velocity/pressure system):
    the adjoint solver, TLM solver, per-dependency TLM right-hand-side templates, and the
    blocked Hessian templates (``NonlinearProblem._get_or_build_hessian_templates``'s
    ``isinstance(self._u, list)`` branch, which otherwise has no caching-focused regression
    coverage at all) must all be compiled exactly once and never rebuilt across repeated
    ``derivative()``/``hessian()`` calls at different control values.
    """
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
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
    ksp_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    forward_options = {
        "snes_type": "newtonls",
        "snes_error_if_not_converged": True,
        "snes_atol": 1e-9,
        "snes_rtol": 1e-9,
        "snes_stol": 1e-12,
    }
    forward_options.update(ksp_options)
    problem = NonlinearProblem(
        [F0, F1],
        u=[uh, ph],
        bcs=[bc],
        petsc_options_prefix="dxa_blocked_nonlinear_reuse_test_",
        petsc_options=forward_options,
        adjoint_petsc_options=ksp_options,
        tlm_petsc_options=ksp_options,
    )
    problem.solve()

    J = assemble_scalar(ufl.inner(uh, uh) ** 2 * dx)
    control = pyadjoint.Control(mu)
    Jh = pyadjoint.ReducedFunctional(J, control)

    dm = Function(Z)
    dm.interpolate(lambda x: 0.2 * np.sin(3 * x[0]))
    Jh.hessian(dm)

    adjoint_solver = problem._get_or_build_adjoint_solver()
    compiled_adjoint_lhs = adjoint_solver._a
    assert compiled_adjoint_lhs is not None

    tlm_solver = problem._get_or_build_tlm_solver()
    compiled_tlm_lhs = tlm_solver._a
    assert compiled_tlm_lhs is not None

    tlm_templates, _, _ = problem._get_or_build_tlm_rhs_templates()
    compiled_tlm_ids = {c: id(form) for c, form in tlm_templates.items()}
    assert compiled_tlm_ids, "no TLM RHS templates were built"

    hessian_templates = problem._get_or_build_hessian_templates()
    assert isinstance(hessian_templates.soa_self, list), "expected a per-row list for a blocked problem"

    mu2 = Function(Z)
    mu2.interpolate(lambda x: 2.0 + np.cos(np.pi * x[0]))
    Jh(mu2)
    Jh.derivative()
    Jh.hessian(dm)

    assert adjoint_solver._a is compiled_adjoint_lhs, "adjoint LHS was rebuilt after evaluating at a new point"
    assert tlm_solver._a is compiled_tlm_lhs, "TLM LHS was rebuilt after evaluating at a new point"

    tlm_templates_after, _, _ = problem._get_or_build_tlm_rhs_templates()
    for c, form in tlm_templates_after.items():
        assert id(form) == compiled_tlm_ids[c], f"TLM RHS template for {c.name} was rebuilt"

    assert problem._get_or_build_hessian_templates() is hessian_templates, (
        "blocked Hessian templates were rebuilt after evaluating at a new point"
    )


def test_recompute_checkpoint_and_tlm_value_preserve_overloaded_function_type():
    """A block's recomputed checkpoint and TLM value must stay the overloaded
    ``dolfinx_adjoint`` ``Function``, never silently downgrade to the plain
    ``dolfinx.fem.Function`` a bare ``.copy()`` would produce.

    Regression test for ``_ProblemBlockBase.recompute_component``/
    ``LinearProblemBlock.__init__`` (and the ``NonlinearProblemBlock`` equivalent)
    switching from ``.copy()`` -- which is hardcoded, in ``dolfinx.fem.Function``, to
    always return the plain base class regardless of the source's concrete type -- to
    ``Function._ad_create_checkpoint()``, which correctly preserves it.
    """
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    uh = Function(V, name="state")
    v = ufl.TestFunction(V)
    u_trial = ufl.TrialFunction(V)
    m = Function(V, name="control")
    m.interpolate(lambda x: 1.0 + x[0] ** 2)

    a = m * ufl.inner(ufl.grad(u_trial), ufl.grad(v)) * ufl.dx
    L = ufl.inner(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), v) * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V)

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = LinearProblem(a, L, bcs=[bc], u=uh, petsc_options=petsc_options)
    problem.solve()

    # create_block_variable() (called by problem.solve()'s own tape recording)
    # stores the new BlockVariable back onto uh itself, so uh.block_variable is
    # exactly the one LinearProblemBlock will recompute into below.
    output_bv = uh.block_variable

    # Using uh as an AssembleBlock coefficient here is what first turns its
    # block_variable into a dependency of another block, which is what triggers
    # pyadjoint to freeze a real checkpoint for it (BlockVariable.will_add_as_dependency
    # -> save_output -> Function._ad_create_checkpoint) -- exercised by every ordinary
    # J = assemble_scalar(f(uh) * dx) call, not something special-cased for this test.
    J = assemble_scalar(uh * uh * ufl.dx)
    control = pyadjoint.Control(m)
    Jh = pyadjoint.ReducedFunctional(J, control)

    assert isinstance(output_bv.checkpoint, Function), (
        "the checkpoint frozen when uh was first used as AssembleBlock's dependency "
        "should already be the overloaded Function"
    )

    # Re-evaluating at a new point forces LinearProblemBlock.prepare_recompute_component/
    # recompute_component; the checkpoint above already exists, so this exercises the
    # in-place-reuse branch (mirroring InterpolationBlock/Firedrake's GenericSolveBlock).
    m2 = Function(V)
    m2.interpolate(lambda x: 2.0 + np.sin(x[0]))
    Jh(m2)
    assert isinstance(output_bv.saved_output, Function), (
        "recompute_component's checkpoint reuse must preserve the overloaded Function type"
    )

    # A Hessian evaluation drives a TLM sweep first, exercising evaluate_tlm_component.
    dm = Function(V)
    dm.interpolate(lambda x: np.cos(np.pi * x[0]))
    Jh.hessian(dm)
    assert isinstance(output_bv.tlm_value, Function), (
        "evaluate_tlm_component's returned tlm_value must be the overloaded Function type"
    )
