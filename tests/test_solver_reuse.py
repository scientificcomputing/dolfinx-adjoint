"""Regression tests for solver/form reuse across repeated LinearProblem.solve() calls.

These guard the "problem owns its solvers, blocks share them" refactor: recomputing
a LinearProblem's forward solve (e.g. during a tape replay, Taylor test, or
optimisation iteration) must not recompile the underlying UFL forms via FFCx on
every call, and must never mutate the user's own dependency/control objects in
place (doing so would corrupt them for any later use of that same object, such as
a Taylor test that perturbs the original control directly).
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
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
    ``LinearProblem._get_or_build_recompute_forms``) are compiled once per Problem,
    the first time any of its blocks replays -- not once per block/timestep. So the
    number of ``dolfinx.fem.form`` calls triggered by replaying the whole tape must
    be the same whether the tape has 3 timesteps or 8.
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

    NonlinearProblemBlock has always mutated its dependency coefficients in
    place during recompute (SNES's residual/Jacobian callbacks are bound to
    fixed compiled Form objects, so there is no equivalent of swapping in a
    differently-compiled form the way LinearProblemBlock historically did).
    Fixed by routing every non-``u`` coefficient through a dedicated
    placeholder from the moment the SNES is built (see
    ``NonlinearProblem._value_placeholders``), so recompute writes into the
    placeholder instead of the user's own object.
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
    dm.interpolate(lambda x: 2 * np.sin(x[0] * np.pi))

    # Passing `f` itself (not a copy) as the expansion point.
    min_rate = pyadjoint.taylor_test(Jh, f, dm, dJdm=0)

    assert np.allclose(f.x.array, f_original), (
        "the original control object was mutated by recompute; it must stay pristine"
    )
    assert np.isclose(min_rate, 1.0, rtol=1e-1, atol=1e-1), f"Expected convergence rate close to 1.0, got {min_rate}"
