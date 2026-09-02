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

        # Scale perturbation for hessian
        Jh(d)
        dJdm = Jh.derivative()._ad_dot(e)
        hessian = Jh.hessian(e)
        dHddu = hessian._ad_dot(e)
        min_rate = pyadjoint.taylor_test(Jh, d, e, dJdm=dJdm, Hm=dHddu)
        assert np.isclose(min_rate, 3.0, rtol=0.1, atol=0.1), f"Expected convergence rate close to 3.0, got {min_rate}"

        z = Function(Z)
        z.interpolate(
            lambda x: (5 * baseline * np.sin(x[1]) + 0.3 * baseline * x[1] * x[0], -2 * baseline * (x[0]) + (1 - x[1]))
        )
        f = Function(Z)
        f.interpolate(
            lambda x: (0.8 * baseline * x[1] ** 2, 2 * baseline * x[0] ** 2)
        )  # NOTE: Has to be divergence free
        f.x.scatter_forward()
        Jh(z)
        dJdm = Jh.derivative()._ad_dot(f)
        hessian = Jh.hessian(f)
        dHddu = hessian._ad_dot(f)
        min_rate = pyadjoint.taylor_test(Jh, z, f, dJdm=dJdm, Hm=dHddu)
        assert np.isclose(min_rate, 3.0, rtol=0.1, atol=0.1), f"Expected convergence rate close to 3.0, got {min_rate}"


def test_vector_valued_solver(mesh_2D):
    """Regression test for a spurious block-extraction bug on plain vector-valued spaces.

    Unlike ``test_solver``'s two parametrizations -- a genuine ``ufl.MixedFunctionSpace``
    (``use_mixed_space=True``) and an explicit list-of-lists two-field system
    (``use_mixed_space=False``, but still two separate function spaces and a
    ``u=[uh, ph]`` list) -- this problem has a *single* state ``Function`` (not a list)
    on one plain vector-*shaped* space (``("Lagrange", 1, (gdim,))``), exactly the shape
    every vector-PDE problem (e.g. elasticity) uses.

    Before the fix, ``compute_adjoint`` called ``ufl.extract_blocks`` unconditionally,
    which still decomposes a shaped (non-mixed) argument into spurious blocks even
    though there is no genuine block structure here. That first raised
    ``AttributeError: 'FunctionSpace' object has no attribute '_cpp_object'`` (a bare
    ``ufl.FunctionSpace`` lost its dolfinx wrapper during the spurious extraction), and
    after passing ``replace_argument=False``, a PETSc size mismatch instead (each
    spurious block still referenced the original, full-space ``Argument``, so the
    assembled adjoint operator came out sized for several redundant copies of the
    space). The fix threads ``blocked=isinstance(self._u, list)`` through
    ``compute_adjoint`` so block-extraction only runs for a genuinely blocked/mixed
    problem.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = mesh_2D
    gdim = mesh.geometry.dim
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1, (gdim,)))
    Z = dolfinx.fem.functionspace(mesh, ("DG", 0))

    # The control multiplies the bilinear form (not just the right-hand side), as in
    # a SIMP-style density-dependent stiffness -- the structure that first surfaced
    # this bug in a linear-elasticity topology optimization demo.
    kappa = Function(Z, name="control")
    kappa.interpolate(lambda x: 1.0 + 0.5 * np.sin(np.pi * x[0]))

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = kappa * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.as_vector((ufl.sin(ufl.pi * x[1]), ufl.cos(ufl.pi * x[0])))
    L = ufl.inner(f, v) * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc_val = dolfinx.fem.Constant(mesh, np.zeros((gdim,), dtype=dolfinx.default_scalar_type))
    bc = dolfinx.fem.dirichletbc(bc_val, boundary_dofs, V)

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

    # Quartic in the state, for the same round-off-avoidance reason as test_solver's
    # objective.
    J = assemble_scalar(ufl.inner(uh, uh) ** 2 * ufl.dx)

    control = pyadjoint.Control(kappa)
    Jh = pyadjoint.ReducedFunctional(J, control)
    d = Function(Z)
    d.interpolate(lambda x: 1.0 + 0.3 * np.cos(np.pi * x[1]))
    e = Function(Z)
    e.interpolate(lambda x: 0.2 * np.sin(3 * x[0]))

    min_rate = pyadjoint.taylor_test(Jh, d, e, dJdm=0)
    assert np.isclose(min_rate, 1.0, rtol=1e-1, atol=1e-1), f"Expected convergence rate close to 1.0, got {min_rate}"

    Jh.derivative()
    min_rate = pyadjoint.taylor_test(Jh, d, e)
    assert np.isclose(min_rate, 2.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 2.0, got {min_rate}"

    Jh(d)
    dJdm = Jh.derivative()._ad_dot(e)
    hessian = Jh.hessian(e)
    dHddu = hessian._ad_dot(e)
    min_rate = pyadjoint.taylor_test(Jh, d, e, dJdm=dJdm, Hm=dHddu)
    assert np.isclose(min_rate, 3.0, rtol=0.1, atol=0.1), f"Expected convergence rate close to 3.0, got {min_rate}"


@pytest.mark.parametrize("use_mixed_space", [True, False])
def test_nonlinear_solver(use_mixed_space: bool, mesh_2D):
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
        "snes_monitor": None,
        "snes_type": "newtonls",
        # Default ("bt") backtracking line search: globalizes genuinely-far Newton
        # steps (e.g. the largest Taylor-test perturbations). Disabling it
        # ("snes_linesearch_type": "none") let an undamped step stall -- SNES then
        # reported SNES_CONVERGED_SNORM_RELATIVE (step size below snes_stol) even
        # though the residual was still ~1e-1, and snes_error_if_not_converged only
        # raises on a *diverged* reason, so that false convergence passed silently.
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

        Jh(d)
        dJdm = Jh.derivative()._ad_dot(e)
        hessian = Jh.hessian(e)
        dHddu = hessian._ad_dot(e)
        min_rate = pyadjoint.taylor_test(Jh, d, e, dJdm=dJdm, Hm=dHddu)
        assert np.isclose(min_rate, 3.0, rtol=0.1, atol=0.1), f"Expected convergence rate close to 3.0, got {min_rate}"

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
        Jh(mu2)
        dJdm = Jh.derivative()._ad_dot(h2)
        hessian = Jh.hessian(h2)
        dHddu = hessian._ad_dot(h2)
        min_rate = pyadjoint.taylor_test(Jh, mu2, h2, dJdm=dJdm, Hm=dHddu)
        assert np.isclose(min_rate, 3.0, rtol=0.1, atol=0.1), f"Expected convergence rate close to 3.0, got {min_rate}"
