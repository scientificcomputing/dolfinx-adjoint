from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import ufl
from pyadjoint.overloaded_type import Weakref

from dolfinx_adjoint import Constant, Function, LinearProblem, assemble_scalar, assign, dirichletbc
from dolfinx_adjoint.blocks.dirichletbc import DirichletBCBlock
from dolfinx_adjoint.blocks.interpolation import ExprInterpolationBlock

direct_solve = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "ksp_error_if_not_converged": True,
    "pc_factor_mat_solver_type": "mumps",
}


def _poisson_bc_control_problem(mesh, g):
    """A scalar Poisson problem with `g` as the sole Dirichlet bc value on the whole
    boundary, no volumetric source -- shared by the Function- and Constant-valued
    control tests below, which differ only in what `g` is.
    """
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = ufl.inner(dolfinx.fem.Constant(mesh, 0.0), v) * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dirichletbc(g, boundary_dofs, V=V)

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
    J = assemble_scalar(ufl.inner(uh, uh) * ufl.dx)
    # Return `problem` too: the tape only holds a *weak* reference to it, and letting it
    # be garbage-collected before later replay/differentiation still works but silently
    # rebuilds an equivalent one each time (see pyadjoint's own warning) -- the caller
    # keeps it alive for the lifetime of the test to avoid that cost.
    return J, problem


def _assert_bc_control_gradient_and_hessian(Jhat, m0, h, *, hessian_atol):
    """Shared verification ladder for a Dirichlet bc control: gradient via the standard
    rate-2 (gradient-corrected) taylor_test, Hessian via a direct exact-quadratic
    remainder check rather than the standard rate-3 taylor_test.

    J is exactly quadratic in a Dirichlet bc control entering a *linear* PDE (the state
    is an affine function of the bc value, and J is itself quadratic in the state), so
    there is no cubic remainder for taylor_test's rate-3 check to measure -- it reduces
    to measuring pure floating-point noise, not a real signal (see the dxa boundary-
    control implementation notes / the emi_membrane_current_control demo's own Hessian
    verification for the same phenomenon). The mathematically meaningful check is that
    the *quadratic* Taylor model matches J exactly (to floating-point precision) for any
    perturbation size, not just asymptotically small ones.
    """
    Jhat(m0)
    min_rate = pyadjoint.taylor_test(Jhat, m0, h)
    assert np.isclose(min_rate, 2.0, rtol=1e-1, atol=1e-1), f"Expected gradient rate 2.0, got {min_rate}"

    Jhat(m0)
    J0 = float(Jhat(m0))
    dJdm0 = Jhat.derivative()._ad_dot(h)
    Hm0 = Jhat.hessian(h)._ad_dot(h)
    for scale in [1.0, 0.3, 0.1, 0.01]:
        perturbed_array = m0.x.array[:] + scale * h.x.array[:]
        if isinstance(m0, Constant):
            # Constant.__init__ infers value_shape from numpy.shape(c) -- a bare Python
            # float (matching how `g`/`g0` were originally constructed, `Constant(mesh,
            # 2.0)`) gives shape () for this scalar test, whereas the raw length-1 array
            # would give shape (1,) and mismatch it under ufl.replace.
            mp = Constant(m0.function_space.mesh, float(perturbed_array[0]))
        else:
            mp = Function(m0.function_space)
            mp.x.array[:] = perturbed_array
        Jp = float(Jhat(mp))
        predicted = J0 + scale * dJdm0 + 0.5 * scale**2 * Hm0
        assert np.isclose(Jp, predicted, rtol=0, atol=hessian_atol), (
            f"scale={scale}: exact 2nd-order remainder {Jp - predicted:.3e} exceeds {hessian_atol:.3e}"
        )


def test_scalar_dirichletbc_control_function_value():
    """Gradient and Hessian of a scalar Poisson objective w.r.t. a Function-valued
    Dirichlet bc, the case where DirichletBCBlock's dependency (bc.g, see
    types/dirichletbc.py::_pack_bc_value) already lives on exactly the same space as
    the user's own control -- no broadcast/reduction involved.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    g = Function(V, name="bc_value")
    g.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

    J, _problem = _poisson_bc_control_problem(mesh, g)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(g))

    g0 = Function(V)
    g0.x.array[:] = g.x.array
    h = Function(V)
    h.interpolate(lambda x: 100.0 * np.cos(np.pi * x[0]) * x[1])

    _assert_bc_control_gradient_and_hessian(Jhat, g0, h, hessian_atol=1e-2)


def test_scalar_dirichletbc_control_constant_value():
    """Gradient and Hessian of a scalar Poisson objective w.r.t. a Constant-valued
    Dirichlet bc: `g` lives on its own private single-dof real space, broadcast across
    every constrained dof, so DirichletBCBlock's dependency (bc.g, always packed onto
    the constrained state space V -- see types/dirichletbc.py::_pack_bc_value) differs
    in space from the user's own control `g` here. The broadcast/reduction between the
    two is handled entirely by ExprInterpolationBlock's own adjoint machinery (packing
    `g` is a `ufl.as_ufl` no-op, but it is still routed through the same interpolation
    machinery as the Function case), not by any bc-specific code.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)

    g = Constant(mesh, 2.0)
    J, _problem = _poisson_bc_control_problem(mesh, g)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(g))

    g0 = Constant(mesh, 2.0)
    h = Constant(mesh, 1.0)

    _assert_bc_control_gradient_and_hessian(Jhat, g0, h, hessian_atol=1e-6)


def test_scalar_dirichletbc_general_expression():
    """A Dirichlet bc value built from a genuine nonlinear UFL expression of a control
    (``m**3``, mirroring legacy dolfin-adjoint's ``test_simple_expression`` -- ported
    directly to a UFL expression rather than a c-string ``Expression`` with a hand-
    supplied ``user_defined_derivatives``, since ``ExprInterpolationBlock`` already
    differentiates a genuine UFL expression automatically, see
    types/dirichletbc.py::_pack_bc_value).

    Unlike the bare Function/Constant cases (test_scalar_dirichletbc_control_*_value),
    the state -- and hence J -- is genuinely *cubic* (not quadratic) in `m` here, so the
    standard rate-3 taylor_test is a real, meaningful signal (not floating-point noise).
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)

    m = Constant(mesh, 1.5)
    J, _problem = _poisson_bc_control_problem(mesh, m**3)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(m))

    m0 = Constant(mesh, 1.5)
    h = Constant(mesh, 1.0)

    Jhat(m0)
    min_rate = pyadjoint.taylor_test(Jhat, m0, h, dJdm=0)
    assert np.isclose(min_rate, 1.0, rtol=1e-1, atol=1e-1), f"Expected rate 1.0, got {min_rate}"

    Jhat(m0)
    min_rate = pyadjoint.taylor_test(Jhat, m0, h)
    assert np.isclose(min_rate, 2.0, rtol=1e-1, atol=1e-1), f"Expected rate 2.0, got {min_rate}"

    Jhat(m0)
    dJdm = Jhat.derivative()._ad_dot(h)
    Hm = Jhat.hessian(h)._ad_dot(h)
    min_rate = pyadjoint.taylor_test(Jhat, m0, h, dJdm=dJdm, Hm=Hm)
    assert np.isclose(min_rate, 3.0, rtol=1e-1, atol=1e-1), f"Expected rate 3.0, got {min_rate}"


def test_dirichletbc_recording():
    """Test that creating an overloaded dirichletbc correctly registers both blocks it
    always produces: the ExprInterpolationBlock that packs the value into bc.g (see
    types/dirichletbc.py::_pack_bc_value -- used even for a bare Function value), then
    the DirichletBCBlock itself."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    c = Function(V, name="boundary_value")
    c.interpolate(lambda x: x[0])

    dofs = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = dirichletbc(c, dofs)

    tape = pyadjoint.get_working_tape()
    blocks = tape.get_blocks()

    assert len(blocks) == 2
    assert isinstance(blocks[0], ExprInterpolationBlock)
    assert isinstance(blocks[1], DirichletBCBlock)

    # The interpolation block's dependency is the user's own control, `c`.
    assert len(blocks[0].get_dependencies()) == 1
    assert blocks[0].get_dependencies()[0].output is c

    # The DirichletBCBlock's single dependency is the packed value, bc.g -- not `c`
    # itself; pyadjoint.Control(c) still works end to end via ordinary tape chaining
    # through the interpolation block.
    assert len(blocks[1].get_dependencies()) == 1
    assert blocks[1].get_dependencies()[0].output is bc.g

    # The returned BC object should now possess the injected block_variable
    assert hasattr(bc, "block_variable")


def test_dirichletbc_no_annotate():
    """Test that setting annotate=False bypasses tape recording entirely."""

    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    c = Function(V, name="boundary_value")
    c.interpolate(lambda x: x[0])

    dofs = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))

    # Run with annotation off
    bc = dirichletbc(c, dofs, annotate=False)

    tape = pyadjoint.get_working_tape()

    assert len(tape.get_blocks()) == 0
    # FIX: Check the underlying weak reference rather than invoking the property
    assert getattr(bc, "_block_variable", Weakref())() is None


def test_dirichletbc_recompute():
    """Position-aware recompute: replaying the tape at several different control values
    must each refresh bc.g's *live* array to the value belonging to that specific
    position, not silently keep whatever the live array happens to already hold.

    DirichletBC._ad_create_checkpoint/_ad_restore_at_checkpoint both `return self` --
    the bc's own "checkpoint" aliases the live bc object, so this can only pass because
    DirichletBCBlock.recompute_component explicitly resyncs bc.g's array from its own
    dependency's (correctly, weakly checkpointed) recomputed value every time -- see
    blocks/dirichletbc.py. Reading `bc.g` directly inside the functional (rather than
    mutating it from outside the tape and observing the same aliased object, as an
    earlier version of this test did) is what makes it possible for this test to fail
    if that resync were ever broken.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    c = Function(V, name="boundary_value")
    c.interpolate(lambda x: np.full_like(x[0], 5.0))

    dofs = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = dirichletbc(c, dofs)
    assert isinstance(pyadjoint.get_working_tape().get_blocks()[-1], DirichletBCBlock)

    J = assemble_scalar(bc.g**2 * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(c))

    for value in [5.0, 15.0, 42.0]:
        c_new = Function(V)
        c_new.interpolate(lambda x, value=value: np.full_like(x[0], value))
        J_value = float(Jhat(c_new))
        assert np.isclose(J_value, value**2)
        assert np.allclose(bc.g.x.array, value)


def test_time_dependent_bc_replay():
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    dt = 0.1
    num_steps = 3

    m = Function(V, name="control")
    m.interpolate(lambda x: np.sin(x[0] * np.pi))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    uh = Function(V, name="state")
    assign(0.0, uh)

    u_prev = Function(V, name="state_prev")
    assign(0.0, u_prev)

    F = (u - u_prev) / dt * v * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - m * v * ufl.dx
    a, L = ufl.system(F)

    bc_func = Function(V, name="bc_func")
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)

    # Use native dolfinx here! PyAdjoint traces the bc_func inside it.
    bc = dirichletbc(bc_func, boundary_dofs)
    petsc_options = {
        "ksp_monitor": None,
        "ksp_type": "preonly",
        "pc_type": "lu",
        "ksp_error_if_not_converged": True,
        "pc_factor_mat_solver_type": "mumps",
    }
    problem = LinearProblem(a, L, bcs=[bc], u=uh, petsc_options=petsc_options)

    J = 0.0

    for i in range(num_steps):
        assign(float(i + 1), bc_func)
        problem.solve()
        J += assemble_scalar(0.5 * ufl.inner(uh, uh) * ufl.dx)
        assign(uh, u_prev)

    J_forward = float(J)

    control = pyadjoint.Control(m)
    Jhat = pyadjoint.ReducedFunctional(J, control)
    J_replay = Jhat(m)

    assert np.isclose(J_replay, J_forward, atol=1e-10, rtol=1e-10)

    # Gradient/Taylor equivalence through the multi-solve() loop: replaying,
    # differentiating, and perturbing a control across several timesteps on
    # one shared LinearProblem must give the same Taylor convergence as
    # before the solver/form-reuse refactor (see the module-level plan in
    # dolfinx-adjoint-knowledge's solver-reuse spec, "Behaviour is
    # unchanged").
    # J is dominated by the (non-controlled) time-varying Dirichlet value, so
    # J's sensitivity to m is comparatively small: scale the perturbation up
    # so that pyadjoint.taylor_test's fixed step sizes resolve the quadratic
    # remainder well above solver/roundoff noise (matching the disproportionate
    # perturbation-to-control scaling already used in
    # test_nonlinear_problem.py's own taylor tests).
    pert = Function(V)
    pert.interpolate(lambda x: 20.0 * np.cos(x[1] * np.pi))

    Jhat(m)
    pyadjoint.taylor_test(Jhat, m, pert, dJdm=0)
    Jhat(m)
    min_rate_grad = pyadjoint.taylor_test(Jhat, m, pert)
    assert np.isclose(min_rate_grad, 2.0, rtol=1e-2, atol=5e-2), f"Expected 2.0, got {min_rate_grad}"
