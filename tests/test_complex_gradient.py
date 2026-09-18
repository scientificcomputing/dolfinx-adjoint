"""Gradients of a real-valued Functional through a complex-valued (sesquilinear) PDE solve.

Under a complex-PETSc build a Control is real-valued -- complex-valued controls are out of
scope -- but is represented as a complex Function whose imaginary part is identically zero,
which is simply what constructing one with the default dtype gives. The gradient of a real
Functional with respect to such a control is ``2*Re[lambda^H . dR/dm]``, and that extraction
happens in exactly one place: ``Function._ad_convert_riesz``, which pyadjoint calls only on a
Control's own object. These tests pin that down against finite differences.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl
from ufl.algorithms.check_arities import ArityMismatch

from dolfinx_adjoint import Constant, Function, assemble_scalar, assign, dirichletbc, interpolate
from dolfinx_adjoint.solvers import LinearProblem, NonlinearProblem
from dolfinx_adjoint.utils import scalar_type_mismatch_message

complex_only = pytest.mark.skipif(
    not np.issubdtype(dolfinx.default_scalar_type, np.complexfloating),
    reason="Exercises the complex-scalar adjoint path.",
)

pytestmark = complex_only

_PETSC_LU = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "ksp_error_if_not_converged": True,
    "pc_factor_mat_solver_type": "mumps",
}


@pytest.fixture
def mesh():
    return dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 20)


@pytest.fixture
def V(mesh):
    return dolfinx.fem.functionspace(mesh, ("Lagrange", 1))


@pytest.fixture
def target(V):
    """A fixed, untracked profile to build a misfit against.

    Deliberately complex-valued: a real-valued target equals its own conjugate, which hides
    every error that consists of conjugating (or failing to conjugate) the adjoint seed.
    """
    t = dolfinx.fem.Function(V)
    t.interpolate(lambda x: np.exp(x[0]) + 1j * np.sin(x[0]))
    return t


def _solve_helmholtz(mesh, V, source, potential=None) -> tuple[LinearProblem, Function]:
    """A complex sesquilinear Helmholtz solve driven by ``source``.

    ``k`` is given a small imaginary part so the operator is genuinely complex (damped)
    rather than a real operator that merely happens to be stored in a complex dtype.

    ``potential``, when given, is an extra coefficient inside the *operator*. A control in
    ``source`` alone leaves the solve affine in the control, so its Hessian is exactly zero
    and would validate nothing; a control in the potential makes the state a genuinely
    nonlinear function of it while the PDE stays linear in the state.
    """
    uh = Function(V, name="state")
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = dolfinx.default_scalar_type(4.0 + 0.5j)  # type: ignore[arg-type]
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * ufl.inner(u, v)) * ufl.dx
    if potential is not None:
        a += ufl.inner(potential * u, v) * ufl.dx
    L = ufl.inner(source, v) * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)),
        dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets),
        V,
    )

    problem = LinearProblem(a, L, u=uh, bcs=[bc], petsc_options=_PETSC_LU, adjoint_petsc_options=_PETSC_LU)
    problem.solve()
    # Returned, not dropped: pyadjoint replays this block during the adjoint sweep, and a
    # garbage-collected LinearProblem forces it to rebuild an equivalent one.
    return problem, uh


def _central_difference(Jhat, m, perturbation, eps=1.0e-3):
    """Central difference of ``Jhat`` at ``m`` along ``perturbation``.

    Used where the Functional is quadratic in the control rather than affine -- a misfit that
    is not linear in the state -- for which a central difference is still exact but for
    roundoff, while the forward difference below carries an O(eps) error.
    """
    values = []
    for step in (eps, -eps):
        m_step = m._ad_copy()
        m_step.x.array[:] = m.x.array[:] + step * perturbation.x.array[:]
        values.append(float(Jhat(m_step)))
    return (values[0] - values[1]) / (2 * eps)


def _finite_difference(Jhat, m, perturbation, J0, eps=1.0e-3):
    """Forward difference of ``Jhat`` at ``m`` along ``perturbation``.

    The problems here are exactly affine in the control (a linear PDE with the control in the
    right-hand side only, paired with a Functional linear in the state), so ``J(m + eps*h)``
    is exactly linear in ``eps``. That leaves no curvature for ``taylor_test``'s second-order
    remainder to converge against -- it degenerates into floating-point noise -- while a
    forward difference is exact but for roundoff at any reasonable step size.
    """
    m_plus = m._ad_copy()
    m_plus.x.array[:] = m.x.array[:] + eps * perturbation.x.array[:]
    return (float(Jhat(m_plus)) - float(J0)) / eps


def test_function_control_gradient_matches_finite_difference(mesh, V, target):
    pyadjoint.get_working_tape().clear_tape()
    # A random (not reflection-symmetric) profile: this domain, BC and operator are symmetric
    # about x=0.5, so a symmetric base point with an antisymmetric perturbation would make the
    # true first-order term vanish by parity and accidentally validate a wrong gradient.
    rng = np.random.default_rng(0)

    f = Function(V, name="control")
    f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
    f.x.scatter_forward()

    problem, uh = _solve_helmholtz(mesh, V, f)

    # A holomorphic-in-state misfit against a fixed target: ufl.inner conjugates its *second*
    # argument, so inner(uh, target) is holomorphic in uh. The anti-holomorphic and the
    # genuinely non-holomorphic shapes are covered below.
    J = assemble_scalar(ufl.inner(uh, target) * ufl.dx)
    assert isinstance(J, float)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))

    h = Function(V)
    h.x.array[:] = rng.uniform(-1.0, 1.0, size=h.x.array.shape)
    h.x.scatter_forward()

    dJdm = Jhat.derivative()._ad_dot(h)
    assert isinstance(dJdm, float), f"_ad_dot should return a real scalar, got {type(dJdm)}"
    assert np.isclose(dJdm, _finite_difference(Jhat, f, h, J), rtol=1e-6, atol=1e-8)
    del problem


def test_gradient_of_real_functional_is_real(mesh, V, target):
    pyadjoint.get_working_tape().clear_tape()
    rng = np.random.default_rng(1)

    f = Function(V, name="control")
    f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
    f.x.scatter_forward()

    problem, uh = _solve_helmholtz(mesh, V, f)
    J = assemble_scalar(ufl.inner(uh, target) * ufl.dx)
    grad = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f)).derivative()

    # The gradient is mathematically real, but stays in the control's complex dtype so that
    # `control + step * gradient` in an optimisation loop does not mix dtypes.
    assert np.issubdtype(grad.x.array.dtype, np.complexfloating)
    assert np.allclose(grad.x.array.imag, 0.0, atol=1e-12), f"max |imag| = {np.abs(grad.x.array.imag).max()}"
    del problem


def test_constant_control_gradient_matches_finite_difference(mesh, V, target):
    """A `Constant` control, which the previous Block-level gating never reached.

    `Constant` inherits `_ad_convert_riesz` from `Function` unchanged, so putting the
    extraction there covers it with no `Constant`-specific code.
    """
    pyadjoint.get_working_tape().clear_tape()

    alpha = Constant(mesh, 2.0)
    f = Function(V, name="source")
    f.interpolate(lambda x: np.sin(np.pi * x[0]) + 0.3 * x[0])

    problem, uh = _solve_helmholtz(mesh, V, alpha * f)
    J = assemble_scalar(ufl.inner(uh, target) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(alpha))

    h = alpha._ad_copy()
    h.x.array[:] = 1.0
    dJdm = Jhat.derivative()._ad_dot(h)
    assert np.isclose(dJdm, _finite_difference(Jhat, alpha, h, J), rtol=1e-6, atol=1e-8)
    del problem


def test_anti_holomorphic_functional_gradient_matches_finite_difference(mesh, V, target):
    """`inner(target, uh)` is anti-holomorphic in `uh`, since `ufl.inner` conjugates its
    *second* argument -- so argument order alone decides whether a misfit is holomorphic in
    the state, a slip a user cannot see. Both orders have to give the right gradient.
    """
    pyadjoint.get_working_tape().clear_tape()
    rng = np.random.default_rng(2)

    f = Function(V, name="control")
    f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
    f.x.scatter_forward()

    problem, uh = _solve_helmholtz(mesh, V, f)
    J = assemble_scalar(ufl.inner(target, uh) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))

    h = Function(V)
    h.x.array[:] = rng.uniform(-1.0, 1.0, size=h.x.array.shape)
    h.x.scatter_forward()

    dJdm = Jhat.derivative()._ad_dot(h)
    assert np.isclose(dJdm, _finite_difference(Jhat, f, h, J), rtol=1e-6, atol=1e-8)
    del problem


def test_non_holomorphic_misfit_gradient_matches_finite_difference(mesh, V, target):
    """The standard misfit `|uh - target|**2`, which is neither holomorphic nor
    anti-holomorphic in the state: `uh` appears conjugated in some terms and not in others.

    Its derivative is not determined by the holomorphic part alone, so the adjoint seed has to
    carry both Wirtinger derivatives -- which is why the seed is assembled from a derivative in
    the real direction *and* one in the imaginary direction.
    """
    pyadjoint.get_working_tape().clear_tape()
    rng = np.random.default_rng(3)

    f = Function(V, name="control")
    f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
    f.x.scatter_forward()

    problem, uh = _solve_helmholtz(mesh, V, f)
    J = assemble_scalar(ufl.inner(uh - target, uh - target) * ufl.dx)
    assert isinstance(J, float)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))

    h = Function(V)
    h.x.array[:] = rng.uniform(-1.0, 1.0, size=h.x.array.shape)
    h.x.scatter_forward()

    dJdm = Jhat.derivative()._ad_dot(h)
    assert np.isclose(dJdm, _central_difference(Jhat, f, h), rtol=1e-6, atol=1e-10)
    del problem


def test_gradient_is_unchanged_by_the_order_of_a_squared_misfit(mesh, V, target):
    """`inner(a, b)` and `inner(b, a)` are conjugates of one another, so a misfit built from
    either order has the same real part -- and therefore the same gradient. A seed that
    conjugates one Wirtinger part but not the other would split them apart.
    """
    pyadjoint.get_working_tape().clear_tape()
    rng = np.random.default_rng(4)

    gradients = []
    for order in (lambda a, b: ufl.inner(a, b), lambda a, b: ufl.inner(b, a)):
        pyadjoint.get_working_tape().clear_tape()
        f = Function(V, name="control")
        f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
        f.x.scatter_forward()
        problem, uh = _solve_helmholtz(mesh, V, f)
        J = assemble_scalar(order(uh, target) * ufl.dx)
        gradients.append(pyadjoint.ReducedFunctional(J, pyadjoint.Control(f)).derivative().x.array.copy())
        del problem
        rng = np.random.default_rng(4)

    assert np.allclose(gradients[0], gradients[1])


def test_gradient_through_an_interpolation_step_matches_finite_difference(mesh, V, target):
    """A Control reaching the Functional through an interpolation into another space.

    Interpolation between two spaces is a real linear map, but the field it is applied to is
    complex, and the matrix DOLFINx builds for it is real-valued even under a complex build --
    so the adjoint of this step has to apply a real operator to a complex vector without
    dropping half of it.
    """
    pyadjoint.get_working_tape().clear_tape()
    rng = np.random.default_rng(5)

    W = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))

    f = Function(V, name="control")
    f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
    f.x.scatter_forward()

    problem, uh = _solve_helmholtz(mesh, V, f)
    # The state is genuinely complex here (a damped Helmholtz solve), so an interpolation
    # that dropped its imaginary part would still produce a plausible number.
    target_W = dolfinx.fem.Function(W)
    target_W.interpolate(target)

    interpolated = interpolate(uh, W)
    J = assemble_scalar(ufl.inner(interpolated, target_W) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))

    h = Function(V)
    h.x.array[:] = rng.uniform(-1.0, 1.0, size=h.x.array.shape)
    h.x.scatter_forward()

    dJdm = Jhat.derivative()._ad_dot(h)
    assert np.isclose(dJdm, _finite_difference(Jhat, f, h, J), rtol=1e-6, atol=1e-8)
    del problem


def test_gradient_through_a_complex_coefficient_interpolation_matches_finite_difference(mesh, V, target):
    """An interpolated *expression* carrying a complex coefficient.

    Unlike interpolation between two spaces, which is a real linear map, the operator behind
    `interpolate(alpha * uh, W)` has genuinely complex entries. Its adjoint action is then the
    Hermitian transpose, not the plain one -- a plain transpose returns a seed conjugated
    relative to every other seed in the package, which the terminal `2*Re[.]` cannot repair.
    """
    pyadjoint.get_working_tape().clear_tape()
    rng = np.random.default_rng(6)

    W = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    target_W = dolfinx.fem.Function(W)
    target_W.interpolate(target)
    alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(2.0 + 1.0j))  # type: ignore[arg-type]

    f = Function(V, name="control")
    f.x.array[:] = rng.uniform(-1.0, 1.0, size=f.x.array.shape)
    f.x.scatter_forward()

    problem, uh = _solve_helmholtz(mesh, V, f)
    J = assemble_scalar(ufl.inner(interpolate(alpha * uh, W), target_W) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))

    h = Function(V)
    h.x.array[:] = rng.uniform(-1.0, 1.0, size=h.x.array.shape)
    h.x.scatter_forward()

    dJdm = Jhat.derivative()._ad_dot(h)
    assert np.isclose(dJdm, _finite_difference(Jhat, f, h, J), rtol=1e-6, atol=1e-8)
    del problem


def _random_pair(V, seed, low=-1.0, high=1.0):
    """A base point and a direction, both real-valued but stored in the build's dtype."""
    rng = np.random.default_rng(seed)
    made = []
    for _ in range(2):
        g = Function(V)
        g.x.array[:] = rng.uniform(low, high, size=g.x.array.shape)
        g.x.scatter_forward()
        made.append(g)
    return made


def test_hessian_through_a_linear_solve_matches_finite_difference(
    mesh, V, target, assert_hessian_matches_finite_difference
):
    """A Functional holomorphic in the state, with the control inside the operator.

    Holomorphic because `ufl.inner` conjugates its *second* argument, so `inner(uh, target)`
    carries `uh` unconjugated throughout. The control sits in the operator rather than the
    right-hand side because a control in the right-hand side leaves this Functional affine in
    it, and an exactly zero Hessian would pass any accuracy check for free.
    """
    pyadjoint.get_working_tape().clear_tape()
    f, h = _random_pair(V, seed=10)

    problem, uh = _solve_helmholtz(mesh, V, ufl.as_ufl(1.0), potential=f)
    J = assemble_scalar(ufl.inner(uh, target) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))

    assert_hessian_matches_finite_difference(Jhat, f, h, fd_eps=1e-4, rtol=1e-5, atol=1e-12)
    assert abs(Jhat.hessian(h)._ad_dot(h)) > 0.0, "a zero Hessian would satisfy the check for free"
    del problem


def test_hessian_of_a_non_holomorphic_misfit_matches_finite_difference(
    mesh, V, target, assert_hessian_matches_finite_difference
):
    """`|uh - target|**2`, the misfit people actually write.

    Neither holomorphic nor anti-holomorphic in the state, so -- exactly as at first order --
    the adjoint seed has to carry both Wirtinger derivatives. At second order it is the
    *directional derivative* of the Functional that is seeded that way, and the two Wirtinger
    parts are taken of that.
    """
    pyadjoint.get_working_tape().clear_tape()
    f, h = _random_pair(V, seed=11)

    problem, uh = _solve_helmholtz(mesh, V, f)
    J = assemble_scalar(ufl.inner(uh - target, uh - target) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))

    assert_hessian_matches_finite_difference(Jhat, f, h, fd_eps=1e-4, rtol=1e-5, atol=1e-12)
    assert abs(Jhat.hessian(h)._ad_dot(h)) > 0.0
    del problem


def test_hessian_of_a_state_intensity_matches_finite_difference(mesh, V, assert_hessian_matches_finite_difference):
    """`|uh|**2` with the control in the operator, which is the metalens demo's shape.

    Both sources of curvature at once: the Functional is quadratic in the state and the state
    is a nonlinear function of the control.
    """
    pyadjoint.get_working_tape().clear_tape()
    f, h = _random_pair(V, seed=12)

    problem, uh = _solve_helmholtz(mesh, V, ufl.as_ufl(1.0), potential=f)
    J = assemble_scalar(ufl.inner(uh, uh) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))

    assert_hessian_matches_finite_difference(Jhat, f, h, fd_eps=1e-4, rtol=1e-5, atol=1e-12)
    assert abs(Jhat.hessian(h)._ad_dot(h)) > 0.0
    del problem


def test_hessian_is_unchanged_by_the_order_of_a_misfit(mesh, V, target):
    """`inner(a, b)` and `inner(b, a)` are conjugates of one another, so a misfit built from
    either order has the same real part -- and therefore the same Hessian. The second-order
    counterpart of `test_gradient_is_unchanged_by_the_order_of_a_squared_misfit`, and the check
    that would catch a seed conjugating one Wirtinger part but not the other.

    The swap has to be of `ufl.inner`'s own two slots, which is what makes the two functionals a
    conjugate pair; negating both arguments of one `inner` (`inner(u - t, u - t)` against
    `inner(t - u, t - u)`) looks like a swap but is the identity, since `inner` conjugates its
    second slot and the two signs cancel.

    The control goes in the operator because `inner(uh, target)` is linear in the state: with
    the control in the right-hand side the Hessian would be exactly zero, and two zeros agree
    for free.
    """
    products = []
    for order in (lambda a, b: ufl.inner(a, b), lambda a, b: ufl.inner(b, a)):
        pyadjoint.get_working_tape().clear_tape()
        f, h = _random_pair(V, seed=13)
        problem, uh = _solve_helmholtz(mesh, V, ufl.as_ufl(1.0), potential=f)
        Jhat = pyadjoint.ReducedFunctional(assemble_scalar(order(uh, target) * ufl.dx), pyadjoint.Control(f))
        Jhat.derivative()
        products.append(Jhat.hessian(h)._ad_dot(h))
        del problem

    assert np.isclose(products[0], products[1], rtol=1e-10, atol=1e-14)
    assert abs(products[0]) > 0.0


def test_hessian_with_a_constant_control_matches_finite_difference(
    mesh, V, target, assert_hessian_matches_finite_difference
):
    """A `Constant` control, which reaches the Hessian through the Real space rather than a
    mesh-resolved one -- the same distinction the first-order suite carries.
    """
    pyadjoint.get_working_tape().clear_tape()

    alpha = Constant(mesh, 2.0)
    source = Function(V, name="source")
    source.interpolate(lambda x: np.sin(np.pi * x[0]) + 0.3 * x[0])

    problem, uh = _solve_helmholtz(mesh, V, source, potential=alpha)
    Jhat = pyadjoint.ReducedFunctional(assemble_scalar(ufl.inner(uh, target) * ufl.dx), pyadjoint.Control(alpha))

    h = alpha._ad_copy()
    h.x.array[:] = 1.0
    assert_hessian_matches_finite_difference(Jhat, alpha, h, fd_eps=1e-4, rtol=1e-5, atol=1e-12)
    assert abs(Jhat.hessian(h)._ad_dot(h)) > 0.0
    del problem


def test_hessian_through_a_nonlinear_solve_matches_finite_difference(
    mesh, V, target, assert_hessian_matches_finite_difference
):
    """A residual with a genuine second derivative in the state.

    `uh**3` is holomorphic in `uh`, which is what keeps it inside the Sesquilinear phase: a
    holomorphic residual has a holomorphic Jacobian and a holomorphic second derivative, so
    the second-order-adjoint self-term needs no Wirtinger splitting of its own. The linear
    solves above leave that term structurally zero, so this is the only test here that
    exercises it.
    """
    pyadjoint.get_working_tape().clear_tape()
    f, h = _random_pair(V, seed=14, low=0.4, high=0.6)

    uh = Function(V, name="state")
    v = ufl.TestFunction(V)
    k = dolfinx.default_scalar_type(4.0 + 0.5j)  # type: ignore[arg-type]
    residual = (
        ufl.inner(ufl.grad(uh), ufl.grad(v)) - k**2 * ufl.inner(uh, v) + ufl.inner(uh**3, v) - ufl.inner(f, v)
    ) * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.0)),
        dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets),
        V,
    )
    problem = NonlinearProblem(
        residual,
        uh,
        bcs=[bc],
        petsc_options=_PETSC_LU,
        adjoint_petsc_options=_PETSC_LU,
        tlm_petsc_options=_PETSC_LU,
    )
    problem.solve()

    Jhat = pyadjoint.ReducedFunctional(
        assemble_scalar(ufl.inner(uh - target, uh - target) * ufl.dx), pyadjoint.Control(f)
    )
    assert_hessian_matches_finite_difference(Jhat, f, h, fd_eps=1e-4, rtol=1e-5, atol=1e-12)
    assert abs(Jhat.hessian(h)._ad_dot(h)) > 0.0
    del problem


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.skipif(
    MPI.COMM_WORLD.size > 1,
    reason=(
        "Deadlocks in parallel, upstream: dolfinx.jit.mpi_jit_decorator has rank 0 compile and "
        "broadcast the outcome under `except Exception`, but UFL raises ArityMismatch from "
        "BaseException -- so rank 0 leaves the decorator without reaching its bcast while every "
        "other rank waits there forever. A user hitting this residual under MPI hangs rather "
        "than seeing the rejection; nothing in this package can repair a collective that one "
        "rank has already walked out of."
    ),
)
def test_non_holomorphic_residual_is_refused(mesh, V):
    """A residual non-holomorphic in the state stays out of scope, and says so by raising.

    The Kerr term `|uh|**2 * uh` is the canonical case. Its Jacobian carries the test function
    both conjugated and not, which UFL's complex-mode arity rules reject outright -- so the
    refusal is UFL's, it arrives at Problem construction long before any tape exists, and
    there is no path by which such a residual could return an unverified number.

    The filter is for the construction failing *partway*: DOLFINx's own
    `NonlinearProblem.__del__` unconditionally reads attributes its `__init__` had not reached,
    so collecting the half-built object raises inside a destructor. Upstream, and unrelated to
    what is being asserted here.
    """
    pyadjoint.get_working_tape().clear_tape()

    f = Function(V, name="control")
    f.x.array[:] = 0.5
    uh = Function(V, name="state")
    v = ufl.TestFunction(V)
    residual = (ufl.inner(ufl.grad(uh), ufl.grad(v)) + ufl.inner(uh * ufl.conj(uh) * uh, v) - ufl.inner(f, v)) * ufl.dx

    with pytest.raises(ArityMismatch):
        NonlinearProblem(residual, uh, bcs=[], petsc_options=_PETSC_LU)


def test_hessian_through_an_interpolation_step_matches_finite_difference(
    mesh, V, target, assert_hessian_matches_finite_difference
):
    """The second-order counterpart of the interpolation gradient test above: a real linear
    map applied to a genuinely complex field, now carrying a second-order seed.
    """
    pyadjoint.get_working_tape().clear_tape()
    f, h = _random_pair(V, seed=15)

    W = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    target_W = dolfinx.fem.Function(W)
    target_W.interpolate(target)

    problem, uh = _solve_helmholtz(mesh, V, ufl.as_ufl(1.0), potential=f)
    interpolated = interpolate(uh, W)
    Jhat = pyadjoint.ReducedFunctional(
        assemble_scalar(ufl.inner(interpolated, target_W) * ufl.dx), pyadjoint.Control(f)
    )

    assert_hessian_matches_finite_difference(Jhat, f, h, fd_eps=1e-4, rtol=1e-5, atol=1e-12)
    assert abs(Jhat.hessian(h)._ad_dot(h)) > 0.0
    del problem


def test_hessian_through_a_complex_coefficient_interpolation_matches_finite_difference(
    mesh, V, target, assert_hessian_matches_finite_difference
):
    """An interpolated *expression* carrying a complex coefficient, whose operator has
    genuinely complex entries -- so its second-order seed, like its first-order one, has to
    travel back through the Hermitian transpose rather than the plain one.
    """
    pyadjoint.get_working_tape().clear_tape()
    f, h = _random_pair(V, seed=16)

    W = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    target_W = dolfinx.fem.Function(W)
    target_W.interpolate(target)
    alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(2.0 + 1.0j))  # type: ignore[arg-type]

    problem, uh = _solve_helmholtz(mesh, V, ufl.as_ufl(1.0), potential=f)
    Jhat = pyadjoint.ReducedFunctional(
        assemble_scalar(ufl.inner(interpolate(alpha * uh, W), target_W) * ufl.dx),
        pyadjoint.Control(f),
    )

    assert_hessian_matches_finite_difference(Jhat, f, h, fd_eps=1e-4, rtol=1e-5, atol=1e-12)
    assert abs(Jhat.hessian(h)._ad_dot(h)) > 0.0
    del problem


def test_hessian_through_a_linear_combination_assignment_matches_finite_difference(
    mesh, V, target, assert_hessian_matches_finite_difference
):
    """A control reaching the operator through `assign(3 * f - g, ...)`.

    An assignment mostly hands a seed onward, so the risk here is the places where a value is
    converted to a Python float or a real dtype on the way through -- which at first order was
    where the complex path broke.
    """
    pyadjoint.get_working_tape().clear_tape()
    f, h = _random_pair(V, seed=17)
    g = Function(V, name="offset")
    g.interpolate(lambda x: 0.25 * np.cos(np.pi * x[0]))

    potential = Function(V, name="potential")
    assign(3 * f - g, potential)

    problem, uh = _solve_helmholtz(mesh, V, ufl.as_ufl(1.0), potential=potential)
    Jhat = pyadjoint.ReducedFunctional(assemble_scalar(ufl.inner(uh, target) * ufl.dx), pyadjoint.Control(f))

    assert_hessian_matches_finite_difference(Jhat, f, h, fd_eps=1e-4, rtol=1e-5, atol=1e-12)
    assert abs(Jhat.hessian(h)._ad_dot(h)) > 0.0
    del problem


def test_hessian_with_a_dirichletbc_control_matches_finite_difference(
    mesh, V, assert_hessian_matches_finite_difference
):
    """A Dirichlet boundary value as the control.

    A bc contributes no `d2F/dm2` or `d2F/dudm` of its own, so its whole Hessian contribution
    is the boundary reaction taken against the *second-order* adjoint solution -- a different
    code path from every other control here, and one that at first order had to be masked onto
    the bc's own dofs.
    """
    pyadjoint.get_working_tape().clear_tape()

    boundary_value = Function(V, name="boundary_value")
    boundary_value.interpolate(lambda x: 0.5 + x[0])
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    bc = dirichletbc(boundary_value, dofs)

    uh = Function(V, name="state")
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    k = dolfinx.default_scalar_type(4.0 + 0.5j)  # type: ignore[arg-type]
    a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - k**2 * ufl.inner(u, v)) * ufl.dx
    L = ufl.inner(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), v) * ufl.dx
    problem = LinearProblem(
        a, L, u=uh, bcs=[bc], petsc_options=_PETSC_LU, adjoint_petsc_options=_PETSC_LU, tlm_petsc_options=_PETSC_LU
    )
    problem.solve()

    Jhat = pyadjoint.ReducedFunctional(assemble_scalar(ufl.inner(uh, uh) * ufl.dx), pyadjoint.Control(boundary_value))

    h = Function(V)
    h.x.array[:] = 1.0
    h.x.scatter_forward()
    assert_hessian_matches_finite_difference(Jhat, boundary_value, h, fd_eps=1e-4, rtol=1e-5, atol=1e-12)
    assert abs(Jhat.hessian(h)._ad_dot(h)) > 0.0
    del problem


def test_assemble_scalar_accepts_a_compiled_form(mesh, V):
    """`assemble_scalar` is the single rank-0 path, so it must take a compiled form too."""
    u = dolfinx.fem.Function(V)
    u.x.array[:] = 2.0
    compiled = dolfinx.fem.form(ufl.inner(u, u) * ufl.dx)

    value = assemble_scalar(compiled, annotate=False)
    assert isinstance(value, float)
    assert np.isclose(value, 4.0)

    with pytest.raises(ValueError, match="already-compiled"):
        assemble_scalar(compiled)


def test_mixed_scalar_type_form_names_the_offending_coefficient(mesh, V):
    """A real/complex dtype mix must say so, not fail deep in the nanobind bindings."""
    complex_fn = dolfinx.fem.Function(V, name="complex_one")
    complex_fn.x.array[:] = 1.0
    real_fn = dolfinx.fem.Function(V, name="real_one", dtype=np.float64)
    real_fn.x.array[:] = 1.0

    with pytest.raises((TypeError, RuntimeError), match="mixes real- and complex-dtype"):
        assemble_scalar(ufl.inner(complex_fn, real_fn) * ufl.dx, annotate=False)


@pytest.mark.parametrize(
    "not_a_ufl_form",
    [
        pytest.param(None, id="absent-preconditioner"),
        pytest.param(0, id="every-term-dropped"),
        pytest.param("compiled", id="already-compiled-form"),
        pytest.param("zero-base-form", id="ufl-ZeroBaseForm"),
    ],
)
def test_dtype_diagnostic_declines_rather_than_raises_on_a_non_form(mesh, V, not_a_ufl_form):
    """The dtype diagnostic must never raise, because it runs inside an ``except`` block.

    {py:func}`dolfinx_adjoint.utils.scalar_type_mismatch_message` is reached only from
    ``_explaining_scalar_type_mismatch``, i.e. while a real failure is already propagating. Not
    everything handed to it is a ``ufl.Form`` or a nesting of them: a blocked problem passes
    ``None`` for an absent preconditioner, replacing every term of a form leaves a
    ``ufl.ZeroBaseForm`` or a plain ``0`` behind, and ``assemble_scalar`` accepts an
    already-compiled form. Walking those as if they were sequences raised
    ``TypeError: ... is not iterable`` *from inside the handler*, which replaced the original
    error with one about the diagnostic itself -- the exact opposite of what it is for.

    Declining (returning ``None``, so the original error is re-raised untouched) is the right
    answer for all of them: none carries UFL coefficients whose dtypes could be compared.
    """
    if not_a_ufl_form == "compiled":
        u = dolfinx.fem.Function(V)
        not_a_ufl_form = dolfinx.fem.form(ufl.inner(u, u) * ufl.dx)
    elif not_a_ufl_form == "zero-base-form":
        not_a_ufl_form = ufl.form.ZeroBaseForm((ufl.Argument(V, 0),))

    assert scalar_type_mismatch_message(not_a_ufl_form) is None
    assert scalar_type_mismatch_message([not_a_ufl_form, None]) is None


def _pml_metalens(mesh_size: int, seed: int):
    """A miniature of `demos/helmholtz_metalens_topology_optimization.py`.

    Returns `(rho, beta, J, keepalive)`: the density control, the projection-sharpness
    `Constant`, the objective, and the objects that must outlive the tape.

    The starting density is random rather than the demo's uniform grey, for two reasons: the
    geometry is symmetric about `y = 0`, so a uniform base point lets a wrong gradient cancel
    itself by parity; and the projection maps 0.5 to 0.5 for every sharpness, which would make
    `beta` unobservable. Every material constant is the demo's, so that the two copies of this
    physics drift visibly rather than silently -- nothing executes the demo itself, since it is
    kept out of `_toc.yml` (the documentation is built in real mode). Two things deliberately do
    not carry over, neither of which the adjoint can tell apart. This mesh is far coarser, so the
    filter here is a sub-element smoother rather than the minimum length scale it imposes in the
    demo. And the mesh is a plain `create_rectangle` grid with the regions selected by cell
    midpoint, rather than the demo's conforming gmsh geometry: region boundaries falling mid-cell
    cost accuracy, not correctness, and gmsh is a documentation dependency rather than a test
    one.

    This is the demo's whole chain at a mesh coarse enough to run in a test, and it is the
    only place in this file where the control reaches the Functional through the *bilinear*
    form rather than the right-hand side, through a second recorded solve, across two meshes
    joined by an entity map, and against an operator carrying genuinely complex (PML)
    coefficients.
    """
    wavelength = 1.0
    k0 = 2.0 * np.pi / wavelength
    pml_width, x_phys, y_phys = 0.5, 1.0, 0.8
    lx, ly = x_phys + pml_width, y_phys + pml_width
    design, focus_centre, focus_radius = (-0.25, 0.25, -0.6, 0.6), (0.6, 0.0), 0.15
    eps_material, pml_strength, penal, filter_radius = 4.0, 4.0, 3.0, 0.10
    design_tag, focus_tag, bulk_tag = 1, 2, 3

    msh = dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([-lx, -ly]), np.array([lx, ly])],
        (int(2 * lx * mesh_size), int(2 * ly * mesh_size)),
        dolfinx.mesh.CellType.triangle,
    )
    tdim = msh.topology.dim
    index_map = msh.topology.index_map(tdim)
    all_cells = np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    # By midpoint rather than `locate_entities`, which needs the predicate to hold at every
    # vertex and so drops every cell straddling a region boundary. Mirrors the demo.
    midpoints = dolfinx.mesh.compute_midpoints(msh, tdim, all_cells)
    design_cells = all_cells[
        (midpoints[:, 0] > design[0])
        & (midpoints[:, 0] < design[1])
        & (midpoints[:, 1] > design[2])
        & (midpoints[:, 1] < design[3])
    ]
    focus_cells = all_cells[
        (midpoints[:, 0] - focus_centre[0]) ** 2 + (midpoints[:, 1] - focus_centre[1]) ** 2 < focus_radius**2
    ]
    markers = np.full(all_cells.size, bulk_tag, dtype=np.int32)
    markers[design_cells] = design_tag
    markers[focus_cells] = focus_tag
    cell_tags = dolfinx.mesh.meshtags(msh, tdim, np.arange(markers.size, dtype=np.int32), markers)
    dx = ufl.Measure("dx", domain=msh, subdomain_data=cell_tags)

    # The density lives on the design region alone, as a mesh of its own; the Helmholtz solve
    # reaches its permittivity across the two meshes through an entity map. Mirrors the demo.
    design_mesh, design_to_parent, _, _ = dolfinx.mesh.create_submesh(msh, tdim, design_cells)
    dx_design = ufl.Measure("dx", domain=design_mesh)

    Vf = dolfinx.fem.functionspace(msh, ("Lagrange", 2))
    Q = dolfinx.fem.functionspace(design_mesh, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(msh)

    def stretch(coordinate, half_width):
        return 1.0 + 1j * pml_strength * (ufl.max_value(abs(coordinate) - half_width, 0.0) / pml_width) ** 2

    sx, sy = stretch(x[0], x_phys), stretch(x[1], y_phys)
    pml_tensor = ufl.as_matrix([[sy / sx, 0], [0, sx / sy]])
    pml_scale = sx * sy
    incident = ufl.exp(1j * k0 * x[0])

    rho = Function(Q, name="density")
    rho.x.array[:] = np.random.default_rng(seed).uniform(0.0, 1.0, size=rho.x.array.size)
    rho.x.scatter_forward()

    # Helmholtz density filter: a second recorded solve between the control and the operator,
    # posed on the design mesh, where homogeneous Neumann is the natural boundary condition and
    # no Dirichlet condition is needed at all.
    trial_rho, w = ufl.TrialFunction(Q), ufl.TestFunction(Q)
    rho_filtered = Function(Q, name="filtered_density")
    filter_problem = LinearProblem(
        (filter_radius**2 * ufl.inner(ufl.grad(trial_rho), ufl.grad(w)) + ufl.inner(trial_rho, w)) * dx_design,
        ufl.inner(rho, w) * dx_design,
        u=rho_filtered,
        petsc_options=_PETSC_LU,
        adjoint_petsc_options=_PETSC_LU,
    )
    filter_problem.solve()

    beta, eta = dolfinx.fem.Constant(design_mesh, dolfinx.default_scalar_type(4.0)), 0.5
    projected = (ufl.tanh(beta * eta) + ufl.tanh(beta * (rho_filtered - eta))) / (
        ufl.tanh(beta * eta) + ufl.tanh(beta * (1 - eta))
    )
    permittivity = 1.0 + projected**penal * (eps_material - 1.0)

    scattered = Function(Vf, name="scattered_field")
    u, v = ufl.TrialFunction(Vf), ufl.TestFunction(Vf)
    a = (
        ufl.inner(pml_tensor * ufl.grad(u), ufl.grad(v)) - k0**2 * pml_scale * ufl.inner(u, v)
    ) * dx - k0**2 * ufl.inner((permittivity - 1.0) * u, v) * dx(design_tag)
    L = k0**2 * ufl.inner((permittivity - 1.0) * incident, v) * dx(design_tag)

    msh.topology.create_connectivity(tdim - 1, tdim)
    outer_facets = dolfinx.mesh.exterior_facet_indices(msh.topology)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.default_scalar_type(0.0),
        dolfinx.fem.locate_dofs_topological(Vf, tdim - 1, outer_facets),
        Vf,
    )
    problem = LinearProblem(
        a,
        L,
        u=scattered,
        bcs=[bc],
        petsc_options=_PETSC_LU,
        adjoint_petsc_options=_PETSC_LU,
        entity_maps=[design_to_parent],
    )
    problem.solve()

    total = scattered + incident
    focus_area = assemble_scalar(1.0 * dx(focus_tag), annotate=False)
    J = -assemble_scalar(ufl.inner(total, total) * dx(focus_tag)) / focus_area
    return rho, beta, J, (problem, filter_problem)


def test_metalens_design_chain_gradient_converges_at_second_order():
    """The spec's validation case: a real density control inside a complex PML operator.

    Every other test in this file puts the control in the right-hand side, where the adjoint
    never differentiates the operator. Here the control reaches the Functional only through
    the sesquilinear form's permittivity, so the sensitivity comes from `ufl.adjoint` applied
    to a derivative of the operator -- the Load-bearing block's other half.
    """
    pyadjoint.get_working_tape().clear_tape()
    rho, _, J, keepalive = _pml_metalens(mesh_size=8, seed=7)
    assert isinstance(J, float)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(rho))
    h = Function(rho.function_space)
    h.x.array[:] = np.random.default_rng(7).standard_normal(h.x.array.size)
    h.x.scatter_forward()

    with pyadjoint.stop_annotating():
        # The direct check the rest of this file makes, first: the Taylor rates below only
        # constrain how the remainder shrinks, and a gradient wrong by a constant factor can
        # still produce a clean rate of 2. `|E|**2` is quadratic in the state, hence in the
        # control, so the central difference is used rather than the forward one. Its step is
        # tighter than the default: this base point is random on a coarse mesh, so the third
        # derivative carried by the O(eps**2) truncation term is large enough to show at 1e-3.
        gradient = Jhat.derivative()
        assert np.isclose(gradient._ad_dot(h), _central_difference(Jhat, rho, h, eps=1e-4), rtol=1e-5, atol=1e-10)

        # The gradient stays in the control's complex dtype but is mathematically real, and is
        # not the zero vector, which would satisfy every other assertion here for free.
        assert np.issubdtype(gradient.x.array.dtype, np.complexfloating)
        assert np.allclose(gradient.x.array.imag, 0.0, atol=1e-12)
        assert np.abs(gradient.x.array.real).max() > 0.0

        assert np.isclose(pyadjoint.taylor_test(Jhat, rho, h, dJdm=0), 1.0, atol=0.15)
        assert np.isclose(pyadjoint.taylor_test(Jhat, rho, h), 2.0, atol=0.15)
    del keepalive


def test_projection_sharpness_is_replayed_from_its_constant():
    """`beta` is a plain DOLFINx `Constant`, so continuation must not need a re-recorded tape.

    The optimization loop in the demo raises `beta` between stages while reusing one
    `ReducedFunctional`. That only works if the recorded blocks read the `Constant`'s value at
    replay time rather than having baked it in when the form was compiled.
    """
    pyadjoint.get_working_tape().clear_tape()
    rho, beta, J, keepalive = _pml_metalens(mesh_size=8, seed=11)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(rho))
    at_beta_4 = Jhat(rho)
    beta.value = dolfinx.default_scalar_type(32.0)
    at_beta_32 = Jhat(rho)

    assert not np.isclose(at_beta_4, at_beta_32), "raising beta changed nothing, so it was baked into the form"
    del keepalive
