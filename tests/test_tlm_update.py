"""Regression tests for the tangent-linear solver cached on ``LinearProblemBlock``.

``construct_tlm_solver`` compiles ``_compute_residual_derivative()`` once and caches the
resulting solver on the block.  That form is built from ``block_variable.saved_output``,
which is a fresh object after every control update, so the cached operator can stay
pinned to the evaluation point it was first built at.  When the control appears in the
bilinear form -- so that dF/du genuinely depends on it -- every Hessian computed after
the first one is then silently wrong.
"""

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

from dolfinx_adjoint import Function, assemble_scalar
from dolfinx_adjoint.solvers import LinearProblem

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

    The control has to enter the bilinear form for this test to have any teeth: if it
    only enters ``L``, dF/du is independent of the control and a stale tangent-linear
    operator is indistinguishable from a fresh one.
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
    # A rotational (non-conservative) body force.  A constant force would be balanced
    # exactly by the pressure in a closed incompressible box, leaving u == 0 and making
    # the functional independent of the control.  The 1e3 only sets the scale of the
    # state, which keeps the Taylor remainders clear of pyadjoint's absolute
    # machine-precision warning threshold.
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
    h.interpolate(lambda x: 0.3 + 0.2 * np.sin(3 * x[0]))

    if warm_up_at_another_point:
        Jh(m1)
        Jh.derivative()
        Jh.hessian(h)

    Jh(m2)
    dJdm = Jh.derivative()._ad_dot(h)
    Hm = Jh.hessian(h)._ad_dot(h)

    min_rate = pyadjoint.taylor_test(Jh, m2, h, dJdm=dJdm, Hm=Hm)
    assert np.isclose(min_rate, 3.0, rtol=0.1, atol=0.1), f"Expected convergence rate close to 3.0, got {min_rate}"
