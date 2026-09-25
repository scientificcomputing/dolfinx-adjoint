"""Shape control: differentiating through the geometry of the mesh a problem is posed on.

Every test here builds its own mesh. ``move`` mutates a mesh's coordinates in place and
promotes it to an overloaded type, so a mesh shared between tests would carry one test's
displacement, and its tape blocks, into the next.
"""

from mpi4py import MPI

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import pyadjoint
import pytest
import ufl

import dolfinx_adjoint as dxa
from dolfinx_adjoint.blocks.interpolation import ExprInterpolationBlock

# A direct LU solve: the Taylor remainders checked below fall to ~1e-10, which an
# iterative solve's own tolerance would swamp.
_LU = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
_SNES = _LU | {"snes_type": "newtonls", "snes_atol": 1e-12, "snes_rtol": 1e-12, "snes_stol": 0.0}


def _unit_square(n: int = 8) -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)


def _dilation_values(x: np.ndarray) -> np.ndarray:
    """A uniform dilation about the origin, ``s(x) = x``: moves the boundary, tangles no cell."""
    return np.vstack((x[0], x[1]))


def _interior_bump_values(x: np.ndarray) -> np.ndarray:
    """A displacement vanishing on the whole boundary of the unit square.

    Leaves the domain itself untouched and only relocates interior nodes, so the discrete
    solution moves but nothing posed *on* the boundary does. Needed wherever a Dirichlet
    value is obtained by interpolating onto the boundary: a direction that moved the boundary
    would change those dof values too, and the adjoint holds them fixed (that dependence is
    not implemented -- see the geometry-dependent-bc-values issue).
    """
    bump = np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])
    return np.vstack((bump, bump))


def _interior_bump(S: dolfinx.fem.FunctionSpace) -> dxa.Function:
    """A displacement vanishing on the whole boundary, as a dxa Function."""
    bump = dxa.Function(S)
    bump.interpolate(_interior_bump_values)
    return bump


def _cell_jacobians(mesh: dolfinx.mesh.Mesh) -> np.ndarray:
    """The Jacobian determinant of every cell this rank owns."""
    DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    detJ = dolfinx.fem.Function(DG0)
    detJ.interpolate(dolfinx.fem.Expression(ufl.JacobianDeterminant(mesh), DG0.element.interpolation_points))
    return detJ.x.array[: DG0.dofmap.index_map.size_local].copy()


def _assert_mesh_is_valid(mesh: dolfinx.mesh.Mesh, reference: np.ndarray) -> None:
    """Assert no cell of ``mesh`` has inverted relative to its undisplaced state.

    The test is that each cell's Jacobian determinant keeps the sign it had in
    ``reference``, not that it is positive. DOLFINx does not orient cells consistently --
    a pristine unit square has as many negative determinants as positive ones -- and
    integrates against ``|detJ|``, so the absolute sign says nothing. A cell whose sign has
    flipped has turned itself inside out, and one whose determinant has reached zero has
    collapsed; either makes the domain, and so the problem posed on it, meaningless.
    """
    detJ = _cell_jacobians(mesh)
    tangled = int(np.count_nonzero(np.sign(detJ) != np.sign(reference)))
    assert mesh.comm.allreduce(tangled, op=MPI.SUM) == 0, "cells inverted: the perturbed mesh is tangled"
    smallest = mesh.comm.allreduce(np.abs(detJ).min() if detJ.size else np.inf, op=MPI.MIN)
    assert smallest > 0.0, "a cell collapsed: the perturbed mesh is degenerate"


def _shape_setup(n: int = 8) -> tuple[dolfinx.mesh.Mesh, dolfinx.fem.FunctionSpace, dxa.Function, np.ndarray]:
    """A mesh with a zero displacement recorded on the tape, that displacement, and the
    undisplaced cell Jacobians to check later configurations against."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = _unit_square(n)
    reference = _cell_jacobians(mesh)
    S = dxa.geometry_function_space(mesh)
    s = dxa.Function(S)
    mesh = dxa.Mesh(mesh)
    dxa.move(mesh, s)
    return mesh, S, s, reference


def _homogeneous_bc(V: dolfinx.fem.FunctionSpace) -> dolfinx.fem.DirichletBC:
    mesh = V.mesh
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, facets)
    return dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), dofs, V)


def test_move_records_the_displacement():
    """``move`` displaces the geometry and leaves a differentiable record of having done so."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = _unit_square(4)
    before = mesh.geometry.x.copy()

    S = dxa.geometry_function_space(mesh)
    s = dxa.Function(S)
    s.x.array[:] = 0.01
    mesh = dxa.Mesh(mesh)
    moved = dxa.move(mesh, s)

    assert moved is mesh, "the mesh must be promoted in place, so existing forms stay valid"
    assert isinstance(mesh, dxa.types.Mesh)
    gdim = mesh.geometry.dim
    assert np.allclose(mesh.geometry.x[:, :gdim], before[:, :gdim] + 0.01)
    assert np.allclose(mesh.geometry.x[:, gdim:], before[:, gdim:]), "padding columns must not move"


def test_move_by_a_bare_spatial_coordinate():
    """``SpatialCoordinate`` is a legal displacement: it dilates the mesh about the origin.

    Previously refused outright, along with every other expression. It has the right shape for
    the geometry space, so it interpolates and moves each node to twice its position.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = dxa.Mesh(_unit_square(4))
    before = mesh.geometry.x.copy()
    dxa.move(mesh, ufl.SpatialCoordinate(mesh))
    gdim = mesh.geometry.dim
    assert np.allclose(mesh.geometry.x[:, :gdim], 2.0 * before[:, :gdim])


def test_shape_derivative_of_a_functional():
    """A functional of the coordinates alone, with no PDE in between."""
    mesh, S, s, reference = _shape_setup()
    X = ufl.SpatialCoordinate(mesh)
    J = dxa.assemble_scalar(ufl.sin(X[0]) * ufl.cos(X[1]) * ufl.dx + ufl.inner(X, X) * ufl.ds)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    h = dxa.Function(S)
    h.interpolate(_dilation_values)
    assert np.isclose(Jhat(s), float(J))
    assert pyadjoint.taylor_test(Jhat, s, h) > 1.9
    _assert_mesh_is_valid(mesh, reference)


def test_shape_derivative_through_a_linear_problem():
    """Poisson, with both the source and the domain depending on the geometry."""
    mesh, S, s, reference = _shape_setup()
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    f = ufl.sin(ufl.pi * X[0]) * ufl.cos(ufl.pi * X[1])

    problem = dxa.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        ufl.inner(f, v) * ufl.dx,
        bcs=[_homogeneous_bc(V)],
        petsc_options=_LU,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_linear_",
    )
    uh = problem.solve()
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    h = dxa.Function(S)
    h.interpolate(_dilation_values)
    assert np.isclose(Jhat(s), float(J))
    assert pyadjoint.taylor_test(Jhat, s, h) > 1.9
    _assert_mesh_is_valid(mesh, reference)


def test_shape_derivative_through_a_nonlinear_problem():
    """A nonlinear diffusivity ``1 + u**2``, strictly positive for every ``u``, so the
    problem stays coercive at the base point and along the whole Taylor perturbation."""
    mesh, S, s, reference = _shape_setup()
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    uh = dxa.Function(V)
    v = ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    f = ufl.sin(ufl.pi * X[0]) * ufl.cos(ufl.pi * X[1])
    F = ufl.inner((1 + uh**2) * ufl.grad(uh), ufl.grad(v)) * ufl.dx - ufl.inner(f, v) * ufl.dx

    problem = dxa.NonlinearProblem(
        F,
        u=uh,
        bcs=[_homogeneous_bc(V)],
        petsc_options=_SNES,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_nonlinear_",
    )
    problem.solve()
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    h = dxa.Function(S)
    h.interpolate(_dilation_values)
    assert np.isclose(Jhat(s), float(J))
    assert pyadjoint.taylor_test(Jhat, s, h) > 1.9
    _assert_mesh_is_valid(mesh, reference)


def test_shape_gradient_matches_a_finite_difference():
    """Check the gradient's *value*, not only its convergence rate.

    A Taylor test confirms the gradient is consistent with the functional the tape
    replays; it would still pass if the assembled shape derivative and the displacement
    ``move`` applies were both wrong in the same way -- for instance if the geometry
    space's dofs did not line up with the rows of ``mesh.geometry.x``. Comparing
    ``<dJ/ds, h>`` against a central difference of ``J`` computed on independently built,
    explicitly moved meshes pins that down.
    """
    mesh, S, s, reference = _shape_setup()
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    f = ufl.sin(ufl.pi * X[0]) * ufl.cos(ufl.pi * X[1])
    problem = dxa.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        ufl.inner(f, v) * ufl.dx,
        bcs=[_homogeneous_bc(V)],
        petsc_options=_LU,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_fd_",
    )
    uh = problem.solve()
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    h = dxa.Function(S)
    h.interpolate(_dilation_values)
    directional = Jhat.derivative()._ad_dot(h)

    def J_at(step: float) -> float:
        """Rebuild everything from scratch at ``s = step*h``, with no tape involved."""
        with pyadjoint.stop_annotating():
            m = _unit_square(8)
            d = dolfinx.fem.Function(dxa.geometry_function_space(m))
            d.interpolate(_dilation_values)
            d.x.array[:] *= step
            reference_m = _cell_jacobians(m)
            m.geometry.x[:, : m.geometry.dim] += d.x.array.reshape(-1, m.geometry.dim)
            _assert_mesh_is_valid(m, reference_m)

            Vm = dolfinx.fem.functionspace(m, ("Lagrange", 1))
            um, vm = ufl.TrialFunction(Vm), ufl.TestFunction(Vm)
            Xm = ufl.SpatialCoordinate(m)
            fm = ufl.sin(ufl.pi * Xm[0]) * ufl.cos(ufl.pi * Xm[1])
            prob = dolfinx.fem.petsc.LinearProblem(
                ufl.inner(ufl.grad(um), ufl.grad(vm)) * ufl.dx,
                ufl.inner(fm, vm) * ufl.dx,
                bcs=[_homogeneous_bc(Vm)],
                petsc_options=_LU,
                petsc_options_prefix=f"test_shape_fd_ref_{abs(step):g}_",
            )
            uh_m = prob.solve()
            uh_m = uh_m[0] if isinstance(uh_m, tuple) else uh_m
            local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(uh_m, uh_m) * ufl.dx))
            return m.comm.allreduce(local, op=MPI.SUM)

    eps = 1e-4
    fd = (J_at(eps) - J_at(-eps)) / (2 * eps)
    # Without this, two zeros would compare equal and the test would pass vacuously.
    assert abs(fd) > 1e-6, f"the finite difference is ~0 ({fd}): this direction tests nothing"
    assert np.isclose(directional, fd, rtol=1e-5), f"adjoint {directional} vs finite difference {fd}"


def test_shape_and_coefficient_controls_together():
    """A geometry control and an ordinary coefficient control on the same tape."""
    mesh, S, s, reference = _shape_setup()
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    Q = dolfinx.fem.functionspace(mesh, ("DG", 0))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    m = dxa.Function(Q)
    m.x.array[:] = 1.0

    problem = dxa.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        ufl.inner(m, v) * ufl.dx,
        bcs=[_homogeneous_bc(V)],
        petsc_options=_LU,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_joint_",
    )
    uh = problem.solve()
    X = ufl.SpatialCoordinate(mesh)
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx + ufl.inner(X, X) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, [pyadjoint.Control(s), pyadjoint.Control(m)])
    hs = dxa.Function(S)
    hs.interpolate(_dilation_values)
    hm = dxa.Function(Q)
    hm.x.array[:] = 0.5
    assert pyadjoint.taylor_test(Jhat, [s, m], [hs, hm]) > 1.9
    _assert_mesh_is_valid(mesh, reference)


def test_repeated_replay_is_stable():
    """Replaying the tape must re-apply the displacement, never accumulate it.

    ``MoveBlock.recompute_component`` adds the displacement to the geometry the mesh had
    *before* the move. If it instead incremented the current geometry, every replay would
    shift the mesh further and this would drift -- which a checkpoint schedule, replaying
    the forward many times, would turn into silent nonsense.
    """
    mesh, S, s, reference = _shape_setup()
    s.x.array[:] = 0.0
    X = ufl.SpatialCoordinate(mesh)
    J = dxa.assemble_scalar(ufl.inner(X, X) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))

    displaced = dxa.Function(S)
    displaced.interpolate(lambda x: 0.05 * _dilation_values(x))
    first = Jhat(displaced)
    for _ in range(3):
        assert np.isclose(Jhat(displaced), first, rtol=0, atol=1e-14)
    assert not np.isclose(first, Jhat(s)), "the displacement must actually change the functional"


def test_mesh_is_not_usable_as_a_control():
    """The control of a shape optimization is the displacement, not the mesh itself."""
    mesh, _, _, _ = _shape_setup(4)
    with pytest.raises(NotImplementedError, match="displacement"):
        mesh._ad_dot(mesh)


def test_shape_hessian_of_a_functional():
    """A second shape derivative, where no PDE solve stands between control and functional."""
    mesh, S, s, reference = _shape_setup()
    X = ufl.SpatialCoordinate(mesh)
    J = dxa.assemble_scalar(ufl.sin(X[0]) * ufl.cos(X[1]) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    h = dxa.Function(S)
    h.interpolate(_dilation_values)

    Jhat(s)
    Jhat.derivative()
    Hh = Jhat.hessian(h)._ad_dot(h)

    def directional_gradient_at(scale: float) -> float:
        Jhat(s._ad_add(h._ad_mul(scale)))
        return Jhat.derivative()._ad_dot(h)

    eps = 1e-4
    fd = (directional_gradient_at(eps) - directional_gradient_at(-eps)) / (2 * eps)
    Jhat(s)
    assert abs(fd) > 1e-6, f"the finite difference is ~0 ({fd}): this direction tests nothing"
    assert np.isclose(Hh, fd, rtol=1e-5), f"hessian {Hh} vs finite difference {fd}"
    _assert_mesh_is_valid(mesh, reference)


def test_shape_hessian_through_a_linear_problem(assert_hessian_matches_finite_difference):
    """A second shape derivative across a PDE solve.

    Needs the whole second-order chain to carry a shape term: ``dF/dX[dX]`` in the
    tangent-linear right-hand side, the mixed ``d2F/dudX`` and pure ``d2F/dX2`` contributions
    to the second-order-adjoint right-hand side, and the Hessian-action output on the geometry
    space. The mesh is registered as an ordinary differentiation target alongside the
    residual's coefficients (``_ProblemBase._differentiation_targets``), differentiating
    against ``ufl.SpatialCoordinate`` where a coefficient differentiates against its
    placeholder, so all of that reuses the coefficient machinery.
    """
    mesh, S, s, reference = _shape_setup()
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    problem = dxa.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        ufl.inner(ufl.sin(ufl.pi * X[0]) * ufl.cos(ufl.pi * X[1]), v) * ufl.dx,
        bcs=[_homogeneous_bc(V)],
        petsc_options=_LU,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_hessian_linear_",
    )
    uh = problem.solve()
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx + ufl.inner(X, X) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    assert_hessian_matches_finite_difference(Jhat, s, _interior_bump(S))
    _assert_mesh_is_valid(mesh, reference)


def test_shape_hessian_through_a_nonlinear_problem(assert_hessian_matches_finite_difference):
    """The same, where ``dF/du`` genuinely depends on the state.

    ``d2F/du2`` is structurally zero for a linear residual, so the linear test above never
    exercises the ``soa_self`` term against a shape direction. A ``1 + u**2`` diffusivity --
    strictly positive for every ``u``, so the problem stays coercive along the whole
    perturbation -- does.
    """
    mesh, S, s, reference = _shape_setup()
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    uh = dxa.Function(V)
    v = ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    F = (
        ufl.inner((1 + uh**2) * ufl.grad(uh), ufl.grad(v)) * ufl.dx
        - ufl.inner(ufl.sin(ufl.pi * X[0]) * ufl.cos(ufl.pi * X[1]), v) * ufl.dx
    )
    problem = dxa.NonlinearProblem(
        F,
        u=uh,
        bcs=[_homogeneous_bc(V)],
        petsc_options=_SNES,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_hessian_nonlinear_",
    )
    problem.solve()
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    assert_hessian_matches_finite_difference(Jhat, s, _interior_bump(S))
    _assert_mesh_is_valid(mesh, reference)


def test_shape_and_coefficient_hessian_together():
    """A shape control and a coefficient control on one tape, to second order.

    Exercises the ``cross`` templates in both directions -- the shape term differentiated
    along the coefficient's tangent-linear direction and the coefficient term differentiated
    along the shape's -- which a single-control test cannot reach.
    """
    mesh, S, s, reference = _shape_setup()
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    Q = dolfinx.fem.functionspace(mesh, ("DG", 0))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    m = dxa.Function(Q)
    m.x.array[:] = 1.0

    problem = dxa.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        ufl.inner(m * ufl.sin(ufl.pi * X[0]), v) * ufl.dx,
        bcs=[_homogeneous_bc(V)],
        petsc_options=_LU,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_hessian_joint_",
    )
    uh = problem.solve()
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx + ufl.inner(X, X) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, [pyadjoint.Control(s), pyadjoint.Control(m)])
    hm = dxa.Function(Q)
    hm.x.array[:] = 0.5
    controls, directions = [s, m], [_interior_bump(S), hm]

    # The conftest Hessian checker takes a single control, so the same central-difference
    # check is done over the pair here: the directional second derivative is the sum over
    # controls of <H_i[h], h_i>, and the gradient likewise.
    def directional_gradient(scale: float) -> float:
        Jhat([c._ad_add(d._ad_mul(scale)) for c, d in zip(controls, directions)])
        return sum(g._ad_dot(d) for g, d in zip(Jhat.derivative(), directions))

    Jhat(controls)
    Jhat.derivative()
    Hh = sum(Hi._ad_dot(d) for Hi, d in zip(Jhat.hessian(directions), directions))
    fd_eps = 1e-3
    fd = (directional_gradient(fd_eps) - directional_gradient(-fd_eps)) / (2 * fd_eps)
    Jhat(controls)
    assert np.isclose(Hh, fd, rtol=1e-2, atol=1e-2), f"hessian {Hh} vs finite difference {fd}"
    _assert_mesh_is_valid(mesh, reference)


def test_gradient_is_correct_when_the_mesh_is_moved_twice():
    """Two successive moves, with a solve between them.

    Each block has to be differentiated at the geometry *it* saw, not at the final one --
    here the solve's geometry and the functional's differ. pyadjoint rewinds a mesh
    dependency for us, by reading its ``saved_output`` before every ``prepare_evaluate_adj``,
    which restores the coordinates in place; this pins that behaviour down, since nothing
    in this package would notice if it stopped.
    """
    mesh, S, s, reference = _shape_setup()
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    problem = dxa.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        ufl.inner(ufl.sin(ufl.pi * X[0]) * ufl.cos(ufl.pi * X[1]), v) * ufl.dx,
        bcs=[_homogeneous_bc(V)],
        petsc_options=_LU,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_two_moves_",
    )
    uh = problem.solve()
    J_first = ufl.inner(uh, uh) * ufl.dx

    second = dxa.Function(S)
    second.interpolate(lambda x: 0.05 * _dilation_values(x))
    mesh = dxa.Mesh(mesh)
    dxa.move(mesh, second)
    J = dxa.assemble_scalar(J_first + ufl.inner(X, X) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    h = dxa.Function(S)
    h.interpolate(_dilation_values)
    assert np.isclose(Jhat(s), float(J))
    assert pyadjoint.taylor_test(Jhat, s, h) > 1.9
    _assert_mesh_is_valid(mesh, reference)


def _heat_loop(n_steps: int, displacement_values: np.ndarray | None = None, schedule=None):
    """A ``n_steps``-step backward-Euler heat equation on a movable mesh.

    Returns the functional, the displacement control, its space, and the problem -- the
    caller has to keep the last alive, or its solvers are collected and every replay pays to
    rebuild them. The state update is an
    explicit, tape-recorded ``assign``: writing into ``u_prev.x.array`` directly would
    bypass annotation and silently break the coupling between steps.
    """
    tape = pyadjoint.get_working_tape()
    tape.clear_tape()
    if schedule is not None:
        # Both must precede every block: pyadjoint refuses to enable checkpointing on a
        # non-empty tape, and disk checkpointing has the same requirement so that every
        # checkpoint is stored the same way.
        if "Disk" in type(schedule).__name__:
            dxa.enable_disk_checkpointing()
        tape.enable_checkpointing(schedule)

    mesh = _unit_square(6)
    S = dxa.geometry_function_space(mesh)
    s = dxa.Function(S, name="displacement")
    if displacement_values is not None:
        s.x.array[:] = displacement_values
    mesh = dxa.Mesh(mesh)
    dxa.move(mesh, s)

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u_prev = dxa.Function(V, name="u_prev")
    u_prev.x.array[:] = 1.0
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    dt = 0.1
    problem = dxa.LinearProblem(
        ufl.inner(u, v) * ufl.dx + dt * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        ufl.inner(u_prev, v) * ufl.dx + dt * ufl.inner(ufl.sin(ufl.pi * X[0]), v) * ufl.dx,
        petsc_options=_LU,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix=f"test_shape_heat_{n_steps}_{id(schedule)}_",
    )

    J = 0.0
    for _ in tape.timestepper(iter(range(n_steps))):
        uh = problem.solve()
        dxa.assign(uh, u_prev)
        J = J + dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx)
    return J, s, S, problem


def test_time_dependent_shape_gradient_matches_a_finite_difference():
    """The shape gradient of a time loop, checked by value against a finite difference.

    Every step's residual carries the previous step's solution, so ``dF/dX`` depends on that
    state -- unlike ``dF/dm`` for a coefficient of a linear residual, which does not
    reference it at all. A gradient that read those states at the wrong time would still
    pass a Taylor test, since the tape would be self-consistent; only a finite difference
    of an independently rebuilt forward catches it.
    """
    n_steps = 4
    _, s0, S0, _keep = _heat_loop(n_steps)
    direction = dolfinx.fem.Function(S0)
    direction.interpolate(_dilation_values)
    h_values = direction.x.array.copy()

    def J_at(step: float) -> float:
        J, _, _, _problem = _heat_loop(n_steps, step * h_values)
        return float(J)

    eps = 1e-5
    fd = (J_at(eps) - J_at(-eps)) / (2 * eps)

    J, s, S, problem = _heat_loop(n_steps)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    gradient = Jhat.derivative()
    owned = S.dofmap.index_map.size_local * S.dofmap.index_map_bs
    directional = MPI.COMM_WORLD.allreduce(float(np.dot(gradient.x.array[:owned], h_values[:owned])), op=MPI.SUM)

    assert abs(fd) > 1e-6, f"the finite difference is ~0 ({fd}): this direction tests nothing"
    assert np.isclose(directional, fd, rtol=1e-7), f"adjoint {directional} vs finite difference {fd}"


def test_shape_derivative_under_a_recomputing_schedule_is_refused():
    """A schedule that recomputes releases the checkpoints ``dF/dX`` needs.

    pyadjoint marks the released dependency as a functional dependency, so it knows the value
    matters, but the set deciding what is retained for the adjoint is only populated during the
    first reverse traversal -- after the data has already been dropped. ``saved_output`` then
    returns the function's *live* value with no error, and the residual is differentiated at
    whatever state the last recomputed step left behind. The gradient comes out wrong by
    double-digit percentages and still passes a Taylor test, so this has to fail rather than
    return a number.

    The refusal lands at ``move``, before the forward is run at all -- ``_heat_loop`` moves the
    mesh -- rather than from inside the adjoint sweep.
    """
    from checkpoint_schedules import Revolve

    n_steps = 4
    with pytest.raises(NotImplementedError, match="not supported under the checkpoint schedule"):
        _heat_loop(n_steps, schedule=Revolve(n_steps, 2))


def test_shape_derivative_under_a_retaining_schedule_is_allowed():
    """A schedule that stores every step releases nothing, so the shape gradient is fine.

    The guard above has to distinguish the two: refusing every schedule outright would rule
    out a case that is demonstrably correct.
    """
    from checkpoint_schedules import SingleMemoryStorageSchedule

    n_steps = 4
    J_ref, s_ref, S_ref, problem_ref = _heat_loop(n_steps)
    Jhat_ref = pyadjoint.ReducedFunctional(J_ref, pyadjoint.Control(s_ref))
    reference = Jhat_ref.derivative().x.array.copy()

    J, s, _, problem = _heat_loop(n_steps, schedule=SingleMemoryStorageSchedule())
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    assert np.allclose(Jhat.derivative().x.array, reference, rtol=1e-10, atol=1e-12)


def test_shape_derivative_of_a_blocked_problem():
    """A blocked (Taylor-Hood Stokes) problem, whose residual's test space is mixed.

    ``_shape_sensitivity`` substitutes the adjoint solution for the residual's test function
    before differentiating, which for a blocked problem means substituting each test function
    part by its own adjoint component. Checked by value against a finite difference, since a
    per-part mix-up would still give a self-consistent tape and a rate-2 Taylor test.
    """
    import basix.ufl

    def solve_stokes(displacement_values: np.ndarray | None = None):
        pyadjoint.get_working_tape().clear_tape()
        mesh = _unit_square(6)
        S = dxa.geometry_function_space(mesh)
        s = dxa.Function(S)
        if displacement_values is not None:
            s.x.array[:] = displacement_values
        mesh = dxa.Mesh(mesh)
        dxa.move(mesh, s)

        V = dolfinx.fem.functionspace(
            mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
        )
        Q = dolfinx.fem.functionspace(mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), 1))
        W = ufl.MixedFunctionSpace(V, Q)
        u, p = ufl.TrialFunctions(W)
        v, q = ufl.TestFunctions(W)
        X = ufl.SpatialCoordinate(mesh)
        # A forcing with non-zero curl, so the velocity response is not absorbed
        # wholesale into the pressure and the functional actually moves.
        f = ufl.as_vector((ufl.sin(ufl.pi * X[1]), ufl.cos(ufl.pi * X[0])))
        a = ufl.extract_blocks(
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.div(v) * p * ufl.dx - ufl.div(u) * q * ufl.dx
        )
        rhs = ufl.extract_blocks(ufl.inner(f, v) * ufl.dx + dolfinx.fem.Constant(mesh, 0.0) * q * ufl.dx)
        # Inlet on the left, no-slip on top and bottom, and the right edge left free so the
        # outflow condition fixes the pressure. Clamping the velocity on the whole boundary
        # instead would leave the pressure determined only up to a constant, and a functional
        # of it meaningless.
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        inlet_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda pt: np.isclose(pt[0], 0.0)
        )
        wall_facets = dolfinx.mesh.locate_entities_boundary(
            mesh, mesh.topology.dim - 1, lambda pt: np.isclose(pt[1], 0.0) | np.isclose(pt[1], 1.0)
        )
        inlet = dxa.Function(V)
        inlet.interpolate(lambda pt: np.vstack((np.sin(np.pi * pt[1]), np.zeros_like(pt[0]))))
        no_slip = np.zeros(mesh.geometry.dim, dtype=dolfinx.default_scalar_type)
        bcs = [
            dolfinx.fem.dirichletbc(
                inlet,
                dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, inlet_facets),
            ),
            dolfinx.fem.dirichletbc(
                no_slip,
                dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, wall_facets),
                V,
            ),
        ]
        uh, ph = dxa.Function(V), dxa.Function(Q)
        problem = dxa.LinearProblem(
            a,
            rhs,
            u=[uh, ph],
            bcs=bcs,
            petsc_options=_LU,
            adjoint_petsc_options=_LU,
            tlm_petsc_options=_LU,
            petsc_options_prefix="test_shape_blocked_",
        )
        problem.solve()
        J = dxa.assemble_scalar(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx)
        return J, s, S, problem

    J, s, S, problem = solve_stokes()
    direction = dolfinx.fem.Function(S)
    direction.interpolate(_interior_bump_values)
    h_values = direction.x.array.copy()

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    gradient = Jhat.derivative()
    owned = S.dofmap.index_map.size_local * S.dofmap.index_map_bs
    directional = MPI.COMM_WORLD.allreduce(float(np.dot(gradient.x.array[:owned], h_values[:owned])), op=MPI.SUM)

    def J_at(step: float) -> float:
        value, _, _, _keep = solve_stokes(step * h_values)
        return float(value)

    eps = 1e-5
    fd = (J_at(eps) - J_at(-eps)) / (2 * eps)
    assert abs(fd) > 1e-6, f"the finite difference is ~0 ({fd}): this direction tests nothing"
    assert np.isclose(directional, fd, rtol=1e-7), f"adjoint {directional} vs finite difference {fd}"


def test_shape_derivative_through_an_interpolated_expression():
    """Interpolating an expression of the coordinates moves when the mesh does.

    ``ExprInterpolationBlock`` reads the geometry, so it carries a shape derivative of its
    own. Before it did, this returned a plausible number that was 5% wrong, with no error and
    a passing Taylor test.
    """

    def forward(displacement_values: np.ndarray | None = None):
        pyadjoint.get_working_tape().clear_tape()
        mesh = _unit_square()
        S = dxa.geometry_function_space(mesh)
        s = dxa.Function(S)
        if displacement_values is not None:
            s.x.array[:] = displacement_values
        mesh = dxa.Mesh(mesh)
        dxa.move(mesh, s)
        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        X = ufl.SpatialCoordinate(mesh)
        uh = dxa.interpolate(ufl.sin(ufl.pi * X[0]) * ufl.cos(ufl.pi * X[1]), V)
        return dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx), s, S

    J, s, S = forward()
    direction = dolfinx.fem.Function(S)
    direction.interpolate(_dilation_values)
    h_values = direction.x.array.copy()

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    owned = S.dofmap.index_map.size_local * S.dofmap.index_map_bs
    directional = MPI.COMM_WORLD.allreduce(
        float(np.dot(Jhat.derivative().x.array[:owned], h_values[:owned])), op=MPI.SUM
    )

    eps = 1e-5
    fd = (float(forward(eps * h_values)[0]) - float(forward(-eps * h_values)[0])) / (2 * eps)
    assert abs(fd) > 1e-6, f"the finite difference is ~0 ({fd}): this direction tests nothing"
    assert np.isclose(directional, fd, rtol=1e-5), f"adjoint {directional} vs finite difference {fd}"


def test_interpolating_a_coefficient_carries_no_geometry_dependence():
    """A Function interpolated into another space on a moved mesh must not gain a shape term.

    Between Lagrange spaces the interpolation evaluates nodal values at reference points, so
    moving the mesh moves the points and the basis functions together and the dof values are
    unchanged. The gradient is therefore exactly the one a still mesh would give -- this pins
    down that ``_reads_geometry`` does not over-trigger and start differentiating a term that
    is genuinely zero (which UFL would refuse to do anyway).
    """

    def forward(displacement_values: np.ndarray | None = None):
        pyadjoint.get_working_tape().clear_tape()
        mesh = _unit_square()
        S = dxa.geometry_function_space(mesh)
        s = dxa.Function(S)
        if displacement_values is not None:
            s.x.array[:] = displacement_values
        mesh = dxa.Mesh(mesh)
        dxa.move(mesh, s)
        V1 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        V2 = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
        source = dxa.Function(V1)
        source.x.array[:] = 1.0
        uh = dxa.interpolate(source, V2)
        return dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx), s, S

    J, s, S = forward()
    direction = dolfinx.fem.Function(S)
    direction.interpolate(_dilation_values)
    h_values = direction.x.array.copy()

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    owned = S.dofmap.index_map.size_local * S.dofmap.index_map_bs
    directional = MPI.COMM_WORLD.allreduce(
        float(np.dot(Jhat.derivative().x.array[:owned], h_values[:owned])), op=MPI.SUM
    )
    eps = 1e-5
    fd = (float(forward(eps * h_values)[0]) - float(forward(-eps * h_values)[0])) / (2 * eps)
    assert np.isclose(directional, fd, rtol=1e-6), f"adjoint {directional} vs finite difference {fd}"


def test_shape_derivative_through_an_interpolated_dirichlet_value():
    """A Dirichlet value built from the coordinates, on a boundary that moves.

    The whole chain has to be annotated for this to be right: ``dxa.interpolate`` to build the
    value (so its geometry dependence is recorded) and ``dxa.dirichletbc`` to attach it (so the
    solve depends on it at all). With a plain ``dolfinx.fem.dirichletbc`` the value is simply
    not on the tape and the gradient is wrong by more than half.
    """

    def forward(displacement_values: np.ndarray | None = None, counter=[0]):
        counter[0] += 1
        pyadjoint.get_working_tape().clear_tape()
        mesh = _unit_square()
        S = dxa.geometry_function_space(mesh)
        s = dxa.Function(S)
        if displacement_values is not None:
            s.x.array[:] = displacement_values
        mesh = dxa.Mesh(mesh)
        dxa.move(mesh, s)

        V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        X = ufl.SpatialCoordinate(mesh)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
        boundary_value = dxa.interpolate(X[0] ** 2 + X[1], V)
        bc = dxa.dirichletbc(boundary_value, dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets))
        problem = dxa.LinearProblem(
            ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
            ufl.inner(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), v) * ufl.dx,
            bcs=[bc],
            petsc_options=_LU,
            adjoint_petsc_options=_LU,
            tlm_petsc_options=_LU,
            petsc_options_prefix=f"test_shape_bc_{counter[0]}_",
        )
        uh = problem.solve()
        return dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx), s, S, problem

    J, s, S, problem = forward()
    direction = dolfinx.fem.Function(S)
    direction.interpolate(_dilation_values)
    h_values = direction.x.array.copy()

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    owned = S.dofmap.index_map.size_local * S.dofmap.index_map_bs
    directional = MPI.COMM_WORLD.allreduce(
        float(np.dot(Jhat.derivative().x.array[:owned], h_values[:owned])), op=MPI.SUM
    )

    eps = 1e-5
    fd = (float(forward(eps * h_values)[0]) - float(forward(-eps * h_values)[0])) / (2 * eps)
    assert abs(fd) > 1e-6, f"the finite difference is ~0 ({fd}): this direction tests nothing"
    assert np.isclose(directional, fd, rtol=1e-5), f"adjoint {directional} vs finite difference {fd}"


_COEFFICIENT_INTERPOLATIONS = {
    # The one case with no geometry dependence: mesh and basis functions move together.
    "value-into-lagrange": (("Lagrange", 2), lambda u, x: u, ("Lagrange", 1)),
    "grad-into-dg": (("Lagrange", 2), lambda u, x: ufl.grad(u), ("DG", 1, (2,))),
    "div-into-dg": (("Lagrange", 2, (2,)), lambda u, x: ufl.div(u), ("DG", 1)),
    "coefficient-times-coordinates": (("Lagrange", 2), lambda u, x: u * ufl.sin(ufl.pi * x[0]), ("Lagrange", 1)),
    "lagrange-into-rt": (("Lagrange", 2, (2,)), lambda u, x: u, ("RT", 1)),
    "rt-into-dg": (("RT", 1), lambda u, x: u, ("DG", 1, (2,))),
    "n1curl-into-dg": (("N1curl", 1), lambda u, x: u, ("DG", 1, (2,))),
}


def _coefficient_interpolation_forward(case: str, step, direction):
    """``int |interpolate(expr(u), Q)|^4 dx`` on a moved mesh, with ``u``'s dofs set beforehand.

    ``u`` is filled before the move and so held fixed in its dofs, as the adjoint holds it.
    The functional is quartic so that the second derivative is not trivially quadratic.
    """
    element_u, expression, element_q = _COEFFICIENT_INTERPOLATIONS[case]
    pyadjoint.get_working_tape().clear_tape()
    mesh = _unit_square(6)
    V = dolfinx.fem.functionspace(mesh, element_u)
    u = dxa.Function(V)
    if V.value_shape == ():
        u.interpolate(lambda x: np.sin(2.0 * x[0]) + x[1] ** 3)
    else:
        u.interpolate(lambda x: np.vstack((np.sin(2.0 * x[0]) + x[1] ** 3, x[0] * x[1] ** 2 + 1.0)))
    S = dxa.geometry_function_space(mesh)
    s = dxa.Function(S)
    if step is not None:
        s.x.array[:] = step * direction
    mesh = dxa.Mesh(mesh)
    dxa.move(mesh, s)
    v = dxa.interpolate(expression(u, ufl.SpatialCoordinate(mesh)), dolfinx.fem.functionspace(mesh, element_q))
    return dxa.assemble_scalar(ufl.inner(v, v) ** 2 * ufl.dx), s, S, u


def _shear(x: np.ndarray) -> np.ndarray:
    """A shape direction that is not a uniform dilation, to which Piola-mapped fields are blind."""
    return np.vstack((x[0] * (1.0 + 0.5 * x[1]), x[1] ** 2))


@pytest.mark.parametrize("case", sorted(_COEFFICIENT_INTERPOLATIONS))
def test_shape_derivative_of_interpolating_a_coefficient(case):
    """A coefficient's interpolation moves with the mesh unless it is a plain value into Lagrange.

    ``grad(u) = ReferenceGrad(u) . K`` carries the inverse Jacobian, a Piola-mapped coefficient's
    push-forward carries ``J`` and ``det J``, and a Piola-mapped target pulls back with them.
    Without a mesh dependency for these the gradient was silently wrong -- 10x and of the wrong
    sign for ``grad(u)`` -- so each is checked by value against a rebuilt forward.
    """
    shear = dxa.Function(dxa.geometry_function_space(_unit_square(6)))
    shear.interpolate(_shear)
    h = shear.x.array.copy()
    eps = 1e-6
    fd = (
        float(_coefficient_interpolation_forward(case, eps, h)[0])
        - float(_coefficient_interpolation_forward(case, -eps, h)[0])
    ) / (2 * eps)
    assert abs(fd) > 1e-8, "the direction must actually change the functional"

    J, s, S, _ = _coefficient_interpolation_forward(case, None, h)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    hf = dxa.Function(S)
    hf.x.array[:] = h
    assert abs(_directional(Jhat.derivative(), hf, S) - fd) < 1e-6 * abs(fd)


@pytest.mark.parametrize("case", ["grad-into-dg", "coefficient-times-coordinates", "lagrange-into-rt", "rt-into-dg"])
def test_shape_hessian_of_interpolating_a_coefficient(case):
    """Second order for the same interpolations, with the coefficient as a second control.

    The coefficient control exercises the cross terms: the coordinate derivative of the
    coefficient derivative, and the reverse.
    """
    J, s, S, u = _coefficient_interpolation_forward(case, None, None)
    h = dxa.Function(S)
    h.interpolate(_shear)
    du = dxa.Function(u.function_space)
    if u.function_space.value_shape == ():
        du.interpolate(lambda x: np.cos(x[0]) * x[1])
    else:
        du.interpolate(lambda x: np.vstack((np.cos(x[0]) * x[1], x[0] ** 2)))
    Jhat = pyadjoint.ReducedFunctional(J, [pyadjoint.Control(s), pyadjoint.Control(u)])
    Hh, fd = _hessian_and_finite_difference(Jhat, [s, u], [h, du])
    assert abs(fd) > 1e-8, "the direction must actually curve the functional"
    assert abs(Hh - fd) < 1e-5 * abs(fd), f"hessian {Hh} vs finite difference {fd}"


def test_move_refuses_a_recomputing_schedule():
    """Refuse at ``move``, not at ``derivative``.

    The alternative failure comes from inside the adjoint sweep, by which point a whole
    forward run has been paid for. pyadjoint requires ``enable_checkpointing`` to precede
    every block, so the schedule is always known by the time ``move`` is called.
    """
    from checkpoint_schedules import Revolve

    tape = pyadjoint.get_working_tape()
    tape.clear_tape()
    tape.enable_checkpointing(Revolve(4, 2))
    mesh = _unit_square(4)
    s = dxa.Function(dxa.geometry_function_space(mesh))
    with pytest.raises(NotImplementedError, match="not supported under the checkpoint schedule"):
        mesh = dxa.Mesh(mesh)
        dxa.move(mesh, s)


@pytest.mark.parametrize("schedule_name", ["SingleMemoryStorageSchedule", "SingleDiskStorageSchedule"])
def test_move_accepts_a_retaining_schedule(schedule_name):
    """A schedule that retains every step releases nothing, so there is nothing to refuse.

    Both are verified to give the right shape gradient, which is why they are on the
    allowlist; refusing them would make shape control and checkpointing mutually exclusive
    for no reason.
    """
    import checkpoint_schedules

    tape = pyadjoint.get_working_tape()
    tape.clear_tape()
    if "Disk" in schedule_name:
        dxa.enable_disk_checkpointing()
    tape.enable_checkpointing(getattr(checkpoint_schedules, schedule_name)())
    mesh = _unit_square(4)
    s = dxa.Function(dxa.geometry_function_space(mesh))
    mesh = dxa.Mesh(mesh)
    dxa.move(mesh, s)  # must not raise


def test_shape_gradient_is_correct_under_a_disk_retaining_schedule():
    """``SingleDiskStorageSchedule`` is on the allowlist, so it has to earn its place."""
    from checkpoint_schedules import SingleDiskStorageSchedule

    n_steps = 4
    J_ref, s_ref, _, problem_ref = _heat_loop(n_steps)
    Jhat_ref = pyadjoint.ReducedFunctional(J_ref, pyadjoint.Control(s_ref))
    reference = Jhat_ref.derivative().x.array.copy()

    J, s, _, problem = _heat_loop(n_steps, schedule=SingleDiskStorageSchedule())
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    assert np.allclose(Jhat.derivative().x.array, reference, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("quantity", ["CellDiameter", "Circumradius", "MinCellEdgeLength"])
def test_geometry_without_a_shape_derivative_is_refused(quantity):
    """UFL differentiates most geometric quantities to *zero* w.r.t. the coordinates.

    ``CoordinateDerivativeRuleset`` registers one rule for the whole ``GeometricQuantity`` base
    class returning an independent terminal. A few survive anyway because ``compute_form_data``
    lowers them into Jacobian terms before the coordinate derivative runs; the ones that stay
    terminals contribute nothing. Measured before this guard existed: ``CellDiameter`` and
    ``MinCellEdgeLength`` gave 2/3 of the true derivative -- only the measure's contribution --
    and ``Circumradius`` gave exactly 0, all with no error.
    """
    mesh, _, _, _ = _shape_setup(4)
    with pytest.raises(NotImplementedError, match="differentiates every geometric quantity"):
        dxa.assemble_scalar(getattr(ufl, quantity)(mesh) * ufl.dx)


@pytest.mark.parametrize(
    "integrand",
    [
        pytest.param(lambda m: ufl.inner(ufl.SpatialCoordinate(m), ufl.SpatialCoordinate(m)) * ufl.dx, id="x.x*dx"),
        pytest.param(lambda m: ufl.dot(ufl.SpatialCoordinate(m), ufl.FacetNormal(m)) * ufl.ds, id="x.n*ds"),
        pytest.param(lambda m: ufl.CellVolume(m) * ufl.dx, id="CellVolume*dx"),
        pytest.param(lambda m: ufl.FacetArea(m) * ufl.ds, id="FacetArea*ds"),
        pytest.param(
            lambda m: ufl.avg(ufl.inner(ufl.SpatialCoordinate(m), ufl.SpatialCoordinate(m))) * ufl.dS, id="avg(x.x)*dS"
        ),
    ],
)
def test_assemble_scalar_shape_derivative_over_measures_and_geometry(integrand):
    """``assemble_scalar``'s shape derivative, across measures and the geometry UFL handles.

    Cell, exterior facet and interior facet integrals, and the geometric quantities that are
    lowered into Jacobian terms before differentiation, so they are genuinely differentiated
    rather than dropped. Checked by value: a dropped term still gives a self-consistent tape.
    """

    def forward(displacement_values: np.ndarray | None = None):
        pyadjoint.get_working_tape().clear_tape()
        mesh = _unit_square(6)
        S = dxa.geometry_function_space(mesh)
        s = dxa.Function(S)
        if displacement_values is not None:
            s.x.array[:] = displacement_values
        mesh = dxa.Mesh(mesh)
        dxa.move(mesh, s)
        return dxa.assemble_scalar(integrand(mesh)), s, S

    J, s, S = forward()
    direction = dolfinx.fem.Function(S)
    direction.interpolate(_dilation_values)
    h_values = direction.x.array.copy()

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    owned = S.dofmap.index_map.size_local * S.dofmap.index_map_bs
    directional = MPI.COMM_WORLD.allreduce(
        float(np.dot(Jhat.derivative().x.array[:owned], h_values[:owned])), op=MPI.SUM
    )

    eps = 1e-6
    fd = (float(forward(eps * h_values)[0]) - float(forward(-eps * h_values)[0])) / (2 * eps)
    assert abs(fd) > 1e-8, f"the finite difference is ~0 ({fd}): this integrand tests nothing"
    assert np.isclose(directional, fd, rtol=1e-5), f"adjoint {directional} vs finite difference {fd}"


def test_move_accepts_a_ufl_expression():
    """``move`` takes any UFL expression over the mesh, not only a geometry-space Function.

    The expression is interpolated into the geometry space by the annotating
    ``dolfinx_adjoint.interpolate``, so the interpolation contributes its own block and the
    chain rule through it is recorded -- the same packing ``dolfinx_adjoint.dirichletbc`` does
    for a boundary value. Checked by value: the gradient with respect to the coefficient the
    expression is built from has to match a finite difference of an independently rebuilt
    forward, which it cannot if the interpolation is untracked.
    """

    def forward(amplitude: float):
        pyadjoint.get_working_tape().clear_tape()
        mesh = dxa.Mesh(_unit_square())
        Q = dolfinx.fem.functionspace(mesh, ("DG", 0))
        alpha = dxa.Function(Q, name="amplitude")
        alpha.x.array[:] = amplitude
        x = ufl.SpatialCoordinate(mesh)
        # A dilation about the origin, scaled by the control. Deliberately not a rigid
        # rotation: rotating the unit square leaves an integrand with the symmetry of
        # sin(pi x) cos(pi y) stationary at alpha = 0, so the finite difference comes out at
        # 1e-12 and the test proves nothing. The domain here becomes [0, 1+alpha]^2 and the
        # functional (1+alpha)^3 / 2, whose derivative is 3/2 at alpha = 0.
        dxa.move(mesh, alpha * ufl.as_vector((x[0], x[1])))
        J = dxa.assemble_scalar(x[0] * ufl.dx)
        return J, alpha, Q

    J, alpha, Q = forward(0.1)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(alpha))
    direction = dxa.Function(Q)
    direction.x.array[:] = 1.0
    owned = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
    directional = MPI.COMM_WORLD.allreduce(
        float(np.dot(Jhat.derivative().x.array[:owned], direction.x.array[:owned])), op=MPI.SUM
    )

    eps = 1e-6
    fd = (float(forward(0.1 + eps)[0]) - float(forward(0.1 - eps)[0])) / (2 * eps)
    assert abs(fd) > 1e-8, f"the finite difference is ~0 ({fd}): this direction tests nothing"
    assert np.isclose(directional, fd, rtol=1e-5), f"adjoint {directional} vs finite difference {fd}"


def test_move_by_a_coordinate_expression_matches_the_equivalent_function():
    """Moving by an expression and by its interpolation must land on the same geometry.

    Pins the forward half: whatever the tape does with it, ``move`` applied to an expression
    has to displace the mesh exactly as ``move`` applied to that expression interpolated by
    hand does. A mismatch here would mean the packing changed the displacement itself.
    """
    pyadjoint.get_working_tape().clear_tape()
    by_expression = dxa.Mesh(_unit_square(6))
    x = ufl.SpatialCoordinate(by_expression)
    dxa.move(by_expression, ufl.as_vector((0.1 * x[1], -0.1 * x[0])))

    pyadjoint.get_working_tape().clear_tape()
    by_function = dxa.Mesh(_unit_square(6))
    S = dxa.geometry_function_space(by_function)
    s = dxa.Function(S)
    s.interpolate(lambda p: np.vstack((0.1 * p[1], -0.1 * p[0])))
    dxa.move(by_function, s)

    assert np.allclose(by_expression.geometry.x, by_function.geometry.x)


def test_move_by_an_expression_in_another_space_is_interpolated():
    """A Function in some other space on this mesh is interpolated, not refused.

    Only a Function already in the geometry space takes the fast path; anything else goes
    through the annotating interpolation, so a P2 vector field is a legal displacement.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = dxa.Mesh(_unit_square(6))
    reference = _cell_jacobians(mesh)
    W = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim,)))
    s = dxa.Function(W)
    s.interpolate(_interior_bump_values)
    # Elsewhere this field is a Taylor direction and is always scaled; applied at full
    # amplitude it is the size of the domain and tangles the mesh.
    s.x.array[:] *= 0.05
    before = mesh.geometry.x.copy()

    dxa.move(mesh, s)

    assert not np.allclose(mesh.geometry.x, before), "the mesh did not move"
    _assert_mesh_is_valid(mesh, reference)


def test_move_refuses_a_displacement_from_another_mesh():
    """Interpolating between meshes has its own adjoint, so it is not done silently here."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = dxa.Mesh(_unit_square(4))
    other = _unit_square(4)
    foreign = dxa.Function(dxa.geometry_function_space(other))
    with pytest.raises(ValueError, match="different mesh"):
        dxa.move(mesh, foreign)


def test_move_requires_an_explicitly_tracked_mesh():
    """Tracking is opt-in: ``move`` will not promote a plain mesh behind the user's back.

    Promoting here would be too late for anything already posed on the mesh, and would leave
    every later form on it carrying a shape dependency nobody asked for.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = _unit_square(4)
    s = dxa.Function(dxa.geometry_function_space(mesh))
    with pytest.raises(ValueError, match="dolfinx_adjoint.Mesh"):
        dxa.move(mesh, s)


def test_tracking_a_mesh_copies_nothing():
    """``dolfinx_adjoint.Mesh(m)`` returns ``m`` itself, promoted -- not a copy."""
    pyadjoint.get_working_tape().clear_tape()
    plain = _unit_square(4)
    coordinates = plain.geometry.x
    tracked = dxa.Mesh(plain)

    assert tracked is plain, "tracking must not create a second mesh"
    # `geometry.x` hands back a fresh view on each access, so identity is not the question --
    # whether the two views address the same buffer is.
    assert np.shares_memory(tracked.geometry.x, coordinates), "tracking must not copy the coordinates"
    assert dxa.Mesh(tracked) is tracked, "tracking twice must be a no-op"


def test_a_form_on_an_untracked_mesh_takes_no_mesh_dependency():
    """A block only depends on the mesh if the user opted that mesh in.

    Tracking is explicit, so every problem that is not a shape optimization -- which is most of
    them -- must be untouched by shape control: no mesh among the block's dependencies, and no
    refusal of the geometric quantities (here ``FacetNormal``) that a shape derivative cannot
    carry.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = _unit_square(4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = dxa.Function(V)
    u.x.array[:] = 1.0
    n = ufl.FacetNormal(mesh)
    dxa.assemble_scalar(ufl.inner(n, n) * u * ufl.ds)

    blocks = pyadjoint.get_working_tape().get_blocks()
    assert len(blocks) == 1
    dependencies = [dep.output for dep in blocks[0].get_dependencies()]
    assert not any(isinstance(dep, dolfinx.mesh.Mesh) for dep in dependencies)


def test_tracking_a_mesh_late_is_refused_by_move_not_by_the_wrap():
    """Wrapping late is harmless; *moving* a mesh that has stale blocks on it is not.

    A block built before the wrap does not depend on the mesh, so replaying the tape would
    re-run it on whatever geometry the previous replay left behind -- a gradient that drifts
    from the second control value on while every Taylor test still passes. Nothing can go wrong
    until the geometry actually changes, so the wrap is allowed and ``move`` is what refuses.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh = _unit_square(4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = dxa.Function(V)
    u.x.array[:] = 1.0
    dxa.assemble_scalar(u * ufl.dx)

    tracked = dxa.Mesh(mesh)  # allowed: no geometry has changed yet
    s = dxa.Function(dxa.geometry_function_space(tracked))
    with pytest.raises(RuntimeError, match="built before the mesh was annotated"):
        dxa.move(tracked, s)


def _directional(vec, direction, S) -> float:
    """``<vec, direction>``, summed over owned dofs only."""
    owned = S.dofmap.index_map.size_local * S.dofmap.index_map_bs
    return MPI.COMM_WORLD.allreduce(float(np.dot(vec.x.array[:owned], direction.x.array[:owned])), op=MPI.SUM)


@pytest.mark.parametrize("family", ["RT", "N1curl", "Lagrange"])
def test_shape_derivative_of_interpolation_into_a_piola_mapped_space(family):
    """Interpolating into H(div)/H(curl) differentiates the interpolation operator too.

    The dof is not a point evaluation: the expression is evaluated at points on the facet or
    edge and pulled back to the reference cell by a Piola map built from the cell Jacobian
    before the interpolation matrix is applied. The operator is therefore a function of the
    geometry, and differentiating only the expression -- which is the whole answer for a
    Lagrange target, included here as the control -- loses about three quarters of this one.
    """

    def forward(step, direction):
        pyadjoint.get_working_tape().clear_tape()
        mesh = _unit_square(6)
        S = dxa.geometry_function_space(mesh)
        s = dxa.Function(S)
        if step is not None:
            s.x.array[:] = step * direction
        mesh = dxa.Mesh(mesh)
        dxa.move(mesh, s)
        X = ufl.SpatialCoordinate(mesh)
        spec = (family, 1, (mesh.geometry.dim,)) if family == "Lagrange" else (family, 1)
        V = dolfinx.fem.functionspace(mesh, spec)
        u = dxa.interpolate(ufl.as_vector((X[1] ** 2 + 1.0, X[0] ** 2 + 2.0)), V)
        return dxa.assemble_scalar(ufl.inner(u, u) * ufl.dx), s, S

    d = dxa.Function(dxa.geometry_function_space(_unit_square(6)))
    d.interpolate(_dilation_values)
    h = d.x.array.copy()
    eps = 1e-6
    fd = (float(forward(eps, h)[0]) - float(forward(-eps, h)[0])) / (2 * eps)
    assert abs(fd) > 1e-8, "the direction must actually change the functional"

    J, s, S = forward(None, h)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    hf = dxa.Function(S)
    hf.x.array[:] = h
    assert abs(_directional(Jhat.derivative(), hf, S) - fd) < 1e-6 * abs(fd)


@pytest.mark.parametrize("family", ["RT", "N1curl"])
def test_shape_derivative_of_a_form_with_a_piola_mapped_coefficient(family):
    """The form path needs no such handling: UFL pulls back before differentiating.

    Worth pinning separately from the interpolation case above, because the two get the
    Jacobian terms by entirely different routes -- here the form compiler's, there
    :py:meth:`ExprInterpolationBlock._coordinate_derivative`'s. The coefficient is held fixed in
    its dofs, which is what the adjoint assumes.

    The direction is a shear rather than the dilation used everywhere else in this file, because
    ``int |u|^2 dx`` is *invariant* under a uniform dilation for both Piola families: scaling the
    square by ``1 + h`` scales ``u`` by ``1 / (1 + h)`` -- contravariant ``J / det J`` and
    covariant ``J^-T`` both do -- while ``dx`` scales by ``(1 + h)^2``. The derivative is then
    exactly zero and the test would pass without testing anything.
    """

    def forward(step, direction):
        pyadjoint.get_working_tape().clear_tape()
        mesh = _unit_square(6)
        S = dxa.geometry_function_space(mesh)
        s = dxa.Function(S)
        if step is not None:
            s.x.array[:] = step * direction
        mesh = dxa.Mesh(mesh)
        dxa.move(mesh, s)
        V = dolfinx.fem.functionspace(mesh, (family, 1))
        u = dxa.Function(V)
        # A pattern keyed on the *global* dof index, so every rank fills the same field.
        imap = V.dofmap.index_map
        indices = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)
        u.x.array[:] = np.sin(imap.local_to_global(indices).astype(np.float64))
        return dxa.assemble_scalar(ufl.inner(u, u) * ufl.dx), s, S

    shear = dxa.Function(dxa.geometry_function_space(_unit_square(6)))
    shear.interpolate(lambda x: np.vstack((x[0] * (1.0 + 0.5 * x[1]), x[1])))
    h = shear.x.array.copy()
    eps = 1e-6
    fd = (float(forward(eps, h)[0]) - float(forward(-eps, h)[0])) / (2 * eps)
    assert abs(fd) > 1e-8, "the direction must actually change the functional"

    J, s, S = forward(None, h)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    hf = dxa.Function(S)
    hf.x.array[:] = h
    assert abs(_directional(Jhat.derivative(), hf, S) - fd) < 1e-6 * abs(fd)


_PULLBACK_OPERANDS = {
    "IdentityPullback": "scalar",
    "ContravariantPiola": "vector",
    "CovariantPiola": "vector",
    "L2Piola": "scalar",
    "DoubleContravariantPiola": "tensor",
    "DoubleCovariantPiola": "tensor",
    "CovariantContravariantPiola": "tensor",
}


@pytest.mark.parametrize("name", sorted(_PULLBACK_OPERANDS))
def test_the_manual_pullback_inverse_matches_ufls(name):
    """The closed forms in ``compat`` must agree with UFL's own ``apply_inverse``.

    They exist only for a UFL predating FEniCS/ufl#511, so on a current UFL they are never
    reached and would rot unnoticed. Comparing them against the implementation they stand in for
    is the only thing keeping them honest.
    """
    import ufl.pullback

    from dolfinx_adjoint.compat import _INVERSE_PULLBACKS

    pullback = getattr(ufl.pullback, name)()
    if not hasattr(pullback, "apply_inverse"):
        pytest.skip(f"this UFL has no {name}.apply_inverse to compare against")

    mesh = _unit_square(4)
    domain = mesh.ufl_domain()
    X = ufl.SpatialCoordinate(mesh)
    operand = {
        "scalar": lambda: X[0] ** 2 + X[1] + 1.0,
        "vector": lambda: ufl.as_vector((X[1] ** 2 + 1.0, X[0] ** 2 + 2.0)),
        "tensor": lambda: ufl.as_matrix(((X[0] + 1.0, X[1]), (X[1] ** 2, X[0] * X[1] + 2.0))),
    }[_PULLBACK_OPERANDS[name]]()

    reference = pullback.apply_inverse(operand, domain)
    manual = _INVERSE_PULLBACKS[name](operand, domain)

    def norm(e):
        local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(e, e) * ufl.dx))
        return np.sqrt(MPI.COMM_WORLD.allreduce(local, op=MPI.SUM))

    assert norm(reference) > 1e-8, "the comparison must not be against zero"
    assert norm(reference - manual) < 1e-12 * norm(reference)


def _nonmatching_forward(
    step, direction, moved: str, vector: bool = False, same_mesh: bool = False, target_degree: int = 1
):
    """A functional of ``interpolate_nonmatching(u, V_B)``, with one mesh displaced.

    ``u``'s dofs are set on the *undeformed* mesh, before the move. The adjoint holds a
    coefficient's dofs fixed, so the finite difference has to as well; interpolating after the
    move would re-sample the function at the moved nodes and hold the function fixed instead --
    for a quadratic in P2 that makes a source-mesh displacement change nothing at all.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh_a = _unit_square(7)
    mesh_b = mesh_a if same_mesh else _unit_square(5)
    tracked = dxa.Mesh(mesh_a if moved == "source" else mesh_b)

    shape = (2,) if vector else ()
    V_a = dolfinx.fem.functionspace(mesh_a, ("Lagrange", 2, shape))
    u = dxa.Function(V_a)
    if vector:
        u.interpolate(lambda x: np.vstack((x[0] ** 2 + 2.0 * x[1], x[1] ** 2 - x[0])))
    else:
        u.interpolate(lambda x: x[0] ** 2 + 2.0 * x[1])

    S = dxa.geometry_function_space(tracked)
    s = dxa.Function(S)
    if step is not None:
        s.x.array[:] = step * direction
    dxa.move(tracked, s)

    u_b = dxa.interpolate_nonmatching(u, dolfinx.fem.functionspace(mesh_b, ("Lagrange", target_degree, shape)))
    return dxa.assemble_scalar(ufl.inner(u_b, u_b) * ufl.dx), s, S


def _contraction(n: int) -> np.ndarray:
    """A displacement towards the centre, so the moved mesh stays inside the other one."""
    d = dolfinx.fem.Function(dxa.geometry_function_space(_unit_square(n)))
    d.interpolate(lambda x: np.vstack((0.3 * (0.5 - x[0]), 0.3 * (0.5 - x[1]))))
    return d.x.array.copy()


@pytest.mark.parametrize(
    "moved,vector,same_mesh,target_degree",
    [
        ("target", False, False, 1),
        ("source", False, False, 1),
        ("target", True, False, 1),
        ("source", True, False, 1),
        ("target", False, True, 1),
        ("target", False, False, 2),
        ("source", False, False, 2),
    ],
    ids=["target", "source", "target-vector", "source-vector", "same-mesh", "target-P2", "source-P2"],
)
def test_shape_derivative_of_nonmatching_interpolation(moved, vector, same_mesh, target_degree):
    """Both ways a non-matching interpolation depends on the geometry, checked by value.

    Writing ``c_i = u_A(x_i)`` for the target dofs: displacing the target moves the points,
    ``dc_i = grad(u_A)(x_i) . dx_B(x_i)``; displacing the source carries ``u_A`` along with its
    mesh, ``du_A = -grad(u_A) . s_A``. On one mesh the two cancel, leaving only the measure's
    contribution from the functional -- which a wrong sign on either would not.

    The P2 targets matter in parallel: a P1 target's interpolation points are the geometry
    vertices, so each row touches only its own vertex, while a P2 target's edge-midpoint rows
    straddle vertices owned by different ranks. A missing ghost reduction was 2.3% wrong on three
    ranks and invisible with P1.
    """
    pytest.importorskip("fenicsx_ii")
    h = _contraction(7 if (moved == "source" or same_mesh) else 5)
    eps = 1e-6
    # Every rebuild first: a clear_tape() under a live ReducedFunctional invalidates it.
    fd = (
        float(_nonmatching_forward(eps, h, moved, vector, same_mesh, target_degree)[0])
        - float(_nonmatching_forward(-eps, h, moved, vector, same_mesh, target_degree)[0])
    ) / (2 * eps)
    assert abs(fd) > 1e-8, "the direction must actually change the functional"

    J, s, S = _nonmatching_forward(None, h, moved, vector, same_mesh, target_degree)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    hf = dxa.Function(S)
    hf.x.array[:] = h
    assert abs(_directional(Jhat.derivative(), hf, S) - fd) < 1e-6 * abs(fd)
    assert abs(float(Jhat.tlm(hf)) - fd) < 1e-6 * abs(fd), "the tangent-linear model must agree too"


@pytest.mark.parametrize("moved", ["source", "target"])
def test_nonmatching_interpolation_replays_at_the_moved_geometry(moved):
    """Where the meshes sit relative to each other has to be recomputed on every replay.

    ``interpolation_data`` locates each target point inside a source cell and stores its
    reference coordinates there. Kept from the first evaluation, it describes a geometry that no
    longer exists, and the replayed *value* is wrong before any derivative is involved -- 2% when
    the target moves and more than 100% when the source does, for this displacement.
    """
    pytest.importorskip("fenicsx_ii")
    h = _contraction(7 if moved == "source" else 5)
    steps = (0.05, 0.10, 0.05)
    rebuilt = {step: float(_nonmatching_forward(step, h, moved)[0]) for step in steps}
    J, s, S = _nonmatching_forward(None, h, moved)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    for step in steps:
        f = dxa.Function(S)
        f.x.array[:] = step * h
        assert abs(float(Jhat(f)) - rebuilt[step]) < 1e-11 * abs(rebuilt[step]), f"s = {step}"


def _hessian_and_finite_difference(Jhat, controls, directions, eps=1e-4) -> tuple[float, float]:
    """``<H h, h>`` summed over the controls, and a central difference of ``<dJ, h>``."""
    Jhat(controls)
    Jhat.derivative()
    hessian = Jhat.hessian(directions)
    hessian = hessian if isinstance(hessian, list) else [hessian]
    Hh = sum(float(Hi._ad_dot(d)) for Hi, d in zip(hessian, directions))

    def directional_gradient(scale: float) -> float:
        Jhat([c._ad_add(d._ad_mul(scale)) for c, d in zip(controls, directions)])
        gradient = Jhat.derivative()
        gradient = gradient if isinstance(gradient, list) else [gradient]
        return sum(float(g._ad_dot(d)) for g, d in zip(gradient, directions))

    fd = (directional_gradient(eps) - directional_gradient(-eps)) / (2 * eps)
    Jhat(controls)
    return Hh, fd


@pytest.mark.parametrize("family", ["Lagrange", "RT", "N1curl"])
def test_shape_hessian_through_an_interpolated_expression(family):
    """Second order through ``interpolate``, including a Piola-mapped target.

    Both coordinate derivatives are taken in the reference frame before anything is converted
    back; converting after the first would leave ``grad`` of the tangent-linear direction in the
    expression, which UFL refuses to differentiate in physical space. The difference is checked
    at a step where its truncation error (~ eps^2) is far below the tolerance.
    """
    mesh, S, s, _ = _shape_setup(6)
    X = ufl.SpatialCoordinate(mesh)
    if family == "Lagrange":
        u = dxa.interpolate(ufl.sin(ufl.pi * X[0]) * X[1] ** 2, dolfinx.fem.functionspace(mesh, ("Lagrange", 2)))
    else:
        V = dolfinx.fem.functionspace(mesh, (family, 1))
        u = dxa.interpolate(ufl.as_vector((X[1] ** 2 + X[0] ** 3, X[0] * X[1] + 2.0)), V)
    J = dxa.assemble_scalar(ufl.inner(u, u) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    h = dxa.Function(S)
    h.interpolate(lambda x: np.vstack((x[0] * (1.0 + 0.5 * x[1]), x[1] ** 2)))
    Hh, fd = _hessian_and_finite_difference(Jhat, [s], [h])
    assert abs(fd) > 1e-8, "the direction must actually curve the functional"
    assert abs(Hh - fd) < 1e-5 * abs(fd), f"hessian {Hh} vs finite difference {fd}"


def _nonmatching_hessian_forward(moved: str, vector: bool, target_degree: int, u_control: bool):
    """A non-matching interpolation with the target strictly inside the source, in general position.

    General position is not cosmetic. At a target point lying on a source cell's edge,
    ``grad(u)`` of a C0 field and ``grad(sigma)`` of the source displacement jump, so the shape
    derivative there is one-sided -- and the second derivative, which reads ``grad(sigma)``, is
    not defined. A target placed on the source's grid (as the first-order tests' is, harmlessly,
    since they read only ``grad(u)`` of an exactly represented quadratic) sits on those edges.
    Strictly inside also keeps every perturbed target point inside the source mesh.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh_a = _unit_square(7)
    mesh_b = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.1, 0.1]), np.array([0.9, 0.9])], [5, 5])
    mesh_b.geometry.x[:, :2] += np.array([0.0123, 0.0071])
    tracked = {"source": [mesh_a], "target": [mesh_b], "both": [mesh_a, mesh_b]}[moved]
    tracked = [dxa.Mesh(mesh) for mesh in tracked]

    shape = (2,) if vector else ()
    u = dxa.Function(dolfinx.fem.functionspace(mesh_a, ("Lagrange", 2, shape)))
    if vector:
        u.interpolate(lambda x: np.vstack((np.sin(2.0 * x[0]) + x[1] ** 3, x[0] * x[1] ** 2)))
    else:
        u.interpolate(lambda x: np.sin(2.0 * x[0]) + x[0] * x[1] ** 3)

    controls, directions = [], []
    for mesh in tracked:
        s = dxa.Function(dxa.geometry_function_space(mesh))
        dxa.move(mesh, s)
        h = dxa.Function(s.function_space)
        h.interpolate(lambda x: np.vstack((0.05 * (0.5 - x[0]) + 0.02 * x[0] * x[1], 0.04 * (0.5 - x[1]))))
        controls.append(s)
        directions.append(h)
    u_b = dxa.interpolate_nonmatching(u, dolfinx.fem.functionspace(mesh_b, ("Lagrange", target_degree, shape)))
    J = dxa.assemble_scalar(ufl.inner(u_b, u_b) * ufl.dx)
    if u_control:
        du = dxa.Function(u.function_space)
        if vector:
            du.interpolate(lambda x: np.vstack((np.cos(x[0]) * x[1], x[0] ** 2)))
        else:
            du.interpolate(lambda x: np.cos(x[0]) * x[1])
        controls.append(u)
        directions.append(du)
    return J, controls, directions


@pytest.mark.parametrize(
    "moved,vector,target_degree,u_control",
    [
        ("source", False, 1, False),
        ("source", True, 1, False),
        ("source", False, 2, False),
        ("target", False, 2, False),
        ("both", False, 1, False),
        ("both", False, 1, True),
        ("source", True, 1, True),
    ],
    ids=["source", "source-vector", "source-P2", "target-P2", "both", "both+coefficient", "source-vector+coefficient"],
)
def test_shape_hessian_of_nonmatching_interpolation(moved, vector, target_degree, u_control):
    r"""Second order through ``interpolate_nonmatching``, against a difference of the gradient.

    With ``d_k = dx_k(x_i) - sigma_k(x_i)`` the motion of each target point relative to the
    source mesh, ``d2c = d1^T H_u d2 - grad(u).(grad(sigma2) d1) - grad(u).(grad(sigma1) d2)
    + grad(u1).d2 + grad(u2).d1``. The cases cover each term: the source Hessian ``H_u`` (P2
    source), ``grad(sigma)`` (source moving), both meshes at once, and the coefficient cross
    terms.
    """
    pytest.importorskip("fenicsx_ii")
    J, controls, directions = _nonmatching_hessian_forward(moved, vector, target_degree, u_control)
    Jhat = pyadjoint.ReducedFunctional(J, [pyadjoint.Control(c) for c in controls])
    Hh, fd = _hessian_and_finite_difference(Jhat, controls, directions)
    assert abs(fd) > 1e-9, "the direction must actually curve the functional"
    assert abs(Hh - fd) < 1e-6 * abs(fd) + 1e-13, f"hessian {Hh} vs finite difference {fd}"


def _piola_nonmatching_forward(element_from, element_to, moved: str, step, direction):
    """``int |interpolate_nonmatching(u, V_B)|^2 dx`` with one mesh displaced, in general position.

    ``"same"`` interpolates onto the source's own mesh. The target otherwise sits strictly inside
    the source and off its grid: a Piola-mapped field's tangential (or normal) component jumps
    across cell boundaries, so a point evaluation there is not merely one-sided but discontinuous
    in the geometry, and a finite difference measures the jump.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh_a = _unit_square(7)
    if moved == "same":
        mesh_b = mesh_a
    else:
        mesh_b = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.1, 0.1]), np.array([0.9, 0.9])], [5, 5])
        mesh_b.geometry.x[:, :2] += np.array([0.0123, 0.0071])
    tracked = dxa.Mesh(mesh_b if moved == "target" else mesh_a)

    V_a = dolfinx.fem.functionspace(mesh_a, element_from)
    u = dxa.Function(V_a)
    u.interpolate(lambda x: np.vstack((np.sin(2.0 * x[0]) + x[1] ** 2, x[0] * x[1] + 1.0)))
    S = dxa.geometry_function_space(tracked)
    s = dxa.Function(S)
    if step is not None:
        s.x.array[:] = step * direction
    dxa.move(tracked, s)
    u_b = dxa.interpolate_nonmatching(u, dolfinx.fem.functionspace(mesh_b, element_to))
    return dxa.assemble_scalar(ufl.inner(u_b, u_b) * ufl.dx), s, S


def _general_motion(x: np.ndarray) -> np.ndarray:
    """A contraction towards the centre with a shear on top, so no Piola invariance hides a term."""
    return np.vstack((0.05 * (0.5 - x[0]) + 0.02 * x[0] * x[1], 0.04 * (0.5 - x[1]) + 0.01 * x[0] ** 2))


@pytest.mark.parametrize("moved", ["target", "source", "same"])
@pytest.mark.parametrize(
    "element_from,element_to",
    [
        (("RT", 1), ("RT", 1)),
        (("Lagrange", 2, (2,)), ("RT", 1)),
        (("RT", 1), ("Lagrange", 1, (2,))),
        (("N1curl", 1), ("N1curl", 1)),
        (("Lagrange", 2, (2,)), ("N1curl", 2)),
    ],
    ids=["RT-RT", "P2-RT", "RT-P1", "N1curl-N1curl", "P2-N1curl2"],
)
def test_shape_derivative_of_nonmatching_interpolation_with_piola_spaces(element_from, element_to, moved):
    """A Piola map on either side adds a term to each mesh's shape derivative.

    A Piola-mapped target's dofs are ``M . P_B^-1(u_A(x_p))``, so moving the target mesh changes
    ``J_B`` as well as the points. A Piola-mapped source changes with its mesh through ``J_A``
    and ``det J_A``, not only by being carried along. On one mesh the terms cancel, leaving the
    functional's own measure term. Checked by value, adjoint and tangent-linear.
    """
    pytest.importorskip("fenicsx_ii")
    if moved == "same" and element_from[0] != "Lagrange" and element_to[0] == "Lagrange":
        pytest.skip("point values of a Piola field at its own mesh's vertices jump with the geometry")
    reference = (
        _unit_square(7)
        if moved != "target"
        else dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.1, 0.1]), np.array([0.9, 0.9])], [5, 5])
    )
    d = dolfinx.fem.Function(dxa.geometry_function_space(reference))
    d.interpolate(_general_motion)
    h = d.x.array.copy()
    eps = 1e-6
    fd = (
        float(_piola_nonmatching_forward(element_from, element_to, moved, eps, h)[0])
        - float(_piola_nonmatching_forward(element_from, element_to, moved, -eps, h)[0])
    ) / (2 * eps)
    assert abs(fd) > 1e-8, "the direction must actually change the functional"

    J, s, S = _piola_nonmatching_forward(element_from, element_to, moved, None, h)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    hf = dxa.Function(S)
    hf.x.array[:] = h
    assert abs(_directional(Jhat.derivative(), hf, S) - fd) < 1e-6 * abs(fd)
    assert abs(float(Jhat.tlm(hf)) - fd) < 1e-6 * abs(fd), "the tangent-linear model must agree too"


_PIOLA_PAIRS = [
    (("RT", 1), ("RT", 1)),
    (("Lagrange", 2, (2,)), ("RT", 1)),
    (("RT", 1), ("Lagrange", 1, (2,))),
    (("N1curl", 1), ("N1curl", 1)),
]
_PIOLA_IDS = ["RT-RT", "P2-RT", "RT-P1", "N1curl-N1curl"]


@pytest.mark.parametrize("element_from,element_to", _PIOLA_PAIRS, ids=_PIOLA_IDS)
def test_piola_nonmatching_forward_matches_dolfinx(element_from, element_to):
    """Recording a Piola case must not change the forward value.

    A Piola-mapped target is split into a quadrature-space step and an expression interpolation,
    which reads only the values at its interpolation points; a Piola-mapped source stays one
    block. Either way the result is DOLFINx's direct interpolation.
    """
    pytest.importorskip("fenicsx_ii")
    d = dolfinx.fem.Function(dxa.geometry_function_space(_unit_square(7)))
    d.interpolate(_general_motion)
    J, _, _ = _piola_nonmatching_forward(element_from, element_to, "source", 0.3, d.x.array.copy())
    blocks = pyadjoint.get_working_tape().get_blocks()
    split = any(isinstance(block, ExprInterpolationBlock) for block in blocks)
    assert split == (element_to[0] != "Lagrange"), "only a Piola-mapped target is split"

    with pyadjoint.stop_annotating():
        mesh_a = _unit_square(7)
        mesh_b = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.1, 0.1]), np.array([0.9, 0.9])], [5, 5])
        mesh_b.geometry.x[:, :2] += np.array([0.0123, 0.0071])
        V_a = dolfinx.fem.functionspace(mesh_a, element_from)
        u = dolfinx.fem.Function(V_a)
        u.interpolate(lambda x: np.vstack((np.sin(2.0 * x[0]) + x[1] ** 2, x[0] * x[1] + 1.0)))
        mesh_a.geometry.x[:, :2] += 0.3 * d.x.array.reshape(-1, 2)  # after u: its dofs move with the mesh
        V_b = dolfinx.fem.functionspace(mesh_b, element_to)
        cells = np.arange(mesh_b.topology.index_map(2).size_local, dtype=np.int32)
        v = dolfinx.fem.Function(V_b)
        v.interpolate_nonmatching(u, cells, dolfinx.fem.create_interpolation_data(V_b, V_a, cells, padding=1e-6))
        local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(ufl.inner(v, v) * ufl.dx))
        direct = MPI.COMM_WORLD.allreduce(local, op=MPI.SUM)
    assert abs(float(J) - direct) < 1e-12 * abs(direct), f"split {float(J)} vs direct {direct}"


def _piola_nonmatching_hessian_forward(element_from, element_to, moved: str):
    """``_piola_nonmatching_forward`` at zero displacement, with every tracked mesh a control
    and the coefficient as a further one."""
    pyadjoint.get_working_tape().clear_tape()
    mesh_a = _unit_square(7)
    mesh_b = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.1, 0.1]), np.array([0.9, 0.9])], [5, 5])
    mesh_b.geometry.x[:, :2] += np.array([0.0123, 0.0071])
    tracked = [dxa.Mesh(mesh) for mesh in {"source": [mesh_a], "target": [mesh_b], "both": [mesh_a, mesh_b]}[moved]]
    V_a = dolfinx.fem.functionspace(mesh_a, element_from)
    u = dxa.Function(V_a)
    u.interpolate(lambda x: np.vstack((np.sin(2.0 * x[0]) + x[1] ** 2, x[0] * x[1] + 1.0)))
    controls, directions = [], []
    for mesh in tracked:
        s = dxa.Function(dxa.geometry_function_space(mesh))
        dxa.move(mesh, s)
        h = dxa.Function(s.function_space)
        h.interpolate(_general_motion)
        controls.append(s)
        directions.append(h)
    u_b = dxa.interpolate_nonmatching(u, dolfinx.fem.functionspace(mesh_b, element_to))
    J = dxa.assemble_scalar(ufl.inner(u_b, u_b) * ufl.dx)
    du = dxa.Function(V_a)
    du.interpolate(lambda x: np.vstack((np.cos(x[0]) * x[1], x[0] ** 2)))
    controls.append(u)
    directions.append(du)
    return J, controls, directions


@pytest.mark.parametrize("moved", ["target", "source", "both"])
@pytest.mark.parametrize("element_from,element_to", _PIOLA_PAIRS, ids=_PIOLA_IDS)
def test_shape_hessian_of_nonmatching_interpolation_with_piola_spaces(element_from, element_to, moved):
    """Second order with a Piola map on either side, against a difference of the gradient.

    Every tracked mesh and the coefficient are controls at once, so the cross terms between the
    Piola maps, the point motion and the coefficient direction are all exercised.
    """
    pytest.importorskip("fenicsx_ii")
    J, controls, directions = _piola_nonmatching_hessian_forward(element_from, element_to, moved)
    Jhat = pyadjoint.ReducedFunctional(J, [pyadjoint.Control(c) for c in controls])
    Hh, fd = _hessian_and_finite_difference(Jhat, controls, directions)
    assert abs(fd) > 1e-9, "the direction must actually curve the functional"
    assert abs(Hh - fd) < 1e-6 * abs(fd), f"hessian {Hh} vs finite difference {fd}"


def test_nonmatching_interpolation_without_shape_control_still_works():
    """The refusal must not reach the ordinary case, which is most of them."""
    pytest.importorskip("fenicsx_ii")
    pyadjoint.get_working_tape().clear_tape()
    mesh_a = _unit_square(7)
    mesh_b = _unit_square(5)
    V = dolfinx.fem.functionspace(mesh_a, ("Lagrange", 2))
    u = dxa.Function(V)
    u.x.array[:] = 1.0

    ub = dxa.interpolate_nonmatching(u, dolfinx.fem.functionspace(mesh_b, ("Lagrange", 1)))
    J = dxa.assemble_scalar(ufl.inner(ub, ub) * ufl.dx)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(u))
    h = dxa.Function(V)
    h.interpolate(lambda x: np.sin(np.pi * x[0]) * np.cos(np.pi * x[1]))
    assert pyadjoint.taylor_test(Jhat, u, h) > 1.9


# --- Coverage of the rows that were untested --------------------------------------------------
#
# Each checked by value: gradients against a central difference of an independently rebuilt
# forward, Hessians against a central difference of the gradient.


def _curved_square(n: int = 4) -> dolfinx.mesh.Mesh:
    """A unit square with a curved (P2) geometry: straight-sided cells mapped by a smooth bend.

    Built on rank 0 and partitioned by ``create_mesh``. Edge nodes follow basix's order,
    edge ``i`` opposite vertex ``i``.
    """
    import basix.ufl

    if MPI.COMM_WORLD.rank == 0:
        m = 2 * n + 1
        grid = np.array([(i / (m - 1), j / (m - 1)) for j in range(m) for i in range(m)])

        def node(i, j):
            return j * m + i

        cells = []
        for j in range(n):
            for i in range(n):
                a, b, c, d = (2 * i, 2 * j), (2 * i + 2, 2 * j), (2 * i, 2 * j + 2), (2 * i + 2, 2 * j + 2)
                mid = (2 * i + 1, 2 * j + 1)
                cells.append(
                    [node(*a), node(*b), node(*d), node(2 * i + 2, 2 * j + 1), node(*mid), node(2 * i + 1, 2 * j)]
                )
                cells.append(
                    [node(*a), node(*d), node(*c), node(2 * i + 1, 2 * j + 2), node(2 * i, 2 * j + 1), node(*mid)]
                )
        x = grid.copy()
        x[:, 0] += 0.08 * grid[:, 1] * (1 - grid[:, 1])  # bends the vertical lines
        x[:, 1] += 0.06 * np.sin(np.pi * grid[:, 0])  # and the horizontal ones
        cells_arr = np.array(cells, dtype=np.int64)
    else:
        x = np.zeros((0, 2))
        cells_arr = np.zeros((0, 6), dtype=np.int64)
    element = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 2, shape=(2,)))
    return dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells_arr, element, x)


_MESHES = {
    "curved-P2": lambda: _curved_square(4),
    "quadrilateral": lambda: dolfinx.mesh.create_unit_square(
        MPI.COMM_WORLD, 5, 5, cell_type=dolfinx.mesh.CellType.quadrilateral
    ),
    "tetrahedron": lambda: dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, 3, 3, 3),
    "hexahedron": lambda: dolfinx.mesh.create_unit_cube(
        MPI.COMM_WORLD, 2, 2, 2, cell_type=dolfinx.mesh.CellType.hexahedron
    ),
}


def _bend(x: np.ndarray, gdim: int) -> np.ndarray:
    """A non-uniform displacement direction in ``gdim`` dimensions."""
    return np.vstack([x[k] * (1.0 + 0.5 * x[(k + 1) % gdim]) for k in range(gdim)])


def _poisson_forward(make_mesh, values=None):
    """``int u^2 + |x|^2 dx`` for a Poisson solve with a coordinate-dependent source."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = make_mesh()
    S = dxa.geometry_function_space(mesh)
    s = dxa.Function(S)
    if values is not None:
        s.x.array[:] = values
    mesh = dxa.Mesh(mesh)
    dxa.move(mesh, s)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    problem = dxa.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        ufl.inner(ufl.sin(ufl.pi * X[0]) * ufl.cos(ufl.pi * X[1]), v) * ufl.dx,
        bcs=[_homogeneous_bc(V)],
        petsc_options=_LU,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_cells_",
    )
    uh = problem.solve()
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx + ufl.inner(X, X) * ufl.dx)
    return J, s, S, problem


def _direction_values(make_mesh, scale: float = 1.0) -> np.ndarray:
    mesh = make_mesh()
    d = dolfinx.fem.Function(dxa.geometry_function_space(mesh))
    gdim = mesh.geometry.dim
    d.interpolate(lambda x: scale * _bend(x, gdim))
    return d.x.array.copy()


def _assert_gradient_by_value(forward, h: np.ndarray, eps: float = 1e-6, rtol: float = 1e-6):
    fd = (float(forward(eps * h)[0]) - float(forward(-eps * h)[0])) / (2 * eps)
    assert abs(fd) > 1e-8, f"the finite difference is ~0 ({fd}): this direction tests nothing"
    J, s, S, _keep = forward()
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    hf = dxa.Function(S)
    hf.x.array[:] = h
    directional = _directional(Jhat.derivative(), hf, S)
    assert abs(directional - fd) < rtol * abs(fd), f"adjoint {directional} vs finite difference {fd}"


def _assert_hessian_by_value(J, controls, directions, rtol: float = 1e-5):
    Jhat = pyadjoint.ReducedFunctional(J, [pyadjoint.Control(c) for c in controls])
    Hh, fd = _hessian_and_finite_difference(Jhat, controls, directions)
    assert abs(fd) > 1e-9, "the direction must actually curve the functional"
    assert abs(Hh - fd) < rtol * abs(fd), f"hessian {Hh} vs finite difference {fd}"


def _non_affine_quadrilateral(n: int = 4) -> dolfinx.mesh.Mesh:
    """Unit-square quadrilaterals bent into non-parallelograms, so each cell map is bilinear."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n, cell_type=dolfinx.mesh.CellType.quadrilateral)
    x = mesh.geometry.x
    x[:, 0] += 0.05 * x[:, 1] * (1 - x[:, 1]) * np.sin(3 * x[:, 0])
    return mesh


_NON_AFFINE_SOURCES = {"curved-P2": lambda: _curved_square(4), "non-affine-quadrilateral": _non_affine_quadrilateral}
_NON_AFFINE_FAMILIES = ["Lagrange", "RT", "N1curl"]


def _non_affine_nonmatching_forward(make_mesh, family: str, values=None, moved=("source",)):
    """``int |interpolate_nonmatching(u, V_B)|^2 dx`` from a non-affine source mesh.

    ``grad(u)`` and a Piola push-forward are rational on these cells, so any intermediate
    polynomial space would make the shape derivative inexact. The target is an identity-pullback
    space on an offset rectangle strictly inside the source. ``values`` displaces the source mesh.
    """
    pyadjoint.get_working_tape().clear_tape()
    mesh_a = make_mesh()
    mesh_b = dolfinx.mesh.create_rectangle(MPI.COMM_WORLD, [np.array([0.2, 0.2]), np.array([0.8, 0.8])], [5, 5])
    mesh_b.geometry.x[:, :2] += np.array([0.0123, 0.0071])
    if family == "Lagrange":
        V_a = dolfinx.fem.functionspace(mesh_a, ("Lagrange", 2))
        V_b = dolfinx.fem.functionspace(mesh_b, ("Lagrange", 1))
        values_a = lambda x: np.sin(2.0 * x[0]) + x[1] ** 3  # noqa: E731
        direction_a = lambda x: np.cos(x[0]) * x[1]  # noqa: E731
    else:
        V_a = dolfinx.fem.functionspace(mesh_a, (family, 1))
        V_b = dolfinx.fem.functionspace(mesh_b, ("Lagrange", 1, (2,)))
        values_a = lambda x: np.vstack((np.sin(2.0 * x[0]) + x[1] ** 2, x[0] * x[1] + 1.0))  # noqa: E731
        direction_a = lambda x: np.vstack((np.cos(x[0]) * x[1], x[0] ** 2))  # noqa: E731
    u = dxa.Function(V_a)
    u.interpolate(values_a)
    controls, directions = [], []
    for mesh in [{"source": mesh_a, "target": mesh_b}[side] for side in moved]:
        tracked = dxa.Mesh(mesh)
        s = dxa.Function(dxa.geometry_function_space(tracked))
        if values is not None:
            s.x.array[:] = values
        dxa.move(tracked, s)
        h = dxa.Function(s.function_space)
        h.interpolate(_general_motion)
        controls.append(s)
        directions.append(h)
    u_b = dxa.interpolate_nonmatching(u, V_b)
    J = dxa.assemble_scalar(ufl.inner(u_b, u_b) * ufl.dx)
    du = dxa.Function(V_a)
    du.interpolate(direction_a)
    return J, controls + [u], directions + [du]


@pytest.mark.parametrize("family", _NON_AFFINE_FAMILIES)
@pytest.mark.parametrize("mesh_kind", sorted(_NON_AFFINE_SOURCES))
def test_shape_derivative_of_nonmatching_interpolation_from_a_non_affine_mesh(mesh_kind, family):
    """Moving a curved or non-affine source mesh, with an identity or a Piola-mapped source."""
    pytest.importorskip("fenicsx_ii")
    make_mesh = _NON_AFFINE_SOURCES[mesh_kind]

    def forward(values=None):
        J, controls, _ = _non_affine_nonmatching_forward(make_mesh, family, values)
        return J, controls[0], controls[0].function_space, None

    d = dolfinx.fem.Function(dxa.geometry_function_space(make_mesh()))
    d.interpolate(_general_motion)
    _assert_gradient_by_value(forward, d.x.array.copy())


@pytest.mark.parametrize("family", _NON_AFFINE_FAMILIES)
@pytest.mark.parametrize("mesh_kind", sorted(_NON_AFFINE_SOURCES))
def test_shape_hessian_of_nonmatching_interpolation_from_a_non_affine_mesh(mesh_kind, family):
    """Second order with both meshes and the coefficient as controls, so every cross term is hit."""
    pytest.importorskip("fenicsx_ii")
    J, controls, directions = _non_affine_nonmatching_forward(
        _NON_AFFINE_SOURCES[mesh_kind], family, moved=("source", "target")
    )
    _assert_hessian_by_value(J, controls, directions)


@pytest.mark.parametrize("mesh_kind", sorted(_MESHES))
def test_shape_gradient_on_curved_3d_and_tensor_product_meshes(mesh_kind):
    """Nothing is specific to affine triangles: P2 geometry, quadrilaterals, tetrahedra, hexahedra."""
    make_mesh = _MESHES[mesh_kind]
    _assert_gradient_by_value(lambda values=None: _poisson_forward(make_mesh, values), _direction_values(make_mesh))


@pytest.mark.parametrize("mesh_kind", sorted(_MESHES))
def test_shape_hessian_on_curved_3d_and_tensor_product_meshes(mesh_kind):
    """The second-order chain on the same meshes."""
    make_mesh = _MESHES[mesh_kind]
    J, s, S, _keep = _poisson_forward(make_mesh)
    h = dxa.Function(S)
    h.x.array[:] = _direction_values(make_mesh, scale=0.1)
    _assert_hessian_by_value(J, [s], [h])


def test_shape_hessian_of_a_time_dependent_problem():
    """Second order through a time loop, where every step's residual carries the previous state."""
    J, s, S, _keep = _heat_loop(3)
    h = dxa.Function(S)
    h.interpolate(_dilation_values)
    h.x.array[:] *= 0.1
    _assert_hessian_by_value(J, [s], [h])


@pytest.mark.parametrize(
    "integrand",
    [
        pytest.param(lambda m: ufl.dot(ufl.SpatialCoordinate(m), ufl.FacetNormal(m)) * ufl.ds, id="x.n*ds"),
        pytest.param(
            lambda m: ufl.FacetArea(m) * ufl.inner(ufl.SpatialCoordinate(m), ufl.SpatialCoordinate(m)) * ufl.ds,
            id="FacetArea*x.x*ds",
        ),
        pytest.param(
            lambda m: ufl.avg(ufl.sin(ufl.SpatialCoordinate(m)[0]) * ufl.SpatialCoordinate(m)[1] ** 2) * ufl.dS,
            id="avg(...)*dS",
        ),
    ],
)
def test_shape_hessian_of_facet_functionals(integrand):
    """Second order over exterior and interior facet integrals."""
    mesh, S, s, _ = _shape_setup(6)
    J = dxa.assemble_scalar(integrand(mesh))
    h = dxa.Function(S)
    h.interpolate(_shear)
    _assert_hessian_by_value(J, [s], [h])


@pytest.mark.parametrize("family", ["RT", "N1curl"])
def test_shape_hessian_of_a_form_with_a_piola_mapped_coefficient(family):
    """Second order for a Piola-mapped coefficient held fixed in its dofs."""
    mesh, S, s, _ = _shape_setup(6)
    V = dolfinx.fem.functionspace(mesh, (family, 1))
    u = dxa.Function(V)
    imap = V.dofmap.index_map
    indices = np.arange(imap.size_local + imap.num_ghosts, dtype=np.int32)
    u.x.array[:] = np.sin(imap.local_to_global(indices).astype(np.float64))
    J = dxa.assemble_scalar(ufl.inner(u, u) ** 2 * ufl.dx)
    h = dxa.Function(S)
    h.interpolate(_shear)
    _assert_hessian_by_value(J, [s], [h])


def test_shape_hessian_through_an_interpolated_dirichlet_value():
    """Second order when a Dirichlet value is built from the coordinates on a moving boundary."""
    mesh, S, s, _ = _shape_setup(6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_value = dxa.interpolate(X[0] ** 2 + X[1], V)
    bc = dxa.dirichletbc(boundary_value, dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets))
    problem = dxa.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        ufl.inner(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), v) * ufl.dx,
        bcs=[bc],
        petsc_options=_LU,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_hessian_bc_",
    )
    uh = problem.solve()
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx)
    h = dxa.Function(S)
    h.interpolate(_dilation_values)
    h.x.array[:] *= 0.1
    _assert_hessian_by_value(J, [s], [h])


def test_shape_hessian_of_a_move_by_an_expression():
    """Second order through ``move`` by an expression, with respect to the coefficient in it."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = dxa.Mesh(_unit_square(6))
    Q = dolfinx.fem.functionspace(mesh, ("DG", 0))
    alpha = dxa.Function(Q, name="amplitude")
    alpha.x.array[:] = 0.1
    x = ufl.SpatialCoordinate(mesh)
    dxa.move(mesh, alpha * ufl.as_vector((x[0] * (1.0 + x[1]), x[1])))
    X = ufl.SpatialCoordinate(mesh)
    J = dxa.assemble_scalar(ufl.sin(X[0]) * X[1] ** 2 * ufl.dx)
    direction = dxa.Function(Q)
    direction.interpolate(lambda p: 1.0 + p[0])
    _assert_hessian_by_value(J, [alpha], [direction])


def _two_moves_forward(values=None):
    """A solve after the first move and a functional after a second, fixed one."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = _unit_square(6)
    S = dxa.geometry_function_space(mesh)
    s = dxa.Function(S)
    if values is not None:
        s.x.array[:] = values
    # Built on the undeformed mesh, so its values do not depend on the first move.
    second = dxa.Function(S)
    second.interpolate(lambda x: np.vstack((0.05 * x[0] * x[1], 0.03 * x[0])))
    mesh = dxa.Mesh(mesh)
    dxa.move(mesh, s)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    problem = dxa.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        ufl.inner(ufl.sin(ufl.pi * X[0]) * ufl.cos(ufl.pi * X[1]), v) * ufl.dx,
        bcs=[_homogeneous_bc(V)],
        petsc_options=_LU,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_two_moves_fd_",
    )
    uh = problem.solve()
    dxa.move(mesh, second)
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx + ufl.inner(X, X) * ufl.dx)
    return J, s, S, problem


def test_shape_gradient_after_two_moves_matches_a_finite_difference():
    """Each block is differentiated at the geometry it saw; checked by value, not by Taylor rate."""
    _assert_gradient_by_value(_two_moves_forward, _direction_values(lambda: _unit_square(6)))


def test_shape_hessian_of_a_blocked_problem():
    """Second order through a blocked Taylor-Hood Stokes solve."""
    import basix.ufl

    mesh, S, s, _ = _shape_setup(4)
    V = dolfinx.fem.functionspace(mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(2,)))
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), 1))
    W = ufl.MixedFunctionSpace(V, Q)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    X = ufl.SpatialCoordinate(mesh)
    f = ufl.as_vector((ufl.sin(ufl.pi * X[1]), ufl.cos(ufl.pi * X[0])))
    a = ufl.extract_blocks(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.div(v) * p * ufl.dx - ufl.div(u) * q * ufl.dx
    )
    rhs = ufl.extract_blocks(ufl.inner(f, v) * ufl.dx + dolfinx.fem.Constant(mesh, 0.0) * q * ufl.dx)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    walls = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda pt: np.isclose(pt[1], 0.0) | np.isclose(pt[1], 1.0) | np.isclose(pt[0], 0.0)
    )
    no_slip = np.zeros(2, dtype=dolfinx.default_scalar_type)
    bcs = [dolfinx.fem.dirichletbc(no_slip, dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, walls), V)]
    uh, ph = dxa.Function(V), dxa.Function(Q)
    problem = dxa.LinearProblem(
        a,
        rhs,
        u=[uh, ph],
        bcs=bcs,
        petsc_options=_LU,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_hessian_blocked_",
    )
    problem.solve()
    J = dxa.assemble_scalar(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx)
    h = _interior_bump(S)
    h.x.array[:] *= 0.1
    _assert_hessian_by_value(J, [s], [h])


def _stokes_with_a_trigonometric_source(values=None):
    """Taylor-Hood Stokes whose source FFCx has to integrate with an estimated quadrature degree."""
    import basix.ufl

    pyadjoint.get_working_tape().clear_tape()
    mesh = _unit_square(4)
    S = dxa.geometry_function_space(mesh)
    s = dxa.Function(S)
    if values is not None:
        s.x.array[:] = values
    mesh = dxa.Mesh(mesh)
    dxa.move(mesh, s)
    V = dolfinx.fem.functionspace(mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(2,)))
    Q = dolfinx.fem.functionspace(mesh, basix.ufl.element("Lagrange", mesh.basix_cell(), 1))
    W = ufl.MixedFunctionSpace(V, Q)
    u, p = ufl.TrialFunctions(W)
    v, q = ufl.TestFunctions(W)
    X = ufl.SpatialCoordinate(mesh)
    f = ufl.as_vector((ufl.sin(ufl.pi * X[1]), ufl.cos(ufl.pi * X[0])))
    a = ufl.extract_blocks(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.div(v) * p * ufl.dx - ufl.div(u) * q * ufl.dx
    )
    rhs = ufl.extract_blocks(ufl.inner(f, v) * ufl.dx + dolfinx.fem.Constant(mesh, 0.0) * q * ufl.dx)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    walls = dolfinx.mesh.locate_entities_boundary(
        mesh, mesh.topology.dim - 1, lambda pt: np.isclose(pt[1], 0.0) | np.isclose(pt[1], 1.0) | np.isclose(pt[0], 0.0)
    )
    no_slip = np.zeros(2, dtype=dolfinx.default_scalar_type)
    bcs = [dolfinx.fem.dirichletbc(no_slip, dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, walls), V)]
    uh, ph = dxa.Function(V), dxa.Function(Q)
    problem = dxa.LinearProblem(
        a,
        rhs,
        u=[uh, ph],
        bcs=bcs,
        petsc_options=_LU,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_quadrature_stokes_",
    )
    problem.solve()
    return dxa.assemble_scalar(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx), s, S, problem


def test_shape_gradient_uses_the_forward_quadrature():
    """A derivative form must be integrated with the rule its forward form was integrated with.

    FFCx estimates a quadrature degree per form, so without pinning, ``dF/dX`` got a different
    degree than ``F`` and was not the exact derivative of what was assembled: 1.5e-3 relative
    here, falling only with mesh refinement. The tolerance is set far below that.
    """
    h = dxa.Function(dxa.geometry_function_space(_unit_square(4)))
    h.interpolate(_interior_bump_values)
    _assert_gradient_by_value(_stokes_with_a_trigonometric_source, h.x.array.copy(), rtol=1e-7)


def _nonlinear_coefficient_forward(amplitude: float):
    """A coarse nonlinear diffusion ``exp(3 m)``, whose derivative in ``m`` FFCx would give its own degree."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = _unit_square(3)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    Q = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    m = dxa.Function(Q, name="m")
    m.interpolate(lambda x: amplitude * (1.0 + x[0] * x[1]))
    uh = dxa.Function(V)
    v = ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    F = ufl.inner(ufl.exp(3 * m) * ufl.grad(uh), ufl.grad(v)) * ufl.dx - ufl.sin(4 * X[0]) * v * ufl.dx
    problem = dxa.NonlinearProblem(
        F, u=uh, bcs=[_homogeneous_bc(V)], petsc_options=_SNES, petsc_options_prefix="test_quadrature_nonlinear_"
    )
    problem.solve()
    return dxa.assemble_scalar(uh * uh * ufl.dx), m, Q, problem


def test_coefficient_gradient_uses_the_forward_quadrature():
    """The same holds for a coefficient derivative: 8.5e-6 relative before pinning."""
    J, m, Q, _keep = _nonlinear_coefficient_forward(0.5)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(m))
    dm = dxa.Function(Q)
    dm.interpolate(lambda x: np.cos(x[0]) + x[1])
    directional = _directional(Jhat.derivative(), dm, Q)
    eps = 1e-6
    values = m.x.array.copy()

    def J_at(step: float) -> float:
        f = dxa.Function(Q)
        f.x.array[:] = values + step * dm.x.array
        return float(Jhat(f))

    fd = (J_at(eps) - J_at(-eps)) / (2 * eps)
    assert abs(fd) > 1e-8
    assert abs(directional - fd) < 1e-7 * abs(fd), f"adjoint {directional} vs finite difference {fd}"


def test_shape_tangent_linear_through_an_interpolated_dirichlet_value():
    """The tangent-linear model when a Dirichlet value is built from the coordinates.

    The geometry's direction reaches the solve twice, as the shape term in its right-hand side
    and as the bc value's own direction; in parallel that combination used to double-count ghost
    contributions (3e-3 on two ranks), and the Hessian built on it inherited the error.
    """
    mesh, S, s, _ = _shape_setup(6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    X = ufl.SpatialCoordinate(mesh)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_value = dxa.interpolate(X[0] ** 2 + X[1], V)
    bc = dxa.dirichletbc(boundary_value, dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets))
    problem = dxa.LinearProblem(
        ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
        ufl.inner(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), v) * ufl.dx,
        bcs=[bc],
        petsc_options=_LU,
        adjoint_petsc_options=_LU,
        tlm_petsc_options=_LU,
        petsc_options_prefix="test_shape_tlm_bc_",
    )
    uh = problem.solve()
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx)
    h = dxa.Function(S)
    h.interpolate(_dilation_values)
    h.x.array[:] *= 0.1
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    tlm = float(Jhat.tlm(h))
    eps = 1e-6
    f = dxa.Function(S)
    f.x.array[:] = eps * h.x.array
    J_plus = float(Jhat(f))
    f.x.array[:] = -eps * h.x.array
    J_minus = float(Jhat(f))
    fd = (J_plus - J_minus) / (2 * eps)
    assert abs(fd) > 1e-8
    assert abs(tlm - fd) < 1e-7 * abs(fd), f"tangent-linear {tlm} vs finite difference {fd}"
