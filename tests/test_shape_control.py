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

# A direct LU solve: the Taylor remainders checked below fall to ~1e-10, which an
# iterative solve's own tolerance would swamp.
_LU = {"ksp_type": "preonly", "pc_type": "lu"}
_SNES = _LU | {"snes_type": "newtonls", "snes_atol": 1e-12, "snes_rtol": 1e-12, "snes_stol": 0.0}


def _unit_square(n: int = 8) -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)


def _dilation_values(x: np.ndarray) -> np.ndarray:
    """A uniform dilation about the origin, ``s(x) = x``, as interpolation values."""
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


def _dilation(S: dolfinx.fem.FunctionSpace) -> dxa.Function:
    """A displacement direction that is a genuine shape change, and admissible at every step.

    The mesh geometry plays the role that a positive diffusivity plays in a coefficient
    control: the problem is well posed only while the perturbed mesh is untangled, so the
    direction has to be admissible at every ``m + h*dm``, not merely at the base point. A
    dilation maps the unit square to ``(1 + h)`` times itself, so no cell can invert for
    any ``h > -1``, far outside the range a Taylor test walks.
    :py:func:`_assert_mesh_is_valid` checks that rather than assuming it.

    It must also *move the boundary*. A displacement supported strictly inside the domain
    only relabels which point of the domain each node sits at: the domain is unchanged, so
    the shape derivative of a functional like ``int_Omega f dx`` is exactly zero and a
    Taylor test on it measures nothing but round-off.
    """
    h = dxa.Function(S)
    h.interpolate(_dilation_values)
    return h


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
    moved = dxa.move(mesh, s)

    assert moved is mesh, "the mesh must be promoted in place, so existing forms stay valid"
    assert isinstance(mesh, dxa.types.Mesh)
    gdim = mesh.geometry.dim
    assert np.allclose(mesh.geometry.x[:, :gdim], before[:, :gdim] + 0.01)
    assert np.allclose(mesh.geometry.x[:, gdim:], before[:, gdim:]), "padding columns must not move"


def test_move_rejects_a_displacement_outside_the_geometry_space():
    """A displacement in the wrong space is refused, pointing at the interpolation that fixes it."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = _unit_square(4)
    W = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim,)))
    with pytest.raises(ValueError, match="geometry function space"):
        dxa.move(mesh, dxa.Function(W))


def test_move_rejects_a_non_function_displacement():
    pyadjoint.get_working_tape().clear_tape()
    mesh = _unit_square(4)
    with pytest.raises(ValueError, match="Function"):
        dxa.move(mesh, ufl.SpatialCoordinate(mesh))  # type: ignore[arg-type]


def test_shape_derivative_of_a_functional():
    """A functional of the coordinates alone, with no PDE in between."""
    mesh, S, s, reference = _shape_setup()
    X = ufl.SpatialCoordinate(mesh)
    J = dxa.assemble_scalar(ufl.sin(X[0]) * ufl.cos(X[1]) * ufl.dx + ufl.inner(X, X) * ufl.ds)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    h = _dilation(S)
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
        petsc_options_prefix="test_shape_linear_",
    )
    uh = problem.solve()
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    h = _dilation(S)
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
        petsc_options_prefix="test_shape_nonlinear_",
    )
    problem.solve()
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    h = _dilation(S)
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
        petsc_options_prefix="test_shape_fd_",
    )
    uh = problem.solve()
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    h = _dilation(S)
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
        petsc_options_prefix="test_shape_joint_",
    )
    uh = problem.solve()
    X = ufl.SpatialCoordinate(mesh)
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx + ufl.inner(X, X) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, [pyadjoint.Control(s), pyadjoint.Control(m)])
    hs = _dilation(S)
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
    displaced.x.array[:] = _dilation(S).x.array * 0.05
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
    h = _dilation(S)

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
        petsc_options_prefix="test_shape_two_moves_",
    )
    uh = problem.solve()
    J_first = ufl.inner(uh, uh) * ufl.dx

    second = dxa.Function(S)
    second.x.array[:] = 0.05 * _dilation(S).x.array
    dxa.move(mesh, second)
    J = dxa.assemble_scalar(J_first + ufl.inner(X, X) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    h = _dilation(S)
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
    assert np.isclose(directional, fd, rtol=1e-4), f"adjoint {directional} vs finite difference {fd}"


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

    saddle_point = _LU | {"pc_factor_mat_solver_type": "mumps"}

    def solve_stokes(displacement_values: np.ndarray | None = None):
        pyadjoint.get_working_tape().clear_tape()
        mesh = _unit_square(6)
        S = dxa.geometry_function_space(mesh)
        s = dxa.Function(S)
        if displacement_values is not None:
            s.x.array[:] = displacement_values
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
            petsc_options=saddle_point,
            adjoint_petsc_options=saddle_point,
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
    assert np.isclose(directional, fd, rtol=1e-4), f"adjoint {directional} vs finite difference {fd}"


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


def test_expression_mixing_a_coefficient_and_the_coordinates_is_refused():
    """UFL cannot differentiate a coefficient w.r.t. the coordinates in physical space.

    That term would otherwise be dropped silently, so it has to raise.
    """
    mesh, S, s, _ = _shape_setup(4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    W = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    coefficient = dxa.Function(W)
    coefficient.x.array[:] = 2.0
    X = ufl.SpatialCoordinate(mesh)
    uh = dxa.interpolate(coefficient * ufl.sin(ufl.pi * X[0]), V)
    J = dxa.assemble_scalar(ufl.inner(uh, uh) * ufl.dx)

    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))
    with pytest.raises(NotImplementedError, match="coefficient with respect to the coordinates"):
        Jhat.derivative()


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
