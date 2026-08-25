"""Checkpointing of time-dependent adjoint computations.

The forward model is a heat equation advanced over a number of tape timesteps, with one
control per timestep. Every test compares a checkpointed run against the same run with
checkpointing disabled: a checkpoint schedule only changes *when* forward state is stored
and recomputed, never the value of the derivative.
"""

from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import h5py
import numpy as np
import pyadjoint
import pytest
import ufl
from checkpoint_schedules import Revolve, SingleDiskStorageSchedule
from pyadjoint.checkpointing import CheckpointError

import dolfinx_adjoint

_PETSC_OPTIONS = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "ksp_error_if_not_converged": True,
}


@pytest.fixture(autouse=True)
def isolated_tape():
    """Keep these tests from leaking tape state into the rest of the suite.

    Two things would otherwise escape. The working tape: pyadjoint's Tape.clear_tape() resets
    the checkpoint manager but leaves `_eagerly_checkpoint_outputs` and `latest_checkpoint` set,
    so a tape that has once been checkpointed keeps checkpointing outputs eagerly even after
    being cleared. And the PETSc options database: LinearProblem writes its options there under
    a fixed default prefix and does not remove them, so a later solver constructed without
    explicit options silently inherits whatever these tests set.
    """
    previous_tape = pyadjoint.get_working_tape()
    previous_options = dict(PETSc.Options().getAll())
    try:
        yield
    finally:
        dolfinx_adjoint.checkpointing.disable_disk_checkpointing()
        pyadjoint.set_working_tape(previous_tape)
        options = PETSc.Options()
        for key in set(options.getAll()) - set(previous_options):
            options.delValue(key)


def _perturbation_directions(V, n):
    """Perturbation directions for a Taylor test.

    Built by interpolating analytic expressions rather than from random numbers: the
    directions must agree across processes, and per-rank random values do not.
    """
    directions = []
    for k in range(n):
        h = dolfinx_adjoint.Function(V, name=f"direction_{k}")
        h.interpolate(lambda x, k=k: np.sin((k + 1) * np.pi * x[0]) * np.cos(np.pi * x[1]))
        directions.append(h)
    return directions


def _tape_heat_equation(n_steps, schedule=None, disk=False, use_mpio=None):
    """Tape a heat equation with one control per tape timestep.

    Args:
        n_steps: Number of tape timesteps to advance.
        schedule: A ``checkpoint_schedules`` schedule, or None to disable checkpointing.
        disk: Whether to store checkpoints on disk.
        use_mpio: Passed through to ``enable_disk_checkpointing``, selecting the file layout.

    Returns:
        A tuple of the reduced functional, the controls, and perturbation directions.
    """
    tape = pyadjoint.Tape()
    pyadjoint.set_working_tape(tape)
    # Both of these must happen before anything is recorded on this tape.
    if disk:
        dolfinx_adjoint.enable_disk_checkpointing(use_mpio=use_mpio)
    if schedule is not None:
        tape.enable_checkpointing(schedule)

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # type: ignore[arg-type]

    dt = 0.1
    nu = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0e-2))

    controls = []
    for i in range(n_steps):
        c = dolfinx_adjoint.Function(V, name=f"control_{i}")
        c.interpolate(lambda x, i=i: 0.5 + 0.1 * (i + 1) * x[0])
        controls.append(c)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = dolfinx_adjoint.Function(V, name="source")
    uh = dolfinx_adjoint.Function(V, name="solution")

    F = ((u - uh) / dt * v + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) - f * v) * ufl.dx
    a, L = ufl.system(F)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(0.0, boundary_dofs, V)

    problem = dolfinx_adjoint.LinearProblem(
        a,
        L,
        u=uh,
        bcs=[bc],
        petsc_options=_PETSC_OPTIONS,
        adjoint_petsc_options=_PETSC_OPTIONS,
    )

    J = dolfinx_adjoint.assemble_scalar(dt * uh**2 * ufl.dx)
    # NOTE: `iter(...)` is required. Tape.timestepper calls next() on what it is given,
    # so a bare range() raises TypeError even though pyadjoint's own docstring shows one.
    for i in tape.timestepper(iter(range(n_steps))):
        dolfinx_adjoint.assign(controls[i], f)
        problem.solve()
        J = J + dolfinx_adjoint.assemble_scalar(dt * uh**2 * ufl.dx)

    rf = pyadjoint.ReducedFunctional(J, [pyadjoint.Control(c) for c in controls])
    return rf, controls, _perturbation_directions(V, n_steps)


def _gradient(rf, controls):
    rf(controls)
    return [np.copy(g.x.array) for g in rf.derivative()]


@pytest.mark.parametrize("n_steps, snapshots", [(6, 2), (10, 3)])
def test_gradient_matches_uncheckpointed(n_steps, snapshots):
    """A checkpoint schedule does not change the gradient."""
    rf_plain, controls_plain, _ = _tape_heat_equation(n_steps)
    expected = _gradient(rf_plain, controls_plain)

    rf_ckpt, controls_ckpt, _ = _tape_heat_equation(n_steps, Revolve(n_steps, snapshots))
    actual = _gradient(rf_ckpt, controls_ckpt)

    assert len(actual) == len(expected)
    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(a, e, rtol=1e-12, atol=1e-14, err_msg=f"control {i}")


@pytest.mark.parametrize("n_steps, snapshots", [(6, 2), (10, 3)])
def test_taylor_test_under_checkpointing(n_steps, snapshots):
    """The checkpointed gradient is the actual derivative, not merely a reproducible one."""
    rf, controls, directions = _tape_heat_equation(n_steps, Revolve(n_steps, snapshots))
    rate = pyadjoint.taylor_test(rf, controls, directions)
    assert rate > 1.95


def _tape_snes_heat_equation(n_steps, schedule=None, solution_dependent_diffusivity=False):
    """Tape a heat equation solved as a residual problem via SNES.

    Unlike the linear model this cannot step in place: the unknown and the previous state
    must be distinct functions, so the state update is an explicit assignment.

    Args:
        n_steps: Number of tape timesteps to advance.
        schedule: A ``checkpoint_schedules`` schedule, or None to disable checkpointing.
        solution_dependent_diffusivity: If True the residual is genuinely nonlinear in the
            unknown, which puts the unknown into the Jacobian and hence into the block's own
            dependencies. See ``test_solution_dependent_jacobian_is_unsupported``.
    """
    tape = pyadjoint.Tape()
    pyadjoint.set_working_tape(tape)
    if schedule is not None:
        tape.enable_checkpointing(schedule)

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # type: ignore[arg-type]
    dt = 0.1

    controls = []
    for i in range(n_steps):
        c = dolfinx_adjoint.Function(V, name=f"control_{i}")
        c.interpolate(lambda x, i=i: 0.5 + 0.1 * (i + 1) * x[0])
        controls.append(c)

    v = ufl.TestFunction(V)
    f = dolfinx_adjoint.Function(V, name="source")
    uh = dolfinx_adjoint.Function(V, name="solution")
    u_prev = dolfinx_adjoint.Function(V, name="previous")

    nu = (1 + uh**2) if solution_dependent_diffusivity else 1.0
    F = ((uh - u_prev) / dt * v + nu * ufl.inner(ufl.grad(uh), ufl.grad(v)) - f * v) * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(0.0, boundary_dofs, V)

    snes_options = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "none",
        "snes_error_if_not_converged": True,
        "snes_atol": 1e-14,
        "snes_rtol": 1e-14,
    }
    snes_options.update(_PETSC_OPTIONS)
    problem = dolfinx_adjoint.NonlinearProblem(
        F,
        uh,
        bcs=[bc],
        petsc_options=snes_options,
        adjoint_petsc_options=_PETSC_OPTIONS,
    )

    J = dolfinx_adjoint.assemble_scalar(dt * uh**2 * ufl.dx)
    for i in tape.timestepper(iter(range(n_steps))):
        dolfinx_adjoint.assign(controls[i], f)
        problem.solve()
        dolfinx_adjoint.assign(uh, u_prev)
        J = J + dolfinx_adjoint.assemble_scalar(dt * uh**2 * ufl.dx)

    rf = pyadjoint.ReducedFunctional(J, [pyadjoint.Control(c) for c in controls])
    return rf, controls, _perturbation_directions(V, n_steps)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pre-existing NonlinearProblemBlock defect, unrelated to checkpointing: a residual "
        "problem advanced over several timesteps gets a wrong adjoint. The unknown is a "
        "coefficient of the residual and so is registered as one of the block's own "
        "dependencies; once its incoming value is itself control-dependent (which is what a "
        "time loop creates) the adjoint contribution for it is computed against a residual in "
        "which that value no longer appears, and ufl.adjoint raises IndexError on the "
        "resulting argument-less form. Suppressing the error is not a fix: the gradient is "
        "then silently wrong (observed Taylor rate -0.41 rather than 2). Checkpointing is not "
        "involved -- this test does not enable a schedule. Until it is fixed, NonlinearProblem "
        "cannot be used in a time loop and so cannot be covered by the checkpointing tests "
        "above."
    ),
)
def test_snes_time_loop_gradient_is_correct():
    """Records that NonlinearProblem cannot yet be advanced over timesteps."""
    rf, controls, directions = _tape_snes_heat_equation(4)
    assert pyadjoint.taylor_test(rf, controls, directions) > 1.95


@pytest.mark.parametrize(
    "use_mpio",
    [
        None,
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(not h5py.get_config().mpi, reason="h5py is not built against MPI"),
        ),
    ],
)
def test_disk_gradient_matches_uncheckpointed(use_mpio):
    """Storing checkpoints on disk does not change the gradient, in either file layout.

    `use_mpio=None` picks the layout automatically, and resolves to the per-process one on a
    single process, so `True` is passed explicitly to reach the shared MPI-IO file as well.
    """
    n_steps = 6
    rf_plain, controls_plain, _ = _tape_heat_equation(n_steps)
    expected = _gradient(rf_plain, controls_plain)

    rf_disk, controls_disk, _ = _tape_heat_equation(n_steps, SingleDiskStorageSchedule(), disk=True, use_mpio=use_mpio)
    actual = _gradient(rf_disk, controls_disk)
    dolfinx_adjoint.checkpointing.disable_disk_checkpointing()

    for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
        np.testing.assert_allclose(a, e, rtol=1e-12, atol=1e-14, err_msg=f"control {i}")


def test_disk_taylor_test():
    """The gradient from disk-stored checkpoints is the actual derivative."""
    n_steps = 6
    rf, controls, directions = _tape_heat_equation(n_steps, SingleDiskStorageSchedule(), disk=True)
    rate = pyadjoint.taylor_test(rf, controls, directions)
    dolfinx_adjoint.checkpointing.disable_disk_checkpointing()
    assert rate > 1.95


def test_disk_schedule_without_enabling_is_refused():
    """A disk-using schedule with no disk backend configured fails loudly, and says why."""
    with pytest.raises(CheckpointError, match="enable_disk_checkpointing"):
        _tape_heat_equation(4, SingleDiskStorageSchedule())
