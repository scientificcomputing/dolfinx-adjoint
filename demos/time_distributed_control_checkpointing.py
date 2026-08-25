# # Time-distributed control with checkpointing
#
# This is the [time-distributed control](./time_distributed_control.py) demo again, with
# checkpointing switched on.
#
# Taping a time-dependent model keeps every intermediate state alive, because the adjoint
# sweep needs each of them on the way back. For a long simulation that is the thing that
# exhausts memory first. Checkpointing trades that memory for repeated work: only some states
# are kept, and the rest are recomputed from the nearest stored one when the adjoint asks for
# them. A schedule decides which to keep and when to recompute. The schedules come from
# `checkpoint_schedules` {cite}`tdcc-Dolci2024`; for how step-based checkpointing combines
# with high-level algorithmic differentiation, see {cite}`tdcc-Maddison2024`.
#
# Everything here comes from `pyadjoint` and `checkpoint_schedules` directly. The only thing
# `dolfinx_adjoint` adds is `enable_disk_checkpointing`, used at the end.

from collections import OrderedDict

from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import ufl
from checkpoint_schedules import Revolve

import dolfinx_adjoint

# ## Enabling a schedule
#
# A schedule has to be enabled on an empty tape, before anything is recorded, so that every
# timestep is treated the same way. {py:class}`Revolve <checkpoint_schedules.hrevolve.Revolve>`
# keeps at most `snapshots` states in memory and recomputes whatever else the adjoint needs.

num_steps = 10
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # type: ignore[arg-type]

nu = dolfinx.fem.Constant(mesh, np.float64(1e-5))
dt = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.1))

x = ufl.SpatialCoordinate(mesh)

petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "ksp_error_if_not_converged": True,
}


def solve_heat(schedule=None, disk=False):
    """Tape the heat equation over `num_steps` timesteps, optionally under a schedule.

    Returns the reduced functional and the controls, one control per timestep.
    """
    tape = pyadjoint.Tape()
    pyadjoint.set_working_tape(tape)
    # Both of these configure how the tape stores state, so both have to happen before
    # anything is recorded on it.
    if disk:
        dolfinx_adjoint.enable_disk_checkpointing()
    if schedule is not None:
        tape.enable_checkpointing(schedule)

    t = dolfinx_adjoint.Constant(mesh, dolfinx.default_scalar_type(0.0))
    t.name = "time"
    d = 16 * x[0] * (x[0] - 1) * x[1] * (x[1] - 1) * ufl.sin(ufl.pi * t)

    ctrls = OrderedDict()
    for i in range(num_steps):
        ctrls[i] = dolfinx_adjoint.Function(V, name=f"control_{i}")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = dolfinx_adjoint.Function(V, name="source")
    u_0 = dolfinx_adjoint.Function(V, name="solution")

    F = ((u - u_0) / dt * v + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) - f * v) * ufl.dx
    a, L = ufl.system(F)

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    exterior_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, exterior_facets)
    bc = dolfinx.fem.dirichletbc(0.0, exterior_dofs, V)

    problem = dolfinx_adjoint.LinearProblem(
        a,
        L,
        u=u_0,
        bcs=[bc],
        petsc_options=petsc_options,
        adjoint_petsc_options=petsc_options,
    )

    j = 0.5 * float(dt) * dolfinx_adjoint.assemble_scalar((u_0 - d) ** 2 * ufl.dx)

    # `iter(...)` because timestepper calls next() on what it is given, and the default
    # progress bar passes it straight through. Setting tape.progress_bar works too.
    for i in tape.timestepper(iter(range(num_steps))):
        t_val = float(dt) * (i + 1)
        dolfinx_adjoint.assign(t_val, t)
        dolfinx_adjoint.assign(ctrls[i], f)

        problem.solve()

        weight = 0.5 if i == num_steps - 1 else 1.0
        j += weight * float(dt) * dolfinx_adjoint.assemble_scalar((u_0 - d) ** 2 * ufl.dx)

    controls = list(ctrls.values())
    rf = pyadjoint.ReducedFunctional(j, [pyadjoint.Control(c) for c in controls])
    return rf, controls


# ## Checkpointing does not change the answer
#
# A schedule only changes when state is stored and recomputed. The functional and its gradient
# are unchanged, which is worth checking explicitly the first time you enable one.

rf_plain, controls_plain = solve_heat()
J_plain = rf_plain(controls_plain)
grad_plain = [np.copy(g.x.array) for g in rf_plain.derivative()]

rf_ckpt, controls_ckpt = solve_heat(Revolve(num_steps, 3))
J_ckpt = rf_ckpt(controls_ckpt)
grad_ckpt = [np.copy(g.x.array) for g in rf_ckpt.derivative()]

assert np.isclose(J_plain, J_ckpt)
for a, e in zip(grad_ckpt, grad_plain, strict=True):
    np.testing.assert_allclose(a, e)

if mesh.comm.rank == 0:
    print(f"J without checkpointing: {J_plain:.12g}")
    print(f"J with Revolve({num_steps}, 3):  {J_ckpt:.12g}")

# ## A Taylor test through the schedule
#
# The check above shows the two gradients agree with each other. It does not show that either
# is correct, since both could be wrong in the same way. A Taylor test checks that directly: the
# first-order remainder must converge at second order.

directions = []
for k in range(num_steps):
    h = dolfinx_adjoint.Function(V, name=f"direction_{k}")
    # Interpolated rather than random: the direction has to be the same on every process, and
    # per-process random numbers are not.
    h.interpolate(lambda x, k=k: np.sin((k + 1) * np.pi * x[0]) * np.cos(np.pi * x[1]))
    directions.append(h)

rf_ckpt, controls_ckpt = solve_heat(Revolve(num_steps, 3))
rate = pyadjoint.taylor_test(rf_ckpt, controls_ckpt, directions)
assert rate > 1.9

# ## Storing checkpoints on disk
#
# {py:class}`Revolve <checkpoint_schedules.hrevolve.Revolve>` keeps its checkpoints in memory.
# When even those do not fit, a schedule such as
# {py:class}`SingleDiskStorageSchedule <checkpoint_schedules.basic_schedules.SingleDiskStorageSchedule>`
# can put them on disk instead, and {py:func}`dolfinx_adjoint.enable_disk_checkpointing`
# provides the storage.
#
# These are *snapshot* checkpoints: they hold just this process's values for the function, and
# assume the mesh and its partition are unchanged, so they are valid only within the run that
# wrote them. They are deleted automatically. For a checkpoint that outlives the run, or that
# can be read back on a different number of processes, use
# [io4dolfinx](https://github.com/scientificcomputing/io4dolfinx) instead.
#
# Like the schedule, it must be enabled before anything is recorded on the tape.

from checkpoint_schedules import SingleDiskStorageSchedule  # noqa: E402

rf_disk, controls_disk = solve_heat(SingleDiskStorageSchedule(), disk=True)
J_disk = rf_disk(controls_disk)
grad_disk = [np.copy(g.x.array) for g in rf_disk.derivative()]

assert np.isclose(J_plain, J_disk)
for a, e in zip(grad_disk, grad_plain, strict=True):
    np.testing.assert_allclose(a, e)

if mesh.comm.rank == 0:
    print(f"J with checkpoints on disk:   {J_disk:.12g}")
    print("Gradients agree to machine precision in all three cases.")

# Turning it off again deletes the checkpoint files. Every process must call it, because
# closing a shared checkpoint file is collective.

dolfinx_adjoint.checkpointing.disable_disk_checkpointing()

# ## References
# ```{bibliography}
# :filter: cited
# :labelprefix:
# :keyprefix: tdcc-
# ```
