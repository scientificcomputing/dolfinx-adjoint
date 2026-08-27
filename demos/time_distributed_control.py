# # Time-distributed control
# Based on example from https://dolfin-adjoint.github.io/dolfin-adjoint/documentation/time-distributed-control/time-distributed-control.html

from collections import OrderedDict

from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import ufl

import dolfinx_adjoint

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
x = ufl.SpatialCoordinate(mesh)

nu = dolfinx.fem.Constant(mesh, np.float64(1e-5))
nu.name = "nu"  # type: ignore

t = dolfinx_adjoint.Constant(mesh, dolfinx.default_scalar_type(0.0))  # type: ignore
t.name = "time"
d = 16 * x[0] * (x[0] - 1) * x[1] * (x[1] - 1) * ufl.sin(ufl.pi * t)

dt = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.1))  # type: ignore
T = 1

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # type: ignore[arg-type]
ctrls = OrderedDict()
t_val = float(dt)
while t_val <= T:
    ctrls[t_val] = dolfinx_adjoint.Function(V, name=f"control_{t_val}")
    t_val += float(dt)


def solve_heat(ctrls):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = dolfinx_adjoint.Function(V, name="source")

    u_prev = dolfinx_adjoint.Function(V, name="u_prev")
    uh = dolfinx_adjoint.Function(V, name="solution")

    F = ((u - u_prev) / dt * v + nu * ufl.inner(ufl.grad(u), ufl.grad(v)) - f * v) * ufl.dx
    a, L = ufl.system(F)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    exterior_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, exterior_facets)

    bc = dolfinx.fem.dirichletbc(0.0, exterior_dofs, V)

    j = 0.5 * float(dt) * dolfinx_adjoint.assemble_scalar((uh - d) ** 2 * ufl.dx)

    t_val = float(dt)
    problem = dolfinx_adjoint.LinearProblem(
        a,
        L,
        u=uh,
        bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
        adjoint_petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
        tlm_petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
    )
    dolfinx_adjoint.assign(t_val, t)
    while t_val <= T:
        # Update source term from control array
        dolfinx_adjoint.assign(ctrls[t_val], f)

        # Update data function
        dolfinx_adjoint.assign(uh, u_prev)

        # Solve PDE
        problem.solve()

        # Implement a trapezoidal rule
        if t_val > T - float(dt):
            weight = 0.5
        else:
            weight = 1
        j += weight * float(dt) * dolfinx_adjoint.assemble_scalar((uh - d) ** 2 * ufl.dx)
        # Update time
        t_val += float(dt)
        dolfinx_adjoint.assign(t_val, t)

    return uh, d, j


u, d, j = solve_heat(ctrls)

alpha = dolfinx.fem.Constant(mesh, np.float64(1.0e-1))
regularisation = (
    alpha
    / 2
    * sum([1 / dt * (fb - fa) ** 2 * ufl.dx for fb, fa in zip(list(ctrls.values())[1:], list(ctrls.values())[:-1])])
)


J = j + dolfinx_adjoint.assemble_scalar(regularisation)
m = [pyadjoint.Control(c) for c in ctrls.values()]

rf = pyadjoint.ReducedFunctional(J, m)

# Check accuracy of gradient and Hessian using Taylor test
with pyadjoint.stop_annotating():
    h = [dolfinx_adjoint.Function(ci.function_space) for ci in m]
    for hi in h:
        hi.x.array[:] = np.random.random(hi.x.array.shape)

    # Prove the perturbation is mathematically exact
    min_val = pyadjoint.taylor_test(rf, list(ctrls.values()), h, dJdm=0)
    assert np.isclose(min_val, 1.0, rtol=5e-2, atol=1e-5), f"Expected convergence rate close to 1.0, got {min_val}"

    # Prove the gradient is mathematically exact
    min_val = pyadjoint.taylor_test(rf, list(ctrls.values()), h)
    assert np.isclose(min_val, 2.0, rtol=1e-2, atol=1e-2), f"Expected convergence rate close to 2.0, got {min_val}"

    # Prove the Hessian is mathematically exact
    rf(list(ctrls.values()))
    dJdm = sum(drfi._ad_dot(hi) for drfi, hi in zip(rf.derivative(), h, strict=True))
    H = rf.hessian(h)
    dHddu = sum(Hi._ad_dot(hi) for Hi, hi in zip(H, h, strict=True))
    pyadjoint.taylor_test(rf, list(ctrls.values()), h, dJdm=dJdm, Hm=dHddu)
    rf(list(ctrls.values()))


tape = pyadjoint.get_working_tape()
tape.visualise_dot("test.dot")

opt_ctrls = pyadjoint.minimize(
    rf,
    method="BFGS",
    options={"maxiter": 100, "disp": True},
)

out_ctrl = dolfinx.fem.Function(V, name="optimal_control")
with dolfinx.io.VTXWriter(mesh.comm, "opt_ctrl.bp", [out_ctrl]) as vtx:
    for t_val, c in zip(ctrls.keys(), opt_ctrls):
        out_ctrl.x.array[:] = c.x.array[:]
        vtx.write(t_val)


assert np.isclose(np.linalg.norm(opt_ctrls[0].x.array), 4.930056079391683)
assert np.isclose(np.linalg.norm(opt_ctrls[-1].x.array), 2.8756312728703963)
