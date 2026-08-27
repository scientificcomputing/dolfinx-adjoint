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
    dolfinx_adjoint.assign(0.0, uh)
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
    # Insert this diagnostic section into your demo right after J and m are defined:

    # 1. Generate a non-zero base control point m_pert
    m_pert = [dolfinx_adjoint.Function(V, name=f"pert_ctrl_{t_val}") for t_val in ctrls.keys()]
    for c in m_pert:
        c.x.array[:] = np.random.uniform(0.1, 1.0, size=c.x.array.shape)

    # 2. Define random directions h
    h = [pyadjoint.Control(dolfinx_adjoint.Function(V)) for _ in m]
    for hi in h:
        hi.control.x.array[:] = np.random.uniform(-0.1, 0.1, size=hi.control.x.array.shape)

    print("\n=== 1. Taylor Test at NON-ZERO Control Point ===")
    min_val_pert = pyadjoint.taylor_test(rf, m_pert, h)
    print(f"Convergence rate at perturbed point: {min_val_pert:.4f}")

    print("\n=== 2. Direct Finite Difference Gradient Verification ===")
    eps = 1e-6

    # Compute Adjoint Directional Derivative at m_pert
    rf(m_pert)
    grad_adj = rf.derivative()
    adj_dir_deriv = sum(g._ad_dot(hi) for g, hi in zip(grad_adj, h, strict=True))

    # Forward Perturbation J(m + eps*h)
    m_plus = [dolfinx_adjoint.Function(V) for _ in m_pert]
    for mp, m_p, hi in zip(m_plus, m_pert, h, strict=True):
        mp.x.array[:] = m_p.x.array[:] + eps * hi.control.x.array[:]
    J_plus = float(rf(m_plus))

    # Backward Perturbation J(m - eps*h)
    m_minus = [dolfinx_adjoint.Function(V) for _ in m_pert]
    for mm, m_p, hi in zip(m_minus, m_pert, h, strict=True):
        mm.x.array[:] = m_p.x.array[:] - eps * hi.control.x.array[:]
    J_minus = float(rf(m_minus))

    # Central Finite Difference
    fd_dir_deriv = (J_plus - J_minus) / (2 * eps)

    print(f"Adjoint Directional Derivative:    {adj_dir_deriv:.10e}")
    print(f"Finite Difference Directional Dev: {fd_dir_deriv:.10e}")
    rel_diff = abs(adj_dir_deriv - fd_dir_deriv) / (abs(fd_dir_deriv) + 1e-15)
    print(f"Relative Mismatch:                 {rel_diff:.4e}")

    assert rel_diff < 1e-4, f"Adjoint gradient mismatches finite differences! Relative error: {rel_diff}"

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
