# # Optimal Control with Non-Linear Expressions and Non-Matching Grids
#
# This demo minimizes the tracking functional:
#
# $$ \min_{u \in Q} J(y, u) = \frac{1}{2} \int_{\Omega_s} (y - d)^2 ~dx + \frac{\alpha}{2} \int_{\Omega_c} u^2 ~dx $$
#
# Subject to the state equation on a fine grid (\Omega_s) and a control on a coarse grid (\Omega_c):
#
# $$ - \Delta y = I_{\Omega_c \to \Omega_s}(u + 0.1 u^3) \quad \text{in } \Omega_s $$

from mpi4py import MPI

import dolfinx
import moola
import numpy as np
import pyadjoint
import ufl
from dolfinx.mesh import create_unit_square
from moola.adaptors import DolfinxPrimalVector

import dolfinx_adjoint
from dolfinx_adjoint import interpolate, interpolate_nonmatching

# ## 1. Define Non-Matching Domains
# Coarse grid for the heater control

mesh_control = create_unit_square(MPI.COMM_WORLD, 8, 8)

# Fine grid for the physical state

mesh_state = create_unit_square(MPI.COMM_WORLD, 32, 32)

Q = dolfinx.fem.functionspace(mesh_control, ("Lagrange", 1))
V = dolfinx.fem.functionspace(mesh_state, ("Lagrange", 1))

# ## 2. Setup Control Variable

u_c = dolfinx_adjoint.Function(Q, name="Control")
u_c.interpolate(lambda x: np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]))

# ## 3. Apply Non-Linear UFL Expression (`ExprInterpolationBlock`)
# The heater output scales non-linearly with the control input

f_expr = u_c + 0.1 * u_c**3

# Interpolate the UFL expression onto the control space.
# PyAdjoint will tape this via {py:class}`ExprInterpolationBlock`.

f_c = interpolate(f_expr, Q, ad_block_tag="nonlinear_heater_eval")

# ## 4. Cross-Mesh Transfer (NonmatchingInterpolationBlock)
# Transfer the evaluated heater power to the fine physics grid.
# PyAdjoint will tape this via {py:class}`NonmatchingInterpolationBlock`.

f_s = interpolate_nonmatching(f_c, V, ad_block_tag="grid_transfer")

# ## 5. Solve the PDE

y = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

F = ufl.inner(ufl.grad(y), ufl.grad(v)) * ufl.dx - f_s * v * ufl.dx
a, L = ufl.system(F)

mesh_state.topology.create_connectivity(mesh_state.topology.dim - 1, mesh_state.topology.dim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh_state.topology)
exterior_dofs = dolfinx.fem.locate_dofs_topological(V, mesh_state.topology.dim - 1, exterior_facets)
zero = dolfinx.fem.Constant(mesh_state, 0.0)
bc = dolfinx.fem.dirichletbc(zero, exterior_dofs, V)

yh = dolfinx_adjoint.Function(V, name="State")
petsc_opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
problem = dolfinx_adjoint.LinearProblem(
    a,
    L,
    u=yh,
    bcs=[bc],
    petsc_options=petsc_opts,
    adjoint_petsc_options=petsc_opts,
    tlm_petsc_options=petsc_opts,
)
problem.solve()

# ## 6. Define Objective and Optimize

x = ufl.SpatialCoordinate(mesh_state)
d = ufl.sin(2 * ufl.pi * x[0]) * ufl.sin(2 * ufl.pi * x[1])

alpha = dolfinx.fem.Constant(mesh_control, 1e-4)

# Note the integration domains: state tracking on $\Omega_s$, regularization on $\Omega_c$

J = dolfinx_adjoint.assemble_scalar(0.5 * ufl.inner(yh - d, yh - d) * ufl.dx(domain=mesh_state))
J += dolfinx_adjoint.assemble_scalar(0.5 * alpha * ufl.inner(u_c, u_c) * ufl.dx(domain=mesh_control))

# Configure PyAdjoint Reduced Functional

control = pyadjoint.Control(u_c)
Jhat = pyadjoint.ReducedFunctional(J, control)

# ## 7. Taylor Tests (Gradient and Hessian)

print("\n--- Running Taylor Tests ---")

# Define a distinct perturbation direction for the Taylor expansion

h = dolfinx_adjoint.Function(Q, name="Perturbation")
h.interpolate(lambda x: 10 * np.cos(np.pi * x[0]) * np.cos(np.pi * x[1]))

print("\n1st-order Taylor test (Testing Gradient exactness):")

# Test Gradient (Expect convergence rate ~ 2.0)

rate_grad = pyadjoint.taylor_test(Jhat, u_c, h)

print("\n2nd-order Taylor test (Testing Hessian exactness):")

# To test the Hessian, we evaluate the action of the first and second derivatives
# in the direction of the perturbation 'h'

Jhat(u_c)
dJdm = Jhat.derivative()._ad_dot(h)
dHddu = Jhat.hessian(h)._ad_dot(h)

# Test Hessian (Expect convergence rate ~ 3.0)

rate_hess = pyadjoint.taylor_test(Jhat, u_c, h, dJdm=dJdm, Hm=dHddu)

print(f"\nGradient Taylor Test Rate: {rate_grad:.4f}")
print(f"Hessian Taylor Test Rate: {rate_hess:.4f}")

# Solve with Moola

opt_problem = pyadjoint.MoolaOptimizationProblem(Jhat)
u_moola = DolfinxPrimalVector(u_c)
solver = moola.NewtonCG(opt_problem, u_moola, options={"gtol": 1e-6, "maxiter": 50, "display": 3})
sol = solver.solve()

print(f"Optimization completed in {sol['iteration']} iterations.")
