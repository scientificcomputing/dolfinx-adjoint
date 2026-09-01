# # Optimal Control of the Poisson equation
# *Section author: Jørgen S. Dokken ([dokken@simula.no](mailto:dokken@simula.no))*.
#
# Original implementation in [dolfin-adjoint](https://github.com/dolfin-adjoint/dolfin-adjoint) was by Simon W. Funke.

# This demo solves the *mother problem* of PDE-constrained optimization: the optimal control of the Possion equation.
# Physically, this problem can be interpreted as finding the best heating/cooling of a cooktop to
# achieve a desired temperature profile.

# This example introduces the basics of how to solve optimization problems with DOLFINx-adjoint.

# ## Problem definition
# Mathematically, the goal is to minimize the following tracking type functional:
#
# $$
# \min_{f \in Q} J(u) = \frac{1}{2} \int_{\Omega} (u - d)^2 ~\mathrm{d}x
# + \frac{\alpha}{2}\int_{\Omega} f^2~\mathrm{d} x
# $$
#
# subject to the Poisson equation with Dirichlet boundary conditions:
#
# $$
# \begin{align}
# - \kappa \Delta u &= f  && \text{in } \Omega, \\
# u &= 0 && \text{on } \partial\Omega, \\
# a &\leq f \leq b &&
# \end{align}
# $$
#
# where $\Omega$ is the domain of interest, $u: \Omega \mapsto \mathbb{R}$ is the unknown temperature,
# $\kappa\in\mathbb{R}$ is the thermal diffusivity, $f: \Omega \mapsto \mathbb{R}$ is the
# unknown control function acting as a source term,
# $d:\Omega\mapsto \mathbb{R}$ is the desired temperature profile, and $\alpha\in[0,\infty)$
# is a Tikhonov regularization parameter, and $a,b\in\mathbb{R}$ are the lower and upper bounds
# on the control function $f$.
# Note that $f(x)>0$ corresponds to heating, while $f(x)<0$ corresponds to cooling.

# It can be shown that this problem is well-posed and has a unique solution, see for instance
# Section 1.5 {cite}`Ulbrich2009` or {cite}`troltzsch2010optimal`.

# ## Implementation
# We start by import the necessary modules for this demo, which includes `mpi4py`, `dolfinx` and `dolfinx_adjoint`.

# +

from mpi4py import MPI

import dolfinx

# + [markdown]
# Next we import [Moola](https://github.com/funsim/moola/), which is a Python package
# containing a collection of optimization solvers specifically designed for PDE-constrained optimization problems.
# +
import moola
import numpy as np
import pandas
import pyadjoint
import pyvista
import ufl
from moola.adaptors import DolfinxPrimalVector  # noqa: E402

import dolfinx_adjoint

# -

# We configure Pyvista for rendering

# + tags=["hide-input"]
pyvista.set_jupyter_backend("html")

# -

# + [markdown]
# Next we create a regular mesh of a unit square, which will be our domain $\Omega$.
# Some optimization algorithms suffer from bad performance when the mesh is non-uniform
# (i.e. when the mesh is partially refined).
# To demonstrate that Moola does not have this issue, we will refine our mesh in the center of the domain.
# We use the DOLFINx refine function to do this.

# +
n = 16
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, n, n)


def refinement_region(x, tol=1e-14):
    """Define a conditional to determine the refinement region.

    Args:
        x: Coordinates of mesh nodes, shape (num_nodes, 3)
        tol: Tolerance for being within the region.
    """
    clause_x = abs(x[0] - 0.5) < 0.25 + tol
    clause_y = abs(x[1] - 0.5) < 0.25 + tol
    return clause_x & clause_y


mesh.topology.create_connectivity(1, mesh.topology.dim)
edges_to_refine = dolfinx.mesh.locate_entities(mesh, 1, refinement_region)
refined_mesh_data = dolfinx.mesh.refine(mesh, edges_to_refine)
refined_mesh = refined_mesh_data[0]
tdim = refined_mesh.topology.dim
del mesh
# -

# Next we use Pyvista to plot the mesh

# +

grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(refined_mesh))
plotter = pyvista.Plotter()
plotter.add_mesh(grid, show_edges=True, color="lightgrey")
plotter.view_xy(plotter)
if pyvista.OFF_SCREEN:
    plotter.screenshot("poisson_mother_mesh.png")
else:
    plotter.show()
# -

# Then we define the discrete function spaces $V$ and $Q$ for the state and control variable, respectively

V = dolfinx.fem.functionspace(refined_mesh, ("Lagrange", 1))  # type: ignore[arg-type]
Q = dolfinx.fem.functionspace(refined_mesh, ("Discontinuous Lagrange", 0))  # type: ignore[arg-type]

# The optimization algorithm will use the value of the control function $f$ as an initial guess for the optimization.
# A zero intial guess seems to be too simple: For example the L-BFGS algorithm will find the optimial
# control with just two iterations. To make it more interesting, we choose a non-zero initial guess.

f = dolfinx_adjoint.Function(Q, name="Control")
f.interpolate(lambda x: x[0] + x[1])  # Set intial guess

# ```{note}
# As opposed to standard DOLFINx code, we use `dolfinx_adjoint.Function` to create the control function.
# This is so that we can track it throughout the program on the computational tape.
# ```
# We also create a state variable that we will store the solution to the Poisson equation in.

uh = dolfinx_adjoint.Function(V, name="State")

# Next, we define the variational formulation of the Poisson equation.

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
kappa = dolfinx.fem.Constant(refined_mesh, np.dtype(dolfinx.default_scalar_type).type(1.0))  # Thermal diffusivity
F = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
a, L = ufl.system(F)

# We create the Dirichlet BC

refined_mesh.topology.create_connectivity(tdim - 1, tdim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(refined_mesh.topology)
exterior_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, exterior_facets)
zero = dolfinx.fem.Constant(refined_mesh, np.dtype(dolfinx.default_scalar_type).type(0.0))
bc = dolfinx.fem.dirichletbc(zero, exterior_dofs, V)

# Next, we define a `dolfinx_adjoint.LinearProblem` instance, which overloads
# the `dolfinx.fem.petsc.LinearProblem` class.
#
# ```{note}
# When creating the :py:class:`dolfinx_adjoint.LinearProblem`, we can specify the solver options that
# are passed on to the underlying PETSc Krylov subspace solver.
# This is also the place to pass in solver options for the first and second order adjoint equations
# and the tangent linear model (TLM) equation.
# ```
#

petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
problem = dolfinx_adjoint.LinearProblem(
    a,
    L,
    u=uh,
    bcs=[bc],
    petsc_options=petsc_options,
    adjoint_petsc_options=petsc_options,
    tlm_petsc_options=petsc_options,  # type: ignore
)
problem.solve()

# ```{note}
# Note that we can pass in solver options for the adjoint equation via the keyword argument
# `adjoint_petsc_options`. As we are solving a linear, symmetric problem, i.e. a self-adjoint problem,
# we use the same options for both the forward and adjoint problems.
# ```

# Before we can start the optimization, we need to specity the control variable and the functional of interest.
# For this example we use $d(x,y)=\frac{1}{2\pi^2} \sin(\pi x)\sin( \pi y)$ as the desired temperature profile

x, y = ufl.SpatialCoordinate(refined_mesh)
d = 1 / (2 * ufl.pi**2) * ufl.sin(ufl.pi * x) * ufl.sin(ufl.pi * y)

# The functional is written out in `ufl` and assembled with `dolfinx_adjoint.assemble_scalar`

alpha = dolfinx.fem.Constant(
    refined_mesh, np.dtype(dolfinx.default_scalar_type).type(1.0e-6)
)  # Tikhonov regularization parameter
alpha.name = "alpha"  # type: ignore
J_symbolic = 0.5 * ufl.inner(uh - d, uh - d) * ufl.dx + 0.5 * alpha * ufl.inner(f, f) * ufl.dx
J = dolfinx_adjoint.assemble_scalar(J_symbolic)

# The next step is to formulate the so-called reduced optimization problem.
# The idea is that the solution $u$ can be considered as a functin of $f$:
# given a value of $f$, we can solve the Poisson equation and obtain the associated solution $u_h$.
# Bu denoting this solution function as $u(f)$ we can rewrite the original optimization problem as a reduced problem:

# $$
# \min_{f \in Q} \hat J(u(f), f) = \frac{1}{2} \int_{\Omega} (u(f) - d)^2 ~\mathrm{d}x
# + \frac{\alpha}{2}\int_{\Omega} f^2~\mathrm{d} x
# $$

# Note that no PDE-constraint is required anymore, since it is implicitly contained in the solution function.

# *dolfinx-adjoint* can automatically reduce the optimization problem by using `pyadjoint.ReducedFunctional`.
# This object solves the forward PDE using the `pyadjoint.Tape` each time the functional is evaluated.
# Additionally, it derives and solves the adjoint equation each time the function gradient should be evaluated.

control = pyadjoint.Control(f)
Jhat = pyadjoint.ReducedFunctional(J, control)

# Now that all ingredients are in place, we can perform the optimization.
# For this, we employ the `moola.MoolaOptimizationProblem` to generate a problem that
# is compatible with the Moola framework.

optimization_problem = pyadjoint.MoolaOptimizationProblem(Jhat)
f_moola = DolfinxPrimalVector(f)

# Then, we wrap the control function into a Moola object, and create a `moola.BFGS``
# solver for solving the optimisation problem

# + tags=["scroll-output"]
optimization_options = {"jtol": 0, "gtol": 1e-9, "Hinit": "default", "maxiter": 100, "mem_lim": 10, "rjtol": 0}
solver = moola.BFGS(optimization_problem, f_moola, options=optimization_options)
solution = solver.solve()
# -

# Next, we update the control function with the optimal value found by Moola and solve the forward problem to
# get the optimal state variable.

f_opt = solution["control"].data
f.x.array[:] = f_opt.x.array  # f_opt.x.array.copy()
problem.solve(annotate=False)

# ## Error analysis
# For our desired temperature profile, there exist an analytic solution on the following form
#
# $$
# \begin{align*}
# f_{analytic}(x,y) &= \frac{1}{1 + \alpha 4\pi^4} \sin(\pi x) \sin(\pi y)\\
# u_{analytic}(x,y) &= \frac{1}{2\pi^2} f_{analytic}(x,y)
# \end{align*}
# $$

# We use these, and compute the
f_analytic = 1 / (1 + alpha * 4 * pow(ufl.pi, 4)) * ufl.sin(ufl.pi * x) * ufl.sin(ufl.pi * y)
u_analytic = 1 / (2 * ufl.pi**2) * f_analytic


err_u = dolfinx_adjoint.error_norm(u_analytic, uh, norm_type="L2", annotate=False)
err_f = dolfinx_adjoint.error_norm(f_analytic, f, norm_type="L2", annotate=False)
print(f"Error in state variable: {err_u:.3e}")
print(f"Error in control variable: {err_f:.3e}")


# We visualize the results using Pyvista.

# + tags=["hide-input"]

# Plotting space for source
plotting_space = dolfinx.fem.functionspace(refined_mesh, ("Discontinuous Lagrange", 1))  # type: ignore[arg-type]
u_plot = dolfinx.fem.Function(plotting_space)
u_plot.interpolate(uh)
f_plot = dolfinx.fem.Function(plotting_space)
f_plot.interpolate(f)

# Interpolate the analytic solutions to the plotting space
u_expr = dolfinx.fem.Expression(u_analytic, plotting_space.element.interpolation_points)
u_ex = dolfinx.fem.Function(plotting_space)
u_ex.interpolate(u_expr)
f_expr = dolfinx.fem.Expression(f_analytic, plotting_space.element.interpolation_points)
f_ex = dolfinx.fem.Function(plotting_space)
f_ex.interpolate(f_expr)

# Attach data to the plotting grid
plotting_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(plotting_space))  # type: ignore[arg-type]
plotting_grid.point_data["u_optimal"] = u_plot.x.array
plotting_grid.point_data["u_exact"] = u_ex.x.array
plotting_grid.point_data["f_optimal"] = f_plot.x.array
plotting_grid.point_data["f_exact"] = f_ex.x.array

# Create plotter for all variables
plotter = pyvista.Plotter(shape=(2, 2))
plotter.subplot(0, 0)
scale_factor = 25
warped_u = plotting_grid.warp_by_scalar("u_optimal", factor=scale_factor)
plotter.add_mesh(warped_u, color="lightgrey", scalars="u_optimal")
plotter.subplot(0, 1)
warped_u_ex = plotting_grid.warp_by_scalar("u_exact", factor=scale_factor)
plotter.add_mesh(warped_u_ex, color="lightgrey", scalars="u_exact")
plotter.subplot(1, 0)
scale_factor = 1
warped_f = plotting_grid.warp_by_scalar("f_optimal", factor=scale_factor)
plotter.add_mesh(warped_f, scalars="f_optimal", show_edges=True)
plotter.subplot(1, 1)
warped_f_ex = plotting_grid.warp_by_scalar("f_exact", factor=scale_factor)
plotter.add_mesh(warped_f_ex, scalars="f_exact", show_edges=True)
plotter.link_views((0, 1))
plotter.link_views((2, 3))
if pyvista.OFF_SCREEN:
    plotter.screenshot("poisson_mother.png")
else:
    plotter.show()
# -

# ## Convergence analysis (mesh independence)
# It is highly desirable that the optimisation algorithm achieve mesh independence: i.e.,
# that the required number of optimisation iterations is independent of the mesh resolution.
# Achieving mesh independence requires paying careful attention to the inner product structure
# of the function space in which the solution is sought.

# We therefore perform a mesh convergence analysis with the BFGS algorithm and the Newton-CG algorithm.
# By expanding the dropdown below, you will see the same implementation as above, but wrapped in a function
# that can be called with different mesh resolutions.


# + tags=["hide-input"]
def solve_optimal_problem(N: int, use_newton: bool = False) -> dict[str, float | int]:
    """Solve the optimal control problem for a given mesh resolution.

    Args:
        N: Number of elements in each direction of the mesh.
        use_newton: Whether to use Newton-CG instead of BFGS for the optimization.
    Returns:
        A dictionary containing the results of the optimization problem.
    """
    # Reset tape
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)
    mesh.topology.create_connectivity(1, mesh.topology.dim)
    edges_to_refine = dolfinx.mesh.locate_entities(mesh, 1, refinement_region)
    refined_mesh_data = dolfinx.mesh.refine(mesh, edges_to_refine)
    refined_mesh = refined_mesh_data[0]
    refined_mesh = mesh  # Use the original mesh for testing
    tdim = refined_mesh.topology.dim
    del mesh

    V = dolfinx.fem.functionspace(refined_mesh, ("Lagrange", 1))  # type: ignore[arg-type]
    Q = dolfinx.fem.functionspace(refined_mesh, ("Discontinuous Lagrange", 0))  # type: ignore[arg-type]

    f = dolfinx_adjoint.Function(Q, name="Control")
    f.interpolate(lambda x: x[0] + np.sin(2 * x[1]))  # Set intial guess

    uh = dolfinx_adjoint.Function(V, name="State")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    kappa = dolfinx.fem.Constant(refined_mesh, np.dtype(dolfinx.default_scalar_type).type(1.0))  # Thermal diffusivity
    F = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
    a, L = ufl.system(F)

    refined_mesh.topology.create_connectivity(tdim - 1, tdim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(refined_mesh.topology)
    exterior_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, exterior_facets)
    zero = dolfinx.fem.Constant(refined_mesh, np.dtype(dolfinx.default_scalar_type).type(0.0))
    bc = dolfinx.fem.dirichletbc(zero, exterior_dofs, V)

    problem = dolfinx_adjoint.LinearProblem(
        a, L, u=uh, bcs=[bc], petsc_options=petsc_options, adjoint_petsc_options=petsc_options
    )
    problem.solve()

    x, y = ufl.SpatialCoordinate(refined_mesh)
    d = 1 / (2 * ufl.pi**2) * ufl.sin(ufl.pi * x) * ufl.sin(ufl.pi * y)

    alpha = dolfinx.fem.Constant(
        refined_mesh, np.dtype(dolfinx.default_scalar_type).type(1.0e-6)
    )  # Tikhonov regularization parameter
    J_symbolic = 0.5 * ufl.inner(uh - d, uh - d) * ufl.dx + 0.5 * alpha * ufl.inner(f, f) * ufl.dx
    J = dolfinx_adjoint.assemble_scalar(J_symbolic)

    control = pyadjoint.Control(f)
    Jhat = pyadjoint.ReducedFunctional(J, control)

    optimization_problem = pyadjoint.MoolaOptimizationProblem(Jhat)
    f_moola = DolfinxPrimalVector(f)
    if use_newton:
        solver = moola.NewtonCG(
            optimization_problem,
            f_moola,
            options={
                "gtol": 1e-9,
                "maxiter": 20,
                "display": 0,
                "ncg_hesstol": 0,
            },
        )
    else:
        opts = optimization_options.copy()
        opts["display"] = 0  # Turn down verbosity
        solver = moola.BFGS(optimization_problem, f_moola, options=opts)
    sol = solver.solve()
    f_opt = sol["control"].data
    f.x.array[:] = f_opt.x.array.copy()
    problem.solve(annotate=False)
    f_analytic = 1 / (1 + alpha * 4 * pow(ufl.pi, 4)) * ufl.sin(ufl.pi * x) * ufl.sin(ufl.pi * y)
    u_analytic = 1 / (2 * ufl.pi**2) * f_analytic
    err_u = dolfinx_adjoint.error_norm(u_analytic, uh, norm_type="L2", annotate=False)
    err_f = dolfinx_adjoint.error_norm(f_analytic, f, norm_type="L2", annotate=False)

    num_cells = refined_mesh.topology.index_map(tdim).size_local
    local_h = np.max(refined_mesh.h(tdim, np.arange(num_cells, dtype=np.int32)))
    global_h = refined_mesh.comm.allreduce(local_h, op=MPI.MAX)
    return {
        "Number of iterations": sol["iteration"],
        "h": global_h,
        "L2_u": err_u,
        "L2_f": err_f,
        "J": sol["objective"],
        "|dJ/df|": sol["grad_norm"],
    }


# -

# ### BFGS results
# We run the BFGS algorithm for different mesh resolutions and collect the results.

results_bfgs = []
for N in [16, 32, 64, 128]:
    results_bfgs.append(solve_optimal_problem(N, use_newton=False))

bfgs_results = pandas.DataFrame(results_bfgs)
bfgs_results

# We observe that the number of iterations is independent of the mesh resolution, and that the errors in the
# state and control goes down with a rate of 1.

results_newton_cg = []
for N in [16, 32, 64, 128]:
    results_newton_cg.append(solve_optimal_problem(N, use_newton=True))

# ### Newton-CG results
# We can also check the convergence of an algorithm using Hessian information, like Newton-CG.
# If we set the tolerance of the Conjugate Gradient solver to zero, the algorithm will converge in one iteration.
# However, if we set it to a small value, we should observe a low number of iterations, that is also independent
# of the mesh resolution.

newton_results = pandas.DataFrame(results_newton_cg)
newton_results

# + [markdown]
# ## References
# ```{bibliography}
# :filter: cited and ({"demos/poisson_mother"} >= docnames)
# ```
# -

# + tags=["hide-input"]
assert newton_results["Number of iterations"].max() == 3
# -
