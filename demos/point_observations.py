# # Inverting for a source term from point measurements
# *Section author: Henrik Finsberg ([henriknf@simula.no](mailto:henriknf@simula.no))*.

# The [Poisson mother problem](./poisson_mother.py) demo matches the computed state against a
# desired profile known *everywhere* in the domain. Real data is rarely like that: sensors sit
# at a handful of locations, wells are drilled at specific coordinates, and medical images
# sample the domain on their own grid rather than on the finite element mesh.

# This demo solves the same mother problem, but against data that only exists at **points**.
# The tool for that is `dolfinx_adjoint.PointObservation`, which evaluates a finite element
# function at a given set of coordinates,
#
# $$
# d = B u, \qquad B_{ij} = \phi_j(x_i),
# $$
#
# so that row $i$ picks out the value of $u$ at the point $x_i$. Together with a PDE solve,
# $m \mapsto B u(m)$ is what is known as the *parameter-to-observable map*.

# ## Problem definition
# We recover an unknown source $f$ from noisy measurements of the state at a set of sensor
# locations. The state solves
#
# $$
# \begin{align}
# -\Delta u &= f && \text{in } \Omega, \\
# u &= 0 && \text{on } \partial\Omega,
# \end{align}
# $$
#
# and we minimize
#
# $$
# \min_{f} J(f) = \frac{1}{2} \lVert B u(f) - d \rVert^2
#              + \frac{\alpha}{2} \int_\Omega f^2 ~\mathrm{d}x.
# $$
#
# The first term is a sum over the sensors rather than an integral over $\Omega$. The data is
# never interpolated onto the mesh, so nothing is invented at locations where nothing was
# measured -- which matters, because interpolating sparse data onto a fine mesh manufactures
# exactly the information the inverse problem is supposed to extract.

# ## Implementation

# +
from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import ufl

import dolfinx_adjoint

# -

# We work on the unit square with both the state and the source in $P_1$.

# +
domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 32, 32)
V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))

domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(domain.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facets)


def solve_forward(f: dolfinx_adjoint.Function, prefix: str) -> dolfinx_adjoint.Function:
    """Solve the Poisson problem with source ``f`` and homogeneous Dirichlet conditions."""
    u = dolfinx_adjoint.Function(V, name="state")
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
    L = ufl.inner(f, test) * ufl.dx

    u_bc = dolfinx_adjoint.Function(V, name="u_bc")
    bc = dolfinx_adjoint.dirichletbc(u_bc, boundary_dofs)
    dolfinx_adjoint.LinearProblem(a, L, u=u, bcs=[bc], petsc_options_prefix=prefix).solve()
    return u


# -

# ## Generating synthetic data
#
# The source we are trying to recover is a pair of Gaussian bumps.


# +
def true_source(x):
    return 5.0 * np.exp(-((x[0] - 0.35) ** 2 + (x[1] - 0.65) ** 2) / 0.05) + 3.0 * np.exp(
        -((x[0] - 0.7) ** 2 + (x[1] - 0.3) ** 2) / 0.03
    )


f_true = dolfinx_adjoint.Function(V, name="f_true")
f_true.interpolate(true_source)
# -

# The sensors sit on a slightly jittered grid. Creating the observation operator is a matter of
# handing it the function space and the coordinates.

# +
rng = np.random.default_rng(2024)
axis = np.linspace(0.08, 0.92, 12)
sensors = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
sensors += rng.uniform(-0.01, 0.01, size=sensors.shape)

B = dolfinx_adjoint.PointObservation(V, sensors)
print(f"Placed {B.num_found} of {B.num_points} sensors inside the mesh")
# -

# `B.evaluate(u)` gives the value of `u` at each sensor. We solve the forward problem with the
# true source and add measurement noise to get our synthetic data.

# +
NOISE_STD = 2e-3

with pyadjoint.stop_annotating():
    clean = B.evaluate(solve_forward(f_true, "demo_truth_"))

observations = clean + rng.normal(0.0, NOISE_STD, size=clean.shape)
print(f"Observations: range [{observations.min():.4f}, {observations.max():.4f}], noise std {NOISE_STD}")
# -

# ## The inverse problem
#
# Now we forget the true source, start from zero, and try to recover it from the measurements.
# `point_observation_misfit` records the misfit on the tape, so it composes with the PDE solve
# exactly like `assemble_scalar` does and the two terms can simply be added.

# +
ALPHA = 1e-6  # Tikhonov regularization parameter

pyadjoint.get_working_tape().clear_tape()

f = dolfinx_adjoint.Function(V, name="f")
f.x.array[:] = 0.0

u = solve_forward(f, "demo_inverse_")

misfit = dolfinx_adjoint.point_observation_misfit(u, B, observations)
regularization = dolfinx_adjoint.assemble_scalar(0.5 * ALPHA * ufl.inner(f, f) * ufl.dx)
J = misfit + regularization

Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(f))
# -

# Before optimizing, we check that the gradient of the combined functional is correct, by
# perturbing the source in some direction and confirming that the error left over after the
# first-order Taylor expansion shrinks quadratically.

# +
h = dolfinx_adjoint.Function(V)
h.interpolate(lambda x: 10.0 * np.sin(4 * np.pi * x[0]) * np.cos(3 * np.pi * x[1]))

rate = pyadjoint.taylor_test(Jhat, f, h)
print(f"Taylor convergence rate: {rate:.3f} (expected ~2)")
assert rate > 1.9
# -

# Now minimize. Recovering a source from measurements of the (much smoother) state is a
# classically ill-conditioned problem, so L-BFGS-B needs a fair number of iterations here.

# +
Jhat(f)  # re-evaluate at the initial guess, undoing the Taylor test's perturbations
f_opt = pyadjoint.minimize(Jhat, method="L-BFGS-B", options={"maxiter": 500})
# -

# ## Results
#
# Two things are worth measuring: how well the recovered source reproduces the measurements,
# and how close it is to the source we started from.

# +
with pyadjoint.stop_annotating():
    predicted = B.evaluate(solve_forward(f_opt, "demo_check_"))
    l2_error = dolfinx_adjoint.error_norm(f_true, f_opt, norm_type="L2")
    l2_truth = np.sqrt(dolfinx_adjoint.assemble_scalar(ufl.inner(f_true, f_true) * ufl.dx))

rms = np.sqrt(np.mean((predicted - observations) ** 2))
print(f"Sensor residual RMS:    {rms:.5f} (injected noise std was {NOISE_STD})")
print(f"Relative L2 error in f: {l2_error / l2_truth:.4f}")
# -

# The residual settles *at* the noise floor rather than below it. That is the outcome to hope
# for: driving it to zero would mean fitting the noise, and the regularization term is what
# stops that from happening.
#
# The recovered source is a smoothed version of the truth, with lower peaks than the real
# bumps. That is not a defect of the method but a property of the data: a few hundred
# measurements of a smoothing operator's output do not determine the fine structure of the
# source, and the regularization supplies what the data cannot.

# + tags=["hide-input"]
try:
    import pyvista

except ImportError:
    print("Install pyvista to visualize the result")

else:
    # pyvista.set_jupyter_backend("html")
    cells, types, geometry = dolfinx.plot.vtk_mesh(V)
    sensor_cloud = pyvista.PolyData(np.column_stack([sensors, np.zeros(len(sensors))]))

    plotter = pyvista.Plotter(shape=(1, 2), window_size=(900, 400))
    for column, (field, title) in enumerate([(f_true, "True source"), (f_opt, "Recovered source")]):
        grid = pyvista.UnstructuredGrid(cells, types, geometry)
        grid.point_data["f"] = field.x.array.real
        grid.set_active_scalars("f")
        plotter.subplot(0, column)
        plotter.add_text(title, font_size=10)
        plotter.add_mesh(grid, show_edges=False, clim=[0.0, 5.0])
        plotter.add_mesh(sensor_cloud, color="black", point_size=5, render_points_as_spheres=True)
        plotter.view_xy()
    if pyvista.OFF_SCREEN:
        plotter.screenshot("point_observations.png")
    else:
        plotter.show()

# -

# ## Going further
#
# * **Sensors outside the domain.** `B.found` flags any points that fall outside the mesh, and
#   those are left out of the misfit rather than being treated as measurements of zero.
# * **Unreliable sensors.** `point_observation_misfit(..., weights=...)` reweights individual
#   measurements; a zero weight drops one entirely.
# * **Noisier data.** `point_observation_misfit(..., noise_variance=...)` divides the misfit by
#   $\sigma^2$, which is what makes it a negative log-likelihood when the regularization term is
#   a genuine prior.
# * **Time-dependent data.** Build `B` once and call `point_observation_misfit` once per
#   observation time, summing the results -- the operator only depends on the sensor positions.
