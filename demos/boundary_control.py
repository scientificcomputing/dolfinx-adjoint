# Dirichlet BC control of the Stokes equations
# ============================================
#
# This demo is based on the example ``stokes-bc-control.py`` from the `dolfin-adjoint` source tree,
# authored by Simon W. Funke and André Massing.
#
# The updated version has been made by Jørgen S. Dokken <dokken@simula.no>.
#
# This example demonstrates how to compute the sensitivity with respect to the Dirichlet
# boundary conditions in pyadjoint.
#
# ## Problem definition
#
# Consider the problem of minimising the compliance
#
# $$
# \min_{g, u, p} \ \frac{1}{2}\int_{\Omega} \nabla u \cdot \nabla u~\mathrm{d}x
# + \frac{\alpha}{2} \int_{\partial \Omega_{\mathrm{circle}}} g^2~\mathrm{d}s
# $$
#
# subject to the Stokes equations
#
# $$
# \begin{align}
# -\nu \Delta u + \nabla p &= 0  \qquad \mathrm{in} \ \Omega \\
# \mathrm{div }\  u &= 0  \qquad \mathrm{in} \ \Omega  \\
# \end{align}
# $$
#
# with Dirichlet boundary conditions
#
# $$
# \begin{align}
# u &= g  \qquad \mathrm{on} \ \partial \Omega_{\mathrm{cirlce}} \\
# u &= f  \qquad \mathrm{on} \ \partial \Omega_{\mathrm{in}} \\
# u &= 0  \qquad \mathrm{on} \ \partial \Omega_{\mathrm{walls}} \\
# p &= 0  \qquad \mathrm{on} \ \partial \Omega_{\mathrm{out}}
# \end{align}
# $$
#
# where :math:`\Omega` is the domain of interest,
# :math:`u:\Omega \to \mathbb R^2` is the unknown velocity,
# :math:`p:\Omega \to \mathbb R` is the unknown pressure, :math:`\nu`
# is the viscosity, :math:`\alpha` is the regularisation parameter,
# :math:`f` denotes the value for the Dirichlet inflow boundary
# condition, and :math:`g` is the control variable that specifies the
# Dirichlet boundary condition on the circle.
#
# Physically, this setup corresponds to minimising the loss of flow
# energy into heat by actively controlling the in/outflow at the
# circle boundary. To avoid excessive control solutions, non-zero
# control values are penalised via the regularisation term.
#
# ## Implementation
#
# First, we implement the various modules we will use in this demo

import itertools

from mpi4py import MPI

import basix.ufl
import dolfinx
import gmsh
import numpy as np
import pyadjoint
import pyvista
import ufl

from dolfinx_adjoint import Constant, Function, LinearProblem, assemble_scalar, dirichletbc

try:
    from dolfinx.io import gmsh as gmshio
except ImportError:
    from dolfinx.io import gmshio  # type: ignore[attr-defined, no-redef]

# The geometry is the one the original demo meshes with `mshr` in its own `make-mesh.py`:
# a 30x10 rectangle with a circle of radius 2.5 centred at (10, 5) removed. Keeping those
# dimensions matters, because the inflow profile below is written for them.
#
# The gmsh commands themselves follow
# [DOLFINx-tutorial Navier-Stokes benchmark](https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code2.html#mesh-generation),
# which explains the mesh generation process in more detail. The mesh is graded towards the
# circle, since that is where the control acts and the flow varies most.

# +
gmsh.initialize()

L = 30.0
H = 10.0
c_x, c_y = 10.0, 5.0
r = 2.5
gdim = 2
mesh_comm = MPI.COMM_WORLD
model_rank = 0
inlet_marker, outlet_marker, wall_marker, obstacle_marker = 2, 3, 4, 5
inflow, outflow, walls, obstacle = [], [], [], []
fluid_marker = 1
res_min = r / 3

if mesh_comm.rank == model_rank:
    rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
    _obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)
    fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, _obstacle)])
    gmsh.model.occ.synchronize()
    volumes = gmsh.model.getEntities(dim=gdim)
    assert len(volumes) == 1
    gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], fluid_marker)
    gmsh.model.setPhysicalName(volumes[0][0], fluid_marker, "Fluid")
    boundaries = gmsh.model.getBoundary(volumes, oriented=False)
    for boundary in boundaries:
        center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
        if np.allclose(center_of_mass, [0, H / 2, 0]):
            inflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L, H / 2, 0]):
            outflow.append(boundary[1])
        elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
            walls.append(boundary[1])
        else:
            obstacle.append(boundary[1])
    gmsh.model.addPhysicalGroup(1, walls, wall_marker)
    gmsh.model.setPhysicalName(1, wall_marker, "Walls")
    gmsh.model.addPhysicalGroup(1, inflow, inlet_marker)
    gmsh.model.setPhysicalName(1, inlet_marker, "Inlet")
    gmsh.model.addPhysicalGroup(1, outflow, outlet_marker)
    gmsh.model.setPhysicalName(1, outlet_marker, "Outlet")
    gmsh.model.addPhysicalGroup(1, obstacle, obstacle_marker)
    gmsh.model.setPhysicalName(1, obstacle_marker, "Obstacle")
    distance_field = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
    threshold_field = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
    gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
    gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
    gmsh.option.setNumber("Mesh.Algorithm", 8)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    gmsh.option.setNumber("Mesh.RecombineAll", 1)
    gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(2)
    gmsh.model.mesh.optimize("Netgen")

mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
mesh = mesh_data.mesh
assert mesh_data.facet_tags is not None
ft = mesh_data.facet_tags
ft.name = "Facet markers"
# -

# ## Visualising the mesh and the boundary markers
#
# Before solving anything, we look at what we have meshed. The first panel shows the graded
# mesh; the second draws only the tagged facets, coloured by the marker each carries, over a
# faint wireframe of the domain. The control acts on the `Obstacle` facets alone, so it is
# worth confirming those are the ones that were tagged.

# + tags=["hide-input"]
pyvista.set_jupyter_backend("html")

tdim = mesh.topology.dim
fdim = tdim - 1
mesh.topology.create_connectivity(fdim, tdim)

# Not `vtk_mesh(mesh, tdim)`: this mesh is second order, and pyvista silently tessellates
# the higher-order quadrilaterals into triangles before rendering, so a wireframe drawn from
# it shows diagonals that are not element edges.
# The cost is that curved edges are drawn straight, which is invisible at this resolution.
V_linear = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
mesh_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(V_linear))
# One VTK cell per tagged facet, carrying its marker as cell data.
facet_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh, fdim, ft.indices))
facet_grid.cell_data["Marker"] = ft.values

# Float keys: pyvista annotates a scalar bar by value, and types those values as floats.
marker_names = {
    float(inlet_marker): "Inlet",
    float(outlet_marker): "Outlet",
    float(wall_marker): "Walls",
    float(obstacle_marker): "Obstacle (control)",
}

plotter = pyvista.Plotter(shape=(2, 1), window_size=[1100, 800])
plotter.subplot(0, 0)
plotter.add_text("Mesh", font_size=10)
plotter.add_mesh(mesh_grid, show_edges=True, color="white", edge_color="dimgrey", line_width=1)
# `camera.tight` fills the viewport with the domain; `view_xy` alone leaves wide margins
# around a long, thin geometry like this one.
plotter.camera.tight(padding=0.05, view="xy", adjust_render_window=False)
plotter.subplot(1, 0)
plotter.add_text("Boundary markers", font_size=10)
plotter.add_mesh(mesh_grid, style="wireframe", color="lightgrey", opacity=0.4)
plotter.add_mesh(
    facet_grid,
    scalars="Marker",
    line_width=8,
    cmap="viridis",
    annotations=marker_names,
    scalar_bar_args={
        "title": "",
        "n_labels": 0,
        "vertical": True,
        "position_x": 0.85,
        "position_y": 0.1,
        "width": 0.05,
        "height": 0.8,
    },
)
plotter.camera.tight(padding=0.05, view="xy", adjust_render_window=False)
if pyvista.OFF_SCREEN:
    plotter.screenshot("boundary_control_mesh.png")
else:
    plotter.show()
# -

# ## Defining the function spaces and boundary conditions
# Then, we define the discrete function spaces. A Taylor-Hood
# finite-element pair is a suitable choice for the Stokes equations.
# The control function is the Dirichlet boundary value on the velocity
# field and is hence be a function on the velocity space

el_u = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(gdim,))
el_p = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
V = dolfinx.fem.functionspace(mesh, el_u)
Q = dolfinx.fem.functionspace(mesh, el_p)
W = ufl.MixedFunctionSpace(V, Q)

u, p = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)

uh = Function(V, name="Velocity")
ph = Function(Q, name="Pressure")

g = Function(V, name="Control")

nu = Constant(mesh, 1)

# Our functional requires the computation of a boundary integral
# over :math:`\partial \Omega_{\mathrm{circle}}`.  Therefore, we need
# to create a measure for this integral, which will be accessible as
# :py:data:`ds(2)` in the definition of the functional. In addition, we
# define our strong Dirichlet boundary conditions.

ds = ufl.ds(subdomain_data=ft)

# Define boundary conditions
x = ufl.SpatialCoordinate(mesh)
u_inflow = ufl.as_vector((x[1] * (10 - x[1]) / 25, 0))
noslip = Constant(mesh, (0, 0))

# Locate the degrees of freedom for the various boundary facets.

noslip_dofs = dolfinx.fem.locate_dofs_topological(V, ft.dim, ft.find(wall_marker))
noslip = dirichletbc(noslip, noslip_dofs, V)
inflow_dofs = dolfinx.fem.locate_dofs_topological(V, ft.dim, ft.find(inlet_marker))
inflow = dirichletbc(u_inflow, inflow_dofs, V)
circle_dofs = dolfinx.fem.locate_dofs_topological(V, ft.dim, ft.find(obstacle_marker))
circle = dirichletbc(g, circle_dofs, V)
bcs = [inflow, noslip, circle]

# We derive the standard weak formulation of
# the Stokes problem: Find :math:`u, p` such that for all test
# functions :math:`v, q`
#
# $$
# a(u,p; v,q) = L(u,p;v,q)
# $$
#
# with
#
# $$
# \begin{align}
# a(u,p;v,q) =&\ \nu \left<\nabla (u), \nabla (v)\right>_\Omega \\
#             & - \left<p, \mathrm{div} v \right>_{\Omega}
#             - \left<q, \mathrm{div} u \right>_{\Omega}
#             \\
# L(u,p;v,q) =&\ 0
# \end{align}
# $$

# In code, this becomes:

a = (
    nu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    - ufl.inner(p, ufl.div(v)) * ufl.dx
    - ufl.inner(q, ufl.div(u)) * ufl.dx
)
# Named `L_forms`, not `L`: the channel length above is already called `L`, and rebinding
# it here would leave the plotting code below unable to refer to the geometry.
L_forms = [ufl.ZeroBaseForm((v,)), ufl.ZeroBaseForm((q,))]

# Next we assemble and solve the system once to record it with
# :py:mod:`dolin-adjoint`.

direct_solver_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
problem = LinearProblem(
    ufl.extract_blocks(a),
    L_forms,
    u=[uh, ph],
    bcs=bcs,
    petsc_options=direct_solver_options,
    adjoint_petsc_options=direct_solver_options,
    tlm_petsc_options=direct_solver_options,
)
problem.solve()

# Next we define the functional of interest :math:`J`, the
# optimisation parameter :math:`g`, and create the reduced
# functional.

alpha = Constant(mesh, 10)

J = assemble_scalar(
    1.0 / 2 * ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx + alpha / 2 * ufl.inner(g, g) * ds(obstacle_marker)
)
m = pyadjoint.Control(g)

# The iteration monitor below wants the gradient norm as well as the functional value, but
# SciPy's callback is handed neither. `derivative_cb_post` fires every time the reduced
# functional computes a gradient -- which L-BFGS-B does once per iterate anyway -- so
# stashing the norm here costs no extra adjoint solve. Note pyadjoint assigns this callback's
# return value back over the derivatives, so it has to hand the list straight back.
gradient_norm = [float("nan")]


def record_gradient(_functional_value, derivatives, _control_values):
    (gradient,) = derivatives
    owned = gradient.x.index_map.size_local * gradient.x.block_size
    local_max = np.abs(gradient.x.array[:owned]).max(initial=0.0)
    gradient_norm[0] = mesh.comm.allreduce(local_max, op=MPI.MAX)
    return derivatives


Jhat = pyadjoint.ReducedFunctional(J, m, derivative_cb_post=record_gradient)

# Now, everything is set up to run the optimisation and to plot the
# results. By default, :py:func:`minimize` uses the L-BFGS-B
# algorithm.

# SciPy deprecated L-BFGS-B's own `disp`/`iprint` options in 1.15 and they print nothing as
# of 1.17 (removal is scheduled for 1.18), so `options={"disp": True}` is silently ignored.
# The supported replacement is a callback, which SciPy calls once per accepted iteration;
# {py:func}`pyadjoint.minimize` forwards any extra keyword straight to
# {py:func}`scipy.optimize.minimize`.
# Naming the parameter `intermediate_result` is what selects SciPy's newer callback API --
# the one handed the functional value, rather than only the control vector.
iteration = itertools.count()


def log_iteration(intermediate_result):
    """Log the iteration number, functional value, and gradient norm.

    The gradient reported is the last one computed, which is the one at this iterate.
    `max|dJ/dg|` is the unconstrained analogue of the `|proj g|` column the original
    dolfin-adjoint demo printed.
    """
    print(
        f"  iteration {next(iteration):3d}    J = {intermediate_result.fun:.8e}    max|dJ/dg| = {gradient_norm[0]:.3e}",
        flush=True,
    )


g_opt = pyadjoint.minimize(Jhat, method="L-BFGS-B", callback=log_iteration)

# ## Results
#
# `minimize` returns the optimal control, but it leaves the tape at whatever point the line
# search last probed. Re-evaluating the reduced functional at `g_opt` replays the forward
# problem there, which both reports the optimal functional value and leaves `uh`/`ph` holding
# the corresponding state -- the velocity and pressure we want to look at.

J_opt = Jhat(g_opt)
print(f"J(g_opt) = {J_opt:.6e}")

# We visualise the three fields together: the optimised boundary control on the circle, and
# the velocity and pressure it induces.
#
# The control and the velocity are vector fields, so we draw them as glyphs, following
# [the tutorial's Navier-Stokes demo](https://jsdokken.com/dolfinx-tutorial/chapter2/ns_code1.html#visualization-of-vectors).
# `dolfinx.plot.vtk_mesh(V)` places one point per degree of freedom of the (P2) velocity
# space, so the glyph grid is built on the space rather than on the mesh.

# + tags=["hide-input"]


def _glyph_cloud(fn, name, target_length, only_nonzero=False):
    """A pyvista point cloud of ``fn``'s degrees of freedom, ready for ``glyph``.

    A point cloud rather than an {py:class}`pyvista.UnstructuredGrid`: this mesh
    is built from recombined, second-order quadrilaterals, and VTK's cell-based
    filters (``clip_box`` in particular) do not handle those higher-order cells.
    Instead, they silently return nothing. Glyphs only need the dof coordinates
    and the values there, so the cells are not needed at all.

    ``target_length`` is how long the longest arrow should be, in mesh units. The
    maximum is reduced over all ranks so every process draws the same scale --
    otherwise the two halves of a parallel plot would not be comparable.
    """
    coordinates = fn.function_space.tabulate_dof_coordinates()
    values = fn.x.array.real.reshape(-1, gdim)
    magnitude = np.linalg.norm(values, axis=1)
    if only_nonzero:
        # The control is identically zero away from the circle, so dropping the zeros leaves
        # exactly the facets it acts on -- and lets the camera frame them by itself.
        keep = magnitude > 0.0
        coordinates, values, magnitude = coordinates[keep], values[keep], magnitude[keep]

    padded = np.zeros((coordinates.shape[0], 3), dtype=np.float64)
    padded[:, :gdim] = values
    cloud = pyvista.PolyData(coordinates)
    cloud["vectors"] = padded
    cloud[name] = magnitude

    largest = mesh.comm.allreduce(magnitude.max() if magnitude.size else 0.0, op=MPI.MAX)
    factor = target_length / largest if largest > 0.0 else 1.0
    return cloud.glyph(orient="vectors", scale=name, factor=factor)


control_glyphs = _glyph_cloud(g_opt, "|g|", 0.45 * r, only_nonzero=True)
velocity_glyphs = _glyph_cloud(uh, "|u|", 0.10 * H)

pressure_grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(Q))
pressure_grid.point_data["Pressure"] = ph.x.array.real

# Lift the pressure out of the plane to read it as a surface rather than a colour map. The
# factor is chosen so the tallest feature is a fixed fraction of the channel height, which
# keeps the relief legible whatever the pressure scale happens to be; the maximum is reduced
# over all ranks so every process warps by the same amount.
pressure_peak = mesh.comm.allreduce(np.abs(ph.x.array.real).max() if ph.x.array.size else 0.0, op=MPI.MAX)
warped_pressure = pressure_grid.warp_by_scalar(
    "Pressure", factor=(0.4 * H) / pressure_peak if pressure_peak > 0.0 else 1.0
)

# The control lives on the circle alone, so it gets its own square figure: in a panel shaped
# like this 3:1 domain, a zoom tight enough to show the arrows would crop the geometry.
# `parallel_scale` is the viewport half-height in mesh units, so it sets the zoom directly.
plotter = pyvista.Plotter(window_size=[700, 700])
plotter.add_text("Optimised boundary control", font_size=10)
plotter.add_mesh(mesh_grid, style="wireframe", color="lightgrey", opacity=0.6)
plotter.add_mesh(control_glyphs, scalars="|g|", cmap="viridis")
plotter.enable_parallel_projection()  # type: ignore[call-arg]
plotter.camera.up = (0.0, 1.0, 0.0)
plotter.camera.focal_point = (c_x, c_y, 0.0)
plotter.camera.position = (c_x, c_y, 1.0)
plotter.camera.parallel_scale = 1.6 * r
if pyvista.OFF_SCREEN:
    plotter.screenshot("boundary_control_control.png")
else:
    plotter.show()

# The state it induces spans the whole channel, so velocity and pressure share a wide figure.
plotter = pyvista.Plotter(shape=(2, 1), window_size=[1100, 800])
plotter.subplot(0, 0)
plotter.add_text("Velocity at the optimum", font_size=10)
plotter.add_mesh(mesh_grid, style="wireframe", color="lightgrey", opacity=0.3)
plotter.add_mesh(velocity_glyphs, scalars="|u|", cmap="viridis")
plotter.camera.tight(padding=0.05, view="xy", adjust_render_window=False)
plotter.subplot(1, 0)
plotter.add_text("Pressure at the optimum", font_size=10)
plotter.add_mesh(warped_pressure, scalars="Pressure", cmap="coolwarm", show_edges=False)
# A warped surface only reads in three dimensions, so this panel is tilted rather than kept
# top-down. `view_isometric` would put +x towards the lower left, mirroring the channel
# relative to the velocity panel above; starting from the xz-plane keeps +x to the right, so
# the inlet is on the left in both panels (and in the original demo's own figures).
plotter.view_xz()  # type: ignore[call-arg]
plotter.camera.elevation = 25.0
plotter.camera.azimuth = -20.0
plotter.camera.zoom(1.5)
if pyvista.OFF_SCREEN:
    plotter.screenshot("boundary_control_state.png")
else:
    plotter.show()
# -

# The optimal control draws fluid *into* the circle on the upstream side and expels it
# downstream: rather than forcing the flow around the obstacle, it lets the obstacle pass
# flow through, which is what reduces the energy dissipated into heat. The strongest control
# sits on the upstream face, where the oncoming flow would otherwise stagnate.
#
# The regularisation term `alpha/2 * inner(g, g) * ds(obstacle_marker)` is what keeps that
# control finite; raising `alpha` shrinks it towards zero.
#
# ## Verifying the implementation
#
# Finally we check the derivative that drove the optimisation, with a Taylor test: perturb the
# control by `eps * h` and watch how fast the remainder of a Taylor expansion of the reduced
# functional decays as `eps` shrinks.
#
# Dropping the derivative term (`dJdm=0`) leaves a remainder of size $\mathcal{O}(\epsilon)$,
# so it should converge at rate 1 -- that only confirms the functional is being re-evaluated
# at the perturbed control. Including the adjoint gradient leaves
# $\mathcal{O}(\epsilon^2)$ and so rate 2, and *that* is the test of the adjoint: rate 2 is
# reached only if `Jhat.derivative()` really is the gradient of `Jhat`.
#
# The expansion point is the zero control, not `g_opt`. At the optimum the gradient vanishes,
# which would take the first-order remainder with it and collapse the rate-1 check into the
# rate-2 one, testing nothing. The direction is `g_opt` itself, which is supported exactly on
# the circle -- a direction supported elsewhere would leave the functional unchanged, since
# the control enters the problem only through the boundary condition on that circle.

with pyadjoint.stop_annotating():
    expansion_point = Function(V, name="TaylorExpansionPoint")

rate_0 = pyadjoint.taylor_test(Jhat, expansion_point, g_opt, dJdm=0)
rate_1 = pyadjoint.taylor_test(Jhat, expansion_point, g_opt)
print(f"Taylor convergence rates: zeroth order {rate_0:.3f} (expect 1), first order {rate_1:.3f} (expect 2)")
assert np.isclose(rate_0, 1.0, atol=0.1), f"zeroth-order Taylor rate {rate_0} is not 1"
assert np.isclose(rate_1, 2.0, atol=0.1), f"first-order Taylor rate {rate_1} is not 2"
