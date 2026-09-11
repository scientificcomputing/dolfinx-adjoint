# # Drag minimization over an obstacle in Stokes flow
# *Section author: Jørgen S. Dokken ([dokken@simula.no](mailto:dokken@simula.no))*.
#
# Converted from the [dolfin-adjoint demo of the same
# name](https://github.com/dolfin-adjoint/dolfin-adjoint/tree/main/examples/stokes-shape-opt),
# with the mesh generation folded in rather than kept in a separate script.

# This is the classical shape optimization problem of minimizing the drag on an obstacle in
# Stokes flow, first analyzed by Pironneau {cite}`pironneau1974optimum`, who found the
# optimal geometry to be a rugby-ball shape with a 90 degree wedge front and back. We start
# from a circular obstacle in a duct: inlet on the left, outlet on the right, no-slip walls
# top and bottom.

# ## Deforming the domain
#
# The fluid domain is written as a perturbation of its undeformed state $\Omega_0$,
#
# $$ \Omega(s) = \{x + s(x) \mid x \in \Omega_0\}, $$
#
# where $s$ solves a linear elasticity problem {cite}`schulz2016computational` with a
# variable Lamé parameter $\mu$:
#
# $$
# \begin{align}
# \mathrm{div}(\sigma(s)) &= 0 && \text{in } \Omega_0, \\
# s &= 0 && \text{on } \Lambda_1\cup\Lambda_2\cup\Lambda_3, \\
# \sigma(s) \cdot n &= h && \text{on } \Gamma,
# \end{align}
# $$
#
# with $\sigma(s) = 2\mu_{\mathrm{elas}}\,\epsilon(s)$ and
# $\epsilon(s) = \tfrac12(\nabla s + \nabla s^T)$. Here $\Lambda_1,\Lambda_2,\Lambda_3$ are
# the walls, inlet and outlet, and $\Gamma$ the obstacle. Taking $\mu_{\mathrm{elas}}$ large
# near the obstacle and small at the outer walls makes the mesh near the obstacle behave
# stiffly, so the deformation is spread smoothly instead of tangling the cells adjacent to
# the moving boundary. It solves
#
# $$
# \begin{align}
# \Delta \mu_{\mathrm{elas}} &= 0 && \text{in } \Omega_0, \\
# \mu_{\mathrm{elas}} &= 1 && \text{on } \Lambda_1\cup\Lambda_2\cup\Lambda_3, \\
# \mu_{\mathrm{elas}} &= 500 && \text{on } \Gamma.
# \end{align}
# $$
#
# As in the original demo, the elasticity problem is *not* used as a Riesz map for the shape
# derivative; the traction $h$ on the obstacle is itself the design variable.

# ## The optimization problem
#
# $$
# \min_{h,u,s} \int_{\Omega(s)} \sum_{i,j=1}^2 \left(\frac{\partial u_i}{\partial x_j}\right)^2 \mathrm{d}x
# + \alpha\Big(\mathrm{Vol}(\Omega(s)) - \mathrm{Vol}(\Omega_0)\Big)^2
# + \beta\sum_{j=1}^2\Big(\mathrm{Bc}_j(\Omega(s)) - \mathrm{Bc}_j(\Omega_0)\Big)^2,
# $$
#
# where $\mathrm{Vol}$ and $\mathrm{Bc}_j$ are the volume and the $j$-th barycenter component
# *of the obstacle*. Without those two penalties the obstacle would simply shrink away. The
# velocity $u$ solves the Stokes equations
#
# $$
# \begin{align}
# -\Delta u + \nabla p &= 0 && \text{in } \Omega(s), \\
# \mathrm{div}(u) &= 0 && \text{in } \Omega(s), \\
# u &= 0 && \text{on } \Gamma(s)\cup\Lambda_1, \\
# u &= g && \text{on } \Lambda_2, \\
# \frac{\partial u}{\partial n} + pn &= 0 && \text{on } \Lambda_3.
# \end{align}
# $$

# ## Implementation

# +
from mpi4py import MPI

import basix.ufl
import dolfinx
import dolfinx.fem.petsc
import gmsh
import matplotlib.pyplot as plt
import matplotlib.tri
import numpy as np
import pyadjoint
import ufl
from dolfinx.io import gmsh as gmshio

import dolfinx_adjoint

# -

# Facet markers and the geometry of the duct and the obstacle.

# +
INFLOW, OUTFLOW, WALL, OBSTACLE = 1, 2, 3, 4
L, H = 1.0, 1.0  # duct length and height
c_x, c_y = L / 2, H / 2  # obstacle centre
r_x = 0.126157  # obstacle radius
# -

# ### Mesh generation
#
# The duct with the circular obstacle cut out of it, built directly with the gmsh Python API
# and handed to DOLFINx in memory. The cell size is graded towards the obstacle, where the
# geometry actually moves.


# +
def create_mesh(resolution: float = 0.02, order: int = 2, comm=MPI.COMM_WORLD, rank: int = 0):
    """Build the duct-with-obstacle mesh and its facet markers.

    Second order by default. The obstacle is a circle, and a straight-sided mesh can only
    approximate it -- with ``order=1`` its area comes out 0.4% below the exact one, and the
    boundary the optimizer moves is a polygon. A curved (P2) geometry represents it properly,
    and the geometry function space the displacement lives in becomes vector P2 to match, so
    the mid-edge nodes are part of the design too.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    if comm.rank == rank:
        centre = gmsh.model.occ.addPoint(c_x, c_y, 0)
        west = gmsh.model.occ.addPoint(c_x - r_x, c_y, 0)
        north = gmsh.model.occ.addPoint(c_x, c_y + r_x, 0)
        east = gmsh.model.occ.addPoint(c_x + r_x, c_y, 0)
        south = gmsh.model.occ.addPoint(c_x, c_y - r_x, 0)
        arcs = [
            gmsh.model.occ.addEllipseArc(start, centre, end, end)
            for start, end in [(west, north), (north, east), (east, south), (south, west)]
        ]
        obstacle = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(arcs)])
        duct = gmsh.model.occ.addRectangle(0, 0, 0, L, H)
        fluid = gmsh.model.occ.cut([(2, duct)], [(2, obstacle)])
        gmsh.model.occ.synchronize()

        # Sort the boundary curves by where their centre of mass sits.
        walls, obstacles = [], []
        for dim, tag in gmsh.model.occ.getEntities(dim=1):
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            if np.allclose(com, [0, H / 2, 0]):
                gmsh.model.addPhysicalGroup(1, [tag], INFLOW)
            elif np.allclose(com, [L, H / 2, 0]):
                gmsh.model.addPhysicalGroup(1, [tag], OUTFLOW)
            elif np.allclose(com, [L / 2, 0, 0]) or np.allclose(com, [L / 2, H, 0]):
                walls.append(tag)
            else:
                obstacles.append(tag)
        gmsh.model.addPhysicalGroup(1, walls, WALL)
        gmsh.model.addPhysicalGroup(1, obstacles, OBSTACLE)
        gmsh.model.addPhysicalGroup(2, [surface[1] for surface in fluid[0]], 12)

        # Refine towards the obstacle.
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "CurvesList", obstacles)
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", resolution)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", 4 * resolution)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 0.5 * r_x)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 2 * r_x)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)

    mesh_data = gmshio.model_to_mesh(gmsh.model, comm, rank, gdim=2)
    gmsh.finalize()
    return mesh_data.mesh, mesh_data.facet_tags


mesh, facet_tags = create_mesh()


def triangulation(domain: dolfinx.mesh.Mesh) -> matplotlib.tri.Triangulation:
    """A snapshot of ``domain``'s current triangles, for plotting.

    Takes a copy of the coordinates: the mesh is moved in place, so a triangulation sharing
    its arrays would silently follow it and the "initial" mesh would plot on top of the
    optimized one.
    """
    # A higher-order cell carries mid-edge nodes as well; matplotlib draws straight triangles,
    # so only the three vertex nodes are used. The curvature still shows in the obstacle
    # outline below, which is drawn through every boundary node.
    dofmap = np.asarray(domain.geometry.dofmaps[0])
    nodes_per_cell = dofmap.size // domain.topology.index_map(domain.topology.dim).size_local
    cells = dofmap.reshape(-1, nodes_per_cell)[:, :3]
    x = domain.geometry.x.copy()
    return matplotlib.tri.Triangulation(x[:, 0], x[:, 1], cells)


def obstacle_outline(domain: dolfinx.mesh.Mesh, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The obstacle boundary as a closed curve, for plotting.

    The boundary nodes come back unordered, so they are sorted by angle about the obstacle's
    centre -- valid because the obstacle stays star-shaped throughout.
    """
    x = domain.geometry.x[nodes, 0].copy()
    y = domain.geometry.x[nodes, 1].copy()
    order = np.argsort(np.arctan2(y - c_y, x - c_x))
    return np.append(x[order], x[order][0]), np.append(y[order], y[order][0])


initial_triangulation = triangulation(mesh)

# Cell orientations of the undeformed mesh, to check against once the shape has moved.
_DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
_detJ = dolfinx.fem.Function(_DG0)
_detJ.interpolate(dolfinx.fem.Expression(ufl.JacobianDeterminant(mesh), _DG0.element.interpolation_points))
initial_orientations = np.sign(_detJ.x.array[: _DG0.dofmap.index_map.size_local]).copy()
# -

# Direct solvers throughout. The Stokes system is a saddle point problem, so its factorization
# needs a solver that pivots -- the default `"lu"` reports a missing diagonal entry on the
# zero pressure block.

# +
lu_options = {"ksp_type": "preonly", "pc_type": "lu"}
saddle_point_options = lu_options | {"pc_factor_mat_solver_type": "mumps"}

tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, tdim)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
x = ufl.SpatialCoordinate(mesh)
# -

# **Put the mesh on the tape before anything is posed on it.** The deformation problem below
# is solved on $\Omega_0$, but the mesh it is posed on is the same object that is moved a few
# lines later. Registering it now is what lets every block record *which* geometry it was
# built on, so that replaying the tape rewinds the mesh to $\Omega_0$ before re-solving the
# deformation problem rather than re-solving it on the previously deformed domain. Without
# this the gradient is quietly wrong from the second evaluation onwards.

dolfinx_adjoint.annotate_mesh(mesh)

# The control is the traction $h$ on the obstacle. It lives in the mesh's geometry function
# space, which is also where the displacement has to live for
# {py:func}`dolfinx_adjoint.move`.
#
# The original demo puts $h$ on a `BoundaryMesh` and transfers it into the volume.
# dolfinx-adjoint has no boundary-mesh transfer, so $h$ is a volume field here. Only its
# obstacle-boundary dofs ever enter a form, so every other dof has exactly zero gradient and
# stays at its initial value -- the optimization is over the same set of designs, just
# carried in a larger vector.

# +
S = dolfinx_adjoint.geometry_function_space(mesh)
h = dolfinx_adjoint.Function(S, name="Design")

# The obstacle's boundary nodes, and its outline while the mesh is still undeformed.
obstacle_nodes = dolfinx.fem.locate_dofs_topological(S, tdim - 1, facet_tags.find(OBSTACLE))
initial_outline = obstacle_outline(mesh, obstacle_nodes)
initial_obstacle_coordinates = mesh.geometry.x[obstacle_nodes, :2].copy()

# The obstacle's volume and barycenter in the undeformed configuration.
with pyadjoint.stop_annotating():
    fluid_volume_0 = mesh.comm.allreduce(
        dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * ufl.dx(domain=mesh))), op=MPI.SUM
    )
obstacle_volume_0 = L * H - fluid_volume_0
# -

# ### The variable Lamé parameter
#
# $\mu_{\mathrm{elas}}$ does not depend on the control, so it is computed once with annotation
# switched off. It is still built as a {py:class}`dolfinx_adjoint.Function`, because
# dolfinx-adjoint identifies a form's coefficients by `ufl_id()`, which only the overloaded
# type carries.

# +
with pyadjoint.stop_annotating():
    V_mu = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    trial_mu, test_mu = ufl.TrialFunction(V_mu), ufl.TestFunction(V_mu)
    bcs_mu = [
        dolfinx.fem.dirichletbc(
            dolfinx.default_scalar_type(value),
            dolfinx.fem.locate_dofs_topological(V_mu, tdim - 1, facet_tags.find(marker)),
            V_mu,
        )
        for marker, value in [(INFLOW, 1.0), (OUTFLOW, 1.0), (WALL, 1.0), (OBSTACLE, 500.0)]
    ]
    mu_problem = dolfinx.fem.petsc.LinearProblem(
        ufl.inner(ufl.grad(trial_mu), ufl.grad(test_mu)) * ufl.dx,
        ufl.inner(dolfinx.fem.Constant(mesh, 0.0), test_mu) * ufl.dx,
        bcs=bcs_mu,
        petsc_options=lu_options,
        petsc_options_prefix="mu_",
    )
    mu_solution = mu_problem.solve()
    mu_solution = mu_solution[0] if isinstance(mu_solution, tuple) else mu_solution

mu = dolfinx_adjoint.Function(V_mu, name="mu")
mu.x.array[:] = mu_solution.x.array
# -

# ### Deforming the mesh
#
# The elasticity problem is posed directly in the geometry function space, so its solution can
# be handed straight to {py:func}`dolfinx_adjoint.move` with no interpolation in between.

# +
trial_s, test_s = ufl.TrialFunction(S), ufl.TestFunction(S)


def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u, mu):
    return 2 * mu * epsilon(u)


clamped = np.zeros(mesh.geometry.dim, dtype=dolfinx.default_scalar_type)
bcs_s = [
    dolfinx.fem.dirichletbc(clamped, dolfinx.fem.locate_dofs_topological(S, tdim - 1, facet_tags.find(marker)), S)
    for marker in (INFLOW, OUTFLOW, WALL)
]

deformation = dolfinx_adjoint.LinearProblem(
    ufl.inner(sigma(trial_s, mu), ufl.grad(test_s)) * ufl.dx,
    ufl.inner(h, test_s) * ds(OBSTACLE),
    bcs=bcs_s,
    petsc_options=lu_options,
    petsc_options_prefix="deformation_",
)
s = deformation.solve()
s.name = "Mesh perturbation field"

dolfinx_adjoint.move(mesh, s)
# -

# ### The Stokes equations
#
# Taylor-Hood elements, in blocked form: velocity in $P_2$, pressure in $P_1$.

# +
P2 = basix.ufl.element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
P1 = basix.ufl.element("Lagrange", mesh.basix_cell(), 1)
V = dolfinx.fem.functionspace(mesh, P2)
Q = dolfinx.fem.functionspace(mesh, P1)

W = ufl.MixedFunctionSpace(V, Q)
u, p = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)
a = ufl.extract_blocks(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - ufl.div(v) * p * ufl.dx - ufl.div(u) * q * ufl.dx)
rhs = ufl.extract_blocks(
    ufl.inner(dolfinx.fem.Constant(mesh, (0.0, 0.0)), v) * ufl.dx + dolfinx.fem.Constant(mesh, 0.0) * q * ufl.dx
)

inlet_profile = dolfinx_adjoint.Function(V, name="inlet")
inlet_profile.interpolate(lambda x: np.vstack((np.sin(np.pi * x[1]), np.zeros_like(x[0]))))
no_slip = np.zeros(mesh.geometry.dim, dtype=dolfinx.default_scalar_type)
bcs = [
    dolfinx.fem.dirichletbc(inlet_profile, dolfinx.fem.locate_dofs_topological(V, tdim - 1, facet_tags.find(INFLOW))),
    dolfinx.fem.dirichletbc(no_slip, dolfinx.fem.locate_dofs_topological(V, tdim - 1, facet_tags.find(OBSTACLE)), V),
    dolfinx.fem.dirichletbc(no_slip, dolfinx.fem.locate_dofs_topological(V, tdim - 1, facet_tags.find(WALL)), V),
]

uh = dolfinx_adjoint.Function(V, name="Velocity")
ph = dolfinx_adjoint.Function(Q, name="Pressure")
stokes = dolfinx_adjoint.LinearProblem(
    a,
    rhs,
    u=[uh, ph],
    bcs=bcs,
    petsc_options=saddle_point_options,
    adjoint_petsc_options=saddle_point_options,
    tlm_petsc_options=saddle_point_options,
    petsc_options_prefix="stokes_",
)
stokes.solve()
# -

# ### The functional
#
# The dissipated energy, plus the volume and barycenter penalties that hold the obstacle's
# size and position fixed.

# +
alpha, beta = 1e6, 1e6

dissipation = dolfinx_adjoint.assemble_scalar(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx)
fluid_volume = dolfinx_adjoint.assemble_scalar(1 * ufl.dx(domain=mesh))
obstacle_volume = L * H - fluid_volume

barycenter_x = (L**2 * H / 2 - dolfinx_adjoint.assemble_scalar(x[0] * ufl.dx(domain=mesh))) / obstacle_volume
barycenter_y = (L * H**2 / 2 - dolfinx_adjoint.assemble_scalar(x[1] * ufl.dx(domain=mesh))) / obstacle_volume

J = dissipation
J = J + alpha * (obstacle_volume - obstacle_volume_0) ** 2
J = J + beta * ((barycenter_x - c_x) ** 2 + (barycenter_y - c_y) ** 2)

print(f"Initial dissipation: {float(dissipation):.6f}   obstacle volume: {float(obstacle_volume):.6f}")
# -

# ### Verifying the shape gradient
#
# A first-order Taylor test, in the same direction the original demo uses. Note that only the
# gradient is checked: dolfinx-adjoint refuses a shape *Hessian* across a PDE solve rather
# than return a wrong one, so the original's second-order `taylor_to_dict` check has no
# counterpart yet.

# +
Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(h))

perturbation = dolfinx_adjoint.Function(S)
perturbation.interpolate(lambda x: np.vstack((-x[0], x[1])))
rate = pyadjoint.taylor_test(Jhat, h, perturbation)
print(f"Taylor convergence rate: {rate:.3f}")
assert rate > 1.9
# -

# ### Optimizing

# +
h_opt = pyadjoint.minimize(Jhat, tol=1e-6, options={"gtol": 1e-6, "maxiter": 300, "disp": False})
J_opt = Jhat(h_opt)
print(f"J: {float(J):.6f} -> {J_opt:.6f}")
print(
    f"Dissipation: {float(dissipation):.6f} -> {dissipation.block_variable.checkpoint:.6f}   "
    f"obstacle volume: {obstacle_volume_0:.6f} -> {obstacle_volume.block_variable.checkpoint:.6f}"
)
# -

# The obstacle's extent tells the story numerically: it starts as a circle, so its width and
# height agree, and it should end up elongated along the flow -- Pironneau's rugby ball.

# +
coordinates = mesh.geometry.x[obstacle_nodes]


def global_extent(values: np.ndarray) -> float:
    largest = mesh.comm.allreduce(values.max() if values.size else -np.inf, op=MPI.MAX)
    smallest = mesh.comm.allreduce(values.min() if values.size else np.inf, op=MPI.MIN)
    return largest - smallest


width = global_extent(coordinates[:, 0])
height = global_extent(coordinates[:, 1])

# The deformation is only meaningful while the mesh stays untangled. Each cell must keep the
# sign its Jacobian determinant started with -- not be positive: DOLFINx does not orient cells
# consistently, and integrates against |detJ|, so the absolute sign carries no information.
DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
detJ = dolfinx.fem.Function(DG0)
detJ.interpolate(dolfinx.fem.Expression(ufl.JacobianDeterminant(mesh), DG0.element.interpolation_points))
owned = detJ.x.array[: DG0.dofmap.index_map.size_local]
assert mesh.comm.allreduce(int(np.count_nonzero(np.sign(owned) != initial_orientations)), op=MPI.SUM) == 0, (
    "the optimized mesh is tangled"
)
print(f"Obstacle extent: {width:.4f} along the flow by {height:.4f} across (started {2 * r_x:.4f} both ways)")
print(f"Aspect ratio: {width / height:.3f}")
# -

# `Jhat(h_opt)` leaves the mesh at the optimized shape, so the two configurations can be drawn
# against each other. Both go in *one* pair of axes, initial in blue and optimized in red, as
# in the original demo -- drawn side by side in separate panels each autoscales to its own
# shape, and a genuinely different obstacle then looks much the same size.

# +
optimal_triangulation = triangulation(mesh)
optimal_outline = obstacle_outline(mesh, obstacle_nodes)

figure, (whole, zoom) = plt.subplots(1, 2, figsize=(11, 5))

whole.triplot(initial_triangulation, color="tab:blue", linewidth=0.3)
whole.triplot(optimal_triangulation, color="tab:red", linewidth=0.3)
whole.set_title("Whole duct")

# The duct walls are clamped by the mesh-deformation problem, so all the movement is at the
# obstacle -- worth a panel of its own, or the interesting part is a few percent of the figure.
# There the mesh is drawn faintly and the two obstacle boundaries on top of it, since the
# shape is the point and two full meshes overlaid mostly obscure it.
zoom.triplot(initial_triangulation, color="0.85", linewidth=0.3)
zoom.triplot(optimal_triangulation, color="0.85", linewidth=0.3)
zoom.plot(*initial_outline, color="tab:blue", linewidth=2.0)
zoom.plot(*optimal_outline, color="tab:red", linewidth=2.0)

# The displacement each boundary node actually underwent, drawn at true length. Taken
# per node rather than by differencing the two outlines: those are each sorted by angle
# about the centre, and nothing guarantees the two orderings agree.
displacement = mesh.geometry.x[obstacle_nodes, :2] - initial_obstacle_coordinates
zoom.quiver(
    initial_obstacle_coordinates[:, 0],
    initial_obstacle_coordinates[:, 1],
    displacement[:, 0],
    displacement[:, 1],
    angles="xy",
    scale_units="xy",
    scale=1.0,
    width=0.003,
    color="0.25",
    zorder=3,
)
zoom.set_title("Obstacle, with the deformation it underwent")
margin = 0.62 * max(width, height)  # sized from the optimized shape, so none of it is cropped
zoom.set_xlim(c_x - margin, c_x + margin)
zoom.set_ylim(c_y - margin, c_y + margin)

for axes in (whole, zoom):
    axes.set_aspect("equal")
    axes.axis("off")

figure.legend(
    handles=[
        plt.Line2D([], [], color="tab:blue", label="Initial mesh"),
        plt.Line2D([], [], color="tab:red", label="Optimized mesh"),
        plt.Line2D([], [], color="0.25", label="Boundary displacement"),
    ],
    loc="lower center",
    ncols=3,
)
figure.savefig("stokes_shape_optimization.png", dpi=200, bbox_inches="tight")
# -

# ```{bibliography}
# :filter: cited and ({"demos/stokes_shape_optimization"} >= docnames)
# ```
