# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3 (DOLFINx complex)
#     language: python
#     name: python3-complex
# ---

# # Topology optimization of a dielectric metalens
#
# We design a flat dielectric lens that focuses a plane wave onto a small spot. The wave problem
# is a complex-valued Helmholtz equation, while the design density and the objective are real;
# {py:mod}`dolfinx_adjoint` differentiates through the complex solve for us.
#
# ```{admonition} This demo requires a complex PETSc build
# :class: warning
# See the [installation instructions](https://jsdokken.com/dolfinx-tutorial/chapter1/complex_mode.html#installation-of-fenicsx-with-complex-number-support).
# The documentation runs it with the `python3-complex` Jupyter kernel of the DOLFINx images.
# ```
#
# ## Problem definition
#
# In two dimensions with TM polarization, the out-of-plane electric field $E$ satisfies
#
# $$
# \nabla^2 E + k_0^2\,\varepsilon_r(\mathbf{x})\,E = 0,
# \qquad k_0 = \frac{2\pi}{\lambda},
# $$
#
# with an $e^{-i\omega t}$ time convention.
#
# ### Scattered-field formulation
#
# We write $E = E_\mathrm{inc} + E_s$, with the incident plane wave $E_\mathrm{inc} = e^{ik_0 x}$.
# The scattered field then solves
#
# $$
# \nabla^2 E_s + k_0^2 \varepsilon_r E_s
# = -k_0^2\,(\varepsilon_r - 1)\,E_\mathrm{inc},
# $$
#
# whose source is nonzero only inside the design region, where $\varepsilon_r \neq 1$.
#
# ### Perfectly matched layers
#
# The box is surrounded by a perfectly matched layer (PML), a complex coordinate stretch
#
# $$
# s_x(x) = 1 + i\,\sigma\left(\frac{\max(|x| - x_\mathrm{phys},\,0)}{d_\mathrm{PML}}\right)^2,
# $$
#
# and likewise $s_y$, which absorbs outgoing waves. The weak form reads: find $E_s \in V$ such that
#
# $$
# \int_\Omega (\mathbf{A}\nabla E_s)\cdot\overline{\nabla v}~\mathrm{d}x
# - k_0^2 \int_\Omega c\, E_s \overline{v}~\mathrm{d}x
# - k_0^2 \int_{\Omega_d} (\varepsilon_r - 1) E_s \overline{v}~\mathrm{d}x
# = k_0^2 \int_{\Omega_d} (\varepsilon_r - 1) E_\mathrm{inc} \overline{v}~\mathrm{d}x
# $$
#
# for all $v \in V$, with $\mathbf{A} = \operatorname{diag}(s_y/s_x,\ s_x/s_y)$, $c = s_x s_y$ and
# $E_s = 0$ on the outer boundary.
#
# ### Mirror symmetry
#
# The incident wave and the PML are symmetric about $y = 0$, and we want a symmetric lens. The field
# is then symmetric too, so we only solve on the upper half $y \ge 0$, with the symmetry condition
# $\partial E_s / \partial y = 0$ on $y = 0$. This is the natural boundary condition of the weak form,
# so it needs no extra term: we simply leave $y = 0$ out of the Dirichlet condition. The design is
# symmetric by construction, and the problem is half the size.
#
# ### Design parametrization
#
# The design region $\Omega_d$ carries a density $\rho \in [0,1]$, mapped to a permittivity in
# three standard steps:
#
# 1. **Filtering.** A Helmholtz filter $-r^2\nabla^2\tilde\rho + \tilde\rho = \rho$ on
#    $\Omega_d$, with natural boundary conditions, imposes a minimum length scale $r$.
# 2. **Projection.** A smoothed Heaviside pushes $\tilde\rho$ towards 0 or 1,
#
#    $$
#    \bar\rho = \frac{\tanh(\beta\eta) + \tanh(\beta(\tilde\rho - \eta))}
#                    {\tanh(\beta\eta) + \tanh(\beta(1 - \eta))},
#    $$
#
#    with $\eta = 0.5$ and the sharpness $\beta$ increased in stages.
# 3. **Interpolation.** $\varepsilon_r(\bar\rho) = 1 + \bar\rho^{\,p}(\varepsilon_\mathrm{mat} - 1)$.
#
# ### Objective
#
# We maximize the mean intensity over a small square focal spot $\Omega_f$ behind the lens, i.e. minimize
#
# $$
# J(\rho) = -\frac{1}{|\Omega_f|}\int_{\Omega_f} |E_\mathrm{inc} + E_s|^2~\mathrm{d}x.
# $$
#
# ## Implementation

# +
from mpi4py import MPI

import dolfinx
import dolfinx.fem.petsc
import gmsh
import matplotlib.pyplot as plt
import matplotlib.tri
import mmapy
import numpy as np
import pyadjoint
import ufl

import dolfinx_adjoint

try:
    from dolfinx.io import gmsh as gmshio
except ImportError:
    from dolfinx.io import gmshio  # type: ignore[attr-defined, no-redef]

# -

if not np.issubdtype(dolfinx.default_scalar_type, np.complexfloating):
    raise RuntimeError("This demo needs a complex-scalar DOLFINx build.")

# ## Parameters
#
# Lengths are in units of the free-space wavelength $\lambda$.

# +
WAVELENGTH = 1.0
K0 = 2.0 * np.pi / WAVELENGTH

PML_WIDTH = 0.6  # absorbing-layer thickness on every side
PML_STRENGTH = 4.0  # peak of the quadratic conductivity profile
X_PHYS, Y_PHYS = 3.0, 3.2  # half-extent of the physical (non-PML) region
LX, LY = X_PHYS + PML_WIDTH, Y_PHYS + PML_WIDTH  # half-extent of the whole box

# Regions as (xmin, xmax, ymin, ymax), upper half only: the lower half is their mirror image.
DESIGN_BOX = (-0.5, 0.5, 0.0, 3.0)  # the lens slab
FOCUS_BOX = (1.8, 2.2, 0.0, 0.2)  # the square spot whose intensity we maximize

EPS_MATERIAL = 4.0  # relative permittivity of the lens material
PENALIZATION = 1.0  # interpolation exponent p
FILTER_RADIUS = 0.2  # filter length scale; a few cell widths at least
ETA = 0.5  # projection threshold
BETA_STAGES = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)  # projection-sharpness continuation
ITERATIONS_PER_STAGE = 40

CELLS_PER_WAVELENGTH = 16  # cell width h = WAVELENGTH / CELLS_PER_WAVELENGTH

RHO_INIT = 0.5  # uniform grey starting design
LIVE_PREVIEW = False  # redraw the design and field after every optimizer iteration
MMA_MOVE = 0.1  # largest change of any density in one MMA iteration

DESIGN_TAG, FOCUS_TAG, BULK_TAG = 1, 2, 3

PETSC_LU = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
# -

# ## Mesh and subdomains
#
# The mesh covers the upper half of the box. The design slab and focal spot are rectangles on a
# structured quadrilateral grid, so their boundaries conform to the mesh.

# +
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)

gdim = 2
mesh_comm, model_rank = MPI.COMM_WORLD, 0
cell_size = WAVELENGTH / CELLS_PER_WAVELENGTH

if mesh_comm.rank == model_rank:
    occ = gmsh.model.occ

    # Every region boundary lies on one of these lines, which cut the box into a grid of rectangles.
    x_lines = sorted({-LX, -X_PHYS, *DESIGN_BOX[:2], *FOCUS_BOX[:2], X_PHYS, LX})
    y_lines = sorted({*DESIGN_BOX[2:], *FOCUS_BOX[2:], Y_PHYS, LY})
    for x0, x1 in zip(x_lines[:-1], x_lines[1:]):
        for y0, y1 in zip(y_lines[:-1], y_lines[1:]):
            occ.addRectangle(x0, y0, 0.0, x1 - x0, y1 - y0)
    occ.removeAllDuplicates()
    occ.synchronize()

    def inside(box, point):
        return box[0] < point[0] < box[1] and box[2] < point[1] < box[3]

    regions: dict[int, list[int]] = {DESIGN_TAG: [], FOCUS_TAG: [], BULK_TAG: []}
    for _, surface in gmsh.model.getEntities(gdim):
        centre = occ.getCenterOfMass(gdim, surface)
        tag = DESIGN_TAG if inside(DESIGN_BOX, centre) else FOCUS_TAG if inside(FOCUS_BOX, centre) else BULK_TAG
        regions[tag].append(surface)
    gmsh.model.addPhysicalGroup(gdim, regions[DESIGN_TAG], DESIGN_TAG, name="Design")
    gmsh.model.addPhysicalGroup(gdim, regions[FOCUS_TAG], FOCUS_TAG, name="Focus")
    gmsh.model.addPhysicalGroup(gdim, regions[BULK_TAG], BULK_TAG, name="Background")

    # A structured quadrilateral grid, with a cell width close to `cell_size` along every edge.
    for _, curve in gmsh.model.getEntities(1):
        num_cells = max(1, round(occ.getMass(1, curve) / cell_size))
        gmsh.model.mesh.setTransfiniteCurve(curve, num_cells + 1)
    for _, surface in gmsh.model.getEntities(gdim):
        gmsh.model.mesh.setTransfiniteSurface(surface)
        gmsh.model.mesh.setRecombine(gdim, surface)
    gmsh.model.mesh.generate(gdim)

mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
gmsh.finalize()

mesh = mesh_data.mesh
assert mesh_data.cell_tags is not None
cell_tags = mesh_data.cell_tags
tdim = mesh.topology.dim

design_cells = cell_tags.find(DESIGN_TAG)
focus_cells = cell_tags.find(FOCUS_TAG)
dx = ufl.Measure("dx", domain=mesh, subdomain_data=cell_tags)

# The density lives on a submesh of the design region.
design_mesh, design_to_parent, _, _ = dolfinx.mesh.create_submesh(mesh, tdim, design_cells)
dx_design = ufl.Measure("dx", domain=design_mesh)

print(f"{mesh.topology.index_map(tdim).size_global} cells, {len(design_cells)} in the design region")
# -

# ## PML and incident field

# +
x = ufl.SpatialCoordinate(mesh)


def stretch(coordinate, half_width):
    """Complex coordinate-stretch factor for a PML of thickness `PML_WIDTH`."""
    depth = ufl.max_value(abs(coordinate) - half_width, 0.0) / PML_WIDTH
    return 1.0 + 1j * PML_STRENGTH * depth**2


s_x, s_y = stretch(x[0], X_PHYS), stretch(x[1], Y_PHYS)
pml_tensor = ufl.as_matrix([[s_y / s_x, 0], [0, s_x / s_y]])
pml_scale = s_x * s_y

incident = ufl.exp(1j * K0 * x[0])
# -

# ## Control, filter and projection
#
# The density `rho` is the control. `beta` is a plain constant, so the continuation loop can change
# it between stages.

# +
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))  # scattered field, on the upper half of the box
Q = dolfinx.fem.functionspace(design_mesh, ("Lagrange", 1))  # density, on the design region only

rho = dolfinx_adjoint.Function(Q, name="density")
rho.x.array[:] = RHO_INIT
rho.x.scatter_forward()

trial_rho, w = ufl.TrialFunction(Q), ufl.TestFunction(Q)
rho_filtered = dolfinx_adjoint.Function(Q, name="filtered_density")
filter_problem = dolfinx_adjoint.LinearProblem(
    (FILTER_RADIUS**2 * ufl.inner(ufl.grad(trial_rho), ufl.grad(w)) + ufl.inner(trial_rho, w)) * dx_design,
    ufl.inner(rho, w) * dx_design,
    u=rho_filtered,
    petsc_options=PETSC_LU,
    adjoint_petsc_options=PETSC_LU,
    tlm_petsc_options=PETSC_LU,
)
filter_problem.solve()

beta = dolfinx.fem.Constant(design_mesh, dolfinx.default_scalar_type(BETA_STAGES[0]))
projected = (ufl.tanh(beta * ETA) + ufl.tanh(beta * (rho_filtered - ETA))) / (
    ufl.tanh(beta * ETA) + ufl.tanh(beta * (1 - ETA))
)
permittivity = 1.0 + projected**PENALIZATION * (EPS_MATERIAL - 1.0)
# -

# ## Forward solve

# +
scattered = dolfinx_adjoint.Function(V, name="scattered_field")
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)


def helmholtz_forms(eps_r):
    """Scattered-field weak form for a relative permittivity `eps_r` on the design region."""
    a = (ufl.inner(pml_tensor * ufl.grad(u), ufl.grad(v)) - K0**2 * pml_scale * ufl.inner(u, v)) * dx
    a -= K0**2 * ufl.inner((eps_r - 1.0) * u, v) * dx(DESIGN_TAG)
    L = K0**2 * ufl.inner((eps_r - 1.0) * incident, v) * dx(DESIGN_TAG)
    return a, L


a, L = helmholtz_forms(permittivity)

mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
# The symmetry line y = 0 keeps its natural boundary condition.
on_symmetry_line = np.isclose(dolfinx.mesh.compute_midpoints(mesh, tdim - 1, boundary_facets)[:, 1], 0.0)
outer_facets = boundary_facets[~on_symmetry_line]
bc = dolfinx.fem.dirichletbc(
    dolfinx.default_scalar_type(0.0),
    dolfinx.fem.locate_dofs_topological(V, tdim - 1, outer_facets),
    V,
)

# `entity_maps` lets the form use the permittivity, which lives on the design submesh.
problem = dolfinx_adjoint.LinearProblem(
    a,
    L,
    u=scattered,
    bcs=[bc],
    petsc_options=PETSC_LU,
    adjoint_petsc_options=PETSC_LU,
    tlm_petsc_options=PETSC_LU,
    entity_maps=[design_to_parent],
)
problem.solve()
# -

# ## Objective and reduced functional

# +
focus_area = dolfinx_adjoint.assemble_scalar(1.0 * dx(FOCUS_TAG), annotate=False)
design_area = dolfinx_adjoint.assemble_scalar(1.0 * dx_design, annotate=False)
total = scattered + incident
J = -dolfinx_adjoint.assemble_scalar(ufl.inner(total, total) * dx(FOCUS_TAG)) / focus_area

Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(rho))
print(f"intensity enhancement of the grey starting design: {-float(J):.4f}")
# -

# ## Verifying the derivatives
#
# Taylor tests check the gradient and the Hessian: the remainders should converge at rates 1
# (no correction), 2 (gradient) and 3 (gradient and Hessian).

# +
with pyadjoint.stop_annotating():
    direction = dolfinx_adjoint.Function(Q)
    direction.x.array[:] = np.random.default_rng(42).standard_normal(direction.x.array.size)
    direction.x.scatter_forward()

    rate_0 = pyadjoint.taylor_test(Jhat, rho, direction, dJdm=0)
    rate_1 = pyadjoint.taylor_test(Jhat, rho, direction)

    # Return to the base point before taking the Hessian.
    Jhat(rho)
    Jhat.derivative()
    Hm = Jhat.hessian(direction)._ad_dot(direction)
    rate_2 = pyadjoint.taylor_test(Jhat, rho, direction, Hm=Hm)
    print(f"0th-order Taylor rate (expect ~1): {rate_0:.4f}")
    print(f"1st-order Taylor rate (expect ~2): {rate_1:.4f}")
    print(f"2nd-order Taylor rate (expect ~3): {rate_2:.4f}")

    gradient = Jhat.derivative()
    print(f"max |Im(dJ/drho)| (expect 0): {np.abs(gradient.x.array.imag).max():.3e}")
# -


# ## Plotting

# +
field_space = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))


def _triangulation(space):
    """Triangulate the mesh of `space` and its mirror image, for plotting the whole box.

    Each quadrilateral cell becomes two triangles. The mirrored nodes and triangles follow the
    originals, so values are mirrored with `np.tile(values, 2)`.
    """
    cells, _, nodes = dolfinx.plot.vtk_mesh(space)
    quads = cells.reshape(-1, 5)[:, 1:]
    triangles = np.stack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=1).reshape(-1, 3)
    mirrored_nodes = nodes[:, :2] * [1.0, -1.0]
    all_nodes = np.concatenate([nodes[:, :2], mirrored_nodes])
    all_triangles = np.concatenate([triangles, triangles + len(nodes)])
    return matplotlib.tri.Triangulation(all_nodes[:, 0], all_nodes[:, 1], all_triangles)


def _nodal_values(expression, space):
    """Interpolate a UFL expression into `space` and return its real nodal values, mirrored."""
    out = dolfinx.fem.Function(space)
    out.interpolate(dolfinx.fem.Expression(expression, space.element.interpolation_points))
    return np.tile(out.x.array.real, 2)


triangulation = _triangulation(field_space)
design_triangulation = _triangulation(Q)


def draw_geometry(axis):
    """Outline the design slab and the focal spot, and crop to the physical region."""
    for xmin, xmax, _, ymax in (DESIGN_BOX, FOCUS_BOX):
        axis.add_patch(plt.Rectangle((xmin, -ymax), xmax - xmin, 2 * ymax, fill=False, edgecolor="red", linewidth=0.8))
    axis.set_xlim(-X_PHYS, X_PHYS)
    axis.set_ylim(-Y_PHYS, Y_PHYS)
    axis.set_aspect("equal")


def plot_state(density, field, figure=None, cellwise=False):
    """Plot a design, its intensity and the real part of its total field.

    Set `cellwise` for a piecewise-constant (DG0) density, drawn with one colour per cell.
    """
    if figure is None:
        figure = plt.figure(figsize=(14, 4.5), layout="constrained")
    figure.clear()
    axes = figure.subplots(1, 3)
    design_options = {"vmin": 0.0, "vmax": 1.0}
    if cellwise:
        design_values = None
        cell_values = density.x.array[density.function_space.dofmap.list[:, 0]].real
        design_options["facecolors"] = np.tile(np.repeat(cell_values, 2), 2)  # two triangles per cell, mirrored
    else:
        design_values = _nodal_values(density, Q)
    panels = (
        (design_triangulation, design_values, r"design $\bar\rho$", "binary", design_options),
        (
            triangulation,
            _nodal_values(ufl.inner(field, field), field_space),
            r"$|E_\mathrm{tot}|^2$",
            "inferno",
            {"shading": "gouraud"},
        ),
        (
            triangulation,
            _nodal_values(ufl.real(field), field_space),
            r"$\mathrm{Re}(E_\mathrm{tot})$",
            "RdBu_r",
            {"shading": "gouraud"},
        ),
    )
    for axis, (tri, values, title, cmap, options) in zip(axes, panels):
        positional = () if values is None else (values,)
        mappable = axis.tripcolor(tri, *positional, cmap=cmap, **options)
        axis.set_title(title)
        figure.colorbar(mappable, ax=axis)
        draw_geometry(axis)
    return figure


# -

# ## Optimization
#
# We use the method of moving asymptotes (MMA) from [`mmapy`](https://pypi.org/project/mmapy/),
# the usual optimizer for density-based topology optimization. It only needs the objective and
# its gradient, which `Jhat` provides, and handles the bounds $0 \le \rho \le 1$ directly. There
# are no other constraints here (`m = 0`). The MMA history carries over from one $\beta$ stage to
# the next: restarting it resets the asymptotes to their widest, and the resulting large steps
# throw the sharply projected design out of its local optimum.
#
# ```{note}
# The design vector is the owned degrees of freedom of `rho`, so this loop runs in serial.
# ```

# +
num_owned = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs

history: list[float] = []  # enhancement after each optimizer iteration
stage_ends: list[int] = []  # index in `history` where each continuation stage finished

if LIVE_PREVIEW:
    plt.ion()
preview = plt.figure(figsize=(14, 4.5), layout="constrained") if LIVE_PREVIEW else None


def evaluate(design):
    """Objective and gradient at the design vector `design`, as MMA column vectors."""
    rho.x.array[:num_owned] = design.ravel()
    rho.x.scatter_forward()
    value = float(Jhat(rho))
    gradient = Jhat.derivative()
    return value, gradient.x.array[:num_owned].real.reshape(-1, 1)


def record_iterate(value):
    history.append(-value)
    if preview is not None:
        print(f"iteration {len(history)}, enhancement = {history[-1]:.4f}")
        plot_state(projected, total, preview)
        plt.pause(0.01)


n_design = num_owned
m = 0  # no constraints besides the bounds
lower, upper = np.zeros((n_design, 1)), np.ones((n_design, 1))
a0, a, c, d = 1.0, np.zeros((m, 1)), np.ones((m, 1)), np.zeros((m, 1))
no_constraints, no_constraint_gradients = np.zeros((m, 1)), np.zeros((m, n_design))

design = rho.x.array[:num_owned].real.reshape(-1, 1).copy()
previous, older = design.copy(), design.copy()
asymptote_low, asymptote_upp = lower.copy(), upper.copy()

for stage_beta in BETA_STAGES:
    beta.value = dolfinx.default_scalar_type(stage_beta)

    for _ in range(ITERATIONS_PER_STAGE):
        value, gradient = evaluate(design)
        record_iterate(value)
        updated, *_, asymptote_low, asymptote_upp = mmapy.mmasub(
            m,
            n_design,
            len(history),
            design,
            lower,
            upper,
            previous,
            older,
            value,
            gradient,
            no_constraints,
            no_constraint_gradients,
            asymptote_low,
            asymptote_upp,
            a0,
            a,
            c,
            d,
            move=MMA_MOVE,
        )
        design, previous, older = updated, design, previous

    stage_ends.append(len(history))
    print(f"beta = {stage_beta:5.1f}: enhancement {history[-1]:8.4f}")

rho.x.array[:num_owned] = design.ravel()
rho.x.scatter_forward()
# -

# ## Binarization
#
# The measure of non-discreteness
# $M_\mathrm{nd} = \frac{4}{|\Omega_d|}\int_{\Omega_d}\bar\rho(1-\bar\rho)~\mathrm{d}x$ is 1 for
# an all-grey design and 0 for a binary one. To get a manufacturable design we threshold the
# filtered and projected density $\bar\rho$ at 0.5, cell by cell (DG0), and solve the Helmholtz
# problem once more with the resulting two-material permittivity.

# +
with pyadjoint.stop_annotating():
    continuous = -Jhat(rho)
    non_discreteness = (
        dolfinx_adjoint.assemble_scalar(4.0 * projected * (1.0 - projected) * dx_design, annotate=False) / design_area
    )

    Q0 = dolfinx.fem.functionspace(design_mesh, ("Discontinuous Lagrange", 0))
    binary = dolfinx.fem.Function(Q0, name="binarized_density")
    binary.interpolate(dolfinx.fem.Expression(projected, Q0.element.interpolation_points))
    binary.x.array[:] = binary.x.array.real > 0.5
    binary.x.scatter_forward()

    scattered_binary = dolfinx.fem.Function(V, name="binarized_scattered_field")
    a_binary, L_binary = helmholtz_forms(1.0 + binary * (EPS_MATERIAL - 1.0))
    dolfinx.fem.petsc.LinearProblem(
        a_binary,
        L_binary,
        petsc_options_prefix="metalens_binary_",
        u=scattered_binary,
        bcs=[bc],
        petsc_options=PETSC_LU,
        entity_maps=[design_to_parent],
    ).solve()

    total_binary = scattered_binary + incident
    binarized = (
        dolfinx_adjoint.assemble_scalar(ufl.inner(total_binary, total_binary) * dx(FOCUS_TAG), annotate=False)
        / focus_area
    )

print(f"measure of non-discreteness: {non_discreteness:.4f}")
print(f"enhancement, continuous design: {continuous:.4f}")
print(f"enhancement, binarized design:  {binarized:.4f}")
# -

# ## Results
#
# The optimized design, the intensity it produces and the real part of the total field.

# +
Jhat(rho)
figure = plot_state(projected, total)
figure.suptitle(f"continuous design, enhancement {continuous:.4f}")
plt.show()
# -

# The same for the binarized design.

# +
figure = plot_state(binary, total_binary, cellwise=True)
figure.suptitle(f"binarized design, enhancement {binarized:.4f}")
plt.show()
# -

# Enhancement against optimizer iteration, with the continuation stages marked.

# +
figure, axis = plt.subplots(figsize=(8, 4), layout="constrained")
axis.plot(range(1, len(history) + 1), history, marker="o", markersize=3)
for end, stage_beta in zip(stage_ends, BETA_STAGES):
    axis.axvline(end + 0.5, color="0.6", linestyle="--", linewidth=0.8)
    axis.annotate(
        rf"$\beta={stage_beta:g}$",
        (end, axis.get_ylim()[1]),
        textcoords="offset points",
        xytext=(-4, -12),
        ha="right",
        fontsize=8,
        color="0.4",
    )
axis.set_xlabel("optimizer iteration")
axis.set_ylabel("intensity enhancement at the focus")
axis.set_title("Metalens topology optimization")
axis.grid(alpha=0.3)
plt.show()
# -
