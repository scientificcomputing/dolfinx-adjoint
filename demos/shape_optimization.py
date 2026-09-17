# # Shape optimization: rounding a square into a disk
# *Section author: Jørgen S. Dokken ([dokken@simula.no](mailto:dokken@simula.no))*.

# The controls in the other demos are *fields* — a source term, a boundary value — posed on a
# fixed domain. This one differentiates with respect to the domain itself.

# ## The shape derivative
#
# Every form carries its dependence on the geometry through
# {py:class}`ufl.SpatialCoordinate`. Differentiating a form with respect to that coordinate,
# in a direction $s$, is exactly the shape derivative:
#
# $$
# \mathrm{d}J(\Omega)[s] = \lim_{\varepsilon \to 0}
#   \frac{J\big((\mathrm{id} + \varepsilon s)(\Omega)\big) - J(\Omega)}{\varepsilon},
# $$
#
# and UFL builds it for us — including the terms that come from the measure transforming
# with the domain — when we differentiate with respect to
# `ufl.SpatialCoordinate(mesh)`.
#
# So a shape optimization is an ordinary optimization whose control is a **displacement
# field** $s$, applied to the mesh with {py:func}`dolfinx_adjoint.move`. That function is the
# annotating counterpart of `scifem.mesh.move`: it records the move on the tape, after which
# every form posed on the mesh depends differentiably on $s$.

# ## The problem
#
# We maximize the *torsional rigidity* of a two-dimensional bar of fixed cross-sectional area.
# Let $u$ solve the Saint-Venant torsion problem
#
# $$
# \begin{align}
# -\Delta u &= 1  && \text{in } \Omega, \\
# u &= 0 && \text{on } \partial\Omega,
# \end{align}
# $$
#
# and let the rigidity be $T(\Omega) = \int_\Omega u \,\mathrm{d}x$. Among all shapes of a
# given area, the disk maximizes $T$ — a classical result of Saint-Venant, proved by Pólya
# {cite}`polya1948`. Starting from the unit square, the optimizer should therefore round it
# off towards a disk, which makes the answer easy to recognize.
#
# We minimize
#
# $$
# J(\Omega) = -\int_\Omega u \,\mathrm{d}x + \mu \left(|\Omega| - 1\right)^2,
# $$
#
# the penalty term holding the area at its initial value so the bar cannot simply grow.

# ## Implementation

# +
from mpi4py import MPI

import dolfinx
import dolfinx.fem.petsc
import matplotlib.pyplot as plt
import matplotlib.tri
import numpy as np
import pyadjoint
import ufl

import dolfinx_adjoint

# -

# A direct solver, used for the state equation, for the adjoint, and for the Riesz map below.

lu_options = {"ksp_type": "preonly", "pc_type": "lu"}

# The control is a displacement in the mesh's **geometry function space**. Its dofs are the
# mesh's own coordinate nodes, which is what lets an assembled shape derivative and an applied
# displacement be the same vector. A displacement in any other space has to be interpolated
# into this one first, with {py:func}`dolfinx_adjoint.interpolate`, so that the interpolation
# is itself recorded.

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 20, 20)


def triangulation(domain: dolfinx.mesh.Mesh) -> matplotlib.tri.Triangulation:
    """A snapshot of ``domain``'s current triangles, for plotting.

    Takes a copy of the coordinates: the mesh is moved in place, so a triangulation sharing its
    arrays would silently follow it and the "initial" mesh would plot on top of the optimized
    one.
    """
    cells = np.asarray(domain.geometry.dofmaps[0]).reshape(-1, 3)
    x = domain.geometry.x.copy()
    return matplotlib.tri.Triangulation(x[:, 0], x[:, 1], cells)


initial_triangulation = triangulation(mesh)

S = dolfinx_adjoint.geometry_function_space(mesh)
s = dolfinx_adjoint.Function(S, name="displacement")

# Moving by a zero displacement changes nothing, but it puts the mesh on the tape: from here
# on, every form posed on `mesh` depends on `s`.

dolfinx_adjoint.move(mesh, s)

# The state equation is an ordinary {py:class}`dolfinx_adjoint.LinearProblem`. Nothing about it
# mentions the geometry — the dependence is implicit in `ufl.dx` and in the coordinates the
# basis functions are evaluated at.

# +
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)

tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, tdim)
boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, boundary_facets)
bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), boundary_dofs, V)

problem = dolfinx_adjoint.LinearProblem(
    ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx,
    ufl.inner(dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0)), v) * ufl.dx,
    bcs=[bc],
    petsc_options=lu_options,
    petsc_options_prefix="torsion_",
    adjoint_petsc_options=lu_options,
    tlm_petsc_options=lu_options,
)
uh = problem.solve()
# -

# The functional, and the area penalty that keeps it honest.

# +
area_penalty = 10.0
rigidity = dolfinx_adjoint.assemble_scalar(ufl.inner(uh, 1.0) * ufl.dx)
area = dolfinx_adjoint.assemble_scalar(1 * ufl.dx(domain=mesh))
J = -rigidity + area_penalty * (area - 1.0) ** 2

print(f"Initial torsional rigidity: {float(rigidity):.6f}   area: {float(area):.6f}")
# -

# ## The reduced functional
#
# The control is the displacement, never the mesh: the mesh is the intermediate value linking
# the two. We attach an $H^1$ Riesz map, so that `derivative(apply_riesz=True)` returns a
# *smooth* displacement field rather than the raw dual vector. That matters more here than for
# a field control — the raw shape derivative is concentrated on the boundary, and following it
# directly would tear the mesh apart within a few steps.

riesz_map = {"riesz_representation": "H1", "petsc_options": lu_options}
Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s, riesz_map=riesz_map))

# A Taylor test confirms the shape derivative before we rely on it. The direction has to keep
# the mesh untangled at every step the test takes, which a smooth, modest displacement does.

# +
direction = dolfinx_adjoint.Function(S)
direction.interpolate(lambda x: np.vstack((np.sin(np.pi * x[0]), np.sin(np.pi * x[1]))))
direction.x.array[:] *= 10
rates = pyadjoint.taylor_to_dict(Jhat, s, direction)
print(rates)

print(
    f"Taylor rates: R0 {min(rates['R0']['Rate']):.3f}, "
    f"R1 {min(rates['R1']['Rate']):.3f}, R2 {min(rates['R2']['Rate']):.3f}"
)
assert min(rates["R0"]["Rate"]) > 0.9
assert min(rates["R1"]["Rate"]) > 1.9
assert min(rates["R2"]["Rate"]) > 2.9
# -

# ## Steepest descent
#
# Each step follows the smoothed shape gradient, rescaled so that no node moves further than
# `max_move`. Normalizing the *step* rather than trusting the gradient's magnitude is what
# keeps the mesh valid: it bounds how far the geometry can travel before we look at it again.
#
# A step is accepted only if it decreases $J$ *and* leaves every cell with the orientation it
# started with. A cell whose Jacobian determinant has changed sign has turned inside out, and
# the domain — along with every integral over it — is meaningless from then on. The check is
# for a *change* of sign, not for a positive determinant: DOLFINx does not orient cells
# consistently, so the absolute sign carries no information.


# +
def cell_orientations(domain: dolfinx.mesh.Mesh) -> np.ndarray:
    """The sign of every locally owned cell's Jacobian determinant."""
    DG0 = dolfinx.fem.functionspace(domain, ("DG", 0))
    detJ = dolfinx.fem.Function(DG0)
    detJ.interpolate(dolfinx.fem.Expression(ufl.JacobianDeterminant(domain), DG0.element.interpolation_points))
    return np.sign(detJ.x.array[: DG0.dofmap.index_map.size_local]).copy()


reference_orientations = cell_orientations(mesh)

current = dolfinx_adjoint.Function(S)
current_J = Jhat(current)
max_move_cap = 0.01
max_move = max_move_cap

for iteration in range(1, 151):
    gradient = Jhat.derivative(apply_riesz=True)
    largest = mesh.comm.allreduce(np.abs(gradient.x.array).max(), op=MPI.MAX)
    if largest == 0.0:
        break

    trial = dolfinx_adjoint.Function(S)
    trial.x.array[:] = current.x.array - (max_move / largest) * gradient.x.array
    trial_J = Jhat(trial)

    untangled = np.array_equal(cell_orientations(mesh), reference_orientations)
    if trial_J < current_J and untangled:
        current, current_J = trial, trial_J
        # Let the step recover after a rejection, or the first bad step would throttle the
        # whole run: `max_move` only ever halves otherwise.
        max_move = min(1.2 * max_move, max_move_cap)
    else:
        # Reject the step, put the mesh back where it was, and try a shorter one.
        max_move *= 0.5
        Jhat(current)
        if max_move < 1e-5:
            break

    if iteration % 25 == 0:
        print(f"iteration {iteration:3d}   J = {current_J: .8f}   max_move = {max_move:g}")

Jhat(current)
# -

# ## The result
#
# `Jhat(current)` above leaves the tape — and so the mesh — at the optimized shape. Note that
# `rigidity` and `area` are the values *recorded* when the tape was built: replaying the tape
# does not, and cannot, write back into those Python floats. The replayed value of a scalar
# lives on its block variable, which is where we read it from.

final_rigidity = rigidity.block_variable.checkpoint
final_area = area.block_variable.checkpoint
print(f"Final torsional rigidity:   {final_rigidity:.6f}   area: {final_area:.6f}")
print(f"Rigidity gained: {100 * (final_rigidity / float(rigidity) - 1):.1f}%")

# The corners are what move: a disk of unit area has radius $1/\sqrt{\pi} \approx 0.564$,
# against the unit square's corner distance of $\sqrt{2}/2 \approx 0.707$.

# +
coordinates = mesh.geometry.x
radii = np.sqrt((coordinates[:, 0] - 0.5) ** 2 + (coordinates[:, 1] - 0.5) ** 2)
furthest = mesh.comm.allreduce(radii.max(), op=MPI.MAX)
print(f"Furthest node from the centre: {furthest:.4f}  (square: 0.707, disk of unit area: 0.564)")
# -

# Both meshes go in *one* pair of axes, initial in blue and optimized in red. Drawn side by
# side in separate panels each would autoscale to its own shape, and the rounding -- which is a
# few percent of the domain -- would be invisible.

# +
figure, axes = plt.subplots(figsize=(6, 6))
axes.triplot(initial_triangulation, color="tab:blue", linewidth=0.4, label="Initial mesh")
axes.triplot(triangulation(mesh), color="tab:red", linewidth=0.4, label="Optimized mesh")
axes.set_aspect("equal")
axes.axis("off")
axes.legend(
    handles=[
        plt.Line2D([], [], color="tab:blue", label="Initial mesh"),
        plt.Line2D([], [], color="tab:red", label="Optimized mesh"),
    ],
    loc="upper center",
    ncols=2,
)
figure.savefig("shape_optimization.png", dpi=200, bbox_inches="tight")
# -

# ```{bibliography}
# :filter: cited and ({"demos/shape_optimization"} >= docnames)
# ```
