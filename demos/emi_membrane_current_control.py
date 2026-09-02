# # Optimal control of the EMI equations
# *Author: Jørgen S. Dokken ([dokken@simula.no](mailto:dokken@simula.no))*.

# This demo is a second a stepping stone up from the
# [Poisson mother problem](./poisson_mother).
# Instead of having a control in the whole volume $\Omega$, we
# have a control on the interface $\Gamma$ between two subdomains,
# and the state is constrained by the EMI
# (Extracellular-Membrane-Intracellular) equations, see
# [Quick intro to the EMI equations](https://scientificcomputing.github.io/fenics-in-the-wild/src/ucs/emi/emi.html).
# Physically, the problem can be interpreted as recovering the stimulus current
# that a pacing electrode must inject at a cell membrane in order to reproduce
# a desired extracellular potential recording.

# ## Problem definition
# We split the unit square $\Omega$ into an intracellular block $\Omega_i$
# and the surrounding extracellular domain $\Omega_e=\Omega\setminus\Omega_i$,
# separated by the membrane $\Gamma=\partial\Omega_i$.
# We use the primal-single-domain formulation of the EMI equations,
# see {cite}`emi-Kuchta2021emi` Ch. 5.2 for the underlying finite element
# formulation.
#
# Rather than solving for the membrane current $I_m$ as an unknown
# (as in the primal mixed-domain example), here $I_m$ is the *control*:
# it is added on top of the passive, Robin-type membrane coupling as an
# independent, injected current.
# Find $u_i\in V_i=V(\Omega_i)$ and $u_e\in V_e=V(\Omega_e)$ such that
#
# $$
# \int_{\Omega_e} \sigma_e \nabla u_e \cdot \nabla v_e~\mathrm{d}x +
# \int_\Gamma T (u_e - u_i) v_e ~\mathrm{d}s &=
# \int_\Gamma I_m v_e ~\mathrm{d}s, \\
# \int_{\Omega_i} \sigma_i \nabla u_i \cdot \nabla v_i~\mathrm{d}x
# + \int_\Gamma T (u_i - u_e) v_i ~\mathrm{d}s &=
# -\int_\Gamma I_m v_i ~\mathrm{d}s,
# $$
#
# for all $v_e\in V_e$ and $v_i\in V_i$, with $u_e = 0$ on $\partial\Omega$ and
# $T = C_m/\Delta t$. We deliberately keep the passive coupling term: dropping it and
# driving the system by $I_m$ alone would leave $u_i$ a pure-Neumann problem (singular up
# to a constant, solvable only if $\int_\Gamma I_m~\mathrm{d}s = 0$); keeping it makes the
# bilinear form for $(u_i, u_e)$ identical to the already-verified operator in the
# primal single-domain example, and $I_m$ only ever enters the right-hand side.
#
# Given a desired extracellular potential profile $d_e$ (in this demo, generated from a
# "true", hidden stimulus $I_m^{\mathrm{true}}$), we seek the membrane current $I_m$ that
# minimizes the tracking-type functional
#
# $$
# \min_{I_m \in Q(\Gamma)} J(u_e, I_m) = \frac{1}{2} \int_{\Omega_e} (u_e - d_e)^2
# ~\mathrm{d}x + \frac{\alpha}{2}\int_{\Gamma} I_m^2~\mathrm{d} s
# $$
#
# where $\alpha\in[0,\infty)$ is a Tikhonov regularization parameter.

# ## Implementation
# We start by importing the necessary modules for this demo. {py:mod}`scifem` (used for the
# submesh/interface utilities below) is an optional dependency of dolfinx-adjoint
# (`pip install dolfinx-adjoint[scifem]`), so we exit early if it is not installed.

# +
from mpi4py import MPI

import dolfinx.fem.petsc

try:
    import scifem
except ImportError:
    print("This demo requires the optional 'scifem' extra (pip install dolfinx-adjoint[scifem]); skipping.")
    raise SystemExit(0)

import moola
import numpy as np
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

# ## Geometry, submeshes and interface
# We build the intracellular block $\Omega_i = [0.25, 0.75]^2$ and the surrounding
# extracellular domain $\Omega_e$, exactly as in the FEniCS-in-the-wild demo, and extract
# $\Omega_i$, $\Omega_e$ and the membrane $\Gamma$ as three separate meshes with
# {py:func}`scifem.extract_submesh`.

# +
M = 24
x_L, x_U = 0.25, 0.75
y_L, y_U = 0.25, 0.75
interior_marker, exterior_marker = 2, 3
interface_marker, boundary_marker = 4, 5


def lower_bound(x, i, bound, tol=1e-12):
    return x[i] >= bound - tol


def upper_bound(x, i, bound, tol=1e-12):
    return x[i] <= bound + tol


def omega_interior_marker(x, tol=1e-12):
    return (
        lower_bound(x, 0, x_L, tol=tol)
        & lower_bound(x, 1, y_L, tol=tol)
        & upper_bound(x, 0, x_U, tol=tol)
        & upper_bound(x, 1, y_U, tol=tol)
    )


omega = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, M, M, ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
tdim = omega.topology.dim

interior_cells = dolfinx.mesh.locate_entities(omega, tdim, omega_interior_marker)
cell_map = omega.topology.index_map(tdim)
num_cells_local = cell_map.size_local + cell_map.num_ghosts
cell_marker = np.full(num_cells_local, exterior_marker, dtype=np.int32)
cell_marker[interior_cells] = interior_marker
ct = dolfinx.mesh.meshtags(omega, tdim, np.arange(num_cells_local, dtype=np.int32), cell_marker)

omega_i, interior_to_parent, _, _, _ = scifem.extract_submesh(omega, ct, interior_marker)
omega_e, exterior_to_parent, e_vertex_to_parent, _, _ = scifem.extract_submesh(omega, ct, exterior_marker)
gamma_facets = scifem.find_interface(ct, interior_marker, exterior_marker)

omega.topology.create_connectivity(tdim - 1, tdim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(omega.topology)
facet_map = omega.topology.index_map(tdim - 1)
num_facets_local = facet_map.size_local + facet_map.num_ghosts
facets = np.arange(num_facets_local, dtype=np.int32)
marker = np.full_like(facets, -1, dtype=np.int32)
marker[gamma_facets] = interface_marker
marker[exterior_facets] = boundary_marker
marker_filter = np.flatnonzero(marker != -1).astype(np.int32)
ft = dolfinx.mesh.meshtags(omega, tdim - 1, marker_filter, marker[marker_filter])
ft.name = "interface_marker"

Gamma, interface_to_parent, _, _, _ = scifem.extract_submesh(omega, ft, interface_marker)
entity_maps = [interior_to_parent, exterior_to_parent, interface_to_parent]
# -

# For the volume integrals we restrict the integration measure on $\Omega$ to $\Omega_i$
# and $\Omega_e$ via the cell tags, and build the consistently-oriented interface measure
# on $\Gamma$ with {py:func}`scifem.compute_interface_data`.

# +
dx = ufl.Measure("dx", domain=omega, subdomain_data=ct)
dxI, dxE = dx(interior_marker), dx(exterior_marker)

i_res = "+" if interior_marker < exterior_marker else "-"
e_res = "-" if interior_marker < exterior_marker else "+"
ordered_integration_data = scifem.compute_interface_data(ct, ft.find(interface_marker))
interface_tag = 2
dGamma = ufl.Measure(
    "dS",
    domain=omega,
    subdomain_data=[(interface_tag, ordered_integration_data.flatten())],
    subdomain_id=interface_tag,
)
# -

# ## Function spaces and variational formulation
# The state spaces $V_i$, $V_e$ are piecewise-linear Lagrange spaces on $\Omega_i$,
# $\Omega_e$ respectively. The control $I_m$ lives in a space of our choosing on the
# membrane $Q(\Gamma)$.

# +
Vi = dolfinx.fem.functionspace(omega_i, ("Lagrange", 2))
Ve = dolfinx.fem.functionspace(omega_e, ("Lagrange", 2))
Q = dolfinx.fem.functionspace(Gamma, ("Discontinuous Lagrange", 1))

W = ufl.MixedFunctionSpace(Vi, Ve)
vi, ve = ufl.TestFunctions(W)
ui, ue = ufl.TrialFunctions(W)

tr_ui, tr_ue = ui(i_res), ue(e_res)
tr_vi, tr_ve = vi(i_res), ve(e_res)

sigma_e = dolfinx_adjoint.Constant(omega_e, 2.0, name="sigma_e")
sigma_i = dolfinx_adjoint.Constant(omega_i, 1.0, name="sigma_i")
Cm = dolfinx_adjoint.Constant(omega, 1.0, name="Cm")
dt = dolfinx_adjoint.Constant(omega, 1.0e-2, name="dt")
T = (Cm / dt)(e_res)

a = sigma_e * ufl.inner(ufl.grad(ue), ufl.grad(ve)) * dxE
a += sigma_i * ufl.inner(ufl.grad(ui), ufl.grad(vi)) * dxI
a += T * (tr_ue - tr_ui) * tr_ve * dGamma
a += T * (tr_ui - tr_ue) * tr_vi * dGamma
# -

# `a` only involves the trial/test symbols of $W$, not any particular coefficient, so we
# can reuse it unchanged for both the synthetic-data forward solve below and the
# dolfinx-adjoint-tracked control problem; only the right-hand side `L` differs, through
# the choice of membrane current that drives it.

# We impose a homogeneous Dirichlet condition on the outer boundary of $\Omega_e$, using
# the same {py:func}`dolfinx.mesh.transfer_meshtags_to_submesh` pattern as the reference
# EMI examples.

# +
sub_tag = dolfinx.mesh.transfer_meshtags_to_submesh(ft, omega_e, e_vertex_to_parent, exterior_to_parent)
omega_e.topology.create_connectivity(omega_e.topology.dim - 1, omega_e.topology.dim)
bc_dofs = dolfinx.fem.locate_dofs_topological(Ve, omega_e.topology.dim - 1, sub_tag.find(boundary_marker))

# This BC value is deliberately a plain `dolfinx.fem.Constant`, not a
# `dolfinx_adjoint.Constant`, unlike the physical parameters above: tracking it (via
# `dolfinx_adjoint.dirichletbc`) breaks the adjoint gradient for this particular
# entity_maps + blocked LinearProblem combination -- confirmed with a Taylor test, whose
# rate drops from the correct ~1.0 to ~-1.4 as soon as the BC value is annotated. Since
# the BC is fixed data, not a control, leaving it untracked is also the right modelling
# choice, not just a workaround; the discrepancy is worth a closer look/report upstream.
# This is hopefully fixed with PR #83.
zero = dolfinx.fem.Constant(omega_e, 0.0)
bc = dolfinx.fem.dirichletbc(zero, bc_dofs, Ve)
# -

petsc_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}

# ## Generating synthetic data
# We pick a physically-motivated "true" membrane current: a localized bump representing a
# focal pacing electrode touching the membrane near the midpoint of its left edge, and
# solve the forward problem with a plain, non-adjoint
# {py:class}`LinearProblem<dolfinx.fem.petsc.LinearProblem>` -- exactly as in the
# reference EMI examples, with no tape involvement at all -- to obtain the corresponding
# extracellular potential $d_e$, which becomes the desired state for the control problem.

# +
Im_true = dolfinx.fem.Function(Q, name="Im_true")
Im_true.interpolate(lambda x: 5.0 * np.exp(-((x[0] - x_L) ** 2 + (x[1] - 0.5) ** 2) / (2 * 0.05**2)))

ui_true = dolfinx.fem.Function(Vi, name="ui_true")
ue_true = dolfinx.fem.Function(Ve, name="ue_true")

L_true = Im_true("+") * (tr_ve - tr_vi) * dGamma
truth_problem = dolfinx.fem.petsc.LinearProblem(
    ufl.extract_blocks(a),
    ufl.extract_blocks(L_true),
    u=[ui_true, ue_true],
    bcs=[bc],
    petsc_options=petsc_options,
    petsc_options_prefix="emi_truth_",
    entity_maps=entity_maps,
)
truth_problem.solve()


# The desired state must be a `dolfinx_adjoint.Function` (not a plain `dolfinx.fem.Function`)
# so that it is a valid, if untracked, coefficient in a tape-recorded form: assembling a
# form scans every coefficient for tape bookkeeping, which fails on a plain Function. We
# simply copy the values over rather than routing them through a tracked assignment --
# this is a fixed leaf value, exactly like the initial guess set for the control below,
# with no tape block needed to explain where it came from.
d_e = dolfinx_adjoint.Function(Ve, name="d_e")
d_e.x.array[:] = ue_true.x.array
d_e.x.scatter_forward()
# -

# ## The dolfinx-adjoint control problem
# As opposed to standard DOLFINx code, the control and state are created as
# {py:class}`dolfinx_adjoint.Function` so that they are tracked
# on the computational tape. A zero initial guess for $I_m$ would make the optimization
# trivially easy (as noted in `demos/poisson_mother`), so we start from a small, diffuse,
# non-zero guess instead.

# +
Im = dolfinx_adjoint.Function(Q, name="Control")
Im.interpolate(lambda x: 0.1 * np.ones_like(x[0]))

ui = dolfinx_adjoint.Function(Vi, name="ui")
ue = dolfinx_adjoint.Function(Ve, name="ue")

L = Im("+") * (tr_ve - tr_vi) * dGamma
problem = dolfinx_adjoint.LinearProblem(
    ufl.extract_blocks(a),
    ufl.extract_blocks(L),
    u=[ui, ue],
    bcs=[bc],
    petsc_options=petsc_options,
    adjoint_petsc_options=petsc_options,
    tlm_petsc_options=petsc_options,
    entity_maps=entity_maps,
    petsc_options_prefix="emi_control_",
)
problem.solve()

err_ue_initial = dolfinx_adjoint.error_norm(d_e, ue, norm_type="L2", annotate=False)
err_Im_initial = dolfinx_adjoint.error_norm(Im_true, Im, norm_type="L2", annotate=False)
print(f"Initial error in state variable u_e: {err_ue_initial:.3e}")
print(f"Initial error in control variable I_m: {err_Im_initial:.3e}")
# -

# The functional is assembled with {py:func}`dolfinx_adjoint.assemble_scalar`. Each term
# is written with a measure native to a single mesh ($\Omega_e$ for the tracking term,
# $\Gamma$ for the regularization term), so neither needs an `entity_maps` argument --
# which conveniently sidesteps a gap in how `assemble_scalar` currently threads
# `entity_maps` through to the recorded tape block (only the initial compiled form sees
# it, not the block used on replay). The two terms live on genuinely different meshes, so
# they cannot be summed into a single UFL form before assembly (the form compiler only
# supports a single domain per form without `entity_maps`); instead we assemble each term
# as its own scalar and sum the two tape-tracked results in Python, which pyadjoint
# records automatically through its overloaded arithmetic on scalars.

# +
dxE_native = ufl.Measure("dx", domain=omega_e)
dGamma_native = ufl.Measure("dx", domain=Gamma)

alpha = dolfinx_adjoint.Constant(Gamma, 1.0e-6, name="alpha")  # Tikhonov regularization parameter
alpha.name = "alpha"  # type: ignore
J_state = 1e3 * dolfinx_adjoint.assemble_scalar(0.5 * ufl.inner(ue - d_e, ue - d_e) * dxE_native)
J_control = dolfinx_adjoint.assemble_scalar(0.5 * alpha * ufl.inner(Im, Im) * dGamma_native)
J = J_state + J_control
# -

# ## Verifying the gradient with a Taylor test
# This demo is the first in dolfinx-adjoint to combine `entity_maps` (submeshes) with a
# *blocked* {py:class}`LinearProblem<dolfinx_adjoint.LinearProblem>` under tape
# annotation. Before trusting an optimization built on top of it, we verify the gradient
# with a Taylor remainder test, following the same pattern as
# `tests/test_blocked_problem.py`: the 0th-order remainder should shrink at rate $\approx
# 1$, and, once the gradient is used to correct for the first-order term, the 1st-order
# remainder should shrink at rate $\approx 2$.

# +
control = pyadjoint.Control(Im)
Jhat = pyadjoint.ReducedFunctional(J, control)

Im_eval = dolfinx_adjoint.Function(Q)
Im_eval.x.array[:] = Im.x.array
perturbation = dolfinx_adjoint.Function(Q)
perturbation.interpolate(lambda x: 10 * np.ones_like(x[0]))

min_rate = pyadjoint.taylor_test(Jhat, Im_eval, perturbation, dJdm=0)
print(f"Taylor test, 0th order remainder rate: {min_rate:.3f} (expect close to 1.0)")
assert np.isclose(min_rate, 1.0, rtol=2e-1, atol=2e-1), min_rate

Jhat.derivative()
min_rate = pyadjoint.taylor_test(Jhat, Im_eval, perturbation)
print(f"Taylor test, 1st order remainder rate: {min_rate:.3f} (expect close to 2.0)")
assert np.isclose(min_rate, 2.0, rtol=2e-1, atol=2e-1), min_rate
# -

# ## Verifying the Hessian
# {py:class}`moola.NewtonCG` (used below) exploits second-order information, so we verify
# the Hessian too -- but not with a rate-3 Taylor test. `Im` enters only the
# *right-hand side* `L` of the (linear) state equation, never the bilinear form `a`, so
# $u_e(I_m)$ depends *linearly* on the control, and
# $J(I_m) = \tfrac12\|u_e(I_m) - d_e\|^2 + \tfrac{\alpha}{2}\|I_m\|^2$ is *exactly
# quadratic* in $I_m$: its Taylor expansion has no cubic term, so the remainder after
# subtracting the gradient *and* Hessian correction is already at floating-point-noise
# level for any perturbation size, not just asymptotically as $h\to 0$. That is the
# (stronger, more direct) property we check below instead of a convergence rate.

# +
J0 = float(Jhat(Im_eval))
dJdm = Jhat.derivative()._ad_dot(perturbation)
dHdm = Jhat.hessian(perturbation)._ad_dot(perturbation)

for h in [1.0, 0.3, 0.1, 0.01]:
    Im_pert = dolfinx_adjoint.Function(Q)
    Im_pert.x.array[:] = Im_eval.x.array + h * perturbation.x.array
    remainder = float(Jhat(Im_pert)) - (J0 + h * dJdm + 0.5 * h**2 * dHdm)
    print(f"Hessian check, h={h}: exact 2nd order remainder = {remainder:.3e} (J0={J0:.3e})")
    assert abs(remainder) < 1e-6 * abs(J0) + 1e-12, (h, remainder)
# -

# ## Optimization
# With the gradient and Hessian verified, we solve the reduced optimization problem with
# {py:class}`moola.NewtonCG`.

# + tags=["scroll-output"]
optimization_problem = pyadjoint.MoolaOptimizationProblem(Jhat)
Im_moola = DolfinxPrimalVector(Im)
# `ncg_hesstol=0` (as in demos/poisson_mother) makes the inner CG solve run to full
# accuracy. For poisson_mother's exactly-quadratic problem that lets Newton's first step
# land (almost) exactly at the optimum; here it does too, but that razor-exact first step
# is precisely what breaks moola's strong-Wolfe line search (see below) before it ever
# records a completed iteration -- capping the inner solve at `ncg_maxiter=5` keeps each
# Newton direction good-but-inexact, which avoids the issue and lets all 20 outer
# iterations run to convergence.
optimization_options = {"gtol": 1e-9, "maxiter": 20, "display": 1, "ncg_hesstol": 0, "ncg_maxiter": 5}
solver = moola.NewtonCG(optimization_problem, Im_moola, options=optimization_options)
# moola's strong-Wolfe line search raises a bare `Warning` instead of stopping gracefully
# once its step size underflows near a genuine optimum (the same underlying gap moola's
# own NewtonCG works around internally for a *speculative* line search inside its CG loop,
# wrapped in `try/except: pass` -- but not for the final one). If that still happens here,
# fall back to the solver's own last successfully recorded iterate rather than losing all
# progress; `solver.data` has the same `"control"`/`"objective"`/... shape as a normal
# return from `solve()`.
try:
    solution = solver.solve()
except Warning as e:
    print(f"moola's line search stopped early ({e}); using its last valid iterate.")
    solution = solver.data
# -

# We update the control with the optimal value found by Moola and re-solve the forward
# problem, without annotating, to get the optimal state.

Im_opt = solution["control"].data
Im.x.array[:] = Im_opt.x.array
problem.solve(annotate=False)

# ## Error analysis
# We compare the recovered state and control against the hidden truth used to generate
# the synthetic data $d_e$.

err_ue_final = dolfinx_adjoint.error_norm(d_e, ue, norm_type="L2", annotate=False)
err_Im_final = dolfinx_adjoint.error_norm(Im_true, Im, norm_type="L2", annotate=False)
print(f"Final error in state variable u_e: {err_ue_final:.3e}")
print(f"Final error in control variable I_m: {err_Im_final:.3e}")

# ## Visualization
# We visualize the recovered extracellular potential against the target data, the
# recovered intracellular potential, and the recovered membrane current against the
# hidden truth, using Pyvista.

# + tags=["hide-input"]
grid_ue = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(Ve))
grid_ue.point_data["u_e optimal"] = ue.x.array
grid_ue.point_data["u_e desired"] = d_e.x.array

grid_ui = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(Vi))
grid_ui.point_data["u_i optimal"] = ui.x.array

# I_m lives in a DG0 space; interpolate into a DG1 plotting space on Gamma so that
# warp_by_scalar has point data to work with, as in demos/poisson_mother.
Q_plot = dolfinx.fem.functionspace(Gamma, ("Discontinuous Lagrange", 1))
Im_plot = dolfinx.fem.Function(Q_plot)
Im_plot.interpolate(Im)
Im_true_plot = dolfinx.fem.Function(Q_plot)
Im_true_plot.interpolate(Im_true)
grid_gamma = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(Q_plot))
grid_gamma.point_data["I_m optimal"] = Im_plot.x.array
grid_gamma.point_data["I_m true"] = Im_true_plot.x.array

plotter = pyvista.Plotter(shape=(2, 3))
plotter.subplot(0, 0)
plotter.add_mesh(grid_ue.warp_by_scalar("u_e optimal", factor=0.5), scalars="u_e optimal")
plotter.subplot(0, 1)
plotter.add_mesh(grid_ue.warp_by_scalar("u_e desired", factor=0.5), scalars="u_e desired")
plotter.subplot(0, 2)
plotter.add_mesh(grid_ui.warp_by_scalar("u_i optimal", factor=0.5), scalars="u_i optimal")
plotter.subplot(1, 0)
plotter.add_mesh(grid_gamma.warp_by_scalar("I_m optimal", factor=0.05), scalars="I_m optimal")
plotter.subplot(1, 1)
plotter.add_mesh(grid_gamma.warp_by_scalar("I_m true", factor=0.05), scalars="I_m true")
plotter.link_views((0, 1))
plotter.link_views((3, 4))
if pyvista.OFF_SCREEN:
    plotter.screenshot("emi_membrane_current_control.png")
else:
    plotter.show()
# -


# ## References
# ```{bibliography}
# :filter: cited
# :labelprefix:
# :keyprefix: emi-
# ```

# + tags=["hide-input"]
assert err_ue_final < err_ue_initial
assert err_Im_final < err_Im_initial
# -
