# # Topology Optimization of a Linear-Elastic Cantilever (SIMP)
# *Section author: Jørgen S. Dokken ([dokken@simula.no](mailto:dokken@simula.no))*.
#
# This demo reproduces the 3D cantilever-beam topology optimization problem from
# [Pasteur Labs' Mosaic benchmark suite][mosaic-results], using `dolfinx_adjoint` in
# place of the original FEniCS/dolfin-adjoint solver. The mesh, material model,
# boundary conditions and objective follow Mosaic's own reference implementation
# exactly:
#
# - [`fenics-structural/tesseract_api.py`][mosaic-tesseract] — the FEniCS/dolfin-adjoint
#   physics kernel this demo's weak form and compliance functional mirror.
# - [`benchmarks/problems/structural_mesh/{physics,optimization,config}.py`][mosaic-benchmark] —
#   the cantilever boundary-condition builder and the canonical run configuration
#   (mesh resolution, material parameters, volume-fraction target) reproduced below.
#
# The optimizer differs from Mosaic's own Adam-based recipe: this demo uses
# `scipy.optimize.minimize(method="trust-constr")`, driven by a
# `pyadjoint.ReducedFunctionalNumPy`, which — unlike Mosaic's Adam + manual clipping —
# enforces the SIMP density bounds natively via `bounds=` while also using exact
# Hessian-vector products from the adjoint/tangent-linear system (`hessp=`), something
# Mosaic's first-order Adam recipe does not use at all.
#
# The per-iteration density animation uses the pyvista GIF-writing pattern from
# [`dolfinx-tutorial/chapter2/amr.py`][tutorial-amr].
#
# [mosaic-results]: https://docs.pasteurlabs.ai/projects/mosaic/stable/docs/results_structural_mesh.html
# [mosaic-tesseract]: https://github.com/pasteurlabs/mosaic/blob/main/mosaic/tesseracts/structural-mesh/fenics-structural/tesseract_api.py
# [mosaic-benchmark]: https://github.com/pasteurlabs/mosaic/tree/main/mosaic/benchmarks/problems/structural_mesh
# [tutorial-amr]: https://github.com/jorgensd/dolfinx-tutorial/blob/c752d010ddbabf98b3e71e7a562f07ad7a738c26/chapter2/amr.py#L241

# ## Problem definition
#
# We minimize the structural compliance $C = \mathbf F^\top \mathbf u$ of a linear
# elastic body $\Omega$ subject to
#
# $$
# -\operatorname{div}(\sigma(\mathbf u)) = 0 \quad \text{in } \Omega, \qquad
# \mathbf u = 0 \quad \text{on } \Gamma_D, \qquad
# \sigma(\mathbf u)\cdot n = \mathbf t \quad \text{on } \Gamma_N,
# $$
#
# with a SIMP-interpolated, density-dependent stiffness
#
# $$
# E(\rho) = E_\min + (E_\max - E_\min)\,\rho^p, \qquad E_\min = x_\min E_\max,
# $$
#
# and a soft volume-fraction penalty added to the objective,
#
# $$
# \min_{x_\min \le \rho \le 1} \; J(\rho) = C(\rho) + w\,(\bar\rho - v_\mathrm{frac})^2,
# \qquad \bar\rho = \frac{1}{|\Omega|}\int_\Omega \rho \, \mathrm{d}x.
# $$
#
# The domain is a $[0,2]\times[0,1]\times[0,1]$ cantilever beam meshed with HEX8
# elements ($16\times2\times8$ elements), clamped at $x=0$. A prescribed total force
# $F_\mathrm{total}$ is applied at $x=2$, either as a uniform downward traction over
# the whole face, or as a concentrated upward traction on a single corner patch — both
# cases are run below.

# ## Implementation
#
# We start by importing the necessary modules.

# +
import time

from mpi4py import MPI

import dolfinx
import numpy as np
import pandas
import pyadjoint
import pyvista
import scipy.optimize
import ufl

import dolfinx_adjoint

# -

# We configure Pyvista for rendering.

# + tags=["hide-input"]
pyvista.set_jupyter_backend("html")
# -

# ## Material and problem parameters
#
# These match Mosaic's canonical `optimization/topopt` run exactly (see
# `mosaic/benchmarks/problems/structural_mesh/config.py`).

# +
Lx, Ly, Lz = 2.0, 1.0, 1.0
F_total = 1.0

E_max = 70_000.0  # Young's modulus of the fully solid material [MPa]
nu = 0.3  # Poisson's ratio
x_min = 1.0e-3  # void stiffness ratio (E_min = x_min * E_max) and density lower bound
penal = 3.0  # SIMP penalization exponent

v_frac = 0.5  # target volume fraction
penalty_weight = 50.0  # volume-fraction penalty weight
# -


# ## Mesh and boundary conditions
#
# `build_mesh_and_bcs` mirrors Mosaic's own `_cantilever_bcs`
# (`mosaic/benchmarks/problems/structural_mesh/physics.py`): all nodes at $x=0$ are
# clamped, and the load on the $x=L_x$ face is either a uniform downward traction, or
# a concentrated upward traction on the single corner patch $y\in[0,\Delta y]$,
# $z\in[0,\Delta z]$. `dolfinx.mesh.locate_entities_boundary` selects a facet only when
# *all* of its vertices satisfy the marker, matching the "all vertices in group"
# semantics Mosaic's own node-mask approach uses.


def build_mesh_and_bcs(nx: int, ny: int, nz: int, corner_load: bool):
    """Build the cantilever mesh, Dirichlet facets, Neumann measure and traction."""
    msh = dolfinx.mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0, 0.0]), np.array([Lx, Ly, Lz])],
        (nx, ny, nz),
        dolfinx.mesh.CellType.hexahedron,
    )
    fdim = msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, msh.topology.dim)
    tol = 1.0e-6 * max(Lx, Ly, Lz)

    left_facets = dolfinx.mesh.locate_entities_boundary(msh, fdim, lambda x: x[0] < tol)

    if corner_load:
        dy, dz = Ly / ny, Lz / nz
        right_facets = dolfinx.mesh.locate_entities_boundary(
            msh, fdim, lambda x: (x[0] > Lx - tol) & (x[1] < dy + tol) & (x[2] < dz + tol)
        )
        traction = (0.0, 0.0, F_total / (dy * dz))  # concentrated, upward
    else:
        right_facets = dolfinx.mesh.locate_entities_boundary(msh, fdim, lambda x: x[0] > Lx - tol)
        traction = (0.0, 0.0, -F_total / (Ly * Lz))  # uniform, downward

    right_facets = np.sort(right_facets)
    facet_tags = dolfinx.mesh.meshtags(msh, fdim, right_facets, np.full(len(right_facets), 1, dtype=np.int32))
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    return msh, left_facets, ds, np.array(traction, dtype=dolfinx.default_scalar_type)


# ## Per-iteration density visualization
#
# We record the density field once per outer optimizer iteration into a GIF, using the
# pyvista pattern from `dolfinx-tutorial/chapter2/amr.py`. Since a SIMP density field
# lives in a `("DG", 0)` space, its dof array maps 1-to-1 onto the local cell
# numbering, so it can be attached directly as `cell_data` on a grid built from the
# mesh itself (not from the function space).


def make_gif_writer(msh: dolfinx.mesh.Mesh, rho: dolfinx_adjoint.Function, gif_path: str):
    """Return (plotter, callback) that appends one frame per optimizer iteration."""
    plotter = pyvista.Plotter(off_screen=True)
    plotter.open_gif(gif_path, fps=10)

    def write_frame():
        grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(msh))
        grid.cell_data["rho"] = rho.x.array
        actor = plotter.add_mesh(grid, scalars="rho", clim=(0.0, 1.0), cmap="viridis", show_edges=False)
        plotter.view_isometric()
        plotter.write_frame()
        plotter.remove_actor(actor)

    def gif_callback(intermediate_result):
        rho.x.array[:] = intermediate_result.x
        rho.x.scatter_forward()
        write_frame()

    write_frame()  # record the initial, uniform density as frame 0
    return plotter, gif_callback


# ## Forward model, gradient verification and optimization
#
# `run_topopt` builds the forward model once (SIMP weak form, linear solve, compliance
# + volume-penalty objective), verifies the resulting adjoint gradient and Hessian with
# 0th/1st/2nd-order Taylor tests, times a single tape recompute and a single adjoint
# solve, then optimizes the density with a bound- and curvature-aware scipy solver.
#
# ```{note}
# A `pyadjoint.ReducedFunctional` *replays* the tape it was built from — it does not
# re-execute Python code. The forward model below therefore calls `problem.solve()`
# with annotation on exactly once; every subsequent evaluation goes through
# `Jhat(...)`/`Jhat.derivative()`/`Jhat.hessian(...)`. `problem` is also kept alive as a
# local variable for as long as `Jhat` is used, since letting it go out of scope would
# silently fall back to a much slower solver-rebuild path.
# ```


def run_topopt(corner_load: bool, nx: int = 16, ny: int = 2, nz: int = 8) -> dict:
    case_name = "corner_load" if corner_load else "full_face_load"
    print(f"\n=== Running topology optimization: {case_name} ===")

    pyadjoint.get_working_tape().clear_tape()

    msh, left_facets, ds, traction = build_mesh_and_bcs(nx, ny, nz, corner_load)
    fdim = msh.topology.dim - 1

    # ### Function spaces and control
    V = dolfinx.fem.functionspace(msh, ("Lagrange", 1, (3,)))  # displacement
    Q = dolfinx.fem.functionspace(msh, ("DG", 0))  # density

    rho = dolfinx_adjoint.Function(Q, name="Density")
    rho.x.array[:] = v_frac
    rho.x.scatter_forward()

    uh = dolfinx_adjoint.Function(V, name="Displacement")

    # ### SIMP material law and weak form
    E_min = x_min * E_max
    E = E_min + (E_max - E_min) * rho**penal
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return lam * ufl.tr(eps(w)) * ufl.Identity(3) + 2 * mu * eps(w)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    t = dolfinx.fem.Constant(msh, traction)
    L = ufl.inner(t, v) * ds(1)

    # ### Dirichlet BC and forward solve (once)
    bdofs = dolfinx.fem.locate_dofs_topological(V, fdim, left_facets)
    bc = dolfinx.fem.dirichletbc(np.zeros(3, dtype=dolfinx.default_scalar_type), bdofs, V)

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
        tlm_petsc_options=petsc_options,
    )

    t_forward_start = time.perf_counter()
    problem.solve()
    forward_time = time.perf_counter() - t_forward_start
    print(f"[{case_name}] initial forward solve: {forward_time:.4e} s")

    # ### Objective: compliance + volume-fraction penalty
    compliance = dolfinx_adjoint.assemble_scalar(ufl.action(L, uh))
    vol_frac = dolfinx_adjoint.assemble_scalar(rho * ufl.dx) / (Lx * Ly * Lz)
    J = compliance + penalty_weight * (vol_frac - v_frac) ** 2

    control = pyadjoint.Control(rho)
    Jhat = pyadjoint.ReducedFunctional(J, control)

    # ### Gradient and Hessian verification: 0th, 1st and 2nd-order Taylor tests
    #
    # Matches the convention used throughout `dxa/tests/` (e.g.
    # `tests/test_linear_solver.py`, `demos/demo_nonmatching_grids.py`) rather than the
    # (unused-in-this-repo) upstream `pyadjoint.taylor_to_dict` helper.
    with pyadjoint.stop_annotating():
        h = dolfinx_adjoint.Function(Q)
        rng = np.random.default_rng(seed=42)
        h.x.array[:] = rng.standard_normal(len(h.x.array))
        h.x.scatter_forward()

        rate0 = pyadjoint.taylor_test(Jhat, rho, h, dJdm=0)
        print(f"[{case_name}] 0th-order Taylor rate (expect ~1): {rate0:.4f}")
        rate1 = pyadjoint.taylor_test(Jhat, rho, h)
        print(f"[{case_name}] 1st-order Taylor rate (expect ~2): {rate1:.4f}")

        Jhat(rho)
        dJdm = Jhat.derivative()._ad_dot(h)
        dHddu = Jhat.hessian(h)._ad_dot(h)
        rate2 = pyadjoint.taylor_test(Jhat, rho, h, dJdm=dJdm, Hm=dHddu)
        print(f"[{case_name}] 2nd-order Taylor rate (expect ~3): {rate2:.4f}")

    # ### Cost of one recompute vs. one adjoint solve
    #
    # Mirrors the forward-vs-VJP cost split Mosaic itself tracks separately
    # (`cost/spatial_cost` vs `cost/vjp_cost`).
    t_recompute_start = time.perf_counter()
    Jhat(rho)
    recompute_time = time.perf_counter() - t_recompute_start
    print(f"[{case_name}] one tape recompute (Jhat(rho)): {recompute_time:.4e} s")

    t_derivative_start = time.perf_counter()
    Jhat.derivative()
    derivative_time = time.perf_counter() - t_derivative_start
    print(f"[{case_name}] one adjoint solve (Jhat.derivative()): {derivative_time:.4e} s")

    # ### Bound- and curvature-aware optimization
    #
    # `scipy.optimize.minimize(method="trust-constr")` is the scipy method that accepts
    # *both* `bounds=` and a Hessian-vector product (`hessp=`); `pyadjoint.minimize`'s
    # convenience wrapper only wires `hessp` automatically for `method="Newton-CG"`
    # (which has no bounds support), so we drive `scipy.optimize.minimize` directly
    # from a `pyadjoint.reduced_functional_numpy.ReducedFunctionalNumPy` — the same public class
    # `pyadjoint.minimize` itself wraps every call in.
    plotter, gif_callback = make_gif_writer(msh, rho, f"topopt_{case_name}.gif")

    rf_np = pyadjoint.reduced_functional_numpy.ReducedFunctionalNumPy(Jhat)
    m0 = rf_np.get_controls()

    t_optim_start = time.perf_counter()
    res = scipy.optimize.minimize(
        fun=rf_np.__call__,
        x0=m0,
        jac=lambda m: rf_np.derivative(),
        hessp=lambda m, p: rf_np.hessian(p),
        method="trust-constr",
        bounds=scipy.optimize.Bounds(x_min, 1.0),
        callback=gif_callback,
        options={"maxiter": 200, "verbose": 2},
    )
    optim_time = time.perf_counter() - t_optim_start
    print(f"[{case_name}] trust-constr optimization: {optim_time:.4e} s over {res.nit} iterations")

    rho_opt = rf_np.set_controls(res.x)[0]
    rho.x.array[:] = rho_opt.x.array
    rho.x.scatter_forward()
    problem.solve(annotate=False)
    plotter.close()

    # Final compliance/volume fraction as plain floats.
    final_compliance = dolfinx_adjoint.assemble_scalar(ufl.action(L, uh), annotate=False)
    final_vol_frac = dolfinx_adjoint.assemble_scalar(rho * ufl.dx, annotate=False) / (Lx * Ly * Lz)

    # Static screenshot of the converged density field.
    grid = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(msh))
    grid.cell_data["rho"] = rho.x.array
    final_plotter = pyvista.Plotter(off_screen=pyvista.OFF_SCREEN)
    final_plotter.add_mesh(grid, scalars="rho", clim=(0.0, 1.0), cmap="viridis", show_edges=False)
    final_plotter.view_isometric()
    if pyvista.OFF_SCREEN:
        final_plotter.screenshot(f"topopt_{case_name}_final.png")
    else:
        final_plotter.show()

    return {
        "case": case_name,
        "compliance": float(final_compliance),
        "vol_frac": float(final_vol_frac),
        "n_iterations": int(res.nit),
        "forward_time": forward_time,
        "recompute_time": recompute_time,
        "derivative_time": derivative_time,
        "optim_time": optim_time,
    }


# ## Running both load cases
#
# Mosaic's canonical `optimization/topopt` run uses `corner_load=True`; we additionally
# run the uniform full-face load for comparison.

results = [run_topopt(corner_load=True), run_topopt(corner_load=False)]

# ## Summary

summary = pandas.DataFrame(results).set_index("case")
print("\n=== Summary ===")
print(summary)
