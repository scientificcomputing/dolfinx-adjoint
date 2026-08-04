import typing

from mpi4py import MPI

import dolfinx.fem.petsc
import numpy as np
import numpy.typing as npt
import pyadjoint
import pytest
import ufl
from ufl.algorithms import expand_derivatives

from dolfinx_adjoint import Function, LinearProblem, NonlinearProblem, assemble_scalar


def convergence_rates(r, p):
    cr = []  # convergence rates
    for i in range(1, len(p)):
        cr.append(np.log(r[i] / r[i - 1]) / np.log(p[i] / p[i - 1]))
    return cr


def reference_solution(
    mesh,
    d: ufl.core.expr.Expr,
    alpha: dolfinx.fem.Constant,
    m_func: typing.Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    step_func: typing.Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    step_length: float = 0.01,
    num_steps: int = 4,
) -> tuple[float, float, float, list[float]]:
    """Compute the functional value `J(m_func)`, `dJ/dm(m_func)[step_func]`,
    `step_func^T d2J/dm2[m_func] step_func` and `J(m_func + (0.5**i)*step_length * step_func)`
    for i in `range(steps)` using UFL to form the first and second order adjoint,
    the TLM and first and second order derivatives of J."""

    # Set up function spaces
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # type: ignore[arg-type]
    Q = dolfinx.fem.functionspace(mesh, ("DG", 0))  # type: ignore[arg-type]
    uh = dolfinx.fem.Function(V)  # Unknown
    du = ufl.TrialFunction(V)
    dv = ufl.TestFunction(V)
    m = dolfinx.fem.Function(Q)

    m.interpolate(m_func)

    F = ufl.inner(ufl.grad(uh), ufl.grad(dv)) * ufl.dx - ufl.inner(m, dv) * ufl.dx
    J = 1 / 2 * ufl.inner(uh - d, uh - d) * ufl.dx + alpha / 2 * m**2 * ufl.dx

    dFdu = ufl.derivative(F, uh, du)
    dFdudm = expand_derivatives(ufl.derivative(dFdu, m))
    assert dFdudm.empty()
    d2Fdudu = expand_derivatives(ufl.derivative(dFdu, uh))
    assert d2Fdudu.empty()
    dFdm = ufl.derivative(F, m)
    d2Fdmdm = expand_derivatives(ufl.derivative(dFdm, m))
    assert d2Fdmdm.empty()
    dJdm = ufl.derivative(J, m)
    dJdu = ufl.derivative(J, uh)
    d2Jdmdu = expand_derivatives(ufl.derivative(dJdu, m))
    assert d2Jdmdu.empty()
    d2Jdudu = ufl.derivative(dJdu, uh)
    d2Jdmdm = ufl.derivative(dJdm, m)

    u_dot = dolfinx.fem.Function(V)  # TLM solution
    dm = dolfinx.fem.Function(Q)  # Perturbation direction
    dm.interpolate(step_func)
    lmbda = dolfinx.fem.Function(V)  # Adjoint solution
    lmbda_dot = dolfinx.fem.Function(V)  # Second order adjoint solution

    # Solve forward problem
    a, L = ufl.system(ufl.replace(F, {uh: du}))
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    exterior_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, exterior_facets)
    bc = dolfinx.fem.dirichletbc(
        dolfinx.fem.Constant(mesh, np.dtype(dolfinx.default_scalar_type).type(0.0)), exterior_dofs, V
    )
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    }
    forward_problem = dolfinx.fem.petsc.LinearProblem(
        a, L, u=uh, petsc_options=petsc_options, petsc_options_prefix="forward_problem_", bcs=[bc]
    )
    forward_problem.solve()

    # Solve TLM
    a_tlm = dFdu
    L_tlm = -ufl.action(dFdm, dm)
    problem_tlm = dolfinx.fem.petsc.LinearProblem(
        a_tlm, L_tlm, u=u_dot, petsc_options=petsc_options, petsc_options_prefix="tlm_problem_", bcs=[bc]
    )
    problem_tlm.solve()

    # Solve adjoint problem
    a_adj = ufl.adjoint(dFdu)
    L_adj = dJdu
    problem_adj = dolfinx.fem.petsc.LinearProblem(
        a_adj, L_adj, u=lmbda, petsc_options=petsc_options, petsc_options_prefix="adjoint_problem_", bcs=[bc]
    )
    problem_adj.solve()

    # Solve second order adjoint problem
    a_soa = ufl.adjoint(dFdu)
    L_soa = ufl.action(ufl.adjoint(d2Jdudu), u_dot)
    # NOTE: These contributions are zero (maybe extend adjoint operator to handle this?)
    # + ufl.action(ufl.adjoint(d2Jdmdu), dm)
    # - ufl.action(ufl.action(ufl.adjoint(d2Fdudu), u_dot), lmbda)
    # - ufl.action(ufl.adjoint(ufl.action(d2Fdmdm,dm)),lmbda)
    problem_soa = dolfinx.fem.petsc.LinearProblem(
        a_soa, L_soa, u=lmbda_dot, petsc_options=petsc_options, petsc_options_prefix="soa_problem_", bcs=[bc]
    )
    problem_soa.solve()

    # [delta m]^T  d^2 J / dm^2 [delta m]
    Hmdm = -ufl.action(ufl.adjoint(dFdm), lmbda_dot) + ufl.action(ufl.adjoint(d2Jdmdm), dm)
    # NOTE: These contributions are zero (maybe extend adjoint operator to handle this?)
    #  - ufl.action(ufl.action(ufl.adjoint(d2Fdmdm), dm))

    # dJ/dm [delta m]
    Jac_adj = -ufl.action(ufl.adjoint(dFdm), lmbda) + dJdm

    Jac_vec = dolfinx.fem.assemble_vector(dolfinx.fem.form(Jac_adj))
    Jac_vec.scatter_reverse(dolfinx.la.InsertMode.add)
    Jac_vec.scatter_forward()
    Hm_vec = dolfinx.fem.assemble_vector(dolfinx.fem.form(Hmdm))
    Hm_vec.scatter_reverse(dolfinx.la.InsertMode.add)
    Hm_vec.scatter_forward()

    J_compiled = dolfinx.fem.form(J)

    m_org = m.x.array.copy()
    J_org = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(J_compiled)), op=MPI.SUM)
    steps = [step_length * (1 / 2) ** i for i in range(num_steps)]

    dJac_dm = dolfinx.cpp.la.inner_product(Jac_vec._cpp_object, dm.x._cpp_object)  # type: ignore[arg-type]
    Hm_dm = dolfinx.cpp.la.inner_product(Hm_vec._cpp_object, dm.x._cpp_object)  # type: ignore[arg-type]

    errors = []
    errors_der = []
    errors_hess = []
    functional_values = []
    for step in steps:
        m.x.array[:] = m_org + step * dm.x.array[:]

        forward_problem.solve()
        J_perturbed = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(J_compiled), op=MPI.SUM)
        functional_values.append(J_perturbed)
        errors.append(J_perturbed - J_org)
        errors_der.append(J_perturbed - J_org - step * dJac_dm)
        errors_hess.append(J_perturbed - J_org - step * dJac_dm - step**2 / 2 * Hm_dm)
    errors = np.abs(np.array(errors))
    errors_der = np.abs(np.array(errors_der))
    errors_hess = np.abs(np.array(errors_hess))

    np.testing.assert_allclose(convergence_rates(errors, steps), 1, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(convergence_rates(errors_der, steps), 2, atol=1e-6)
    np.testing.assert_allclose(errors_hess, 0.0, atol=1e-14)
    return J_org, dJac_dm, Hm_dm, functional_values


@pytest.mark.parametrize("linear_solver", [True, False])
@pytest.mark.parametrize("cell_type", [dolfinx.mesh.CellType.triangle, dolfinx.mesh.CellType.quadrilateral])
def test_poisson_mother(cell_type: dolfinx.mesh.CellType, linear_solver: bool):
    """Compare differentiation of the Poisson mother problem with a hand-written implementation."""
    steps = 4
    step_length = 0.01
    # Create a mesh
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 7, cell_type=cell_type)

    # Define the step function and reference solution
    def m_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Reference function for the control."""
        return x[0] + x[1]

    def step_func(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Step function for the control perturbation."""
        # This is a simple perturbation function
        return x[0] + np.sin(x[1])

    x, y = ufl.SpatialCoordinate(mesh)
    d = 1 / (2 * ufl.pi**2) * ufl.sin(ufl.pi * x) * ufl.sin(ufl.pi * y)

    alpha = dolfinx.fem.Constant(
        mesh, np.dtype(dolfinx.default_scalar_type).type(1.0e-6)
    )  # Tikhonov regularization parameter
    alpha.name = "alpha"  # type: ignore

    # Get reference values
    J_ref, dJ_ref, H_ref, ref_perturbations = reference_solution(
        mesh, d, alpha, m_func, step_func, num_steps=steps, step_length=step_length
    )

    pyadjoint.get_working_tape().clear_tape()

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))  # type: ignore[arg-type]
    Q = dolfinx.fem.functionspace(mesh, ("DG", 0))  # type: ignore[arg-type]
    f = Function(Q, name="Control")
    f.interpolate(m_func)
    uh = Function(V, name="State")
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - f * v * ufl.dx
    if not linear_solver:
        F = ufl.replace(F, {u: uh})
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim - 1, tdim)
    exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    exterior_dofs = dolfinx.fem.locate_dofs_topological(V, tdim - 1, exterior_facets)
    zero = dolfinx.fem.Constant(mesh, np.dtype(dolfinx.default_scalar_type).type(0.0))
    bc = dolfinx.fem.dirichletbc(zero, exterior_dofs, V)

    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    }
    problem: typing.Union[LinearProblem, NonlinearProblem]
    if linear_solver:
        problem = LinearProblem(
            *ufl.system(F),
            u=uh,
            bcs=[bc],
            petsc_options=petsc_options,
            adjoint_petsc_options=petsc_options,
            tlm_petsc_options=petsc_options,  # type: ignore
        )
    else:
        snes_options = {
            "snes_type": "newtonls",
            "snes_linesearch_type": "none",
            "snes_error_if_not_converged": True,
            "snes_monitor": None,
            "snes_atol": 1e-15,
            "snes_rtol": 1e-15,
        }
        snes_options.update(petsc_options)
        problem = NonlinearProblem(
            F,
            uh,
            bcs=[bc],
            petsc_options=snes_options,
            adjoint_petsc_options=petsc_options,
            tlm_petsc_options=petsc_options,
        )
    problem.solve()

    J_symbolic = 0.5 * ufl.inner(uh - d, uh - d) * ufl.dx + 0.5 * alpha * ufl.inner(f, f) * ufl.dx
    J = assemble_scalar(J_symbolic)

    control = pyadjoint.Control(f)
    Jhat = pyadjoint.ReducedFunctional(J, control)

    Jh = Jhat(f)
    df = Function(Q)
    df.interpolate(step_func)
    dJdm = Jhat.derivative()._ad_dot(df)
    hessian = Jhat.hessian(df)
    dHddu = hessian._ad_dot(df)

    assert np.isclose(Jh, J_ref)
    assert np.isclose(dJdm, dJ_ref)
    assert np.isclose(dHddu, H_ref)

    f_org = f.x.array.copy()
    for i in range(steps):
        f.x.array[:] = f_org + (0.5**i) * step_length * df.x.array[:]
        J_perturnbed = Jhat(f)
        assert np.isclose(J_perturnbed, ref_perturbations[i])
