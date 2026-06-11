from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import ufl

from dolfinx_adjoint import Function, LinearProblem, assemble_scalar, assign, dirichletbc
from dolfinx_adjoint.types.dirichletbc import DirichletBCBlock


def test_dirichletbc_recording():
    """Test that creating an overloaded dirichletbc correctly registers a block and dependency on the tape."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    c = Function(V, name="boundary_value")
    c.interpolate(lambda x: x[0])

    dofs = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = dirichletbc(c, dofs)

    tape = pyadjoint.get_working_tape()
    blocks = tape.get_blocks()

    # The tape should have 1 block: DirichletBCBlock
    assert len(blocks) == 1
    assert isinstance(blocks[0], DirichletBCBlock)

    # The block should have exactly 1 dependency (the function 'c')
    assert len(blocks[0].get_dependencies()) == 1
    assert blocks[0].get_dependencies()[0].output is c

    # The returned BC object should now possess the injected block_variable
    assert hasattr(bc, "block_variable")


def test_dirichletbc_no_annotate():
    """Test that setting annotate=False bypasses tape recording entirely."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    c = Function(V, name="boundary_value")
    c.interpolate(lambda x: x[0])

    dofs = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))

    # Run with annotation off
    bc = dirichletbc(c, dofs, annotate=False)

    tape = pyadjoint.get_working_tape()

    assert len(tape.get_blocks()) == 0
    assert not hasattr(bc, "block_variable")


def test_dirichletbc_recompute():
    """Test the PyAdjoint internal recompute logic specifically for the DirichletBCBlock."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    c = Function(V, name="boundary_value")
    c.interpolate(lambda x: np.full_like(x[0], 5.0))

    dofs = dolfinx.fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0))
    bc = dirichletbc(c, dofs)

    tape = pyadjoint.get_working_tape()
    block = tape.get_blocks()[0]

    # Simulate an optimizer changing the function value
    c.interpolate(lambda x: np.full_like(x[0], 15.0))

    # Replay the PyAdjoint mechanics manually
    prepared = block.prepare_recompute_component([c], None)
    new_bc = block.recompute_component([c], bc.block_variable, 0, prepared)

    # Assert that the re-instantiated C++ object captured the updated control value
    assert isinstance(new_bc, dolfinx.fem.bcs.DirichletBC)
    assert np.isclose(new_bc.g.x.array[0], 15.0)


def test_time_dependent_bc_replay():
    pyadjoint.get_working_tape().clear_tape()

    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    dt = 0.1
    num_steps = 3

    m = Function(V, name="control")
    m.interpolate(lambda x: np.sin(x[0] * np.pi))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    uh = Function(V, name="state")
    assign(0.0, uh)

    u_prev = Function(V, name="state_prev")
    assign(0.0, u_prev)

    F = (u - u_prev) / dt * v * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - m * v * ufl.dx
    a, L = ufl.system(F)

    bc_func = Function(V, name="bc_func")
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    boundary_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, boundary_facets)

    # Use native dolfinx here! PyAdjoint traces the bc_func inside it.
    bc = dolfinx.fem.dirichletbc(bc_func, boundary_dofs)

    problem = LinearProblem(a, L, bcs=[bc], u=uh)

    J = 0.0

    for i in range(num_steps):
        assign(float(i + 1), bc_func)
        problem.solve()
        J += assemble_scalar(0.5 * ufl.inner(uh, uh) * ufl.dx)
        assign(uh, u_prev)

    J_forward = float(J)

    control = pyadjoint.Control(m)
    Jhat = pyadjoint.ReducedFunctional(J, control)
    J_replay = Jhat(m)

    assert np.isclose(J_replay, J_forward, atol=1e-10, rtol=1e-10)
