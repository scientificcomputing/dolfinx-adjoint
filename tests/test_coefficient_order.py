from mpi4py import MPI

import dolfinx
import pytest
import ufl

from dolfinx_adjoint import Constant, Function
from dolfinx_adjoint.solvers import LinearProblem, NonlinearProblem


def test_function_and_constant_share_ufl_id_counter():
    """A Function and a Constant never get the same ufl_id, and a Constant uses up only one id."""
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    ids = [Function(V).ufl_id(), Constant(mesh, 1.0).ufl_id(), Constant(mesh, 2.0).ufl_id(), Function(V).ufl_id()]
    assert ids == list(range(ids[0], ids[0] + len(ids)))

    assert Constant(mesh, 1.0, ufl_id=ids[0]).ufl_id() == ids[0]


@pytest.mark.parametrize("reverse", [False, True])
def test_placeholder_order_independent_of_hash(monkeypatch, reverse):
    """Placeholder creation is collective, so its order must not follow set iteration order.

    That order follows the coefficient hashes, which depend on the per-process
    ``PYTHONHASHSEED`` and so differ between ranks. Coefficients with equal ufl_ids were
    created in set iteration order, which deadlocked the ranks. The seed is fixed at
    interpreter startup, so the hashes are patched instead to reverse the iteration order
    on odd ranks, and in serial between the two parametrizations. Both coefficients live
    in the same space, so a wrong order fails the assertion rather than deadlocking.
    """
    comm = MPI.COMM_WORLD
    order = ["g", "f"] if reverse != (comm.rank % 2 == 1) else ["f", "g"]
    original_hash = Function.__hash__

    def rank_dependent_hash(self):
        name = getattr(self, "name", None)
        return order.index(name) if name in order else original_hash(self)

    monkeypatch.setattr(Function, "__hash__", rank_dependent_hash)

    mesh = dolfinx.mesh.create_unit_square(comm, 2, 2)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    f = Function(V, name="f")
    g = Function(V, name="g", ufl_id=f.ufl_id())
    assert [c.name for c in {f, g}] == order

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + f * u * v * ufl.dx
    L = g * v * ufl.dx
    options = {"ksp_type": "preonly", "pc_type": "lu"}
    linear = LinearProblem(a, L, petsc_options=options, petsc_options_prefix="order_linear_")

    uh = Function(V, name="u")
    F = ufl.inner(ufl.grad(uh), ufl.grad(v)) * ufl.dx + f * uh * v * ufl.dx - g * v * ufl.dx
    nonlinear = NonlinearProblem(F, uh, petsc_options=options, petsc_options_prefix="order_nonlinear_")

    # f was created before g, so its placeholder comes first on every rank
    for problem in (linear, nonlinear):
        assert [c.name for c in problem.value_placeholders] == ["f", "g"]
