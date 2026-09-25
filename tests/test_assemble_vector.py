"""Derivatives through ``assemble_vector``, and the vectors ``AssembleBlock`` reuses between sweeps."""

from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

import dolfinx_adjoint as dxa


def _owned_dot(a, b) -> float:
    space = a.function_space
    n = space.dofmap.index_map.size_local * space.dofmap.index_map_bs
    return MPI.COMM_WORLD.allreduce(float(np.dot(a.x.array[:n], b.x.array[:n])), op=MPI.SUM)


def _coefficient_forward(values=None):
    """``int b^2 dx`` for ``b = assemble_vector(m^2 sin(x) v dx)``, with ``m`` the control."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 4)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    m = dxa.Function(V, name="m")
    m.interpolate(lambda x: 1.0 + x[0] * x[1])
    if values is not None:
        m.x.array[:] = values
    x = ufl.SpatialCoordinate(mesh)
    b = dxa.assemble_vector(m**2 * ufl.sin(3 * x[0]) * ufl.TestFunction(V) * ufl.dx)
    return dxa.assemble_scalar(b * b * m * ufl.dx), m


def _shape_forward(values=None):
    """The same, with the mesh moved by the control ``s``."""
    pyadjoint.get_working_tape().clear_tape()
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 4)
    S = dxa.geometry_function_space(mesh)
    s = dxa.Function(S, name="s")
    if values is not None:
        s.x.array[:] = values
    mesh = dxa.Mesh(mesh)
    dxa.move(mesh, s)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    x = ufl.SpatialCoordinate(mesh)
    b = dxa.assemble_vector(ufl.sin(3 * x[0]) * (1 + x[1] ** 2) * ufl.TestFunction(V) * ufl.dx)
    return dxa.assemble_scalar(b * b * ufl.dx), s


def _direction(control) -> dxa.Function:
    h = dxa.Function(control.function_space)
    gdim = control.function_space.mesh.geometry.dim
    if control.function_space.value_shape == ():
        h.interpolate(lambda x: np.cos(x[0]) + x[1] ** 2)
    else:
        h.interpolate(lambda x: np.vstack([0.1 * x[k] * (1 + x[(k + 1) % gdim]) for k in range(gdim)]))
    return h


@pytest.mark.parametrize("forward", [_coefficient_forward, _shape_forward], ids=["coefficient", "shape"])
def test_assemble_vector_derivatives_by_value(forward):
    """Gradient, tangent-linear model and Hessian through a vector assembly, by central differences."""
    J, control = forward()
    base, direction = control.x.array.copy(), _direction(control).x.array.copy()
    eps = 1e-6
    fd = (float(forward(base + eps * direction)[0]) - float(forward(base - eps * direction)[0])) / (2 * eps)
    assert abs(fd) > 1e-8, "the direction must actually change the functional"

    # Every forward builds its own mesh, so the direction is rebuilt on the last one.
    J, control = forward()
    h = dxa.Function(control.function_space)
    h.x.array[:] = direction
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(control))
    assert abs(_owned_dot(Jhat.derivative(), h) - fd) < 1e-6 * abs(fd)
    assert abs(float(Jhat.tlm(h)) - fd) < 1e-6 * abs(fd)

    Jhat.derivative()
    Hh = _owned_dot(Jhat.hessian(h), h)

    def directional_gradient(step: float) -> float:
        shifted = dxa.Function(control.function_space)
        shifted.x.array[:] = base + step * h.x.array
        Jhat(shifted)
        return _owned_dot(Jhat.derivative(), h)

    eps = 1e-4
    fd2 = (directional_gradient(eps) - directional_gradient(-eps)) / (2 * eps)
    assert abs(fd2) > 1e-9
    assert abs(Hh - fd2) < 1e-5 * abs(fd2), f"hessian {Hh} vs finite difference {fd2}"


def test_derivatives_are_independent_of_the_cached_vectors():
    """A gradient handed to the user is a copy: the next sweep reuses the block's vector."""
    J, m = _coefficient_forward()
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(m))
    first = Jhat.derivative()
    kept = first.x.array.copy()
    shifted = dxa.Function(m.function_space)
    shifted.x.array[:] = 2.0 * m.x.array
    Jhat(shifted)
    second = Jhat.derivative()
    assert not np.allclose(second.x.array, kept), "the functional must depend on the control"
    assert np.array_equal(first.x.array, kept), "the first gradient was overwritten by the second"


def test_assemble_block_reuses_its_vectors():
    """Repeated sweeps assemble into the same vectors rather than creating new ones."""
    J, m = _coefficient_forward()
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(m))
    blocks = [b for b in pyadjoint.get_working_tape().get_blocks() if isinstance(b, dxa.blocks.AssembleBlock)]
    Jhat.derivative()
    Jhat.hessian(_direction(m))
    before = {id(block): {key: id(v) for key, v in block._vectors.items()} for block in blocks}
    assert all(before.values()), "every assembly block should have cached a vector"
    Jhat.derivative()
    Jhat.hessian(_direction(m))
    after = {id(block): {key: id(v) for key, v in block._vectors.items()} for block in blocks}
    assert after == before


def test_assemble_vector_refuses_a_bilinear_form():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 2, 2)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    with pytest.raises(ValueError, match="linear form"):
        dxa.assemble_vector(ufl.TrialFunction(V) * ufl.TestFunction(V) * ufl.dx)
