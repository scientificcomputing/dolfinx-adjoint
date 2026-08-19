"""Tests for the pointwise observation operator and its misfit."""

from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pyadjoint
import pytest
import ufl

import dolfinx_adjoint
from dolfinx_adjoint import observation


def unit_square(comm: MPI.Intracomm, n: int = 8) -> dolfinx.mesh.Mesh:
    return dolfinx.mesh.create_unit_square(comm, n, n)


def sample_points(n: int = 37, seed: int = 0) -> np.ndarray:
    """Points strictly inside the unit square, identical on every rank."""
    rng = np.random.default_rng(seed)
    return 0.05 + 0.9 * rng.random((n, 2))


def owned_dofs(V: dolfinx.fem.FunctionSpace) -> int:
    return V.dofmap.index_map.size_local * V.dofmap.bs


# ---------------------------------------------------------------------------
# The observation operator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [1, 2])
def test_exact_on_polynomials(degree):
    """B u reproduces the exact values of a function that lies in the FE space."""
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm), ("Lagrange", degree))

    def expression(x):
        if degree == 1:
            return 1.0 + 2.0 * x[0] - 3.0 * x[1]
        return 1.0 + 2.0 * x[0] - 3.0 * x[1] + 0.5 * x[0] * x[1] - x[0] ** 2 + 2 * x[1] ** 2

    u = dolfinx.fem.Function(V)
    u.interpolate(expression)

    points = sample_points()
    B = dolfinx_adjoint.PointObservation(V, points)

    assert B.found.all()
    assert np.allclose(B.gather(B.apply(u)), expression(points.T), atol=1e-12)


def test_evaluate_combines_apply_and_gather():
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm), ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: 2.0 * x[0] - x[1])

    points = sample_points(17, seed=101)
    B = dolfinx_adjoint.PointObservation(V, points)

    assert np.allclose(B.evaluate(u), B.gather(B.apply(u)), equal_nan=True)
    assert np.allclose(B.evaluate(u), 2.0 * points[:, 0] - points[:, 1], atol=1e-12)


def test_evaluate_marks_points_outside_the_mesh():
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm), ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0])

    B = dolfinx_adjoint.PointObservation(V, np.array([[0.5, 0.5], [7.0, 7.0]]))
    values = B.evaluate(u)
    assert np.isclose(values[0], 0.5)
    assert np.isnan(values[1])
    # The fill value is configurable, for callers that prefer a sentinel to NaN.
    assert B.evaluate(u, fill=0.0)[1] == 0.0


def test_partition_of_unity():
    """Each row of B sums to one, so a constant is observed as that constant."""
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm), ("Lagrange", 2))
    u = dolfinx.fem.Function(V)
    u.x.array[:] = 1.0

    B = dolfinx_adjoint.PointObservation(V, sample_points())
    assert np.allclose(B.gather(B.apply(u)), 1.0)


def test_matches_dolfinx_point_evaluation():
    """B u agrees with dolfinx' own evaluation of a non-polynomial function."""
    comm = MPI.COMM_WORLD
    mesh = unit_square(comm, 12)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.sin(3 * x[0]) * np.cos(2 * x[1]))

    points = sample_points(23, seed=3)
    values = dolfinx_adjoint.PointObservation(V, points).gather(dolfinx_adjoint.PointObservation(V, points).apply(u))

    points_3d = np.zeros((points.shape[0], 3))
    points_3d[:, :2] = points
    tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
    candidates = dolfinx.geometry.compute_collisions_points(tree, points_3d)
    colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, points_3d)
    for i in range(points.shape[0]):
        cells = colliding.links(i)
        if len(cells) == 0:
            continue  # this point is not on this rank
        assert np.isclose(u.eval(points_3d[i], cells[0])[0], values[i], atol=1e-12)


def test_vector_space_rows_are_component_fastest():
    """Row i*bs + c observes component c at point i."""
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm), ("Lagrange", 1, (2,)))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: np.vstack([x[0] + 2 * x[1], 3.0 - x[0]]))

    points = sample_points(11, seed=5)
    B = dolfinx_adjoint.PointObservation(V, points)
    assert B.block_size == 2

    values = B.gather(B.apply(u)).reshape(-1, 2)
    assert np.allclose(values[:, 0], points[:, 0] + 2 * points[:, 1], atol=1e-12)
    assert np.allclose(values[:, 1], 3.0 - points[:, 0], atol=1e-12)


def test_points_outside_the_mesh_are_reported_and_dropped():
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm), ("Lagrange", 1))
    points = np.array([[0.5, 0.5], [5.0, 5.0], [0.25, 0.75], [-1.0, 0.5]])

    B = dolfinx_adjoint.PointObservation(V, points)
    assert B.found.tolist() == [True, False, True, False]
    assert B.num_found == 2
    assert B.owner[1] == -1 and B.owner[3] == -1

    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] + x[1])
    values = B.gather(B.apply(u))
    assert np.isclose(values[0], 1.0) and np.isclose(values[2], 1.0)
    assert np.isnan(values[1]) and np.isnan(values[3])


def test_points_on_the_boundary_are_found():
    """The default padding makes points sitting exactly on the boundary detectable."""
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm), ("Lagrange", 1))
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.5], [0.5, 1.0]])
    B = dolfinx_adjoint.PointObservation(V, points)
    assert B.found.all()

    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] + x[1])
    assert np.allclose(B.gather(B.apply(u)), [0.0, 2.0, 0.5, 1.5], atol=1e-12)


def test_ownership_is_unique_and_agreed_on_by_every_rank():
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 10), ("Lagrange", 1))
    B = dolfinx_adjoint.PointObservation(V, sample_points(101, seed=7))

    claimed = np.zeros(B.num_points, dtype=np.int64)
    claimed[B.local_indices] = 1
    total = np.empty_like(claimed)
    comm.Allreduce(claimed, total, op=MPI.SUM)
    assert np.all(total[B.found] == 1)
    assert np.all(total[~B.found] == 0)

    for other in comm.allgather(B.owner):
        assert np.array_equal(other, B.owner)


def test_ambiguous_points_are_owned_exactly_once():
    """Points on cell vertices and facets lie in several cells, and often on several processes.

    These are the cases the ownership tie-break exists for, so they are worth pinning down
    separately from randomly scattered interior points.
    """
    comm = MPI.COMM_WORLD
    n = 8
    mesh = unit_square(comm, n)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    axis = np.linspace(0.0, 1.0, n + 1)
    vertices = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    midpoints = 0.5 * (vertices[:-1] + vertices[1:])
    points = np.vstack([vertices, midpoints])

    B = dolfinx_adjoint.PointObservation(V, points)
    assert B.found.all()

    claimed = np.zeros(B.num_points, dtype=np.int64)
    claimed[B.local_indices] = 1
    total = np.empty_like(claimed)
    comm.Allreduce(claimed, total, op=MPI.SUM)
    assert np.all(total == 1), "every point must be owned by exactly one process"

    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: 1.0 + 2.0 * x[0] - 3.0 * x[1])
    expected = 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1]
    assert np.allclose(B.evaluate(u), expected, atol=1e-12)


def test_transpose_is_the_adjoint_of_apply():
    """<Bu, v> == <u, B^T v> globally, including ghost contributions."""
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 9), ("Lagrange", 2))
    rng = np.random.default_rng(11)

    u = dolfinx.fem.Function(V)
    u.x.array[:] = rng.random(u.x.array.shape)
    u.x.scatter_forward()

    B = dolfinx_adjoint.PointObservation(V, sample_points(47, seed=13))
    v = B.restrict(rng.random(B.num_points))

    lhs = comm.allreduce(float(np.dot(B.apply(u), v)), op=MPI.SUM)
    Btv = B.apply_transpose(v)
    n = owned_dofs(V)
    rhs = comm.allreduce(float(np.dot(Btv.array[:n], u.x.array[:n])), op=MPI.SUM)

    assert np.isclose(lhs, rhs)


def test_replicated_points_are_required():
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 4), ("Lagrange", 1))
    if comm.size == 1:
        pytest.skip("differing point counts require more than one rank")
    points = sample_points(5) if comm.rank == 0 else sample_points(6)
    with pytest.raises(ValueError, match="replicated"):
        dolfinx_adjoint.PointObservation(V, points)


def test_mixed_element_is_rejected_on_every_process():
    """A mixed space must fail identically everywhere, not only where points landed.

    The element is validated up front rather than during assembly: a process that owns no
    points skips assembly entirely, so a lazy check would raise on some processes and not
    others and deadlock at the next collective call instead of surfacing the error.
    """
    comm = MPI.COMM_WORLD
    mesh = unit_square(comm, 6)
    element = basix.ufl.mixed_element(
        [
            basix.ufl.element("Lagrange", mesh.basix_cell(), 1),
            basix.ufl.element("Lagrange", mesh.basix_cell(), 2),
        ]
    )
    W = dolfinx.fem.functionspace(mesh, element)

    # A single point, so at most one process would ever reach assembly.
    with pytest.raises(NotImplementedError, match="Basix element"):
        dolfinx_adjoint.PointObservation(W, np.array([[0.3, 0.4]]))


def test_points_must_match_across_processes():
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 4), ("Lagrange", 1))
    if comm.size == 1:
        pytest.skip("mismatched points require more than one process")

    # Same number of points, different coordinates: caught by the checksum.
    points = sample_points(5, seed=109)
    if comm.rank == 1:
        points = points + 0.01
    with pytest.raises(ValueError, match="differing coordinates"):
        dolfinx_adjoint.PointObservation(V, points)


def test_permuted_points_are_rejected():
    """Reordering the rows leaves the sum of coordinates unchanged, so a plain-sum checksum
    would miss it; the checksum must weight by position to catch this too."""
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 4), ("Lagrange", 1))
    if comm.size == 1:
        pytest.skip("mismatched points require more than one process")

    points = sample_points(6, seed=131)
    if comm.rank == 1:
        points = points[::-1].copy()
    with pytest.raises(ValueError, match="differing coordinates"):
        dolfinx_adjoint.PointObservation(V, points)


def test_bad_point_shape_on_one_process_does_not_deadlock():
    """A per-process malformed `points` array must raise on every rank, not just the bad one.

    A validation failure on only some ranks would leave those ranks raising while the others
    proceed into the collective replication check below and block forever.
    """
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 4), ("Lagrange", 1))
    if comm.size == 1:
        pytest.skip("a per-process shape mismatch requires more than one process")

    points = np.zeros((2, 5)) if comm.rank == 1 else sample_points(5, seed=151)
    with pytest.raises(ValueError, match="at most 3 components"):
        dolfinx_adjoint.PointObservation(V, points)


def test_padding_does_not_depend_on_the_partition():
    """The default padding comes from the global bounding box, not each process's own."""
    comm = MPI.COMM_WORLD
    mesh = unit_square(comm, 8)
    padding = observation._default_padding(mesh)
    for other in comm.allgather(padding):
        assert other == padding
    # Unit square: the diagonal is sqrt(2) regardless of how the mesh is split up.
    assert np.isclose(padding, 1e-10 * np.sqrt(2.0))


def test_operator_survives_a_process_owning_no_points():
    """Clustered points leave some processes with nothing to do."""
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 8), ("Lagrange", 1))
    points = np.array([[0.01, 0.01], [0.02, 0.02], [0.03, 0.01]])

    B = dolfinx_adjoint.PointObservation(V, points)
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] + x[1])

    assert np.allclose(B.evaluate(u), points.sum(axis=1), atol=1e-12)
    # An empty local block must still transpose, and contribute nothing.
    result = B.apply_transpose(np.zeros(B.num_local_rows))
    assert np.allclose(result.array, 0.0)


def test_operator_with_no_points_at_all():
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 4), ("Lagrange", 1))
    B = dolfinx_adjoint.PointObservation(V, np.zeros((0, 2)))

    u = dolfinx.fem.Function(V)
    u.x.array[:] = 1.0
    assert B.num_points == 0
    assert B.num_local_rows == 0
    assert B.evaluate(u).shape == (0,)
    assert np.allclose(B.apply_transpose(np.zeros(0)).array, 0.0)


def test_too_many_components_raises():
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 4), ("Lagrange", 1))
    with pytest.raises(ValueError, match="at most 3 components"):
        dolfinx_adjoint.PointObservation(V, np.zeros((3, 4)))


@pytest.mark.parametrize(
    "cell_type",
    [
        dolfinx.mesh.CellType.tetrahedron,
        dolfinx.mesh.CellType.hexahedron,
        dolfinx.mesh.CellType.prism,
    ],
)
@pytest.mark.parametrize("degree", [1, 2])
def test_three_dimensional_cell_types(cell_type, degree):
    """Simplex and tensor-product cells alike, in 3D.

    Only the simplex has an affine coordinate map; the others go through the general
    Newton pull-back, which must be just as exact.
    """
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 3, 3, 3, cell_type=cell_type)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: 1.0 + 2.0 * x[0] - 3.0 * x[1] + 0.5 * x[2])

    points = np.array([[0.21, 0.33, 0.47], [0.62, 0.11, 0.88], [0.5, 0.5, 0.5]])
    B = dolfinx_adjoint.PointObservation(V, points)
    expected = 1.0 + 2.0 * points[:, 0] - 3.0 * points[:, 1] + 0.5 * points[:, 2]

    assert B.found.all()
    assert np.allclose(B.evaluate(u), expected, atol=1e-12)


def test_distorted_hexahedra_are_handled():
    """A hexahedron's coordinate map is trilinear, so a distorted mesh is truly non-affine."""
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_unit_cube(comm, 4, 4, 4, cell_type=dolfinx.mesh.CellType.hexahedron)
    # Warp the geometry so that cell faces are no longer planar.
    x = mesh.geometry.x
    x[:, 0] += 0.08 * np.sin(3.0 * x[:, 1]) * x[:, 2]
    x[:, 1] += 0.06 * np.cos(2.0 * x[:, 0]) * x[:, 2]

    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    # Q1 on a distorted hex reproduces affine fields exactly, unlike general quadratics.
    u.interpolate(lambda x: 0.75 + x[0] - 2.0 * x[1] + 3.0 * x[2])

    points = np.array([[0.5, 0.5, 0.5], [0.3, 0.62, 0.25], [0.7, 0.2, 0.8]])
    B = dolfinx_adjoint.PointObservation(V, points)
    expected = 0.75 + B.points[:, 0] - 2.0 * B.points[:, 1] + 3.0 * B.points[:, 2]

    assert B.found.all()
    assert np.allclose(B.evaluate(u), expected, atol=1e-11)


def test_matrix_is_distributed_over_the_points():
    """The operator is an interpolation matrix onto a point mesh of the observed points."""
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 8), ("Lagrange", 1))
    points = sample_points(31, seed=91)
    B = dolfinx_adjoint.PointObservation(V, points)

    rows, columns = B.matrix.getSizes()
    assert rows[1] == B.num_found  # one global row per located point
    assert columns[1] == V.dofmap.index_map.size_global * V.dofmap.bs
    assert rows[0] == B.num_local_rows
    assert comm.allreduce(B.num_local_rows, op=MPI.SUM) == B.num_found

    # The point mesh carries exactly the points this process owns.
    owned = B.point_mesh.geometry.x[: B.num_local_rows, :2]
    assert np.allclose(owned, points[B.local_indices])


def test_quadrilateral_mesh_uses_the_generic_pull_back():
    """Non-simplex geometry falls back to the Newton pull-back and stays exact."""
    comm = MPI.COMM_WORLD
    mesh = dolfinx.mesh.create_rectangle(
        comm,
        [np.array([0.0, 0.0]), np.array([1.0, 1.0])],
        [5, 5],
        cell_type=dolfinx.mesh.CellType.quadrilateral,
    )
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: 2.0 * x[0] + 3.0 * x[1])

    points = sample_points(13, seed=83)
    B = dolfinx_adjoint.PointObservation(V, points)
    assert np.allclose(B.gather(B.apply(u)), 2.0 * points[:, 0] + 3.0 * points[:, 1], atol=1e-12)


def test_matrix_action_matches_apply():
    """`apply` is exactly a matrix-vector product with the interpolation matrix."""
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 6), ("Lagrange", 1))
    B = dolfinx_adjoint.PointObservation(V, sample_points(15, seed=89))

    rng = np.random.default_rng(97)
    u = dolfinx.fem.Function(V)
    u.x.array[:] = rng.random(u.x.array.shape)
    u.x.scatter_forward()

    result = dolfinx.fem.Function(B.observation_space)
    B.matrix.mult(u.x.petsc_vec, result.x.petsc_vec)
    result.x.scatter_forward()
    assert np.allclose(result.x.array[: B.num_local_rows], B.apply(u))

    # Rows form a partition of unity, so the matrix reproduces constants exactly.
    ones = dolfinx.fem.Function(V)
    ones.x.array[:] = 1.0
    assert np.allclose(B.apply(ones), 1.0)


# ---------------------------------------------------------------------------
# The misfit
# ---------------------------------------------------------------------------


def test_misfit_matches_a_manual_computation():
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm), ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0] + x[1])

    points = sample_points(15, seed=19)
    B = dolfinx_adjoint.PointObservation(V, points)
    data = np.random.default_rng(23).random(B.num_points)
    noise_variance = 0.25

    with pyadjoint.stop_annotating():
        J = dolfinx_adjoint.point_observation_misfit(u, B, data, noise_variance=noise_variance)

    expected = 0.5 * np.sum(((points[:, 0] + points[:, 1]) - data) ** 2) / noise_variance
    assert np.isclose(float(J), expected)


def test_misfit_vanishes_at_the_exact_data():
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm), ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: 2.0 * x[0] - x[1])

    B = dolfinx_adjoint.PointObservation(V, sample_points(13, seed=29))
    with pyadjoint.stop_annotating():
        J = dolfinx_adjoint.point_observation_misfit(u, B, B.gather(B.apply(u)))
    assert np.isclose(float(J), 0.0, atol=1e-20)


def test_misfit_weights_mask_rows_out():
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm), ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    u.interpolate(lambda x: x[0])

    points = sample_points(10, seed=31)
    B = dolfinx_adjoint.PointObservation(V, points)
    weights = np.zeros(B.num_points)
    weights[:4] = 1.0

    with pyadjoint.stop_annotating():
        J = dolfinx_adjoint.point_observation_misfit(u, B, np.zeros(B.num_points), weights=weights)
    assert np.isclose(float(J), 0.5 * np.sum(points[:4, 0] ** 2))


def test_zero_noise_variance_raises():
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 4), ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    B = dolfinx_adjoint.PointObservation(V, sample_points(4, seed=37))
    with pytest.raises(ZeroDivisionError):
        dolfinx_adjoint.point_observation_misfit(u, B, np.zeros(B.num_points), noise_variance=0.0)


def test_mismatched_data_length_raises():
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 4), ("Lagrange", 1))
    u = dolfinx.fem.Function(V)
    B = dolfinx_adjoint.PointObservation(V, sample_points(6, seed=41))
    with pytest.raises(ValueError, match="Expected an array of length"):
        dolfinx_adjoint.point_observation_misfit(u, B, np.zeros(B.num_points + 3))


def test_reusing_a_data_buffer_does_not_corrupt_earlier_blocks():
    comm = MPI.COMM_WORLD
    tape = pyadjoint.get_working_tape()
    tape.clear_tape()
    V = dolfinx.fem.functionspace(unit_square(comm, 4), ("Lagrange", 1))
    u = dolfinx_adjoint.Function(V, name="u")
    u.interpolate(lambda x: x[0])

    B = dolfinx_adjoint.PointObservation(V, sample_points(5, seed=17))
    buffer = B.restrict(np.zeros(B.num_points))

    buffer[:] = 1.0
    dolfinx_adjoint.point_observation_misfit(u, B, buffer)
    first_block = tape.get_blocks()[-1]

    buffer[:] = 2.0  # mutated after the block was built, as a caller reusing the array would
    dolfinx_adjoint.point_observation_misfit(u, B, buffer)
    second_block = tape.get_blocks()[-1]

    assert np.allclose(first_block.data, 1.0)
    assert np.allclose(second_block.data, 2.0)
    tape.clear_tape()


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------


def test_gradient_taylor_test():
    comm = MPI.COMM_WORLD
    pyadjoint.get_working_tape().clear_tape()
    V = dolfinx.fem.functionspace(unit_square(comm, 6), ("Lagrange", 1))

    u = dolfinx_adjoint.Function(V, name="u")
    u.interpolate(lambda x: x[0] * x[1])

    B = dolfinx_adjoint.PointObservation(V, sample_points(21, seed=37))
    rng = np.random.default_rng(41)
    data = rng.random(B.num_points)

    control = pyadjoint.Control(u)
    J = dolfinx_adjoint.point_observation_misfit(u, B, data, noise_variance=0.5)
    Jhat = pyadjoint.ReducedFunctional(J, control)

    h = dolfinx_adjoint.Function(V)
    h.x.array[:] = rng.random(h.x.array.shape)
    h.x.scatter_forward()

    assert pyadjoint.taylor_test(Jhat, u, h) > 1.9
    pyadjoint.get_working_tape().clear_tape()


def test_gradient_matches_the_closed_form():
    """dJ/du == sigma^-2 B^T (Bu - d)."""
    comm = MPI.COMM_WORLD
    pyadjoint.get_working_tape().clear_tape()
    V = dolfinx.fem.functionspace(unit_square(comm, 6), ("Lagrange", 1))

    u = dolfinx_adjoint.Function(V, name="u")
    u.interpolate(lambda x: x[0] + 0.5 * x[1])

    B = dolfinx_adjoint.PointObservation(V, sample_points(19, seed=43))
    data = np.random.default_rng(47).random(B.num_points)
    noise_variance = 0.75

    J = dolfinx_adjoint.point_observation_misfit(u, B, data, noise_variance=noise_variance)
    gradient = pyadjoint.ReducedFunctional(J, pyadjoint.Control(u)).derivative()

    expected = B.apply_transpose((B.apply(u) - B.restrict(data)) / noise_variance)
    n = owned_dofs(V)
    assert np.allclose(gradient.x.array[:n], expected.array[:n])
    pyadjoint.get_working_tape().clear_tape()


def test_weighted_derivatives_apply_the_weights_twice():
    """With weights, dJ/du is sigma^-2 B^T W^2 (Bu - d), not W applied once.

    Differentiating ||W r||^2 brings down one W from the norm and one from the residual,
    so a non-binary weight vector is the case that would expose getting this wrong.
    """
    pyadjoint.get_working_tape().clear_tape()
    comm = MPI.COMM_WORLD
    V = dolfinx.fem.functionspace(unit_square(comm, 6), ("Lagrange", 1))

    rng = np.random.default_rng(5)
    B = dolfinx_adjoint.PointObservation(V, sample_points(17, seed=113))
    data = rng.random(B.num_points)
    weights = rng.random(B.num_points)
    noise_variance = 0.7

    u = dolfinx_adjoint.Function(V, name="u")
    u.interpolate(lambda x: x[0] * x[1])

    J = dolfinx_adjoint.point_observation_misfit(u, B, data, noise_variance=noise_variance, weights=weights)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(u))

    h = dolfinx_adjoint.Function(V)
    h.interpolate(lambda x: np.sin(3.0 * x[0]) + x[1])
    assert pyadjoint.taylor_test(Jhat, u, h) > 1.9

    # taylor_test leaves the control perturbed, so re-evaluate before differentiating.
    Jhat(u)
    gradient = Jhat.derivative()

    local_weights = B.restrict(weights)
    residual = B.apply(u) - B.restrict(data)
    expected = B.apply_transpose(local_weights**2 * residual / noise_variance)

    n = owned_dofs(V)
    assert np.allclose(gradient.x.array[:n], expected.array[:n])

    curvature = Jhat.hessian(h)
    expected_curvature = B.apply_transpose(local_weights**2 * B.apply(h) / noise_variance)
    assert np.allclose(curvature.x.array[:n], expected_curvature.array[:n])
    pyadjoint.get_working_tape().clear_tape()


def test_hessian_is_the_gauss_newton_operator():
    """The misfit is quadratic, so its Hessian action is sigma^-2 B^T B h."""
    comm = MPI.COMM_WORLD
    pyadjoint.get_working_tape().clear_tape()
    V = dolfinx.fem.functionspace(unit_square(comm, 6), ("Lagrange", 1))

    u = dolfinx_adjoint.Function(V, name="u")
    u.interpolate(lambda x: x[0] - x[1])

    B = dolfinx_adjoint.PointObservation(V, sample_points(17, seed=53))
    rng = np.random.default_rng(59)
    noise_variance = 2.0

    J = dolfinx_adjoint.point_observation_misfit(u, B, rng.random(B.num_points), noise_variance=noise_variance)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(u))
    Jhat(u)
    Jhat.derivative()

    h = dolfinx_adjoint.Function(V)
    h.x.array[:] = rng.random(h.x.array.shape)
    h.x.scatter_forward()

    expected = B.apply_transpose(B.apply(h) / noise_variance)
    n = owned_dofs(V)
    assert np.allclose(Jhat.hessian(h).x.array[:n], expected.array[:n])
    pyadjoint.get_working_tape().clear_tape()


def test_taylor_test_through_a_pde_solve():
    """The full parameter-to-observable map is differentiable: m -> u(m) -> B u(m)."""
    comm = MPI.COMM_WORLD
    pyadjoint.get_working_tape().clear_tape()
    mesh = unit_square(comm, 6)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

    # -div(exp(m) grad u) = f, with m the log-diffusivity we differentiate against.
    m = dolfinx_adjoint.Function(V, name="m")
    m.x.array[:] = 0.3

    u = dolfinx_adjoint.Function(V, name="u")
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(mesh)
    f = ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1])
    a = ufl.inner(ufl.exp(m) * ufl.grad(trial), ufl.grad(test)) * ufl.dx
    L = ufl.inner(f, test) * ufl.dx

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    u_bc = dolfinx_adjoint.Function(V, name="u_bc")
    bc = dolfinx_adjoint.dirichletbc(u_bc, dofs)

    problem = dolfinx_adjoint.LinearProblem(a, L, u=u, bcs=[bc], petsc_options_prefix="test_observation_")
    problem.solve()

    B = dolfinx_adjoint.PointObservation(V, sample_points(15, seed=61))
    rng = np.random.default_rng(67)
    data = B.gather(B.apply(u)) + 0.01 * rng.standard_normal(B.num_points)

    J = dolfinx_adjoint.point_observation_misfit(u, B, data, noise_variance=0.01)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(m))

    h = dolfinx_adjoint.Function(V)
    h.x.array[:] = 0.1 * rng.standard_normal(h.x.array.shape)
    h.x.scatter_forward()

    assert pyadjoint.taylor_test(Jhat, m, h) > 1.9
    pyadjoint.get_working_tape().clear_tape()


def test_recovers_a_known_source_from_point_data():
    """A small end-to-end inversion: point data is enough to pin down a scalar source."""
    comm = MPI.COMM_WORLD
    pyadjoint.get_working_tape().clear_tape()
    mesh = unit_square(comm, 8)
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
    true_scale = 3.0

    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, facets)

    def solve_for(scale_value, prefix):
        """-div(grad u) = scale * sin(pi x) sin(pi y), with u = 0 on the boundary."""
        scale = dolfinx_adjoint.Constant(mesh, scale_value)
        state = dolfinx_adjoint.Function(V, name="u")
        trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
        x = ufl.SpatialCoordinate(mesh)
        a = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
        L = ufl.inner(scale * ufl.sin(np.pi * x[0]) * ufl.sin(np.pi * x[1]), test) * ufl.dx
        u_bc = dolfinx_adjoint.Function(V, name="u_bc")
        bc = dolfinx_adjoint.dirichletbc(u_bc, dofs)
        dolfinx_adjoint.LinearProblem(a, L, u=state, bcs=[bc], petsc_options_prefix=prefix).solve()
        return scale, state

    points = sample_points(12, seed=71)
    with pyadjoint.stop_annotating():
        _, truth = solve_for(true_scale, "test_recover_truth_")
        B_truth = dolfinx_adjoint.PointObservation(V, points)
        observations = B_truth.gather(B_truth.apply(truth))

    scale, state = solve_for(1.0, "test_recover_")
    B = dolfinx_adjoint.PointObservation(V, points)
    J = dolfinx_adjoint.point_observation_misfit(state, B, observations)
    Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(scale))

    # The observations depend linearly on the source scale, so the misfit is an exact
    # quadratic and a single Newton step from the initial guess lands on the true value.
    gradient = Jhat.derivative()
    curvature = Jhat.hessian(dolfinx_adjoint.Constant(mesh, 1.0))
    recovered = 1.0 - float(gradient.x.array[0]) / float(curvature.x.array[0])

    assert np.isclose(recovered, true_scale, rtol=1e-8)
    pyadjoint.get_working_tape().clear_tape()
