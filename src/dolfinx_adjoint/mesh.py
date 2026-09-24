from __future__ import annotations

import typing

import dolfinx
import pyadjoint
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

from .blocks.mesh import MoveBlock
from .types.mesh import Mesh, annotate_mesh, geometry_function_space

__all__ = ["move", "annotate_mesh", "geometry_function_space", "apply_displacement"]


# Checkpoint schedules verified to give a correct shape derivative. Both retain every step,
# so nothing the adjoint reads is ever released. A schedule that *recomputes* (Revolve and
# relatives) releases dependency checkpoints the shape derivative needs, and
# `BlockVariable.saved_output` then silently returns the function's live value instead --
# measured at 17% error on a four-step heat equation, with a Taylor test still reporting
# rate 2. This is an allowlist rather than a denylist on purpose: an unrecognised schedule is
# refused, which is the safe direction to be wrong in.
_SHAPE_SAFE_SCHEDULES = frozenset({"SingleMemoryStorageSchedule", "SingleDiskStorageSchedule"})


def _reject_schedule_that_breaks_shape_derivatives() -> None:
    """Refuse, now, if the working tape carries a checkpoint schedule that recomputes.

    The alternative failure comes much later, from inside the adjoint sweep, by which point the
    user has paid for a whole forward run. pyadjoint requires ``enable_checkpointing`` to
    precede every block, so by the time :py:func:`move` is called the schedule is always
    already known and this can be said up front.

    Raises:
        NotImplementedError: If a checkpoint schedule is active and is not one of the
            schedules known to retain every step.
    """
    manager = getattr(get_working_tape(), "_checkpoint_manager", None)
    if manager is None:
        return
    schedule = getattr(manager, "_schedule", None)
    name = type(schedule).__name__ if schedule is not None else "unknown"
    if name in _SHAPE_SAFE_SCHEDULES:
        return
    raise NotImplementedError(
        f"Shape derivatives are not supported under the checkpoint schedule in use ({name}). "
        "A schedule that recomputes the forward releases dependency checkpoints that the "
        "shape derivative needs, and the gradient would be silently wrong -- wrong by 17% on "
        "a four-step heat equation, with a Taylor test still reporting rate 2. Use a schedule "
        f"that retains every step -- {', '.join(sorted(_SHAPE_SAFE_SCHEDULES))} -- or no "
        "schedule at all."
    )


def apply_displacement(mesh: dolfinx.mesh.Mesh, displacement: dolfinx.fem.Function) -> None:
    """Add ``displacement`` to ``mesh``'s coordinates, without touching the tape.

    The unannotated core of {py:func}`move`, shared with
    {py:meth}`~dolfinx_adjoint.blocks.mesh.MoveBlock.recompute_component`.

    Args:
        mesh: The mesh to move. Must already be tracked -- wrap it with
            {py:class}`dolfinx_adjoint.Mesh` where you create or read it.
        displacement: The displacement, in the mesh's geometry function space.
    """
    displacement.x.scatter_forward()  # Ensure that ghost nodes are up to date
    gdim = mesh.geometry.dim
    mesh.geometry.x[:, :gdim] += displacement.x.array.reshape(-1, gdim)


def _is_geometry_function(mesh: dolfinx.mesh.Mesh, candidate: typing.Any, V_geom: dolfinx.fem.FunctionSpace) -> bool:
    """Whether ``candidate`` is already a Function in *this* mesh's geometry space.

    **Collective.** The answer is reduced across the communicator, because the two branches at
    the call site diverge into collective code: a rank taking the direct path while another
    interpolates would hang. Nothing here guarantees per-rank agreement on its own, as dof
    orderings are a property of the local partition.

    Args:
        mesh: The mesh being moved.
        candidate: The object offered as a displacement.
        V_geom: ``mesh``'s geometry function space.

    Returns:
        True on every rank, or False on every rank.
    """
    local = isinstance(candidate, dolfinx.fem.Function)
    if local:
        space = candidate.function_space
        candidate_map = space.dofmap.index_map
        geometry_map = mesh.geometry.index_map()
        same_map = getattr(candidate_map, "_cpp_object", candidate_map) is getattr(
            geometry_map, "_cpp_object", geometry_map
        )
        local = (
            space.mesh is mesh
            and same_map
            and space.dofmap.index_map_bs == V_geom.dofmap.index_map_bs
            and space.element == V_geom.element
            and np.array_equal(np.asarray(space.dofmap.list), np.asarray(mesh.geometry.dofmaps[0]))
        )
    return bool(mesh.comm.allreduce(local, op=MPI.LAND))


def _as_geometry_displacement(
    mesh: dolfinx.mesh.Mesh,
    displacement: typing.Any,
    V_geom: dolfinx.fem.FunctionSpace,
    annotate: bool,
) -> dolfinx.fem.Function:
    """Return ``displacement`` as a Function in ``mesh``'s geometry space.

    A Function already living in that exact space is used as it stands, so no transformation required.
    Anything else; a {py:class}`UFL-expression<ufl.core.expr.Expr>` or a
    {py:class}`dolfinx.fem.Function` in another space on this mesh is
    interpolated into it by the annotating :py:func:`~dolfinx_adjoint.interpolate`, which
    contributes its own block so the chain rule through the interpolation is recorded.

    Args:
        mesh: The mesh being moved.
        displacement: A Function or any UFL expression over ``mesh``.
        V_geom: ``mesh``'s geometry function space.
        annotate: Whether the interpolation should be recorded on the tape.

    Returns:
        A Function in ``V_geom``.

    Raises:
        ValueError: If ``displacement`` is defined over a different mesh. Interpolating across
            meshes is a different operation with a different adjoint
            (:py:func:`~dolfinx_adjoint.interpolate_nonmatching`), not something to do silently
            here.
    """
    from .interpolation import interpolate

    if _is_geometry_function(mesh, displacement, V_geom):
        return displacement

    # A Function names its mesh directly; an expression names a ufl.Mesh domain, which is what
    # `mesh.ufl_domain()` is compared against below -- hence the deliberately loose type.
    source_mesh: typing.Any
    if isinstance(displacement, dolfinx.fem.Function):
        source_mesh = displacement.function_space.mesh
    else:
        source_mesh = ufl.domain.extract_unique_domain(ufl.as_ufl(displacement))
    if source_mesh is not None and source_mesh is not mesh and source_mesh is not mesh.ufl_domain():
        raise ValueError(
            "The displacement is defined over a different mesh than the one being moved. "
            "Interpolating between meshes has its own adjoint -- map it across with "
            "dolfinx_adjoint.interpolate_nonmatching() first, then pass the result."
        )

    return interpolate(displacement, V_geom, annotate=annotate)


def move(
    mesh: dolfinx.mesh.Mesh,
    displacement: dolfinx.fem.Function,
    **kwargs,
) -> Mesh:
    """Move a mesh's geometry by ``displacement``, recording the move on the tape.

    This adds ``displacement`` as a dependence to all future blocks through
    {py:class}`ufl.SpatialCoordinate`.

    The control of a shape optimization is ``displacement``, not the mesh.
    Example::

        ..code-block:: python

            mesh = dolfinx_adjoint.Mesh(mesh)          # track it, before posing anything on it
            S = dolfinx_adjoint.geometry_function_space(mesh)
            s = dolfinx_adjoint.Function(S)
            dolfinx_adjoint.move(mesh, s)
            ...
            Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))

    Args:
        mesh: The mesh to move. Must already be tracked -- wrap it with
            {py:class}`dolfinx_adjoint.Mesh` where you create or read it.
        displacement: The displacement. Either a {py:class}`~dolfinx_adjoint.Function` in
            ``mesh``'s geometry function space (see
            {py:func}`~dolfinx_adjoint.geometry_function_space`), which is used as it stands,
            or any UFL expression over ``mesh``. If an UFL expression or a Function in another
            space, this function interpolates it into the geometry space prior to moving the mesh.
        kwargs: ``"annotate"`` to control whether the move is recorded on the tape, and
            ``"ad_block_tag"`` to tag the resulting block.

    Returns:
        The mesh, promoted to an overloaded {py:class}`~dolfinx_adjoint.types.mesh.Mesh`.

    Raises:
        ValueError: If ``mesh`` is not a tracked {py:class}`dolfinx_adjoint.Mesh`, or if
            ``displacement`` is defined over a different mesh than ``mesh``.

    Note:
        Interpolating *between meshes* is a different operation with a different adjoint, so
        it is refused here rather than done silently -- map the field across with
        {py:func}`~dolfinx_adjoint.interpolate_nonmatching` first.
    """
    ad_block_tag = kwargs.pop("ad_block_tag", None)
    annotate = annotate_tape(kwargs)

    V_geom = geometry_function_space(mesh)

    if not annotate:
        # With annotation off this is a plain geometric operation, so it neither needs nor
        # requires a tracked mesh -- an untracked one is moved and handed straight back.
        with stop_annotating():
            geometry_disp = _as_geometry_displacement(mesh, displacement, V_geom, False)
            apply_displacement(mesh, geometry_disp)
        return typing.cast(Mesh, mesh)

    _reject_schedule_that_breaks_shape_derivatives()

    if not isinstance(mesh, Mesh):
        raise ValueError(
            "move() needs a mesh that is tracked for shape differentiation, and tracking is "
            "something to opt into explicitly: wrap it once, where you create or read it, with "
            "`mesh = dolfinx_adjoint.Mesh(mesh)` -- which copies nothing and hands back the same "
            "object -- and build your function spaces and forms on the result. Promoting it here "
            "instead would be too late for anything already posed on it, and would leave every "
            "later form on this mesh paying for a shape dependency nobody asked for."
        )
    overloaded = typing.cast(Mesh, mesh)
    geometry_disp = _as_geometry_displacement(overloaded, displacement, V_geom, True)

    overloaded_disp = pyadjoint.create_overloaded_object(geometry_disp)
    block = MoveBlock(overloaded, overloaded_disp, ad_block_tag=ad_block_tag)
    get_working_tape().add_block(block)

    with stop_annotating():
        apply_displacement(overloaded, geometry_disp)

    block.add_output(overloaded.create_block_variable())

    return overloaded
