from __future__ import annotations

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
        mesh: The mesh to move.
        displacement: The displacement, in the mesh's geometry function space.
    """
    gdim = mesh.geometry.dim
    mesh.geometry.x[:, :gdim] += displacement.x.array.reshape(-1, gdim)


def move(
    mesh: dolfinx.mesh.Mesh,
    displacement: dolfinx.fem.Function,
    **kwargs,
) -> Mesh:
    """Move a mesh's geometry by ``displacement``, recording the move on the tape.

    This is the annotating counterpart of {py:func}`scifem.mesh.move`, and the entry point
    for shape control: after this call every form posed on ``mesh`` carries a
    differentiable dependence on ``displacement`` through
    {py:class}`ufl.SpatialCoordinate`. ``mesh`` is promoted to an overloaded
    {py:class}`~dolfinx_adjoint.types.mesh.Mesh` in place, so existing function spaces and
    forms built on it stay valid.

    The control of a shape optimization is ``displacement``, not the mesh::

        S = dolfinx_adjoint.geometry_function_space(mesh)
        s = dolfinx_adjoint.Function(S)
        dolfinx_adjoint.move(mesh, s)
        ...
        Jhat = pyadjoint.ReducedFunctional(J, pyadjoint.Control(s))

    Args:
        mesh: The mesh to move.
        displacement: The displacement field. Must live in ``mesh``'s geometry function
            space (see {py:func}`~dolfinx_adjoint.geometry_function_space`).
        kwargs: ``"annotate"`` to control whether the move is recorded on the tape, and
            ``"ad_block_tag"`` to tag the resulting block.

    Returns:
        The mesh, promoted to an overloaded {py:class}`~dolfinx_adjoint.types.mesh.Mesh`.

    Raises:
        ValueError: If ``displacement`` does not live in the geometry function space.

    Note:
        Unlike {py:func}`scifem.mesh.move`, a UFL expression or a callable is not accepted:
        the displacement has to be a {py:class}`~dolfinx_adjoint.Function` for the tape to
        have anything to hold a derivative against. To drive the geometry from a field in
        a different space, interpolate it first with the annotating
        {py:func}`~dolfinx_adjoint.interpolate`, which contributes its own (differentiable)
        block::

            s_geom = dolfinx_adjoint.interpolate(s, geometry_function_space(mesh))
            dolfinx_adjoint.move(mesh, s_geom)
    """
    ad_block_tag = kwargs.pop("ad_block_tag", None)
    annotate = annotate_tape(kwargs)

    if not isinstance(displacement, dolfinx.fem.Function):
        raise ValueError(
            f"move() needs a Function as the displacement, got {type(displacement).__name__}. "
            "Interpolate an expression into the geometry function space with "
            "dolfinx_adjoint.interpolate() first, so the move stays differentiable."
        )

    V_geom = geometry_function_space(mesh)
    if displacement.function_space.dofmap.index_map_bs != V_geom.dofmap.index_map_bs or (
        displacement.function_space.element != V_geom.element
    ):
        raise ValueError(
            "The displacement must live in the mesh's geometry function space "
            f"({V_geom.ufl_element()}), got {displacement.function_space.ufl_element()}. "
            "Use dolfinx_adjoint.interpolate(displacement, "
            "dolfinx_adjoint.geometry_function_space(mesh)) to map it there first."
        )

    overloaded = annotate_mesh(mesh)

    if annotate:
        _reject_schedule_that_breaks_shape_derivatives()
        displacement = pyadjoint.create_overloaded_object(displacement)
        block = MoveBlock(overloaded, displacement, ad_block_tag=ad_block_tag)
        get_working_tape().add_block(block)

    with stop_annotating():
        apply_displacement(overloaded, displacement)

    if annotate:
        block.add_output(overloaded.create_block_variable())

    return overloaded
