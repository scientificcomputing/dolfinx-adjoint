from __future__ import annotations

import typing

import dolfinx
import pyadjoint
import ufl
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

from .blocks.mesh import MoveBlock
from .interpolation import interpolate
from .types.mesh import Mesh, annotate_mesh, apply_displacement, geometry_function_space

__all__ = ["move", "annotate_mesh", "geometry_function_space", "apply_displacement"]


# Checkpoint schedules verified to give a correct shape derivative. The property that matters
# is not which schedule it is but whether it *recomputes*: these two store every step and
# replay nothing, so no checkpoint is ever released and the question below never arises.
#
# Why a recomputing schedule (Revolve and relatives) breaks the shape derivative
# ------------------------------------------------------------------------------
# Checkpointing trades memory for recomputation: it keeps a few steps and regenerates the rest
# from them. Regenerating means the intermediate values are discarded and later rebuilt, and
# pyadjoint's `Forward` handler discards them bluntly -- `checkpointing.py` clears the previous
# step's `checkpointable_state` and every block output outright, keeping only what
# `TimeStep.checkpoint(...)` was told to store.
#
# What it stores *for the adjoint* is `TimeStep.adjoint_dependencies`, and that set is
# populated in the `Reverse` handler during the first reverse traversal of a step, **after**
# `block.evaluate_adj()` has already run on it. So on the pass that matters the set is not yet
# known, and the fallback is `checkpointable_state`: the data needed to *restart the forward*,
# which is not the same as the data the adjoint reads.
#
# A block whose adjoint needs only the *structure* of its form never notices. For a linear
# residual `dF/dm` with respect to a coefficient does not reference the released values at all
# -- which is exactly why every coefficient-control test in this suite passes under Revolve.
# `dF/dX` does reference them: a coordinate derivative differentiates the whole integrand, the
# measure and the basis functions included, so every coefficient in the residual is dragged
# into it and evaluated.
#
# What turns that into a wrong number rather than an exception is `BlockVariable.saved_output`,
# which returns `self.output` when `checkpoint is None` -- the *live* function, holding whatever
# the last recomputed step left in it. Measured at 17% error on a four-step heat equation, with
# a Taylor test still reporting rate 2, because the tape stays self-consistent: it is simply no
# longer a model of the forward problem.
#
# The marking is not the problem, so this is not a missing declaration on our side: the released
# dependency carries `is_functional_dependency=True`, i.e. pyadjoint knows the value matters and
# clears it anyway. A fix belongs upstream -- either the adjoint-dependency set has to be known
# before the data is dropped, or `saved_output` has to refuse rather than guess.
#
# An allowlist rather than a denylist on purpose: nothing on a `checkpoint_schedules` schedule
# advertises whether it recomputes, so an unrecognised one is refused, which is the safe
# direction to be wrong in.
_SHAPE_SAFE_SCHEDULES = frozenset({"SingleMemoryStorageSchedule", "SingleDiskStorageSchedule"})


def _reject_schedule_that_breaks_shape_derivatives() -> None:
    """Refuse, now, if the working tape carries a checkpoint schedule that recomputes.

    See :py:data:`_SHAPE_SAFE_SCHEDULES` for why recomputation is the property that matters and
    why the resulting gradient is wrong rather than merely unavailable.

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


def _is_geometry_function(mesh: dolfinx.mesh.Mesh, candidate: typing.Any, V_geom: dolfinx.fem.FunctionSpace) -> bool:
    """Whether ``candidate`` is already a Function in *this* mesh's geometry space.

    :py:func:`apply_displacement` adds ``candidate``'s dof array onto ``mesh.geometry.x`` row by
    row, so what has to hold is that dof block *i* of the candidate is geometry node *i*.

    Identity is checked for the index map, not equality: ``FiniteElement::operator==`` compares
    the underlying basix element by value -- it carries no dofmap, no index map and no mesh -- so
    an element comparison alone accepts a displacement belonging to a *different* mesh with the
    same coordinate element, and the displacement would then be added to the wrong nodes.

    Note:
        Comparing the dofmaps themselves (``space.dofmap.list`` against
        ``mesh.geometry.dofmaps[0]``) would look like a stronger check, and it is tempting
        because node ordering is the property actually relied on. It is deliberately not done,
        for a reason that only shows up in parallel.

        Every other input here is partition-independent: both spaces are built by collective
        calls on the same mesh, so the element, the block size and the index map object are the
        same on every rank by construction, and this predicate therefore answers the same on
        every rank without having to communicate. The dofmap *contents* are per-rank data. Add
        them and the predicate can, in principle, answer differently on different ranks -- and
        the caller's two branches diverge into collective code, since interpolating is
        collective. A rank taking the direct path while another interpolates hangs.

        Making it collective instead (an ``allreduce`` over ``mesh.comm``) would fix that, at
        the cost of a communication in ``move()`` and of a predicate that can no longer be
        evaluated locally. It is not worth it here: the case the dofmap comparison would catch
        -- a space sharing the geometry index map but ordering its dofs differently -- cannot be
        built through the public API, since ``create_geometry_function_space`` is the only thing
        that makes a space on that index map and it uses the geometry dofmap. Keeping the
        predicate local and partition-independent is both cheaper and easier to reason about.

        Measured, for the record: a plain ``("Lagrange", 1, (gdim,))`` space is *rejected* here
        (its index map is a different object) even though its dofmap does match the geometry
        dofmap on triangles, quadrilaterals and tetrahedra. That is the conservative direction --
        it is then interpolated, which is exact -- so nothing is lost by not recognising it.

    Args:
        mesh: The mesh being moved.
        candidate: The object offered as a displacement.
        V_geom: ``mesh``'s geometry function space.

    Returns:
        Whether ``candidate`` can be written onto the geometry directly.
    """
    if not isinstance(candidate, dolfinx.fem.Function):
        return False
    space = candidate.function_space
    candidate_map = space.dofmap.index_map
    geometry_map = mesh.geometry.index_map()
    same_map = getattr(candidate_map, "_cpp_object", candidate_map) is getattr(
        geometry_map, "_cpp_object", geometry_map
    )
    # Note: Should really compare dofmap arrays, but expensive and not parition
    # independent. Would require a global reduction.
    return (
        space.mesh is mesh
        and same_map
        and space.dofmap.index_map_bs == V_geom.dofmap.index_map_bs
        and space.element == V_geom.element
    )


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


def _reject_blocks_predating_annotation(mesh: dolfinx.mesh.Mesh) -> None:
    """Refuse to move a mesh that has tape blocks posed on it from before it was annotated.

    Blocks skip the mesh dependency when the mesh is not annotated, and record the domain they
    were built on instead (``_unannotated_domain``). If any of them names this mesh, the tape
    cannot be replayed correctly once the geometry starts changing, and no later call can
    repair it, so {py:func}`~dolfinx_adjoint.move` raises rather than move the mesh.

    Args:
        mesh: The mesh about to be moved.

    Raises:
        RuntimeError: If such a block is on the working tape.
    """
    domain = mesh.ufl_domain()
    if domain is None:
        return
    ufl_id = domain.ufl_id()
    stale = [
        type(block).__name__
        for block in get_working_tape().get_blocks()
        if getattr(block, "_unannotated_domain", None) == ufl_id
    ]
    if stale:
        raise RuntimeError(
            f"This mesh already has {len(stale)} block(s) recorded on the tape "
            f"({', '.join(sorted(set(stale)))}), built before the mesh was annotated. Those "
            "blocks do not depend on the mesh, so replaying the tape will not rewind the "
            "geometry before re-running them and the gradient would be silently wrong from the "
            "second evaluation onwards. Wrap the mesh with dolfinx_adjoint.Mesh(mesh) before "
            "posing anything on it, or clear the tape and rebuild."
        )


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
        RuntimeError: If the tape holds a block posed on ``mesh`` from before it was tracked.
            Such a block does not depend on the mesh, so replaying the tape would re-run it on
            whatever geometry the previous replay left behind.

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
    _reject_blocks_predating_annotation(overloaded)
    geometry_disp = _as_geometry_displacement(overloaded, displacement, V_geom, True)

    overloaded_disp = pyadjoint.create_overloaded_object(geometry_disp)
    block = MoveBlock(overloaded, overloaded_disp, ad_block_tag=ad_block_tag)
    get_working_tape().add_block(block)

    with stop_annotating():
        apply_displacement(overloaded, geometry_disp)

    block.add_output(overloaded.create_block_variable())

    return overloaded
