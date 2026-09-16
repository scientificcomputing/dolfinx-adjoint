from __future__ import annotations

import typing
import weakref

import dolfinx
import numpy as np
import numpy.typing as npt
import ufl
from pyadjoint.overloaded_type import OverloadedType
from pyadjoint.tape import get_working_tape, no_annotations

__all__ = ["Mesh", "annotate_mesh", "geometry_function_space", "overloaded_mesh"]


# Maps a `ufl.Mesh` domain to the annotated `dolfinx.mesh.Mesh` carrying it, so that a
# block holding only a form can recover the mesh to depend on. Keyed by the domain's
# `ufl_id()` rather than the domain itself: `ufl.Mesh` hashes on that id, and holding the
# domain as a key would keep it alive for as long as the registry lives.
_annotated_meshes: weakref.WeakValueDictionary[int, "Mesh"] = weakref.WeakValueDictionary()


def geometry_function_space(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.FunctionSpace:
    """Return the function space a displacement of ``mesh``'s geometry lives in.

    This is {py:func}`scifem.mesh.create_geometry_function_space`'s space: it is built on
    the geometry dofmap, so its dofs correspond one-to-one, in order, with the rows of
    ``mesh.geometry.x``. That correspondence is what makes an assembled shape derivative
    and a displacement applied by {py:func}`~dolfinx_adjoint.move` the same vector.

    Args:
        mesh: The mesh whose geometry is to be displaced.

    Returns:
        The vector-valued function space of the mesh's coordinate element.
    """
    try:
        import scifem.mesh
    except ImportError as e:
        raise ImportError("scifem is required for shape control: pip install scifem") from e
    try:
        return scifem.mesh.create_geometry_function_space(mesh)
    except TypeError:
        # `create_geometry_function_space` hands `mesh.geometry.index_map()` to
        # `dolfinx.cpp.fem.DofMap`. From DOLFINx 0.12 (FEniCS/dolfinx#4496) the accessor
        # returns the Python `dolfinx.common.IndexMap` wrapper while the nanobind constructor
        # still wants the raw `dolfinx.cpp.common.IndexMap`, so the call raises before the
        # space exists -- and this is the one object every shape workflow starts from.
        # `blocks._vector` routes around the same change by delegating to a factory that
        # follows the release; there is no such factory here, so unwrap for the duration of
        # the call. Harmless on 0.11, where the accessor already returns the raw object and
        # `getattr(..., "_cpp_object", ...)` is the identity.
        return _create_geometry_function_space_unwrapped(mesh)


def _create_geometry_function_space_unwrapped(mesh: dolfinx.mesh.Mesh) -> dolfinx.fem.FunctionSpace:
    """Build scifem's geometry function space with the index map unwrapped.

    See {py:func}`geometry_function_space`. Retries the same scifem call with
    ``dolfinx.cpp.fem.DofMap`` temporarily replaced by a subclass that accepts either the
    wrapped or the raw index map, so a single DOLFINx-version difference does not take the
    whole feature out. Restores the real class on the way out, success or not.

    Args:
        mesh: The mesh whose geometry is to be displaced.

    Returns:
        The vector-valued function space of the mesh's coordinate element.
    """
    import scifem.mesh

    real = dolfinx.cpp.fem.DofMap

    class _UnwrappingDofMap(real):  # type: ignore[misc,valid-type]
        def __init__(self, layout, index_map, index_map_bs, dofmap, bs):
            super().__init__(
                layout, getattr(index_map, "_cpp_object", index_map), index_map_bs, dofmap, bs
            )

    dolfinx.cpp.fem.DofMap = _UnwrappingDofMap  # type: ignore[misc]
    scifem.mesh.dolfinx.cpp.fem.DofMap = _UnwrappingDofMap
    try:
        return scifem.mesh.create_geometry_function_space(mesh)
    finally:
        dolfinx.cpp.fem.DofMap = real  # type: ignore[misc]
        scifem.mesh.dolfinx.cpp.fem.DofMap = real


class Mesh(dolfinx.mesh.Mesh, OverloadedType):
    """A {py:class}`dolfinx.mesh.Mesh` extended so that its geometry can be differentiated
    through.

    Instances are not constructed directly. An existing mesh is promoted in place by
    {py:func}`annotate_mesh`, which is called for you by {py:func}`~dolfinx_adjoint.move`.

    Note:
        The base order is load-bearing and must stay
        ``(dolfinx.mesh.Mesh, OverloadedType)``. CPython only permits assigning to
        ``__class__`` between types whose instance layouts agree, and with
        {py:class}`~pyadjoint.OverloadedType` listed first the resulting layout no longer
        matches a plain {py:class}`dolfinx.mesh.Mesh` -- the promotion in
        {py:func}`annotate_mesh` then fails with ``object layout differs``.

    Note:
        The value this type carries on the tape is the mesh's coordinates. It is not
        itself usable as a {py:class}`pyadjoint.Control`; the control in a shape
        optimization is the displacement passed to {py:func}`~dolfinx_adjoint.move`, and
        this type is the intermediate block variable linking that displacement to every
        form posed on the mesh.
    """

    def _ad_init_mesh(self) -> None:
        """Initialise the pyadjoint side of an already-constructed mesh.

        Separate from ``__init__`` because instances are produced by reassigning
        ``__class__`` on a live mesh (see {py:func}`annotate_mesh`), so the DOLFINx
        constructor has already run and must not run again.
        """
        OverloadedType.__init__(self)
        self._ad_coordinate_space: dolfinx.fem.FunctionSpace | None = None

    def _ad_function_space(self) -> dolfinx.fem.FunctionSpace:
        """The geometry function space, built once and cached on the mesh."""
        if self._ad_coordinate_space is None:
            self._ad_coordinate_space = geometry_function_space(self)
        return self._ad_coordinate_space

    @no_annotations
    def _ad_create_checkpoint(self) -> npt.NDArray[np.floating]:
        """Checkpoint the geometry by copying the coordinate array.

        A plain array copy, not a {py:class}`~dolfinx_adjoint.Function`: the geometry is
        owned by the mesh and there is no coordinate Function in DOLFINx to hand back.
        """
        return self.geometry.x.copy()

    @no_annotations
    def _ad_restore_at_checkpoint(self, checkpoint: npt.NDArray[np.floating]) -> "Mesh":
        """Restore the geometry from a checkpoint, returning the mesh itself.

        The mesh is restored in place rather than rebuilt: every form, function space and
        compiled kernel already built on it holds the same mesh object, so replacing it
        would silently leave them pointing at the old geometry.
        """
        self.geometry.x[:] = checkpoint
        return self

    def _ad_dot(self, other: "Mesh", options: dict | None = None) -> float:
        raise NotImplementedError(
            "A mesh cannot be used as a Control directly. Use the displacement passed to "
            "dolfinx_adjoint.move() as the control instead."
        )


def annotate_mesh(mesh: dolfinx.mesh.Mesh) -> Mesh:
    """Promote ``mesh`` in place so that its geometry can be differentiated through.

    The mesh's ``__class__`` is reassigned to {py:class}`Mesh`. Promoting in place, rather
    than returning a new object, is what lets a mesh created by any of DOLFINx's many
    entry points -- {py:func}`dolfinx.mesh.create_unit_square`, ``gmshio``, XDMF, a
    submesh -- take part in a shape optimization without this package having to overload
    each of them. Every function space, form and compiled kernel already built on the mesh
    keeps working, since the object's identity is unchanged.

    Idempotent: a mesh that is already annotated is returned unchanged, keeping the block
    variable it has accumulated on the tape.

    Must be called before anything is posed on the mesh. A block built earlier cannot take the
    mesh as a dependency, so replaying the tape does not rewind the geometry before re-running
    it -- the block is re-evaluated on whatever the previous replay left behind, and the
    gradient drifts from the second distinct control value onwards while every Taylor test
    still passes. :py:func:`_reject_blocks_predating_annotation` refuses that up front.

    Args:
        mesh: The mesh to promote.

    Returns:
        The same object, now an overloaded {py:class}`Mesh`.

    Raises:
        RuntimeError: If the working tape already holds a block posed on ``mesh``.
    """
    if not isinstance(mesh, Mesh):
        _reject_blocks_predating_annotation(mesh)
        mesh.__class__ = Mesh  # type: ignore[assignment]
        typing.cast(Mesh, mesh)._ad_init_mesh()
    domain = mesh.ufl_domain()
    assert domain is not None
    _annotated_meshes[domain.ufl_id()] = typing.cast(Mesh, mesh)
    return typing.cast(Mesh, mesh)


def _reject_blocks_predating_annotation(mesh: dolfinx.mesh.Mesh) -> None:
    """Refuse to annotate a mesh that already has tape blocks posed on it.

    Blocks that would have taken a mesh dependency record the domain they were built on when
    the lookup came back empty (``_unannotated_domain``). If any of them names this mesh, the
    tape cannot be replayed correctly and no later call can repair it, so this raises instead.

    Args:
        mesh: The mesh about to be promoted.

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
            "second evaluation onwards. Call dolfinx_adjoint.annotate_mesh(mesh) (or move()) "
            "before posing anything on the mesh, or clear the tape and rebuild."
        )


def overloaded_mesh(domain: ufl.Mesh | None) -> Mesh | None:
    """Return the annotated mesh carrying ``domain``, or ``None`` if there is none.

    A block generally holds a form, and a form knows only its {py:class}`ufl.Mesh` domain
    -- whose ``ufl_cargo()`` is the *C++* mesh, not the Python one that carries the tape's
    block variable. This is the lookup back to the Python mesh, and returning ``None`` is
    the ordinary answer for any problem that is not a shape optimization.

    Args:
        domain: The form's UFL domain, or ``None``.

    Returns:
        The annotated mesh, or ``None`` if ``domain`` is ``None`` or its mesh was never
        passed to {py:func}`~dolfinx_adjoint.move`.
    """
    if domain is None:
        return None
    return _annotated_meshes.get(domain.ufl_id())
