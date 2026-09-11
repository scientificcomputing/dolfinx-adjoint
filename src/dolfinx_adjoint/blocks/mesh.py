from __future__ import annotations

import typing

import dolfinx
import numpy as np
import numpy.typing as npt
import pyadjoint
from pyadjoint.tape import no_annotations

from ..types.mesh import Mesh
from ._vector import _SpecialVector, _vector


def _displacement_vector(space: dolfinx.fem.FunctionSpace) -> _SpecialVector:
    """Allocate a zeroed sensitivity vector on ``space``."""
    vec = _vector(space.dofmap.index_map, space.dofmap.index_map_bs, function_space=space)
    vec.array[:] = 0.0
    return vec


class MoveBlock(pyadjoint.Block):
    r"""Block recording a displacement of a mesh's geometry, :math:`x \mapsto x + s`.

    The map is a translation in the displacement, so its derivative is the identity: the
    adjoint, tangent-linear and Hessian actions all pass their input straight through to
    both dependencies. All the geometry-specific differentiation lives in the blocks that
    consume the moved mesh -- they differentiate their forms with respect to
    {py:class}`ufl.SpatialCoordinate` -- not here.

    Args:
        mesh: The mesh being moved, already annotated.
        displacement: The displacement field, in the mesh's geometry function space.
        ad_block_tag: Tag for the block on the tape.
    """

    def __init__(
        self,
        mesh: Mesh,
        displacement: dolfinx.fem.Function,
        ad_block_tag: str | None = None,
    ):
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_dependency(mesh)
        self.add_dependency(displacement)

    def __str__(self) -> str:
        return "move(mesh, s)"

    def evaluate_adj_component(
        self,
        inputs: typing.Sequence[typing.Any],
        adj_inputs: typing.Sequence[typing.Any],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        prepared: typing.Any = None,
    ) -> typing.Any:
        """Pass the adjoint value through unchanged to both the mesh and the displacement."""
        return adj_inputs[0]

    def evaluate_tlm_component(
        self,
        inputs: typing.Sequence[typing.Any],
        tlm_inputs: typing.Sequence[typing.Any],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        prepared: typing.Any = None,
    ) -> typing.Any:
        """Sum the incoming tangent-linear directions of the mesh and the displacement.

        The result is a {py:class}`~dolfinx_adjoint.Function` in the geometry space, not a
        bare array: it is consumed as the direction of a
        {py:func}`ufl.derivative` with respect to {py:class}`ufl.SpatialCoordinate`, so it
        has to be something UFL can treat as a coefficient.

        Either dependency may carry no direction -- the mesh does not when the displacement
        is the control, which is the usual case -- so ``None`` entries are skipped rather
        than treated as zero vectors, and an all-``None`` input yields ``None``.
        """
        tlm_output = None
        for tlm_input in tlm_inputs:
            if tlm_input is None:
                continue
            if tlm_output is None:
                tlm_output = tlm_input._ad_copy()
            else:
                tlm_output.x.array[:] += tlm_input.x.array[:]
        return tlm_output

    def evaluate_hessian_component(
        self,
        inputs: typing.Sequence[typing.Any],
        hessian_inputs: typing.Sequence[typing.Any],
        adj_inputs: typing.Sequence[typing.Any],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        relevant_dependencies: typing.Sequence[typing.Any],
        prepared: typing.Any = None,
    ) -> typing.Any:
        """Identity, as for the adjoint: the map is linear, so it has no second derivative."""
        return hessian_inputs[0]

    @no_annotations
    def recompute_component(
        self,
        inputs: typing.Sequence[typing.Any],
        block_variable: pyadjoint.block_variable.BlockVariable,
        idx: int,
        prepared: typing.Any = None,
    ) -> npt.NDArray[np.floating]:
        """Re-apply the displacement to the undisplaced geometry.

        ``inputs[0]`` is the mesh itself, already rewound to the geometry it had *before*
        this block ran: reading a dependency's ``saved_output`` restores it from its
        checkpoint, and for a mesh that restore rewrites the coordinates in place. So the
        displacement is applied to the pre-move geometry, never accumulated on top of an
        earlier recompute -- which matters because a checkpoint schedule replays the
        forward many times.
        """
        from ..mesh import apply_displacement

        mesh, displacement = inputs[0], inputs[1]
        apply_displacement(mesh, displacement)
        return mesh._ad_create_checkpoint()
