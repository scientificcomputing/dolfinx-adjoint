"""The transpose of scifem's surface-submesh extension, for the adjoint of ``transfer_from_boundary``."""

import basix
import dolfinx
import numpy as np
from scifem.compat import compute_integration_domains
from scifem.interpolation import SurfaceSubmeshExtension
from scifem.mesh import get_entity_map


class TransposableSurfaceSubmeshExtension(SurfaceSubmeshExtension):
    """``scifem.interpolation.SurfaceSubmeshExtension`` with its transpose.

    The extension is ``u[volume_dofs] += weights * (basis @ q[surface_dofs])`` followed by a
    reverse scatter, so its transpose reads the dual vector at ``volume_dofs``, scales it by the
    same ``weights``, contracts it with ``basis`` and adds it into ``surface_dofs``.
    """

    def apply_transpose(self, dual_volume: dolfinx.la.Vector, dual_surface: dolfinx.la.Vector) -> None:
        """Set ``dual_surface`` to the transpose of the extension applied to ``dual_volume``.

        Args:
            dual_volume: A dual vector on the volume space, read at its owned entries; its ghost
                entries are overwritten by a forward scatter.
            dual_surface: The result on the surface space, ghosts included.
        """
        dual_volume.scatter_forward()
        weighted = dual_volume.array[self.volume_dofs] * self.weights
        local = np.einsum("fnd,fn->fd", self.basis, weighted)
        dual_surface.array[:] = 0.0
        np.add.at(dual_surface.array, self.surface_dofs.reshape(-1), local.ravel())
        dual_surface.scatter_reverse(dolfinx.la.InsertMode.add)
        dual_surface.scatter_forward()


def create_surface_extension(
    V_boundary: dolfinx.fem.FunctionSpace, V: dolfinx.fem.FunctionSpace, entity_map
) -> TransposableSurfaceSubmeshExtension:
    """The extension from ``V_boundary``, on a facet submesh made by ``create_submesh``, into ``V``.

    Args:
        V_boundary: A space on the facet submesh.
        V: A continuous Lagrange space on the parent mesh.
        entity_map: The submesh-to-parent entity map returned by ``create_submesh``, in any
            DOLFINx version's form; ``scifem.mesh.get_entity_map`` and
            ``scifem.compat.compute_integration_domains`` handle the version differences.

    Returns:
        The extension, over every facet of the submesh owned by this process.

    Raises:
        ValueError: If ``V`` is not Lagrange.
    """
    if V.element.basix_element.family != basix.ElementFamily.P:
        # A Piola-mapped extension depends on the parent Jacobian, which BoundaryTransferBlock
        # does not differentiate.
        raise ValueError("transfer_from_boundary needs a Lagrange volume space V.")
    mesh, submesh = V.mesh, V_boundary.mesh
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, fdim + 1)
    num_facets = submesh.topology.index_map(fdim).size_local
    parent_facets = get_entity_map(entity_map)[:num_facets]
    entities = compute_integration_domains(dolfinx.fem.IntegralType.exterior_facet, mesh.topology, parent_facets)
    submesh_facets = np.arange(num_facets, dtype=np.int32)
    return TransposableSurfaceSubmeshExtension(V_boundary, V, submesh_facets, entities)
