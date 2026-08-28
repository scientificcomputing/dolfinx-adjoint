import dolfinx
import numpy as np
import numpy.typing as npt
from pyadjoint.block import Block


def sync_bc_values(bcs, dependencies) -> None:
    """Refresh each BC's live backing Function from its value at this recorded position.

    A DirichletBC's C++ binding reads bc.g's array directly, by reference, not through the tape,
    so it has to be refreshed by hand before every solve that is not the original one. This must
    read from the calling block's own pinned dependencies (dependencies, i.e. self.get_dependencies())
    rather than bc.g.block_variable directly: that property always points at bc.g's most recently
    created BlockVariable, which -- once the full tape has been recorded -- is simply the last
    timestep's, regardless of which point in the replay this call is for.
    """
    values = {dep.output: dep.saved_output for dep in dependencies}
    for bc in bcs:
        value = values.get(bc.g)
        if value is not None:
            bc.g.x.array[:] = value.x.array[:]


class DirichletBCBlock(Block):
    """A block representing a DirichletBC in the adjoint framework.

    Args:
        value: The value of the Dirichlet BC.
        dofs: An array of degree-of-freedom indices in `V` where the BC should be applied.
        V: The function space associated with the Dirichlet BC.
        ad_block_tag: An optional tag to identify this block in the adjoint framework.

    """

    def __init__(
        self,
        value: dolfinx.fem.Function | dolfinx.fem.Constant,
        dofs: npt.NDArray[np.int32],
        V: dolfinx.fem.FunctionSpace | None = None,
        ad_block_tag: str | None = None,
    ):
        super().__init__(ad_block_tag=ad_block_tag)
        self._dofs = dofs
        self._V = V
        self.add_dependency(value)

    @property
    def dofs(self):
        return self._dofs

    @property
    def V(self):
        return self._V

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return inputs[0] if inputs else None

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return block_variable.saved_output
