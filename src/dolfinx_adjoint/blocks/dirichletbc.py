import dolfinx
import numpy as np
import numpy.typing as npt
from pyadjoint.block import Block


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
