import dolfinx
import numpy as np
import numpy.typing as npt
from pyadjoint.block import Block


class DirichletBCBlock(Block):
    def __init__(
        self,
        value: dolfinx.fem.Function | dolfinx.fem.Constant,
        dofs: npt.NDArray[np.int32],
        V: dolfinx.fem.FunctionSpace | None = None,
        ad_block_tag=None,
    ):
        super().__init__(ad_block_tag=ad_block_tag)
        self.dofs = dofs
        self.V = V
        self.add_dependency(value)

    def prepare_recompute_component(self, inputs, relevant_outputs):
        return inputs[0] if inputs else None

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return block_variable.saved_output
