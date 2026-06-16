import dolfinx
from pyadjoint.block import Block
from pyadjoint.tape import stop_annotating


class DirichletBCBlock(Block):
    def __init__(self, value, dofs, V=None, ad_block_tag=None):
        super().__init__(ad_block_tag=ad_block_tag)
        self.dofs = dofs
        self.V = V

        # Add dependency on the underlying overloaded Function or Constant
        self.add_dependency(value)

    def __str__(self):
        return "dirichletbc"

    def prepare_recompute_component(self, inputs, relevant_outputs):
        # Extract the checkpointed `value` from the inputs
        return inputs[0] if inputs else None

    def recompute_component(self, inputs, block_variable, idx, prepared):
        # Re-instantiate the FEniCSx boundary condition with the rewound tape value
        with stop_annotating():
            return dolfinx.fem.dirichletbc(prepared, self.dofs, self.V)

    # Empty stubs required by the PyAdjoint Block interface for passive nodes
    def prepare_evaluate_adj(self, inputs, adj_inputs, relevant_dependencies):
        pass

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        pass

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        pass

    def evaluate_hessian_component(
        self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None
    ):
        pass
