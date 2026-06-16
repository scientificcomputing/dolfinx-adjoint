from pyadjoint.block import Block


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
        return inputs[0] if inputs else None

    def recompute_component(self, inputs, block_variable, idx, prepared):
        # PyAdjoint relies on checkpoints. The dynamic `_cpp_object` property
        # in DirichletBC ensures FEniCSx handles the C++ updates internally.
        return block_variable.saved_output

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
