import dolfinx
import numpy as np
import numpy.typing as npt
from packaging.version import Version
from pyadjoint.block import Block


def build_cpp_bc_and_kwargs(
    g: dolfinx.fem.Function, dofs: npt.NDArray[np.int32], V: dolfinx.fem.FunctionSpace | None
) -> tuple[
    dolfinx.cpp.fem.DirichletBC_float32
    | dolfinx.cpp.fem.DirichletBC_float64
    | dolfinx.cpp.fem.DirichletBC_complex64
    | dolfinx.cpp.fem.DirichletBC_complex128,
    dict,
]:
    """Build the cpp Dirichlet bc object and the kwargs a Python
    :py:class:`dolfinx.fem.DirichletBC` wrapper needs, decoupling the *Python-level*
    constrained space ``V`` from the cpp-level construction.

    Shared by :py:class:`~dolfinx_adjoint.types.dirichletbc.DirichletBC` (the tape-tracked
    constructor) and :py:meth:`DirichletBCBlock.evaluate_tlm_component` (a plain,
    untracked bc built fresh each TLM call): both need the same fix for the same reason.

    The cpp constructor is always called the V-less way regardless of whether the caller
    passed ``V``: the 3-arg cpp overload for a `Function`-valued ``g`` requires ``dofs`` to
    be a *paired* ``(dofs_in_V, dofs_in_g_space)`` sequence -- the mechanism for
    constraining a sub-space with a value on its collapsed counterpart, not what a flat
    ``dofs`` array plus a broadcast-style value (e.g. a
    :py:class:`~dolfinx_adjoint.Constant`, itself a `Function` on a single-dof real space)
    needs. The V-less overload already broadcasts a real-space `Function`'s single value
    across an arbitrary flat ``dofs`` array correctly (the whole point of implementing
    `Constant` via a real space). ``V`` is instead threaded through purely at the *Python*
    level, which :py:class:`dolfinx.fem.DirichletBC` keeps entirely independent of the cpp
    object: ``bc.function_space`` is whatever ``V`` is passed to the Python wrapper, never
    introspected from the cpp bc.
    """
    dtype = g.dtype
    V_used = V if V is not None else g.function_space
    cpp_bc: (
        dolfinx.cpp.fem.DirichletBC_float32
        | dolfinx.cpp.fem.DirichletBC_float64
        | dolfinx.cpp.fem.DirichletBC_complex64
        | dolfinx.cpp.fem.DirichletBC_complex128
    )
    # cpp_bc is constructed inside each branch, immediately after its own isinstance
    # assert, rather than deferred to one call after the if/elif chain -- mypy can only
    # narrow g._cpp_object's union type within the branch the assert itself is in.
    if np.issubdtype(dtype, np.float32):
        assert isinstance(g._cpp_object, dolfinx.cpp.fem.Function_float32)
        cpp_bc = dolfinx.cpp.fem.DirichletBC_float32(g._cpp_object, dofs)
    elif np.issubdtype(dtype, np.float64):
        assert isinstance(g._cpp_object, dolfinx.cpp.fem.Function_float64)
        cpp_bc = dolfinx.cpp.fem.DirichletBC_float64(g._cpp_object, dofs)
    elif np.issubdtype(dtype, np.complex64):
        assert isinstance(g._cpp_object, dolfinx.cpp.fem.Function_complex64)
        cpp_bc = dolfinx.cpp.fem.DirichletBC_complex64(g._cpp_object, dofs)
    elif np.issubdtype(dtype, np.complex128):
        assert isinstance(g._cpp_object, dolfinx.cpp.fem.Function_complex128)
        cpp_bc = dolfinx.cpp.fem.DirichletBC_complex128(g._cpp_object, dofs)
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")

    bc_kwargs: dict = {}
    # If dolfinx-version is 0.12 we need to pass the following
    # due to https://github.com/FEniCS/dolfinx/pull/4342/
    if Version(dolfinx.__version__).minor > 11:
        bc_kwargs["V"] = V_used
        bc_kwargs["g"] = g
    return cpp_bc, bc_kwargs


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
        """Return the (aliased) bc, having first resynced its live ``g.x.array`` from the
        value's own checkpoint at this tape position.

        ``DirichletBC._ad_create_checkpoint``/``_ad_restore_at_checkpoint``
        (types/dirichletbc.py) both ``return self`` -- the bc's own "checkpoint" aliases
        the *live* bc object rather than snapshotting a value -- so nothing else writes a
        replayed/perturbed value back into ``bc.g``'s array. ``prepared`` (this block's
        single dependency's own, correctly-checkpointed value, from
        ``prepare_recompute_component``) is exactly that value: writing it into ``bc.g``
        here, at the position in the tape this block itself occupies (always *before* any
        solve block that consumes ``bc``, since the bc must be constructed first), is what
        makes a later solve block see the right value regardless of which tape position is
        being replayed. Without this, replaying an earlier position (a Taylor-test
        perturbation, an optimizer step, a later solve in a time loop reusing the same bc)
        would silently keep using whatever ``g.x.array`` currently holds -- observed
        directly as an all-zero Taylor-test residual, since the "perturbed" forward solve
        never actually saw a different bc value.
        """
        bc = block_variable.saved_output
        if prepared is not None:
            bc.g.x.array[:] = prepared.x.array[:]
            bc.g.x.scatter_forward()
        return bc

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        """Return this bc's own tangent-linear value: itself a (plain, untracked)
        DirichletBC, with its value replaced by the perturbation direction.

        A bc perturbation is *not* an ordinary right-hand-side contribution -- it enters
        the tangent-linear solve as an inhomogeneous condition (`u_dot = g_dot` on this
        bc's dofs, see ``HomogeneousBCLinearProblem.tlm_bcs``/``solve()``), consumed by
        ``_ProblemBlockBase.prepare_evaluate_tlm``. Built plain (``dolfinx.fem.dirichletbc``,
        not the overloaded ``dolfinx_adjoint`` one) so it is never itself tape-recorded --
        it exists only for this one TLM evaluation, not as a new control.
        """
        tlm_input = tlm_inputs[0]
        if tlm_input is None:
            return None
        cpp_bc, bc_kwargs = build_cpp_bc_and_kwargs(tlm_input, self._dofs, self._V)
        return dolfinx.fem.DirichletBC(cpp_bc, **bc_kwargs)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        """Return this bc's contribution to the adjoint action.

        ``adj_inputs[0]`` is the boundary reaction the consuming solve block(s) already
        computed (see ``_ProblemBlockBase._mask_reaction_to_bc``/``prepare_evaluate_adj``),
        already living on this bc's own constrained space -- no reduction needed here:
        ``types.dirichletbc._pack_bc_value`` always packs the bc's value into a Function on
        exactly that space before this block is ever created, whatever the original value
        was (a plain `Function`, a broadcasting `Constant`, or a general expression), so
        this block's single dependency and the masked reaction always agree.
        """
        return adj_inputs[0]

    def evaluate_hessian_component(
        self, inputs, hessian_inputs, adj_inputs, block_variable, idx, relevant_dependencies, prepared=None
    ):
        """Return this bc's contribution to the Hessian action.

        Same pass-through as ``evaluate_adj_component``, applied to the second-order
        boundary reaction (``prepare_evaluate_hessian``'s ``_adj_sol2_bdy``) instead of the
        first-order one -- correct because ``d2F/dm2 = d2F/dudm = 0`` for a Dirichlet bc
        control (it only ever enters the residual through its own linear right-hand side),
        so the boundary reaction against the second-order adjoint solution is the *entire*
        Hessian-action contribution.
        """
        return hessian_inputs[0]
