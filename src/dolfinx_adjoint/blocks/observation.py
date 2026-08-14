"""Tape block for the pointwise observation misfit."""

from __future__ import annotations

import typing

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt
from pyadjoint import Block
from pyadjoint.overloaded_type import create_overloaded_object

from ._vector import _SpecialVector, _vector

if typing.TYPE_CHECKING:  # pragma: no cover
    from ..observation import PointObservation

__all__ = ["PointObservationBlock"]


class PointObservationBlock(Block):
    r"""Block for :math:`J(u) = \frac{1}{2\sigma^2}\,\lVert W(Bu - d)\rVert^2`.

    The functional is quadratic in the state, so the derivatives are available in closed
    form: the adjoint is :math:`\sigma^{-2} B^T W^2 (Bu - d)` and the Hessian action is
    :math:`\sigma^{-2} B^T W^2 B \hat{u}`. No linearization point needs to be stored.

    Args:
        u: The observed state.
        observation: The observation operator :math:`B`.
        data: Measured values, already restricted to this rank's rows.
        noise_variance: :math:`\sigma^2`.
        weights: Optional per-row weights :math:`W`, restricted to this rank's rows.
        ad_block_tag: Optional tag for the block on the tape.
    """

    def __init__(
        self,
        u: dolfinx.fem.Function,
        observation: "PointObservation",
        data: npt.NDArray[np.float64],
        noise_variance: float,
        weights: npt.NDArray[np.float64] | None = None,
        ad_block_tag: str | None = None,
    ) -> None:
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_dependency(u)
        self.observation = observation
        self.data = data
        self.noise_variance = noise_variance
        self.weights = weights

    def __str__(self) -> str:
        return f"point_observation_misfit({self.observation.num_found} points)"

    def _residual(self, u: dolfinx.fem.Function) -> npt.NDArray[np.float64]:
        return self.observation.apply(u) - self.data

    def _apply_weights(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.weights is None:
            return values
        return self.weights * values

    def _transpose_action(self, residual: npt.NDArray[np.float64], scale: float) -> _SpecialVector:
        """:math:`\\mathrm{scale} \\cdot \\sigma^{-2} B^T W^2 r`, as a DOF vector."""
        V = self.observation.function_space
        out = _vector(V.dofmap.index_map, V.dofmap.bs, V, dtype=V.mesh.geometry.x.dtype)
        out.array[:] = 0.0
        # W is applied twice: once to the residual, once from differentiating ||W r||^2.
        weighted = self._apply_weights(self._apply_weights(residual)) * (scale / self.noise_variance)
        self.observation.apply_transpose(weighted, out=out)
        return out

    def recompute_component(self, inputs, block_variable, idx, prepared=None):
        from ..observation import misfit_value

        value = misfit_value(self.observation, inputs[0], self.data, self.noise_variance, self.weights)
        return create_overloaded_object(value)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = 1.0 if adj_inputs[0] is None else float(adj_inputs[0])
        return self._transpose_action(self._residual(inputs[0]), adj_input)

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        tlm_u = tlm_inputs[0]
        if tlm_u is None:
            return None
        residual = self._apply_weights(self._residual(inputs[0]))
        directional = self._apply_weights(self.observation.apply(tlm_u))
        local = float(np.dot(residual, directional)) / self.noise_variance
        return self.observation.comm.allreduce(local, op=MPI.SUM)

    def evaluate_hessian_component(
        self,
        inputs,
        hessian_inputs,
        adj_inputs,
        block_variable,
        idx,
        relevant_dependencies,
        prepared=None,
    ):
        hessian_input = 0.0 if hessian_inputs[0] is None else float(hessian_inputs[0])
        adj_input = 1.0 if adj_inputs[0] is None else float(adj_inputs[0])

        # Second-order seed, propagated through the first derivative ...
        out = self._transpose_action(self._residual(inputs[0]), hessian_input)
        # ... plus the curvature of J applied to the TLM direction.
        tlm_u = block_variable.tlm_value
        if tlm_u is not None:
            out.array[:] += self._transpose_action(self.observation.apply(tlm_u), adj_input).array[:]
        return out
