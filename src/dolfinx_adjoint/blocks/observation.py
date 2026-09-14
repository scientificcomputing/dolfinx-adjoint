"""Tape block for the pointwise observation misfit."""

from __future__ import annotations

from mpi4py import MPI

import dolfinx
import numpy as np
import numpy.typing as npt
from pyadjoint import Block
from pyadjoint.overloaded_type import create_overloaded_object

from ..types.function import _create_function

__all__ = ["PointObservationMisfitBlock", "misfit_value"]


def misfit_value(
    v: dolfinx.fem.Function,
    data: npt.NDArray[np.float64],
    noise_variance: float,
    weights: npt.NDArray[np.float64] | None,
) -> float:
    """The misfit value, summed over ranks. Used by both the forward pass and the block.

    Args:
        v: The observed state, already evaluated at the observation points (i.e. the output
            of an observation operator such as :func:`dolfinx_adjoint.interpolate_nonmatching`),
            restricted to this rank's owned rows.
        data: Measured values, restricted to this rank's rows.
        noise_variance: :math:`\\sigma^2`.
        weights: Optional per-row weights, restricted to this rank's rows.
    """
    residual = v.x.array[: data.shape[0]] - data
    if weights is not None:
        residual = weights * residual
    local = 0.5 * float(np.dot(residual, residual)) / noise_variance
    return v.function_space.mesh.comm.allreduce(local, op=MPI.SUM)


class PointObservationMisfitBlock(Block):
    r"""Block for :math:`J(v) = \frac{1}{2\sigma^2}\,\lVert W(v - d)\rVert^2`.

    ``v`` is the observed state -- the output of an observation operator, evaluated at the
    observation points -- not the state itself; this block never touches the observation
    operator or its transfer matrix. The functional is quadratic in ``v``, so the
    derivatives are available in closed form: the adjoint is :math:`\sigma^{-2} W^2 (v - d)`
    and the Hessian action is :math:`\sigma^{-2} W^2 \hat{v}`. No linearization point needs
    to be stored beyond ``v`` itself, which pyadjoint already keeps.

    Args:
        v: The observed state, in the observation space.
        data: Measured values, already restricted to this rank's rows.
        noise_variance: :math:`\sigma^2`.
        weights: Optional per-row weights :math:`W`, restricted to this rank's rows.
        ad_block_tag: Optional tag for the block on the tape.
    """

    def __init__(
        self,
        v: dolfinx.fem.Function,
        data: npt.NDArray[np.float64],
        noise_variance: float,
        weights: npt.NDArray[np.float64] | None = None,
        ad_block_tag: str | None = None,
    ) -> None:
        super().__init__(ad_block_tag=ad_block_tag)
        self.add_dependency(v)
        self.observation_space = v.function_space
        # Copied: the tape holds this block for as long as it is needed for differentiation,
        # so a caller mutating a data/weights buffer it passed in (for instance reusing one
        # local-length array across a time-stepping loop) must not retroactively change it.
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.noise_variance = noise_variance
        self.weights = None if weights is None else np.array(weights, dtype=np.float64, copy=True)

    def __str__(self) -> str:
        return f"point_observation_misfit({self.data.shape[0]} local rows)"

    def _residual(self, v: dolfinx.fem.Function) -> npt.NDArray[np.float64]:
        return v.x.array[: self.data.shape[0]] - self.data

    def _apply_weights(self, values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.weights is None:
            return values
        return self.weights * values

    def _weighted_row(self, values: npt.NDArray[np.float64], scale: float) -> npt.NDArray[np.float64]:
        """:math:`\\mathrm{scale} \\cdot \\sigma^{-2} W^2 v`, in ``v``'s row layout."""
        # W is applied twice: once to the residual, once from differentiating ||W r||^2.
        return self._apply_weights(self._apply_weights(values)) * (scale / self.noise_variance)

    def _as_output(self, values: npt.NDArray[np.float64]) -> dolfinx.la.Vector:
        """``values`` as a freshly built vector on the observation space.

        Built fresh on every call rather than reused: pyadjoint stores the returned vector by
        reference on the tape (`BlockVariable.add_adj_output`/`add_hessian_output`), so handing
        back the same buffer twice would let a later call silently overwrite a value pyadjoint
        is still holding.
        """
        out = _create_function(self.observation_space)
        out.x.array[: self.data.shape[0]] = values
        out.x.scatter_forward()
        return out.x

    def recompute_component(self, inputs, block_variable, idx, prepared=None):
        value = misfit_value(inputs[0], self.data, self.noise_variance, self.weights)
        return create_overloaded_object(value)

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        adj_input = 1.0 if adj_inputs[0] is None else float(adj_inputs[0])
        return self._as_output(self._weighted_row(self._residual(inputs[0]), adj_input))

    def evaluate_tlm_component(self, inputs, tlm_inputs, block_variable, idx, prepared=None):
        tlm_v = tlm_inputs[0]
        if tlm_v is None:
            return None
        residual = self._apply_weights(self._residual(inputs[0]))
        directional = self._apply_weights(tlm_v.x.array[: self.data.shape[0]])
        local = float(np.dot(residual, directional)) / self.noise_variance
        return inputs[0].function_space.mesh.comm.allreduce(local, op=MPI.SUM)

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

        # Second-order seed, propagated through the first derivative, plus the curvature of J
        # applied to the TLM direction, fused into one combined row before returning -- so
        # that the interpolation block's own B^T action (a full communication round-trip)
        # runs once per Hessian action, not once per contribution.
        combined = self._weighted_row(self._residual(inputs[0]), hessian_input)
        tlm_v = block_variable.tlm_value
        if tlm_v is not None:
            combined = combined + self._weighted_row(tlm_v.x.array[: self.data.shape[0]], adj_input)
        return self._as_output(combined)
