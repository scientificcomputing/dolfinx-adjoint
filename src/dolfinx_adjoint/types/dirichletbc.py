import dolfinx
import numpy as np
import numpy.typing as npt
import pyadjoint
from pyadjoint.overloaded_type import FloatingType

from ..blocks.dirichletbc import DirichletBCBlock
from .function import Function


class DirichletBC(dolfinx.fem.DirichletBC, FloatingType):
    """A class overloading `dolfinx.fem.DirichletBC` to support it being used as a control variable
    in the adjoint framework.

    Args:
        g: The value of the Dirichlet BC.
        dofs: An array of degree-of-freedom indices in `V` where the BC should be applied.
        **kwargs: Additional keyword arguments to pass to the `pyadjoint.overloaded_type.FloatingType` constructor.

    """

    def __init__(self, g: Function, dofs: npt.NDArray[np.int32], **kwargs):
        dtype = g.dtype
        if np.issubdtype(dtype, np.float32):
            bctype = dolfinx.cpp.fem.DirichletBC_float32
        elif np.issubdtype(dtype, np.float64):
            bctype = dolfinx.cpp.fem.DirichletBC_float64
        elif np.issubdtype(dtype, np.complex64):
            bctype = dolfinx.cpp.fem.DirichletBC_complex64
        elif np.issubdtype(dtype, np.complex128):
            bctype = dolfinx.cpp.fem.DirichletBC_complex128
        else:
            raise NotImplementedError(f"Type {dtype} not supported.")

        kwargs = {}
        # If dolfinx-version is 0.12 we need to pass the following
        # due to https://github.com/FEniCS/dolfinx/pull/4342/
        # TODO: Add conditional check if we want backwards compatibility with dolfinx < 0.12
        kwargs["V"] = g.function_space
        kwargs["g"] = g

        super().__init__(bctype(g._cpp_object, dofs), **kwargs)

        annotate = kwargs.pop("annotate", True)
        annotate = annotate and pyadjoint.annotate_tape()

        FloatingType.__init__(
            self,
            g,
            dtype=dtype,
            block_class=kwargs.pop("block_class", DirichletBCBlock),
            _ad_floating_active=False,
            _ad_args=kwargs.pop("_ad_args", (g, dofs)),
            annotate=annotate,
            **kwargs,
        )

        if annotate:
            self._ad_annotate_block()

    def _ad_create_checkpoint(self):
        return self

    def _ad_restore_at_checkpoint(self, checkpoint):
        return self


def dirichletbc(value: Function, dofs: npt.NDArray[np.int32], **kwargs) -> DirichletBC:
    """Overloaded DirichletBC constructor that creates an adjoint-aware DirichletBC"""
    return DirichletBC(value, dofs, **kwargs)
