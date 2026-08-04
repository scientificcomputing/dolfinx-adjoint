import dolfinx
import numpy as np
import numpy.typing as npt
import pyadjoint
from packaging.version import Version
from pyadjoint.overloaded_type import FloatingType

from ..blocks.dirichletbc import DirichletBCBlock
from .function import Function


class DirichletBC(dolfinx.fem.DirichletBC, FloatingType):
    """A class overloading :py:class:`dolfinx.fem.DirichletBC` to support
    it being used as a control variable in the adjoint framework.

    Args:
        g: The value of the Dirichlet BC.
        dofs: An array of degree-of-freedom indices in `V` where the BC should be applied.
        **kwargs: Additional keyword arguments to pass to the
            :py:func:`pyadjoint.overloaded_type.FloatingType` constructor.

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

        bc_kwargs = {}
        # If dolfinx-version is 0.12 we need to pass the following
        # due to https://github.com/FEniCS/dolfinx/pull/4342/
        if Version(dolfinx.__version__).minor >= 11:
            bc_kwargs["V"] = g.function_space
            bc_kwargs["g"] = g

        super().__init__(bctype(g._cpp_object, dofs), **bc_kwargs)

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
    """Overloaded DirichletBC constructor that creates an adjoint-aware DirichletBC

    Args:
        value: The value of the Dirichlet BC. Should be a :py:class:`dolfinx_adjoint.Function`.
            This means you can also pass in a :py:class:`dolfinx_adjoint.Constant` but not
            a :py:class:`dolfinx.fem.Constant`.
        dofs: An array of degree-of-freedom indices in `V` where the BC should be applied.
        **kwargs: Additional keyword arguments to pass to the
            :py:class:`dolfinx_adjoint.types.dirichletbc.DirichletBC` constructor.


    """
    assert isinstance(value, Function), "value must be a dolfinx_adjoint.Function"
    return DirichletBC(value, dofs, **kwargs)
