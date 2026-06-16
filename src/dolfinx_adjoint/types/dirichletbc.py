import dolfinx
import numpy as np
import numpy.typing as npt
import pyadjoint
import ufl
from pyadjoint.overloaded_type import FloatingType

from ..blocks.dirichletbc import DirichletBCBlock


def extract_dtype(expr: ufl.core.expr.Expr) -> npt.DTypeLike:
    """Extract the dtype from an expression.

    Looks for any constants or coefficients and returning their dtype.
    This is necessary for determining which DOLFINx DirichletBC constructor
    to use when packing UFL expressions into DOLFINx Expressions for use in
    BC reconstruction.
    """
    consts = ufl.algorithms.analysis.extract_constants(expr)
    for c in consts:
        if hasattr(c, "dtype"):
            return c.dtype
    coeffs = ufl.algorithms.extract_coefficients(expr)
    for c in coeffs:
        if hasattr(c, "dtype"):
            return c.dtype
    raise ValueError(
        "Could not extract dtype from expression, "
        "please ensure that all constants and coefficients have a "
        "dtype attribute"
    )


class DirichletBC(dolfinx.fem.DirichletBC, FloatingType):
    def __init__(self, g, dofs, V, name="dirichletbc", **kwargs):
        self.name = name
        self._ufl_space = V.ufl_function_space()

        if not isinstance(g, (dolfinx.fem.Function, dolfinx.fem.Constant, int, float, complex)):
            self._ufl_expr = g
        else:
            self._ufl_expr = None

        if V.component() != []:
            V_sub, sub_to_parent = V.collapse()
            if len(sub_to_parent) != 1:
                raise NotImplementedError("Mixed topology is not supported for reconstructing BCs")
            else:
                sub_to_parent = sub_to_parent[0]
                parent_to_sub = np.full(
                    (V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts) * V.dofmap.index_map_bs,
                    -1,
                    dtype=np.int32,
                )
                parent_to_sub[sub_to_parent] = np.arange(len(sub_to_parent))
                sub_dofs = parent_to_sub[dofs]
                dofs = (dofs, sub_dofs)

        if isinstance(g, (dolfinx.fem.Function, dolfinx.fem.Constant)):
            val = g
            self._pack_expression = None
        else:
            val = dolfinx.fem.Function(V_sub if V.component() != [] else V, name=f"bc_{str(g)}")
            self._pack_expression = dolfinx.fem.Expression(g, V.element.interpolation_points())
            val.interpolate(self._pack_expression)

        dtype = extract_dtype(g)
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

        # Save internal references for dynamic C++ object generation
        self._g_val = val
        self._dofs_array = dofs
        self._V_space = V
        self._bctype = bctype

        # Initialize FEniCSx wrapper. This will trigger our _cpp_object.setter
        super().__init__(self._generate_cpp_object())

        annotate = kwargs.pop("annotate", True)
        annotate = annotate and pyadjoint.annotate_tape()

        FloatingType.__init__(
            self,
            V,
            val,
            dtype=dtype,
            block_class=kwargs.pop("block_class", DirichletBCBlock),
            _ad_floating_active=False,
            _ad_args=kwargs.pop("_ad_args", (val, dofs, V)),
            annotate=annotate,
            **kwargs,
        )

        if annotate:
            self._ad_annotate_block()

    def _generate_cpp_object(self):
        """Dynamically construct a C++ BC reflecting the current array memory."""
        val_cpp = self._g_val._cpp_object if hasattr(self._g_val, "_cpp_object") else self._g_val
        if isinstance(self._g_val, dolfinx.fem.Function):
            return self._bctype(val_cpp, self._dofs_array)
        else:
            try:
                return self._bctype(val_cpp, self._dofs_array, self._V_space._cpp_object)
            except TypeError:
                return self._bctype(val_cpp, self._dofs_array)

    @property
    def _cpp_object(self):
        # Solvers internally read this property every time they assemble/set_bcs
        return self._generate_cpp_object()

    @_cpp_object.setter
    def _cpp_object(self, value):
        # Absorb the assignment from dolfinx.fem.DirichletBC.__init__
        self._initial_cpp_object = value

    def _ad_create_checkpoint(self):
        return self

    def _ad_restore_at_checkpoint(self, checkpoint):
        return self


def dirichletbc(
    value: ufl.core.expr.Expr,
    dofs: npt.NDArray[np.int32],
    V: dolfinx.fem.FunctionSpace | None = None,
    **kwargs,
) -> DirichletBC:
    """Overloaded DirichletBC so that we can reconstruct BCs with UFL expressions.

    .. note::
        This class is user-facing.

    :param value: A UFL expression representing the boundary condition.
    :param dofs: An array of degree-of-freedom indices in `V` where the BC should be applied.
    :param V: The function space on which the BC applies. It can be a subspace of a mixed/blocked space.
    """
    if isinstance(value, dolfinx.fem.Function):
        V = value.function_space
    return DirichletBC(value, dofs, V, **kwargs)
