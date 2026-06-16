import dolfinx
import numpy as np
import numpy.typing as npt
import ufl
from pyadjoint.overloaded_type import (
    FloatingType,
    create_overloaded_object,
)

# from pyadjoint.block_variable import BlockVariable
# from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating


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
    _pack_expression: dolfinx.fem.Expression | None
    _ufl_expr: ufl.core.expr.Expr | None  # Store original UFL expression

    def __init__(
        self,
        g: ufl.core.expr.Expr,
        dofs: npt.NDArray[np.int32],
        V: dolfinx.fem.FunctionSpace,
        name: str = "dirichletbc",
        **kwargs,
    ):
        """
        Create an Irksome compatible DirichletBC from an existing DOLFINx bc.

        :param g: The boundary condition expression
        :param dofs: An array of degree-of-freedom indices in V
        :param V: The space to construct the BC on.
        :param name: The name of the boundary condition.
        """
        # Attach UFL function space (to be able to reconstruct functions and constants on the same UFL domain)
        self.name = name
        self._ufl_space = V.ufl_function_space()

        # Store original UFL expression for time-varying BCs
        if not isinstance(g, (dolfinx.fem.Function, dolfinx.fem.Constant, int, float, complex)):
            self._ufl_expr = g  # Save the symbolic expression
        else:
            self._ufl_expr = None
        self._ufl_space = V.ufl_function_space()

        # If reconstructing with a sub space, we need to get the subspace dof indices
        # If working with a subspace of a single stage, we need to create the (parent_dof, sub_dof) mapping
        if V.component() != []:
            V_sub, sub_to_parent = V.collapse()
            if len(sub_to_parent) != 1:
                msg = "Mixed topology is not supported for reconstructing BCs with UFL expressions"
                raise NotImplementedError(msg)
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

        # If we are not reconstructing the BC with a new value,
        #  we can reuse existing C++ objects
        self._pack_expression = None

        # If we are reconstructing the BC with a new value,
        # we need to check if the new value is a DOLFINx function or Constant.
        # If True, we do not need to do anything for reconstruction.
        if isinstance(g, (dolfinx.fem.Function, dolfinx.fem.Constant)):
            val = g
            self._pack_expression = None
        else:
            # If not, we need to take the ufl.core.expr.Expr and pack it into a DOLFINx Expression
            if V.component() != []:
                val = dolfinx.fem.Function(V_sub, name=f"bc_{str(g)}")._cpp_object
            else:
                val = dolfinx.fem.Function(V, name=f"bc_{str(g)}")._cpp_object
            self._pack_expression = dolfinx.fem.Expression(g, V.element.interpolation_points)

        # Get correct C++ implementation based on dtype of expression
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

        if (
            isinstance(
                val,
                (
                    dolfinx.cpp.fem.Function_complex128,
                    dolfinx.cpp.fem.Function_complex64,
                    dolfinx.cpp.fem.Function_float32,
                    dolfinx.cpp.fem.Function_float64,
                ),
            )
            and val.function_space == V._cpp_object
        ):
            new_cpp_object = bctype(val, dofs)
        elif isinstance(val, dolfinx.fem.Function):
            new_cpp_object = bctype(val._cpp_object, dofs)
        else:
            # Depending on your FEniCSx version, the C++ constructor might strictly
            # expect the C++ FunctionSpace instead of the Python FunctionSpace wrapper.
            try:
                new_cpp_object = bctype(val, dofs, V._cpp_object)
            except TypeError:
                new_cpp_object = bctype(val._cpp_object, dofs, V._cpp_object)

        # 4. Initialize the parent dolfinx.fem.DirichletBC wrapper with the newly minted C++ object
        super().__init__(new_cpp_object)

        # 5. Store your custom properties
        # self._orig_g = val
        FloatingType.__init__(
            self,
            V,
            val,
            # name=name,
            dtype=dtype,
            block_class=kwargs.pop("block_class", None),
            _ad_floating_active=kwargs.pop("_ad_floating_active", False),
            _ad_args=kwargs.pop("_ad_args", None),
            output_block_class=kwargs.pop("output_block_class", None),
            _ad_output_args=kwargs.pop("_ad_output_args", None),
            _ad_outputs=kwargs.pop("_ad_outputs", None),
            annotate=kwargs.pop("annotate", True),
            **kwargs,
        )

    def _ad_create_checkpoint(self):
        checkpoint = create_overloaded_object(self)
        checkpoint.name = self.name + "_checkpoint"
        return checkpoint

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint


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
