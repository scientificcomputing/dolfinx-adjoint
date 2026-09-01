import typing
from functools import singledispatchmethod

import dolfinx
import numpy
import numpy.typing as npt
import ufl
from ufl.corealg.dag_traverser import DAGTraverser

from .compat import extract_linear_combination


def function_from_vector(
    V: dolfinx.fem.FunctionSpace,
    vector: typing.Union[
        dolfinx.la.Vector,
        dolfinx.cpp.la.Vector_float32,
        dolfinx.cpp.la.Vector_float64,
        dolfinx.cpp.la.Vector_complex64,
        dolfinx.cpp.la.Vector_complex128,
        dolfinx.cpp.la.Vector_int8,
        dolfinx.cpp.la.Vector_int32,
        dolfinx.cpp.la.Vector_int64,
    ],
) -> dolfinx.fem.Function:
    """Create a new Function from a vector.

    Arguments:
        V: The function space
        vector: The vector data.
    Returns:
        A new {py:class}`dolfinx.fem.Function` instance that has been assigned the
        values from the vector (deep-copy)
    """
    ret = dolfinx.fem.Function(V, dtype=vector.array.dtype)
    ret.x.array[:] = vector.array[:]
    return ret


def gather(vector: dolfinx.la.Vector) -> npt.NDArray[numpy.number]:
    """Gather a vector on all processes.

    Args:
        vector: The vector to gather.
    Returns:
        A numpy array containing the gathered vector.
    """
    local_size = vector.index_map.size_local * vector.block_size
    comm = vector.index_map.comm
    data = comm.allgather(vector.array[:local_size])
    return numpy.hstack(data)


class ad_kwargs(typing.TypedDict):
    ad_block_tag: typing.NotRequired[str]
    """Tag for the block in the adjoint tape."""
    annotate: typing.NotRequired[bool]
    """Whether to annotate the assignment in the adjoint tape."""


def assign_linear_combination(value: ufl.core.expr.Expr, function: dolfinx.fem.Function) -> None:
    """Assign a linear combination of functions to a function.

    Arguments:
        value: A linear combination of functions, e.g. `2*u + 3*v`.
        function: The function to assign the linear combination to.
    """
    pairs = extract_linear_combination(value)
    function.x.array[:] = 0.0
    floatifier = Floatify()
    for weight, func in pairs:
        assert isinstance(func, dolfinx.fem.Function), "All operands in the linear combination must be Functions."
        if not func.function_space == function.function_space:
            raise ValueError("Function spaces of all functions in the linear combination must match for assignment.")
        function.x.array[:] += floatifier.process(weight) * func.x.array[:]
    function.x.scatter_forward()


class Floatify(DAGTraverser):
    """Traverser to convert a UFL expression into a float."""

    def __init__(self, **kwargs):
        """Convert a ufl expression into a float"""
        super().__init__(**kwargs)

    @singledispatchmethod
    def process(self, o: ufl.classes.Expr, **kwargs):
        return float(o)

    @process.register(dolfinx.fem.Function)
    def _(self, o, **kwargs):
        if ufl.checks.is_scalar_constant_expression(o):
            return o.x.array[0]
        raise NotImplementedError(f"Unsupported UFL node type for floatification: {type(o)}")

    @process.register(ufl.classes.Sum)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        # operands is a tuple of the already-floatified children
        return sum(operands)

    @process.register(ufl.classes.Division)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        # Division always has exactly two operands: numerator and denominator
        return operands[0] / operands[1]

    @process.register(ufl.classes.Power)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        # Power has exactly two operands: base and exponent
        return operands[0] ** operands[1]

    @process.register(ufl.classes.Product)
    @DAGTraverser.postorder
    def _(self, o, *operands, **kwargs):
        # Product has exactly two operands: left and right
        return operands[0] * operands[1]


def unroll_dofmap(dofs: npt.NDArray[numpy.int32], bs: int) -> npt.NDArray[numpy.int32]:
    """
    Given a two-dimensional dofmap of size `(num_cells, num_dofs_per_cell)`
    Expand the dofmap by its block size such that the resulting array
    is of size `(num_cells, bs*num_dofs_per_cell)`
    """
    num_cells, num_dofs_per_cell = dofs.shape
    unrolled_dofmap = numpy.repeat(dofs, bs).reshape(num_cells, num_dofs_per_cell * bs) * bs
    unrolled_dofmap += numpy.tile(numpy.arange(bs), num_dofs_per_cell)
    return unrolled_dofmap
