import typing

import dolfinx
import numpy
import numpy.typing as npt
import ufl


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

    :arg V: The function space
    :arg vector: The vector data.
    """
    ret = dolfinx.fem.Function(V, dtype=vector.array.dtype)
    ret.x.array[:] = vector.array[:]
    return ret


def gather(vector: dolfinx.la.Vector) -> npt.NDArray[numpy.number]:
    """Gather a vector on all processes."""
    local_size = vector.index_map.size_local * vector.block_size
    comm = vector.index_map.comm
    data = comm.allgather(vector.array[:local_size])
    return numpy.hstack(data)


class ad_kwargs(typing.TypedDict):
    ad_block_tag: typing.NotRequired[str]
    """Tag for the block in the adjoint tape."""
    annotate: typing.NotRequired[bool]
    """Whether to annotate the assignment in the adjoint tape."""


def extract_scalar_value(scalar_expr: ufl.core.expr.Expr) -> float:
    """Extract float from a scalar UFL expression."""
    if isinstance(scalar_expr, (ufl.classes.IntValue, ufl.classes.FloatValue)):
        return float(scalar_expr)
    elif isinstance(scalar_expr, dolfinx.fem.Function):
        # Check if it's a RealElement (constant stored as Function)
        if scalar_expr.function_space.ufl_element().is_real and scalar_expr.ufl_shape == ():
            return float(scalar_expr.x.array[0])
        else:
            raise ValueError(f"Cannot extract scalar from spatial Function: {scalar_expr}")
    elif isinstance(scalar_expr, dolfinx.fem.Constant) and scalar_expr.ufl_shape == ():
        val = scalar_expr.value
        return float(val) if hasattr(val, "__float__") else float(val.item())
    elif isinstance(scalar_expr, ufl.classes.ScalarValue):
        return float(scalar_expr._value)
    elif isinstance(scalar_expr, ufl.classes.Product):
        result = 1.0
        for op in scalar_expr.ufl_operands:
            result *= extract_scalar_value(op)
        return result
    elif isinstance(scalar_expr, ufl.classes.Division):
        num, den = scalar_expr.ufl_operands
        return extract_scalar_value(num) / extract_scalar_value(den)
    else:
        raise ValueError(f"Cannot extract scalar from {type(scalar_expr)}: {scalar_expr}")


def extract_function(expr) -> tuple[bool, dolfinx.fem.Function | None]:
    """Recursively extract a Function from nested UFL expressions."""
    if isinstance(expr, dolfinx.fem.Function):
        is_real = expr.function_space.ufl_element().is_real
        if is_real:
            return (False, None)
        return (False, expr)
    elif isinstance(expr, (ufl.classes.Indexed, ufl.classes.ComponentTensor)):
        return extract_function(expr.ufl_operands[0])
    elif hasattr(expr, "ufl_operands"):
        found_func = None
        for op in expr.ufl_operands:
            is_real, func = extract_function(op)
            if func is not None:
                if found_func is not None:
                    raise ValueError(f"Non-linear expression detected: multiple spatial functions in {expr}")
                found_func = func
        return (False, found_func)
    return (False, None)


def extract_term(term: ufl.core.expr.Expr) -> tuple[float, dolfinx.fem.Function] | None:
    """Extract (weight, function) from a single term."""
    if isinstance(term, dolfinx.fem.Function):
        is_real = term.function_space.ufl_element().is_real
        if is_real:
            return None
        return (1.0, term)
    elif isinstance(term, ufl.classes.ComponentTensor):
        return extract_term(term.ufl_operands[0])
    elif isinstance(term, ufl.classes.Indexed):
        is_real, func = extract_function(term)
        if func is None:
            return None
        return (1.0, func)
    elif isinstance(term, ufl.classes.Product):
        weight = 1.0
        func = None
        for op in term.ufl_operands:
            is_real, extracted_func = extract_function(op)
            if extracted_func is not None:
                if func is not None:
                    raise ValueError(f"Non-linear term detected: multiple spatial functions in {term}")
                func = extracted_func
            else:
                weight *= extract_scalar_value(op)
        return (weight, func) if func is not None else None
    elif isinstance(term, ufl.classes.Division):
        num, den = term.ufl_operands
        denom_val = extract_scalar_value(den)
        if isinstance(num, dolfinx.fem.Function):
            is_real = num.function_space.ufl_element().is_real
            if is_real:
                return None
            return (1.0 / denom_val, num)
        elif isinstance(num, ufl.classes.Product):
            result = extract_term(num)
            return (result[0] / denom_val, result[1]) if result else None
    return None


def extract_linear_combination(expr: ufl.core.expr.Expr) -> list[tuple[float, dolfinx.fem.Function]]:
    """Extract (weight, function) pairs from a UFL linear combination.

    Analyzes expressions like: 0.5*u + 0.3*v + 0.2*w
    Returns: [(0.5, u), (0.3, v), (0.2, w)]

    :param expr: UFL expression (Sum, Product, or single Function)
    :returns: List of (weight, function) tuples
    """

    # Parse the expression, flattening nested Sums recursively
    summands: list[ufl.core.expr.Expr] | tuple[ufl.core.terminal.FormArgument, ...]
    if isinstance(expr, ufl.classes.Sum):
        summands = expr.ufl_operands
    else:
        summands = [expr]
    terms = []
    for summand in summands:
        if isinstance(summand, ufl.classes.Sum):
            # Recursively flatten nested Sum structures
            terms.extend(extract_linear_combination(summand))
        else:
            result = extract_term(summand)
            if result is not None:
                terms.append(result)
    return terms


def assign_linear_combination(value: ufl.core.expr.Expr, function: dolfinx.fem.Function):
    pairs = extract_linear_combination(value)
    function.x.array[:] = 0.0
    for weight, func in pairs:
        if not func.function_space == function.function_space:
            raise ValueError("Function spaces of all functions in the linear combination must match for assignment.")
        function.x.array[:] += weight * func.x.array[:]
    function.x.scatter_forward()
