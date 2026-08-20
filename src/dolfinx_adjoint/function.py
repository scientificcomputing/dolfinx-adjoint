from __future__ import annotations

import typing

import dolfinx
import numpy
import ufl
from pyadjoint.overloaded_type import (
    create_overloaded_object,
)
from pyadjoint.tape import annotate_tape, get_working_tape, stop_annotating

from .blocks.function_assigner import FunctionAssignBlock
from .types.function import Function as _Function
from .utils import ad_kwargs, assign_linear_combination


def assign(value: typing.Union[numpy.inexact, float, int], function: _Function, **kwargs: typing.Unpack[ad_kwargs]):
    """Assign a `value` to a :py:func:`dolfinx_adjoint.Function`.

    Args:
        value: The value to assign to the function.
        function: The function to assign the value to.
        *args: Additional positional arguments to pass to the assign method.
        **kwargs: Additional keyword arguments to pass to the assign method.
    """
    # do not annotate in case of self assignment
    ad_block_tag = kwargs.pop("ad_block_tag", None)
    annotate = annotate_tape(kwargs) and value != function
    if annotate:
        if not isinstance(value, ufl.core.operator.Operator):
            value = create_overloaded_object(value)
        block = FunctionAssignBlock(value, function, ad_block_tag=ad_block_tag)
        tape = get_working_tape()
        tape.add_block(block)

    with stop_annotating():
        if isinstance(value, (numpy.inexact, float, int)):
            function.x.array[:] = value
        elif isinstance(value, dolfinx.fem.Function):
            if value.function_space == function.function_space:
                function.x.array[:] = value.x.array[:]
            elif value.ufl_element().is_real and value.ufl_shape == ():
                function.x.array[:] = value.x.array[0]
            else:
                raise ValueError("Function spaces of the value and function must match for assignment.")
        elif isinstance(value, ufl.core.expr.Expr):
            # Linear combination of functions, e.g., 2*u + 3*v
            assign_linear_combination(value, function)
        else:
            raise ValueError(f"Unsupported value type for assignment: {type(value)})")
    if annotate:
        block.add_output(function.create_block_variable())
