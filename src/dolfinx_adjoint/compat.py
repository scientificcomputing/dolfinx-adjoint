import dolfinx
from ufl.algebra import Conj
from ufl.algorithms.formsplitter import extract_blocks
from ufl.algorithms.map_integrands import map_integrands
from ufl.algorithms.replace import replace
from ufl.argument import Argument

try:
    from ufl.algorithms.extract_linear_combination import extract_linear_combination

except ImportError:
    # This is a workaround until dolfinx-adjoint only supports the version of UFL that has
    # this feature

    from functools import singledispatchmethod

    import ufl
    from ufl.corealg.dag_traverser import DAGTraverser

    class LinearCombinationExtractor(DAGTraverser):
        """Bottom-up DAG traverser for extracting linear combinations.

        To process an arbitrary mathematical expression, this traverser categorizes
        every node in the DAG into one of two states, returning different types for each:

        1. Scalar Weights (Returns: {py:class}`ufl.core.expr.Expr`)
        If a node and all its children represent a global scalar value (e.g.,
        {py:class}`ufl.FloatValue`, {py:class}`ufl.Constant`), the traverser
        propagates the actual UFL expression upwards. It does not evaluate them
        to Python floats, preserving the full UFL AST of the constants.

        2. Spatial Fields (Returns: `list[tuple[ufl.core.expr.Expr, ufl.Coefficient]]`)
        If a node contains spatial functions (standard Coefficients), it must
        maintain the strict algebraic structure of a linear combination. Therefore,
        it returns a list of `(weight, function)` tuples, where `weight` is the
        accumulated UFL expression and `function` is the base spatial field.

        By strictly distinguishing between the two (checking `isinstance(..., list)`),
        the traverser can safely apply algebraic rules (e.g., multiplying a list by a
        scalar weight expression distributes the weight) and instantly catch illegal
        non-linear operations (e.g., attempting to multiply two lists together).
        """

        def __init__(self, **kwargs):
            """Initialize LinearCombinationExtractor with memoization and no compression.

            Compression is disabled to avoid hashing unhashable return types (like lists)
            while preserving the `_visited_cache` memoization.
            """
            kwargs["compress"] = False
            super().__init__(**kwargs)

        @singledispatchmethod
        def process(self, o: ufl.classes.Expr, **kwargs):
            """Fallback for any unsupported node types."""
            raise ValueError(f"Unsupported UFL node type for linear combinations: {type(o)}")

        @process.register(ufl.coefficient.BaseCoefficient)
        def _(self, o, **kwargs):
            raise NotImplementedError(f"Unsupported UFL node type for linear combinations: {type(o)}")

        # ---------------------------------------------------------
        # 1. Terminals (Leaves) - No children to evaluate
        # ---------------------------------------------------------
        @process.register(ufl.classes.IntValue)
        @process.register(ufl.classes.FloatValue)
        @process.register(ufl.classes.ScalarValue)
        def _(self, o, **kwargs):
            # Return the UFL expression itself
            return o

        @process.register(ufl.classes.Zero)
        def _(self, o, **kwargs):
            return o if o.ufl_shape == () else []

        @process.register(ufl.Constant)
        def _(self, o, **kwargs):
            if o.ufl_shape == ():
                return o
            raise ValueError(f"Only scalar constants are supported, got shape {o.ufl_shape}")

        @process.register(ufl.Cofunction)
        @process.register(ufl.Matrix)
        def _(self, o, **kwargs):
            return [(ufl.as_ufl(1.0), o)]

        @process.register(ufl.classes.Coefficient)
        def _(self, o, **kwargs):
            # Check for real-valued elements
            if ufl.checks.is_scalar_constant_expression(o):
                return o
            return [(ufl.as_ufl(1.0), o)]

        # ---------------------------------------------------------
        # 2. Operators - Use @postorder to evaluate operands first
        # ---------------------------------------------------------
        @process.register(ufl.classes.Sum)
        @DAGTraverser.postorder
        def _(self, o, *operands, **kwargs):
            # If no operands are lists, this is a pure scalar addition.
            # We construct a new UFL expression by safely summing them.
            if all(not isinstance(op, list) for op in operands):
                res = operands[0]
                for op in operands[1:]:
                    res = res + op
                return res

            # Otherwise, accumulate the spatial functions
            res = []
            for op_res in operands:
                if isinstance(op_res, list):
                    res.extend(op_res)
                else:
                    raise ValueError("Cannot directly add a raw scalar expression to a spatial function.")
            return res

        @process.register(ufl.Action)
        def _(self, o, **kwargs):
            # An Action node represents a matrix-vector product (e.g., A * u).
            # This cannot be reduced to a simple algebraic linear combination of arrays.
            raise ValueError("Non-linear expression detected: product of two spatial functions.")

        @process.register(ufl.classes.FormSum)
        @process.register(ufl.form.FormSum)
        def _(self, o, **kwargs):
            res = []
            components = o.components()
            weights = o.weights()
            for weight, comp in zip(weights, components):
                # Evaluate the base component (e.g., Matrix or Cofunction)
                comp_res = self(comp, **kwargs)

                # Evaluate the weight (in case it contains sub-expressions)
                w_res = self(weight, **kwargs) if isinstance(weight, ufl.classes.Expr) else weight

                if isinstance(comp_res, list):
                    # Distribute this FormSum weight into the component's linear combination
                    res.extend([(w_res * w, f) for w, f in comp_res])
                else:
                    raise ValueError("Cannot directly add a raw scalar expression to a spatial function.")

            return res

        @process.register(ufl.classes.Product)
        @DAGTraverser.postorder
        def _(self, o, *operands, **kwargs):
            op1_res, op2_res = operands
            # Each of the operands are either a scalar UFL expression (float, Constant, etc.)
            # or a list of (weight, function) tuples.
            # The following cases are possible:
            # 1. Both operands are scalars: return the product of the two UFL expressions.
            # 2. One operand is a scalar, the other is a list: distribute the scalar across the list.
            # 3. Both operands are lists: this is a non-linear operation and should raise an error.
            is_list1 = isinstance(op1_res, list)
            is_list2 = isinstance(op2_res, list)
            if not is_list1 and not is_list2:
                return op1_res * op2_res  # UFL operator overloading takes over
            elif not is_list1 and is_list2:
                return [(op1_res * w, f) for w, f in op2_res]
            elif not is_list2 and is_list1:
                return [(op2_res * w, f) for w, f in op1_res]
            else:
                raise ValueError("Non-linear expression detected: product of two spatial functions.")

        @process.register(ufl.classes.Division)
        @DAGTraverser.postorder
        def _(self, o, *operands, **kwargs):
            num_res, den_res = operands
            if isinstance(den_res, list):
                raise ValueError("Non-linear expression detected: division by a spatial function.")

            if not isinstance(num_res, list):
                return num_res / den_res
            return [(w / den_res, f) for w, f in num_res]

        @process.register(ufl.classes.Power)
        @DAGTraverser.postorder
        def _(self, o, *operands, **kwargs):
            base_res, exp_res = operands
            if isinstance(base_res, list) or isinstance(exp_res, list):
                raise ValueError("Non-linear expression detected: power involving a spatial function.")
            return base_res**exp_res

        # ---------------------------------------------------------
        # 3. Forbidden Operations
        # ---------------------------------------------------------
        @process.register(ufl.classes.Indexed)
        @process.register(ufl.classes.ComponentTensor)
        def _(self, o, **kwargs):
            raise NotImplementedError("Direct array assignment of indexed vector components is not supported.")

    def extract_linear_combination(
        expr: ufl.core.expr.Expr | ufl.form.BaseForm,
    ) -> list[tuple[ufl.core.expr.Expr, ufl.coefficient.BaseCoefficient]]:
        """Wrapper to initialize traverser and extract linear combinations.

        Returns:
            A list of tuples where the first element is the UFL expression of the
            weight, and the second element is the base UFL Coefficient (spatial function).
        """
        extractor = LinearCombinationExtractor()
        final_result = extractor(expr)  # type: ignore

        if not isinstance(final_result, list):
            raise ValueError("Expression evaluated to a pure scalar, no spatial functions found.")

        return final_result


def get_interpolation_points(V: dolfinx.fem.FunctionSpace):
    """Get the interpolation points for a given function space V."""
    try:
        return V.element.interpolation_points()  # type: ignore[operator]
    except TypeError:
        return V.element.interpolation_points


# Workaround until https://github.com/FEniCS/ufl/pull/508 is in all stable releases we support
def compute_form_adjoint(
    form,
    reordered_arguments: tuple[Argument, Argument] | tuple[tuple[Argument, Argument], ...] | None = None,
):
    """Compute the adjoint of a bilinear form.

    This works simply by swapping the number of the two arguments,
    but keeping their elements and places in the integrand expressions.

    Args:
        form: A UFL bilinear form.
        reordered_arguments: Optional explicit arguments to use for the adjoint form.
            - For standard finite element spaces: A single tuple `(new_u, new_v)`
              representing the replacement trial and test functions.
            - For mixed function spaces: A sequence of tuples, with one `(new_u, new_v)`
              pair for each *subspace*. For example, `((new_u0, new_v0), (new_u1, new_v1))`.
              The test function mappings are extracted using the block row index `i`,
              and the trial function mappings using the block column index `j`.

    Returns:
        The adjoint of the bilinear form.
    """
    if form.empty():
        return form

    arguments = form.arguments()

    # Check if mixed space
    is_mixed = any(arg.part() is not None for arg in arguments)

    def validate_mapping(old_v: Argument, old_u: Argument, new_v: Argument, new_u: Argument, check_parts=False):
        """Validate the mapping of old arguments to new arguments."""
        if new_u.number() >= new_v.number():
            raise ValueError("Ordering of new arguments is the same as the old arguments!")
        if new_u.ufl_function_space() != old_u.ufl_function_space():
            raise ValueError("Element mismatch between new and old arguments (trial functions).")
        if new_v.ufl_function_space() != old_v.ufl_function_space():
            raise ValueError("Element mismatch between new and old arguments (test functions).")

        if check_parts and (new_u.part() != old_v.part() or new_v.part() != old_u.part()):
            raise ValueError("Ordering of new arguments is the same as the old arguments!")

    if not is_mixed:
        if len(arguments) != 2:
            raise ValueError("Expecting bilinear form.")

        v, u = arguments
        if v.number() >= u.number():
            raise ValueError("Mistaken assumption in code!")
        if reordered_arguments is None:
            assert u.part() is None and v.part() is None
            new_u = Argument(u.ufl_function_space(), number=v.number())
            new_v = Argument(v.ufl_function_space(), number=u.number())
        else:
            assert isinstance(reordered_arguments, tuple) and len(reordered_arguments) == 2
            u_arg, v_arg = reordered_arguments[0], reordered_arguments[1]
            assert isinstance(u_arg, Argument) and isinstance(v_arg, Argument)
            new_u, new_v = u_arg, v_arg

        validate_mapping(v, u, new_v, new_u, check_parts=True)

        return map_integrands(Conj, replace(form, {v: new_v, u: new_u}))
    else:
        form_blocked = extract_blocks(form, arity=2)
        # Apply mapping block-by-block and sum
        form_adj = 0
        assert isinstance(form_blocked, tuple)
        for i, row in enumerate(form_blocked):
            assert isinstance(row, tuple)
            for j, block in enumerate(row):
                if block is not None:
                    v, u = block.arguments()
                    if reordered_arguments is not None:
                        new_v = reordered_arguments[i][1]
                        new_u = reordered_arguments[j][0]
                    else:
                        new_v = Argument(v.ufl_function_space(), number=u.number(), part=v.part())
                        new_u = Argument(u.ufl_function_space(), number=v.number(), part=u.part())
                    local_map = {v: new_v, u: new_u}
                    validate_mapping(v, u, new_v, new_u)
                    form_adj += map_integrands(Conj, replace(block, local_map))

        return form_adj
