from collections.abc import Sequence

import dolfinx
from ufl import as_tensor, indices
from ufl.algebra import Conj
from ufl.algorithms.formsplitter import extract_blocks
from ufl.algorithms.map_integrands import map_integrands
from ufl.algorithms.replace import replace
from ufl.argument import Argument
from ufl.classes import Jacobian, JacobianDeterminant, JacobianInverse

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


def bcs_by_block(
    spaces: Sequence[dolfinx.fem.FunctionSpace | dolfinx.cpp.fem.FunctionSpace_float32 | None],
    bcs: Sequence[dolfinx.fem.DirichletBC],
) -> list[list[dolfinx.fem.DirichletBC]]:
    """Arrange boundary conditions by the space they constrain, on every supported dolfinx.

    :py:func:`dolfinx.fem.bcs.bcs_by_block` cannot be called directly because 0.11 and 0.12
    disagree about which side of the containment check is a Python wrapper and which is a
    cpp object, in opposite directions:

    * 0.11 does ``V.contains(bc.function_space)``. ``DirichletBC.function_space`` there
      unwraps to a cpp space, so ``V`` must be a cpp space too -- as it is when it comes
      from ``extract_function_spaces``, whose spaces are cpp on both versions. Passing a
      *Python* ``V`` (e.g. ``u.function_space``) picks up the Python ``contains``, which
      dereferences ``bc.function_space._cpp_object`` and raises ``AttributeError``.
    * 0.12 (FEniCS/dolfinx#4342) made ``DirichletBC`` hold the Python ``V``/``g`` it was
      built with, and ``bcs_by_block`` now normalises ``V`` itself while requiring
      ``bc.function_space`` to be the Python wrapper.

    Normalising both sides here and calling the cpp ``contains`` directly is stable across
    both, and accepts either flavour of space from the caller.
    """

    def _cpp(space):
        return getattr(space, "_cpp_object", space)

    return [[bc for bc in bcs if _cpp(V).contains(_cpp(bc.function_space))] if V is not None else [] for V in spaces]


# --- Pullbacks -----------------------------------------------------------------------------
#
# `AbstractPullback.apply_inverse` -- the physical-to-reference direction -- arrived in
# FEniCS/ufl#511. It is needed to differentiate an interpolation into a space whose pullback is
# not the identity, because the dofs are built from the pulled-back expression rather than from
# the expression itself. Only the inverse is missing on older UFL, not `apply`, and each one is
# a short closed-form expression, so they are written out here rather than requiring the newer
# UFL. `_INVERSE_PULLBACKS` is keyed by class name to avoid importing classes that a given UFL
# may not define.


def _inverse_identity(expr, domain):
    return expr


def _inverse_contravariant_piola(expr, domain):
    """``v = J vhat / detJ`` inverts to ``vhat = detJ K v``."""
    detJ = JacobianDeterminant(Jacobian(domain))
    K = JacobianInverse(domain)
    *k, i, j = indices(len(expr.ufl_shape) + 1)
    return as_tensor(detJ * K[i, j] * expr[(*k, j)], (*k, i))


def _inverse_covariant_piola(expr, domain):
    """``v = K^T vhat`` inverts to ``vhat = J^T v``."""
    J = Jacobian(domain)
    *k, i, j = indices(len(expr.ufl_shape) + 1)
    return as_tensor(J[j, i] * expr[(*k, j)], (*k, i))


def _inverse_l2_piola(expr, domain):
    """``v = vhat / detJ`` inverts to ``vhat = detJ v``."""
    return expr * JacobianDeterminant(domain)


def _inverse_double_contravariant_piola(expr, domain):
    """``v = J vhat J^T / detJ^2`` inverts to ``vhat = detJ^2 K v K^T``."""
    detJ = JacobianDeterminant(Jacobian(domain))
    K = JacobianInverse(domain)
    *k, i, j, m, n = indices(len(expr.ufl_shape) + 2)
    return as_tensor(detJ**2 * K[i, m] * expr[(*k, m, n)] * K[j, n], (*k, i, j))


def _inverse_double_covariant_piola(expr, domain):
    """``v = K^T vhat K`` inverts to ``vhat = J^T v J``."""
    J = Jacobian(domain)
    *k, i, j, m, n = indices(len(expr.ufl_shape) + 2)
    return as_tensor(J[m, i] * expr[(*k, m, n)] * J[n, j], (*k, i, j))


def _inverse_covariant_contravariant_piola(expr, domain):
    """``v = K^T vhat J^T / detJ`` inverts to ``vhat = detJ J^T v K^T``."""
    J = Jacobian(domain)
    detJ = JacobianDeterminant(J)
    K = JacobianInverse(domain)
    *k, i, j, m, n = indices(len(expr.ufl_shape) + 2)
    return as_tensor(detJ * J[m, i] * expr[(*k, m, n)] * K[j, n], (*k, i, j))


_INVERSE_PULLBACKS = {
    "IdentityPullback": _inverse_identity,
    "ContravariantPiola": _inverse_contravariant_piola,
    "CovariantPiola": _inverse_covariant_piola,
    "L2Piola": _inverse_l2_piola,
    "DoubleContravariantPiola": _inverse_double_contravariant_piola,
    "DoubleCovariantPiola": _inverse_double_covariant_piola,
    "CovariantContravariantPiola": _inverse_covariant_contravariant_piola,
}


def apply_pullback_inverse(pullback, expr, domain):
    """Map ``expr`` from the physical cell to the reference cell.

    Args:
        pullback: The element's pullback.
        expr: A physical-cell expression.
        domain: The domain whose Jacobian relates the two cells.

    Returns:
        ``expr`` pulled back to the reference cell.

    Raises:
        NotImplementedError: If this UFL has no ``apply_inverse`` for ``pullback`` and no
            closed form is written out here -- the composite pullbacks (mixed, symmetric) and
            the ones that are not a fixed expression at all (custom, physical, undefined).
    """
    if hasattr(pullback, "apply_inverse"):
        return pullback.apply_inverse(expr, domain)
    name = type(pullback).__name__
    try:
        return _INVERSE_PULLBACKS[name](expr, domain)
    except KeyError:
        raise NotImplementedError(
            f"This UFL does not provide {name}.apply_inverse (added in FEniCS/ufl#511) and "
            "dolfinx-adjoint has no closed form for it, so an interpolation into a space with "
            "this pullback cannot be shape-differentiated. Upgrade UFL."
        ) from None
