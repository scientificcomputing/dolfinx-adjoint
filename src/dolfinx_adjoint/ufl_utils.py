from __future__ import annotations

import typing

import ufl
from ufl.algorithms.check_arities import ArityChecker, ArityMismatch
from ufl.algorithms.map_integrands import map_integrands
from ufl.corealg.map_dag import map_expr_dag

from .compat import compute_form_adjoint
from .typing_utils import NestedSequence
from .utils import _is_complex_build


def recursive_space_discovery(
    obj: NestedSequence[ufl.BaseForm | None], indices: tuple[int, ...], spaces: dict[int, ufl.FunctionSpace]
) -> None:
    """Recursively discover, for each row/column index, the function space of the
    (as yet unassigned) argument occupying that position.

    `indices` will be `(row,)` for vectors and `(row, col)` for matrices.

    Arguments:
        obj: A UFL form or nested iterable of forms.
        indices: The current row/column indices in the nested structure.
        spaces: A dictionary mapping row/column indices to discovered function spaces.
            This dictionary is updated in-place as the function traverses the structure.
    """
    if isinstance(obj, ufl.BaseForm):
        for arg in obj.arguments():
            if arg.part() is None:
                # The argument number corresponds to the index of the row/column
                # in the nested structure
                num = arg.number()
                if num < len(indices):
                    spaces.setdefault(indices[num], arg.ufl_function_space())
    elif isinstance(obj, typing.Iterable):
        for i, item in enumerate(obj):
            if item is not None:
                recursive_space_discovery(item, indices + (i,), spaces)
    elif obj is None:
        return
    else:
        raise TypeError(f"Expected ufl.Form or iterable, got {type(obj)}")


def build_argument_replacement_map(
    obj: NestedSequence[ufl.BaseForm | None],
    indices: tuple[int, ...],
    test_functions: typing.Sequence[ufl.Argument],
    trial_functions: typing.Sequence[ufl.Argument],
    replace_map: dict[ufl.Argument, ufl.Argument],
) -> None:
    """
    Recursively build a mapping from ufl arguments that does not have a `part`-index
    to their replacements in a {py:class}`ufl.MixedFunctionSpace`.

    Arguments:
        obj: A UFL form or nested iterable of forms.
        indices: The current row/column indices in the nested structure.
        test_functions: A sequence of test functions used for replacement.
        trial_functions: A sequence of trial functions used for replacement.
        replace_map: A dictionary mapping old arguments to new arguments.
    """
    if isinstance(obj, ufl.Form):
        for arg in obj.arguments():
            if arg.part() is None and arg not in replace_map:
                num = arg.number()
                if num < len(indices):
                    replace_map[arg] = (test_functions if num == 0 else trial_functions)[indices[num]]
    elif isinstance(obj, typing.Iterable):
        for i, item in enumerate(obj):
            if item is not None:
                build_argument_replacement_map(item, indices + (i,), test_functions, trial_functions, replace_map)
    elif obj is None:
        return
    else:
        raise TypeError(f"Expected ufl.Form or iterable, got {type(obj)}")


@typing.overload
def assign_mixed_parts[T: NestedSequence[ufl.BaseForm | None]](form1: T, /) -> T: ...
@typing.overload
def assign_mixed_parts[T: NestedSequence[ufl.BaseForm | None], S: NestedSequence[ufl.BaseForm | None]](
    form1: T, form2: S, /
) -> tuple[T, S]: ...
def assign_mixed_parts(
    *form_structs: NestedSequence[ufl.BaseForm | None],
) -> NestedSequence[ufl.BaseForm | None] | tuple[NestedSequence[ufl.BaseForm | None], ...]:
    """
    Recursively assigns mixed-space `part` indices to {py:class}`ufl.Argument`
    (test and trial functions), within nested iterables of forms.

    When solving monolithic block systems in FEniCSx, the UFL arguments must have the
    method {py:meth}`ufl.Argument.part` return the index corresponding to their block position.
    For a block matrix (list of lists), the TestFunction corresponds to the row index, and the
    TrialFunction corresponds to the column index.

    This utility traverses arbitrary nested structures (e.g., a 2D list for the LHS
    matrix `a` and a 1D list for the RHS vector `L` simultaneously), extracts arguments
    that lack a part index, builds a unified replacement map, and applies it.

    Args:
        *form_structs: One or more UFL forms, or nested iterables (lists/tuples) of
            UFL forms. Passing multiple structures (like `a` and `L`) ensures they
            share the same replacement map, preventing mismatched compilation.

    Returns:
        The modified form structures with identical nesting, where all unassigned
        TestFunction and TrialFunction arguments have been mapped. Returns a single
        structure if one was passed, otherwise returns a tuple.

    Note:
        The replacement arguments are drawn from {py:func}`ufl.TestFunctions`
        and {py:func}`ufl.TrialFunctions` of a single
        {py:class}`ufl.MixedFunctionSpace` built from the row/column function spaces
        discovered while walking the structure.
    """
    spaces: dict[int, ufl.FunctionSpace] = {}
    for struct in form_structs:
        recursive_space_discovery(struct, (), spaces)

    # If no replacements are needed, exit early to save computation
    if not spaces:
        return form_structs if len(form_structs) > 1 else form_structs[0]

    num_parts = max(spaces) + 1
    mixed_space = ufl.MixedFunctionSpace(*(spaces[i] for i in range(num_parts)))
    test_functions = ufl.TestFunctions(mixed_space)
    trial_functions = ufl.TrialFunctions(mixed_space)

    replace_map: dict[ufl.Argument, ufl.Argument] = {}
    for struct in form_structs:
        build_argument_replacement_map(struct, (), test_functions, trial_functions, replace_map)

    # Apply the replacements and unpack if necessary
    replaced = tuple(recursive_replace(struct, replace_map) for struct in form_structs)
    return replaced if len(replaced) > 1 else replaced[0]


def get_sorted_arguments(arguments: typing.Iterable[ufl.Argument], number: int) -> typing.Iterable[ufl.Argument]:
    """Extract all arguments of a given number, sorted by part."""
    return sorted(filter(lambda x: x.number() == number, arguments), key=lambda a: a.part())


def sum_form(form: NestedSequence[ufl.Form | None]) -> ufl.Form | None:
    """Sum a blocked form into a single form."""
    # Handle top-level None
    if form is None:
        return None

    if isinstance(form, ufl.BaseForm):
        return form

    elif isinstance(form, typing.Iterable):
        # Recursively sum items, filtering out Nones
        valid_forms: list[ufl.Form] = []
        for fi in form:
            summed_fi = sum_form(fi)
            if summed_fi is not None:
                valid_forms.append(summed_fi)

        # Handle empty case safely
        if not valid_forms:
            return None

        # Safely sum without defaulting to integer 0, removing the need for type: ignore
        return sum(valid_forms[1:], start=valid_forms[0])

    else:
        raise TypeError(f"Cannot sum form of type {type(form)}")


def compute_adjoint(form: ufl.Form, blocked: bool = True) -> typing.Sequence[typing.Sequence[ufl.Form]] | ufl.Form:
    """Compute the adjoint of a (possibly blocked) bilinear form.

    Args:
        form: A bilinear form :math:`a(u, v)`. Blocked forms should be summed with
            {py:func}`sum_form` before passing to this function.
        blocked: Whether ``form``'s arguments come from a genuine blocked/mixed
            problem (multiple ``Argument``s with distinct ``part()`` tags). When
            ``False``, ``ufl.extract_blocks`` is skipped entirely: a plain scalar
            or vector-*shaped* (non-mixed) argument still reports multiple
            "parts" to UFL, so ``extract_blocks`` would otherwise decompose the
            single bilinear form into spurious blocks that each still reference
            the original, full-space ``Argument`` -- producing a system sized
            for several redundant copies of the space once assembled.

    Returns:
        The transposed form :math:`a(v, u)`: a single ``ufl.Form`` when
        ``blocked=False``, else decomposed back into blocks via
        ``ufl.extract_blocks`` (a no-op decomposition for a scalar form).
    """
    adjoint_form = compute_form_adjoint(form)
    if not blocked:
        return adjoint_form
    return ufl.extract_blocks(adjoint_form)


def recursive_replace(
    form: NestedSequence[ufl.BaseForm | None], placeholders: dict
) -> NestedSequence[ufl.BaseForm | None]:
    """Recursively apply {py:func}`ufl.replace` to a (possibly nested) form structure.

    Args:
        form: A single form, ``None``, or an arbitrarily nested sequence of
            forms/``None`` (e.g. a blocked system).
        placeholders: Map passed straight through to {py:func}`ufl.replace` at
            each form encountered.

    Returns:
        A structure with the same nesting as ``form``, each form replaced via
        {py:func}`ufl.replace`; ``None`` in, ``None`` out.
    """
    if form is None:
        return None
    if isinstance(form, ufl.BaseForm):
        return ufl.replace(form, placeholders)
    return [recursive_replace(f, placeholders) for f in form]


def _conjugate_hermitian_pairing(form: ufl.Form) -> ufl.Form:
    r"""Conjugate a form built by contracting a residual against an adjoint solution.

    ``ufl.action(F, lmbda)`` substitutes an adjoint solution into a residual's test-function
    slot, and a sesquilinear residual is conjugate-linear there, so the result is the pairing
    :math:`\lambda^H F` rather than the :math:`(\cdot)^H \lambda` this package's seeds are
    defined by. Differentiating that pairing with respect to a control, along a test function
    of the control's space, therefore produces the *conjugate* of the seed wanted -- and a
    form that is linear, rather than conjugate-linear, in its own test function, which is
    exactly what UFL's complex-mode arity rules reject.

    Conjugating the whole integrand fixes both at once, and fixes them for the same reason:
    the seed and its arity are two views of the same conjugation. It is exact rather than a
    relabelling -- unlike {py:func}`_conjugate_for_complex_mode`, which moves the conjugation
    UFL records without changing a value -- because a real directional derivative commutes
    with conjugation, so conjugating the contracted form conjugates every derivative of it
    too, however many were taken and in whatever directions.

    The first-order control path does not need this: it differentiates the residual *before*
    contracting, so {py:func}`ufl.adjoint` conjugates on its behalf. Only the second-order
    templates, which differentiate an already-contracted rank-0 form, arrive here.

    Args:
        form: The contracted form, or a derivative of one.

    Returns:
        The conjugated form under a complex-scalar build; ``form`` unchanged under a real one,
        where conjugation is the identity and UFL imposes no arity rule to repair.
    """
    if not _is_complex_build():
        return form
    return map_integrands(ufl.conj, form)


def _argument_conjugation(expression: ufl.core.expr.Expr, argument: ufl.Argument) -> bool | None:
    """Report how ``argument`` enters ``expression``: conjugated, bare, or not at all.

    Raises:
        ufl.algorithms.check_arities.ArityMismatch: If a sum somewhere inside the
            expression adds a conjugated occurrence to a bare one, so that no single answer
            describes the expression.
    """
    arities = map_expr_dag(ArityChecker((argument,)), expression, compress=False)
    conjugations = {conjugated for arg, conjugated in arities if arg.number() == argument.number()}
    if not conjugations:
        return None
    (conjugation,) = conjugations
    return conjugation


def _conjugate_argument(expression: ufl.core.expr.Expr, argument: ufl.Argument) -> ufl.core.expr.Expr:
    """Flip whether ``argument`` counts as conjugated inside ``expression``.

    The value is unchanged: the argument ranges over a real-valued (Lagrange) basis, for
    which ``conj(phi) == phi`` pointwise. Only the conjugation UFL *records* moves, and that
    is what the complex-mode arity rules are about.
    """
    return ufl.replace(expression, {argument: ufl.conj(argument)})


def _agree_on_conjugation(expression: ufl.core.expr.Expr, argument: ufl.Argument) -> ufl.core.expr.Expr:
    """Rewrite ``expression`` so that no sum inside it mixes conjugation states.

    Descends only where the arity checker reports a mixture -- an expression it already
    accepts is returned untouched -- and repairs each offending sum by conjugating the
    argument in the terms that lack it.
    """
    try:
        _argument_conjugation(expression, argument)
    except ArityMismatch:
        pass
    else:
        return expression

    operands = [_agree_on_conjugation(operand, argument) for operand in expression.ufl_operands]
    if isinstance(expression, ufl.classes.Sum):
        conjugations = [_argument_conjugation(operand, argument) for operand in operands]
        if True in conjugations and False in conjugations:
            operands = [
                _conjugate_argument(operand, argument) if conjugation is False else operand
                for operand, conjugation in zip(operands, conjugations)
            ]
    return expression._ufl_expr_reconstruct_(*operands)


def _conjugate_for_complex_mode(integrand: ufl.core.expr.Expr, argument: ufl.Argument) -> ufl.core.expr.Expr:
    """Rewrite an integrand so that ``argument`` is conjugated, as complex mode requires."""
    integrand = _agree_on_conjugation(integrand, argument)
    if _argument_conjugation(integrand, argument) is False:
        integrand = _conjugate_argument(integrand, argument)
    return integrand


def _derivative_along(form: ufl.Form, coefficient: ufl.core.expr.Expr, direction, argument) -> ufl.Form:
    """Differentiate ``form`` along ``direction``, repaired for complex-mode arity rules."""
    dform = ufl.algorithms.expand_derivatives(ufl.derivative(form, coefficient, direction))
    return map_integrands(lambda integrand: _conjugate_for_complex_mode(integrand, argument), dform)


def _wirtinger_derivative_forms(
    form: ufl.Form, coefficient: ufl.core.expr.Expr, argument: ufl.Argument
) -> tuple[ufl.Form, ufl.Form | None]:
    r"""Forms whose assembly gives the adjoint seed of ``form`` with respect to ``coefficient``.

    Under a real-scalar build this is just ``ufl.derivative(form, coefficient, argument)``,
    returned alone.

    Under a complex-scalar build one derivative is not enough. ``form`` is a real-differentiable
    function of a complex coefficient, so its differential splits into a holomorphic and an
    anti-holomorphic part,

    .. math::

        dF = \frac{\partial F}{\partial c}\,dc + \frac{\partial F}{\partial \bar c}\,\overline{dc},

    and the adjoint seed this codebase carries is :math:`\overline{\partial F/\partial c} +
    \partial F/\partial \bar c` -- conjugated on the holomorphic part because every pairing
    downstream of it is Hermitian ({py:func}`ufl.adjoint`), and carrying the anti-holomorphic
    part is what lets a Functional such as :math:`|u - u_d|^2` be differentiated at all.

    {py:func}`ufl.derivative` returns neither part on its own: differentiating along
    ``argument`` gives their sum, :math:`v_1 = \partial F/\partial c + \partial F/\partial
    \bar c` (the derivative along a *real* perturbation, since the argument ranges over a
    real-valued basis). Differentiating along ``1j * argument`` gives
    :math:`v_2 = i(\partial F/\partial c - \partial F/\partial \bar c)`, and the two
    together separate the parts. The seed is then, vector by vector,

    .. math::

        \mathrm{Re}(v_1) + i\,\mathrm{Re}(v_2).

    Forming that combination is not left to callers: this function is private to
    {py:func}`dolfinx_adjoint.blocks.assembly._assemble_wirtinger_seed`, which assembles both
    forms and combines them, and is the only thing that should call it. The split into two
    functions is only the split between symbolic work and assembly.

    Each form is also repaired for UFL's complex-mode arity rules, which require argument
    number 0 to appear conjugated in every term. {py:func}`ufl.derivative` leaves the direction
    it is handed exactly as given, so the raw derivative generally violates that; conjugating
    the direction up front instead only moves the violation to the form shapes where the
    differentiated occurrence of ``coefficient`` sits in the second slot of an
    {py:func}`ufl.inner`, which conjugates it a second time. The conjugation therefore has to be
    decided per term, after differentiating. Doing so is free of numerical consequence: the
    argument ranges over a real-valued (Lagrange) basis, so only the conjugation UFL records
    changes, never a value.

    Args:
        form: The rank-0 form to differentiate.
        coefficient: The coefficient to differentiate with respect to.
        argument: The direction to differentiate in, an argument over a real-valued basis.
    Returns:
        The derivative along ``argument``, and -- under a complex-scalar build -- the
        derivative along ``1j * argument``, which is ``None`` otherwise.
    """
    if not _is_complex_build():
        return ufl.derivative(form, coefficient, argument), None
    return (
        _derivative_along(form, coefficient, argument, argument),
        _derivative_along(form, coefficient, 1j * argument, argument),
    )
