from __future__ import annotations

import typing

import ufl
from ufl.algorithms.analysis import extract_type

from .compat import compute_form_adjoint
from .typing_utils import NestedSequence

# Geometric quantities whose shape derivative UFL gets right. Everything else in
# `ufl.classes.GeometricQuantity` is differentiated to *zero*: `CoordinateDerivativeRuleset`
# registers a rule for the whole base class that returns an independent terminal
# ("Explicitly defining dg/dw == 0"). These four survive because `compute_form_data` runs
# `apply_geometry_lowering` first, rewriting them in terms of the Jacobian and so of the
# coordinates, before the coordinate derivative is applied; the ones that stay terminals do
# not. Measured: CellDiameter and MinCellEdgeLength come out at 2/3 of the true derivative
# (only the measure's contribution survives) and Circumradius at exactly 0.
_SHAPE_DIFFERENTIABLE_GEOMETRY = frozenset({"SpatialCoordinate", "FacetNormal", "CellVolume", "FacetArea"})


def geometry_without_shape_derivative(form: NestedSequence[ufl.BaseForm | None]) -> set[str]:
    """Names of geometric quantities in ``form`` that UFL differentiates to zero.

    Args:
        form: A single form, ``None``, or an arbitrarily nested sequence of forms/``None``
            (a blocked system's right-hand side is a list, and ``ufl.extract_blocks`` returns
            tuples).

    Returns:
        The distinct type names of the offending quantities, empty if there are none.
    """
    if form is None:
        return set()
    if isinstance(form, ufl.BaseForm):
        return {
            type(quantity).__name__
            for quantity in extract_type(form, ufl.classes.GeometricQuantity)
            if type(quantity).__name__ not in _SHAPE_DIFFERENTIABLE_GEOMETRY
        }
    offenders: set[str] = set()
    for part in form:
        offenders |= geometry_without_shape_derivative(part)
    return offenders


def reject_geometry_without_shape_derivative(form: NestedSequence[ufl.BaseForm | None]) -> None:
    """Refuse a form whose shape derivative UFL would silently get wrong.

    Called only once a mesh is known to have been moved, so a form using these quantities on a
    mesh nobody differentiates through is left alone.

    Args:
        form: The form, or nested structure of forms, about to gain a mesh dependency.

    Raises:
        NotImplementedError: If ``form`` contains a geometric quantity that UFL differentiates
            to zero with respect to the coordinates.
    """
    offenders = geometry_without_shape_derivative(form)
    if offenders:
        raise NotImplementedError(
            f"Cannot take a shape derivative of a form containing {', '.join(sorted(offenders))}: "
            "UFL differentiates every geometric quantity except "
            f"{', '.join(sorted(_SHAPE_DIFFERENTIABLE_GEOMETRY))} to zero with respect to the "
            "coordinates, so the contribution would be dropped and the gradient would be "
            "silently wrong rather than merely incomplete. Express the quantity through "
            "SpatialCoordinate instead, or do not move this mesh."
        )


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
