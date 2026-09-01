from __future__ import annotations

import typing

import ufl

from .compat import compute_form_adjoint
from .typing_utils import NestedSequence


def recursive_space_discovery(
    obj: NestedSequence[ufl.Form], indices: tuple[int, ...], spaces: dict[int, ufl.FunctionSpace]
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
    if isinstance(obj, ufl.Form):
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
    else:
        raise TypeError(f"Expected ufl.Form or iterable, got {type(obj)}")


def build_argument_replacement_map(
    obj: NestedSequence[ufl.Form],
    indices: tuple[int, ...],
    test_functions: typing.Sequence[ufl.TestFunction],
    trial_functions: typing.Sequence[ufl.TrialFunction],
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


@typing.overload
def assign_mixed_parts[T: NestedSequence[ufl.Form]](form1: T, /) -> T: ...
@typing.overload
def assign_mixed_parts[T: NestedSequence[ufl.Form], S: NestedSequence[ufl.Form]](
    form1: T, form2: S, /
) -> tuple[T, S]: ...
def assign_mixed_parts(
    *form_structs: NestedSequence[ufl.Form],
) -> NestedSequence[ufl.Form] | tuple[NestedSequence[ufl.Form], ...]:
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
    spaces: dict[int, ufl.functionspace.AbstractFunctionSpace] = {}
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


def collect_coefficients(form: ufl.Form | typing.Sequence | None) -> set[ufl.Coefficient]:
    """Return the set of UFL coefficients appearing anywhere in ``form``.

    ``form`` may be a single form or an arbitrarily nested sequence of forms
    (entries may be ``None``, e.g. a zero block in a blocked system). Plain set
    union rather than ``sum_form``: unlike summing, this never requires the
    sub-forms' arguments to be mutually compatible (e.g. carry matching
    ``part()`` tags), which a blocked ``NonlinearProblem``'s forms are not
    required to be before ``assign_mixed_parts`` runs.
    """
    if form is None:
        return set()
    if isinstance(form, ufl.Form):
        return set(form.coefficients())
    coefficients: set = set()
    for f in form:
        coefficients |= collect_coefficients(f)
    return coefficients


def sum_form(form: NestedSequence[ufl.Form | None]) -> ufl.Form | None:
    """Sum a blocked form into a single form."""
    # Handle top-level None
    if form is None:
        return None

    if isinstance(form, ufl.Form):
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


def compute_adjoint(form: ufl.Form) -> typing.Sequence[typing.Sequence[ufl.Form]] | ufl.Form:
    """Compute the adjoint of a (possibly blocked) bilinear form.

    A module-level function, not a method: it needs no ``Block``/``Problem`` state,
    just ``form`` itself, so both ``_ProblemBlockBase`` (``blocks/solvers.py``) and
    ``_ProblemBase`` (``solvers.py``) can call it directly rather than one reaching
    into a "private" method defined on the other.

    Args:
        form: A bilinear form :math:`a(u, v)`, either a single ``ufl.Form`` or a
            blocked (nested list) system.

    Returns:
        The transposed form :math:`a(v, u)`, decomposed back into blocks (via
        ``ufl.extract_blocks``) -- a no-op decomposition for a scalar form.
    """
    return ufl.extract_blocks(compute_form_adjoint(form))


def recursive_replace(form: ufl.Form | typing.Sequence | None, placeholders: dict) -> ufl.Form | typing.Sequence | None:
    """Recursively apply {py:func}`ufl.replace` to a (possibly nested) form structure.

    A module-level function, not a nested closure: a nested function that
    recurses by calling itself by name captures *itself* as a free variable,
    which makes the function object (and, via ``self`` if the closure also
    needs it) part of a reference cycle -- collected only by the cyclic
    garbage collector, at a moment that differs between MPI ranks, not by
    ordinary refcounting. That is exactly the hazard ``Problem`` owning its
    solvers (rather than each {py:class}`~pyadjoint.Block`) exists to avoid: a
    self-referential ``_replace`` closure inside a
    {py:class}`~dolfinx_adjoint.LinearProblem`/{py:class}`~dolfinx_adjoint.NonlinearProblem`
    ``__init__`` would keep the ``Problem`` itself -- and its PETSc solvers -- alive as
    cyclic garbage. Taking ``placeholders`` as a plain argument instead of
    capturing ``self`` sidesteps this entirely: a module-level function
    referring to itself by name is looked up through the module's namespace,
    not a closure cell, so no cycle is created.

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
    if isinstance(form, ufl.Form):
        return ufl.replace(form, placeholders)
    return [recursive_replace(f, placeholders) for f in form]
