import contextlib
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
        # extract_linear_combination is typed against UFL, which knows nothing of degrees of
        # freedom; assigning a linear combination needs the DOLFINx Function that carries them.
        if not isinstance(func, dolfinx.fem.Function):
            raise TypeError(f"Expected the linear combination to be over dolfinx Functions, got {type(func)}.")
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


def scalar_type_mismatch_message(form: typing.Any) -> str | None:
    """Describe a real/complex dtype mismatch among a form's coefficients, if there is one.

    Accepts a single form or any nesting of forms -- a blocked problem's ``a`` is a list of
    lists, whose entries may be ``None`` -- and considers every coefficient across them, since
    it is their coexistence in one assembly that fails.

    DOLFINx cannot assemble a form whose coefficients do not share one scalar dtype: its
    nanobind bindings are templated on a fixed set of ``(scalar, geometry)`` dtype pairs, so
    a mismatch surfaces several layers down as a template-resolution failure naming neither
    the dtype nor the coefficient responsible.

    Under a complex-scalar build every {py:class}`~dolfinx_adjoint.Function` and
    {py:class}`~dolfinx_adjoint.Constant` built with the default dtype is already complex, so
    a mismatch is reached only when a caller explicitly asked for a real dtype, or passed a
    raw {py:class}`dolfinx.fem.Constant` built from a Python float. This is called only from
    the failure path, so walking the coefficients costs nothing in the normal case.

    Returns:
        A message naming each dtype and the coefficients carrying it, or ``None`` if the
        coefficients agree (in which case the original error was about something else).
    """
    by_dtype: dict[numpy.dtype, list[str]] = {}

    def record(obj) -> None:
        values = getattr(getattr(obj, "x", None), "array", None)
        if values is None:
            values = getattr(obj, "value", None)
        if values is None:
            return
        dtype = numpy.asarray(values).dtype
        if not numpy.issubdtype(dtype, numpy.number):
            return
        name = getattr(obj, "name", None) or repr(obj)
        by_dtype.setdefault(dtype, []).append(str(name))

    def record_form(candidate) -> None:
        if isinstance(candidate, (list, tuple)):
            for nested in candidate:
                record_form(nested)
            return
        if not isinstance(candidate, ufl.Form):
            # Anything that is neither a UFL form nor a nesting of them carries no UFL
            # coefficients to walk: a blocked problem's absent preconditioner (``None``), the
            # ``ufl.ZeroBaseForm`` or plain ``0`` that dropping every term of a form leaves
            # behind, or a form that has already been compiled. Skipping those matters more
            # than it looks, because this runs inside an ``except`` block: raising here
            # ("'ZeroBaseForm' object is not iterable") would replace the very error the
            # diagnostic exists to explain.
            return
        for coefficient in ufl.algorithms.extract_coefficients(candidate):
            record(coefficient)
        try:
            constants = ufl.algorithms.analysis.extract_constants(candidate)
        except AttributeError:  # pragma: no cover - depends on the installed UFL version
            constants = []
        for constant in constants:
            record(constant)

    record_form(form)

    kinds = {numpy.issubdtype(dtype, numpy.complexfloating) for dtype in by_dtype}
    if len(kinds) < 2:
        return None

    listing = "; ".join(f"{dtype}: {', '.join(sorted(names))}" for dtype, names in sorted(by_dtype.items(), key=str))
    return (
        f"This form mixes real- and complex-dtype coefficients, which DOLFINx cannot assemble "
        f"({listing}). Under a complex-scalar build, construct every coefficient with the default "
        f"dtype so they are all complex; a coefficient given an explicit real dtype, or a raw "
        f"dolfinx.fem.Constant built from a Python float, is the usual cause."
    )


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


def _is_complex_build() -> bool:
    """Whether DOLFINx was built against a complex-scalar PETSc."""
    return bool(numpy.issubdtype(dolfinx.default_scalar_type, numpy.complexfloating))


def _compile_form(
    form: typing.Any,
    jit_options: dict | None = None,
    form_compiler_options: dict | None = None,
    entity_maps: typing.Any = None,
) -> typing.Any:
    """Compile a form, explaining a real/complex dtype mix rather than letting it surface raw.

    A thin stand-in for {py:func}`dolfinx.fem.form` with identical arguments and return
    value, differing only in what it does on failure. Every form this package compiles goes
    through it, so the explanation is the default rather than something each call site has to
    remember to opt into: a dtype mix otherwise fails several layers down in the nanobind
    bindings, in a message naming neither the dtype nor the coefficient at fault, and a
    compile site that forgot the wrapper is indistinguishable from one that never needed it.

    See {py:func}`scalar_type_mismatch_message` for what the replacement message says and
    when there is one to say. Any other failure propagates untouched.
    """
    with _explaining_scalar_type_mismatch(form):
        return dolfinx.fem.form(
            form,
            jit_options=jit_options,
            form_compiler_options=form_compiler_options,
            entity_maps=entity_maps,
        )


@contextlib.contextmanager
def _explaining_scalar_type_mismatch(form: typing.Any) -> typing.Iterator[None]:
    """Re-raise a dtype-mix failure from DOLFINx with a message naming the coefficients.

    See {py:func}`scalar_type_mismatch_message` for why the original error names neither.
    Anything else raised inside the block is left alone.
    """
    try:
        yield
    except (TypeError, RuntimeError) as error:
        hint = scalar_type_mismatch_message(form)
        if hint is None:
            raise
        raise type(error)(f"{hint}\n\nOriginal error: {error}") from error
