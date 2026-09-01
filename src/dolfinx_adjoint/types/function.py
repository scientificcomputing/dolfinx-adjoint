from __future__ import annotations

import typing

import basix.ufl
import dolfinx
import numpy
import numpy.typing as npt
import ufl
from pyadjoint.overloaded_type import (
    FloatingType,
    create_overloaded_object,
    register_overloaded_type,
)
from pyadjoint.tape import no_annotations

from ..blocks._vector import _SpecialVector, _vector
from ..blocks.assembly import assemble_compiled_form
from ..utils import function_from_vector, gather


def _create_function(
    V: dolfinx.fem.FunctionSpace,
    dtype: npt.DTypeLike = dolfinx.default_scalar_type,
) -> Function:
    """Create a Function that is compatible with a given function space.

    Args:
        V: A function space.

    Returns:
        A function that is compatible with the function space.
    """
    x = _vector(V.dofmap.index_map, V.dofmap.index_map_bs, dtype=dtype, function_space=V)
    return Function(V, x=x, annotate=False)


class Function(dolfinx.fem.Function, FloatingType):
    """A class overloading `dolfinx.fem.Function` to support it being used as a control variable
    in the adjoint framework.

    Args:
        V: The function space of the function.
        x: Optional vector to initialize the function with. If not provided, a zero vector is created.
        name: Optional name for the function.
        dtype: Data type of the function values, defaults to `dolfinx.default_scalar_type`.
        **kwargs: Additional keyword arguments to pass to the `pyadjoint.overloaded_type.FloatingType` constructor.

    """

    def __init__(
        self,
        V: dolfinx.fem.FunctionSpace,
        x: dolfinx.la.Vector | None = None,
        name: str | None = None,
        dtype: npt.DTypeLike = dolfinx.default_scalar_type,
        **kwargs,
    ):
        super(Function, self).__init__(
            V,
            x,
            name,
            dtype,
        )
        FloatingType.__init__(
            self,
            V,
            x,
            name=name,
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
        if x is not None:
            self._x = x  # Ensure that the input `x` is stored in case it is a _SpecialVector
        if not isinstance(self.x, _SpecialVector):
            self._x = _SpecialVector(self.x, V)  # Wrap the vector in a _SpecialVector for adjoint operations

    @classmethod
    def _ad_init_object(cls, obj):
        return cls(obj.function_space, obj.x, obj.name)

    @property
    def index_map(self) -> dolfinx.cpp.la.IndexMap:  # type: ignore [name-defined]
        """Return the index map of the function's vector."""
        return self.x.index_map

    @no_annotations
    def _ad_create_checkpoint(self):
        # Note: self.copy() (dolfinx.fem.Function.copy) always returns a plain
        # dolfinx.fem.Function regardless of self's concrete type, so wrapping it with
        # create_overloaded_object would silently downcast a Constant checkpoint to a
        # plain Function. Use _ad_new_like() instead to preserve the concrete subclass.
        checkpoint = self._ad_new_like()
        checkpoint.x.array[:] = self.x.array[:]
        checkpoint.name = self.name + "_checkpoint"
        return checkpoint

    def _ad_restore_at_checkpoint(self, checkpoint):
        return checkpoint

    def _ad_dot(self, other: typing.Self, options: dict | None = None):
        """Compute the inner product of the current function with ``other`` in the Riesz representation.

        Args:
            other: Function to compute the inner product with.
        """
        options = {} if options is None else options
        riesz_representation = options.get("riesz_representation", "l2")
        if riesz_representation == "l2":
            return dolfinx.cpp.la.inner_product(self.x._cpp_object, other.x._cpp_object)  # type: ignore[arg-type]
        elif riesz_representation == "L2":
            form_compiler_options = options.get("form_compiler_options", None)
            jit_options = options.get("jit_options", None)
            mass = ufl.inner(self, other) * ufl.dx
            compiled_form = dolfinx.fem.form(
                mass,
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )
            return assemble_compiled_form(compiled_form)
        elif riesz_representation == "H1":
            form_compiler_options = options.get("form_compiler_options", None)
            jit_options = options.get("jit_options", None)
            mass_and_stiffness = ufl.inner(self, other) * ufl.dx + ufl.inner(ufl.grad(self), ufl.grad(other)) * ufl.dx
            compiled_form = dolfinx.fem.form(
                mass_and_stiffness,
                jit_options=jit_options,
                form_compiler_options=form_compiler_options,
            )
            return assemble_compiled_form(compiled_form)
        else:
            raise NotImplementedError("Unknown Riesz representation %s" % riesz_representation)

    def _ad_new_like(self) -> typing.Self:
        """Create a new, zero-valued instance sharing this object's exact overloaded type and
        function space.

        Constructing via ``type(self)(...)`` directly does not work here because subclasses
        such as ``Constant`` take a different constructor signature (mesh and value, not a
        function space). Going through ``__new__`` and ``Function.__init__`` bypasses that
        constructor while still producing an instance of the correct concrete subclass.
        """
        r = type(self).__new__(type(self), self.function_space)  # type: ignore[call-arg]
        Function.__init__(r, self.function_space)
        return r

    @no_annotations
    def _ad_mul(self, other: typing.Union[int, float]) -> typing.Self:
        """Multiplication of self with integer or floating value."""
        r = self._ad_new_like()
        r.x.array[:] = self.x.array * other
        return r

    @no_annotations
    def _ad_add(self, other: typing.Self) -> typing.Self:
        r = self._ad_new_like()
        r.x.array[:] = self.x.array[:] + other.x.array[:]
        return r

    @no_annotations
    def _ad_convert_riesz(self, value: dolfinx.la.Vector, riesz_map: dict | None = None) -> dolfinx.fem.Function:
        """Convert a vector to a Riesz representation of the function."""
        options = {} if riesz_map is None else riesz_map
        riesz_representation = options.get("riesz_representation", "l2")
        if riesz_representation == "l2":
            return create_overloaded_object(function_from_vector(self.function_space, value))
        elif riesz_representation == "L2":
            from dolfinx.fem.petsc import assemble_matrix

            from dolfinx_adjoint.petsc_utils import solve_linear_problem

            u = ufl.TrialFunction(self.function_space)
            v = ufl.TestFunction(self.function_space)
            riesz_form = ufl.inner(u, v) * ufl.dx
            compiled_riesz = dolfinx.fem.form(
                riesz_form,
                jit_options=options.get("jit_options", None),
                form_compiler_options=options.get("form_compiler_options", None),
            )
            ret = dolfinx.fem.Function(self.function_space)
            M = assemble_matrix(compiled_riesz)
            M.assemble()
            petsc_options = options.get("petsc_options", {})
            solve_linear_problem(M, ret.x, value, petsc_options=petsc_options)
            M.destroy()
            return ret
        elif riesz_representation == "H1":
            from dolfinx.fem.petsc import assemble_matrix

            from dolfinx_adjoint.petsc_utils import solve_linear_problem

            u = ufl.TrialFunction(self.function_space)
            v = ufl.TestFunction(self.function_space)
            riesz_form = ufl.inner(u, v) * ufl.dx + ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
            compiled_riesz = dolfinx.fem.form(
                riesz_form,
                jit_options=options.get("jit_options", None),
                form_compiler_options=options.get("form_compiler_options", None),
            )
            ret = dolfinx.fem.Function(self.function_space)
            M = assemble_matrix(compiled_riesz)
            M.assemble()
            petsc_options = options.get("petsc_options", {})
            solve_linear_problem(M, ret.x, value, petsc_options=petsc_options)
            M.destroy()
            return ret
        elif callable(riesz_representation):
            return riesz_representation(value)
        else:
            raise NotImplementedError("Unknown Riesz representation %s" % riesz_representation)

    @staticmethod
    def _ad_to_list(m):
        """Convert a function into a list of (global) values."""
        if not hasattr(m, "array"):
            m_v = m.x
        else:
            m_v = m
        m_a = gather(m_v)
        return m_a.tolist()

    def _ad_copy(self):
        """Create a (deep) copy of the function."""
        r = self._ad_new_like()
        r.x.array[:] = self.x.array[:].copy()
        return r

    @staticmethod
    def _ad_assign_numpy(dst: dolfinx.fem.Function, src: numpy.ndarray, offset: int):
        range_begin, range_end = dst.x.index_map.local_range
        range_begin *= dst.x.block_size
        range_end *= dst.x.block_size
        m_a_local = src[offset + range_begin : offset + range_end]
        dst.x.array[: len(m_a_local)] = m_a_local
        offset += dst.x.index_map.size_local * dst.x.block_size
        dst.x.scatter_forward()
        return dst, offset

    @property
    def x(self) -> dolfinx.la.Vector:
        """Return the underlying vector of the function."""
        return self._x


class Constant(Function):
    """A class overloading {py:class}`dolfinx.fem.Constant`
    to support it being used as a control variable in
    the adjoint framework.

    Args:
        domain: The mesh on which the constant is defined.
        c: The value of the constant. Can be a scalar, a sequence, or a numpy array.

    Note:
        The {py:class}`Constant` class is implemented as a subclass of {py:class}`Function` to leverage the
        existing functionality for handling function spaces and vectors. The value of
        the constant is stored in the underlying vector of the function, and the class
        provides a property to access this value conveniently.

        If {py:func}`basix.ufl.real_element` is not available, the class will attempt to use
        {py:mod}`scifem` to create a function space for the constant (which would then require
        {py:mod}`scifem` to be installed - :code:`pip install scifem`).

    """

    def __init__(
        self,
        domain: dolfinx.mesh.Mesh,
        c: float | numpy.floating | complex | numpy.complexfloating | typing.Sequence | numpy.ndarray,
    ):
        value_shape = numpy.shape(c)
        try:
            el = basix.ufl.real_element(domain.basix_cell(), value_shape=numpy.shape(c))
            V = dolfinx.fem.functionspace(domain, el)

        except AttributeError:
            try:
                import scifem
            except ImportError as e:
                raise ImportError("scifem is required to use Constant 'pip install scifem") from e

            V = scifem.create_real_functionspace(domain, value_shape=value_shape)
        super().__init__(V)
        self.x.array[:] = c

    @property
    def value(self):
        return self.x.array[:]

    @classmethod
    def _ad_init_object(cls, obj):
        return cls(obj.function_space.mesh, obj.x.array[:])


register_overloaded_type(Function, (dolfinx.fem.Function, Function))
register_overloaded_type(Constant, (dolfinx.fem.Constant, Constant))
