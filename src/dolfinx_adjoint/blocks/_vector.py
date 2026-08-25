import dolfinx
import numpy as np
import numpy.typing as npt


class _SpecialVector(dolfinx.la.Vector):
    """Workaround adding __iadd__ to `dolfinx.la.Vector`."""

    def __init__(self, x, function_space: dolfinx.fem.FunctionSpace):
        super().__init__(x._cpp_object)
        self._function_space = function_space

    def __iadd__(self, other):
        self.array[:] += other.array[:]
        return self

    @property
    def function_space(self) -> dolfinx.fem.FunctionSpace:
        return self._function_space

    @property
    def x(self):
        return self

    @property
    def name(self):
        return "SpecialVector"


def _vector(
    map, bs: int, function_space: dolfinx.fem.FunctionSpace, dtype: npt.DTypeLike = np.float64
) -> _SpecialVector:
    """Create a distributed vector.

    Args:
        map: Index map the describes the size and distribution of the
            vector.
        bs: Block size.
        dtype: The scalar type.

    Returns:
        A distributed vector.
    """
    vtype: (
        type[dolfinx.cpp.la.Vector_float32]
        | type[dolfinx.cpp.la.Vector_float64]
        | type[dolfinx.cpp.la.Vector_complex64]
        | type[dolfinx.cpp.la.Vector_complex128]
        | type[dolfinx.cpp.la.Vector_int8]
        | type[dolfinx.cpp.la.Vector_int32]
        | type[dolfinx.cpp.la.Vector_int64]
    )
    if np.issubdtype(dtype, np.float32):
        vtype = dolfinx.cpp.la.Vector_float32
    elif np.issubdtype(dtype, np.float64):
        vtype = dolfinx.cpp.la.Vector_float64
    elif np.issubdtype(dtype, np.complex64):
        vtype = dolfinx.cpp.la.Vector_complex64
    elif np.issubdtype(dtype, np.complex128):
        vtype = dolfinx.cpp.la.Vector_complex128
    elif np.issubdtype(dtype, np.int8):
        vtype = dolfinx.cpp.la.Vector_int8
    elif np.issubdtype(dtype, np.int32):
        vtype = dolfinx.cpp.la.Vector_int32
    elif np.issubdtype(dtype, np.int64):
        vtype = dolfinx.cpp.la.Vector_int64
    else:
        raise NotImplementedError(f"Type {dtype} not supported.")

    return _SpecialVector(dolfinx.la.Vector(vtype(map, bs)), function_space)


def _create_vector(L: dolfinx.fem.Form, space: dolfinx.fem.FunctionSpace) -> _SpecialVector:
    """Create a Vector that is compatible with a given linear form.

    Args:
        L: A linear form.

    Returns:
        A vector that the form can be assembled into.
    """
    # Can just take the first dofmap here, since all dof maps have the same
    # index map in mixed-topology meshes

    dofmap = L.function_spaces[0].dofmaps[0]  # type: ignore

    assert space._cpp_object == L.function_spaces[0], "Function space mismatch when creating vector."
    return _vector(dofmap.index_map, dofmap.index_map_bs, dtype=L.dtype, function_space=space)
