from __future__ import annotations

import dolfinx
import numpy as np
import numpy.typing as npt


class _SpecialVector(dolfinx.la.Vector):
    """A `dolfinx.la.Vector` tagged with the function space it is dual to.

    This is the type dxa stores in `pyadjoint.BlockVariable.adj_value` and
    `hessian_value`. It adds two things a bare `dolfinx.la.Vector` lacks:

    - in-place addition, under both the name pyadjoint uses today
      (`_ad_iadd`, called from `BlockVariable.add_adj_output` and
      `add_hessian_output`) and the `+=` operator older pyadjoint releases
      applied directly;
    - the `function_space`/`x`/`name` attributes that let the vector stand in
      for a `Function`, which is what `pyadjoint.Control.get_derivative` needs
      when it hands the value to `Function._ad_init_object`.
    """

    def __init__(self, x, function_space: dolfinx.fem.FunctionSpace):
        super().__init__(x._cpp_object)
        self._function_space = function_space

    def __iadd__(self, other):
        """Add `other` into this vector in place, returning `self`.

        Owned and ghost entries are both added, which needs no communication
        and leaves the result ghost-consistent because every producer of these
        vectors ends in `scatter_forward()`.

        Returning `self` is what lets the one implementation serve both
        pyadjoint generations: the older one rebinds the result of `+=`, the
        newer one calls `_ad_iadd` and discards what it returns.
        """
        # An adjoint value reaches this as a bare vector or as a Function --
        # `blocks.dirichletbc.DirichletBCBlock` passes `adj_inputs[0]` straight
        # through, so the incoming type is not ours to choose.
        rhs = getattr(other, "x", other)
        if self.array.shape != rhs.array.shape or self.block_size != rhs.block_size:
            raise ValueError(
                f"Cannot add vectors of shape {rhs.array.shape} (block size {rhs.block_size}) "
                f"to {self.array.shape} (block size {self.block_size})."
            )
        self.array[:] += rhs.array[:]
        return self

    _ad_iadd = __iadd__

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
    map: dolfinx.common.IndexMap | dolfinx.cpp.common.IndexMap,
    bs: int,
    function_space: dolfinx.fem.FunctionSpace,
    dtype: npt.DTypeLike = np.float64,
) -> _SpecialVector:
    """Create a distributed vector.

    Args:
        map: Index map the describes the size and distribution of the
            vector.
        bs: Block size.
        function_space: The function space the vector is associated with.
        dtype: The scalar type.

    Returns:
        A distributed vector.
    """
    # Delegate to `dolfinx.la.vector` rather than reaching for the nanobind
    # `dolfinx.cpp.la.Vector_*` constructors. Within a DOLFINx release the index
    # map accessors and this factory agree on whether an index map is the raw
    # C++ object (0.11.0) or the Python wrapper (0.12, FEniCS/dolfinx#4496);
    # constructing by hand picks one side of that contract and breaks on the
    # other. `dtype` must stay keyword -- it is keyword-only from 0.12 -- and
    # `scatterer` must not be passed, as 0.11.0 has no such parameter.
    vec = dolfinx.la.vector(map, bs, dtype=dtype)  # type: ignore
    return _SpecialVector(vec, function_space)


def _create_vector(L: dolfinx.fem.Form, space: dolfinx.fem.FunctionSpace) -> _SpecialVector:
    """Create a Vector that is compatible with a given linear form.

    Args:
        L: A linear form.
        space: The function space ``L``'s (single) argument lives on -- must match
            ``L.function_spaces[0]``.

    Returns:
        A vector that the form can be assembled into.
    """
    # Can just take the first dofmap here, since all dof maps have the same
    # index map in mixed-topology meshes

    first_space = L.function_spaces[0]
    dofmap = first_space.dofmaps[0]  # type: ignore
    if isinstance(first_space, dolfinx.fem.FunctionSpace):
        assert space._cpp_object == first_space._cpp_object, "Function space mismatch when creating vector."
    else:
        assert space._cpp_object == first_space  # type: ignore
    return _vector(dofmap.index_map, dofmap.index_map_bs, dtype=L.dtype, function_space=space)
