"""Accumulation of adjoint/tangent contributions onto a single block variable.

A :py:class:`pyadjoint.block_variable.BlockVariable` collects one contribution
per fan-in edge. Every failure mode here is a silently wrong -- or, for the
tangent path, a loudly broken -- derivative rather than a numerical
inaccuracy, so the invariants are pinned directly rather than being left to
the Taylor tests.
"""

from mpi4py import MPI

import dolfinx
import numpy as np
import pytest
import ufl
from pyadjoint.block_variable import BlockVariable

from dolfinx_adjoint import Function
from dolfinx_adjoint.blocks._vector import _vector


@pytest.fixture(scope="module")
def V():
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
    return dolfinx.fem.functionspace(mesh, ("Lagrange", 1))


def _pyadjoint_accumulates_via_ad_iadd() -> bool:
    """Whether pyadjoint routes accumulation through ``_ad_iadd``.

    Releases up to 2026.9 apply the ``+=`` operator instead, which reaches
    :py:meth:`dolfinx_adjoint.Function.__add__` (i.e. UFL's) rather than
    ``_ad_iadd``. Probed by behaviour rather than by version, since the two
    spellings coexist across branches.
    """

    class _Probe:
        def __init__(self):
            self.hits = 0

        def _ad_iadd(self, other):
            self.hits += 1
            return self

        def __iadd__(self, other):
            # Present so the `+=` spelling resolves instead of raising, which
            # would make this probe report a failure rather than an answer.
            return self

    probe = _Probe()
    bv = BlockVariable(None)
    bv.tlm_value = probe
    bv.add_tlm_output(probe)
    return probe.hits > 0


ACCUMULATES_VIA_AD_IADD = _pyadjoint_accumulates_via_ad_iadd()


# -- the adjoint/Hessian path: vectors -------------------------------------
#
# This is the reachable fan-in today. Instrumenting the suite shows 195
# adjoint and 4 Hessian fan-in events, every one of them a `_SpecialVector`.


@pytest.mark.parametrize(
    "add_output, slot",
    [
        (BlockVariable.add_adj_output, "adj_value"),
        (BlockVariable.add_hessian_output, "hessian_value"),
    ],
)
def test_vector_contributions_accumulate(V, add_output, slot):
    """Two contributions to one block variable must sum, under either pyadjoint spelling.

    ``_SpecialVector`` carries ``__iadd__`` and ``_ad_iadd`` as the same
    function precisely so that the ``+=`` and ``_ad_iadd`` call sites both work.
    """
    bv = BlockVariable(None)
    for value in (1.0, 10.0):
        contribution = _vector(V.dofmap.index_map, V.dofmap.index_map_bs, V)
        contribution.array[:] = value
        add_output(bv, contribution)

    accumulated = getattr(bv, slot)
    assert accumulated.function_space is V
    assert np.allclose(accumulated.array, 11.0)


def test_vector_accumulation_rejects_mismatched_space(V):
    """Adding across function spaces is a bug, not a broadcast."""
    W = dolfinx.fem.functionspace(V.mesh, ("Lagrange", 2))
    a = _vector(V.dofmap.index_map, V.dofmap.index_map_bs, V)
    b = _vector(W.dofmap.index_map, W.dofmap.index_map_bs, W)
    with pytest.raises(ValueError):
        a._ad_iadd(b)


# -- the tangent path: functions -------------------------------------------


def test_function_ad_iadd_sums_dofs(V):
    """``Function._ad_iadd`` must add dof values, not build a symbolic sum.

    ``Function`` is a :py:class:`ufl.Coefficient`, so the inherited
    :py:meth:`pyadjoint.OverloadedType._ad_iadd` -- which applies ``+=`` --
    produces a :py:class:`ufl.algebra.Sum` and discards it. Overriding it is
    what keeps a tangent value a ``Function``.
    """
    f, g = Function(V), Function(V)
    f.x.array[:] = 1.0
    g.x.array[:] = 10.0

    accumulated = f._ad_iadd(g)

    assert accumulated is f
    assert isinstance(accumulated, Function)
    assert np.allclose(f.x.array, 11.0)


def test_function_plus_stays_symbolic(V):
    """``__iadd__`` is deliberately *not* defined: ``f += g`` keeps its UFL meaning.

    Accumulation is spelled ``_ad_iadd`` so that the public type's arithmetic
    stays symbolic; making ``+=`` an in-place array write would silently change
    what every user's ``f += g`` means.
    """
    f, g = Function(V), Function(V)
    f += g
    assert isinstance(f, ufl.algebra.Sum)


@pytest.mark.xfail(
    not ACCUMULATES_VIA_AD_IADD,
    reason="pyadjoint accumulates with '+=', which reaches UFL's __add__ rather than "
    "Function._ad_iadd, so tlm_value degrades to a ufl.algebra.Sum",
    strict=False,
)
def test_tangent_contributions_accumulate(V):
    """A second tangent contribution must accumulate into the Function.

    Not reachable through dxa's own API today -- every block that outputs a
    Function registers a fresh block variable, so instrumenting the suite
    records zero tangent fan-in events against 678 ``add_tlm_output`` calls.
    The invariant is pinned here because the consumers assume it:
    ``_ProblemBlockBase.prepare_evaluate_tlm`` reads ``tlm_value.x.array``,
    which a ``ufl.algebra.Sum`` does not have.
    """
    f, g, h = Function(V), Function(V), Function(V)
    g.x.array[:] = 1.0
    h.x.array[:] = 10.0

    bv = BlockVariable(f)
    bv.add_tlm_output(g)
    bv.add_tlm_output(h)

    assert isinstance(bv.tlm_value, Function), (
        f"tangent value degraded to {type(bv.tlm_value).__name__}; "
        "a consumer reading .x.array will raise AttributeError"
    )
    assert np.allclose(bv.tlm_value.x.array, 11.0)
