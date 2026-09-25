"""Assembling a compiled form into a vector or scalar, with no dependency on dxa's types."""

import typing

from mpi4py import MPI

import dolfinx

from ._vector import _SpecialVector


def assemble_compiled_form(
    form: dolfinx.fem.Form,
    tensor: typing.Union[dolfinx.la.Vector, _SpecialVector | float] | None = None,
    finalize: bool = True,
) -> typing.Union[dolfinx.la.Vector, _SpecialVector, float]:
    """Assemble a compiled form into ``tensor`` (or return a new scalar).

    Args:
        form: Compiled form to assemble.
        tensor: For a rank-1 form, the vector to accumulate the assembled contribution
            into, while it is unused for a rank-0 form.
        finalize: For rank-0 forms, wheter to sum contributions from all ranks.
            For rank-1 forms finalize does a reverse accumulation of ghost entries and a
            scatter forward to update ghost entries.
    Returns:
        For a rank-1 form, ``tensor`` itself (mutated in place). For a rank-0 form, the
        assembled scalar as a Python ``float``.
    Raises:
        NotImplementedError: If the form's rank is not 0 or 1.
    """

    if form.rank == 1:
        if tensor is None:
            raise ValueError("tensor must be provided for rank-1 forms.")
        assert isinstance(tensor, dolfinx.la.Vector)
        dolfinx.fem.assemble._assemble_vector_array(tensor.array, form)
        if finalize:
            tensor.scatter_reverse(dolfinx.la.InsertMode.add)
            tensor.scatter_forward()
    elif form.rank == 0:
        local_val = dolfinx.fem.assemble_scalar(form)
        comm = form.mesh.comm
        tensor = comm.allreduce(local_val, op=MPI.SUM)
    else:
        raise NotImplementedError("Only 1-form assembly is currently supported.")
    assert tensor is not None
    return tensor
