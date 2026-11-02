"""Top-level package for dolfinx_adjoint."""

from importlib.metadata import metadata

import dolfinx as _dolfinx

# Start annotation at import
import pyadjoint as _pyad

from .assembly import assemble_scalar, error_norm
from .checkpointing import enable_disk_checkpointing
from .function import assign
from .interpolation import interpolate, interpolate_nonmatching
from .types import Constant, Function, dirichletbc

# The solvers build on dolfinx.fem.petsc, which DOLFINx built without PETSc (e.g. on Windows) lacks.
if _dolfinx.has_petsc4py:
    from .solvers import LinearProblem, NonlinearProblem

meta = metadata("dolfinx_adjoint")
__version__ = meta.get("Version")
__license__ = meta.get("License")
__author__ = meta.get("Author")
__email__ = meta.get("Author-email")
__program_name__ = meta.get("Name")


_pyad.set_working_tape(_pyad.Tape())
_pyad.continue_annotation()

__all__ = [
    "Constant",
    "Function",
    "dirichletbc",
    "assemble_scalar",
    "assign",
    "enable_disk_checkpointing",
    "error_norm",
    "__version__",
    "__author__",
    "__license__",
    "__email__",
    "__program_name__",
    "interpolate",
    "interpolate_nonmatching",
]

if _dolfinx.has_petsc4py:
    __all__ += ["LinearProblem", "NonlinearProblem"]
else:

    def __getattr__(name: str):
        if name in ("LinearProblem", "NonlinearProblem"):
            raise ImportError(f"dolfinx_adjoint.{name} requires DOLFINx built with PETSc (petsc4py).")
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
