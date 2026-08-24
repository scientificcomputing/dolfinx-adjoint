"""Top-level package for dxa."""

from importlib.metadata import metadata

# Start annotation at import
import pyadjoint as _pyad

from .assembly import assemble_scalar, error_norm
from .function import assign
from .interpolation import interpolate
from .solvers import LinearProblem, NonlinearProblem
from .types import Constant, Function, dirichletbc

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
    "LinearProblem",
    "NonlinearProblem",
    "assemble_scalar",
    "assign",
    "error_norm",
    "__version__",
    "__author__",
    "__license__",
    "__email__",
    "__program_name__",
    "interpolate",
]
