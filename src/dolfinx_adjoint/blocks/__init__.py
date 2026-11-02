import dolfinx

from .assembly import AssembleBlock
from .dirichletbc import DirichletBCBlock
from .function_assigner import FunctionAssignBlock
from .interpolation import ExprInterpolationBlock, InterpolationBlock
from .nonmatching_interpolation import NonmatchingInterpolationBlock

__all__ = [
    "AssembleBlock",
    "DirichletBCBlock",
    "ExprInterpolationBlock",
    "FunctionAssignBlock",
    "InterpolationBlock",
    "NonmatchingInterpolationBlock",
]

# The solver blocks build on dolfinx.fem.petsc, which DOLFINx built without PETSc (e.g. on Windows) lacks.
if dolfinx.has_petsc4py:
    from .solvers import LinearProblemBlock, NonlinearProblemBlock

    __all__ += ["LinearProblemBlock", "NonlinearProblemBlock"]
