# DOLFINx-Adjoint

**DOLFINx-Adjoint** is an algorithmic differentiation (AD) framework for [DOLFINx](https://github.com/FEniCS/dolfinx). It allows you to automatically compute the gradients and Hessians of PDE-constrained optimization problems and track your computational graphs through the `pyadjoint` backend.

Read the [Latest Documentation here](https://scientificcomputing.github.io/dolfinx-adjoint).

> **Note for Legacy FEnICS Users:** If you are using the legacy FEnICS (`dolfin`) library, please refer to the original [dolfin-adjoint repository](https://github.com/dolfin-adjoint/dolfin-adjoint). This repository (DOLFINx-Adjoint) is built specifically for the modern DOLFINx environment and is currently under active development. It is intended to eventually support the same comprehensive feature set and capabilities as the legacy version.

## Features

* **Automated Adjoints:** Seamlessly derive discrete adjoint models from DOLFINx forward models.

* **Overloaded API:** Swap out standard `dolfinx` calls with their `dolfinx_adjoint` equivalents (e.g., `Function`, `Constant`, `LinearProblem`, `NonlinearProblem`, `assemble_scalar`) to automatically record the computational tape.

* **Optimization:** Seamless integration with PDE-constrained optimization frameworks like [Moola](https://github.com/funsim/moola) or SciPy via `pyadjoint.ReducedFunctional`.

## Installation

### Dependencies

DOLFINx-Adjoint requires **DOLFINx** (>=0.11.0) and **pyadjoint-ad**.

### via pip

The main way to install the package is via pip:

```bash
python3 -m pip install dolfinx-adjoint
```

### via conda

You can also install dolfinx-adjoint via conda which also comes with **DOLFINx**

```bash
conda install -c conda-forge dolfinx-adjoint
```

### Development Install

To install the latest development version directly from the repository, use:

```bash
python3 -m pip install git+[https://github.com/scientificcomputing/dolfinx-adjoint.git](https://github.com/scientificcomputing/dolfinx-adjoint.git)
```

If you plan to actively modify the code, clone the repository and install the optional dependencies for testing, development, and documentation generation:

```bash
git clone [https://github.com/scientificcomputing/dolfinx-adjoint.git](https://github.com/scientificcomputing/dolfinx-adjoint.git)
cd dolfinx-adjoint
python3 -m pip install -e ".[all]"
```


## Quick Start

Using `dolfinx_adjoint` is designed to be as close to standard `dolfinx` syntax as possible. Here is a brief overview of how to track a parameter and assemble an objective functional:
<!-- #endregion -->

```python
import dolfinx
from mpi4py import MPI
import pyadjoint
import ufl
import dolfinx_adjoint

# Create mesh and function space
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

# Use dolfinx_adjoint overloaded types
# This ensures operations are tracked on the pyadjoint tape!
f = dolfinx_adjoint.Function(V, name="Control")
f.interpolate(lambda x: x[0] + x[1]) # Initial guess for control
uh = dolfinx_adjoint.Function(V, name="State")

# Define UFL forms for a simple Poisson problem: - \Delta u = f
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx

# Set up Dirichlet boundary condition (u = 0 on boundary)
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
exterior_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim - 1, exterior_facets)
bc = dolfinx.fem.dirichletbc(dolfinx.default_scalar_type(0.0), exterior_dofs, V)

# Use overloaded solvers
problem = dolfinx_adjoint.LinearProblem(a, L, u=uh, bcs=[bc])
problem.solve()

# Define a desired temperature profile 'd' and regularization parameter 'alpha'
x = ufl.SpatialCoordinate(mesh)
d = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1e-6))

# Assemble the objective scalar using the overloaded assembly
J_symbolic = 0.5 * ufl.inner(uh - d, uh - d) * ufl.dx + 0.5 * alpha * ufl.inner(f, f) * ufl.dx
J = dolfinx_adjoint.assemble_scalar(J_symbolic)

# Create a ReducedFunctional for optimization
control = pyadjoint.Control(f)
Jhat = pyadjoint.ReducedFunctional(J, control)

# Evaluate gradient
gradient = Jhat.derivative()
```

<!-- #region -->
For more comprehensive examples, such as solving the optimal control of the Poisson equation or time-distributed control problems, check out the `demos/` directory or the [online documentation](https://scientificcomputing.github.io/dolfinx-adjoint).

## Development and Testing

Code formatting is enforced via `ruff` and type-checking via `mypy`. To set up your local development environment:

```bash
# Install development dependencies
python3 -m pip install -e ".[dev,test]"

# Run formatting checks
ruff check .
ruff format --check .

# Run type checking
python3 -m mypy .

# Run tests
python3 -m pytest -vs tests/
```

## License

MIT License. See [LICENSE](LICENSE) for more details.
<!-- #endregion -->
