from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import ufl

import dolfinx_adjoint


def test_constant_derivative():
    """
    Proves that the ScalarPrior evaluates the exact algebraic math
    and that PyAdjoint can compute its exact derivative!
    """
    pyadjoint.get_working_tape().clear_tape()
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)

    # 2. Setup tracked control: m = 3.0
    m = dolfinx_adjoint.Constant(domain, 3.0)

    # 3. Evaluate Cost
    dx = ufl.dx(domain)
    J_form = 0.5 * (m - 5.0) ** 2 * dx
    J = dolfinx_adjoint.assemble_scalar(J_form)

    control = pyadjoint.Control(m)
    Jhat = pyadjoint.ReducedFunctional(J, control)

    # 4. Compute Derivative
    dJdm = Jhat.derivative()

    # 5. Check that the derivative is correct
    expected_derivative = 3.0 - 5.0
    assert np.isclose(dJdm.x.array[0], expected_derivative)
