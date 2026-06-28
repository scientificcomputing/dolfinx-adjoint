from mpi4py import MPI

import dolfinx
import numpy as np
import pyadjoint
import ufl

import dolfinx_adjoint


def test_constant_derivative():
    pyadjoint.get_working_tape().clear_tape()
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)

    m = dolfinx_adjoint.Constant(domain, 3.0)

    dx = ufl.dx(domain)
    J_form = 0.5 * (m - 5.0) ** 2 * dx
    J = dolfinx_adjoint.assemble_scalar(J_form)

    control = pyadjoint.Control(m)
    Jhat = pyadjoint.ReducedFunctional(J, control)

    dJdm = Jhat.derivative()

    expected_derivative = 3.0 - 5.0
    assert np.isclose(dJdm.x.array[0], expected_derivative)
