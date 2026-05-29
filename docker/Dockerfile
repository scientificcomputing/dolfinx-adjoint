FROM ghcr.io/fenics/dolfinx/dolfinx:nightly

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

WORKDIR /tmp/

COPY . ./dolfinx-adjoint
RUN python3 -m pip install ./dolfinx-adjoint[all]


RUN rm -rf /tmp
