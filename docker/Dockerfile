FROM ghcr.io/fenics/dolfinx/dolfinx:nightly

ENV DEB_PYTHON_INSTALL_LAYOUT=deb_system

WORKDIR /tmp/

COPY . ./dolfinx-adjoint
RUN apt update && apt install -y graphviz-dev &&  apt-get clean
RUN python3 -m pip install ./dolfinx-adjoint[all]
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
