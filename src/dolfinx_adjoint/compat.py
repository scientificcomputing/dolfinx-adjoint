import dolfinx


def get_interpolation_points(V: dolfinx.fem.FunctionSpace):
    """Get the interpolation points for a given function space V."""
    try:
        return V.element.interpolation_points()  # type: ignore[operator]
    except TypeError:
        return V.element.interpolation_points
