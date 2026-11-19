"""Validated forcing and squared interior residual for u=tanh(x+y-1)."""
from verified_arithmetic import tanh_ranges


def tanh_poisson_forcing(arithmetic, bounds):
    """Delta tanh(x+y-1) = 2*tanh''(x+y-1), with outward ranges."""
    if len(bounds) != 2:
        raise ValueError('Expected a two-dimensional box')
    z = arithmetic.interval(*bounds[0]) + arithmetic.interval(*bounds[1]) - arithmetic.interval(1)
    return arithmetic.interval(2) * tanh_ranges(z)[2]


def poisson_residual_integrand(network, bounds):
    if network.dimension != 2 or len(network.weights[-1]) != 1:
        raise ValueError('Expected a scalar two-dimensional network')
    jet = network.evaluate(bounds, order=2)[0]
    # Packed multi-index Hessian order in two dimensions is xx, xy, yy.
    laplacian = jet.hessian[0] + jet.hessian[2]
    return (laplacian - tanh_poisson_forcing(network.a, bounds)).abs_power(2)
