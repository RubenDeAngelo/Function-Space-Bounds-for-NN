"""
W^{1,p} Sobolev norm enclosure: combines phi-Lp and gradient-Lp bounds (Corollary 4.22).
Returns (lower, upper) bounds on ||f||_{W^{1,p}(B)} for a box B centered at x with half-width delta.
"""

from lib_nn_model import extract_numpy_params
from lib_lp_enclosures import compute_phi_lp_bounds, compute_grad_lp_bounds


def compute_w1p_bounds(model, x, delta, p=2.0, volume_lambda=None, jacobian_method='ibp'):
    if volume_lambda is None:
        weights, _ = extract_numpy_params(model)
        d_in = weights[0].shape[1]
        volume_lambda = (2.0 * delta) ** d_in

    phi_low_norm, phi_high_norm, x_l, z_l = compute_phi_lp_bounds(
        model, x, delta, p=p, volume_lambda=volume_lambda
    )

    grad_low_norm, grad_high_norm = compute_grad_lp_bounds(
        model, x, delta, x_l=x_l, z_l=z_l, p=p, volume_lambda=volume_lambda,
        jacobian_method=jacobian_method
    )

    W1p_lower = (phi_low_norm ** p + grad_low_norm ** p) ** (1.0 / p)
    W1p_upper = (phi_high_norm ** p + grad_high_norm ** p) ** (1.0 / p)

    return W1p_lower, W1p_upper
