from flex_model import extract_numpy_params
from lp_bounds import compute_phi_lp_bounds, compute_grad_lp_bounds

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
        jacobian_method=jacobian_method  # Parameter durchreichen
    )

    term_phi_low = phi_low_norm ** p
    term_grad_low = grad_low_norm ** p
    W1p_lower = (term_phi_low + term_grad_low) ** (1.0 / p)

    term_phi_high = phi_high_norm ** p
    term_grad_high = grad_high_norm ** p
    W1p_upper = (term_phi_high + term_grad_high) ** (1.0 / p)

    return W1p_lower, W1p_upper