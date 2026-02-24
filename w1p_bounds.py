"""
w1p_bounds.py

Implements rigorous upper and lower bounds for the W^{1,p} (Sobolev) norm
by aggregating the L_p bounds of the function and its Jacobian.

Relies on lp_bounds.py for the core integral bounding logic.
"""

from flex_model import extract_numpy_params
from lp_bounds import compute_phi_lp_bounds, compute_grad_lp_bounds


def compute_w1p_bounds(model, x, delta, p=2.0, volume_lambda=None):
    """
    Computes the lower and upper bounds for the W^{1,p} norm of the network
    within the L_inf ball B(x, delta).

    Arguments:
        model: FlexibleNet instance
        x: Central input point (scalar or array)
        delta: Radius of the Linf ball
        p: Norm order (e.g., 2 for Hilbertian Sobolev norm)
        volume_lambda: Volume of the domain. If None, calculated as (2*delta)^d_in.

    Returns:
        W1p_lower: Lower bound on ||Phi||_{W^{1,p}}
        W1p_upper: Upper bound on ||Phi||_{W^{1,p}}
    """
    # --- 0. Setup Domain Volume ---
    # We need to ensure volume is consistent across calls if not provided
    if volume_lambda is None:
        weights, _ = extract_numpy_params(model)
        d_in = weights[0].shape[1]
        volume_lambda = (2.0 * delta) ** d_in

    # --- 1. Compute L_p Bounds on Phi (Function Values) ---
    # compute_phi_lp_bounds returns the Norm values (not integrals) and intermediate states
    phi_low_norm, phi_high_norm, x_l, z_l = compute_phi_lp_bounds(
        model, x, delta, p=p, volume_lambda=volume_lambda
    )

    # --- 2. Compute L_p Bounds on Gradient (Jacobian) ---
    # We pass x_l, z_l to reuse the forward pass computations
    grad_low_norm, grad_high_norm = compute_grad_lp_bounds(
        model, x, delta, x_l=x_l, z_l=z_l, p=p, volume_lambda=volume_lambda
    )

    # --- 3. Combine for Sobolev Norm ---
    # Definition: ||Phi||_{W^{1,p}} = ( ||Phi||_{L^p}^p + ||Grad||_{L^p}^p )^{1/p}

    # Lower Bound combination
    # Note: norms are non-negative, so simple exponentiation is safe
    term_phi_low = phi_low_norm ** p
    term_grad_low = grad_low_norm ** p
    W1p_lower = (term_phi_low + term_grad_low) ** (1.0 / p)

    # Upper Bound combination
    term_phi_high = phi_high_norm ** p
    term_grad_high = grad_high_norm ** p
    W1p_upper = (term_phi_high + term_grad_high) ** (1.0 / p)

    return W1p_lower, W1p_upper