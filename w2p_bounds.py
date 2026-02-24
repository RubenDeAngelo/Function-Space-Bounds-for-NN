"""
w2p_bounds.py

Implements rigorous upper and lower bounds for the W^{2,p} (Sobolev) norm
by aggregating the L_p bounds of the function, its Jacobian, and its Hessian.

Definition: ||Phi||_{W^{2,p}} = ( ||Phi||_{L^p}^p + ||D Phi||_{L^p}^p + ||D^2 Phi||_{L^p}^p )^{1/p}
"""

from flex_model import extract_numpy_params
from lp_bounds import (
    compute_phi_lp_bounds,
    compute_grad_lp_bounds,
    compute_hess_lp_bounds
)

def compute_w2p_bounds(model, x, delta, p=2.0, volume_lambda=None):
    """
    Computes the lower and upper bounds for the W^{2,p} norm of the network
    within the L_inf ball B(x, delta).

    Arguments:
        model: FlexibleNet instance
        x: Central input point (scalar or array)
        delta: Radius of the Linf ball
        p: Norm order (e.g., 2 for Hilbertian Sobolev norm)
        volume_lambda: Volume of the domain. If None, calculated as (2*delta)^d_in.

    Returns:
        W2p_lower: Lower bound on ||Phi||_{W^{2,p}}
        W2p_upper: Upper bound on ||Phi||_{W^{2,p}}
    """
    # --- 0. Setup Domain Volume ---
    if volume_lambda is None:
        weights, _ = extract_numpy_params(model)
        d_in = weights[0].shape[1]
        volume_lambda = (2.0 * delta) ** d_in

    # --- 1. Compute L_p Bounds on Phi (Function Values) ---
    # Returns norms and intermediate states (x_l, z_l) to optimize Jacobian step
    phi_low_norm, phi_high_norm, x_l, z_l = compute_phi_lp_bounds(
        model, x, delta, p=p, volume_lambda=volume_lambda
    )

    # --- 2. Compute L_p Bounds on Gradient (Jacobian) ---
    # Reuse x_l, z_l from step 1
    grad_low_norm, grad_high_norm = compute_grad_lp_bounds(
        model, x, delta, x_l=x_l, z_l=z_l, p=p, volume_lambda=volume_lambda
    )

    # --- 3. Compute L_p Bounds on Hessian (Second Derivative) ---
    # Currently compute_hess_lp_bounds recalculates internal states for safety
    hess_low_norm, hess_high_norm = compute_hess_lp_bounds(
        model, x, delta, p=p, volume_lambda=volume_lambda
    )

    # --- 4. Combine for W^{2,p} Norm ---
    # Formula: ||Phi||_{W^{2,p}} = ( ||Phi||_{p}^p + ||Grad||_{p}^p + ||Hess||_{p}^p )^{1/p}

    # Lower Bound combination
    term_phi_low = phi_low_norm ** p
    term_grad_low = grad_low_norm ** p
    term_hess_low = hess_low_norm ** p
    W2p_lower = (term_phi_low + term_grad_low + term_hess_low) ** (1.0 / p)

    # Upper Bound combination
    term_phi_high = phi_high_norm ** p
    term_grad_high = grad_high_norm ** p
    term_hess_high = hess_high_norm ** p
    W2p_upper = (term_phi_high + term_grad_high + term_hess_high) ** (1.0 / p)

    return W2p_lower, W2p_upper