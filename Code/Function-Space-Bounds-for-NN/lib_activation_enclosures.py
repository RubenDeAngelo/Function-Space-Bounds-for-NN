"""
Activation function evaluations and interval enclosures for sigma, sigma', sigma''.
Used by Jacobian and Hessian bounding algorithms.
"""

import numpy as np


def sigma(model, z):
    name = model.activation_name
    if name == 'relu':
        return np.maximum(0, z)
    elif name == 'tanh':
        return np.tanh(z)
    elif name == 'sigmoid':
        zc = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-zc))
    else:
        raise ValueError(f"Unsupported activation: {name}")


def sigma_derivative(model, z):
    name = model.activation_name
    if name == 'relu':
        return (z > 0).astype(z.dtype)
    elif name == 'tanh':
        return 1 - np.tanh(z) ** 2
    elif name == 'sigmoid':
        zc = np.clip(z, -500, 500)
        s = 1 / (1 + np.exp(-zc))
        return s * (1 - s)
    else:
        raise ValueError(f"Unsupported activation: {name}")


def sigma_second_derivative(model, z):
    name = model.activation_name
    if name == 'relu':
        return np.zeros_like(z)
    elif name == 'tanh':
        tanh_z = np.tanh(z)
        return -2 * tanh_z * (1 - tanh_z ** 2)
    elif name == 'sigmoid':
        zc = np.clip(z, -500, 500)
        s = 1 / (1 + np.exp(-zc))
        return s * (1 - s) * (1 - 2 * s)
    else:
        raise ValueError(f"Unsupported activation: {name}")


def get_analytical_lipschitz_constants(model, x, w):
    """
    Returns (L, M): lower and upper bounds on sigma'(z) for z in [x-w, x+w].
    Used to enclose the first-derivative interval for each neuron.
    """
    activation_name = model.activation_name
    a = x - w
    b = x + w

    if w <= 0:
        val = float(sigma_derivative(model, np.array([x]))[0])
        return val, val

    a = np.clip(a, -500, 500)
    b = np.clip(b, -500, 500)

    if activation_name in ('tanh', 'sigmoid'):
        candidates = [a, b]
        if a <= 0 <= b:
            candidates.append(0.0)  # sigma' has its maximum at 0

        vals = sigma_derivative(model, np.array(candidates))
        return float(np.min(vals)), float(np.max(vals))

    elif activation_name == 'relu':
        if b <= 0:
            return 0.0, 0.0
        elif a >= 0:
            return 1.0, 1.0
        else:
            return 0.0, 1.0

    else:
        raise NotImplementedError(f"Analytical Lipschitz (Jacobian) not implemented for {activation_name}")


def get_analytical_lipschitz_constants_hess(model, x, w):
    """
    Returns (gamma, eta): lower and upper bounds on sigma''(z) for z in [x-w, x+w].
    Critical points are precomputed analytically per activation.
    """
    activation = model.activation_name
    a, b = float(x - w), float(x + w)

    if w <= 0:
        val = float(sigma_second_derivative(model, np.array([x]))[0])
        return val, val

    a, b = np.clip(a, -500, 500), np.clip(b, -500, 500)

    candidates = [a, b]

    if activation == 'sigmoid':
        # sigma''(z) extrema at z ≈ ±1.317 (roots of 6s²-6s+1=0) and z=0
        criticals = [0.0, 1.3169579, -1.3169579]

    elif activation == 'tanh':
        # sigma''(z) extrema at z = ±arctanh(1/sqrt(3)) ≈ ±0.658
        criticals = [0.0, 0.65847894, -0.65847894]

    elif activation == 'relu':
        return 0.0, 0.0

    else:
        raise NotImplementedError(f"Hessian interval bounds not implemented for {activation}")

    for c in criticals:
        if a - 1e-9 <= c <= b + 1e-9:
            candidates.append(c)

    vals = sigma_second_derivative(model, np.array(candidates))
    return float(np.min(vals)), float(np.max(vals))


def compute_sigma_jac_interval(model, d_layer, z_flat, omega_vec):
    """Returns (alpha, beta) per-neuron first-derivative bounds for one layer."""
    beta_vec = np.empty(d_layer)
    alpha_vec = np.empty(d_layer)
    for k in range(d_layer):
        alpha_vec[k], beta_vec[k] = get_analytical_lipschitz_constants(
            model, z_flat[k], omega_vec[k]
        )
    return alpha_vec, beta_vec


def compute_sigma_jac_intervals_all_layers(model, z_l, omega_all):
    """
    Returns (alpha_all, beta_all): first-derivative bounds for every layer.
    The final (linear) layer appends [1, 1] bounds.
    """
    L = len(z_l)
    alpha_all = []
    beta_all = []

    for l in range(L - 1):  # hidden layers only
        z_flat = np.atleast_1d(np.squeeze(z_l[l]))
        omega_vec = omega_all[l]
        d_layer = z_flat.shape[0]
        alpha_vec, beta_vec = compute_sigma_jac_interval(model, d_layer, z_flat, omega_vec)
        alpha_all.append(alpha_vec)
        beta_all.append(beta_vec)

    # Final linear layer: sigma' = 1
    d_last_layer = z_l[L - 1].shape[0]
    alpha_all.append(np.ones(d_last_layer))
    beta_all.append(np.ones(d_last_layer))

    return alpha_all, beta_all


def compute_sigma_hess_interval(model, d_layer, z_flat, omega_vec):
    """Returns (gamma, eta) per-neuron second-derivative bounds for one layer."""
    gamma_vec, eta_vec = np.empty(d_layer), np.empty(d_layer)
    for k in range(d_layer):
        gamma_vec[k], eta_vec[k] = get_analytical_lipschitz_constants_hess(
            model, z_flat[k], omega_vec[k]
        )
    return gamma_vec, eta_vec


def compute_sigma_hess_intervals_all_layers(model, z_l, omega_all):
    """
    Returns (gamma_all, eta_all): second-derivative bounds for every layer.
    The final (linear) layer appends [0, 0] bounds.
    """
    L = len(z_l)
    gamma_all, eta_all = [], []

    for l in range(L - 1):  # hidden layers only
        z_flat = np.atleast_1d(np.squeeze(z_l[l]))
        omega_vec = omega_all[l]
        d_layer = z_flat.shape[0]
        gamma_vec, eta_vec = compute_sigma_hess_interval(model, d_layer, z_flat, omega_vec)
        gamma_all.append(gamma_vec)
        eta_all.append(eta_vec)

    # Final linear layer: sigma'' = 0
    d_last_layer = z_l[L - 1].shape[0]
    gamma_all.append(np.zeros(d_last_layer))
    eta_all.append(np.zeros(d_last_layer))

    return gamma_all, eta_all
