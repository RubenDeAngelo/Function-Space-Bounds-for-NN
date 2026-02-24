import numpy as np


# --- EXISTING FUNCTIONS (from Jacobian code) ---


def sigma(model, z):
    """
    Computes the activation function output for the given model.
    """
    name = model.activation_name
    if name == 'relu':
        return np.maximum(0, z)
    elif name == 'tanh':
        return np.tanh(z)
    elif name == 'sigmoid':
        # Clip to avoid overflow in exp
        zc = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-zc))
    else:
        raise ValueError(f"Unsupported activation: {name}")


def sigma_derivative(model, z):
    """
    Computes the activation function's derivative.
    """
    name = model.activation_name
    if name == 'relu':
        return (z > 0).astype(z.dtype)
    elif name == 'tanh':
        return 1 - np.tanh(z) ** 2
    elif name == 'sigmoid':
        # Clip to avoid overflow in exp
        zc = np.clip(z, -500, 500)
        s = 1 / (1 + np.exp(-zc))
        return s * (1 - s)
    else:
        raise ValueError(f"Unsupported activation: {name}")


def get_analytical_lipschitz_constants(model, x, w):
    """
    Computes the lower (L) and upper (M) bounds on the *first* derivative
    of the activation function over the interval [x-w, x+w].
    """
    activation_name = model.activation_name
    a = x - w
    b = x + w

    if w <= 0:
        # Interval is a single point
        val = float(sigma_derivative(model, np.array([x]))[0])
        return val, val

    a = np.clip(a, -500, 500)
    b = np.clip(b, -500, 500)

    if activation_name == 'tanh' or activation_name == 'sigmoid':
        # These functions are symmetric, 0 is the max for sigma'
        candidates = [a, b]
        if a <= 0 <= b:
            candidates.append(0.0)

        vals = sigma_derivative(model, np.array(candidates))
        M = np.max(vals)
        L = np.min(vals)
        return L, M

    elif activation_name == 'relu':
        if b <= 0:
            L, M = 0.0, 0.0
        elif a >= 0:
            L, M = 1.0, 1.0
        else:  # a < 0 < b
            L, M = 0.0, 1.0
        return L, M

    else:
        raise NotImplementedError(f"Analytical Lipschitz (Jacobian) not implemented for {activation_name}")


def compute_sigma_jac_interval(model, d_layer, z_flat, omega_vec):
    """
    Computes the first derivative bounds (alpha, beta) for each neuron
    in a single layer.
    """
    beta_vec = np.empty(d_layer)
    alpha_vec = np.empty(d_layer)
    for k in range(d_layer):
        # Note: Your Hessian code had a different (but equivalent) get_analytical_lipschitz_constants.
        # Reusing the existing one from the Jacobian code for consistency.
        alpha_vec_k, beta_vec_k = get_analytical_lipschitz_constants(
            model, z_flat[k], omega_vec[k]
        )
        alpha_vec[k] = alpha_vec_k
        beta_vec[k] = beta_vec_k
    return alpha_vec, beta_vec


def compute_sigma_jac_intervals_all_layers(model, z_l, omega_all):
    """
    Computes the first derivative bounds (alpha, beta) for all layers.
    """
    L = len(z_l)
    alpha_all = []
    beta_all = []

    # --- (FIX) ---
    # Loop only over HIDDEN layers (from 0 to L-2)
    for l in range(L - 1):
        # --- (END FIX) ---
        z_flat = np.atleast_1d(np.squeeze(z_l[l]))
        omega_vec = omega_all[l]
        d_layer = z_flat.shape[0]
        alpha_vec, beta_vec = compute_sigma_jac_interval(
            model, d_layer, z_flat, omega_vec
        )
        alpha_all.append(alpha_vec)
        beta_all.append(beta_vec)

    # --- (FIX) ---
    # Manually add bounds for the final LINEAR layer (l = L-1)
    # sigma' = 1, so bounds are [1, 1]
    d_last_layer = z_l[L - 1].shape[0]
    alpha_all.append(np.ones(d_last_layer))
    beta_all.append(np.ones(d_last_layer))
    # --- (END FIX) ---

    return alpha_all, beta_all


# --- NEW HESSIAN-SPECIFIC FUNCTIONS ---


def sigma_second_derivative(model, z):
    """
    Computes the activation function's second derivative.
    """
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


def get_analytical_lipschitz_constants_hess(model, x, w):
    """
    Computes the lower (gamma) and upper (eta) bounds on the *second* derivative
    of the activation function over the interval [x-w, x+w].

    This is now a fully analytical function, not based on sampling.
    """
    activation = model.activation_name
    a, b = float(x - w), float(x + w)

    if w <= 0:
        val = float(sigma_second_derivative(model, np.array([x]))[0])
        return val, val

    a, b = np.clip(a, -500, 500), np.clip(b, -500, 500)

    # Start with the endpoints
    candidates = [a, b]

    if activation == 'sigmoid':
        # Critical points of sigma''(z) for sigmoid are where
        # 6s^2 - 6s + 1 = 0, where s = sigmoid(z).
        # z = ln(s / (1-s))
        # z_c1 = ln( (0.2113) / (1-0.2113) ) approx -1.317
        # z_c2 = ln( (0.7887) / (1-0.7887) ) approx +1.317
        # We also add z=0, which is an inflection point
        criticals = [0.0, 1.3169579, -1.3169579]

    elif activation == 'tanh':
        # Critical points of sigma''(z) for tanh are where
        # tanh^2(z) = 1/3
        # z = arctanh( +- 1/sqrt(3) )
        # z_c1 = -0.65847894
        # z_c2 = +0.65847894
        # We also add z=0, which is an inflection point
        criticals = [0.0, 0.65847894, -0.65847894]

    elif activation == 'relu':
        # Second derivative is always 0
        return 0.0, 0.0
    else:
        raise NotImplementedError(f"Hessian interval bounds not implemented for {activation}")

    # Add any critical points that fall within the interval [a, b]
    for c in criticals:
        # Use a small tolerance for floating point
        if a - 1e-9 <= c <= b + 1e-9:
            candidates.append(c)

    # Compute sigma'' at all candidate points
    vals = sigma_second_derivative(model, np.array(candidates))
    return float(np.min(vals)), float(np.max(vals))


def compute_sigma_hess_interval(model, d_layer, z_flat, omega_vec):
    """
    Computes the second derivative bounds (gamma, eta) for each neuron
    in a single layer.
    """
    gamma_vec, eta_vec = np.empty(d_layer), np.empty(d_layer)
    for k in range(d_layer):
        gamma_vec[k], eta_vec[k] = get_analytical_lipschitz_constants_hess(
            model, z_flat[k], omega_vec[k]
        )
    return gamma_vec, eta_vec


def compute_sigma_hess_intervals_all_layers(model, z_l, omega_all):
    """
    Computes the second derivative bounds (gamma, eta) for all layers.
    """
    L = len(z_l)
    gamma_all, eta_all = [], []

    # --- (FIX) ---
    # Loop only over HIDDEN layers (from 0 to L-2)
    for l in range(L - 1):
        # --- (END FIX) ---
        z_flat = np.atleast_1d(np.squeeze(z_l[l]))
        omega_vec = omega_all[l]
        d_layer = z_flat.shape[0]
        gamma_vec, eta_vec = compute_sigma_hess_interval(
            model, d_layer, z_flat, omega_vec
        )
        gamma_all.append(gamma_vec)
        eta_all.append(eta_vec)

    # --- (FIX) ---
    # Manually add bounds for the final LINEAR layer (l = L-1)
    # sigma'' = 0, so bounds are [0, 0]
    d_last_layer = z_l[L - 1].shape[0]
    gamma_all.append(np.zeros(d_last_layer))
    eta_all.append(np.zeros(d_last_layer))
    # --- (END FIX) ---

    return gamma_all, eta_all