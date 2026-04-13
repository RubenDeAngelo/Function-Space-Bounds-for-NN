"""
Forward pass (storing pre/post-activations) and backward pass (Jacobian matrices D_phi).
"""

import numpy as np
from lib_nn_model import extract_numpy_params
from lib_activation_enclosures import sigma, sigma_derivative


def forward_pass(model, x):
    """
    Forward pass storing post-activations x_l and pre-activations z_l per layer.
    The final entry of x_l is set to z_l[-1] (identity output activation).
    """
    weights, biases = extract_numpy_params(model)
    L = len(weights)

    x_l = [x]
    z = (weights[0] @ x_l[0]) + biases[0].reshape(-1, 1)
    z_l = [z]

    for l in range(L - 1):
        x_next = sigma(model, z_l[l])
        x_l.append(x_next)
        z = weights[l + 1] @ x_l[l + 1] + biases[l + 1].reshape(-1, 1)
        z_l.append(z)

    x_l.append(z_l[-1])
    return x_l, z_l


def backward_pass(model, z_l):
    """
    Backward pass computing D_phi[l]: the partial Jacobian from layer l to the output.
    D_phi[0] equals the full network Jacobian J.
    """
    weights, _ = extract_numpy_params(model)
    L = len(weights)
    d_out = weights[-1].shape[0]

    D_phi = [None] * (L + 1)
    D_phi[L] = np.eye(d_out)

    for l in range(L - 1, -1, -1):
        W = weights[l]
        if l > 0:
            sigma_prime = sigma_derivative(model, z_l[l - 1])
            if sigma_prime.ndim > 1:
                sigma_prime = sigma_prime.flatten()
            diag_sigma = np.diag(sigma_prime)
            D_phi[l] = diag_sigma @ (W.T @ D_phi[l + 1])
        else:
            D_phi[0] = W.T @ D_phi[1]

    return D_phi


def compute_jacobian(model, x, z_l=None, J_all=None):
    """Returns the network Jacobian (d_out × d_in) at point x."""
    x = x.reshape(-1, 1)
    if z_l is None or J_all is None:
        _, z_l = forward_pass(model, x)
        J_all = backward_pass(model, z_l)

    return J_all[0].T
