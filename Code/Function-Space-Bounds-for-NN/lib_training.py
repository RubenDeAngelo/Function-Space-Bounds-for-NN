"""
Training loop (Adam + MSE) and autograd-based derivative utilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from math import pi


def generate_sine_data(n_samples, noise_std=0.0):
    X = np.linspace(0, 2 * pi, n_samples).reshape(-1, 1).astype(np.float32)
    y = np.sin(X) + np.random.normal(0, noise_std, X.shape).astype(np.float32)
    return X, y


def train_model(model, X_train, y_train, epochs=8000, lr=0.001, weight_decay=1e-4):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {loss.item():.6f}")


def compute_first_derivative_autograd(model, x_test):
    """Computes the network's first derivative via autograd (scalar input, 1D loop)."""
    model.eval()
    first_derivatives = []
    x_test_flat = x_test.flatten()

    for x_val in x_test_flat:
        x_tensor = torch.tensor([[x_val]], dtype=torch.float32, requires_grad=True)
        y = model(x_tensor)
        first_grad = torch.autograd.grad(
            y, x_tensor, torch.ones_like(y), create_graph=False
        )[0]
        first_derivatives.append(first_grad.item())

    return np.array(first_derivatives)


def compute_second_derivative_autograd(model, x_test):
    """Computes the network's second derivative via autograd (scalar input, 1D loop)."""
    model.eval()
    second_derivatives = []
    x_test_flat = x_test.flatten()

    for x_val in x_test_flat:
        x_tensor = torch.tensor([[x_val]], dtype=torch.float32, requires_grad=True)
        y = model(x_tensor)

        first_grad = torch.autograd.grad(
            y, x_tensor, torch.ones_like(y), create_graph=True
        )[0]
        second_grad = torch.autograd.grad(
            first_grad, x_tensor, torch.ones_like(first_grad), create_graph=False
        )[0]
        second_derivatives.append(second_grad.item())

    return np.array(second_derivatives)
