import numpy as np
import torch
import torch.nn as nn
from math import pi


# --- EXISTING FUNCTIONS (from Jacobian code) ---

def generate_sine_data(n_samples, noise_std=0.0):
    """
    Generates sample data from a sine wave.
    """
    X = np.linspace(0, 2 * pi, n_samples).reshape(-1, 1).astype(np.float32)
    y = np.sin(X) + np.random.normal(0, noise_std, X.shape).astype(np.float32)
    return X, y


def train_model(model, X_train, y_train, epochs=2000, lr=1e-2):
    """
    Trains a flexible_model.FlexibleNet model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print("Training model...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % (epochs // 4) == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    print(f"Training complete. Final loss: {loss.item()}")
    return model


def compute_first_derivative_autograd(model, x_test):
    """
    Computes the network's first derivative using PyTorch autograd.
    """
    model.eval()
    first_derivatives = []
    x_test_flat = x_test.flatten()

    for x_val in x_test_flat:
        x_tensor = torch.tensor([[x_val]], dtype=torch.float32, requires_grad=True)
        y = model(x_tensor)

        grad_outputs = torch.ones_like(y)
        first_grad = torch.autograd.grad(
            y, x_tensor, grad_outputs=grad_outputs, create_graph=False
        )[0]

        first_derivatives.append(first_grad.item())

    return np.array(first_derivatives)


# --- NEW HESSIAN-SPECIFIC FUNCTION ---

def compute_second_derivative_autograd(model, x_test):
    """
    Computes the network's second derivative using PyTorch autograd.
    """
    model.eval()
    second_derivatives = []
    x_test_flat = x_test.flatten()

    for x_val in x_test_flat:
        x_tensor = torch.tensor([[x_val]], dtype=torch.float32, requires_grad=True)
        y = model(x_tensor)

        # Create graph for first gradient
        first_grad = torch.autograd.grad(
            y, x_tensor, torch.ones_like(y), create_graph=True
        )[0]

        # Compute second gradient
        second_grad = torch.autograd.grad(
            first_grad, x_tensor, torch.ones_like(first_grad), create_graph=False
        )[0]

        second_derivatives.append(second_grad.item())

    return np.array(second_derivatives)
