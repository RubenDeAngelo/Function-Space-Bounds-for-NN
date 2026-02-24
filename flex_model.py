from math import pi
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class FlexibleNet(nn.Module):
    def __init__(self, layer_sizes, activation='relu'):
        super().__init__()

        # Store activation name for derivative computation
        self.activation_name = activation if isinstance(activation, str) else 'custom'

        # Get activation function
        self.activation_fn = self._get_activation_function(activation)

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(self.activation_fn)

        # Final layer without activation
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))

        self.net = nn.Sequential(*layers)

    def _get_activation_function(self, activation):
        if isinstance(activation, nn.Module):
            return activation

        activation_dict = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }

        if isinstance(activation, str):
            activation = activation.lower()
            if activation in activation_dict:
                return activation_dict[activation]
            else:
                available_activations = list(activation_dict.keys())
                raise ValueError(f"Unsupported activation function: {activation}. "
                                 f"Available options: {available_activations}")

        raise TypeError("Activation must be a string or a PyTorch nn.Module")

    def forward(self, x):
        return self.net(x)

    def get_activation_derivative(self, x):

        if self.activation_name == 'relu':
            return self.relu_derivative(x)
        elif self.activation_name == 'tanh':
            return self.tanh_derivative(x)
        elif self.activation_name == 'sigmoid':
            return self.sigmoid_derivative(x)
        else:
            raise NotImplementedError(f"Derivative not implemented for activation: {self.activation_name}")

    def relu_derivative(self, x):
        return (x > 0).astype(x.dtype)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z)**2

    def sigmoid_derivative(self, z):
        s = 1 / (1 + np.exp(-z))
        return s * (1 - s)



def extract_numpy_params(model):
    weights = []
    biases = []
    for layer in model.net:
        if isinstance(layer, nn.Linear):
            weights.append(layer.weight.detach().numpy())
            biases.append(layer.bias.detach().numpy())
    return weights, biases


def generate_sine_data(n_samples, noise_std=0.1):
    X = np.linspace(0, 2 * pi, n_samples).reshape(-1, 1)
    y = np.sin(X) + np.random.normal(0, noise_std, X.shape)
    return X.astype(np.float32), y.astype(np.float32)


def train_model(model, X_train, y_train, epochs=1000, lr=1e-2):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    return model, loss.item()


