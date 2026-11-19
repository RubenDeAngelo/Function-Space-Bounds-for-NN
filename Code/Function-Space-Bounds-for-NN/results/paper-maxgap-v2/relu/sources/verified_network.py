"""Interval forward automatic differentiation for affine/tanh networks.

Weights and biases are exact rationals (in particular exact binary checkpoint
values). The certificate concerns the real network with those coefficients,
not the rounded output of torch.forward. Hessians use one mixed entry per
multi-index in the Sobolev integrand.
"""
from dataclasses import dataclass
from verified_arithmetic import rational, tanh_ranges


@dataclass(frozen=True)
class Jet:
    value: object
    gradient: tuple
    hessian: tuple


class VerifiedNetwork:
    def __init__(self, arithmetic, weights, biases, activation="tanh"):
        if activation not in ("tanh", "affine", "relu"):
            raise ValueError("Verified path supports tanh or affine networks only")
        if len(weights) != len(biases) or not weights:
            raise ValueError("Nonempty matching weights and biases required")
        self.a = arithmetic
        self.weights = tuple(tuple(tuple(rational(x) for x in row) for row in w)
                             for w in weights)
        self.biases = tuple(tuple(rational(x) for x in b) for b in biases)
        if not self.weights[0] or not self.weights[0][0]:
            raise ValueError("Empty layer")
        self.dimension = len(self.weights[0][0])
        width = self.dimension
        for w, b in zip(self.weights, self.biases):
            if not w or len(w) != len(b) or any(len(row) != width for row in w):
                raise ValueError("Incompatible layer dimensions")
            width = len(w)
        self.activation = activation
        self._weights = tuple(tuple(tuple(self.a.interval(x) for x in row) for row in w)
                              for w in self.weights)
        self._biases = tuple(tuple(self.a.interval(x) for x in b) for b in self.biases)

    @classmethod
    def from_torch(cls, model, arithmetic):
        import torch
        # Accept precisely the existing FlexibleNet graph or an explicit
        # Sequential; do not silently skip unknown layers or custom forwards.
        from lib_nn_model import FlexibleNet
        if type(model) is FlexibleNet:
            if model.activation_name not in ("tanh", "relu"):
                raise ValueError("Only tanh FlexibleNet is supported")
            layers = list(model.net)
        elif type(model) is torch.nn.Sequential:
            layers = list(model)
        else:
            raise TypeError("Expected FlexibleNet or Sequential")
        if len(layers) % 2 != 1:
            raise ValueError("Expected alternating Linear/Tanh with affine output")
        for i, layer in enumerate(layers):
            expected = torch.nn.Linear if i % 2 == 0 else type(layers[1])
            if i % 2 and expected not in (torch.nn.Tanh, torch.nn.ReLU):
                raise ValueError("Unsupported activation")
            if type(layer) is not expected:
                raise ValueError("Unsupported graph")
        weights, biases = [], []
        for layer in layers[::2]:
            if layer.weight.dtype not in (torch.float32, torch.float64):
                raise ValueError("Only binary32/64 checkpoint parameters supported")
            weights.append(layer.weight.detach().cpu().tolist())
            biases.append(layer.bias.detach().cpu().tolist() if layer.bias is not None
                          else [0] * layer.out_features)
        return cls(arithmetic, weights, biases, "relu" if len(layers)>1 and type(layers[1]) is torch.nn.ReLU else "tanh")

    def evaluate(self, bounds, order=2):
        if order not in (0, 1, 2) or len(bounds) != self.dimension:
            raise ValueError("Invalid derivative order or input dimension")
        if self.activation == "relu" and order > 1:
            raise ValueError("ReLU H2 is not supported")
        d, a = self.dimension, self.a
        zero, one = a.interval(0), a.interval(1)
        pairs = tuple((i, j) for i in range(d) for j in range(i, d)) if order == 2 else ()
        jets = [Jet(a.interval(lo, hi),
                    tuple(one if i == j else zero for j in range(d)) if order else (),
                    tuple(zero for _ in pairs))
                for i, (lo, hi) in enumerate(bounds)]
        for layer_index, (w, b) in enumerate(zip(self._weights, self._biases)):
            outputs = []
            for row, bias in zip(w, b):
                value = a.sum(c * jet.value for c, jet in zip(row, jets)) + bias
                gradient = tuple(a.sum(c * jet.gradient[i] for c, jet in zip(row, jets))
                                 for i in range(d)) if order else ()
                hessian = tuple(a.sum(c * jet.hessian[k] for c, jet in zip(row, jets))
                                for k in range(len(pairs)))
                if layer_index < len(self._weights) - 1 and self.activation == "tanh":
                    if order == 0:
                        value = a.raw(a.down.tanh(value.lo), a.up.tanh(value.hi))
                    else:
                        v, first, second = tanh_ranges(value)
                        hessian = tuple(second * (gradient[i].abs_power(2) if i == j
                                                  else gradient[i] * gradient[j])
                                        + first * hessian[k] for k, (i, j) in enumerate(pairs))
                        gradient = tuple(first * gi for gi in gradient)
                        value = v
                if layer_index < len(self._weights) - 1 and self.activation == "relu":
                    # Lipschitz chain rule: weak derivative a.e.; ambiguous
                    # preactivations include both slopes. Identically zero
                    # preactivations have zero weak gradient.
                    slope = zero if value.hi <= 0 else (one if value.lo >= 0 else a.interval(0, 1))
                    gradient = tuple(slope * gi for gi in gradient)
                    value = a.raw(max(zero.lo, value.lo), max(zero.hi, value.hi))
                outputs.append(Jet(value, gradient, hessian))
            jets = outputs
        return tuple(jets)

    def sobolev_integrand(self, bounds, order=2, p=2):
        if not isinstance(p, int) or p < 1:
            raise ValueError("Only positive integer p supported")
        terms = []
        for jet in self.evaluate(bounds, order):
            terms.extend((jet.value,) + jet.gradient + jet.hessian)
        return self.a.sum(t.abs_power(p) for t in terms)

    def serialise(self):
        return {"activation": self.activation,
                "weights": [[[str(x) for x in row] for row in w] for w in self.weights],
                "biases": [[str(x) for x in b] for b in self.biases]}

    @classmethod
    def from_record(cls, record, arithmetic):
        return cls(arithmetic, record["weights"], record["biases"], record["activation"])
