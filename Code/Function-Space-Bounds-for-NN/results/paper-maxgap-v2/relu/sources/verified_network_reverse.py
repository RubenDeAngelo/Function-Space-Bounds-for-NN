"""Interval reverse automatic differentiation for affine/tanh networks.

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
        if activation not in ("tanh", "affine"):
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
            if model.activation_name != "tanh":
                raise ValueError("Only tanh FlexibleNet is supported")
            layers = list(model.net)
        elif type(model) is torch.nn.Sequential:
            layers = list(model)
        else:
            raise TypeError("Expected FlexibleNet or Sequential")
        if len(layers) % 2 != 1:
            raise ValueError("Expected alternating Linear/Tanh with affine output")
        for i, layer in enumerate(layers):
            expected = torch.nn.Linear if i % 2 == 0 else torch.nn.Tanh
            if type(layer) is not expected:
                raise ValueError("Unsupported graph")
        weights, biases = [], []
        for layer in layers[::2]:
            if layer.weight.dtype not in (torch.float32, torch.float64):
                raise ValueError("Only binary32/64 checkpoint parameters supported")
            weights.append(layer.weight.detach().cpu().tolist())
            biases.append(layer.bias.detach().cpu().tolist() if layer.bias is not None
                          else [0] * layer.out_features)
        return cls(arithmetic, weights, biases)

    def evaluate(self, bounds, order=2):
        """Forward values and backward suffix derivatives (paper recurrences).

        Matrix products are evaluated left to right. Hessian symmetry permits
        storing one triangle; diagonal scaling uses interval multiplication,
        including D[j]*H[j,j]*D[j], exactly as in the printed recurrence.
        """
        if order not in (0, 1, 2) or len(bounds) != self.dimension:
            raise ValueError("Invalid derivative order or input dimension")
        a, d = self.a, self.dimension
        zero = a.interval(0)
        values = [a.interval(lo, hi) for lo, hi in bounds]
        derivatives = []
        for layer, (w, b) in enumerate(zip(self._weights, self._biases)):
            values = [a.sum(c*v for c, v in zip(row, values))+bias
                      for row, bias in zip(w, b)]
            if layer < len(self._weights)-1:
                if self.activation == "tanh":
                    if order:
                        ranges = [tanh_ranges(v) for v in values]
                        values = [r[0] for r in ranges]
                        derivatives.append(([r[1] for r in ranges],
                                            [r[2] for r in ranges]))
                    else:
                        values = [a.raw(a.down.tanh(v.lo), a.up.tanh(v.hi))
                                  for v in values]
                elif order:
                    derivatives.append(([a.interval(1)]*len(values),
                                        [zero]*len(values)))
        if not order:
            return tuple(Jet(v, (), ()) for v in values)
        jac = self._weights[-1]
        width = len(jac[0])
        hess = [[[zero for _ in range(width)] for _ in range(width)]
                for _ in jac] if order == 2 else None
        for layer in range(len(self._weights)-2, -1, -1):
            w = self._weights[layer]
            first, second = derivatives[layer]
            n, previous = len(w), len(w[0])
            if order == 2:
                next_hess = []
                for output, h in enumerate(hess):
                    # B = D H D + diag(sigma'' * suffix Jacobian).
                    b = [[zero]*n for _ in range(n)]
                    for j in range(n):
                        for k in range(j, n):
                            entry = first[j]*h[j][k]*first[k]
                            if j == k:
                                entry = entry + second[j]*jac[output][j]
                            b[j][k] = b[k][j] = entry
                    # H = (W^T B) W; no forward input jets enter this pass.
                    left = [[a.sum(w[j][i]*b[j][k] for j in range(n))
                             for k in range(n)] for i in range(previous)]
                    result = [[zero]*previous for _ in range(previous)]
                    for i in range(previous):
                        for k in range(i, previous):
                            entry = a.sum(left[i][j]*w[j][k] for j in range(n))
                            result[i][k] = result[k][i] = entry
                    next_hess.append(result)
                hess = next_hess
            jac = tuple(tuple(a.sum((row[j]*first[j])*w[j][i]
                                    for j in range(n)) for i in range(previous))
                        for row in jac)
        pairs = tuple((i,j) for i in range(d) for j in range(i,d))
        return tuple(Jet(v, tuple(jac[o]),
                         tuple(hess[o][i][j] for i,j in pairs) if order == 2 else ())
                     for o,v in enumerate(values))

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
