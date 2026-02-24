import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional, Callable
from dataclasses import dataclass


# ==========================================
# 1. NEURAL NETWORK & INITIALIZATION
# ==========================================

class ReLUNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, width=10, depth=1, init_method="default", std=1.0):
        super().__init__()
        self.init_method = init_method
        self.depth = depth
        self.std = std

        layers = [nn.Linear(input_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, output_dim))

        self.layers = nn.Sequential(*layers)
        self._initialize_weights(init_method)

    def _initialize_weights(self, method: str):
        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                if method == "default":
                    pass
                elif method == "xavier_uniform":
                    init.xavier_uniform_(m.weight)
                elif method == "xavier_normal":
                    init.xavier_normal_(m.weight)
                elif method in ("he_uniform", "kaiming_uniform"):
                    init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                elif method in ("he_normal", "kaiming_normal"):
                    init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu")
                elif method == "orthogonal":
                    init.orthogonal_(m.weight)
                elif method == "zeros":
                    init.zeros_(m.weight)
                elif method == "ones":
                    init.ones_(m.weight)
                else:
                    raise ValueError(f"Unknown init_method={method!r}")

                if m.bias is not None:
                    init.normal_(m.bias, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def init_many_affine_pieces(net: nn.Module):
    first = True
    for m in net.modules():
        if isinstance(m, nn.Linear):
            # He init for ReLU
            init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu")

            if m.bias is not None:
                if first:
                    # Make first-layer hyperplanes likely to cut through [-0.5,0.5]^2:
                    t = 0.5 * m.weight.abs().sum(dim=1)
                    m.bias.data.copy_((2 * torch.rand_like(t) - 1) * t)
                    first = False
                else:
                    # keep later biases modest (or zero)
                    init.zeros_(m.bias)


def boxes_inside_activation_region(
        net: nn.Module,
        cr_batch: torch.Tensor,  # (n,d,2) with [:,:,0]=center, [:,:,1]=radius
        eps: float = 0.0,  # treat preact > eps as "active"
) -> torch.Tensor:
    """
    Returns: inside (n,) bool tensor. True if each box's vertices all have identical
    ReLU activation pattern at every ReLU layer.
    """
    if cr_batch.ndim != 3 or cr_batch.shape[-1] != 2:
        raise ValueError(f"cr_batch must have shape (n,d,2), got {tuple(cr_batch.shape)}")

    device = cr_batch.device
    dtype = cr_batch.dtype
    n, d, _ = cr_batch.shape

    # Build all 2^d vertex sign patterns in {-1,+1}^d
    m = 2 ** d
    idx = torch.arange(m, device=device)
    bits = ((idx[:, None] >> torch.arange(d, device=device)) & 1).to(dtype)
    signs = 2 * bits - 1

    c = cr_batch[:, :, 0]
    r = cr_batch[:, :, 1]

    # vertices: (n,m,d) with coords c +/- r
    vertices = c[:, None, :] + signs[None, :, :] * r[:, None, :]
    x = vertices.reshape(n * m, d)

    inside = torch.ones(n, dtype=torch.bool, device=device)

    net.eval()
    with torch.no_grad():
        modules = list(net.layers) if hasattr(net, "layers") else list(net.children())

        for layer in modules:
            if isinstance(layer, nn.Linear):
                x = layer(x)
            elif isinstance(layer, nn.ReLU):
                active = (x > eps)
                active = active.reshape(n, m, -1)

                same = (active.min(dim=1).values == active.max(dim=1).values).all(dim=1)
                inside &= same
                x = layer(x)

                if not inside.any():
                    break
            else:
                x = layer(x)

    return inside


# ==========================================
# 2. INTEGRATION & MATH UTILS
# ==========================================

class TensorGaussLegendreLp:
    """
    Tensor-product Gauss–Legendre quadrature on a batch of axis-aligned boxes in [0,1]^d.
    """

    def __init__(self, p: float):
        self.p = int(math.ceil(p))
        self.m = int(math.ceil((self.p + 1) / 2))

        t_np, w_np = np.polynomial.legendre.leggauss(self.m)
        self.t_1d = torch.tensor(t_np, dtype=torch.float32)
        self.w_1d = torch.tensor(w_np, dtype=torch.float32)
        self._cache = {}

    def _tensor_product_rule(self, d: int, device, dtype):
        if d in self._cache:
            nodes_ref, w_prod = self._cache[d]
            return nodes_ref.to(device=device, dtype=dtype), w_prod.to(device=device, dtype=dtype)

        t = self.t_1d.to(device=device, dtype=dtype)
        w = self.w_1d.to(device=device, dtype=dtype)

        grids = torch.meshgrid(*([t] * d), indexing="ij")
        nodes_ref = torch.stack([g.reshape(-1) for g in grids], dim=1)

        wgrids = torch.meshgrid(*([w] * d), indexing="ij")
        w_prod = torch.stack([g.reshape(-1) for g in wgrids], dim=0).prod(dim=0)

        self._cache[d] = (nodes_ref.detach().cpu(), w_prod.detach().cpu())
        return nodes_ref, w_prod

    @torch.no_grad()
    def integrate_boxes(self, net: nn.Module, cr_batch: torch.Tensor) -> torch.Tensor:
        if cr_batch.ndim != 3 or cr_batch.shape[-1] != 2:
            raise ValueError(f"cr_batch must have shape (n,d,2), got {tuple(cr_batch.shape)}")

        device, dtype = cr_batch.device, cr_batch.dtype
        n, d, _ = cr_batch.shape

        nodes_ref, w_prod = self._tensor_product_rule(d, device, dtype)
        q = nodes_ref.shape[0]

        c = cr_batch[:, :, 0]
        r = cr_batch[:, :, 1]

        X = c[:, None, :] + r[:, None, :] * nodes_ref[None, :, :]
        X = X.reshape(n * q, d)

        y = net(X)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        y = y.reshape(n, q)

        f = torch.abs(y).pow(self.p)
        jac = r.prod(dim=1)
        I = jac * (f * w_prod[None, :]).sum(dim=1)
        return I

    @torch.no_grad()
    def lp_norm(self, net: nn.Module, cr_batch: torch.Tensor) -> float:
        I = self.integrate_boxes(net, cr_batch).sum()
        return float(I.clamp_min(0).pow(1.0 / self.p).item())


@torch.no_grad()
def monte_carlo_lp_norm(
        net: nn.Module,
        d: int,
        p: float,
        n_samples: int = 200_000,
        batch_size: int = 50_000,
        device: str | torch.device | None = None,
) -> float:
    """
    Approximates ||net||_{L^p([0,1]^d)} via Monte Carlo
    """
    net.eval()
    dev = device or next(net.parameters()).device
    p = float(p)

    total = 0.0
    seen = 0

    while seen < n_samples:
        b = min(batch_size, n_samples - seen)
        x = torch.rand((b, d), device=dev)
        y = net(x)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        vals = torch.abs(y).pow(p)
        total += float(vals.sum().item())
        seen += b

    mean = total / n_samples
    return mean ** (1.0 / p)


def a_posteriori_bound(vals: dict[int, float], p: float) -> float:
    p = float(p)
    s = math.fsum((abs(v) ** p) for v in vals.values())
    if s == 0.0:
        return 0.0
    return math.exp(math.log(s) / p)


def a_priori_bound(theta: float, p: float, n: float, L: float) -> float:
    t = 2.0 ** (-p)  # Klassische Schreibweise, funktioniert in jeder Python-Version
    log_base = math.log1p(-theta * (1.0 - t))
    return math.exp((n / p) * log_base) * (L / 2.0)


# ==========================================
# 3. PARTITIONING & REFINEMENT
# ==========================================

class pDoerflerMarker:
    def __init__(self, p: float, theta: float):
        self.p = float(1)  # As originally implemented
        self.theta = float(theta)

    def __call__(self, eta: dict[int, float]) -> set[int]:
        total_sum = sum(v ** self.p for v in eta.values())
        target = self.theta * total_sum

        marked: set[int] = set()
        unmarked: set[int] = set(eta.keys())
        sum_marked = 0.0

        while unmarked and sum_marked < target:
            box_id = max(unmarked, key=eta.get)
            unmarked.remove(box_id)
            marked.add(box_id)
            sum_marked += eta[box_id] ** self.p

        return marked


class Partition:
    def __init__(self, d, theta=0.8, capacity=5000):
        self.leaves = set([])
        self.storage = torch.zeros((capacity, d, 2), dtype=torch.float32)
        self.id2slot = {}
        self.free_slots = list(range(capacity - 1, -1, -1))
        self.next_id = 0
        self.eta = {}

    def register_boxes(
            self,
            cr_batch: torch.Tensor,
            eta_batch: Optional[torch.Tensor] = None
    ) -> list[int]:
        n = int(cr_batch.shape[0])

        if n == 0:
            return []

        while len(self.free_slots) < n:
            self.increase_storage()

        start = self.next_id
        ids = list(range(start, start + n))
        self.next_id += n
        self.leaves.update(ids)

        slots = [self.free_slots.pop() for _ in range(n)]
        slots_t = torch.tensor(slots, dtype=torch.long, device=self.storage.device)
        self.storage.index_copy_(0, slots_t, cr_batch.to(self.storage.device, dtype=self.storage.dtype))

        self.id2slot.update(dict(zip(ids, slots)))

        if eta_batch is not None:
            eta_flat = eta_batch.reshape(-1)
            if eta_flat.numel() != n:
                raise ValueError(f"eta_batch must have {n} entries, got {eta_flat.numel()}")
            for box_id, e in zip(ids, eta_flat):
                self.eta[box_id] = float(e.item())

        return ids

    def deregister_box(self, box_id: int) -> None:
        self.leaves.discard(box_id)
        slot = self.id2slot.pop(box_id)
        self.free_slots.append(slot)
        self.eta.pop(box_id, None)

    def increase_storage(self, factor: int = 2) -> None:
        old_cap = self.storage.shape[0]
        new_cap = max(old_cap * factor, old_cap + 1)

        new_storage = torch.zeros(
            (new_cap, *self.storage.shape[1:]),
            dtype=self.storage.dtype,
            device=self.storage.device,
        )
        new_storage[:old_cap].copy_(self.storage)

        self.storage = new_storage
        self.free_slots.extend(range(new_cap - 1, old_cap - 1, -1))

    def refine_once(
            self,
            marker: Callable[[dict[int, float]], set[int]],
            refiner: str = "all",
            estimator: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:

        marked_ids = marker(self.eta)

        for box_id in marked_ids:
            b = Box(self, box_id)
            children = b.split(kind=refiner)

            self.deregister_box(box_id)

            etas = estimator(children)
            self.register_boxes(children, etas)


@dataclass(slots=True)
class Box:
    partition: "Partition"
    box_id: int

    @property
    def slot(self) -> int:
        return self.partition.id2slot[self.box_id]

    @property
    def cr(self) -> torch.Tensor:
        return self.partition.storage[self.slot]

    @property
    def center(self) -> torch.Tensor:
        return self.cr[:, 0]

    @property
    def radius(self) -> torch.Tensor:
        return self.cr[:, 1]

    @property
    def volume(self) -> float:
        r = self.radius
        return float((2 * r).prod())

    @property
    def diam(self) -> float:
        r = self.radius
        return float(2 * r.max())

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        c, r = self.center, self.radius
        return c - r, c + r

    def split(self, kind: str = "all") -> torch.Tensor:
        c, r = self.center, self.radius
        d = c.numel()
        r_new = r / 2

        if kind == "all":
            idx = torch.arange(2 ** d, device=c.device)
            bits = ((idx[:, None] >> torch.arange(d, device=c.device)) & 1).to(c.dtype)
            signs = 2 * bits - 1

            centers = c[None, :] + signs * r_new[None, :]
            radii = r_new[None, :].expand_as(centers)

            children = torch.stack([centers, radii], dim=2)
            return children

        elif kind == "max":
            k = int(torch.argmax(r).item())
            r_new = r.clone()
            r_new[k] = r[k] / 2

            centers = c.repeat(2, 1)
            centers[0, k] = c[k] - r_new[k]
            centers[1, k] = c[k] + r_new[k]

            radii = r_new.unsqueeze(0).repeat(2, 1)
            children = torch.stack([centers, radii], dim=2)
            return children


# ==========================================
# 4. ORCHESTRATOR
# ==========================================

class AdaptiveLpNorm:
    def __init__(self, d: int, neural_net: nn.Module, is_relu: bool = True):
        self.d = d
        self.neural_net = neural_net
        self.is_relu = bool(is_relu)

    def norm(
            self,
            p: float,
            m: int,
            theta: float = 0.5,
            refiner: str = "all",
            compute_quad_each_step: bool = True,
            eta_type: str = "fval",
    ) -> tuple[float, list[float], list[float], list[float]]:

        # Lokaler Import zur Vermeidung von Zirkelbezuegen
        from eta_computations import EtaEstimator, FValOracle, LipOracle

        marker = pDoerflerMarker(p, theta)

        lip_base_oracle = LipOracle(self.neural_net, input_dim=self.d)

        if eta_type == "lip":
            eta_oracle = lip_base_oracle
        elif eta_type == "fval":
            eta_oracle = FValOracle(self.neural_net, input_dim=self.d)
        else:
            raise ValueError(f"Unknown eta_type={eta_type!r}; expected 'lip' or 'fval'.")

        estimator_obj = EtaEstimator(p, self.neural_net, is_relu=self.is_relu, oracle=eta_oracle)

        def estimator(cr_batch: torch.Tensor) -> torch.Tensor:
            return estimator_obj(cr_batch, mode=eta_type)

        self.partition_adaptive = Partition(self.d)

        unit_cube = torch.full(
            (1, self.d, 2),
            0.5,
            dtype=self.partition_adaptive.storage.dtype,
            device=self.partition_adaptive.storage.device,
        )
        global_lip = float(lip_base_oracle(unit_cube).reshape(-1)[0].item())

        self.partition_adaptive.register_boxes(unit_cube)

        ids0 = sorted(self.partition_adaptive.leaves)
        slots0 = torch.tensor(
            [self.partition_adaptive.id2slot[i] for i in ids0],
            dtype=torch.long,
            device=self.partition_adaptive.storage.device,
        )
        cr0 = self.partition_adaptive.storage.index_select(0, slots0)
        eta0 = estimator(cr0)
        for box_id, e in zip(ids0, eta0.reshape(-1)):
            self.partition_adaptive.eta[box_id] = float(e.item())

        quad = TensorGaussLegendreLp(p=p)

        def current_cr_batch():
            ids = sorted(self.partition_adaptive.leaves)
            slots = torch.tensor(
                [self.partition_adaptive.id2slot[i] for i in ids],
                dtype=torch.long,
                device=self.partition_adaptive.storage.device,
            )
            return self.partition_adaptive.storage.index_select(0, slots)

        approx_hist: list[float] = []
        post_hist: list[float] = []
        prior_hist: list[float] = []
        size_hist: list[int] = []
        size_hist: list[int] = []

        def record(k: int):
            post_hist.append(a_posteriori_bound(self.partition_adaptive.eta, p))
            prior_hist.append(a_priori_bound(theta, p, k, global_lip))
            size_hist.append(len(self.partition_adaptive.leaves))  # <--- NEU
            if compute_quad_each_step:
                approx_hist.append(quad.lp_norm(self.neural_net, current_cr_batch()))
            else:
                approx_hist.append(float("nan"))

        record(k=0)

        for k in range(1, m + 1):
            self.partition_adaptive.refine_once(marker, refiner, estimator)
            record(k=k)

        approx_final = quad.lp_norm(self.neural_net, current_cr_batch())
        approx_hist[-1] = approx_final

        self.out_adap = {
            "approx": approx_final,
            "approx_hist": approx_hist,
            "error": post_hist[-1],
            "post_hist": post_hist,
            "prior_hist": prior_hist,
            "size_hist": size_hist,  # <--- NEU HIER
            "m": m,
            "p": p,
            "theta": theta,
            "refiner": refiner,
            "size": len(self.partition_adaptive.leaves),
            "eta_type": eta_type,
        }
        return approx_final, approx_hist, post_hist, prior_hist