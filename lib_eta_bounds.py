"""
Error estimators (eta) for the adaptive Lp refinement loop (AdaQuad).
Provides EtaEstimator, FValOracle (auto_LiRPA bounds), and LipOracle (global Lipschitz).
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable, Literal
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm

EtaMode = Literal["lip", "fval"]


class EtaEstimator:
    """
    Computes per-box eta estimates used by the Dörfler marker:
      - mode='lip':  eta_K = vol(K)^(1/p) * (L_K * diam(K) / 2)
      - mode='fval': eta_K = vol(K)^(1/p) * (ub_K - lb_K)
    Boxes entirely inside one ReLU activation region get eta=0 (affine, zero error).
    """

    def __init__(
            self,
            p: float,
            net: nn.Module,
            is_relu: bool,
            oracle: Callable[[torch.Tensor], torch.Tensor]
                    | Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
    ):
        self.p = float(p)
        self.net = net
        self.is_relu = bool(is_relu)
        self.oracle = oracle

    def _mask_inside_relu_region(self, cr_batch: torch.Tensor) -> torch.Tensor:
        if not self.is_relu:
            return torch.zeros(cr_batch.shape[0], dtype=torch.bool, device=cr_batch.device)

        from lib_adaquad_lp import boxes_inside_activation_region
        return boxes_inside_activation_region(self.net, cr_batch)

    def _volume_and_diam(self, cr_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r = cr_batch[:, :, 1]
        volume = (2.0 * r).prod(dim=1)
        diam = 2.0 * r.max(dim=1).values
        return volume, diam

    def __call__(self, cr_batch: torch.Tensor, mode: EtaMode = "lip") -> torch.Tensor:
        if cr_batch.ndim != 3 or cr_batch.shape[-1] != 2:
            raise ValueError(f"cr_batch must have shape (n, d, 2), got {tuple(cr_batch.shape)}")

        device = cr_batch.device
        dtype = cr_batch.dtype
        n, d, _ = cr_batch.shape

        volume, diam = self._volume_and_diam(cr_batch)
        inside = self._mask_inside_relu_region(cr_batch)
        mask = ~inside

        eta = torch.zeros(n, device=device, dtype=dtype)

        if not mask.any():
            return eta

        cr_out = cr_batch[mask]
        vol_out = volume[mask]
        diam_out = diam[mask]

        if mode == "lip":
            L = self.oracle(cr_out).to(device=device, dtype=dtype)
            eta_out = vol_out.pow(1.0 / self.p) * (0.5 * L * diam_out)

        elif mode == "fval":
            lb, ub = self.oracle(cr_out)
            lb = lb.to(device=device, dtype=dtype)
            ub = ub.to(device=device, dtype=dtype)
            diff = (ub - lb).clamp_min(0.0)
            eta_out = vol_out.pow(1.0 / self.p) * diff
        else:
            raise ValueError(f"Unknown eta mode {mode!r}; expected 'lip' or 'fval'.")

        eta[mask] = eta_out
        return eta


class FValOracle:
    """
    Computes guaranteed lower (lb) and upper (ub) bounds on network output values
    using auto_LiRPA's CROWN/IBP bounding.
    """

    def __init__(
            self,
            neural_net: torch.nn.Module,
            input_dim: int,
            *,
            device: torch.device | None = None,
            bound_opts: dict | None = None,
    ):
        self.neural_net = neural_net
        self.input_dim = input_dim
        self.bound_opts = {} if bound_opts is None else dict(bound_opts)

        if device is None:
            try:
                device = next(neural_net.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = device

        self.neural_net = self.neural_net.to(self.device)

        dummy_x = torch.zeros(1, self.input_dim, device=self.device)
        self.bounded_model = BoundedModule(
            self.neural_net,
            (dummy_x,),
            bound_opts=self.bound_opts,
            device=self.device,
        )

    def __call__(self, cr_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if cr_batch.ndim != 3 or cr_batch.shape[-1] != 2:
            raise ValueError(f"cr_batch must have shape (n, d, 2), got {tuple(cr_batch.shape)}")

        cr_batch = cr_batch.to(device=self.device, dtype=torch.float32)
        centers = cr_batch[..., 0]
        radii = cr_batch[..., 1]
        n = centers.shape[0]

        x = BoundedTensor(centers, PerturbationLpNorm(norm=np.inf, eps=radii))
        lb, ub = self.bounded_model.compute_bounds(x=(x,))

        # Flatten to (n,) regardless of auto_LiRPA output shape
        lb = lb.reshape(n, -1)[:, 0]
        ub = ub.reshape(n, -1)[:, 0]

        return lb, ub


class LipOracle:
    """
    Conservative global Lipschitz upper bound via weight-matrix product norm.
    Assigns the same constant to every box — no local refinement.
    """

    def __init__(
            self,
            neural_net: torch.nn.Module,
            input_dim: int,
            *,
            device: torch.device | None = None,
            bound_opts: dict | None = None,
    ):
        self.neural_net = neural_net
        self.input_dim = input_dim

        if device is None:
            try:
                device = next(neural_net.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = device

        self.global_lip_bound = self._compute_naive_global_lip()

    def _compute_naive_global_lip(self) -> float:
        """Computes ||W_L| · ... · |W_1||_1 as a safe global Lipschitz bound."""
        W_prod = None
        for m in self.neural_net.modules():
            if isinstance(m, nn.Linear):
                W_abs = torch.abs(m.weight).detach().to(self.device)
                if W_prod is None:
                    W_prod = W_abs
                else:
                    W_prod = W_abs @ W_prod

        if W_prod is None:
            return 1.0

        return float(W_prod.sum().item())

    def __call__(self, cr_batch: torch.Tensor) -> torch.Tensor:
        n = cr_batch.shape[0]
        return torch.full((n,), self.global_lip_bound, dtype=torch.float32, device=self.device)


def compute_local_lipschitz_boxes2(
        net: nn.Module,
        children: torch.Tensor,
        *,
        device=None,
        bound_opts=None,
) -> torch.Tensor:
    """
    Per-box local Lipschitz upper bounds computed with a shared BoundedModule.
    children: (n, d, 2) tensor of center-radius pairs.
    """
    if children.ndim != 3 or children.shape[-1] != 2:
        raise ValueError(f"children must have shape (n, d, 2), got {tuple(children.shape)}")

    if device is None:
        try:
            device = next(net.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    bound_opts = {} if bound_opts is None else bound_opts

    children = children.to(device=device, dtype=torch.float32)
    centers = children[..., 0]
    radii = children[..., 1]
    n, d = centers.shape

    net = net.to(device)

    dummy_x = torch.zeros(1, d, device=device)
    bounded_model = BoundedModule(
        net,
        (dummy_x,),
        bound_opts=bound_opts,
        device=device,
    )

    return _compute_local_lipschitz_boxes_with_bound_module(
        bounded_model,
        centers,
        radii,
        device=device,
    )


import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import logging
import warnings
from contextlib import contextmanager, redirect_stdout, redirect_stderr


@contextmanager
def silence_everything(disable_logging: bool = True, disable_warnings: bool = True):
    """Suppresses stdout, stderr, logging, and warnings (e.g. from auto_LiRPA)."""
    prev_disable_level = logging.root.manager.disable
    prev_warning_filters = warnings.filters[:]

    with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
        try:
            if disable_logging:
                logging.disable(logging.CRITICAL)
            if disable_warnings:
                warnings.simplefilter("ignore")
            yield
        finally:
            if disable_warnings:
                warnings.filters = prev_warning_filters
            if disable_logging:
                logging.disable(prev_disable_level)
