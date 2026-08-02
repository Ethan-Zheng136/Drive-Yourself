"""synthetic.py -- tiny synthetic bimodal trajectory dataset for FMHead smoke tests.

A single "scene" has a fixed content condition ``c0``. Its target trajectory
distribution is BIMODAL: a smooth "go-left" arc (mode A) and a "go-right" arc
(mode B), each a curved 8-waypoint path with small Gaussian jitter.

Two dataset flavours:
  * mode="content"  -> mode chosen at random (style uninformative). The head must
                       represent BOTH modes from c alone  => multimodality test.
  * mode="style"    -> mode determined by the sign of the style condition s
                       (s=+ -> left, s=- -> right)          => style-control test.

Everything is standardized (z-scored) into a well-conditioned space for flow
matching; helpers convert back to metres for reporting/clustering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


HORIZON = 8
TRAJ_DIM = 2
STEP_X = 2.0        # metres of forward progress per 0.5 s waypoint
LATERAL = 4.0       # metres of lateral offset at the final waypoint
JITTER = 0.10       # metres of Gaussian jitter added to each waypoint


def canonical_trajectories(device: torch.device) -> torch.Tensor:
    """Return the two noise-free canonical trajectories, shape (2, T, 2) in metres.

    index 0 = go-LEFT (y>0), index 1 = go-RIGHT (y<0).
    """
    t = torch.arange(1, HORIZON + 1, device=device, dtype=torch.float32)
    x = STEP_X * t
    frac = (t / HORIZON) ** 2
    left = torch.stack([x, +LATERAL * frac], dim=-1)
    right = torch.stack([x, -LATERAL * frac], dim=-1)
    return torch.stack([left, right], dim=0)  # (2, T, 2)


@dataclass
class Normalizer:
    """Per-dimension z-score standardizer computed from the canonical bank."""

    mean: torch.Tensor  # (T, 2)
    std: torch.Tensor   # (T, 2)

    @classmethod
    def from_bank(cls, bank: torch.Tensor) -> "Normalizer":
        # bank: (M, T, 2). Use a global (broadcastable) scale to keep it simple
        # and numerically well conditioned.
        mean = bank.mean(dim=0)
        std = bank.std(dim=0).clamp_min(0.5)  # floor avoids blow-up where modes agree
        return cls(mean=mean, std=std)

    def norm(self, traj: torch.Tensor) -> torch.Tensor:
        return (traj - self.mean) / self.std

    def denorm(self, traj: torch.Tensor) -> torch.Tensor:
        return traj * self.std + self.mean


class SyntheticScene:
    """Fixed-scene bimodal trajectory sampler with configurable style semantics."""

    def __init__(self, d_cond: int, d_style: int, device: torch.device, seed: int = 0):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.device = device
        self.d_cond = d_cond
        self.d_style = d_style
        # fixed content vector for the single scene
        self.c0 = torch.randn(d_cond, generator=g).to(device)
        # fixed style direction; s_plus = +base, s_minus = -base
        base = torch.randn(d_style, generator=g)
        self.style_base = (base / base.norm() * (d_style ** 0.5)).to(device)

        self.canon = canonical_trajectories(device)          # (2, T, 2) metres
        self.normalizer = Normalizer.from_bank(self.canon)
        self.canon_norm = self.normalizer.norm(self.canon)   # (2, T, 2) normalized

    # -- condition helpers --------------------------------------------------
    def content(self, batch: int) -> torch.Tensor:
        return self.c0.unsqueeze(0).expand(batch, -1)

    def style_from_sign(self, sign: torch.Tensor) -> torch.Tensor:
        """sign: (B,) in {+1,-1} -> style vectors (B, d_style)."""
        return sign.view(-1, 1) * self.style_base.unsqueeze(0)

    # -- batch sampling -----------------------------------------------------
    def sample_batch(self, batch: int, mode: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (tau_norm, c, s, mode_idx).

        mode="content": mode_idx ~ Bernoulli(0.5), style random sign (uninformative).
        mode="style":   style sign ~ Bernoulli(0.5); mode_idx = 0 (left) if +, 1 (right) if -.
        """
        dev = self.device
        if mode == "content":
            mode_idx = torch.randint(0, 2, (batch,), device=dev)
            sign = torch.where(torch.rand(batch, device=dev) < 0.5, 1.0, -1.0)
        elif mode == "style":
            sign = torch.where(torch.rand(batch, device=dev) < 0.5, 1.0, -1.0)
            mode_idx = (sign < 0).long()  # +1 -> left(0), -1 -> right(1)
        else:
            raise ValueError(mode)

        tau = self.canon_norm[mode_idx].clone()  # (B, T, 2) normalized
        # add jitter in metric space then renormalize
        tau_metric = self.normalizer.denorm(tau)
        tau_metric = tau_metric + JITTER * torch.randn_like(tau_metric)
        tau = self.normalizer.norm(tau_metric)

        c = self.content(batch)
        s = self.style_from_sign(sign)
        return tau, c, s, mode_idx
