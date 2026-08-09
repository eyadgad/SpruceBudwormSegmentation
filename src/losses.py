"""Segmentation losses for binary, class-imbalanced targets.

All operate on raw logits (sigmoid applied internally). Factory ``create_loss``
selects via config. Focal / Tversky / Focal-Tversky target the severe imbalance
identified in Phase 1/2.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits).reshape(-1)
        t = target.reshape(-1)
        inter = (p * t).sum()
        return 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, pos_weight=None):
        super().__init__()
        self.dw, self.bw = dice_weight, bce_weight
        self.dice = DiceLoss()
        pw = torch.tensor([float(pos_weight)]) if pos_weight else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits, target):
        return self.dw * self.dice(logits, target) + self.bw * self.bce(logits, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, target):
        p = torch.sigmoid(logits).reshape(-1)
        t = target.reshape(-1)
        tp = (p * t).sum()
        fp = (p * (1 - t)).sum()
        fn = ((1 - p) * t).sum()
        return 1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)


class FocalTverskyLoss(nn.Module):
    """Abraham & Khan, 2019 (arXiv:1810.07842): (1 - Tversky)**(1/gamma)."""

    def __init__(self, alpha=0.3, beta=0.7, gamma=1.333, smooth=1.0):
        super().__init__()
        self.tv = TverskyLoss(alpha, beta, smooth)
        self.gamma = gamma

    def forward(self, logits, target):
        # Clamp the base away from 0 before the fractional power. The derivative
        # of x**(1/gamma) blows up as x -> 0 (1/gamma < 1); clamping bounds the
        # gradient and stops NaN/Inf gradients on near-perfect batches.
        return self.tv(logits, target).clamp(min=1e-6) ** (1.0 / self.gamma)


class BoundaryLoss(nn.Module):
    """Boundary loss (Kervadec et al., 2019, arXiv:1812.07032): integral of the
    prediction over the signed distance transform of the ground truth. Negative
    inside / positive outside the target, so it directly penalises the predicted
    contour drifting from the true contour — well suited to fuzzy plume edges and
    class imbalance. Use as a regulariser ADDED to a region loss (see below)."""

    @staticmethod
    def _sdt(target: torch.Tensor) -> torch.Tensor:
        t = (target.detach().cpu().numpy() > 0.5)
        sdt = np.zeros(t.shape, dtype=np.float32)
        for b in range(t.shape[0]):
            m = t[b, 0]
            if m.any() and (~m).any():
                sdt[b, 0] = ndimage.distance_transform_edt(~m) - ndimage.distance_transform_edt(m)
            elif m.all():
                sdt[b, 0] = -ndimage.distance_transform_edt(m)
            # all-background -> zeros (no contour to penalise)
        return torch.from_numpy(sdt).to(target.device)

    def forward(self, logits, target):
        sdt = self._sdt(target)
        return (torch.sigmoid(logits) * sdt).mean()


class RegionBoundaryLoss(nn.Module):
    """region_loss + boundary_weight * boundary_loss (Kervadec-style combo).

    A fixed weight is used; the paper schedules it upward over training, which
    can be added later if needed."""

    def __init__(self, region: nn.Module, boundary_weight: float = 0.5):
        super().__init__()
        self.region = region
        self.boundary = BoundaryLoss()
        self.w = boundary_weight

    def forward(self, logits, target):
        return self.region(logits, target) + self.w * self.boundary(logits, target)


def create_loss(cfg: Dict) -> nn.Module:
    """Build the loss from ``cfg['loss']``. Params use generic keys so a single
    naming scheme works across losses: ``alpha``, ``beta``, ``gamma``,
    ``pos_weight``, ``dice_weight``, ``bce_weight`` (also accepts the older
    ``tversky_alpha``/``tversky_beta`` aliases)."""
    c = cfg["loss"]
    name = c["name"]

    def g(*keys, default):
        for k in keys:
            if k in c:
                return c[k]
        return default

    if name == "dice":
        return DiceLoss()
    if name == "dice_bce":
        return DiceBCELoss(dice_weight=g("dice_weight", default=0.5),
                           bce_weight=g("bce_weight", default=0.5),
                           pos_weight=g("pos_weight", default=10.0))
    if name == "focal":
        return FocalLoss(alpha=g("alpha", "focal_alpha", default=0.25),
                         gamma=g("gamma", "focal_gamma", default=2.0))
    if name == "tversky":
        return TverskyLoss(alpha=g("alpha", "tversky_alpha", default=0.3),
                           beta=g("beta", "tversky_beta", default=0.7))
    if name == "focal_tversky":
        return FocalTverskyLoss(alpha=g("alpha", "tversky_alpha", default=0.3),
                                beta=g("beta", "tversky_beta", default=0.7),
                                gamma=g("gamma", "focal_tversky_gamma", default=1.333))
    if name == "bce":
        pw = torch.tensor([float(g("pos_weight", default=10.0))])
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if name == "dice_boundary":
        return RegionBoundaryLoss(DiceBCELoss(pos_weight=g("pos_weight", default=10.0)),
                                  boundary_weight=g("boundary_weight", default=0.5))
    raise ValueError(f"Unknown loss '{name}'")
