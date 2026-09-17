"""Binary segmentation metrics computed from 0/1 masks (numpy or torch).

Includes region metrics (Dice/IoU/precision/recall) and BOUNDARY-aware metrics
for fuzzy/gradient plume edges, where standard Dice over-penalizes a boundary
drawn a pixel or two too wide:
  * Boundary IoU        — Cheng et al., CVPR 2021 (arXiv:2103.16562)
  * Normalized Surface Dice (NSD) at tolerance tau — Nikolov et al., 2018/2021
  * HD95, ASSD          — 95th-percentile Hausdorff / avg symmetric surface dist.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import ndimage


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    p = np.asarray(pred).reshape(-1).astype(np.float64)
    t = np.asarray(true).reshape(-1).astype(np.float64)
    tp = float((p * t).sum())
    fp = float((p * (1 - t)).sum())
    fn = float(((1 - p) * t).sum())
    tn = float(((1 - p) * (1 - t)).sum())
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {"dice": dice, "iou": iou, "precision": precision,
            "recall": recall, "f1": f1, "accuracy": accuracy}


def _boundary_region(mask: np.ndarray, width: int) -> np.ndarray:
    """Pixels within `width` of the mask boundary: mask AND NOT erode(mask)."""
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m)
    eroded = ndimage.binary_erosion(m, iterations=max(1, width), border_value=1)
    return m & ~eroded


def boundary_iou(pred: np.ndarray, true: np.ndarray, dilation_ratio: float = 0.02) -> float:
    """Boundary IoU (Cheng et al., 2021): IoU restricted to the boundary bands.

    `dilation_ratio` sets band width = round(ratio * image diagonal).
    """
    p = np.asarray(pred).astype(bool); t = np.asarray(true).astype(bool)
    if not p.any() and not t.any():
        return 1.0
    diag = float(np.sqrt(p.shape[0] ** 2 + p.shape[1] ** 2))
    w = max(1, int(round(dilation_ratio * diag)))
    pb, tb = _boundary_region(p, w), _boundary_region(t, w)
    inter = float((pb & tb).sum()); union = float((pb | tb).sum())
    return inter / (union + 1e-8)


def _surface_distances(pred: np.ndarray, true: np.ndarray):
    """Return (d_pred->true surface, d_true->pred surface) for boundary pixels."""
    p = np.asarray(pred).astype(bool); t = np.asarray(true).astype(bool)
    sp = p & ~ndimage.binary_erosion(p, border_value=1) if p.any() else p
    st = t & ~ndimage.binary_erosion(t, border_value=1) if t.any() else t
    if not sp.any() or not st.any():
        return None
    # distance from every pixel to the nearest TRUE-surface / PRED-surface pixel
    dt_from_true = ndimage.distance_transform_edt(~st)
    dt_from_pred = ndimage.distance_transform_edt(~sp)
    return dt_from_true[sp], dt_from_pred[st]  # pred-surface distances, true-surface distances


def surface_metrics(pred: np.ndarray, true: np.ndarray, tau: float = 2.0) -> Dict[str, float]:
    """Normalized Surface Dice at tolerance `tau` (px), HD95, and ASSD.

    NSD = fraction of both surfaces lying within `tau` pixels of the other
    surface — tolerant of fuzzy-boundary jitter within tau, unlike Dice.
    """
    out = {"nsd": float("nan"), "hd95": float("nan"), "assd": float("nan")}
    res = _surface_distances(pred, true)
    if res is None:
        # both empty -> perfect; one empty -> worst
        both_empty = (not np.asarray(pred).any()) and (not np.asarray(true).any())
        if both_empty:
            out.update(nsd=1.0, hd95=0.0, assd=0.0)
        return out
    d_pred, d_true = res  # d_pred: pred-surface -> true; d_true: true-surface -> pred
    n = d_pred.size + d_true.size
    out["nsd"] = float(((d_pred <= tau).sum() + (d_true <= tau).sum()) / (n + 1e-8))
    alld = np.concatenate([d_pred, d_true])
    out["hd95"] = float(np.percentile(alld, 95))
    out["assd"] = float(alld.mean())
    return out
