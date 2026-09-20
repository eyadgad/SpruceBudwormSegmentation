"""Binary segmentation metrics computed from 0/1 masks (numpy or torch).

Includes region metrics (Dice/IoU/precision/recall) and BOUNDARY-aware metrics
for fuzzy/gradient plume edges, where standard Dice over-penalizes a boundary
drawn a pixel or two too wide:
  * Boundary IoU        — Cheng et al., CVPR 2021 (arXiv:2103.16562)
  * Normalized Surface Dice (NSD) at tolerance tau — Nikolov et al., 2018/2021
  * HD95, ASSD          — 95th-percentile Hausdorff / avg symmetric surface dist.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from scipy import ndimage

# XAM Cartesian grid. One cell is 500 m on a side (confirmed in the radar
# product; do not hardcode this literal at call sites).
PIXEL_M = 500.0
PIXEL_KM = PIXEL_M / 1000.0
PIXEL_AREA_KM2 = PIXEL_KM ** 2
DEFAULT_NSD_TAUS_PX = (1, 2, 3, 4, 6, 8, 10, 14, 20)
DEFAULT_BF1_SIGMA_PX = 2.0


def px_to_km(px) -> float:
    return float(px) * PIXEL_KM


def km_to_px(km) -> float:
    return float(km) / PIXEL_KM


def cells_to_km2(n_cells) -> float:
    return float(n_cells) * PIXEL_AREA_KM2


def km2_to_cells(area_km2) -> float:
    return float(area_km2) / PIXEL_AREA_KM2


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


def _bf1_from_distances(d_pred: np.ndarray, d_true: np.ndarray, theta: float) -> float:
    """Hard-tolerance Csurka boundary F1 at distance θ (px)."""
    prec = float((d_pred <= theta).mean()) if d_pred.size else 0.0
    rec = float((d_true <= theta).mean()) if d_true.size else 0.0
    return float(2 * prec * rec / (prec + rec + 1e-8))


def _fuzzy_bf1(d_pred: np.ndarray, d_true: np.ndarray, sigma: float) -> float:
    """Centre-weighted BF1: mean exp(-d² / 2σ²) on each surface, then F1."""
    var = 2.0 * float(sigma) ** 2
    p_soft = float(np.mean(np.exp(-(d_pred ** 2) / var))) if d_pred.size else 0.0
    r_soft = float(np.mean(np.exp(-(d_true ** 2) / var))) if d_true.size else 0.0
    return float(2 * p_soft * r_soft / (p_soft + r_soft + 1e-8))


def surface_metrics(pred: np.ndarray, true: np.ndarray, tau: float = 2.0,
                    taus_px: Optional[Sequence[float]] = None,
                    sigma_px: Optional[float] = None,
                    pixel_m: float = PIXEL_M) -> Dict:
    """Normalized Surface Dice at tolerance `tau` (px), HD95, and ASSD.

    NSD = fraction of both surfaces lying within `tau` pixels of the other
    surface — tolerant of fuzzy-boundary jitter within tau, unlike Dice.

    Extra keys (additive; existing nsd/hd95/assd stay the same when both
    surfaces exist):
      nsd_curve, hd95_km, assd_km, bf1, bf1_fuzzy
    Empty prediction on a non-empty true mask: nsd = 0 (no longer NaN, so
    nanmean cannot drop total misses). hd95/assd stay NaN.
    """
    out: Dict = {"nsd": float("nan"), "hd95": float("nan"), "assd": float("nan")}
    scale_km = float(pixel_m) / 1000.0
    taus = tuple(DEFAULT_NSD_TAUS_PX if taus_px is None else taus_px)
    sigma = float(DEFAULT_BF1_SIGMA_PX if sigma_px is None else sigma_px)
    out["nsd_curve"] = {
        "taus_px": [float(t) for t in taus],
        "taus_km": [float(t) * scale_km for t in taus],
        "nsd": [float("nan")] * len(taus),
    }
    out["bf1"] = float("nan")
    out["bf1_fuzzy"] = float("nan")
    out["hd95_km"] = float("nan")
    out["assd_km"] = float("nan")

    p_any = bool(np.asarray(pred).any())
    t_any = bool(np.asarray(true).any())
    res = _surface_distances(pred, true)
    if res is None:
        if not p_any and not t_any:
            out.update(nsd=1.0, hd95=0.0, assd=0.0, hd95_km=0.0, assd_km=0.0,
                       bf1=1.0, bf1_fuzzy=1.0)
            out["nsd_curve"]["nsd"] = [1.0] * len(taus)
        else:
            # One surface empty: a total miss (or unmatched hallucination).
            out["nsd"] = 0.0
            out["bf1"] = 0.0
            out["bf1_fuzzy"] = 0.0
            out["nsd_curve"]["nsd"] = [0.0] * len(taus)
        return out
    d_pred, d_true = res
    n = d_pred.size + d_true.size
    out["nsd"] = float(((d_pred <= tau).sum() + (d_true <= tau).sum()) / (n + 1e-8))
    alld = np.concatenate([d_pred, d_true])
    out["hd95"] = float(np.percentile(alld, 95))
    out["assd"] = float(alld.mean())
    out["hd95_km"] = float(out["hd95"] * scale_km)
    out["assd_km"] = float(out["assd"] * scale_km)
    out["nsd_curve"]["nsd"] = [
        float(((d_pred <= t).sum() + (d_true <= t).sum()) / (n + 1e-8)) for t in taus
    ]
    out["bf1"] = _bf1_from_distances(d_pred, d_true, tau)
    out["bf1_fuzzy"] = _fuzzy_bf1(d_pred, d_true, sigma)
    return out
