"""Training epoch, cheap patch validation, and realistic full-scene evaluation.

- Training/validation during a run happen on patches (cheap).
- Final performance is judged with full-scene sliding-window inference (+optional
  TTA) at a val-calibrated threshold, on held-out scenes — never patch metrics
  alone (Phase 3 requirement).
- Single GPU: AMP autocast + GradScaler, gradient accumulation for large models.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import channels, dataset
from . import metrics as metrics_mod
from .metrics import compute_metrics
from .models import seg_logits

IMG_SIZE = channels.IMG_SIZE


# ----------------------------------------------------------------------------
# Scheduler: linear warmup -> cosine decay (state-dict-able => resumable)
# ----------------------------------------------------------------------------

def build_scheduler(optimizer, cfg):
    epochs = int(cfg["train"]["epochs"])
    warmup = int(cfg["train"].get("warmup_epochs", 0))

    def lr_lambda(epoch):
        if warmup > 0 and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ----------------------------------------------------------------------------
# Train / validate on patches
# ----------------------------------------------------------------------------

def _reset_nonfinite_bn(model: torch.nn.Module) -> None:
    """Reset BatchNorm running buffers that became NaN/Inf after a bad forward.

    The forward pass updates BN running_mean/running_var *before* we can check
    the loss, so when a non-finite activation occurs the buffers are already
    corrupted.  Resetting them to (mean=0, var=1) lets training recover: BN
    re-accumulates valid statistics over subsequent batches.  Only modules whose
    buffers are actually non-finite are touched."""
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                          torch.nn.BatchNorm3d)):
            corrupt = (
                (m.running_mean is not None and not torch.isfinite(m.running_mean).all())
                or (m.running_var is not None and not torch.isfinite(m.running_var).all())
            )
            if corrupt:
                m.reset_running_stats()


def unpack_batch(batch):
    """Split a loader batch into ``(x, y, extra)``.

    Plain runs yield ``(x, y)``. The multi-task and temporal variants add a third
    dict element carrying ``cls`` (patch presence label) and/or ``pad_mask``
    (which temporal slots are replicated padding), so every consumer can stay
    agnostic about which extras a given experiment enabled.
    """
    if len(batch) == 2:
        return batch[0], batch[1], {}
    return batch[0], batch[1], batch[2]


def forward(model, x, pad_mask=None):
    """Call a model, passing ``pad_mask`` only to the temporal architectures."""
    return model(x, pad_mask) if pad_mask is not None else model(x)


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, accum_steps=1,
                    verbose=False, use_amp=False, amp_dtype=torch.float16, logger=None):
    """One training epoch. AMP dtype is configurable: bfloat16 (recommended,
    default) has fp32's exponent range and does not overflow, avoiding the fp16
    NaN that permanently corrupts BatchNorm buffers in transformer decoders.
    A non-finite-loss guard skips any bad micro-batch so a stray NaN cannot
    propagate into an optimizer step.  If a forward pass corrupts BN running
    buffers, _reset_nonfinite_bn() clears them so training can recover."""
    model.train()
    total, n = 0.0, 0
    n_batches = len(loader)
    n_skipped = 0        # micro-batches dropped for a non-finite loss
    n_bad_grad = 0       # optimizer steps skipped for non-finite gradients
    params = [p for g in optimizer.param_groups for p in g["params"]]
    multitask = bool(getattr(criterion, "is_multitask", False))
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pad_mask = extra.get("pad_mask")
        if pad_mask is not None:
            pad_mask = pad_mask.to(device, non_blocking=True)
        cls_target = extra.get("cls")
        if cls_target is not None:
            cls_target = cls_target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = forward(model, x, pad_mask)
            loss = (criterion(out, y, cls_target) if multitask
                    else criterion(seg_logits(out), y)) / accum_steps
        # Guard: a non-finite loss (fp16 overflow, bad batch) must never reach the
        # optimizer or accumulate — drop this micro-batch instead.
        # ALSO reset any BN running buffers the forward may have corrupted: BN
        # updates its buffers during the forward (before we can check the loss),
        # so without this reset a single bad batch permanently kills all subsequent
        # forwards (the "BN cascade" seen with sparse high-elevation channels).
        if not torch.isfinite(loss):
            n_skipped += 1
            optimizer.zero_grad(set_to_none=True)
            _reset_nonfinite_bn(model)
            continue
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (i + 1) % accum_steps == 0:
            if scaler is not None:
                # fp16 path: GradScaler skips the step itself on inf/nan grads.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                # bf16/fp32 path: no GradScaler, so gate the step on finite
                # gradients. A finite loss can still yield NaN/Inf grads (e.g. the
                # Focal-Tversky power derivative), and stepping on those writes NaN
                # into the weights and kills the model for good. clip_grad_norm_
                # returns the total grad norm; if it is non-finite, skip the step.
                gnorm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                if torch.isfinite(gnorm):
                    optimizer.step()
                else:
                    n_bad_grad += 1
                    _reset_nonfinite_bn(model)
            optimizer.zero_grad(set_to_none=True)
        total += loss.item() * accum_steps * x.size(0)
        n += x.size(0)
    if n_skipped or n_bad_grad:
        msg = (f"epoch guard: {n_skipped}/{n_batches} non-finite-loss batches dropped, "
               f"{n_bad_grad} non-finite-gradient steps skipped")
        (logger.warning(msg) if logger is not None else print(f"    [warn] {msg}", flush=True))
    return total / max(n, 1)


@torch.no_grad()
def validate_patches(model, loader, device, threshold=0.5) -> Dict[str, float]:
    """Cheap patch-grid validation: accumulate confusion counts at a threshold."""
    model.eval()
    tp = fp = fn = tn = 0.0
    for batch in loader:
        x, y, extra = unpack_batch(batch)
        x = x.to(device, non_blocking=True)
        pad_mask = extra.get("pad_mask")
        if pad_mask is not None:
            pad_mask = pad_mask.to(device, non_blocking=True)
        prob = torch.sigmoid(seg_logits(forward(model, x, pad_mask))).cpu().numpy()
        pred = (prob > threshold).astype(np.float64)
        t = y.numpy().astype(np.float64)
        tp += float((pred * t).sum())
        fp += float((pred * (1 - t)).sum())
        fn += float(((1 - pred) * t).sum())
        tn += float(((1 - pred) * (1 - t)).sum())
    eps = 1e-8
    prec = tp / (tp + fp + eps); rec = tp / (tp + fn + eps)
    return {"dice": 2 * tp / (2 * tp + fp + fn + eps),
            "iou": tp / (tp + fp + fn + eps),
            "precision": prec, "recall": rec,
            "accuracy": (tp + tn) / (tp + tn + fp + fn + eps)}


# ----------------------------------------------------------------------------
# Full-scene sliding-window inference (+ TTA)
# ----------------------------------------------------------------------------

_TTA = ["none", "hflip", "vflip", "rot90", "rot180", "rot270"]


def _apply_tta(x, t):
    if t == "none":
        return x
    if t == "hflip":
        return torch.flip(x, dims=[3])
    if t == "vflip":
        return torch.flip(x, dims=[2])
    if t == "rot90":
        return torch.rot90(x, 1, dims=[2, 3])
    if t == "rot180":
        return torch.rot90(x, 2, dims=[2, 3])
    if t == "rot270":
        return torch.rot90(x, 3, dims=[2, 3])
    raise ValueError(t)


def _undo_tta(x, t):
    if t == "none":
        return x
    if t == "hflip":
        return torch.flip(x, dims=[3])
    if t == "vflip":
        return torch.flip(x, dims=[2])
    if t == "rot90":
        return torch.rot90(x, -1, dims=[2, 3])
    if t == "rot180":
        return torch.rot90(x, -2, dims=[2, 3])
    if t == "rot270":
        return torch.rot90(x, -3, dims=[2, 3])
    raise ValueError(t)


def _gaussian_weight(patch_size: int, sigma_scale: float = 0.125) -> np.ndarray:
    """2D Gaussian window (1 at centre -> ~0 at edges), nnU-Net style, to
    down-weight patch borders during sliding-window aggregation."""
    coords = np.arange(patch_size) - (patch_size - 1) / 2.0
    sigma = patch_size * sigma_scale
    g1 = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    w = np.outer(g1, g1)
    return (w / w.max()).astype(np.float64)


@torch.no_grad()
def sliding_window_predict(model, x_full, device, patch_size=256, overlap=0.5,
                           use_tta=False, gaussian=True, pad_mask=None,
                           return_cls=False):
    """Probability map (H,W) for a full scene via overlapped windows.

    ``x_full`` is ``(C,H,W)``, or ``(T,C,H,W)`` for the temporal models -- in
    which case ``pad_mask`` is a length-T boolean array marking replicated
    padding frames. The same window geometry is used either way; only the
    channel/time axes differ.

    ``gaussian=True`` weights each patch's contribution by a centred Gaussian
    (nnU-Net): centre pixels count most, borders least, suppressing tiling seams.

    ``return_cls=True`` additionally returns the MAX auxiliary presence
    probability over all windows -- the standard max-pooling MIL aggregation from
    instance (patch) scores to a bag (scan) score.
    """
    model.eval()
    temporal = x_full.ndim == 4
    H, W = x_full.shape[-2:]
    stride = max(1, int(patch_size * (1 - overlap)))
    prob = np.zeros((H, W), dtype=np.float64)
    weight = np.zeros((H, W), dtype=np.float64)
    wmap = _gaussian_weight(patch_size) if gaussian else np.ones((patch_size, patch_size))
    xt = torch.from_numpy(np.ascontiguousarray(x_full)).float().unsqueeze(0).to(device)
    mask_t = None
    if temporal and pad_mask is not None:
        mask_t = torch.from_numpy(np.asarray(pad_mask, dtype=bool)).unsqueeze(0).to(device)

    rows = list(range(0, H - patch_size + 1, stride))
    cols = list(range(0, W - patch_size + 1, stride))
    if rows[-1] + patch_size < H:
        rows.append(H - patch_size)
    if cols[-1] + patch_size < W:
        cols.append(W - patch_size)

    transforms = _TTA if use_tta else ["none"]
    cls_max = None
    for r in rows:
        for c in cols:
            patch = xt[..., r:r + patch_size, c:c + patch_size]
            acc = torch.zeros((1, 1, patch_size, patch_size), device=device)
            for t in transforms:
                # TTA flips/rotations act on the trailing spatial dims, so they
                # apply unchanged to the (B,T,C,H,W) temporal layout.
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                    out = forward(model, _apply_tta(patch, t), mask_t)
                if isinstance(out, tuple) and out[1] is not None:
                    p = torch.sigmoid(out[1].float()).max().item()
                    cls_max = p if cls_max is None else max(cls_max, p)
                acc += torch.sigmoid(_undo_tta(seg_logits(out), t).float())
            acc /= len(transforms)
            prob[r:r + patch_size, c:c + patch_size] += acc[0, 0].cpu().numpy() * wmap
            weight[r:r + patch_size, c:c + patch_size] += wmap
    result = prob / np.maximum(weight, 1e-8)
    return (result, cls_max) if return_cls else result


def scene_loader(cfg, rows: List[dict], norm_stats):
    """Return ``load(i) -> (x, y, pad_mask)`` for the configured input mode.

    Hides whether an experiment feeds one scan or a short in-night sequence, so
    threshold calibration and full-scene evaluation are identical either way.
    Neighbours are resolved within ``rows``, i.e. within one split -- and since
    whole nights live in one split, no sequence can cross a split boundary.
    """
    tc = dataset.temporal_cfg(cfg)
    if not tc["enabled"]:
        def load(i):
            x, y = dataset.load_full_scene(cfg, rows[i], norm_stats)
            return x, y, None
        return load

    seq_index = dataset.build_sequence_index(rows, tc["radius"], tc["max_gap_minutes"])

    def load(i):
        return dataset.load_full_scene_sequence(cfg, rows, i, norm_stats, seq_index)
    return load


def calibrate_threshold(model, rows: List[dict], cfg, norm_stats, device,
                        thresholds, max_scenes=None) -> float:
    """Pick the threshold maximizing mean full-scene Dice on POSITIVE val scenes.

    Dice is undefined on all-background (negative) scenes, so calibration uses
    only scenes that contain ground-truth positives.
    """
    # Neighbour lookup must span the whole split, so index first and subset after.
    load = scene_loader(cfg, rows, norm_stats)
    pos_idx = [i for i, r in enumerate(rows) if int(r["label"]) == 1]
    if max_scenes is not None and len(pos_idx) > max_scenes:
        pos_idx = pos_idx[:max_scenes]
    ps = int(cfg["patch"]["size"])
    ov = float(cfg["eval"].get("overlap", cfg["eval"].get("sliding_overlap", 0.5)))
    tta = bool(cfg["eval"].get("tta", True))
    probs, trues = [], []
    for i in pos_idx:
        x, y, pad = load(i)
        probs.append(sliding_window_predict(model, x, device, ps, ov, tta, pad_mask=pad))
        trues.append(y)
    best_t, best_d = 0.5, -1.0
    for t in thresholds:
        d = np.mean([compute_metrics(p > t, y)["dice"] for p, y in zip(probs, trues)])
        if d > best_d:
            best_d, best_t = d, float(t)
    return best_t


def evaluate_full_scene(model, rows: List[dict], cfg, norm_stats, device,
                        threshold, tta=True) -> Dict[str, float]:
    """Full-scene evaluation on held-out scenes at a fixed threshold.

    Reports metrics two ways over POSITIVE scenes (both are legitimate; they
    answer different questions and the reference notebook mixed them):
      * MACRO (``dice`` etc.): mean of per-scene metrics. Every scene counts
        equally, so sparse/hard plumes pull it down. This matches the notebook's
        reported full-scene test number and is the headline metric.
      * MICRO (``dice_micro`` etc.): pixel-pooled across all scenes. Dominated by
        dense plumes; matches the patch-level ``val_dice`` reported per epoch.
    For NEGATIVE (all-background) scenes we report ``bg_fp_rate`` = mean fraction
    of pixels wrongly predicted positive.
    """
    ps = int(cfg["patch"]["size"])
    ov = float(cfg["eval"].get("overlap", cfg["eval"].get("sliding_overlap", 0.5)))
    gaussian = bool(cfg["eval"].get("gaussian_window", True))
    tau = float(cfg["eval"].get("nsd_tolerance", 2.0))
    boundary = bool(cfg["eval"].get("boundary_metrics", True))
    pos_metrics, bg_fp = [], []
    TP = FP = FN = TN = 0.0
    load = scene_loader(cfg, rows, norm_stats)
    for i, row in enumerate(rows):
        x, y, pad = load(i)
        prob = sliding_window_predict(model, x, device, ps, ov, tta, gaussian=gaussian,
                                      pad_mask=pad)
        pred = prob > threshold
        if int(row["label"]) == 1:
            m = compute_metrics(pred, y)
            if boundary:
                m["boundary_iou"] = metrics_mod.boundary_iou(pred, y)
                m.update(metrics_mod.surface_metrics(pred, y, tau=tau))
            pos_metrics.append(m)
            p = pred.reshape(-1).astype(np.float64); t = y.reshape(-1).astype(np.float64)
            TP += float((p * t).sum()); FP += float((p * (1 - t)).sum())
            FN += float(((1 - p) * t).sum()); TN += float(((1 - p) * (1 - t)).sum())
        else:
            bg_fp.append(float(pred.mean()))
    keys = ["dice", "iou", "precision", "recall", "f1", "accuracy"]
    if boundary:
        keys += ["boundary_iou", "nsd", "hd95", "assd"]
    out = {k: (float(np.nanmean([m[k] for m in pos_metrics])) if pos_metrics else float("nan"))
           for k in keys}
    eps = 1e-8
    out["dice_micro"] = 2 * TP / (2 * TP + FP + FN + eps)
    out["iou_micro"] = TP / (TP + FP + FN + eps)
    out["precision_micro"] = TP / (TP + FP + eps)
    out["recall_micro"] = TP / (TP + FN + eps)
    out["n_pos_scenes"] = len(pos_metrics)
    out["n_neg_scenes"] = len(bg_fp)
    out["bg_fp_rate"] = float(np.mean(bg_fp)) if bg_fp else float("nan")
    out["threshold"] = float(threshold)
    return out
