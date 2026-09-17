"""Train a scan-level head on a frozen S0 Attention U-Net.

Encoder/decoder stay at ``night_base_attunet9`` weights. Only ``cls_fc`` is
updated, with BCE on the manifest scan label (384 full scenes). That tests
whether bottleneck features already contain a usable swarm/quiet score
without damaging the segmenter.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import checkpoint as ckpt, config as cfgmod, data_prep, engine, logutil, paths
from .classify import (
    ScanClsDataset, _auroc, _pos_weight, _save_resume, calibrate_threshold,
)
from .experiment import seed_everything
from .models import count_params, create_model
from .presence import classification_metrics


def _freeze_backbone(model) -> int:
    for p in model.parameters():
        p.requires_grad = False
    n = 0
    for p in model.cls_fc.parameters():
        p.requires_grad = True
        n += p.numel()
    model.eval()
    return n


def _probe_logits(model, x):
    """Linear probe: stop-grad through the frozen encoder."""
    with torch.no_grad():
        _e1, _e2, _e3, _e4, b = model._encode(x)
    return model.cls_fc(model.cls_pool(b.detach()).flatten(1))


def _loaders(cfg, manifest, norm_stats):
    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"].get("num_workers", 0))
    common = dict(num_workers=nw, pin_memory=bool(cfg["train"].get("pin_memory", False)))
    train_ds = ScanClsDataset(cfg, manifest, "train", norm_stats, train=True)
    val_ds = ScanClsDataset(cfg, manifest, "val", norm_stats, train=False)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device, use_amp, amp_dtype) -> float:
    # Backbone stays in eval (frozen BN). cls_fc still receives gradients.
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = _probe_logits(model, x).reshape(y.shape)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        optimizer.step()
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(n, 1)


@torch.no_grad()
def collect_scores(model, loader, device, use_amp, amp_dtype) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores, truth = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = _probe_logits(model, x).reshape(y.shape)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy().reshape(-1))
        truth.append(y.numpy().reshape(-1))
    return np.concatenate(scores), np.concatenate(truth)


def run_experiment(cfg: Dict, manifest, norm_stats, device, verbose=True) -> Dict:
    name = cfg["name"]
    exp_dir = paths.experiments_dir(cfg)
    ckpt_dir = paths.checkpoint_dir(cfg)
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if ckpt.is_done(exp_dir, name):
        if verbose:
            print(f"[skip] {name}: result exists")
        return ckpt.load_result(exp_dir, name)

    cfgmod.save_config(cfg, exp_dir / f"{name}_config.json")
    log = logutil.get_logger(name, exp_dir / f"{name}_train.log", console=verbose)
    seed_everything(int(cfg["train"].get("seed", cfg["split"]["seed"])))

    model = create_model(cfg).to(device)
    init_path = cfg.get("model", {}).get("init_checkpoint")
    if not init_path:
        raise ValueError("frozen_s0_cls requires model.init_checkpoint")
    init_state = ckpt.load_checkpoint(Path(init_path), device)
    if init_state is None:
        raise FileNotFoundError(init_path)
    missing, unexpected = model.load_state_dict(init_state["model"], strict=False)
    log.info(f"init from {init_path} | missing={list(missing)} unexpected={list(unexpected)}")
    n_trainable = _freeze_backbone(model)
    n_params = count_params(model)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["train"]["lr"]),
                                  weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    scheduler = engine.build_scheduler(optimizer, cfg)
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(cfg["train"].get("amp_dtype", "bfloat16")) == "bfloat16" else torch.float16
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=_pos_weight(manifest).to(device))

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("patience", 10))
    start_epoch, best_auc, best_epoch, no_improve = 0, -1.0, -1, 0
    history: List[Dict] = []
    state = ckpt.load_checkpoint(ckpt.resume_path(ckpt_dir, name), device)
    if state is not None:
        model.load_state_dict(state["model"])
        _freeze_backbone(model)
        optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        best_auc = float(state.get("best_dice", -1.0))
        best_epoch = int(state.get("best_epoch", -1))
        no_improve = int(state.get("no_improve", 0))
        history = state.get("history", [])
        if state.get("training_complete"):
            start_epoch = epochs

    train_loader, val_loader = _loaders(cfg, manifest, norm_stats)
    log.info(f"START {name} | frozen S0 linear probe | trainable={n_trainable} "
             f"total={n_params/1e6:.2f}M | epochs={epochs} batch={cfg['train']['batch_size']} "
             f"lr={cfg['train']['lr']}")
    log.info(f"{'epoch':>9} | {'train_loss':>10} | {'val_auc':>8} {'val_f1':>8} "
             f"{'val_prec':>8} {'val_rec':>8} | {'best':>15} | {'lr':>8} | {'time':>6}")

    if start_epoch < epochs:
        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                      device, use_amp, amp_dtype)
            scheduler.step()
            scores, truth = collect_scores(model, val_loader, device, use_amp, amp_dtype)
            auc = _auroc(truth, scores)
            mid = classification_metrics(truth, scores, 0.5)
            improved = auc > best_auc + float(cfg["train"].get("es_tolerance", 1e-4))
            if improved:
                best_auc, best_epoch, no_improve = auc, epoch, 0
                ckpt.save_model(ckpt.best_path(ckpt_dir, name), model=model, epoch=epoch,
                                val_dice=best_auc, cfg=cfg)
            else:
                no_improve += 1
            history.append({"epoch": epoch, "train_loss": tr_loss, "val_auc": auc,
                            "val_f1": mid.get("f1"), "lr": lr})
            f1 = mid.get("f1") if mid.get("f1") is not None else float("nan")
            prec = mid.get("precision") if mid.get("precision") is not None else float("nan")
            rec = mid.get("recall") if mid.get("recall") is not None else float("nan")
            star = " *" if improved else "  "
            log.info(f"{epoch + 1:4d}/{epochs:<4d} | {tr_loss:10.4f} | "
                     f"{auc:8.4f} {f1:8.4f} {prec:8.4f} {rec:8.4f} | "
                     f"{best_auc:8.4f}@{best_epoch:<3d}{star} | {lr:8.2e} | "
                     f"{time.time() - t0:5.0f}s")
            _save_resume(ckpt_dir, name, model, optimizer, scheduler, epoch + 1,
                         best_auc, best_epoch, no_improve, False, history, cfg)
            if no_improve >= patience:
                log.info(f"early stop at epoch {epoch + 1} (no val AUROC gain for {patience} epochs)")
                break
        ckpt.save_model(ckpt.final_path(ckpt_dir, name), model=model, epoch=epochs,
                        val_dice=best_auc, cfg=cfg)

    best_state = ckpt.load_checkpoint(ckpt.best_path(ckpt_dir, name), device)
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        _freeze_backbone(model)
    scores, truth = collect_scores(model, val_loader, device, use_amp, amp_dtype)
    grid = cfg["eval"].get("threshold_range", [i / 20 for i in range(1, 20)])
    best_t = calibrate_threshold(truth, scores, grid)
    metrics = classification_metrics(truth, scores, best_t)
    result = {
        "name": name,
        "task": "frozen_scan_head",
        "model": cfg["model"]["name"],
        "n_params": int(n_params),
        "n_trainable": int(n_trainable),
        "best_val_auc": float(best_auc),
        "best_epoch": int(best_epoch),
        "calibrated_threshold": float(best_t),
        "val": {
            "auroc": _auroc(truth, scores),
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "youden_j": metrics.get("youden_j"),
            "n": metrics.get("n"),
            "n_positive": metrics.get("n_positive"),
            "n_negative": metrics.get("n_negative"),
            "confusion": metrics.get("confusion"),
            "threshold": float(best_t),
        },
    }
    ckpt.save_result(exp_dir, name, result)
    if history:
        pd.DataFrame(history).to_csv(exp_dir / f"{name}_history.csv", index=False)
    rp = ckpt.resume_path(ckpt_dir, name)
    if rp.exists():
        rp.unlink()
    v = result["val"]
    log.info(f"DONE {name} | val AUROC={v['auroc']:.4f} F1={v['f1']} "
             f"P={v['precision']} R={v['recall']} t={best_t:.2f}")
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Frozen-S0 scan-head probe")
    ap.add_argument("--base-config", default="configs/base_config_frozen_cls.yaml")
    ap.add_argument("--experiments", default="configs/experiments_frozen_cls.yaml")
    ap.add_argument("--name", default="frozen_s0_cls")
    args = ap.parse_args()
    base = cfgmod.load_base_config(args.base_config)
    matches = [e for e in cfgmod.load_experiments(args.experiments) if e["name"] == args.name]
    if not matches:
        raise SystemExit(f"experiment '{args.name}' not found")
    cfg = cfgmod.resolve_experiment(base, matches[0])
    manifest, norm_stats = data_prep.load_artifacts(base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[device] cuda ({torch.cuda.get_device_name(0)})", flush=True)
    run_experiment(cfg, manifest, norm_stats, device)


if __name__ == "__main__":
    main()
