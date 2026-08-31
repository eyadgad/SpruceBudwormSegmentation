"""Run a single experiment end-to-end: train (resume-aware) -> full-scene eval.

Resumability contract:
  * If ``<name>_result.json`` exists -> skip entirely (already done).
  * Else if ``<name>_resume.pt`` exists -> continue training from it (including
    the completed-training flag, so a crash during evaluation does not retrain).
  * Best model saved the instant val Dice improves; resume snapshot saved every
    ``train.snapshot_every`` epochs; final model saved at the end.
"""
from __future__ import annotations

import os
# Set before torch initializes the CUDA allocator: expandable segments prevent
# the memory fragmentation that OOMs long training runs on limited VRAM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import checkpoint as ckpt
from . import config as cfgmod
from . import engine, logutil, paths
from .dataset import NightBalancedSceneSampler, RadarPatchDataset, SceneGroupedSampler
from .losses import create_loss
from .models import create_model, count_params


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch so a run is reproducible from its config.

    Nothing in src/ seeded torch or numpy before this: only the split, the
    negative sample and the sampler order were deterministic, while model init
    and every patch crop (which use the *global* numpy RNG) were not. Without
    this, "same seed -> same result" is not a claim the repo can support.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loaders(cfg, manifest, norm_stats):
    train_ds = RadarPatchDataset(cfg, manifest, "train", norm_stats, mode="train")
    val_ds = RadarPatchDataset(cfg, manifest, "val", norm_stats, mode="eval")
    nw = int(cfg["train"].get("num_workers", 0))
    bs = int(cfg["train"]["batch_size"])
    common = dict(num_workers=nw, pin_memory=bool(cfg["train"].get("pin_memory", False)),
                  persistent_workers=nw > 0)
    seed = int(cfg["split"]["seed"])
    # Night-balanced sampling is a TRAIN-ONLY intervention. The val loader keeps
    # its deterministic patch grid so the model-selection distribution is never
    # reweighted.
    if bool(cfg["train"].get("night_balanced", False)):
        sampler = NightBalancedSceneSampler(train_ds, shuffle=True, seed=seed)
    else:
        sampler = SceneGroupedSampler(train_ds, shuffle=True, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, sampler


def run_experiment(cfg: Dict, manifest, norm_stats, device, verbose=True) -> Dict:
    name = cfg["name"]
    exp_dir = paths.experiments_dir(cfg)
    ckpt_dir = paths.checkpoint_dir(cfg)
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1) Already finished -> skip.
    if ckpt.is_done(exp_dir, name):
        if verbose:
            print(f"[skip] {name}: result exists")
        return ckpt.load_result(exp_dir, name)

    cfgmod.save_config(cfg, exp_dir / f"{name}_config.json")
    log = logutil.get_logger(name, exp_dir / f"{name}_train.log", console=verbose)

    seed_everything(int(cfg["train"].get("seed", cfg["split"]["seed"])))
    model = create_model(cfg).to(device)
    n_params = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                                  weight_decay=float(cfg["train"].get("weight_decay", 1e-5)))
    scheduler = engine.build_scheduler(optimizer, cfg)
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(cfg["train"].get("amp_dtype", "bfloat16")) == "bfloat16" else torch.float16
    # GradScaler is only needed for float16 (bf16 has fp32 range and needs no loss scaling).
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if (use_amp and amp_dtype == torch.float16) else None
    criterion = create_loss(cfg).to(device)

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("patience", 15))
    tol = float(cfg["train"].get("es_tolerance", 1e-3))
    accum = int(cfg["train"].get("accum_steps", 1))
    snap_every = int(cfg["train"].get("snapshot_every", 1))

    start_epoch, best_dice, best_epoch = 0, 0.0, -1
    no_improve, training_complete = 0, False
    history = []

    # 2) Resume if a snapshot exists.
    state = ckpt.load_checkpoint(ckpt.resume_path(ckpt_dir, name), device)
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"]
        best_dice = state["best_dice"]
        best_epoch = state["best_epoch"]
        history = state.get("history", [])
        no_improve = state.get("no_improve", 0)
        training_complete = state.get("training_complete", False)
        ckpt.set_rng_state(state.get("rng", {}))
        log.info(f"resuming from epoch {start_epoch} (best val_dice={best_dice:.4f} @ep{best_epoch})")

    train_loader, val_loader, train_sampler = _loaders(cfg, manifest, norm_stats)

    # 3) Train (unless a prior run already completed training).
    if not training_complete:
        log.info(f"START {name} | model={cfg['model']['name']} loss={cfg['loss']['name']} "
                 f"channels={len(cfg['channels'])} params={n_params/1e6:.2f}M | "
                 f"epochs={epochs} batch={cfg['train']['batch_size']} lr={cfg['train']['lr']} "
                 f"amp={cfg['train'].get('amp_dtype','bfloat16') if use_amp else 'off'}")
        log.info(logutil.EPOCH_HEADER)
        for epoch in range(start_epoch, epochs):
            train_sampler.set_epoch(epoch)
            t0 = time.time()
            lr = optimizer.param_groups[0]["lr"]
            tr_loss = engine.train_one_epoch(model, train_loader, optimizer, criterion,
                                             device, scaler, accum_steps=accum, verbose=verbose,
                                             use_amp=use_amp, amp_dtype=amp_dtype, logger=log)
            scheduler.step()
            val = engine.validate_patches(model, val_loader, device, threshold=0.5)
            history.append({"epoch": epoch, "train_loss": tr_loss, "lr": lr,
                            **{f"val_{k}": v for k, v in val.items()}})

            improved = val["dice"] > best_dice + tol
            if improved:
                best_dice, best_epoch, no_improve = val["dice"], epoch, 0
                ckpt.save_model(ckpt.best_path(ckpt_dir, name), model=model, epoch=epoch,
                                val_dice=best_dice, cfg=cfg)
            else:
                no_improve += 1

            log.info(logutil.format_epoch(epoch, epochs, tr_loss, val, best_dice, best_epoch,
                                          lr, time.time() - t0, improved))

            # Divergence stop: once a forward goes non-finite it corrupts the
            # BatchNorm running buffers, so every later forward is NaN, every
            # batch is skipped (train_loss==0), and val_dice stays 0 with no
            # recovery. Stop and keep the best checkpoint instead of grinding
            # through the remaining epochs.
            diverged = (tr_loss == 0.0) or (val["dice"] == 0.0 and best_dice > 0.05)
            if diverged:
                log.warning(f"training diverged at epoch {epoch + 1} (non-finite outputs, "
                            f"unrecoverable). Stopping; keeping best checkpoint "
                            f"val_dice={best_dice:.4f} @ep{best_epoch}.")
                break

            if (epoch + 1) % snap_every == 0 or epoch == epochs - 1:
                _save_resume(ckpt_dir, name, model, optimizer, scheduler, scaler,
                             epoch + 1, best_dice, best_epoch, no_improve, False, history, cfg)

            if no_improve >= patience:
                log.info(f"early stop at epoch {epoch + 1} (no val-dice improvement for "
                         f"{patience} epochs; best {best_dice:.4f} @ep{best_epoch})")
                break

        ckpt.save_model(ckpt.final_path(ckpt_dir, name), model=model, epoch=epochs,
                        val_dice=history[-1]["val_dice"] if history else 0.0, cfg=cfg)
        # Mark training complete so a crash during eval won't retrain.
        _save_resume(ckpt_dir, name, model, optimizer, scheduler, scaler,
                     epochs, best_dice, best_epoch, no_improve, True, history, cfg)

    # 4) Full-scene evaluation with the BEST checkpoint.
    best_state = ckpt.load_checkpoint(ckpt.best_path(ckpt_dir, name), device)
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    val_rows = manifest[manifest["split"] == "val"].to_dict("records")
    test_rows = manifest[manifest["split"] == "test"].to_dict("records")

    thresholds = cfg["eval"].get("threshold_range",
                                 [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    calib_max = cfg["eval"].get("calib_max_scenes", None)
    tta = bool(cfg["eval"].get("tta", True))
    # Hold the test set back until architecture / lambda / threshold decisions are
    # frozen; `python -m src.finalize` scores it once, deliberately.
    defer_test = bool(cfg["eval"].get("defer_test", False))
    log.info(f"evaluating best checkpoint (ep{best_epoch}, val_dice={best_dice:.4f}) on full "
             f"{'val' if defer_test else 'test'} scenes ...")
    best_t = engine.calibrate_threshold(model, val_rows, cfg, norm_stats, device,
                                        thresholds, max_scenes=calib_max)

    result = {
        "name": name,
        "model": cfg["model"]["name"],
        "loss": cfg["loss"]["name"],
        "channels": cfg["channels"],
        "n_channels": len(cfg["channels"]),
        "n_params": int(n_params),
        "best_val_dice_patch": float(best_dice),
        "best_epoch": int(best_epoch),
        "calibrated_threshold": float(best_t),
        "defer_test": defer_test,
    }
    if defer_test:
        result["val_full_scene"] = engine.evaluate_full_scene(
            model, val_rows, cfg, norm_stats, device, threshold=best_t, tta=tta)
    else:
        result["test_full_scene"] = engine.evaluate_full_scene(
            model, test_rows, cfg, norm_stats, device, threshold=best_t, tta=tta)
    ckpt.save_result(exp_dir, name, result)
    _save_history_csv(exp_dir, name, history)

    # 5) Clean completion -> drop the resume snapshot.
    rp = ckpt.resume_path(ckpt_dir, name)
    if rp.exists():
        rp.unlink()

    tm = result.get("val_full_scene") or result["test_full_scene"]
    label = "VAL " if defer_test else "TEST"
    log.info(f"DONE {name} | threshold={best_t:.2f}"
             + (" | test deferred to src.finalize" if defer_test else ""))
    log.info(f"  {label} full-scene | dice(macro)={tm['dice']:.4f} dice(micro)={tm.get('dice_micro', float('nan')):.4f} "
             f"iou(macro)={tm['iou']:.4f}")
    log.info(f"                  | precision={tm['precision']:.4f} recall={tm['recall']:.4f} "
             f"bg_fp_rate={tm.get('bg_fp_rate', float('nan')):.4f}")
    if "boundary_iou" in tm:
        log.info(f"  boundary        | boundaryIoU={tm['boundary_iou']:.4f} NSD@{cfg['eval'].get('nsd_tolerance',2.0)}px={tm['nsd']:.4f} "
                 f"HD95={tm['hd95']:.1f}px ASSD={tm['assd']:.1f}px")
    return result


def _save_resume(ckpt_dir, name, model, optimizer, scheduler, scaler, epoch,
                 best_dice, best_epoch, no_improve, training_complete, history, cfg):
    path = ckpt.resume_path(ckpt_dir, name)
    # extend checkpoint.save_resume payload with early-stop bookkeeping
    import torch as _t
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    _t.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch, "best_dice": best_dice, "best_epoch": best_epoch,
        "no_improve": no_improve, "training_complete": training_complete,
        "history": history, "rng": ckpt.rng_state(), "cfg": cfg,
    }, tmp)
    tmp.replace(path)


def _save_history_csv(exp_dir, name, history):
    if not history:
        return
    import pandas as pd
    pd.DataFrame(history).to_csv(exp_dir / f"{name}_history.csv", index=False)


if __name__ == "__main__":
    # Run ONE experiment in this process — invoked by src.run for process
    # isolation, so a CUDA OOM/crash here cannot affect other experiments and all
    # GPU memory is reclaimed by the OS when this process exits.
    import argparse
    from . import data_prep
    ap = argparse.ArgumentParser(description="Run a single experiment by name")
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-config", default="configs/base_config.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    args = ap.parse_args()

    base = cfgmod.load_base_config(args.base_config)
    matches = [e for e in cfgmod.load_experiments(args.experiments) if e["name"] == args.name]
    if not matches:
        raise SystemExit(f"experiment '{args.name}' not found in {args.experiments}")
    cfg = cfgmod.resolve_experiment(base, matches[0])
    manifest, norm_stats = data_prep.load_artifacts(base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_experiment(cfg, manifest, norm_stats, device, verbose=True)
