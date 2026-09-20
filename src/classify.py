"""Scan-level swarm vs swarm-free classification on the night split.

One label per radar scan (manifest ``label``: 1 = swarm, 0 = swarm-free).
Training resizes the same 10-channel stack the night-split segmenters use.
After the classifiers finish, ``compare_to_segmentation`` scores the five
night-split segmenters with the rule the user asked for: any pixel above the
locked threshold => swarm.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from . import channels, checkpoint as ckpt, config as cfgmod, data_prep, engine, logutil, paths, sync
from .experiment import seed_everything
from .models import count_params, create_model
from .presence import classification_metrics, roc_analysis


def _resize(x: np.ndarray, size: int) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.squeeze(0).numpy()


class ScanClsDataset(Dataset):
    """One resized scene per item. Target is the scan label, not the pixel mask."""

    def __init__(self, cfg: Dict, manifest, split: str, norm_stats: Dict, train: bool):
        rows = manifest[manifest["split"] == split].reset_index(drop=True)
        self.rows = rows.to_dict("records")
        self.cfg = cfg
        self.norm_stats = norm_stats
        self.train = train
        self.img_size = int(cfg["model"].get("cls_img_size")
                            or cfg["model"].get("img_size", 384))
        self.aug = bool(cfg.get("augment", {}).get("enabled", False)) and train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        x = channels.build_stack(self.cfg, row["x_path"], self.cfg["channels"], self.norm_stats)
        x = _resize(x, self.img_size)
        if self.aug:
            if np.random.random() > 0.5:
                x = np.flip(x, axis=-1).copy()
            if np.random.random() > 0.5:
                x = np.flip(x, axis=-2).copy()
            k = int(np.random.randint(4))
            if k:
                x = np.rot90(x, k, axes=(-2, -1)).copy()
        y = torch.tensor([float(int(row["label"]))], dtype=torch.float32)
        return torch.from_numpy(np.ascontiguousarray(x)).float(), y


def _pos_weight(manifest) -> torch.Tensor:
    train = manifest[manifest["split"] == "train"]
    n_pos = max(int((train["label"] == 1).sum()), 1)
    n_neg = max(int((train["label"] == 0).sum()), 1)
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


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
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits = model(x).reshape(y.shape)
            loss = criterion(logits, y)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            logits = model(x).reshape(y.shape)
        scores.append(torch.sigmoid(logits.float()).cpu().numpy().reshape(-1))
        truth.append(y.numpy().reshape(-1))
    return np.concatenate(scores), np.concatenate(truth)


def _auroc(truth: np.ndarray, scores: np.ndarray) -> float:
    roc = roc_analysis(truth, scores)
    auc = roc.get("auc")
    return float(auc) if auc is not None else float("nan")


def calibrate_threshold(truth: np.ndarray, scores: np.ndarray, grid) -> float:
    """Pick the cutoff maximizing Youden's J on validation."""
    best_t, best_j = 0.5, -1.0
    for t in grid:
        m = classification_metrics(truth, scores, float(t))
        j = m.get("youden_j")
        if j is not None and j > best_j:
            best_j, best_t = float(j), float(t)
    return best_t


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
    n_params = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]),
                                  weight_decay=float(cfg["train"].get("weight_decay", 1e-5)))
    scheduler = engine.build_scheduler(optimizer, cfg)
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if str(cfg["train"].get("amp_dtype", "bfloat16")) == "bfloat16" else torch.float16
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=_pos_weight(manifest).to(device))

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("patience", 15))
    start_epoch, best_auc, best_epoch, no_improve = 0, -1.0, -1, 0
    history: List[Dict] = []
    # Same mirror contract as src.experiment: on a fresh machine the only
    # snapshot is the remote one, so pull before deciding where to resume.
    syncer = sync.RunSync(cfg, name, ckpt_dir, exp_dir, logger=log)
    if syncer.enabled and not ckpt.resume_path(ckpt_dir, name).exists():
        syncer.pull()
    state = ckpt.load_checkpoint(ckpt.resume_path(ckpt_dir, name), device)
    if state is not None:
        model.load_state_dict(state["model"])
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
    log.info(f"START {name} | cls={cfg['model']['name']} img={cfg['model'].get('img_size', 384)} "
             f"params={n_params/1e6:.2f}M | epochs={epochs} batch={cfg['train']['batch_size']}")
    log.info(f"{'epoch':>9} | {'train_loss':>10} | {'val_auc':>8} {'val_f1':>8} "
             f"{'val_prec':>8} {'val_rec':>8} | {'best':>15} | {'lr':>8} | {'time':>6}")

    import time
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
            syncer.maybe_push(epoch + 1, epochs)
            if no_improve >= patience:
                log.info(f"early stop at epoch {epoch + 1} (no val AUROC gain for {patience} epochs)")
                break
        ckpt.save_model(ckpt.final_path(ckpt_dir, name), model=model, epoch=epochs,
                        val_dice=best_auc, cfg=cfg)

    best_state = ckpt.load_checkpoint(ckpt.best_path(ckpt_dir, name), device)
    if best_state is not None:
        model.load_state_dict(best_state["model"])
    scores, truth = collect_scores(model, val_loader, device, use_amp, amp_dtype)
    grid = cfg["eval"].get("threshold_range", [i / 20 for i in range(1, 20)])
    best_t = calibrate_threshold(truth, scores, grid)
    metrics = classification_metrics(truth, scores, best_t)
    result = {
        "name": name,
        "task": "classify",
        "model": cfg["model"]["name"],
        "n_params": int(n_params),
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
    import pandas as pd
    if history:
        pd.DataFrame(history).to_csv(exp_dir / f"{name}_history.csv", index=False)
    syncer.push(tag="complete", include_resume=False)
    rp = ckpt.resume_path(ckpt_dir, name)
    if rp.exists():
        rp.unlink()
    v = result["val"]
    log.info(f"DONE {name} | val AUROC={v['auroc']:.4f} F1={v['f1']} "
             f"P={v['precision']} R={v['recall']} t={best_t:.2f}")
    return result


def _save_resume(ckpt_dir, name, model, optimizer, scheduler, epoch,
                 best_auc, best_epoch, no_improve, training_complete, history, cfg):
    path = ckpt.resume_path(ckpt_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch, "best_dice": best_auc, "best_epoch": best_epoch,
        "no_improve": no_improve, "training_complete": training_complete,
        "history": history, "rng": ckpt.rng_state(), "cfg": cfg,
    }, tmp)
    tmp.replace(path)


@torch.no_grad()
def score_segmenter_any_pixel(cfg: Dict, manifest, norm_stats, device) -> Dict:
    """Any pixel above the locked threshold => swarm (class 1)."""
    from .models import create_model as create_seg
    name = cfg["name"]
    ckpt_dir = paths.checkpoint_dir(cfg)
    exp_dir = paths.experiments_dir(cfg)
    best = ckpt.load_checkpoint(ckpt.best_path(ckpt_dir, name), device)
    if best is None:
        return {"name": name, "skip": "no checkpoint"}
    train_result = ckpt.load_result(exp_dir, name)
    threshold = float(train_result["calibrated_threshold"])
    model = create_seg(cfg).to(device)
    model.load_state_dict(best["model"])
    model.eval()
    rows = manifest[manifest["split"] == "val"].to_dict("records")
    load = engine.scene_loader(cfg, rows, norm_stats)
    ps = int(cfg["patch"]["size"])
    ov = float(cfg["eval"].get("overlap", 0.5))
    tta = bool(cfg["eval"].get("tta", False))
    truth, pred, max_prob, pred_area = [], [], [], []
    for i, row in enumerate(rows):
        x, y, pad = load(i)
        prob = engine.sliding_window_predict(model, x, device, ps, ov, tta, pad_mask=pad)
        hit = bool((prob > threshold).any())
        truth.append(int(row["label"]))
        pred.append(int(hit))
        max_prob.append(float(np.max(prob)))
        pred_area.append(float((prob > threshold).mean()))
    truth_a = np.asarray(truth)
    pred_a = np.asarray(pred)
    # Binary any-pixel rule (no extra cutoff). AUROC uses max probability.
    tp = int(((pred_a == 1) & (truth_a == 1)).sum())
    fp = int(((pred_a == 1) & (truth_a == 0)).sum())
    tn = int(((pred_a == 0) & (truth_a == 0)).sum())
    fn = int(((pred_a == 0) & (truth_a == 1)).sum())
    from .presence import _ratio
    return {
        "name": name,
        "source": "segmentation_any_pixel",
        "pixel_threshold": threshold,
        "val": {
            "auroc": _auroc(truth_a, np.asarray(max_prob)),
            "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
            "precision": _ratio(tp, tp + fp),
            "recall": _ratio(tp, tp + fn),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "n": int(len(truth_a)),
            "n_positive": int(truth_a.sum()),
            "n_negative": int((1 - truth_a).sum()),
            "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        },
    }


def compare_to_segmentation(cls_base: str, cls_exps: str, seg_base: str, seg_exps: str,
                            verbose=True) -> None:
    cls_cfg = cfgmod.load_base_config(cls_base)
    seg_cfg = cfgmod.load_base_config(seg_base)
    manifest, norm_stats = data_prep.load_artifacts(cls_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for exp in cfgmod.load_experiments(cls_exps):
        cfg = cfgmod.resolve_experiment(cls_cfg, exp)
        if not ckpt.is_done(paths.experiments_dir(cfg), cfg["name"]):
            continue
        res = ckpt.load_result(paths.experiments_dir(cfg), cfg["name"])
        v = res["val"]
        rows.append({
            "name": res["name"], "family": "classifier", "model": res["model"],
            "auroc": v.get("auroc"), "accuracy": v.get("accuracy"),
            "precision": v.get("precision"), "recall": v.get("recall"),
            "f1": v.get("f1"), "threshold": v.get("threshold"),
        })
    for exp in cfgmod.load_experiments(seg_exps):
        cfg = cfgmod.resolve_experiment(seg_cfg, exp)
        if not ckpt.is_done(paths.experiments_dir(cfg), cfg["name"]):
            continue
        if verbose:
            print(f"[compare] scoring {cfg['name']} (any-pixel rule) ...", flush=True)
        scored = score_segmenter_any_pixel(cfg, manifest, norm_stats, device)
        if "skip" in scored:
            continue
        v = scored["val"]
        rows.append({
            "name": scored["name"], "family": "segmentation_any_pixel",
            "model": cfg["model"]["name"],
            "auroc": v.get("auroc"), "accuracy": v.get("accuracy"),
            "precision": v.get("precision"), "recall": v.get("recall"),
            "f1": v.get("f1"), "threshold": scored["pixel_threshold"],
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("auroc", ascending=False, na_position="last").reset_index(drop=True)
    out = paths.output_dir(cls_cfg)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scan_cls_comparison.csv", index=False)
    with open(out / "scan_cls_comparison.md", "w", encoding="utf-8") as f:
        f.write("# Scan-level swarm vs swarm-free (night split, val)\n\n")
        f.write("Classifiers are trained on the scan label. Segmenters are converted with "
                "`any pixel > locked threshold => swarm`. AUROC for segmenters uses the "
                "per-scan max probability so the ranking is threshold-free.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    if verbose:
        print(df.to_string(index=False))
        print(f"\n[table] {out / 'scan_cls_comparison.md'}")


def run_all(base_path: str, experiments_path: str, compare: bool = True) -> None:
    base = cfgmod.load_base_config(base_path)
    experiments = cfgmod.load_experiments(experiments_path)
    manifest, norm_stats = data_prep.load_artifacts(base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[device] cuda ({torch.cuda.get_device_name(0)})")
    import subprocess, sys
    for exp in experiments:
        name = exp["name"]
        if ckpt.is_done(paths.experiments_dir(cfgmod.resolve_experiment(base, exp)), name):
            print(f"[skip] {name}: result exists")
            continue
        print(f"\n=== launching {name} ===", flush=True)
        rc = subprocess.run(
            [sys.executable, "-m", "src.classify", "--name", name,
             "--base-config", str(base_path), "--experiments", str(experiments_path)],
        ).returncode
        if rc != 0:
            print(f"[error] {name}: exit {rc} (continuing)")
    if compare:
        compare_to_segmentation(
            base_path, experiments_path,
            "configs/base_config_night.yaml", "configs/experiments_night.yaml",
        )


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Scan-level swarm/swarm-free classification")
    ap.add_argument("--name", default=None)
    ap.add_argument("--base-config", default="configs/base_config_cls.yaml")
    ap.add_argument("--experiments", default="configs/experiments_cls.yaml")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()
    if args.compare_only:
        compare_to_segmentation(args.base_config, args.experiments,
                                "configs/base_config_night.yaml",
                                "configs/experiments_night.yaml")
    elif args.all or args.name is None:
        run_all(args.base_config, args.experiments)
    else:
        base = cfgmod.load_base_config(args.base_config)
        matches = [e for e in cfgmod.load_experiments(args.experiments) if e["name"] == args.name]
        if not matches:
            raise SystemExit(f"experiment '{args.name}' not found")
        cfg = cfgmod.resolve_experiment(base, matches[0])
        manifest, norm_stats = data_prep.load_artifacts(base)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_experiment(cfg, manifest, norm_stats, device)
