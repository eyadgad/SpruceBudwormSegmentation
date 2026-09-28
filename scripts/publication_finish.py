"""Finish the publication evidence the frozen test set can still support.

Scores the four Swin-Tiny checkpoints that exist (seeds 43-46) on the test
split, applies each as a scan gate to the seed-42 segmenters, and writes
three case plates. The seed-42 classifier weights are not on disk; nothing
here refits a threshold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.checkpoint import load_checkpoint  # noqa: E402
from src.classify import ScanClsDataset, collect_scores  # noqa: E402
from src.config import load_base_config  # noqa: E402
from src.data_prep import load_artifacts  # noqa: E402
from src.engine import scene_loader, sliding_window_predict  # noqa: E402
from src.models import create_model  # noqa: E402
from src.presence import pr_analysis, roc_analysis  # noqa: E402

COMP = ROOT / "outputs" / "night_split" / "comparison"
CLS = ROOT / "outputs" / "night_cascade"
OUT = ROOT / "sprucebudworm_progress.github.io" / "assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
THR = 0.15


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def score_swin(device) -> dict:
    base = load_base_config(ROOT / "configs" / "base_config_cascade_cls.yaml")
    manifest, norm = load_artifacts(base)
    records = {}
    for seed in (43, 44, 45, 46):
        name = f"cls_swin_tiny_bal_s{seed}"
        cfg = json.loads((CLS / "experiments" / f"{name}_config.json").read_text(encoding="utf-8"))
        cfg["model"]["pretrained"] = False
        result = json.loads((CLS / "experiments" / f"{name}_result.json").read_text(encoding="utf-8"))
        model = create_model(cfg).to(device)
        state = load_checkpoint(CLS / "checkpoints" / f"{name}_best.pt", device)
        model.load_state_dict(state["model"])
        model.eval()
        use_amp = device.type == "cuda"
        per_split = {}
        for split in ("val", "test"):
            ds = ScanClsDataset(cfg, manifest, split, norm, train=False)
            loader = DataLoader(ds, batch_size=8, shuffle=False)
            scores, truth = collect_scores(model, loader, device, use_amp, torch.bfloat16)
            ts = [int(r["timestamp"]) for r in ds.rows]
            roc = roc_analysis(truth, scores)
            pr = pr_analysis(truth, scores)
            per_split[split] = {
                "ts": ts,
                "truth": [int(v) for v in truth],
                "score": [float(v) for v in scores],
                "auc": roc["auc"],
                "ap": pr["ap"],
                "roc": roc["points"],
                "pr": pr["points"],
            }
            print(f"{name} {split} AUC {roc['auc']:.3f} AP {pr['ap']:.3f} n {len(ts)}")
        records[name] = {"threshold": float(result["calibrated_threshold"]), "splits": per_split}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    path = COMP / "swin_test_scores.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return records


def _gated(rows, keep):
    """Blank a scan's prediction when the classifier rejects it."""
    tp = fp = fn = 0
    dices = []
    for r in rows:
        if int(r["label"]) != 1:
            if keep.get(int(r["ts"]), True):
                fp += int(r["fp"])
            continue
        if keep.get(int(r["ts"]), True):
            tp += int(r["tp"])
            fp += int(r["fp"])
            fn += int(r["fn"])
            dices.append(float(r["dice"]))
        else:
            fn += int(r["tp"]) + int(r["fn"])
            dices.append(0.0)
    macro = float(np.mean(dices)) if dices else float("nan")
    glob = 2 * tp / (2 * tp + fp + fn)
    return macro, glob


def cascade(swin: dict) -> None:
    rows = {}
    for name in ("night_base_attunet9_s42", "unet_night_s42"):
        rows[name] = [r for r in json.loads((COMP / f"{name}_samples.json").read_text(encoding="utf-8"))["samples"]
                      if r["split"] == "test"]
    lines = ["model,gate,macro_dice,global_dice,kept_pos,rejected_pos"]
    for seg, recs in rows.items():
        base_m, base_g = _gated(recs, {})
        lines.append(f"{seg},none,{base_m:.4f},{base_g:.4f},{sum(int(r['label'])==1 for r in recs)},0")
        for cls_name, pack in swin.items():
            test = pack["splits"]["test"]
            thr = pack["threshold"]
            keep = {ts: sc >= thr for ts, sc in zip(test["ts"], test["score"])}
            macro, glob = _gated(recs, keep)
            pos = [ts for ts, y in zip(test["ts"], test["truth"]) if y == 1]
            rejected = sum(1 for ts in pos if not keep[ts])
            lines.append(f"{seg},{cls_name},{macro:.4f},{glob:.4f},{len(pos)-rejected},{rejected}")
            print(lines[-1])
    (COMP / "cascade_test.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pick_cases():
    attn = {int(r["ts"]): r for r in json.loads(
        (COMP / "night_base_attunet9_s42_samples.json").read_text(encoding="utf-8"))["samples"]
            if r["split"] == "test"}
    unet = {int(r["ts"]): r for r in json.loads(
        (COMP / "unet_night_s42_samples.json").read_text(encoding="utf-8"))["samples"]
            if r["split"] == "test"}
    pos = [ts for ts, r in attn.items() if int(r["label"]) == 1 and ts in unet]
    good = max(pos, key=lambda ts: 0.5 * (attn[ts]["dice"] + unet[ts]["dice"]))
    poor = min(pos, key=lambda ts: 0.5 * (attn[ts]["dice"] + unet[ts]["dice"]))
    neg = [ts for ts, r in attn.items() if int(r["label"]) == 0 and ts in unet]
    false = max(neg, key=lambda ts: attn[ts]["pred_area"] + unet[ts]["pred_area"])
    return [("Clear migration night", good), ("Missed or fragmented plume", poor),
            ("Quiet scan, false plume", false)]


def case_plates(device) -> None:
    cfg = json.loads((ROOT / "outputs" / "night_split" / "experiments"
                      / "night_base_attunet9_s42_config.json").read_text(encoding="utf-8"))
    manifest, norm = load_artifacts(cfg)
    picks = _pick_cases()
    wanted = {ts for _, ts in picks}
    rows = [r for r in manifest[manifest["split"] == "test"].to_dict("records")
            if int(r["timestamp"]) in wanted]
    order = [int(r["timestamp"]) for r in rows]
    loader = scene_loader(cfg, rows, norm)
    stacks = {}
    for i, ts in enumerate(order):
        x, y, _ = loader(i)
        stacks[ts] = (x, (np.asarray(y) > 0))

    preds = {}
    for name in ("attn", "unet"):
        full = f"{'night_base_attunet9' if name == 'attn' else 'unet_night'}_s42"
        mcfg = json.loads((ROOT / "outputs" / "night_split" / "experiments"
                           / f"{full}_config.json").read_text(encoding="utf-8"))
        model = create_model(mcfg).to(device)
        state = load_checkpoint(ROOT / "outputs" / "night_split" / "checkpoints" / f"{full}_best.pt", device)
        model.load_state_dict(state["model"])
        model.eval()
        preds[name] = {}
        for ts, (x, _) in stacks.items():
            prob = sliding_window_predict(model, x, device, patch_size=256, overlap=0.5)
            preds[name][ts] = prob >= THR
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fig, axes = plt.subplots(3, 4, figsize=(7.4, 6.6))
    plt.rcParams.update({"font.family": "serif", "font.size": 8})
    for i, (title, ts) in enumerate(picks):
        x, gt = stacks[ts]
        dbz = x[0] * 17.54166844119095 + 13.461288769556035
        valid = x[-1] > 0.5
        show = np.where(valid, dbz, np.nan)
        panels = [
            (show, "Reflectivity, lowest tilt", "turbo", -10, 40),
            (np.where(valid, gt, np.nan), "Label", "gray_r", 0, 1),
            (np.where(valid, preds["attn"][ts], np.nan), "Attention U-Net", "gray_r", 0, 1),
            (np.where(valid, preds["unet"][ts], np.nan), "U-Net", "gray_r", 0, 1),
        ]
        interest = (gt | preds["attn"][ts] | preds["unet"][ts] | (valid & (dbz > 5)))
        ys, xs = np.where(interest if interest.any() else valid)
        if len(ys):
            pad = 20
            r0, r1 = max(0, ys.min() - pad), min(valid.shape[0], ys.max() + pad)
            c0, c1 = max(0, xs.min() - pad), min(valid.shape[1], xs.max() + pad)
        else:
            r0, r1, c0, c1 = 0, valid.shape[0], 0, valid.shape[1]
        for j, (img, lab, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img[r0:r1, c0:c1], cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(lab, fontsize=8)
            if j == 0:
                ax.set_ylabel(title, fontsize=7)
        print(title, ts)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_cases.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig7_cases.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-only", action="store_true")
    args = ap.parse_args()
    device = _device()
    print("device", device, flush=True)
    if args.cases_only:
        case_plates(device)
    else:
        swin = score_swin(device)
        cascade(swin)
        case_plates(device)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
