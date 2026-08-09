"""Re-evaluate finished experiments from their best checkpoints — no retraining.

Loads each ``<name>_best.pt``, runs full-scene evaluation on the test split, and
writes an enriched comparison table with BOTH macro (mean-per-scene) and micro
(pixel-pooled) metrics. Useful after changing the metric/eval code without
wanting to repeat training. Non-destructive: it does not overwrite
``*_result.json`` or the training-time comparison table.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from . import checkpoint as ckpt
from . import config as cfgmod
from . import data_prep, dataset, engine, paths
from . import metrics as metrics_mod
from .models import create_model
from .run import REFERENCE_BASELINES


def reevaluate(base_config_path: str, experiments_path: str, split: str = "test",
               tta: bool = False, verbose: bool = True) -> pd.DataFrame:
    base_cfg = cfgmod.load_base_config(base_config_path)
    experiments = cfgmod.load_experiments(experiments_path)
    manifest, norm_stats = data_prep.load_artifacts(base_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = paths.checkpoint_dir(base_cfg)
    exp_dir = paths.experiments_dir(base_cfg)
    rows_split = manifest[manifest["split"] == split].to_dict("records")

    out_rows: List[Dict] = []
    for exp in experiments:
        cfg = cfgmod.resolve_experiment(base_cfg, exp)
        name = cfg["name"]
        best = ckpt.load_checkpoint(ckpt.best_path(ckpt_dir, name), device)
        if best is None:
            if verbose:
                print(f"[skip] {name}: no best checkpoint")
            continue
        # reuse the calibrated threshold from the training result if available
        t = 0.5
        rp = ckpt.result_path(exp_dir, name)
        if rp.exists():
            t = json.load(open(rp))["calibrated_threshold"]
        model = create_model(cfg).to(device)
        model.load_state_dict(best["model"])
        m = engine.evaluate_full_scene(model, rows_split, cfg, norm_stats, device,
                                       threshold=t, tta=tta)
        if verbose:
            print(f"[eval] {name}: macro_dice={m['dice']:.4f}  micro_dice={m['dice_micro']:.4f}  "
                  f"bIoU={m.get('boundary_iou', float('nan')):.3f} NSD={m.get('nsd', float('nan')):.3f}  "
                  f"prec={m['precision']:.3f} rec={m['recall']:.3f} (t={t})")
        out_rows.append({
            "experiment": name, "model": cfg["model"]["name"], "loss": cfg["loss"]["name"],
            "threshold": t,
            "dice_macro": round(m["dice"], 4), "dice_micro": round(m["dice_micro"], 4),
            "iou_macro": round(m["iou"], 4),
            "boundary_iou": (round(m["boundary_iou"], 4) if "boundary_iou" in m else None),
            "nsd": (round(m["nsd"], 4) if "nsd" in m else None),
            "hd95": (round(m["hd95"], 2) if "hd95" in m else None),
            "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
            "bg_fp_rate": (round(m["bg_fp_rate"], 5) if m["bg_fp_rate"] == m["bg_fp_rate"] else None),
            "n_pos_scenes": m["n_pos_scenes"], "source": "this_framework",
        })

    for b in REFERENCE_BASELINES:
        out_rows.append({
            "experiment": b["name"], "model": b["model"], "loss": b["loss"], "threshold": None,
            "dice_macro": b["test_dice"], "dice_micro": None,
            "iou_macro": b["test_iou"], "iou_micro": None,
            "precision": None, "recall": None, "bg_fp_rate": None,
            "n_pos_scenes": None, "source": b["source"] + " (macro, ~10 scenes)",
        })

    df = pd.DataFrame(out_rows).sort_values("dice_macro", ascending=False, na_position="last").reset_index(drop=True)
    out = paths.output_dir(base_cfg)
    df.to_csv(out / f"comparison_table_reeval_{split}.csv", index=False)
    with open(out / f"comparison_table_reeval_{split}.md", "w", encoding="utf-8") as f:
        f.write(f"# Re-evaluation on {split} split (macro + micro), TTA={tta}\n\n")
        f.write("`dice_macro` = mean per-scene Dice (headline, comparable to the notebook). "
                "`dice_micro` = pixel-pooled Dice (comparable to per-epoch patch val_dice). "
                "`REF:` rows are the reference-notebook macro numbers over ~10 scenes.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    if verbose:
        print("\n" + df.to_string(index=False))
    return df


def ensemble_evaluate(base_config_path: str, experiments_path: str, names=None,
                      split: str = "test", tta: bool = False, verbose: bool = True) -> Dict:
    """Average sigmoid probabilities across several finished models (no retraining).

    Ensembling reduces per-scene variance and reliably lifts macro Dice a few
    points. Only experiments sharing the same input channels can be ensembled
    (their inputs must match); the base config's channel list is used.
    """
    base_cfg = cfgmod.load_base_config(base_config_path)
    experiments = cfgmod.load_experiments(experiments_path)
    manifest, norm_stats = data_prep.load_artifacts(base_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = paths.checkpoint_dir(base_cfg)

    # Members must share input channels AND target so a single loaded scene
    # (its x and y) is valid for every model. The first selected member fixes the
    # channel/target set used to load scenes; the rest must match it.
    members = []
    load_cfg = None
    for exp in experiments:
        cfg = cfgmod.resolve_experiment(base_cfg, exp)
        if names and cfg["name"] not in names:
            continue
        if load_cfg is None:
            load_cfg = cfg
        elif cfg["channels"] != load_cfg["channels"] or cfg["target"] != load_cfg["target"]:
            raise RuntimeError(
                f"ensemble members must share channels and target; '{cfg['name']}' "
                f"differs from '{load_cfg['name']}'")
        best = ckpt.load_checkpoint(ckpt.best_path(ckpt_dir, cfg["name"]), device)
        if best is None:
            if verbose:
                print(f"[skip] {cfg['name']}: no best checkpoint")
            continue
        m = create_model(cfg).to(device)
        m.load_state_dict(best["model"])
        m.eval()
        members.append((cfg["name"], m))
    if not members:
        raise RuntimeError("No finished checkpoints to ensemble (check --names)")

    ps = int(load_cfg["patch"]["size"])
    ov = float(load_cfg["eval"].get("overlap", load_cfg["eval"].get("sliding_overlap", 0.5)))
    gaussian = bool(load_cfg["eval"].get("gaussian_window", True))
    tau = float(load_cfg["eval"].get("nsd_tolerance", 2.0))
    thresholds = base_cfg["eval"].get("threshold_range", [0.2, 0.3, 0.4, 0.45, 0.5, 0.6])
    from .metrics import compute_metrics

    @torch.no_grad()
    def ensemble_prob(x):
        return np.mean([engine.sliding_window_predict(m, x, device, ps, ov, tta, gaussian=gaussian)
                        for _, m in members], axis=0)

    # calibrate the threshold on positive val scenes (never on test)
    val_pos = manifest[(manifest.split == "val") & (manifest.label == 1)].to_dict("records")
    calib_max = base_cfg["eval"].get("calib_max_scenes")
    if calib_max:
        val_pos = val_pos[:calib_max]
    vp = []
    for r in val_pos:
        x, y = dataset.load_full_scene(load_cfg, r, norm_stats)
        vp.append((ensemble_prob(x), y))
    best_t = max(thresholds, key=lambda t: np.mean([compute_metrics(p > t, y)["dice"] for p, y in vp]))

    rows = manifest[manifest["split"] == split].to_dict("records")
    pos_m, bg = [], []
    TP = FP = FN = 0.0
    for r in rows:
        x, y = dataset.load_full_scene(load_cfg, r, norm_stats)
        pred = ensemble_prob(x) > best_t
        if int(r["label"]) == 1:
            m = compute_metrics(pred, y)
            m["boundary_iou"] = metrics_mod.boundary_iou(pred, y)
            m.update(metrics_mod.surface_metrics(pred, y, tau=tau))
            pos_m.append(m)
            p = pred.reshape(-1).astype(np.float64); t = y.reshape(-1).astype(np.float64)
            TP += float((p * t).sum()); FP += float((p * (1 - t)).sum()); FN += float(((1 - p) * t).sum())
        else:
            bg.append(float(pred.mean()))
    avg = lambda k: float(np.nanmean([m[k] for m in pos_m])) if pos_m else float("nan")
    out = {"members": [n for n, _ in members], "tta": bool(tta), "threshold": float(best_t),
           "dice_macro": avg("dice"), "dice_micro": 2 * TP / (2 * TP + FP + FN + 1e-8),
           "iou_macro": avg("iou"), "precision": avg("precision"), "recall": avg("recall"),
           "boundary_iou": avg("boundary_iou"), "nsd": avg("nsd"),
           "hd95": avg("hd95"), "assd": avg("assd"),
           "bg_fp_rate": float(np.mean(bg)) if bg else float("nan"),
           "n_pos_scenes": len(pos_m)}
    if verbose:
        print(f"[ensemble] {len(members)} members {out['members']} (tta={tta})")
        print(f"[ensemble] {split}: macro_dice={out['dice_macro']:.4f} micro_dice={out['dice_micro']:.4f} "
              f"bIoU={out['boundary_iou']:.3f} NSD={out['nsd']:.3f} "
              f"prec={out['precision']:.3f} rec={out['recall']:.3f} (t={best_t})")
    with open(paths.output_dir(base_cfg) / f"ensemble_{split}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Re-evaluate best checkpoints (no retraining)")
    ap.add_argument("--base-config", default="configs/base_config.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--ensemble", action="store_true", help="average sigmoid probs across members")
    ap.add_argument("--names", default=None,
                    help="comma-separated experiment names to include (ensemble members)")
    args = ap.parse_args()
    names = [s.strip() for s in args.names.split(",")] if args.names else None
    if args.ensemble:
        ensemble_evaluate(args.base_config, args.experiments, names=names,
                          split=args.split, tta=args.tta)
    else:
        reevaluate(args.base_config, args.experiments, split=args.split, tta=args.tta)
