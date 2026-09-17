"""Two-model cascade: scan classifier then segmenter.

Hard gate (cascade-equivalent):
  if p_cls < t_cls -> zeros (quiet scan)
  else             -> unscaled segmenter map (no plume shrink)

Compares against the one-model hard-gated Attention U-Net and against
S0 ungated on the same balanced val split. Test stays deferred.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import channels, checkpoint as ckpt, config as cfgmod, data_prep, engine, paths
from .models import create_model


def _load_done(base_path: str, experiments_path: str, name: str):
    base = cfgmod.load_base_config(base_path)
    matches = [e for e in cfgmod.load_experiments(experiments_path) if e["name"] == name]
    if not matches:
        raise SystemExit(f"experiment '{name}' not found in {experiments_path}")
    cfg = cfgmod.resolve_experiment(base, matches[0])
    exp_dir = paths.experiments_dir(cfg)
    if not ckpt.is_done(exp_dir, name):
        return cfg, None, None
    return cfg, ckpt.load_result(exp_dir, name), ckpt.best_path(paths.checkpoint_dir(cfg), name)


def _restore(cfg, ckpt_path, device):
    model = create_model(cfg).to(device)
    state = ckpt.load_checkpoint(ckpt_path, device)
    if state is None:
        raise FileNotFoundError(ckpt_path)
    model.load_state_dict(state["model"])
    model.eval()
    return model


SWEEP_T = [
    0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10,
    0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
]
DICE_BUDGET = 0.01
CACHE_PATH = Path("outputs/night_cascade/cache/s0_val.npz")


@torch.no_grad()
def _cls_probs(model, rows: List[dict], cfg, norm_stats, device) -> List[float]:
    size = int(cfg["model"].get("img_size", 384))
    use_amp = device.type == "cuda"
    use_classify = bool(getattr(model, "cls_head_enabled", False))
    scores = []
    for row in rows:
        x = channels.build_stack(cfg, row["x_path"], cfg["channels"], norm_stats)
        t = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0).to(device)
        t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            z = model.classify(t) if use_classify else model(t)
        scores.append(float(torch.sigmoid(z.float()).reshape(-1)[0].item()))
    return scores


def _save_items(items: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        label=np.asarray([it["label"] for it in items], dtype=np.int16),
        y=np.stack([it["y"] for it in items]).astype(np.uint8),
        prob=np.stack([it["prob"] for it in items]).astype(np.float16),
    )


def _load_items(path: Path) -> List[dict]:
    z = np.load(path)
    items = []
    for i in range(len(z["label"])):
        items.append({
            "label": int(z["label"][i]),
            "y": z["y"][i].astype(np.float32),
            "prob": z["prob"][i].astype(np.float32),
            "p_cls": 1.0,
        })
    return items


def _collect_s0(seg_cfg, seg_ckpt, val_rows, norm_stats, device, tta, verbose=True):
    if CACHE_PATH.exists():
        if verbose:
            print(f"[seg] loading cached S0 maps from {CACHE_PATH}", flush=True)
        return _load_items(CACHE_PATH)
    if verbose:
        print(f"[seg] collecting sliding-window maps ({len(val_rows)} val scenes) ...",
              flush=True)
    seg_model = _restore(seg_cfg, seg_ckpt, device)
    items = engine.collect_scene_cache(seg_model, val_rows, seg_cfg, norm_stats,
                                       device, tta=tta)
    del seg_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if verbose:
        print(f"[seg] writing cache {CACHE_PATH}", flush=True)
    _save_items(items, CACHE_PATH)
    return items


def _bind_scores(items: List[dict], scores: List[float]) -> List[dict]:
    out = []
    for it, p in zip(items, scores):
        row = dict(it)
        row["p_cls"] = float(p)
        out.append(row)
    return out


def _sweep_gate(items: List[dict], seg_t: float, grid) -> pd.DataFrame:
    rows = []
    for t in grid:
        m = engine.metrics_from_cache(items, seg_t, float(t), gate="hard")
        rows.append({
            "cls_threshold": float(t),
            "dice": m.get("dice"),
            "iou": m.get("iou"),
            "precision": m.get("precision"),
            "recall": m.get("recall"),
            "bg_fp_rate": m.get("bg_fp_rate"),
            "scan_auroc": m.get("scan_auroc"),
            "scan_f1": m.get("scan_f1"),
            "scan_precision": m.get("scan_precision"),
            "scan_recall": m.get("scan_recall"),
            "n_pos": m.get("n_pos_scenes"),
            "n_neg": m.get("n_neg_scenes"),
        })
    return pd.DataFrame(rows)


def _pick_operating_point(sweep: pd.DataFrame, dice_ref: float, budget: float = DICE_BUDGET):
    """Highest t whose Dice stays within ``budget`` of the ungated segmenter."""
    floor = dice_ref - budget
    keep = sweep.loc[sweep["dice"] >= floor]
    if keep.empty:
        # Stay as close as possible to the budget; prefer higher Dice, then higher t.
        return sweep.sort_values(["dice", "cls_threshold"], ascending=[False, False]).iloc[0]
    return keep.sort_values("cls_threshold", ascending=False).iloc[0]


def _recall_row(sweep: pd.DataFrame, target: float):
    hit = sweep.loc[sweep["scan_recall"] >= target]
    if hit.empty:
        return None
    return hit.sort_values("cls_threshold", ascending=False).iloc[0]


def _row(name: str, family: str, metrics: Dict, extra: Optional[Dict] = None) -> Dict:
    out = {
        "name": name,
        "family": family,
        "dice": metrics.get("dice"),
        "iou": metrics.get("iou"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "bg_fp_rate": metrics.get("bg_fp_rate"),
        "scan_auroc": metrics.get("scan_auroc"),
        "scan_f1": metrics.get("scan_f1"),
        "scan_precision": metrics.get("scan_precision"),
        "scan_recall": metrics.get("scan_recall"),
        "seg_threshold": metrics.get("threshold"),
        "cls_threshold": metrics.get("cls_threshold"),
        "n_pos": metrics.get("n_pos_scenes"),
        "n_neg": metrics.get("n_neg_scenes"),
    }
    if extra:
        out.update(extra)
    return out


def evaluate(seg_base: str, seg_exps: str, seg_name: str,
             cls_base: str, cls_exps: str, cls_name: str,
             one_base: str, one_exps: str, one_name: str,
             verbose: bool = True) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose and device.type == "cuda":
        print(f"[device] cuda ({torch.cuda.get_device_name(0)})", flush=True)

    seg_cfg, seg_res, seg_ckpt = _load_done(seg_base, seg_exps, seg_name)
    if seg_res is None:
        raise SystemExit(f"segmenter '{seg_name}' is not finished")
    manifest, norm_stats = data_prep.load_artifacts(seg_cfg)
    val_rows = manifest[manifest["split"] == "val"].to_dict("records")
    seg_t = float(seg_res["calibrated_threshold"])
    items = _collect_s0(
        seg_cfg, seg_ckpt, val_rows, norm_stats, device,
        tta=bool(seg_cfg["eval"].get("tta", False)), verbose=verbose)

    rows = [_row(f"{seg_name}_ungated", "segmentation",
                 engine.metrics_from_cache(items, seg_t, 0.0, gate="off"))]

    cls_cfg, cls_res, cls_ckpt = _load_done(cls_base, cls_exps, cls_name)
    if cls_res is None:
        # Fall back to the original (unbalanced-val) Swin-Tiny so a cascade
        # number still exists if the balanced retrain has not finished.
        alt_base, alt_exps, alt_name = (
            "configs/base_config_cls.yaml",
            "configs/experiments_cls.yaml",
            "cls_swin_tiny",
        )
        if verbose:
            print(f"[cls] {cls_name} not finished -> fallback {alt_name}", flush=True)
        cls_cfg, cls_res, cls_ckpt = _load_done(alt_base, alt_exps, alt_name)
        cls_name = alt_name
    if cls_res is not None:
        if verbose:
            print(f"[cls] {cls_name} scoring {len(val_rows)} val scans ...", flush=True)
        cls_model = _restore(cls_cfg, cls_ckpt, device)
        probs = _cls_probs(cls_model, val_rows, cls_cfg, norm_stats, device)
        del cls_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for it, p in zip(items, probs):
            it["p_cls"] = float(p)
        youden_t = float(cls_res["val"]["threshold"])
        dice_t = engine.calibrate_cls_threshold(
            items, seg_t, cls_cfg["eval"].get("threshold_range",
                                              [i / 10 for i in range(2, 10)]))
        rows.append(_row(f"cascade_{cls_name}+{seg_name}_youden", "cascade",
                         engine.metrics_from_cache(items, seg_t, youden_t, gate="hard"),
                         extra={"note": "cls cutoff = Youden J"}))
        rows.append(_row(f"cascade_{cls_name}+{seg_name}_dice", "cascade",
                         engine.metrics_from_cache(items, seg_t, dice_t, gate="hard"),
                         extra={"note": "cls cutoff max val Dice"}))
        # Classifier-only row (scan metrics; pixel maps unused).
        cm = engine.metrics_from_cache(items, seg_t, youden_t, gate="hard")
        rows.append(_row(cls_name, "classifier", cm,
                         extra={"note": "scan metrics; pixel maps are gated S0"}))

    one_cfg, one_res, one_ckpt = _load_done(one_base, one_exps, one_name)
    if one_res is not None:
        vf = one_res.get("val_full_scene") or {}
        ug = one_res.get("val_ungated") or {}
        if vf:
            rows.append(_row(one_name, "one_model_hard", vf,
                             extra={"note": "hard gate, Kendall UW, S0 init"}))
        if ug:
            rows.append(_row(f"{one_name}_ungated", "one_model_ungated", ug))
    elif verbose:
        print(f"[one] {one_name} not finished (skip)", flush=True)

    df = pd.DataFrame(rows)
    out = Path("outputs/night_cascade")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cascade_comparison.csv", index=False)
    md = out / "cascade_comparison.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Cascade comparison — night-split val (test deferred)\n\n")
        f.write("Hard gate: if `p_cls < t` the scan is all zeros, otherwise the ")
        f.write("segmenter map is left unscaled. Soft multiply is not used.\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    if verbose:
        print(df.to_string(index=False))
        print(f"\n[table] {md}", flush=True)
    return df


def sweep(seg_base: str, seg_exps: str, seg_name: str,
          cls_base: str, cls_exps: str, cls_name: str,
          frozen_base: str, frozen_exps: str, frozen_name: str,
          verbose: bool = True) -> None:
    """High-recall hard-gate sweep for every available scan score on S0 maps."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose and device.type == "cuda":
        print(f"[device] cuda ({torch.cuda.get_device_name(0)})", flush=True)

    seg_cfg, seg_res, seg_ckpt = _load_done(seg_base, seg_exps, seg_name)
    if seg_res is None:
        raise SystemExit(f"segmenter '{seg_name}' is not finished")
    manifest, norm_stats = data_prep.load_artifacts(seg_cfg)
    val_rows = manifest[manifest["split"] == "val"].to_dict("records")
    seg_t = float(seg_res["calibrated_threshold"])
    items = _collect_s0(
        seg_cfg, seg_ckpt, val_rows, norm_stats, device,
        tta=bool(seg_cfg["eval"].get("tta", False)), verbose=verbose)

    ungated = engine.metrics_from_cache(items, seg_t, 0.0, gate="off")
    dice_ref = float(ungated["dice"])
    if verbose:
        print(f"[seg] ungated Dice={dice_ref:.4f} bg_fp={ungated['bg_fp_rate']:.4f} "
              f"(budget {DICE_BUDGET:.2f})", flush=True)

    classifiers = []
    cls_cfg, cls_res, cls_ckpt = _load_done(cls_base, cls_exps, cls_name)
    if cls_res is not None:
        classifiers.append((cls_name, "two_model_swin", cls_cfg, cls_ckpt))
    fr_cfg, fr_res, fr_ckpt = _load_done(frozen_base, frozen_exps, frozen_name)
    if fr_res is not None:
        classifiers.append((frozen_name, "one_model_frozen_head", fr_cfg, fr_ckpt))
    elif verbose:
        print(f"[frozen] {frozen_name} not finished (skip)", flush=True)

    out = Path("outputs/night_cascade")
    out.mkdir(parents=True, exist_ok=True)
    summary_rows = [{
        "name": f"{seg_name}_ungated",
        "family": "segmentation",
        "cls_threshold": 0.0,
        "dice": ungated["dice"],
        "bg_fp_rate": ungated["bg_fp_rate"],
        "scan_recall": 1.0,
        "scan_auroc": float("nan"),
        "note": "S0 maps, no gate",
    }]
    md_parts = [
        "# Cascade threshold sweep — night-split val (test deferred)\n\n",
        "Hard gate: if `p_cls < t` the scan is zeros, otherwise the S0 map is unscaled.\n",
        f"Ungated S0 Dice = **{dice_ref:.4f}**. Operating point = highest `t` with "
        f"Dice ≥ {dice_ref:.4f} − {DICE_BUDGET:.2f} = **{dice_ref - DICE_BUDGET:.4f}**.\n\n",
    ]

    for name, family, cfg, ckpt_path in classifiers:
        if verbose:
            print(f"[cls] {name} scoring {len(val_rows)} val scans ...", flush=True)
        model = _restore(cfg, ckpt_path, device)
        scores = _cls_probs(model, val_rows, cfg, norm_stats, device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        bound = _bind_scores(items, scores)
        table = _sweep_gate(bound, seg_t, SWEEP_T)
        table.insert(0, "classifier", name)
        table.to_csv(out / f"sweep_{name}.csv", index=False)
        op = _pick_operating_point(table, dice_ref)
        rec_rows = []
        for tgt in (0.95, 0.97, 0.99):
            r = _recall_row(table, tgt)
            if r is not None:
                rec_rows.append((tgt, r))
        summary_rows.append({
            "name": f"cascade_{name}_op",
            "family": family,
            "cls_threshold": float(op["cls_threshold"]),
            "dice": float(op["dice"]),
            "bg_fp_rate": float(op["bg_fp_rate"]),
            "scan_recall": float(op["scan_recall"]),
            "scan_auroc": float(op["scan_auroc"]),
            "note": f"highest t with Dice within {DICE_BUDGET:.2f} of S0",
        })
        md_parts.append(f"## {name}\n\n")
        md_parts.append(table.to_markdown(index=False, floatfmt=".4f"))
        md_parts.append("\n\n")
        md_parts.append(
            f"**Operating point:** t = {float(op['cls_threshold']):.3f} → "
            f"Dice {float(op['dice']):.4f}, bg_fp {float(op['bg_fp_rate']):.4f}, "
            f"scan recall {float(op['scan_recall']):.4f}, "
            f"scan AUROC {float(op['scan_auroc']):.4f}.\n\n"
        )
        if rec_rows:
            md_parts.append("High-recall cutoffs (highest t meeting the recall floor):\n\n")
            for tgt, r in rec_rows:
                md_parts.append(
                    f"- recall ≥ {tgt:.2f}: t = {float(r['cls_threshold']):.3f}, "
                    f"Dice {float(r['dice']):.4f}, bg_fp {float(r['bg_fp_rate']):.4f}, "
                    f"scan recall {float(r['scan_recall']):.4f}\n"
                )
            md_parts.append("\n")
        if verbose:
            print(table.to_string(index=False))
            print(f"[op] {name} t={float(op['cls_threshold']):.3f} "
                  f"dice={float(op['dice']):.4f} bg_fp={float(op['bg_fp_rate']):.4f} "
                  f"scan_rec={float(op['scan_recall']):.4f}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "cascade_sweep_summary.csv", index=False)
    md_parts.insert(4, "## Operating points\n\n")
    md_parts.insert(5, summary.to_markdown(index=False, floatfmt=".4f"))
    md_parts.insert(6, "\n\n")
    md = out / "cascade_sweep.md"
    md.write_text("".join(md_parts), encoding="utf-8")
    if verbose:
        print(f"\n[table] {md}", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Evaluate the two-model cascade")
    ap.add_argument("--seg-base", default="configs/base_config_night.yaml")
    ap.add_argument("--seg-experiments", default="configs/experiments_night.yaml")
    ap.add_argument("--seg-name", default="night_base_attunet9")
    ap.add_argument("--cls-base", default="configs/base_config_cascade_cls.yaml")
    ap.add_argument("--cls-experiments", default="configs/experiments_cascade_cls.yaml")
    ap.add_argument("--cls-name", default="cls_swin_tiny_bal")
    ap.add_argument("--one-base", default="configs/base_config_cascade.yaml")
    ap.add_argument("--one-experiments", default="configs/experiments_cascade.yaml")
    ap.add_argument("--one-name", default="gated_attn_unet")
    ap.add_argument("--frozen-base", default="configs/base_config_frozen_cls.yaml")
    ap.add_argument("--frozen-experiments", default="configs/experiments_frozen_cls.yaml")
    ap.add_argument("--frozen-name", default="frozen_s0_cls")
    ap.add_argument("--sweep", action="store_true",
                    help="high-recall threshold sweep on cached S0 maps")
    args = ap.parse_args()
    if args.sweep:
        sweep(args.seg_base, args.seg_experiments, args.seg_name,
              args.cls_base, args.cls_experiments, args.cls_name,
              args.frozen_base, args.frozen_experiments, args.frozen_name)
    else:
        evaluate(args.seg_base, args.seg_experiments, args.seg_name,
                 args.cls_base, args.cls_experiments, args.cls_name,
                 args.one_base, args.one_experiments, args.one_name)


if __name__ == "__main__":
    main()
