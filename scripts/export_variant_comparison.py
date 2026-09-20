"""Compare night-split ablation variants on the full dashboard metric set.

For each variant this runs full-scene inference over val+test, assembles the
``samples.json`` / ``dataset.json`` shaped documents the dashboard uses, and
feeds them to ``src.presence.analyze_presence`` UNCHANGED. Reusing that module
rather than reimplementing it is the point: scan/night ROC-AUC, Youden cutoffs,
confusion matrices and Mann-Whitney statistics are then computed by exactly the
code that produced the published numbers.

Nothing here needs the website checkout, the SBW1 packs or the raw PPI volumes.

    .venv\\Scripts\\python.exe scripts\\export_variant_comparison.py \
        --base-config configs\\base_config_night.yaml \
        --experiments configs\\experiments_night.yaml

Writes to ``outputs/night_split/comparison/``:
  <name>_samples.json     per-scene records for one variant
  presence_<name>.json    full presence analysis for one variant
  dataset.json            shared scene/grid document
  comparison.{csv,md}     segmentation + presence metrics, all variants
  classification.{csv,md} auxiliary classifier metrics, reported separately

Segmentation metrics come from ``engine.evaluate_full_scene``; presence metrics
are derived from segmentation ``pred_area`` so every variant is measured the
same way. Auxiliary classifier scores are kept in a separate table precisely so
they cannot be mistaken for the headline presence result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401  (adds the project root to sys.path)

from src import checkpoint as ckpt
from src import config as cfgmod
from src import data_prep, engine, metrics as metrics_mod, paths
from src.finalize import load_best_or_final
from src.models import create_model
from src.channels import GRID
from src.presence import analyze_presence
from src.stats import paired_cluster_bootstrap, wilcoxon_paired
SEG_COLUMNS = [
    ("dice", "dice_macro"), ("dice_micro", "dice_micro"),
    # Global = pixel-pooled over ALL scans; the only overlap metric a presence
    # gate can move, and the paper's integrated headline.
    ("dice_global", "dice_global"),
    ("iou", "iou_macro"), ("iou_micro", "iou_micro"),
    ("precision", "precision"), ("recall", "recall"),
    ("boundary_iou", "boundary_iou"), ("nsd", "nsd"),
    ("bf1", "bf1"), ("bf1_fuzzy", "bf1_fuzzy"),
    ("hd95", "hd95"), ("assd", "assd"), ("bg_fp_rate", "bg_fp_rate"),
    ("far_scan", "far_scan"), ("sensitivity_retained", "sensitivity_retained"),
]


def _r(x, nd=4):
    """Round a scalar for JSON; None for anything not a finite number.

    Must tolerate non-scalars: ``metrics.surface_metrics`` returns ``nsd_curve``
    as a nested dict alongside its scalars, and callers map this over whole
    metric dicts.
    """
    if x is None or isinstance(x, (dict, list, tuple, set)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def json_safe(obj):
    """Recursively replace non-finite floats with None so JSON stays strict.

    A degenerate variant (constant predicted areas) makes scipy return a NaN
    Mann-Whitney p-value, which ``json.dumps(allow_nan=False)`` rejects. Emitting
    null matches how presence.py already reports undefined statistics, and keeps
    NaN from silently becoming the invalid literal ``NaN`` in the output.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, (np.floating, np.integer)):
        return json_safe(float(obj))
    return obj


def dataset_document(manifest: pd.DataFrame) -> Dict:
    """``dataset.json``-shaped document: every manifest scan with its dbZ>=0 area."""
    scenes = []
    for row in manifest.itertuples():
        area = None
        if int(row.label) == 1 and row.target_path:
            with np.load(row.target_path) as z:
                dbz = z[z.files[0]]
            area = int((np.isfinite(dbz) & (dbz >= 0.0)).sum())
        scenes.append({"ts": int(row.timestamp), "year": int(row.year),
                       "split": str(row.split), "label": int(row.label),
                       "night": str(row.night), "area": area})
    return {"grid": GRID, "scenes": scenes}


def score_variant(cfg: Dict, manifest: pd.DataFrame, norm_stats: Dict, device,
                  threshold: float, key: str, verbose: bool = True):
    """Per-scene records for one variant over val+test, plus split metrics.

    Runs one forward pass per scene and derives everything from it, so the
    segmentation metrics and the presence areas can never disagree.
    """
    state, weights = load_best_or_final(paths.checkpoint_dir(cfg), cfg["name"], device)
    if state is None:
        return None, None
    if weights == "final" and verbose:
        print(f"  [warn] {cfg['name']}: no best checkpoint; using final weights")
    model = create_model(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    ps = int(cfg["patch"]["size"])
    ov = float(cfg["eval"].get("overlap", 0.5))
    gaussian = bool(cfg["eval"].get("gaussian_window", True))
    tau = float(cfg["eval"].get("nsd_tolerance", 2.0))
    tta = bool(cfg["eval"].get("tta", False))

    samples: List[Dict] = []
    cls_scores: List[Dict] = []
    per_split: Dict[str, Dict] = {}
    for split in ("val", "test"):
        rows = manifest[manifest["split"] == split].to_dict("records")
        load = engine.scene_loader(cfg, rows, norm_stats)
        pos_metrics, bg_fp = [], []
        for i, row in enumerate(rows):
            x, y, pad = load(i)
            out = engine.sliding_window_predict(
                model, x, device, ps, ov, tta, gaussian=gaussian,
                pad_mask=pad, return_cls=True)
            prob, cls_prob = out
            pred = prob > threshold
            truth = y.astype(bool)
            tp = int((pred & truth).sum()); fp = int((pred & ~truth).sum())
            fn = int((~pred & truth).sum()); tn = int((~pred & ~truth).sum())
            is_pos = int(row["label"]) == 1

            record = {"ts": int(row["timestamp"]), "split": split,
                      "label": int(row["label"]), "year": int(row["year"]),
                      "night": str(row["night"]), "thr": threshold,
                      "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                      "gt_area": int(truth.sum()), "pred_area": int(pred.sum())}
            model_block = {"pred_area": int(pred.sum())}
            if is_pos:
                m = metrics_mod.compute_metrics(pred, y)
                m["boundary_iou"] = metrics_mod.boundary_iou(pred, y)
                m.update(metrics_mod.surface_metrics(pred, y, tau=tau))
                pos_metrics.append(m)
                # Scalars only: nsd_curve is a nested dict and is kept verbatim
                # (its lists are already JSON-safe) so the NSD-vs-distance figure
                # can be built with a night-clustered band.
                record.update({k: _r(v) for k, v in m.items()
                               if not isinstance(v, dict)})
                if isinstance(m.get("nsd_curve"), dict):
                    record["nsd_curve"] = m["nsd_curve"]
                model_block.update({k: _r(m.get(k)) for k in
                                    ("dice", "iou", "precision", "recall",
                                     "boundary_iou", "nsd", "bf1", "bf1_fuzzy")})
                model_block.update({"tp": tp, "fp": fp, "fn": fn})
            else:
                rate = _r(float(pred.mean()), 6)
                bg_fp.append(float(pred.mean()))
                record["bg_fp_rate"] = rate
                model_block["bg_fp_rate"] = rate
            record["models"] = {key: model_block}
            samples.append(record)
            if cls_prob is not None:
                # MIL max-pooling: bag (scan) score = max over instance (window) scores.
                cls_scores.append({"ts": int(row["timestamp"]), "split": split,
                                   "truth": int(truth.sum() > 0), "score": float(cls_prob)})
            if verbose and (i + 1) % 50 == 0:
                print(f"    {cfg['name']} {split} {i + 1}/{len(rows)}")

        keys = ["dice", "iou", "precision", "recall", "f1", "accuracy",
                "boundary_iou", "nsd", "hd95", "assd", "bf1", "bf1_fuzzy"]
        block = {k: (float(np.nanmean([m[k] for m in pos_metrics])) if pos_metrics else float("nan"))
                 for k in keys}
        rows_split = [s for s in samples if s["split"] == split]
        pos_rows = [s for s in rows_split if s["label"] == 1]
        neg_rows = [s for s in rows_split if s["label"] == 0]
        TP = sum(s["tp"] for s in pos_rows)
        FP = sum(s["fp"] for s in pos_rows)
        FN = sum(s["fn"] for s in pos_rows)
        block["dice_micro"] = 2 * TP / (2 * TP + FP + FN + 1e-8)
        block["iou_micro"] = TP / (TP + FP + FN + 1e-8)
        # Pixel-pooled over ALL scans: quiet scans contribute false positives, so
        # this is the only overlap metric a presence gate can move. dice_micro
        # stays positives-only so existing numbers do not shift.
        FP_all = sum(s["fp"] for s in rows_split)
        block["dice_global"] = 2 * TP / (2 * TP + FP_all + FN + 1e-8)
        block["iou_global"] = TP / (TP + FP_all + FN + 1e-8)
        block["bg_fp_rate"] = float(np.mean(bg_fp)) if bg_fp else float("nan")
        # Operational false-alarm rate at the frozen minimum swarm area.
        a_km2 = float(cfg["eval"].get("far_min_area_km2", 25.0))
        a_cells = max(1, int(round(metrics_mod.km2_to_cells(a_km2))))
        block["far_min_area_km2"] = a_km2
        block["far_scan"] = (sum(1 for s in neg_rows if s["pred_area"] >= a_cells)
                             / len(neg_rows)) if neg_rows else float("nan")
        block["sensitivity_retained"] = (
            sum(1 for s in pos_rows if s["pred_area"] >= a_cells) / len(pos_rows)
        ) if pos_rows else float("nan")
        curves = [m["nsd_curve"] for m in pos_metrics if isinstance(m.get("nsd_curve"), dict)]
        if curves:
            block["nsd_curve"] = {
                "taus_px": curves[0]["taus_px"],
                "taus_km": curves[0]["taus_km"],
                "nsd": [float(v) for v in
                        np.nanmean(np.asarray([c["nsd"] for c in curves], float), axis=0)],
            }
        block["n_pos_scenes"] = len(pos_rows)
        block["n_neg_scenes"] = len(neg_rows)
        per_split[split] = block
    return samples, {"per_split": per_split, "cls_scores": cls_scores}


def paired_table(samples_a: List[Dict], samples_b: List[Dict],
                 name_a: str, name_b: str,
                 keys=("dice", "iou", "precision", "recall", "nsd", "bf1_fuzzy"),
                 split: str = "test") -> pd.DataFrame:
    """Night-clustered paired comparison on positive scans sharing a timestamp.

    ``split`` defaults to ``test``: thresholds and the model choice were selected
    on validation, so pooling val into the confirmatory interval would report a
    partly in-sample effect. Pass ``split=None`` only for exploratory views.
    """
    def _keep(s):
        return int(s.get("label", 0)) == 1 and (split is None or s.get("split") == split)

    by_a = {int(s["ts"]): s for s in samples_a if _keep(s)}
    by_b = {int(s["ts"]): s for s in samples_b if _keep(s)}
    common = sorted(set(by_a) & set(by_b))
    rows = []
    for ts in common:
        a, b = by_a[ts], by_b[ts]
        rows.append({"ts": ts, "night": a.get("night") or b.get("night")})
        for k in keys:
            rows[-1][f"{k}_a"] = a.get(k)
            rows[-1][f"{k}_b"] = b.get(k)
    nights = [r["night"] for r in rows]
    out = []
    for k in keys:
        av = [r[f"{k}_a"] for r in rows]
        bv = [r[f"{k}_b"] for r in rows]
        boot = paired_cluster_bootstrap(av, bv, nights)
        wil = wilcoxon_paired(av, bv)
        out.append({
            "metric": k,
            "n": len(rows),
            "mean_a": float(np.nanmean(av)) if rows else None,
            "mean_b": float(np.nanmean(bv)) if rows else None,
            "delta_a_minus_b": boot["point"],
            "ci_lo": boot["lo"],
            "ci_hi": boot["hi"],
            "n_nights": boot["n_nights"],
            "wilcoxon_p": wil["p_value"],
            "split": split or "val+test",
            "name_a": name_a,
            "name_b": name_b,
        })
    return pd.DataFrame(out)


def attach_swin_probs(samples: List[Dict], manifest, cls_cfg, cls_ckpt, device) -> None:
    """Write Swin p_cls onto each sample (val+test) for presence score_field=p_cls."""
    from src.cascade import _cls_probs, _restore
    rows = [r for r in manifest.to_dict("records")
            if str(r["split"]) in {"val", "test"}]
    # Score in the same order as `rows`; map by timestamp.
    model = _restore(cls_cfg, cls_ckpt, device)
    from src import data_prep
    _, norm_stats = data_prep.load_artifacts(cls_cfg)
    # Use the passed manifest's matching rows in timestamp order.
    by_ts = {int(r["timestamp"]): r for r in rows}
    ordered = [by_ts[int(s["ts"])] for s in samples if int(s["ts"]) in by_ts]
    probs = _cls_probs(model, ordered, cls_cfg, norm_stats, device)
    ts_to_p = {int(r["timestamp"]): p for r, p in zip(ordered, probs)}
    del model
    for s in samples:
        p = ts_to_p.get(int(s["ts"]))
        if p is None:
            continue
        s["p_cls"] = float(p)
        models = s.setdefault("models", {})
        for block in models.values():
            block["p_cls"] = float(p)


def roc_auc(truth: List[int], scores: List[float]) -> float | None:
    """Rank-based ROC-AUC (ties averaged); None unless both classes are present."""
    y = np.asarray(truth, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    srt = s[order]
    i = 0
    while i < len(srt):                       # average ranks within tie groups
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", default="configs/base_config_night.yaml")
    ap.add_argument("--experiments", default="configs/experiments_night.yaml")
    ap.add_argument("--names", default=None, help="comma-separated subset")
    ap.add_argument("--baseline", default="night_base_attunet9")
    ap.add_argument("--cls-base", default="configs/base_config_cascade_cls.yaml")
    ap.add_argument("--cls-experiments", default="configs/experiments_cascade_cls.yaml")
    ap.add_argument("--cls-name", default="cls_swin_tiny_bal")
    ap.add_argument("--pair-baseline", default="unet_night_s42")
    ap.add_argument("--pair-final", default="night_base_attunet9_s42")
    args = ap.parse_args()

    base = cfgmod.load_base_config(args.base_config)
    experiments = cfgmod.load_experiments(args.experiments)
    if args.names:
        wanted = {n.strip() for n in args.names.split(",")}
        experiments = [e for e in experiments if e["name"] in wanted]

    manifest, norm_stats = data_prep.load_artifacts(base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = paths.output_dir(base) / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_doc = dataset_document(manifest)
    (out_dir / "dataset.json").write_text(
        json.dumps(json_safe(dataset_doc), separators=(",", ":"), allow_nan=False),
        encoding="utf-8")

    seg_rows, cls_rows = [], []
    samples_by_name: Dict[str, List[Dict]] = {}
    for exp in experiments:
        cfg = cfgmod.resolve_experiment(base, exp)
        name = cfg["name"]
        exp_dir = paths.experiments_dir(cfg)
        # Prefer the finalized result (test scored after freezing); fall back to
        # the training result, which under defer_test carries validation only.
        final = exp_dir / f"{name}_final_result.json"
        train = ckpt.result_path(exp_dir, name)
        source = final if final.exists() else train
        if not source.exists():
            print(f"[skip] {name}: no result yet")
            continue
        result = json.loads(source.read_text(encoding="utf-8"))
        threshold = float(result["calibrated_threshold"])
        print(f"[variant] {name} (threshold {threshold}, from {source.name})")

        key = name.replace("night_", "").replace("_attunet9", "") or name
        samples, extra = score_variant(cfg, manifest, norm_stats, device, threshold, key)
        if samples is None:
            print(f"[skip] {name}: no checkpoint")
            continue

        samples_doc = {"selected": name, "compare": name, "threshold": threshold,
                       "models": [{"key": key, "name": name,
                                   "disp": name, "thr": threshold}],
                       "samples": samples}
        samples_by_name[name] = samples
        (out_dir / f"{name}_samples.json").write_text(
            json.dumps(json_safe(samples_doc), separators=(",", ":"), allow_nan=False),
            encoding="utf-8")

        presence = analyze_presence(samples_doc, dataset_doc, score_field="pred_area")
        (out_dir / f"presence_{name}.json").write_text(
            json.dumps(json_safe(presence), separators=(",", ":"), allow_nan=False),
            encoding="utf-8")

        model_block = presence["models"][0]
        for split, label in (("validation", "val"), ("test", "test")):
            seg = extra["per_split"][label]
            row = {"experiment": name, "split": label,
                   "calibrated_threshold": threshold,
                   "n_params": result.get("n_params"),
                   "best_epoch": result.get("best_epoch")}
            row.update({out_key: _r(seg.get(src)) for src, out_key in SEG_COLUMNS})
            scan = model_block["scan"]
            row["scan_auc"] = _r(scan["splits"][split]["roc"]["auc"])
            row["scan_cutoff"] = scan["selected_cutoff"]["cells"]
            row["scan_confusion"] = json.dumps(
                scan["splits"][split]["operating_points"]["validation_selected"]["confusion"])
            for agg in ("max", "mean"):
                block = model_block["night"][agg]["splits"][split]
                row[f"night_{agg}_auc"] = _r(block["roc"]["auc"])
                row[f"night_{agg}_cutoff"] = model_block["night"][agg]["selected_cutoff"]["cells"]
                row[f"night_{agg}_confusion"] = json.dumps(
                    block["operating_points"]["validation_selected"]["confusion"])
                row[f"night_{agg}_mw_p"] = _r(block["mann_whitney"]["p_value"], 6)
                row[f"night_{agg}_mw_auc"] = _r(block["mann_whitney"]["common_language_auc"])
            seg_rows.append(row)

        # Auxiliary classifier: reported SEPARATELY so it is never mistaken for
        # the segmentation-derived presence result.
        for split in ("val", "test"):
            subset = [c for c in extra["cls_scores"] if c["split"] == split]
            if subset:
                cls_rows.append({
                    "experiment": name, "split": split, "n": len(subset),
                    "n_positive": sum(c["truth"] for c in subset),
                    "scan_cls_auc": _r(roc_auc([c["truth"] for c in subset],
                                               [c["score"] for c in subset])),
                    "note": "MIL max-pool over sliding windows; auxiliary only",
                })

    if not seg_rows:
        raise SystemExit("no variants scored")

    df = pd.DataFrame(seg_rows)
    baseline = df[(df.experiment == args.baseline)].set_index("split")
    for metric in ("dice_macro", "dice_micro", "scan_auc", "night_max_auc"):
        if args.baseline in set(df.experiment):
            df[f"d_{metric}"] = df.apply(
                lambda r: (None if r.experiment == args.baseline or r.split not in baseline.index
                           or r[metric] is None or baseline.loc[r.split, metric] is None
                           else round(r[metric] - baseline.loc[r.split, metric], 4)), axis=1)
    df.to_csv(out_dir / "comparison.csv", index=False)
    with open(out_dir / "comparison.md", "w", encoding="utf-8") as f:
        f.write("# Night-split ablation — variants vs the night-split baseline\n\n")
        f.write(f"Baseline: `{args.baseline}`. `d_*` columns are the difference from it "
                f"on the same split. Presence metrics are derived from segmentation "
                f"`pred_area`, so every variant is measured identically.\n\n")
        f.write("The historical dashboard model is NOT in this table: it was trained on the "
                "scan-level split where 172 of 248 nights straddled train/val/test, so its "
                "numbers are not comparable.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")

    if cls_rows:
        cdf = pd.DataFrame(cls_rows)
        cdf.to_csv(out_dir / "classification.csv", index=False)
        with open(out_dir / "classification.md", "w", encoding="utf-8") as f:
            f.write("# Auxiliary scan-presence classifier (reported separately)\n\n")
            f.write("These come from the multi-task classification head, aggregated over "
                    "sliding windows by max-pooling (MIL bag score). They are diagnostic "
                    "only: the presence metrics in `comparison.md` are derived from "
                    "segmentation for direct comparability across all variants.\n\n")
            f.write(cdf.to_markdown(index=False))
            f.write("\n")

    from src.cascade import _load_done
    cls_cfg, cls_res, cls_ckpt = _load_done(args.cls_base, args.cls_experiments, args.cls_name)
    if cls_res is not None and cls_ckpt is not None:
        print(f"[cls] attaching {args.cls_name} p_cls to samples")
        for name, samples in samples_by_name.items():
            attach_swin_probs(samples, manifest, cls_cfg, cls_ckpt, device)
            key = name.replace("night_", "").replace("_attunet9", "") or name
            samples_doc = json.loads((out_dir / f"{name}_samples.json").read_text(encoding="utf-8"))
            samples_doc["samples"] = samples
            (out_dir / f"{name}_samples.json").write_text(
                json.dumps(json_safe(samples_doc), separators=(",", ":"), allow_nan=False),
                encoding="utf-8")
            presence_cls = analyze_presence(samples_doc, dataset_doc, score_field="p_cls")
            (out_dir / f"presence_{name}_pcls.json").write_text(
                json.dumps(json_safe(presence_cls), separators=(",", ":"), allow_nan=False),
                encoding="utf-8")

    if args.pair_baseline in samples_by_name and args.pair_final in samples_by_name:
        # TEST is the confirmatory comparison; VAL is reported separately for
        # reference only. Pooling them would fold the split that selected the
        # model and thresholds into the interval.
        frames = {s: paired_table(samples_by_name[args.pair_final],
                                  samples_by_name[args.pair_baseline],
                                  args.pair_final, args.pair_baseline, split=s)
                  for s in ("test", "val")}
        pdf = pd.concat([frames["test"], frames["val"]], ignore_index=True)
        pdf.to_csv(out_dir / f"paired_{args.pair_baseline}.csv", index=False)
        with open(out_dir / f"paired_{args.pair_baseline}.md", "w", encoding="utf-8") as f:
            f.write(f"# Paired {args.pair_final} vs {args.pair_baseline} "
                    f"(positive scans, night-clustered)\n\n")
            f.write("Lead interval is the night-clustered bootstrap on the difference. "
                    "Wilcoxon over scans is anticonservative under within-night "
                    "correlation.\n\n")
            f.write("**TEST is the confirmatory result.** The validation block is "
                    "reference only: the model and every threshold were selected on "
                    "it, so its effect is partly in-sample.\n\n")
            for s in ("test", "val"):
                f.write(f"## {s}\n\n")
                f.write(frames[s].to_markdown(index=False, floatfmt=".4f"))
                f.write("\n\n")

    print(f"\n[done] wrote {out_dir}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
