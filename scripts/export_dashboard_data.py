"""Generate every data asset the evaluation dashboard reads.

The website is static (no server), so all analysis happens here and is written
to ``sprucebudworm_progress.github.io/data/``. Nothing in the site is computed
from placeholder values: every number traces back to the manifest, the cached
targets, the experiment outputs under ``outputs/experiments``, or a forward pass
of a trained checkpoint performed by this script.

Stages (each can be run alone with --only):
  experiments  57 experiment configs/metrics/histories + parsed training logs
  dataset      split / year / night / time / target-area distributions
  predict      forward pass of available registered viewer models over val+test,
               preserving validated records for registered models whose local
               checkpoints are absent,
               producing per-scene metrics, threshold sweeps, calibration
               histograms, connected components and radial error profiles
  presence     GPU-free scan/night presence analysis from samples.json and
               dataset.json, with validation-selected operating cutoffs
  images       legacy PNG intermediates for probability/ground truth migration
  packs        GPU-free SBW1 packs + WebP thumbnails for the sample explorer

Run:
  .venv\\Scripts\\python.exe scripts\\export_dashboard_data.py --only experiments,dataset
  .venv\\Scripts\\python.exe scripts\\export_dashboard_data.py --only packs
      --data-root ..\\Data --site-dir ..\\sprucebudworm_progress.github.io

The raw-data and website roots default to those sibling paths, so the explicit
options above are needed only when either checkout lives elsewhere.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import struct
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SITE = ROOT.parent / "sprucebudworm_progress.github.io"
DATA_OUT = SITE / "data"
IMG_OUT = DATA_OUT / "samples"

SBW_HEADER = struct.Struct("<4sHHBBBB")
SBW_MAGIC = b"SBW1"
SBW_FLAGS = 0x03  # bit 0: MSB-first packed GT; bit 1: categorical reflectivity
SBW_VERSION_PREFIX = "sbw1-max6-v1"
VIEWER_MODELS = (
    {"key": "attunet9", "name": "sweep_attunet_dbz0_e012345678_focaltv",
     "disp": "Attention UNet (9 elev)"},
    {"key": "unetpp9", "name": "sweep_unetpp_dbz0_e012345678_focaltv",
     "disp": "UNet++ (9 elev)"},
    {"key": "attunet7", "name": "sweep_attunet_dbz0_e0123456_focaltv",
     "disp": "Attention UNet (7 elev)"},
    {"key": "attunet8", "name": "sweep_attunet_dbz0_e01234567_focaltv",
     "disp": "Attention UNet (8 elev)"},
)
VIEWER_MODEL_KEYS = tuple(m["key"] for m in VIEWER_MODELS)
REFLECTIVITY_BINS = (-1.0, 2.0, 7.0, 12.0, 19.0)
REFLECTIVITY_COLORS = np.asarray([
    (0, 0, 0), (190, 222, 230), (117, 231, 137), (42, 220, 18),
    (247, 235, 39), (247, 139, 20), (242, 31, 23),
], dtype=np.uint8)


def configure_site(site_dir: Path | str) -> None:
    """Point all generated dashboard outputs at a website checkout."""
    global SITE, DATA_OUT, IMG_OUT
    SITE = Path(site_dir).expanduser().resolve()
    DATA_OUT = SITE / "data"
    IMG_OUT = DATA_OUT / "samples"

# The configuration selected by Experiments 1-5, and the runner-up used for
# model-vs-model comparison in the sample explorer.
SELECTED = VIEWER_MODELS[0]["name"]
COMPARE = VIEWER_MODELS[1]["name"]

# Thresholds swept for the calibration/threshold section. The project's
# officially selected operating point (0.15) is included and never overwritten.
SWEEP_T = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8]
PROB_BINS = 50  # histogram resolution for probability distributions


def _w(name: str, obj) -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    p = DATA_OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), allow_nan=False)
    try:
        shown = p.relative_to(ROOT.parent)
    except ValueError:
        shown = p
    print(f"  wrote {shown}  ({p.stat().st_size/1024:.0f} KB)")


def _num(x):
    """JSON-safe float: NaN/Inf -> None, numpy scalar -> python."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _r(x, nd=4):
    v = _num(x)
    return None if v is None else round(v, nd)


# --------------------------------------------------------------------------
# stage: experiments
# --------------------------------------------------------------------------
LOG_START = re.compile(r"^(\S+ \S+) \| START (\S+) \|.*params=([\d.]+)M")
LOG_EPOCH = re.compile(r"^(\S+ \S+) \|\s*(\d+)/(\d+)\s*\|.*?\|\s*(\d+)s\s*$")


def parse_log(path: Path) -> Dict:
    """Training duration and per-epoch seconds from a run's train.log."""
    out = {"train_seconds": None, "epoch_seconds": [], "wall_start": None, "wall_end": None}
    if not path.exists():
        return out
    first = last = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LOG_EPOCH.match(line.strip())
        if m:
            out["epoch_seconds"].append(int(m.group(4)))
        ts = line.split(" | ")[0].strip()
        try:
            t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        first = first or t
        last = t
    if first and last:
        out["wall_start"] = first.isoformat(sep=" ")
        out["wall_end"] = last.isoformat(sep=" ")
        out["train_seconds"] = int((last - first).total_seconds())
    return out


def stage_experiments() -> None:
    print("[experiments]")
    exp_dir = ROOT / "outputs" / "experiments"
    rows, histories = [], {}
    for res_path in sorted(exp_dir.glob("*_result.json")):
        name = res_path.name[: -len("_result.json")]
        res = json.loads(res_path.read_text(encoding="utf-8"))
        cfg_path = exp_dir / f"{name}_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        log = parse_log(exp_dir / f"{name}_train.log")
        t = res.get("test_full_scene", {}) or {}

        hp = cfg.get("train", {})
        tgt = cfg.get("target", {})
        chans = res.get("channels", cfg.get("channels", []))
        n_elev = sum(1 for c in chans if c.startswith("th_e"))
        rows.append({
            "name": name,
            "model": res.get("model"),
            "loss": res.get("loss"),
            "target_mode": tgt.get("mode"),
            "dbz_threshold": _num(tgt.get("dbz_threshold")),
            "channels": chans,
            "n_channels": res.get("n_channels", len(chans)),
            "n_elev": n_elev,
            "has_dem": "dem" in chans,
            "has_beam": any(c.startswith("bh_e") for c in chans),
            "n_params": res.get("n_params"),
            "best_val_dice_patch": _r(res.get("best_val_dice_patch")),
            "best_epoch": res.get("best_epoch"),
            "epochs_budget": hp.get("epochs"),
            "lr": _num(hp.get("lr")),
            "batch_size": hp.get("batch_size"),
            "accum_steps": hp.get("accum_steps"),
            "patch_size": (cfg.get("patch") or {}).get("size"),
            "patches_per_image": (cfg.get("patch") or {}).get("patches_per_image"),
            "threshold": _num(res.get("calibrated_threshold")),
            "train_seconds": log["train_seconds"],
            "sec_per_epoch": (int(np.median(log["epoch_seconds"])) if log["epoch_seconds"] else None),
            "n_epochs_run": len(log["epoch_seconds"]) or None,
            # held-out test, full-scene
            "dice": _r(t.get("dice")), "dice_micro": _r(t.get("dice_micro")),
            "iou": _r(t.get("iou")), "iou_micro": _r(t.get("iou_micro")),
            "precision": _r(t.get("precision")), "recall": _r(t.get("recall")),
            "f1": _r(t.get("f1")), "accuracy": _r(t.get("accuracy")),
            "boundary_iou": _r(t.get("boundary_iou")), "nsd": _r(t.get("nsd")),
            "hd95": _r(t.get("hd95"), 2), "assd": _r(t.get("assd"), 2),
            "bg_fp_rate": _r(t.get("bg_fp_rate"), 6),
            "n_pos_scenes": t.get("n_pos_scenes"), "n_neg_scenes": t.get("n_neg_scenes"),
            "selected": name == SELECTED,
        })

        h_path = exp_dir / f"{name}_history.csv"
        if h_path.exists():
            h = pd.read_csv(h_path)
            keep = [c for c in ["epoch", "train_loss", "lr", "val_dice", "val_iou",
                                "val_precision", "val_recall", "val_accuracy"] if c in h.columns]
            hh = {c: [_r(v, 5) for v in h[c].tolist()] for c in keep}
            hh["sec_per_epoch"] = log["epoch_seconds"] or None
            histories[name] = hh

    _w("experiments.json", {"generated": datetime.now().isoformat(timespec="seconds"),
                            "selected": SELECTED, "compare": COMPARE,
                            "n": len(rows), "experiments": rows})
    _w("histories.json", histories)
    print(f"  {len(rows)} experiments, {len(histories)} histories")


# --------------------------------------------------------------------------
# stage: dataset
# --------------------------------------------------------------------------
def stage_dataset() -> None:
    print("[dataset]")
    from src import config as cfgmod, paths

    base = cfgmod.load_base_config(str(ROOT / "configs" / "base_config.yaml"))
    man = pd.read_csv(ROOT / "artifacts" / "manifest.csv")
    man["hour"] = (man.timestamp % 10000) // 100
    man["date"] = man.timestamp // 10000

    # target area per positive scene, read from the cached masks (no netCDF needed).
    # Areas are reported for BOTH label definitions so the label-threshold effect
    # is measurable rather than asserted.
    tdir = paths.targets_dir(base)
    areas = {}
    pos = man[man.label == 1]
    for i, r in enumerate(pos.itertuples(), 1):
        f = tdir / f"y_{r.timestamp}.npz"
        if not f.exists():
            continue
        with np.load(f) as z:
            dbz = z[z.files[0]]
        finite = np.isfinite(dbz)
        areas[int(r.timestamp)] = {
            "isfinite": int(finite.sum()),
            "dbz0": int((finite & (dbz >= 0.0)).sum()),
            "dbz5": int((finite & (dbz >= 5.0)).sum()),
            "dbz_mean": _r(np.nanmean(dbz[finite]) if finite.any() else None, 2),
            "dbz_p95": _r(np.nanpercentile(dbz[finite], 95) if finite.any() else None, 2),
        }
        if i % 250 == 0:
            print(f"  areas {i}/{len(pos)}")

    npx = 960 * 960
    scenes = []
    for r in man.itertuples():
        a = areas.get(int(r.timestamp), {})
        scenes.append({
            "ts": int(r.timestamp), "year": int(r.year), "split": r.split,
            "label": int(r.label),
            "night": (None if not isinstance(r.night, str) else r.night),
            "hour": int(r.hour), "date": int(r.date),
            "area": a.get("dbz0"), "area_isfinite": a.get("isfinite"),
            "area_dbz5": a.get("dbz5"),
            "dbz_mean": a.get("dbz_mean"), "dbz_p95": a.get("dbz_p95"),
            "pos_frac": (_r(a["dbz0"] / npx, 6) if a.get("dbz0") is not None else None),
        })

    # night leakage across splits: the manifest is split by year/scene, not by
    # night, so nights can span splits. This is measured, not assumed.
    p = man[man.label == 1]
    by_night = p.groupby("night")["split"].agg(lambda s: sorted(set(s)))
    train_nights = set(p[p.split == "train"].night)
    leak = {
        "n_nights": int(p.night.nunique()),
        "nights_multi_split": int((by_night.map(len) > 1).sum()),
        "nights_all_three": int((by_night.map(len) == 3).sum()),
        "test_scenes_night_in_train": int(p[(p.split == "test") & (p.night.isin(train_nights))].shape[0]),
        "test_scenes_total": int(p[p.split == "test"].shape[0]),
        "val_scenes_night_in_train": int(p[(p.split == "val") & (p.night.isin(train_nights))].shape[0]),
        "val_scenes_total": int(p[p.split == "val"].shape[0]),
    }

    split_summary = json.loads((ROOT / "artifacts" / "split_summary.json").read_text())
    from src.channels import GRID
    grid = dict(GRID)
    _w("dataset.json", {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "grid": grid,
        "split_summary": split_summary,
        "leakage": leak,
        "scenes": scenes,
    })
    # Compact companion for the overview, which needs the headline counts but not
    # the 2000+ per-scene records (keeps the first page load small).
    _w("summary.json", {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "grid": grid, "split_summary": split_summary, "leakage": leak,
        "n_scenes": len(scenes),
        "years": sorted({s["year"] for s in scenes}),
    })
    print(f"  {len(scenes)} scenes, {len(areas)} target areas")


# --------------------------------------------------------------------------
# stage: predict
# --------------------------------------------------------------------------
def _radial_index(h=960, w=960, n_ring=12, pixel_km=0.5):
    """Ring index per pixel (radar assumed at grid centre) + ring edges in km."""
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) * pixel_km
    edges = np.linspace(0, d.max(), n_ring + 1)
    idx = np.clip(np.digitize(d, edges) - 1, 0, n_ring - 1)
    return idx, edges, d


def _components(mask: np.ndarray, min_size: int = 10):
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return 0, []
    sizes = np.bincount(lab.ravel())[1:]
    sizes = sizes[sizes >= min_size]
    return int(len(sizes)), sorted(int(s) for s in sizes)


def _load_model(name, device, required: bool = True):
    from src import checkpoint as ckpt, config as cfgmod, paths
    from src.models import create_model
    base = cfgmod.load_base_config(str(ROOT / "configs" / "base_config.yaml"))
    exps = cfgmod.load_experiments(str(ROOT / "configs" / "experiments_elev.yaml"))
    exp = next(e for e in exps if e["name"] == name)
    cfg = cfgmod.resolve_experiment(base, exp)
    st = ckpt.load_checkpoint(ckpt.best_path(paths.checkpoint_dir(base), name), device)
    if st is None:
        if required:
            raise SystemExit(f"no checkpoint for {name}")
        res = json.loads((ROOT / "outputs" / "experiments" / f"{name}_result.json").read_text())
        return None, cfg, float(res["calibrated_threshold"])
    m = create_model(cfg).to(device)
    m.load_state_dict(st["model"])
    m.eval()
    res = json.loads((ROOT / "outputs" / "experiments" / f"{name}_result.json").read_text())
    return m, cfg, float(res["calibrated_threshold"])


def _prediction_metrics(prob: np.ndarray, truth: np.ndarray, threshold: float,
                        is_positive: bool, metrics_module) -> tuple[Dict, np.ndarray]:
    """Compute the per-model scene record used by samples.json."""
    pred = prob > threshold
    tp = float((pred & truth).sum()); fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum()); tn = float((~pred & ~truth).sum())
    eps = 1e-8
    out: Dict = {
        "pred_area": int(pred.sum()), "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
    }
    if is_positive:
        out.update({
            "dice": _r(2 * tp / (2 * tp + fp + fn + eps)),
            "iou": _r(tp / (tp + fp + fn + eps)),
            "precision": _r(tp / (tp + fp + eps)),
            "recall": _r(tp / (tp + fn + eps)),
            "accuracy": _r((tp + tn) / (tp + tn + fp + fn + eps), 5),
            "specificity": _r(tn / (tn + fp + eps), 5),
            "boundary_iou": _r(metrics_module.boundary_iou(pred, truth)),
        })
        out.update({k: _r(v, 3) for k, v in
                    metrics_module.surface_metrics(pred, truth, tau=2.0).items()})
        n_pred, pred_sizes = _components(pred)
        out.update({"n_pred_regions": n_pred,
                    "pred_region_max": (pred_sizes[-1] if pred_sizes else 0)})
    else:
        out["bg_fp_rate"] = _r(float(pred.mean()), 6)
    return out, pred


def _public_model_metrics(values: Dict, is_positive: bool) -> Dict:
    """Keep the stable compact schema consumed by the website."""
    if not is_positive:
        return {k: values[k] for k in ("pred_area", "bg_fp_rate")}
    keys = ("pred_area", "dice", "iou", "precision", "recall", "accuracy",
            "specificity", "boundary_iou", "tp", "fp", "fn", "nsd", "hd95",
            "assd", "n_pred_regions")
    return {k: values.get(k) for k in keys}


def _validate_preserved_model_lineage(previous: Dict, reused_models,
                                      expected_artifact_version: str) -> None:
    """Refuse to relabel preserved scene metrics as a changed model artifact."""
    if not reused_models:
        return
    previous_version = (previous.get("sample_assets") or {}).get("model_artifact_version")
    if previous_version != expected_artifact_version:
        raise SystemExit("cannot reuse missing-checkpoint scene metrics: previous "
                         f"model_artifact_version={previous_version!r}, expected "
                         f"{expected_artifact_version!r}")

    previous_models = list(previous.get("models") or [])
    for spec, threshold in reused_models:
        matches = [m for m in previous_models if m.get("key") == spec["key"]]
        if len(matches) != 1:
            raise SystemExit("cannot reuse missing-checkpoint scene metrics: previous "
                             f"model key {spec['key']!r} has {len(matches)} matches")
        old = matches[0]
        try:
            old_threshold = float(old["thr"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit("cannot reuse missing-checkpoint scene metrics: previous "
                             f"model {spec['key']!r} has no numeric threshold") from exc
        if old.get("name") != spec["name"] or old_threshold != float(threshold):
            raise SystemExit("cannot reuse missing-checkpoint scene metrics: model lineage "
                             f"changed for {spec['key']!r}; previous name/thr="
                             f"{old.get('name')!r}/{old.get('thr')!r}, expected "
                             f"{spec['name']!r}/{float(threshold)!r}")


def stage_predict(splits=("test", "val"), limit=None) -> None:
    print("[predict]")
    import torch
    from src import data_prep, dataset as dsmod, engine, metrics as M, config as cfgmod

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = cfgmod.load_base_config(str(ROOT / "configs" / "base_config.yaml"))
    man, norm = data_prep.load_artifacts(base)

    previous = {}
    samples_path = DATA_OUT / "samples.json"
    if samples_path.exists():
        previous = json.loads(samples_path.read_text(encoding="utf-8"))
    previous_by_ts = {int(s["ts"]): s for s in previous.get("samples", [])}

    loaded_models = []
    for spec in VIEWER_MODELS:
        model, cfg, threshold = _load_model(spec["name"], device, required=False)
        loaded_models.append((spec, model, cfg, threshold))
        source = "checkpoint" if model is not None else "preserved packed export"
        print(f"  {spec['key']}={spec['name']} thr={threshold} [{source}]")
    if any(model is None for _spec, model, _cfg, _threshold in loaded_models[:2]):
        raise SystemExit("the selected and comparison checkpoints are required by stage predict")
    current_model_version = _model_artifact_version()
    reused_models = [(spec, threshold) for spec, model, _cfg, threshold in loaded_models
                     if model is None]
    _validate_preserved_model_lineage(previous, reused_models, current_model_version)
    THR = loaded_models[0][3]
    THR2 = loaded_models[1][3]
    ring_idx, ring_edges, dist_km = _radial_index()
    n_ring = len(ring_edges) - 1

    samples: List[Dict] = []
    # global accumulators
    sweep = {s: {t: dict(tp=0.0, fp=0.0, fn=0.0, dice_sum=0.0, n=0) for t in SWEEP_T} for s in splits}
    hist_pos = {s: np.zeros(PROB_BINS) for s in splits}
    hist_neg = {s: np.zeros(PROB_BINS) for s in splits}
    calib = {s: {"sum_p": np.zeros(PROB_BINS), "sum_y": np.zeros(PROB_BINS),
                 "n": np.zeros(PROB_BINS)} for s in splits}
    ring_acc = {s: dict(tp=np.zeros(n_ring), fp=np.zeros(n_ring),
                        fn=np.zeros(n_ring), gt=np.zeros(n_ring)) for s in splits}
    edges = np.linspace(0.0, 1.0, PROB_BINS + 1)

    rows = man[man.split.isin(splits)].to_dict("records")
    if limit:
        rows = rows[:limit]
    print(f"  {len(rows)} scenes")

    for i, r in enumerate(rows, 1):
        split = r["split"]
        is_pos = int(r["label"]) == 1
        yb = None
        model_values: Dict[str, Dict] = {}
        probabilities: Dict[str, np.ndarray] = {}
        predictions: Dict[str, np.ndarray] = {}
        for spec, model, cfg, threshold in loaded_models:
            if model is None:
                old = (previous_by_ts.get(int(r["timestamp"]), {}).get("models") or {}).get(spec["key"])
                if not old:
                    raise SystemExit(f"no checkpoint or preserved metrics for {spec['key']} / {r['timestamp']}")
                model_values[spec["key"]] = old
                continue
            x, y = dsmod.load_full_scene(cfg, r, norm)
            candidate_truth = y.astype(bool)
            if yb is None:
                yb = candidate_truth
            elif not np.array_equal(yb, candidate_truth):
                raise RuntimeError(f"target mismatch across viewer models for {r['timestamp']}")
            ps = int(cfg["patch"]["size"])
            ov = float(cfg["eval"].get("overlap", 0.5))
            prob_i = engine.sliding_window_predict(model, x, device, ps, ov, False, gaussian=True)
            values, pred_i = _prediction_metrics(prob_i, yb, threshold, is_pos, M)
            probabilities[spec["key"]] = prob_i
            predictions[spec["key"]] = pred_i
            model_values[spec["key"]] = values

        prob = probabilities[VIEWER_MODEL_KEYS[0]]
        pred = predictions[VIEWER_MODEL_KEYS[0]]
        selected_values = model_values[VIEWER_MODEL_KEYS[0]]
        compare_values = model_values[VIEWER_MODEL_KEYS[1]]
        tp = float(selected_values["tp"]); fp = float(selected_values["fp"])
        fn = float(selected_values["fn"]); tn = float(selected_values["tn"])
        eps = 1e-8
        rec = {
            "ts": int(r["timestamp"]), "split": split, "label": int(r["label"]),
            "year": int(r["year"]),
            "night": (r["night"] if isinstance(r.get("night"), str) else None),
            "hour": int(int(r["timestamp"]) % 10000 // 100),
            "thr": THR,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "gt_area": int(yb.sum()), "pred_area": selected_values["pred_area"],
            "prob_mean": _r(float(prob.mean()), 5),
            "prob_max": _r(float(prob.max()), 4),
            "models": {key: _public_model_metrics(model_values[key], is_pos)
                       for key in VIEWER_MODEL_KEYS},
        }
        if is_pos:
            rec.update({k: selected_values[k] for k in
                        ("dice", "iou", "precision", "recall", "accuracy", "specificity",
                         "boundary_iou", "nsd", "hd95", "assd", "n_pred_regions",
                         "pred_region_max")})
            n_gt, gt_sizes = _components(yb)
            rec.update({"n_gt_regions": n_gt,
                        "gt_region_max": (gt_sizes[-1] if gt_sizes else 0)})
            # mean radial distance of GT signal and of each error type
            if yb.any():
                rec["gt_dist_km"] = _r(float(dist_km[yb].mean()), 1)
            if (pred & ~yb).any():
                rec["fp_dist_km"] = _r(float(dist_km[pred & ~yb].mean()), 1)
            if (~pred & yb).any():
                rec["fn_dist_km"] = _r(float(dist_km[~pred & yb].mean()), 1)
        else:
            # negatives have no positive pixels: only a false-alarm rate is defined
            rec["bg_fp_rate"] = selected_values["bg_fp_rate"]

        # Stable top-level comparison aliases retained for existing sections.
        if is_pos:
            rec["dice_cmp"] = compare_values["dice"]
        else:
            rec["bg_fp_rate_cmp"] = compare_values["bg_fp_rate"]
        rec["pred_area_cmp"] = compare_values["pred_area"]

        # ---- global accumulators (positives drive metric curves) ----
        if is_pos:
            for t in SWEEP_T:
                pt = prob > t
                a = float((pt & yb).sum()); b = float((pt & ~yb).sum()); c = float((~pt & yb).sum())
                s = sweep[split][t]
                s["tp"] += a; s["fp"] += b; s["fn"] += c
                s["dice_sum"] += 2 * a / (2 * a + b + c + eps); s["n"] += 1
            hp, _ = np.histogram(prob[yb], bins=edges)
            hn, _ = np.histogram(prob[~yb], bins=edges)
            hist_pos[split] += hp
            hist_neg[split] += hn
            bi = np.clip(np.digitize(prob.ravel(), edges) - 1, 0, PROB_BINS - 1)
            np.add.at(calib[split]["sum_p"], bi, prob.ravel())
            np.add.at(calib[split]["sum_y"], bi, yb.ravel().astype(float))
            np.add.at(calib[split]["n"], bi, 1.0)
            for k, m in (("tp", pred & yb), ("fp", pred & ~yb), ("fn", ~pred & yb), ("gt", yb)):
                ring_acc[split][k] += np.bincount(ring_idx[m], minlength=n_ring)

        samples.append(rec)
        if i % 25 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    # Preserve valid packed-asset metadata across a metrics rebuild. This makes
    # the default predict stage safe after the PNG-to-pack migration.
    sample_timestamps = {int(s["ts"]) for s in samples}
    previous_timestamps = {int(s["ts"]) for s in previous.get("samples", [])}
    previous_assets = previous.get("sample_assets") or {}
    assets_match = (previous_timestamps == sample_timestamps and
                    previous_assets.get("model_artifact_version") == current_model_version)
    have_img = sorted({split for split in splits if assets_match and all(
        (IMG_OUT / f"{ts}.sbw.gz").exists() and (IMG_OUT / f"{ts}.webp").exists()
        for ts in (int(s["ts"]) for s in samples if s["split"] == split)
    )})
    result = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "selected": SELECTED, "compare": COMPARE, "threshold": THR,
        "threshold_cmp": THR2, "image_splits": have_img,
        "models": [{**spec, "thr": threshold}
                   for spec, _model, _cfg, threshold in loaded_models],
        "samples": samples,
    }
    if previous_assets and previous_timestamps == sample_timestamps:
        result["sample_assets"] = previous_assets
    _w("samples.json", result)

    def curve(split):
        out = []
        for t in SWEEP_T:
            s = sweep[split][t]
            if not s["n"]:
                continue
            tp, fp, fn = s["tp"], s["fp"], s["fn"]
            out.append({"t": t,
                        "dice_macro": _r(s["dice_sum"] / s["n"]),
                        "dice_micro": _r(2 * tp / (2 * tp + fp + fn + 1e-8)),
                        "iou_micro": _r(tp / (tp + fp + fn + 1e-8)),
                        "precision": _r(tp / (tp + fp + 1e-8)),
                        "recall": _r(tp / (tp + fn + 1e-8)),
                        "n": s["n"]})
        return out

    def hists(split):
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(PROB_BINS)]
        n = calib[split]["n"]
        rel = [{"p": _r(calib[split]["sum_p"][i] / n[i], 4),
                "y": _r(calib[split]["sum_y"][i] / n[i], 4),
                "n": int(n[i])} for i in range(PROB_BINS) if n[i] > 0]
        return {"centers": [round(c, 3) for c in centers],
                "pos": [int(v) for v in hist_pos[split]],
                "neg": [int(v) for v in hist_neg[split]],
                "reliability": rel}

    _w("threshold.json", {
        "selected_threshold": THR, "swept": SWEEP_T,
        "curves": {s: curve(s) for s in splits},
        "distributions": {s: hists(s) for s in splits},
        "radial": {s: {"edges_km": [round(float(e), 1) for e in ring_edges],
                       "tp": [int(v) for v in ring_acc[s]["tp"]],
                       "fp": [int(v) for v in ring_acc[s]["fp"]],
                       "fn": [int(v) for v in ring_acc[s]["fn"]],
                       "gt": [int(v) for v in ring_acc[s]["gt"]]} for s in splits},
    })


# --------------------------------------------------------------------------
# stage: presence
# --------------------------------------------------------------------------
def stage_presence() -> None:
    """Build scan/night binary-detection results without model inference."""
    print("[presence]")
    from src.presence import analyze_presence

    samples_path = DATA_OUT / "samples.json"
    dataset_path = DATA_OUT / "dataset.json"
    missing = [str(p) for p in (samples_path, dataset_path) if not p.exists()]
    if missing:
        raise SystemExit("presence stage requires generated samples.json and dataset.json; "
                         f"missing: {', '.join(missing)}")
    samples_doc = json.loads(samples_path.read_text(encoding="utf-8"))
    dataset_doc = json.loads(dataset_path.read_text(encoding="utf-8"))
    model_keys = tuple(m.get("key") for m in samples_doc.get("models", []))
    if model_keys != VIEWER_MODEL_KEYS:
        raise SystemExit(f"presence stage requires viewer models {list(VIEWER_MODEL_KEYS)}; "
                         f"got {list(model_keys)}")
    result = analyze_presence(
        samples_doc,
        dataset_doc,
        generated=datetime.now().isoformat(timespec="seconds"),
    )
    _w("presence.json", result)


# --------------------------------------------------------------------------
# stage: images
# --------------------------------------------------------------------------
def _to_png(arr_u8: np.ndarray, path: Path, mode="L") -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_u8, mode=mode).save(path, optimize=True)


def _downsample(a: np.ndarray, size: int) -> np.ndarray:
    """Block-max downsample (keeps thin plumes visible, unlike subsampling)."""
    h = a.shape[0]
    k = h // size
    if k <= 1:
        return a
    return a[: size * k, : size * k].reshape(size, k, size, k).max(axis=(1, 3))


def _nan_block_max(a: np.ndarray, size: int) -> np.ndarray:
    """Block maximum that ignores NaNs and preserves all-missing blocks."""
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] % size:
        raise ValueError(f"cannot block-downsample shape {a.shape} to {size}x{size}")
    k = a.shape[0] // size
    blocks = a.reshape(size, k, size, k)
    finite = np.isfinite(blocks)
    out = np.where(finite, blocks, -np.inf).max(axis=(1, 3))
    out[~finite.any(axis=(1, 3))] = np.nan
    return out.astype(np.float32, copy=False)


def _reflectivity_from_array(raw: np.ndarray, size: int = 480) -> tuple[np.ndarray, np.ndarray]:
    """Return display categories and raw block-max dBZ for six TH scans."""
    raw = (np.ma.filled(raw, np.nan) if np.ma.isMaskedArray(raw) else np.asarray(raw)).astype(np.float32)
    if raw.ndim != 3 or raw.shape[0] < 6:
        raise ValueError(f"need at least six TH elevations; got shape {raw.shape}")
    raw = raw[:6]
    finite = np.isfinite(raw)
    composite = np.where(finite, raw, -np.inf).max(axis=0)
    composite[~finite.any(axis=0)] = np.nan
    down = _nan_block_max(composite, size)
    categories = np.zeros(down.shape, dtype=np.uint8)
    # Values below the displayed legend floor are background, as are cells for
    # which every one of the six elevations is missing.
    valid = np.isfinite(down) & (down >= -10.0)
    categories[valid] = np.digitize(down[valid], REFLECTIVITY_BINS, right=True).astype(np.uint8) + 1
    return categories, down


def _reflectivity_composite(ppi_path: Path, size: int = 480) -> tuple[np.ndarray, np.ndarray]:
    """Read raw TH[0:6], then return its categorical block-max composite."""
    import netCDF4 as nc

    with nc.Dataset(ppi_path) as ds:
        if "TH" not in ds.variables:
            raise ValueError(f"{ppi_path} has no TH variable")
        v = ds.variables["TH"]
        if v.shape[0] < 6:
            raise ValueError(f"{ppi_path} has only {v.shape[0]} TH elevations; need 6")
        raw = v[:6]
    return _reflectivity_from_array(raw, size)


def _raw_ppi_index(data_root: Path) -> Dict[int, Path]:
    """Index raw PPI files by timestamp and reject ambiguous duplicates."""
    out: Dict[int, Path] = {}
    duplicates: Dict[int, List[Path]] = {}
    for p in data_root.rglob("*_filtered_ppi.nc"):
        m = re.search(r"(\d{12})", p.name)
        if not m:
            continue
        ts = int(m.group(1))
        if ts in out:
            duplicates.setdefault(ts, [out[ts]]).append(p)
        else:
            out[ts] = p
    if duplicates:
        detail = "; ".join(f"{ts}: {', '.join(map(str, ps))}" for ts, ps in sorted(duplicates.items()))
        raise RuntimeError(f"duplicate raw PPI timestamps: {detail}")
    return out


def _read_l_png(path: Path, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    if not path.exists():
        raise FileNotFoundError(path)
    with Image.open(path) as im:
        if im.mode != "L" or im.size != (shape[1], shape[0]):
            raise ValueError(f"{path} must be an L-mode {shape[1]}x{shape[0]} PNG; got {im.mode} {im.size}")
        return np.asarray(im, dtype=np.uint8).copy()


def _write_deterministic_gzip(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=9, mtime=0) as gz:
            gz.write(payload)
    tmp.replace(path)


def _read_existing_pack(path: Path, model_order: List[str], shape: tuple[int, int]) -> tuple[List[np.ndarray], bytes]:
    """Read probability planes and packed GT from a valid existing SBW1 file."""
    raw = gzip.decompress(path.read_bytes())
    if len(raw) < SBW_HEADER.size:
        raise ValueError(f"{path} has a truncated SBW1 header")
    magic, width, height, model_count, flags, header_size, reserved = SBW_HEADER.unpack_from(raw)
    expected_shape = (height, width)
    if (magic != SBW_MAGIC or expected_shape != shape or model_count != len(model_order) or
            flags != SBW_FLAGS or header_size != SBW_HEADER.size or reserved != 0):
        raise ValueError(f"{path} has an incompatible SBW1 header")
    pixels = width * height
    gt_size = (pixels + 7) // 8
    expected_size = header_size + model_count * pixels + gt_size + pixels
    if len(raw) != expected_size:
        raise ValueError(f"{path} has {len(raw)} uncompressed bytes; expected {expected_size}")
    offset = header_size
    probabilities = []
    for _ in model_order:
        probabilities.append(np.frombuffer(raw, np.uint8, pixels, offset).reshape(shape).copy())
        offset += pixels
    return probabilities, bytes(raw[offset:offset + gt_size])


def _asset_version(paths: List[Path]) -> str:
    """Content-address every delivered sample asset for browser cache busting."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        digest.update(path.name.encode("ascii"))
        digest.update(path.read_bytes())
    return f"{SBW_VERSION_PREFIX}-{digest.hexdigest()[:12]}"


def _model_artifact_version(root: Path | None = None,
                            exporter_path: Path | None = None,
                            viewer_models=None) -> str:
    """Fingerprint every input that can change exported probability/area data."""
    root = Path(root) if root is not None else ROOT
    exporter_path = (Path(exporter_path) if exporter_path is not None
                     else Path(__file__).resolve())
    viewer_models = tuple(viewer_models) if viewer_models is not None else VIEWER_MODELS
    digest = hashlib.sha256()
    paths = [
        root / "configs" / "base_config.yaml",
        root / "configs" / "experiments_elev.yaml",
        root / "artifacts" / "norm_stats.json",
        root / "src" / "channels.py",
        root / "src" / "config.py",
        root / "src" / "checkpoint.py",
        root / "src" / "dataset.py",
        root / "src" / "engine.py",
        exporter_path,
    ]
    # Includes create_model in models/__init__.py and every architecture module.
    paths.extend(sorted((root / "src" / "models").glob("*.py"),
                        key=lambda p: p.as_posix()))
    for spec in viewer_models:
        digest.update(json.dumps(spec, sort_keys=True).encode("utf-8"))
        paths.extend([
            root / "outputs" / "experiments" / f"{spec['name']}_result.json",
            root / "outputs" / "checkpoints" / f"{spec['name']}_best.pt",
        ])
    for path in paths:
        try:
            label = path.resolve().relative_to(root.resolve())
        except ValueError:
            label = path.resolve()
        digest.update(str(label).replace("\\", "/").encode("utf-8"))
        if not path.exists():
            digest.update(b"\0MISSING\0")
            continue
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return f"models-{digest.hexdigest()[:16]}"


def _write_thumbnail(path: Path, reflectivity: np.ndarray, probability: np.ndarray,
                     threshold: float, size: int = 120) -> None:
    from PIL import Image

    refl = _downsample(reflectivity, size)
    prob = _downsample(probability, size)
    rgb = REFLECTIVITY_COLORS[refl]
    pred = prob > threshold * 255.0
    rgb = rgb.copy()
    rgb[pred] = (rgb[pred].astype(np.float32) * 0.25 +
                 np.asarray((47, 125, 209), np.float32) * 0.75).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.webp")
    Image.fromarray(rgb).save(tmp, format="WEBP", lossless=True, method=6)
    tmp.replace(path)


def stage_packs(data_root: Path | str, size: int = 480, thumb: int = 120,
                expected_scenes: int = 615, limit: int | None = None) -> None:
    """Migrate exact PNG probability/GT bytes into per-scene SBW1 gzip packs.

    Reflectivity is regenerated from the raw radar volumes as the per-cell
    maximum of the six lowest TH elevations. This stage is GPU-free.
    """
    print("[packs]")
    data_root = Path(data_root).expanduser().resolve()
    samples_path = DATA_OUT / "samples.json"
    if not samples_path.exists():
        raise SystemExit(f"missing {samples_path}")
    doc = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = doc.get("samples") or []
    timestamps = [int(s["ts"]) for s in samples]
    if len(samples) != expected_scenes or len(set(timestamps)) != expected_scenes:
        raise SystemExit(f"samples.json must contain exactly {expected_scenes} unique scenes; "
                         f"got {len(samples)} rows / {len(set(timestamps))} unique")
    models = doc.get("models") or []
    model_order = [m.get("key") for m in models]
    if tuple(model_order) != VIEWER_MODEL_KEYS:
        raise SystemExit(f"samples.json model order must be {list(VIEWER_MODEL_KEYS)}; got {model_order}")

    raw_index = _raw_ppi_index(data_root)
    missing_raw = [ts for ts in timestamps if ts not in raw_index]
    if missing_raw:
        report = DATA_OUT / "missing_sample_ppi.txt"
        report.write_text("\n".join(map(str, missing_raw)) + "\n", encoding="utf-8")
        raise SystemExit(f"raw PPI coverage is {expected_scenes - len(missing_raw)}/{expected_scenes}; "
                         f"missing timestamps written to {report}")
    print(f"  raw PPI coverage {expected_scenes}/{expected_scenes} under {data_root}")

    shape = (size, size)
    threshold = float(doc.get("threshold", 0.15))
    todo = samples[:limit] if limit else samples
    legacy_complete = all(
        (IMG_OUT / f"{ts}_gt.png").exists() and
        all((IMG_OUT / f"{ts}_prob_{key}.png").exists() for key in model_order)
        for ts in timestamps
    )
    pack_complete = all((IMG_OUT / f"{ts}.sbw.gz").exists() for ts in timestamps)
    if legacy_complete:
        source = "legacy PNG probability/GT layers"
    elif pack_complete:
        source = "existing SBW1 probability/GT payloads"
        prior_model_version = (doc.get("sample_assets") or {}).get("model_artifact_version")
        current_model_version = _model_artifact_version()
        if prior_model_version != current_model_version:
            raise SystemExit("existing packs are tied to different model artifacts; run stage images "
                             "before rebuilding packs")
    else:
        legacy_missing = sum(not (IMG_OUT / f"{ts}_gt.png").exists() or any(
            not (IMG_OUT / f"{ts}_prob_{key}.png").exists() for key in model_order) for ts in timestamps)
        pack_missing = sum(not (IMG_OUT / f"{ts}.sbw.gz").exists() for ts in timestamps)
        raise SystemExit("cannot build packs: inputs are incomplete; "
                         f"legacy PNG sets missing for {legacy_missing} scenes and packs missing for {pack_missing}")
    print(f"  probability/GT source: {source}")

    for i, sample in enumerate(todo, 1):
        ts = int(sample["ts"])
        if legacy_complete:
            probabilities = [_read_l_png(IMG_OUT / f"{ts}_prob_{key}.png", shape) for key in model_order]
            gt = _read_l_png(IMG_OUT / f"{ts}_gt.png", shape) > 127
            gt_packed = np.packbits(gt.reshape(-1), bitorder="big").tobytes()
        else:
            probabilities, gt_packed = _read_existing_pack(
                IMG_OUT / f"{ts}.sbw.gz", model_order, shape)
        reflectivity, _ = _reflectivity_composite(raw_index[ts], size)
        header = SBW_HEADER.pack(SBW_MAGIC, size, size, len(model_order), SBW_FLAGS,
                                 SBW_HEADER.size, 0)
        payload = (header + b"".join(p.tobytes(order="C") for p in probabilities) +
                   gt_packed + reflectivity.tobytes(order="C"))
        pack_path = IMG_OUT / f"{ts}.sbw.gz"
        thumb_path = IMG_OUT / f"{ts}.webp"
        _write_deterministic_gzip(pack_path, payload)
        _write_thumbnail(thumb_path, reflectivity, probabilities[0], threshold, thumb)
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)}")

    if limit:
        print("  limited run: samples.json metadata was not changed")
        return

    pack_files = list(IMG_OUT.glob("*.sbw.gz"))
    thumb_files = list(IMG_OUT.glob("*.webp"))
    total = sum(p.stat().st_size for p in pack_files + thumb_files)
    if len(pack_files) != expected_scenes or len(thumb_files) != expected_scenes:
        raise RuntimeError(f"expected {expected_scenes} packs and thumbnails; "
                           f"got {len(pack_files)} packs / {len(thumb_files)} thumbnails")
    if total > 21 * 1024 * 1024:
        raise RuntimeError(f"packed sample assets use {total / 1024 / 1024:.2f} MiB; limit is 21 MiB")
    version = _asset_version(pack_files + thumb_files)

    doc["image_splits"] = sorted({str(s["split"]) for s in samples})
    doc["sample_assets"] = {
        "format": "sbw1-gzip", "header_size": SBW_HEADER.size,
        "width": size, "height": size, "thumbnail_width": thumb,
        "thumbnail_height": thumb, "model_order": model_order,
        "pack_path": "data/samples/{ts}.sbw.gz",
        "thumbnail_path": "data/samples/{ts}.webp",
        "version": version,
        "model_artifact_version": _model_artifact_version(),
        "reflectivity_source": "max_th_e0_th_e5",
        "reflectivity_elevations": [0, 1, 2, 3, 4, 5],
        "reflectivity_categories": [
            {"code": 0, "label": "background / missing / below -10 dBZ"},
            {"code": 1, "label": "-10 to -1 dBZ"},
            {"code": 2, "label": ">-1 to 2 dBZ"},
            {"code": 3, "label": ">2 to 7 dBZ"},
            {"code": 4, "label": ">7 to 12 dBZ"},
            {"code": 5, "label": ">12 to 19 dBZ"},
            {"code": 6, "label": ">19 dBZ"},
        ],
    }
    tmp_json = samples_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(doc, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    tmp_json.replace(samples_path)
    print(f"  wrote {len(pack_files)} packs + {len(thumb_files)} thumbnails, "
          f"{total / 1024 / 1024:.2f} MiB, version {version}")


def stage_images(size=480, thumb=120, splits=("test", "val"), limit=None) -> None:
    """Generate four-model probability/GT PNG migration intermediates.

    These files preserve the dashboard's established 8-bit preview precision.
    They are consumed by ``stage_packs`` and are not deployed by the website.
    """
    print("[images]")
    import torch
    from src import data_prep, dataset as dsmod, engine, config as cfgmod

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = cfgmod.load_base_config(str(ROOT / "configs" / "base_config.yaml"))
    man, norm = data_prep.load_artifacts(base)
    loaded_models = []
    for spec in VIEWER_MODELS:
        model, cfg, threshold = _load_model(spec["name"], device, required=False)
        loaded_models.append((spec, model, cfg, threshold))
    reused_models = [(spec, threshold) for spec, model, _cfg, threshold in loaded_models
                     if model is None]
    missing_keys = [spec["key"] for spec, _threshold in reused_models]
    if missing_keys:
        print(f"  missing checkpoints for {', '.join(missing_keys)}; reusing those planes from existing SBW1 packs")
        samples_path = DATA_OUT / "samples.json"
        if not samples_path.exists():
            raise SystemExit("cannot reuse missing-checkpoint packed probability planes: "
                             f"missing lineage metadata {samples_path}")
        previous = json.loads(samples_path.read_text(encoding="utf-8"))
        _validate_preserved_model_lineage(
            previous, reused_models, _model_artifact_version())

    rows = man[man.split.isin(splits)].to_dict("records")
    if limit:
        rows = rows[:limit]
    print(f"  {len(rows)} scenes -> {IMG_OUT}")

    for i, r in enumerate(rows, 1):
        ts = int(r["timestamp"])
        scene_truth = None
        packed_probabilities = None
        for spec, model, cfg, _threshold in loaded_models:
            if model is None:
                if packed_probabilities is None:
                    pack_path = IMG_OUT / f"{ts}.sbw.gz"
                    if not pack_path.exists():
                        raise SystemExit(f"no checkpoint for {spec['key']} and no reusable pack for {ts}")
                    packed_probabilities, packed_gt = _read_existing_pack(
                        pack_path, list(VIEWER_MODEL_KEYS), (size, size))
                    bits = np.unpackbits(np.frombuffer(packed_gt, np.uint8), bitorder="big")
                    packed_truth = bits[:size * size].reshape(size, size).astype(bool)
                    if scene_truth is None:
                        scene_truth = packed_truth
                    elif not np.array_equal(scene_truth, packed_truth):
                        raise RuntimeError(f"ground truth mismatch between inference and pack for {ts}")
                plane = packed_probabilities[VIEWER_MODEL_KEYS.index(spec["key"])]
                _to_png(plane, IMG_OUT / f"{ts}_prob_{spec['key']}.png")
                continue
            x, y = dsmod.load_full_scene(cfg, r, norm)
            truth = y.astype(bool)
            ps = int(cfg["patch"]["size"])
            ov = float(cfg["eval"].get("overlap", 0.5))
            prob = engine.sliding_window_predict(model, x, device, ps, ov, False, gaussian=True)
            p_s = _downsample(prob, size)
            _to_png((p_s * 255).astype(np.uint8), IMG_OUT / f"{ts}_prob_{spec['key']}.png")
            truth_s = _downsample(truth.astype(np.float32), size).astype(bool)
            if scene_truth is None:
                scene_truth = truth_s
            elif not np.array_equal(scene_truth, truth_s):
                raise RuntimeError(f"target mismatch across viewer models/pack for {ts}")
        y_s = scene_truth.astype(np.uint8)
        _to_png((y_s * 255).astype(np.uint8), IMG_OUT / f"{ts}_gt.png")
        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")


def _validate_stage_request(todo, limit) -> None:
    """Reject a partial predict+presence rebuild before either stage writes."""
    if limit is not None and "predict" in todo and "presence" in todo:
        raise SystemExit("--limit cannot be combined with predict+presence: presence requires "
                         "the complete validation/test sample set. Use --only predict for a "
                         "limited smoke run.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="experiments,dataset,predict,presence")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--img-splits", default="test,val")
    ap.add_argument("--data-root", type=Path, default=ROOT.parent / "Data",
                    help="Raw radar root used by the GPU-free packs stage (default: ../Data)")
    ap.add_argument("--site-dir", type=Path, default=ROOT.parent / "sprucebudworm_progress.github.io",
                    help="Website checkout that receives generated data")
    args = ap.parse_args()
    configure_site(args.site_dir)
    todo = [s.strip() for s in args.only.split(",") if s.strip()]
    _validate_stage_request(todo, args.limit)
    if "experiments" in todo:
        stage_experiments()
    if "dataset" in todo:
        stage_dataset()
    if "predict" in todo:
        stage_predict(limit=args.limit)
    if "presence" in todo:
        stage_presence()
    if "images" in todo:
        stage_images(splits=tuple(args.img_splits.split(",")), limit=args.limit)
    if "packs" in todo:
        stage_packs(args.data_root, limit=args.limit)
    print("done")


if __name__ == "__main__":
    main()
