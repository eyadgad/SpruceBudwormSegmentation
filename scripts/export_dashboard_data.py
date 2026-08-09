"""Generate every JSON/PNG asset the evaluation dashboard reads.

The website is static (no server), so all analysis happens here and is written
to ``sprucebudworm_progress.github.io/data/``. Nothing in the site is computed
from placeholder values: every number traces back to the manifest, the cached
targets, the experiment outputs under ``outputs/experiments``, or a forward pass
of a trained checkpoint performed by this script.

Stages (each can be run alone with --only):
  experiments  57 experiment configs/metrics/histories + parsed training logs
  dataset      split / year / night / time / target-area distributions
  predict      forward pass of the selected (and comparison) model over val+test,
               producing per-scene metrics, threshold sweeps, calibration
               histograms, connected components and radial error profiles
  images       prob / ground-truth / reflectivity PNGs for the sample explorer

Run:
  .venv\\Scripts\\python.exe scripts\\export_dashboard_data.py --only experiments,dataset
  .venv\\Scripts\\python.exe scripts\\export_dashboard_data.py --only predict,images
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SITE = ROOT / "sprucebudworm_progress.github.io"
DATA_OUT = SITE / "data"
IMG_OUT = DATA_OUT / "samples"

# The configuration selected by Experiments 1-5, and the runner-up used for
# model-vs-model comparison in the sample explorer.
SELECTED = "sweep_attunet_dbz0_e012345678_focaltv"
COMPARE = "sweep_unetpp_dbz0_e012345678_focaltv"

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
    print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size/1024:.0f} KB)")


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
    grid = {"h": 960, "w": 960, "pixel_m": 500, "radar": "XAM Val d'Irene, Quebec"}
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


def _load_model(name, device):
    from src import checkpoint as ckpt, config as cfgmod, paths
    from src.models import create_model
    base = cfgmod.load_base_config(str(ROOT / "configs" / "base_config.yaml"))
    exps = cfgmod.load_experiments(str(ROOT / "configs" / "experiments_elev.yaml"))
    exp = next(e for e in exps if e["name"] == name)
    cfg = cfgmod.resolve_experiment(base, exp)
    st = ckpt.load_checkpoint(ckpt.best_path(paths.checkpoint_dir(base), name), device)
    if st is None:
        raise SystemExit(f"no checkpoint for {name}")
    m = create_model(cfg).to(device)
    m.load_state_dict(st["model"])
    m.eval()
    res = json.loads((ROOT / "outputs" / "experiments" / f"{name}_result.json").read_text())
    return m, cfg, float(res["calibrated_threshold"])


def stage_predict(splits=("test", "val"), limit=None) -> None:
    print("[predict]")
    import torch
    from src import data_prep, dataset as dsmod, engine, metrics as M, config as cfgmod

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = cfgmod.load_base_config(str(ROOT / "configs" / "base_config.yaml"))
    man, norm = data_prep.load_artifacts(base)

    model, cfg, THR = _load_model(SELECTED, device)
    model2, cfg2, THR2 = _load_model(COMPARE, device)
    print(f"  selected={SELECTED} thr={THR}  compare={COMPARE} thr={THR2}")

    ps = int(cfg["patch"]["size"])
    ov = float(cfg["eval"].get("overlap", 0.5))
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
        x, y = dsmod.load_full_scene(cfg, r, norm)
        prob = engine.sliding_window_predict(model, x, device, ps, ov, False, gaussian=True)
        pred = prob > THR
        yb = y.astype(bool)
        is_pos = int(r["label"]) == 1

        tp = float((pred & yb).sum()); fp = float((pred & ~yb).sum())
        fn = float((~pred & yb).sum()); tn = float((~pred & ~yb).sum())
        eps = 1e-8
        rec = {
            "ts": int(r["timestamp"]), "split": split, "label": int(r["label"]),
            "year": int(r["year"]),
            "night": (r["night"] if isinstance(r.get("night"), str) else None),
            "hour": int(int(r["timestamp"]) % 10000 // 100),
            "thr": THR,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "gt_area": int(yb.sum()), "pred_area": int(pred.sum()),
            "prob_mean": _r(float(prob.mean()), 5),
            "prob_max": _r(float(prob.max()), 4),
        }
        if is_pos:
            rec.update({
                "dice": _r(2 * tp / (2 * tp + fp + fn + eps)),
                "iou": _r(tp / (tp + fp + fn + eps)),
                "precision": _r(tp / (tp + fp + eps)),
                "recall": _r(tp / (tp + fn + eps)),
                "accuracy": _r((tp + tn) / (tp + tn + fp + fn + eps), 5),
                "specificity": _r(tn / (tn + fp + eps), 5),
                "boundary_iou": _r(M.boundary_iou(pred, yb)),
            })
            rec.update({k: _r(v, 3) for k, v in M.surface_metrics(pred, yb, tau=2.0).items()})
            n_gt, gt_sizes = _components(yb)
            n_pr, pr_sizes = _components(pred)
            rec.update({"n_gt_regions": n_gt, "n_pred_regions": n_pr,
                        "gt_region_max": (gt_sizes[-1] if gt_sizes else 0),
                        "pred_region_max": (pr_sizes[-1] if pr_sizes else 0)})
            # mean radial distance of GT signal and of each error type
            if yb.any():
                rec["gt_dist_km"] = _r(float(dist_km[yb].mean()), 1)
            if (pred & ~yb).any():
                rec["fp_dist_km"] = _r(float(dist_km[pred & ~yb].mean()), 1)
            if (~pred & yb).any():
                rec["fn_dist_km"] = _r(float(dist_km[~pred & yb].mean()), 1)
        else:
            # negatives have no positive pixels: only a false-alarm rate is defined
            rec["bg_fp_rate"] = _r(float(pred.mean()), 6)

        # second model (same scene, same threshold rule) for model comparison
        x2, _ = dsmod.load_full_scene(cfg2, r, norm)
        prob2 = engine.sliding_window_predict(model2, x2, device, ps, ov, False, gaussian=True)
        pred2 = prob2 > THR2
        tp2 = float((pred2 & yb).sum()); fp2 = float((pred2 & ~yb).sum()); fn2 = float((~pred2 & yb).sum())
        if is_pos:
            rec["dice_cmp"] = _r(2 * tp2 / (2 * tp2 + fp2 + fn2 + eps))
        else:
            rec["bg_fp_rate_cmp"] = _r(float(pred2.mean()), 6)
        rec["pred_area_cmp"] = int(pred2.sum())

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

    # `image_splits` tells the site which scenes have stored pixel layers, so it
    # never advertises imagery that stage `images` was not run for.
    have_img = sorted({s for s in splits
                       if any((IMG_OUT / f"{r['timestamp']}_prob.png").exists()
                              for r in man[man.split == s].head(5).to_dict("records"))})
    _w("samples.json", {"generated": datetime.now().isoformat(timespec="seconds"),
                        "selected": SELECTED, "compare": COMPARE, "threshold": THR,
                        "threshold_cmp": THR2, "image_splits": have_img,
                        "samples": samples})

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


def stage_images(size=480, thumb=120, splits=("test",), limit=None) -> None:
    """Probability / ground-truth / reflectivity PNGs for the sample explorer.

    Probability is stored as an 8-bit map so the browser can re-threshold it
    interactively without shipping raw float arrays. Preview resolution is
    ``size`` (down from 960) -- the authoritative metrics in samples.json are
    always computed at full resolution by stage_predict.
    """
    print("[images]")
    import torch
    from src import data_prep, dataset as dsmod, engine, config as cfgmod

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = cfgmod.load_base_config(str(ROOT / "configs" / "base_config.yaml"))
    man, norm = data_prep.load_artifacts(base)
    model, cfg, THR = _load_model(SELECTED, device)
    ps = int(cfg["patch"]["size"]); ov = float(cfg["eval"].get("overlap", 0.5))

    rows = man[man.split.isin(splits)].to_dict("records")
    if limit:
        rows = rows[:limit]
    print(f"  {len(rows)} scenes -> {IMG_OUT.relative_to(ROOT)}")

    for i, r in enumerate(rows, 1):
        ts = int(r["timestamp"])
        x, y = dsmod.load_full_scene(cfg, r, norm)
        prob = engine.sliding_window_predict(model, x, device, ps, ov, False, gaussian=True)

        # reflectivity preview: th_e0 is channel 0, normalized; map to 0..255
        th = x[0]
        th = np.nan_to_num(th, nan=0.0)
        lo, hi = np.percentile(th, 1), np.percentile(th, 99.5)
        thn = np.clip((th - lo) / max(hi - lo, 1e-6), 0, 1)

        p_s = _downsample(prob, size)
        y_s = _downsample(y.astype(np.float32), size)
        t_s = _downsample(thn, size)
        _to_png((p_s * 255).astype(np.uint8), IMG_OUT / f"{ts}_prob.png")
        _to_png((y_s * 255).astype(np.uint8), IMG_OUT / f"{ts}_gt.png")
        _to_png((t_s * 255).astype(np.uint8), IMG_OUT / f"{ts}_th.png")
        # small thumbnail for the grid: prediction at the selected threshold
        _to_png((_downsample((prob > THR).astype(np.float32), thumb) * 255).astype(np.uint8),
                IMG_OUT / f"{ts}_thumb.png")
        if i % 20 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="experiments,dataset,predict,images")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--img-splits", default="test")
    args = ap.parse_args()
    todo = [s.strip() for s in args.only.split(",") if s.strip()]
    if "experiments" in todo:
        stage_experiments()
    if "dataset" in todo:
        stage_dataset()
    if "predict" in todo:
        stage_predict(limit=args.limit)
    if "images" in todo:
        stage_images(splits=tuple(args.img_splits.split(",")), limit=args.limit)
    print("done")


if __name__ == "__main__":
    main()
