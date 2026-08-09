"""Focused tests for the dashboard data pipeline.

Covers the two things that would silently corrupt the site: the export helpers
(JSON-safety, log parsing, radial binning, component counting) and the integrity
of the generated files against the training pipeline's own recorded results.

Run:  .venv\\Scripts\\python.exe scripts\\test_dashboard.py
Exits non-zero on the first failure, so it can gate a rebuild.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import export_dashboard_data as X  # noqa: E402

DATA = ROOT / "sprucebudworm_progress.github.io" / "data"
FAILS: list[str] = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  -> ' + str(detail)}")
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------- helpers
def test_num_json_safety():
    print("\n[export helpers] JSON safety")
    check("_num(nan) is None", X._num(float("nan")) is None)
    check("_num(inf) is None", X._num(float("inf")) is None)
    check("_num(None) is None", X._num(None) is None)
    check("_num('abc') is None", X._num("abc") is None)
    check("_num(np.float32) -> float", isinstance(X._num(np.float32(0.5)), float))
    check("_r rounds", X._r(0.123456, 3) == 0.123)
    check("_r(nan) is None", X._r(float("nan")) is None)


def test_radial_index():
    print("\n[export helpers] radial binning")
    idx, edges, dist = X._radial_index(h=100, w=100, n_ring=5, pixel_km=0.5)
    check("ring index shape", idx.shape == (100, 100), idx.shape)
    check("ring ids within range", idx.min() >= 0 and idx.max() <= 4, (idx.min(), idx.max()))
    check("edge count = n_ring+1", len(edges) == 6, len(edges))
    centre = dist[50, 50]
    corner = dist[0, 0]
    check("centre nearer than corner", centre < corner, (centre, corner))
    check("centre ring is 0", idx[50, 50] == 0, idx[50, 50])


def test_components():
    print("\n[export helpers] connected components")
    m = np.zeros((60, 60), bool)
    m[5:20, 5:20] = True          # 225 px
    m[40:55, 40:55] = True        # 225 px
    m[0, 59] = True               # 1 px, below the min size
    n, sizes = X._components(m, min_size=10)
    check("two regions above min size", n == 2, (n, sizes))
    check("sizes correct", sizes == [225, 225], sizes)
    n0, s0 = X._components(np.zeros((10, 10), bool))
    check("empty mask -> 0 regions", n0 == 0 and s0 == [], (n0, s0))


def test_downsample():
    print("\n[export helpers] block-max downsample")
    a = np.zeros((8, 8), np.float32)
    a[3, 3] = 1.0
    d = X._downsample(a, 4)
    check("shape halved", d.shape == (4, 4), d.shape)
    check("thin signal preserved by max", d.max() == 1.0, d.max())
    check("no-op when already small", X._downsample(a, 8).shape == (8, 8))


def test_log_parse():
    print("\n[export helpers] training-log parsing")
    logs = sorted((ROOT / "outputs" / "experiments").glob("*_train.log"))
    if not logs:
        check("log files present", False, "none found")
        return
    info = X.parse_log(logs[0])
    check("duration parsed", isinstance(info["train_seconds"], int) and info["train_seconds"] > 0, info["train_seconds"])
    check("per-epoch times parsed", len(info["epoch_seconds"]) > 0, len(info["epoch_seconds"]))
    check("epoch times positive", all(s > 0 for s in info["epoch_seconds"]))
    check("missing file is handled", X.parse_log(ROOT / "does_not_exist.log")["train_seconds"] is None)


# ---------------------------------------------------------------- artefacts
def load(name):
    p = DATA / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def test_files_exist():
    print("\n[artefacts] generated files")
    for n in ["experiments", "histories", "dataset", "summary", "samples", "threshold"]:
        check(f"{n}.json exists", (DATA / f"{n}.json").exists())


def test_json_finite():
    print("\n[artefacts] no NaN/Infinity leaked into JSON")
    for n in ["experiments", "dataset", "summary", "samples", "threshold"]:
        p = DATA / f"{n}.json"
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        bad = [t for t in ("NaN", "Infinity", "-Infinity") if t in txt]
        check(f"{n}.json is strict JSON", not bad, bad)


def test_samples_match_training():
    """The dashboard must reproduce the training pipeline's own test metrics."""
    print("\n[integrity] recomputed metrics match outputs/experiments/*_result.json")
    sm = load("samples")
    if sm is None:
        check("samples.json present", False)
        return
    name = sm["selected"]
    res = json.loads((ROOT / "outputs" / "experiments" / f"{name}_result.json").read_text())
    ref = res["test_full_scene"]
    pos = [s for s in sm["samples"] if s["split"] == "test" and s["label"] == 1]
    neg = [s for s in sm["samples"] if s["split"] == "test" and s["label"] == 0]

    check("test positive count", len(pos) == ref["n_pos_scenes"], (len(pos), ref["n_pos_scenes"]))
    check("test negative count", len(neg) == ref["n_neg_scenes"], (len(neg), ref["n_neg_scenes"]))

    def close(a, b, tol=5e-4):
        return abs(a - b) <= tol

    for key in ["dice", "iou", "precision", "recall", "boundary_iou", "nsd"]:
        # surface metrics are undefined (None) when a mask has no boundary; the
        # training pipeline uses nanmean, so the same scenes must be skipped here
        vals = [s[key] for s in pos if s.get(key) is not None]
        check(f"macro {key} has values", len(vals) > 0, key)
        got = float(np.mean(vals))
        check(f"macro {key}", close(got, ref[key]), f"got {got:.5f} vs {ref[key]:.5f} over n={len(vals)}")

    TP = sum(s["tp"] for s in pos); FP = sum(s["fp"] for s in pos); FN = sum(s["fn"] for s in pos)
    micro = 2 * TP / (2 * TP + FP + FN)
    check("micro dice", close(micro, ref["dice_micro"]), f"got {micro:.5f} vs {ref['dice_micro']:.5f}")
    bg = float(np.mean([s["bg_fp_rate"] for s in neg]))
    check("bg_fp_rate", close(bg, ref["bg_fp_rate"], 1e-5), f"got {bg:.6f} vs {ref['bg_fp_rate']:.6f}")

    thr_used = {s["thr"] for s in sm["samples"]}
    check("single threshold used", thr_used == {res["calibrated_threshold"]}, thr_used)


def test_sample_consistency():
    print("\n[integrity] per-scene arithmetic is self-consistent")
    sm = load("samples")
    if sm is None:
        return
    pos = [s for s in sm["samples"] if s["label"] == 1]
    npx = 960 * 960
    bad_total = [s["ts"] for s in pos if s["tp"] + s["fp"] + s["fn"] + s["tn"] != npx]
    check("TP+FP+FN+TN = 921600 for every scene", not bad_total, bad_total[:3])
    bad_area = [s["ts"] for s in pos if s["tp"] + s["fn"] != s["gt_area"]]
    check("TP+FN = truth area", not bad_area, bad_area[:3])
    bad_pred = [s["ts"] for s in pos if s["tp"] + s["fp"] != s["pred_area"]]
    check("TP+FP = predicted area", not bad_pred, bad_pred[:3])
    bad_dice = []
    for s in pos:
        d = 2 * s["tp"] / (2 * s["tp"] + s["fp"] + s["fn"]) if s["tp"] else 0.0
        if abs(d - s["dice"]) > 1e-3:
            bad_dice.append((s["ts"], d, s["dice"]))
    check("stored Dice matches its own TP/FP/FN", not bad_dice, bad_dice[:2])
    rng = [s["ts"] for s in pos if not (0 <= s["dice"] <= 1 and 0 <= s["precision"] <= 1)]
    check("metrics within [0,1]", not rng, rng[:3])


def test_dataset_split_matches_manifest():
    print("\n[integrity] dataset.json matches the manifest and split summary")
    ds = load("dataset")
    if ds is None:
        return
    cnt = ds["split_summary"]["counts"]
    for sp in ["train", "val", "test"]:
        p = sum(1 for s in ds["scenes"] if s["split"] == sp and s["label"] == 1)
        n = sum(1 for s in ds["scenes"] if s["split"] == sp and s["label"] == 0)
        check(f"{sp} positives", p == cnt[sp]["positives"], (p, cnt[sp]["positives"]))
        check(f"{sp} negatives", n == cnt[sp]["negatives"], (n, cnt[sp]["negatives"]))
    lk = ds["leakage"]
    check("leakage counts are consistent",
          lk["test_scenes_night_in_train"] <= lk["test_scenes_total"], lk)
    check("label thresholds are nested (dbz5 <= dbz0 <= isfinite)",
          all((s.get("area_dbz5") or 0) <= (s.get("area") or 0) <= (s.get("area_isfinite") or 0)
              for s in ds["scenes"] if s["label"] == 1 and s.get("area") is not None))


def test_summary_matches_dataset():
    print("\n[integrity] summary.json agrees with dataset.json")
    ds, su = load("dataset"), load("summary")
    if ds is None or su is None:
        return
    check("scene count", su["n_scenes"] == len(ds["scenes"]), (su["n_scenes"], len(ds["scenes"])))
    check("leakage block identical", su["leakage"] == ds["leakage"])
    check("split summary identical", su["split_summary"] == ds["split_summary"])


def test_threshold_file():
    print("\n[integrity] threshold.json")
    th = load("threshold")
    if th is None:
        return
    check("selected threshold present in sweep",
          any(abs(c["t"] - th["selected_threshold"]) < 1e-9 for c in th["curves"]["test"]))
    for sp, curve in th["curves"].items():
        mono = all(curve[i]["recall"] >= curve[i + 1]["recall"] - 1e-9 for i in range(len(curve) - 1))
        check(f"{sp}: recall is non-increasing with threshold", mono)
    d = th["distributions"]["test"]
    check("histogram bins align", len(d["centers"]) == len(d["pos"]) == len(d["neg"]))
    check("reliability bins carry counts", all(r["n"] > 0 for r in d["reliability"]))
    r = th["radial"]["test"]
    check("radial arrays align", len(r["tp"]) == len(r["fp"]) == len(r["fn"]) == len(r["gt"]) == len(r["edges_km"]) - 1)


def test_experiments_file():
    print("\n[integrity] experiments.json")
    ex = load("experiments")
    if ex is None:
        return
    rows = ex["experiments"]
    check("one and only one selected run", sum(1 for r in rows if r["selected"]) == 1)
    names = [r["name"] for r in rows]
    check("names unique", len(set(names)) == len(names))
    check("every run has a result on test", all(r["dice"] is not None for r in rows))
    sel = next(r for r in rows if r["selected"])
    res = json.loads((ROOT / "outputs" / "experiments" / f"{sel['name']}_result.json").read_text())
    # experiments.json stores metrics rounded to 4 dp on purpose (file size),
    # so compare at that precision rather than exactly
    check("selected dice matches result.json",
          abs(sel["dice"] - res["test_full_scene"]["dice"]) < 5e-5,
          (sel["dice"], res["test_full_scene"]["dice"]))
    check("selected best_epoch matches", sel["best_epoch"] == res["best_epoch"])

    # The selection claim shown on the site: leads on test Dice, not on everything.
    lead_dice = max(rows, key=lambda r: r["dice"])
    check("selected run leads test Dice", lead_dice["name"] == sel["name"])
    others = [k for k in ["boundary_iou", "nsd", "best_val_dice_patch"]
              if max((r for r in rows if r.get(k) is not None), key=lambda r: r[k])["name"] != sel["name"]]
    check("site's 'does not lead everything' claim holds", len(others) > 0,
          "selected unexpectedly leads all metrics; update the wording in experiments.js")


def test_images():
    print("\n[artefacts] sample imagery")
    sm = load("samples")
    d = DATA / "samples"
    if sm is None or not d.exists():
        check("image folder exists", False)
        return
    declared = sm.get("image_splits")
    check("samples.json declares image_splits", bool(declared), declared)

    # every scene in a declared split must have all four layers, or the site
    # would advertise imagery it cannot load
    expected = [s["ts"] for s in sm["samples"] if s["split"] in (declared or [])]
    missing = [f"{t}_{suf}" for t in expected for suf in ("prob", "gt", "th", "thumb")
               if not (d / f"{t}_{suf}.png").exists()]
    check(f"all {len(expected)} scenes in {declared} have four layers", not missing, missing[:4])
    check("file count = 4 per declared scene",
          len(list(d.glob("*.png"))) == 4 * len(expected),
          (len(list(d.glob("*.png"))), 4 * len(expected)))

    # and nothing outside the declared splits claims coverage
    undeclared = [s["ts"] for s in sm["samples"] if s["split"] not in (declared or [])]
    stray = [t for t in undeclared if (d / f"{t}_prob.png").exists()]
    check("no imagery outside the declared splits", not stray, stray[:3])


def main():
    print("dashboard pipeline tests")
    for fn in [test_num_json_safety, test_radial_index, test_components, test_downsample,
               test_log_parse, test_files_exist, test_json_finite, test_samples_match_training,
               test_sample_consistency, test_dataset_split_matches_manifest,
               test_summary_matches_dataset, test_threshold_file, test_experiments_file,
               test_images]:
        try:
            fn()
        except Exception as e:  # a crashing test is a failing test
            print(f"  ERROR in {fn.__name__}: {e}")
            FAILS.append(fn.__name__)
    print("\n" + ("-" * 60))
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
