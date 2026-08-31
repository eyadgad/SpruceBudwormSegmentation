"""One-time data preparation (run before any training).

Produces, under ``artifacts/``:
  manifest.csv      one row per usable scene (positives + sampled negatives),
                    with locally-resolved paths, night, split, target path.
  split_summary.json  year->split assignment and per-split counts.
  norm_stats.json   per-channel z-score mean/std computed from TRAIN scenes only.
  targets/y_<ts>.npz  cached raw dBZ mask slice for each positive (NaN = background).

Separating this fiddly, verified-once logic from training keeps the tricky
mask-indexing in a single place and makes training fast and reproducible.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import netCDF4 as nc

from . import channels, nights, paths

# ----------------------------------------------------------------------------
# Year-balanced split
# ----------------------------------------------------------------------------

def build_year_split(counts: Dict[int, int], fractions: Dict[str, float],
                     explicit: Dict[str, List[int]] | None, seed: int) -> Dict[int, str]:
    """Assign whole years to train/val/test so positive counts match target
    fractions as closely as possible (greedy). No year appears in two splits.

    ``explicit`` (optional) pins specific years to specific splits and overrides
    the greedy assignment for those years.
    """
    assignment: Dict[int, str] = {}
    explicit = explicit or {}
    pinned = set()
    for split, years in explicit.items():
        for y in (years or []):
            assignment[int(y)] = split
            pinned.add(int(y))

    total = sum(counts.values())
    targets = {s: fractions[s] * total for s in ("train", "val", "test")}
    running = {s: sum(counts[y] for y, sp in assignment.items() if sp == s)
               for s in ("train", "val", "test")}

    # Assign remaining years largest-first to whichever split is most under target.
    remaining = sorted((y for y in counts if y not in pinned),
                       key=lambda y: counts[y], reverse=True)
    rng = np.random.default_rng(seed)
    for y in remaining:
        deficits = {s: targets[s] - running[s] for s in ("train", "val", "test")}
        best = max(deficits, key=lambda s: (deficits[s], rng.random()))
        assignment[y] = best
        running[best] += counts[y]
    return assignment


# ----------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------

def _list_negatives(cfg: Dict, year: int) -> List[int]:
    d = paths.data_root(cfg) / str(year) / "negatives"
    if not d.exists():
        return []
    return sorted(paths.timestamp_from_name(p.name) for p in d.glob("*.nc"))


def stratified_split_labels(years, fractions: Dict[str, float], seed: int) -> np.ndarray:
    """Per-row split labels, stratified WITHIN each year so every year contributes
    to train/val/test at the requested fractions (default 70/20/10). Deterministic
    for a given seed, so the split is identical across all experiments."""
    years = np.asarray(years)
    labels = np.empty(len(years), dtype=object)
    rng = np.random.default_rng(seed)
    for y in np.unique(years):
        idx = np.where(years == y)[0]
        n = len(idx)
        perm = rng.permutation(n)
        n_tr = int(round(fractions["train"] * n))
        n_va = int(round(fractions["val"] * n))
        sel = np.array(["train"] * n, dtype=object)
        sel[perm[n_tr:n_tr + n_va]] = "val"
        sel[perm[n_tr + n_va:]] = "test"
        labels[idx] = sel
    return labels


def _max_elev_from_channels(channel_list: List[str]) -> int:
    """Return the highest TH/height elevation index referenced by a channel list,
    or -1 if none.  Used to determine the minimum tilt count a PPI file must
    have to be usable by any configured experiment."""
    mx = -1
    for spec in channel_list:
        kind, idx = channels.parse_spec(spec)
        if kind in ("th", "height") and idx is not None:
            mx = max(mx, idx)
    return mx


def build_manifest(cfg: Dict) -> pd.DataFrame:
    """Positives (from the CSV, paths re-resolved locally) + sampled negatives,
    with a split column.

    split.mode:
      'stratified' (default) -> per-year 70/20/10; each year in every split.
      'year'                 -> whole years held out (build_year_split).
      'night'                -> whole operational nights held out (see nights.py).
                                Leakage-free: consecutive 30-minute scans of one
                                moth exodus can no longer straddle splits.
    """
    csv = pd.read_csv(paths.matched_csv(cfg))
    split_cfg = cfg["split"]
    mode = split_cfg.get("mode", "stratified")
    seed = int(split_cfg["seed"])
    fractions = split_cfg["fractions"]

    # Assemble all scenes (positives + sampled negatives) as rows first.
    rows = []
    for _, r in csv.iterrows():
        ts, year = int(r["timestamp"]), int(r["year"])
        rows.append({"timestamp": ts, "year": year, "label": 1, "night": str(r["night"]),
                     "x_path": str(paths.ppi_file(cfg, year, 1, ts)),
                     "target_path": str(paths.targets_dir(cfg) / f"y_{ts}.npz")})

    neg_cfg = cfg.get("negatives", {"include": True, "ratio": 0.3})
    if neg_cfg.get("include", True):
        ratio = float(neg_cfg.get("ratio", 0.3))
        rng = np.random.default_rng(seed + 1)
        pos_per_year = csv.groupby("year").size().to_dict()
        for year in sorted(pos_per_year):
            negs = _list_negatives(cfg, int(year))
            budget = int(round(ratio * pos_per_year[year]))
            if budget <= 0 or not negs:
                continue
            sel = rng.choice(len(negs), size=min(budget, len(negs)), replace=False)
            for i in sorted(sel):
                ts = negs[i]
                rows.append({"timestamp": ts, "year": int(year), "label": 0, "night": "",
                             "x_path": str(paths.ppi_file(cfg, int(year), 0, ts)),
                             "target_path": ""})

    df = pd.DataFrame(rows)

    # The manifest's own ``night`` is the mask-file variable name and exists only
    # for positives; keep it as ``mask_night`` (cache_targets indexes the netCDF
    # by it) and make ``night`` the timestamp-derived operational night, which is
    # defined for negatives too.
    df["mask_night"] = df["night"].fillna("").astype(str)
    df["night"] = df["timestamp"].map(nights.night_id)

    # Assign splits.
    if mode == "stratified":
        df["split"] = stratified_split_labels(df["year"].values, fractions, seed)
    elif mode == "year":
        counts = csv.groupby("year").size().to_dict()
        assignment = build_year_split({int(k): int(v) for k, v in counts.items()},
                                      fractions, split_cfg.get("years"), seed)
        df["split"] = df["year"].map(assignment)
    elif mode == "night":
        df["split"] = nights.assign_night_split(df, fractions, seed)
        nights.verify_no_leakage(df)
    else:
        raise ValueError(f"split.mode must be 'stratified', 'year' or 'night', got {mode}")

    df = df.sort_values(["split", "timestamp"]).reset_index(drop=True)
    # Fail loudly if any radar file is missing (real data-loading boundary).
    missing = [p for p in df["x_path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} radar files missing, e.g. {missing[:3]}")

    # Drop scenes whose PPI has fewer elevation tilts than the channel superset
    # requires.  A handful of scans were stored with a truncated volume (e.g. only
    # the lowest 5 tilts), making channels like th_e5+ all-NaN for that scene.
    norm_ch = cfg.get("data_prep", {}).get("norm_channels") or cfg["channels"]
    min_tilts = _max_elev_from_channels(norm_ch) + 1  # e.g. th_e9 -> need 10 tilts
    if min_tilts > 0:
        def _n_tilts(path: str) -> int:
            with nc.Dataset(path) as ds:
                return int(ds.variables["TH"].shape[0])
        mask = df["x_path"].apply(_n_tilts) >= min_tilts
        n_dropped = (~mask).sum()
        if n_dropped:
            import warnings
            dropped_ts = df.loc[~mask, "timestamp"].tolist()
            warnings.warn(
                f"build_manifest: dropped {n_dropped} scene(s) with fewer than "
                f"{min_tilts} elevation tilts (timestamps: {dropped_ts})"
            )
            df = df[mask].reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------
# Target cache (positives only)
# ----------------------------------------------------------------------------

def _scan_timestamps(mask_ds, night_var: str) -> np.ndarray:
    scan_dim = mask_ds.variables[night_var].dimensions[0]
    return np.rint(mask_ds.variables[scan_dim][:]).astype(np.int64)


def cache_targets(cfg: Dict, df: pd.DataFrame, overwrite: bool = False) -> int:
    """Extract each positive's raw dBZ mask slice and cache as compressed npz.

    Slice lookup = (night variable, scan index whose timestamp == sample ts).
    Stored raw (NaN = background) so any target mode (isfinite / dbz>=t) can be
    derived later without re-extraction.

    Uses ``mask_night`` (the netCDF variable name), NOT ``night`` -- the latter is
    the timestamp-derived operational-night ID used for splitting.
    """
    tdir = paths.targets_dir(cfg)
    tdir.mkdir(parents=True, exist_ok=True)
    pos = df[df["label"] == 1]
    written = 0
    for year, grp in pos.groupby("year"):
        with nc.Dataset(paths.mask_file(cfg, int(year))) as mask:
            scan_ts_by_night = {}
            for _, r in grp.iterrows():
                out_path = Path(r["target_path"])
                if out_path.exists() and not overwrite:
                    continue
                night = r["mask_night"] if "mask_night" in r else r["night"]
                if night not in mask.variables:
                    raise KeyError(f"night '{night}' not a variable in {year} mask")
                if night not in scan_ts_by_night:
                    scan_ts_by_night[night] = _scan_timestamps(mask, night)
                idx = np.where(scan_ts_by_night[night] == int(r["timestamp"]))[0]
                if len(idx) == 0:
                    raise ValueError(f"ts {r['timestamp']} not found in night '{night}' ({year})")
                sl = _to_nan(mask.variables[night][int(idx[0])])
                np.savez_compressed(out_path, dbz=sl.astype(np.float32))
                written += 1
    return written


def _to_nan(arr) -> np.ndarray:
    return np.ma.filled(arr.astype(np.float32), np.nan) if np.ma.isMaskedArray(arr) else np.asarray(arr, np.float32)


# ----------------------------------------------------------------------------
# Normalization statistics (train split only)
# ----------------------------------------------------------------------------

def compute_norm_stats(cfg: Dict, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Per-channel z-score mean/std from TRAIN scenes only (prevents leakage).

    Static channels (dem, bh_e*) use their single grid. Radar channels
    (th_e*, height_e*) accumulate finite values over up to
    ``data_prep.norm_max_scenes`` training scenes, sampled deterministically.
    """
    # Compute stats for a superset so channel-varying experiments work without
    # re-prep. Defaults to the base channel list; set data_prep.norm_channels to
    # the union of every channel any experiment uses.
    superset = cfg["data_prep"].get("norm_channels") or cfg["channels"]
    channel_list = [c for c in superset if channels.norm_key(c) is not None]
    train = df[df["split"] == "train"].reset_index(drop=True)
    max_scenes = int(cfg["data_prep"]["norm_max_scenes"])
    if len(train) > max_scenes:
        rng = np.random.default_rng(cfg["split"]["seed"])
        train = train.iloc[np.sort(rng.choice(len(train), max_scenes, replace=False))]

    stats: Dict[str, Dict[str, float]] = {}
    for spec in channel_list:
        kind, _ = channels.parse_spec(spec)
        if spec == "dem" or kind == "bh":
            vals = channels.raw_channel_values(cfg, None, spec)
            stats[spec] = {"mean": float(np.mean(vals)), "std": float(np.std(vals) + 1e-8)}
    # Radar channels (incl. per-pixel statistical summaries): stream over scenes
    # with running sums.
    radar_specs = [s for s in channel_list if channels.parse_spec(s)[0] in ("th", "height", "stat")]
    if radar_specs:
        acc = {s: {"n": 0, "s": 0.0, "ss": 0.0} for s in radar_specs}
        for _, r in train.iterrows():
            for s in radar_specs:
                v = channels.raw_channel_values(cfg, r["x_path"], s)
                acc[s]["n"] += v.size
                acc[s]["s"] += float(v.sum())
                acc[s]["ss"] += float(np.square(v, dtype=np.float64).sum())
        for s in radar_specs:
            n = max(acc[s]["n"], 1)
            mean = acc[s]["s"] / n
            var = max(acc[s]["ss"] / n - mean * mean, 0.0)
            stats[s] = {"mean": float(mean), "std": float(np.sqrt(var) + 1e-8)}
    return stats


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def prepare(cfg: Dict, overwrite_targets: bool = False) -> Dict:
    adir = paths.artifacts_dir(cfg)
    adir.mkdir(parents=True, exist_ok=True)

    df = build_manifest(cfg)
    df.to_csv(adir / "manifest.csv", index=False)

    # Per-year x per-split positive counts confirm every year is in every split.
    per_year = {}
    for year in sorted(df["year"].unique()):
        per_year[int(year)] = {
            s: int(((df.year == year) & (df.split == s) & (df.label == 1)).sum())
            for s in ("train", "val", "test")
        }
    summary = {
        "mode": cfg["split"].get("mode", "stratified"),
        "fractions": cfg["split"]["fractions"],
        "counts": {
            s: {
                "positives": int(((df.split == s) & (df.label == 1)).sum()),
                "negatives": int(((df.split == s) & (df.label == 0)).sum()),
            } for s in ("train", "val", "test")
        },
        "positives_per_year_per_split": per_year,
    }
    # Night accounting is reported for every mode, so the leakage of a
    # scan-level split is visible in the artifact rather than only in a doc.
    summary["night_split"] = nights.night_split_report(df)
    with open(adir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    n_written = cache_targets(cfg, df, overwrite=overwrite_targets)

    stats = compute_norm_stats(cfg, df)
    with open(adir / "norm_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return {"manifest_rows": len(df), "targets_written": n_written,
            "split_summary": summary, "norm_channels": list(stats.keys())}


def load_artifacts(cfg: Dict):
    adir = paths.artifacts_dir(cfg)
    manifest = pd.read_csv(adir / "manifest.csv")
    manifest["night"] = manifest["night"].fillna("").astype(str)
    manifest["target_path"] = manifest["target_path"].fillna("").astype(str)
    if "mask_night" in manifest.columns:
        manifest["mask_night"] = manifest["mask_night"].fillna("").astype(str)
    else:
        # Manifests written before the night split carry only the mask variable
        # name in ``night``; keep them loadable.
        manifest["mask_night"] = manifest["night"]
    with open(adir / "norm_stats.json", "r", encoding="utf-8") as f:
        norm_stats = json.load(f)
    return manifest, norm_stats


if __name__ == "__main__":
    import argparse
    from . import config as cfgmod
    ap = argparse.ArgumentParser(description="One-time data preparation")
    ap.add_argument("--base-config", default="configs/base_config.yaml")
    ap.add_argument("--overwrite-targets", action="store_true")
    args = ap.parse_args()
    cfg = cfgmod.load_base_config(args.base_config)
    info = prepare(cfg, overwrite_targets=args.overwrite_targets)
    print(f"[prepare] {info['manifest_rows']} scenes; {info['targets_written']} targets cached")
    print(f"[prepare] split counts: {info['split_summary']['counts']}")
    print(f"[prepare] normalized channels: {info['norm_channels']}")
