"""Mean ± SD over publication seed runs.

Strips ``_s\\d+`` from experiment names, reads ``*_final_result.json`` when
present else ``*_result.json`` (val-only under defer_test).

    .venv\\Scripts\\python.exe scripts\\aggregate_seeds.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from era_guard import is_current_era  # noqa: E402

SEED_RE = re.compile(r"_s(\d+)$")
KEYS = ("dice", "dice_micro", "dice_global", "iou", "precision", "recall",
        "nsd", "bf1", "bf1_fuzzy", "bg_fp_rate", "far_scan")


def _stem(name: str) -> str:
    return SEED_RE.sub("", name)


def _seed(name: str, result: dict):
    m = SEED_RE.search(name)
    if m:
        return int(m.group(1))
    return (result.get("train") or {}).get("seed")


def _metrics_block(result: dict) -> dict:
    return result.get("test_full_scene") or result.get("val_full_scene") or {}


def collect(exp_dirs, allow_superseded=False):
    """Group seed runs by family, skipping superseded-era runs.

    Without the era filter a development-screening run silently joins its own
    seed family: ``night_base_attunet9`` and ``night_base_attunet9_s42`` both
    stem to ``night_base_attunet9``, so an old-split number would be averaged
    into a published mean.
    """
    groups = defaultdict(list)
    skipped = []
    for exp_dir in exp_dirs:
        exp_dir = Path(exp_dir)
        if not exp_dir.is_dir():
            continue
        for path in sorted(exp_dir.glob("*_result.json")):
            if path.name.endswith("_final_result.json"):
                continue
            name = path.name.replace("_result.json", "")
            if not allow_superseded and not is_current_era(exp_dir / f"{name}_config.json"):
                skipped.append(name)
                continue
            final = exp_dir / f"{name}_final_result.json"
            source = final if final.exists() else path
            result = json.loads(source.read_text(encoding="utf-8"))
            block = _metrics_block(result)
            if not block:
                continue
            row = {"name": name, "family": _stem(name),
                   "source": source.name, "seed": _seed(name, result)}
            for k in KEYS:
                row[k] = block.get(k)
            groups[_stem(name)].append(row)
    if skipped:
        print(f"[era] skipped {len(skipped)} superseded-split run(s): "
              f"{', '.join(sorted(skipped))}", file=sys.stderr)
    return groups


def summarise(groups) -> pd.DataFrame:
    rows = []
    for family, items in sorted(groups.items()):
        n = len(items)
        rec = {"family": family, "n": n}
        for k in KEYS:
            vals = np.asarray([i[k] for i in items if i.get(k) is not None],
                              dtype=np.float64)
            if not vals.size:
                rec[f"{k}_mean"] = None
                rec[f"{k}_sd"] = None
                rec[f"{k}_min"] = None
                rec[f"{k}_max"] = None
                continue
            rec[f"{k}_mean"] = float(np.nanmean(vals))
            rec[f"{k}_sd"] = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
            rec[f"{k}_min"] = float(np.nanmin(vals))
            rec[f"{k}_max"] = float(np.nanmax(vals))
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*",
                    default=["outputs/night_split/experiments",
                             "outputs/night_cascade/experiments"])
    ap.add_argument("--out", default="outputs/night_split/comparison")
    ap.add_argument("--allow-superseded", action="store_true",
                    help="include old-split runs (development screening only; "
                         "never for publication numbers)")
    args = ap.parse_args()
    groups = collect(args.dirs, allow_superseded=args.allow_superseded)
    if not groups:
        raise SystemExit(
            "no current-era seed results found. The publication seed runs "
            "(night_base_attunet9_s*, unet_night_s*, cls_swin_tiny_bal*) have "
            "not been trained yet; pass --allow-superseded only to inspect "
            "development screening.")
    df = summarise(groups)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "seeds.csv", index=False)
    with open(out / "seeds.md", "w", encoding="utf-8") as f:
        f.write("# Seed aggregate (mean ± SD)\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    print(df.to_string(index=False))
    print(f"[done] {out / 'seeds.md'}")


if __name__ == "__main__":
    main()
