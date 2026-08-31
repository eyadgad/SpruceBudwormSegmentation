"""Rank the multi-task lambda probes on VALIDATION only.

    .venv\\Scripts\\python.exe scripts\\show_lambda_probes.py

Prints validation patch Dice and validation full-scene Dice per lambda, and names
the winner. Copy that lambda into ``lambda_cls`` for night_mtl_attunet9,
night_tstack_attunet9 and night_ltae_attunet9 in configs/experiments_night.yaml
before running the main chain.

The test set is not read here and must not be: lambda is a hyperparameter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/night_split/experiments")
    args = ap.parse_args()

    paths = sorted((ROOT / args.dir).glob("probe_lambda_*_result.json"))
    if not paths:
        raise SystemExit(f"no probe results yet under {ROOT / args.dir}")

    rows = []
    for p in paths:
        r = json.loads(p.read_text(encoding="utf-8"))
        cfg_path = p.with_name(p.name.replace("_result.json", "_config.json"))
        lam = None
        if cfg_path.exists():
            lam = json.loads(cfg_path.read_text(encoding="utf-8"))["loss"].get("lambda_cls")
        if "test_full_scene" in r:
            raise SystemExit(f"{p.name} contains test metrics; lambda must be chosen on "
                             f"validation only (expected eval.defer_test: true)")
        val = r.get("val_full_scene") or {}
        rows.append({"name": r["name"], "lambda": lam,
                     "val_dice_patch": r["best_val_dice_patch"],
                     "val_dice_scene": val.get("dice"),
                     "threshold": r.get("calibrated_threshold"),
                     "best_epoch": r.get("best_epoch")})

    print(f"{'lambda':>8s} {'val_dice_patch':>15s} {'val_dice_scene':>15s} "
          f"{'thr':>5s} {'best_ep':>8s}")
    print("-" * 58)
    for row in sorted(rows, key=lambda r: (r["lambda"] is None, r["lambda"])):
        scene = row["val_dice_scene"]
        print(f"{str(row['lambda']):>8s} {row['val_dice_patch']:>15.4f} "
              f"{(f'{scene:.4f}' if scene is not None else '-'):>15s} "
              f"{str(row['threshold']):>5s} {str(row['best_epoch']):>8s}")

    # Selection criterion is validation patch Dice, matching what drives early
    # stopping and best-checkpoint selection during training.
    best = max(rows, key=lambda r: r["val_dice_patch"])
    print(f"\nwinner: lambda={best['lambda']} "
          f"(val_dice_patch={best['val_dice_patch']:.4f}, from {best['name']})")
    print("set this as lambda_cls in configs/experiments_night.yaml for "
          "night_mtl_attunet9, night_tstack_attunet9 and night_ltae_attunet9")
    if len(rows) > 1:
        ordered = sorted(rows, key=lambda r: -r["val_dice_patch"])
        gap = ordered[0]["val_dice_patch"] - ordered[1]["val_dice_patch"]
        if gap < 0.005:
            print(f"note: top two lambdas differ by only {gap:.4f} val patch Dice — "
                  f"treat the choice as weakly determined and say so in the write-up")


if __name__ == "__main__":
    main()
