"""Orchestrate all experiments and emit one comparison table.

Runs experiments sequentially on the single GPU (skipping any already finished),
then writes ``outputs/comparison_table.{csv,md}`` ranking every architecture/config
by held-out full-scene Dice, with the reference-notebook baseline as anchor rows.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from . import config as cfgmod
from . import data_prep, paths

# Reference-notebook baselines (old-budworm-training.ipynb) — the numbers every
# new architecture/config is compared against. Full-scene = sliding-window+TTA
# test Dice; RandomForest is pixel-level test Dice. See progress.md / Phase 1.
REFERENCE_BASELINES = [
    {"name": "REF: AttentionU-Net+Focal (notebook best DL)", "model": "attention_unet",
     "loss": "focal", "source": "reference_notebook",
     "test_dice": 0.6412, "test_iou": 0.4719, "note": "full-scene sliding-window+TTA"},
    {"name": "REF: RandomForest (notebook best ML)", "model": "random_forest",
     "loss": "-", "source": "reference_notebook",
     "test_dice": 0.6832, "test_iou": None, "note": "pixel-level test Dice"},
]


def ensure_artifacts(base_cfg: Dict, verbose=True) -> None:
    adir = paths.artifacts_dir(base_cfg)
    if (adir / "manifest.csv").exists() and (adir / "norm_stats.json").exists():
        return
    if verbose:
        print("[prepare] artifacts missing -> running data preparation ...")
    info = data_prep.prepare(base_cfg)
    if verbose:
        print(f"[prepare] {info['manifest_rows']} scenes, "
              f"{info['targets_written']} targets cached, split={info['split_summary']['counts']}")


def _load_finished_results(base_cfg: Dict, experiments) -> List[Dict]:
    """Load result JSONs for experiments that have already finished on disk."""
    import json
    from . import checkpoint as ckpt
    exp_dir = paths.experiments_dir(base_cfg)
    out = []
    for exp in experiments:
        name = exp["name"]
        rp = ckpt.result_path(exp_dir, name)
        if rp.exists():
            out.append(json.load(open(rp, encoding="utf-8")))
    return out


def build_table(results: List[Dict], base_cfg: Dict) -> pd.DataFrame:
    rows = []
    for r in results:
        ts = r["test_full_scene"]
        micro = ts.get("dice_micro")  # present only for runs evaluated with the updated engine
        rows.append({
            "experiment": r["name"], "model": r["model"], "loss": r["loss"],
            "n_channels": r["n_channels"], "n_params": r["n_params"],
            "val_dice_patch": round(r["best_val_dice_patch"], 4),
            "threshold": r["calibrated_threshold"],
            "test_dice_macro": round(ts["dice"], 4),
            "test_dice_micro": (round(micro, 4) if micro is not None else None),
            "test_precision": round(ts["precision"], 4), "test_recall": round(ts["recall"], 4),
            "boundary_iou": (round(ts["boundary_iou"], 4) if "boundary_iou" in ts else None),
            "nsd": (round(ts["nsd"], 4) if "nsd" in ts else None),
            "bg_fp_rate": (round(ts["bg_fp_rate"], 5) if ts["bg_fp_rate"] == ts["bg_fp_rate"] else None),
            "source": "this_framework",
        })
    for b in REFERENCE_BASELINES:
        rows.append({
            "experiment": b["name"], "model": b["model"], "loss": b["loss"],
            "n_channels": None, "n_params": None, "val_dice_patch": None, "threshold": None,
            "test_dice_macro": b["test_dice"], "test_dice_micro": None,
            "test_precision": None, "test_recall": None, "boundary_iou": None, "nsd": None,
            "bg_fp_rate": None, "source": b["source"] + " (macro, ~10 scenes)",
        })
    df = pd.DataFrame(rows).sort_values("test_dice_macro", ascending=False, na_position="last")
    return df.reset_index(drop=True)


def write_table(df: pd.DataFrame, base_cfg: Dict) -> Path:
    out = paths.output_dir(base_cfg)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "comparison_table.csv", index=False)
    with open(out / "comparison_table.md", "w", encoding="utf-8") as f:
        f.write("# Architecture comparison — held-out full-scene evaluation\n\n")
        f.write("Test Dice/IoU are averaged over positive test scenes (comparable to the\n")
        f.write("reference notebook). `bg_fp_rate` is the mean false-positive pixel fraction on\n")
        f.write("negative (all-background) test scenes. `REF:` rows are the reference-notebook\n")
        f.write("baselines this comparison is measured against.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    return out / "comparison_table.md"


def clean_run(base_cfg: Dict, verbose=True) -> None:
    """Remove previous experiment outputs + the stale split/normalization
    artifacts so a fresh end-to-end run retrains everything under the current
    config. Keeps the split-independent target cache (artifacts/targets/)."""
    import shutil
    for d in (paths.experiments_dir(base_cfg), paths.checkpoint_dir(base_cfg)):
        if d.exists():
            shutil.rmtree(d)
    out = paths.output_dir(base_cfg)
    if out.exists():
        for f in list(out.glob("comparison_table*")) + list(out.glob("ensemble_*.json")):
            f.unlink()
    adir = paths.artifacts_dir(base_cfg)
    for nm in ("manifest.csv", "norm_stats.json", "split_summary.json"):
        p = adir / nm
        if p.exists():
            p.unlink()
    if verbose:
        print("[fresh] cleared previous experiment outputs + stale artifacts "
              "(kept artifacts/targets/ cache)")


def run_all(base_config_path: str, experiments_path: str, verbose=True, fresh=False) -> pd.DataFrame:
    base_cfg = cfgmod.load_base_config(base_config_path)
    experiments = cfgmod.load_experiments(experiments_path)
    if fresh:
        clean_run(base_cfg, verbose=verbose)
    ensure_artifacts(base_cfg, verbose=verbose)  # runs data prep in-parent if needed

    if verbose and torch.cuda.is_available():
        print(f"[device] cuda ({torch.cuda.get_device_name(0)})")

    # Run each experiment in its OWN process. On limited VRAM a CUDA OOM corrupts
    # the process's CUDA context (even torch.cuda.empty_cache() then re-raises),
    # so an in-process loop lets one failure kill the whole sweep. A fresh
    # subprocess per experiment reclaims all GPU memory on exit and fully isolates
    # failures; the parent just orchestrates and never touches CUDA.
    import subprocess
    import sys
    from . import checkpoint as ckpt
    exp_dir = paths.experiments_dir(base_cfg)
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for exp in experiments:
        name = exp["name"]
        if ckpt.is_done(exp_dir, name):
            if verbose:
                print(f"[skip] {name}: result exists")
            continue
        if verbose:
            print(f"\n=== launching {name} (isolated subprocess) ===", flush=True)
        rc = subprocess.run(
            [sys.executable, "-m", "src.experiment", "--name", name,
             "--base-config", str(base_config_path), "--experiments", str(experiments_path)],
            env=env,
        ).returncode
        if rc != 0:
            print(f"[error] {name}: subprocess exited with code {rc} "
                  f"(continuing to the next experiment)")

    # Build the table from whatever finished on disk (robust to a mid-sweep crash).
    results = _load_finished_results(base_cfg, experiments)
    df = build_table(results, base_cfg)
    path = write_table(df, base_cfg)
    if verbose:
        print(f"\n[table] written to {path}\n")
        print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run all experiments end to end and emit comparison table")
    parser.add_argument("--base-config", type=str, default="configs/base_config.yaml",
                        help="Path to base config YAML file")
    parser.add_argument("--experiments", type=str, default="configs/experiments.yaml",
                        help="Path to experiments YAML file")
    parser.add_argument("--fresh", action="store_true",
                        help="clear previous outputs + stale artifacts, re-prep, and retrain everything")
    parser.add_argument("--ensemble", action="store_true",
                        help="after training, also compute the ensemble on the test split")
    args = parser.parse_args()

    run_all(args.base_config, args.experiments, verbose=True, fresh=args.fresh)
    if args.ensemble:
        from .evaluate import ensemble_evaluate
        ensemble_evaluate(args.base_config, args.experiments, split="test")
    # ONE COMMAND to rerun everything end to end (from the project root):
    #   python -m src.run --fresh --ensemble