"""Score the held-out TEST split, once, after every decision is frozen.

Training runs under ``eval.defer_test: true`` never touch test: they calibrate
the probability threshold on validation positives and report ``val_full_scene``.
This module is the single, deliberate place where test is read.

It writes ``<name>_final_result.json`` and does NOT modify ``<name>_result.json``
-- ``src.run`` treats the presence of the latter as "already trained, skip", so
mutating it in place would be indistinguishable from a finished training run.

    python -m src.finalize --base-config configs/base_config_night.yaml \
        --experiments configs/experiments_night.yaml --names night_base_attunet9,...

Reusing the threshold calibrated during training is deliberate: recalibrating
here would be selecting a hyperparameter on the test set.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch

from . import checkpoint as ckpt
from . import config as cfgmod
from . import data_prep, engine, paths
from .models import create_model


def load_best_or_final(ckpt_dir, name: str, device):
    """Best checkpoint, falling back to the final one.

    ``_best.pt`` is only written when an epoch beats ``train.es_tolerance``. A run
    that never improves (a diverged or very short run) therefore has no best
    checkpoint, and ``run_experiment`` scores whatever weights it ended with.
    Returns ``(state, "best"|"final"|None)`` so callers can say which they used.
    """
    state = ckpt.load_checkpoint(ckpt.best_path(ckpt_dir, name), device)
    if state is not None:
        return state, "best"
    state = ckpt.load_checkpoint(ckpt.final_path(ckpt_dir, name), device)
    return (state, "final") if state is not None else (None, None)


def finalize_one(cfg: Dict, manifest, norm_stats, device, verbose: bool = True) -> Dict | None:
    """Evaluate one experiment's best checkpoint on test at its frozen threshold."""
    name = cfg["name"]
    exp_dir = paths.experiments_dir(cfg)
    ckpt_dir = paths.checkpoint_dir(cfg)

    result_path = ckpt.result_path(exp_dir, name)
    if not result_path.exists():
        print(f"[skip] {name}: no training result yet")
        return None
    with open(result_path, "r", encoding="utf-8") as f:
        train_result = json.load(f)

    final_path = exp_dir / f"{name}_final_result.json"
    if final_path.exists():
        if verbose:
            print(f"[skip] {name}: already finalized")
        with open(final_path, "r", encoding="utf-8") as f:
            return json.load(f)

    state, weights = load_best_or_final(ckpt_dir, name, device)
    if state is None:
        print(f"[skip] {name}: no checkpoint")
        return None
    if weights == "final":
        # No epoch ever beat train.es_tolerance, so no best checkpoint was written
        # and run_experiment already scored the final weights. Match that rather
        # than silently skipping the run.
        print(f"[warn] {name}: no best checkpoint; using final weights")
    model = create_model(cfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    threshold = float(train_result["calibrated_threshold"])
    test_rows = manifest[manifest["split"] == "test"].to_dict("records")
    if verbose:
        print(f"[finalize] {name}: {len(test_rows)} test scenes at frozen threshold {threshold}")
    test_metrics = engine.evaluate_full_scene(
        model, test_rows, cfg, norm_stats, device, threshold=threshold,
        tta=bool(cfg["eval"].get("tta", True)))

    per_scene = test_metrics.pop("per_scene", None)
    out = dict(train_result)
    out["test_full_scene"] = test_metrics
    if per_scene is not None:
        per_path = exp_dir / f"{name}_test_per_scene.json"
        with open(per_path, "w", encoding="utf-8") as f:
            json.dump(per_scene, f, indent=2, default=str)
    out["threshold_source"] = "calibrated on validation during training; not refit on test"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    if verbose:
        m = test_metrics
        print(f"[finalize] {name}: test dice(macro)={m['dice']:.4f} "
              f"dice(micro)={m.get('dice_micro', float('nan')):.4f} "
              f"precision={m['precision']:.4f} recall={m['recall']:.4f} "
              f"bg_fp_rate={m.get('bg_fp_rate', float('nan')):.5f}")
    return out


def finalize(base_config_path: str, experiments_path: str, names: List[str] | None = None,
             verbose: bool = True) -> Dict[str, Dict]:
    base = cfgmod.load_base_config(base_config_path)
    experiments = cfgmod.load_experiments(experiments_path)
    if names:
        wanted = set(names)
        unknown = wanted - {e["name"] for e in experiments}
        if unknown:
            raise SystemExit(f"unknown experiment(s): {sorted(unknown)}")
        experiments = [e for e in experiments if e["name"] in wanted]

    manifest, norm_stats = data_prep.load_artifacts(base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: Dict[str, Dict] = {}
    for exp in experiments:
        cfg = cfgmod.resolve_experiment(base, exp)
        res = finalize_one(cfg, manifest, norm_stats, device, verbose=verbose)
        if res is not None:
            out[exp["name"]] = res
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score the held-out test split once, after freezing")
    ap.add_argument("--base-config", default="configs/base_config_night.yaml")
    ap.add_argument("--experiments", default="configs/experiments_night.yaml")
    ap.add_argument("--names", default=None,
                    help="comma-separated experiment names (default: all in the YAML)")
    args = ap.parse_args()
    names = [n.strip() for n in args.names.split(",")] if args.names else None
    finalize(args.base_config, args.experiments, names=names, verbose=True)
