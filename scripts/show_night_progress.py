"""Show progress of the night-split experiment chain.

    .venv\\Scripts\\python.exe scripts\\show_night_progress.py

Read-only: safe to run while training. Prints one line per experiment with its
current epoch, best validation Dice, recent epoch time and state.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import _bootstrap  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
EPOCH = re.compile(r"\|\s*(\d+)/(\d+)\s*\|.*?\|\s*([\d.]+)\s+.*?\|\s*([\d.]+)@(\d+)\s*\*?\s*\|.*?\|\s*(\d+)s\s*$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/night_split")
    args = ap.parse_args()

    exp_dir = ROOT / args.dir / "experiments"
    if not exp_dir.exists():
        raise SystemExit(f"no experiments yet under {exp_dir}")

    print(f"{'experiment':26s} {'epoch':>9s} {'best_val_dice':>14s} {'last_s':>7s}  state")
    print("-" * 78)
    for log in sorted(exp_dir.glob("*_train.log")):
        name = log.name[: -len("_train.log")]
        epoch = best = last = None
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = EPOCH.search(line.strip())
            if m:
                epoch = f"{m.group(1)}/{m.group(2)}"
                best = float(m.group(4))
                last = int(m.group(6))

        finished = (exp_dir / f"{name}_result.json").exists()
        finalized = (exp_dir / f"{name}_final_result.json").exists()
        resume = (ROOT / args.dir / "checkpoints" / f"{name}_resume.pt").exists()
        if finalized:
            state = "finalized (test scored)"
        elif finished:
            result = json.loads((exp_dir / f"{name}_result.json").read_text(encoding="utf-8"))
            split = "val" if result.get("defer_test") else "test"
            metrics = result.get("val_full_scene") or result.get("test_full_scene") or {}
            state = (f"done | thr={result['calibrated_threshold']} "
                     f"{split}_dice={metrics.get('dice', float('nan')):.4f}")
        elif resume:
            state = "running / resumable"
        else:
            state = "started, no snapshot yet"

        print(f"{name:26s} {epoch or '-':>9s} "
              f"{(f'{best:.4f}' if best is not None else '-'):>14s} "
              f"{(str(last) if last else '-'):>7s}  {state}")


if __name__ == "__main__":
    main()
