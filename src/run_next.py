"""Train the frozen-S0 scan head, then sweep hard-gate thresholds.

  python -m src.run_next
"""
from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _run(title: str, argv: list[str]) -> int:
    print(f"\n=== {title} ===", flush=True)
    rc = subprocess.run(argv).returncode
    if rc != 0:
        print(f"[error] {title}: exit {rc}", flush=True)
    return rc


def main() -> None:
    py = sys.executable
    rc1 = _run(
        "frozen S0 scan head (encoder locked, BCE)",
        [py, "-m", "src.frozen_cls"],
    )
    rc2 = _run(
        "high-recall hard-gate sweep (Swin + frozen head on S0 maps)",
        [py, "-m", "src.cascade", "--sweep"],
    )
    if rc1 or rc2:
        sys.exit(1)


if __name__ == "__main__":
    main()
