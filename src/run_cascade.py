"""Unattended sequence: one-model hard-gated AttnUNet, then Swin-Tiny, then eval.

  python -m src.run_cascade
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
        "one-model gated Attention U-Net (hard gate, S0 init)",
        [py, "-m", "src.run",
         "--base-config", "configs/base_config_cascade.yaml",
         "--experiments", "configs/experiments_cascade.yaml"],
    )
    rc2 = _run(
        "cascade classifier Swin-Tiny (balanced split)",
        [py, "-m", "src.classify",
         "--name", "cls_swin_tiny_bal",
         "--base-config", "configs/base_config_cascade_cls.yaml",
         "--experiments", "configs/experiments_cascade_cls.yaml"],
    )
    rc3 = _run(
        "cascade evaluation (S0 + Swin-Tiny vs one-model)",
        [py, "-m", "src.cascade"],
    )
    if rc1 or rc2 or rc3:
        sys.exit(1)


if __name__ == "__main__":
    main()
