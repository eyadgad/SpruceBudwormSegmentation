"""Train remaining publication-plan runs on the FROZEN night split.

Never rebuilds artifacts. Never passes --fresh. Skips any run that already
has a result JSON.

    .venv\\Scripts\\python.exe scripts\\run_plan_remaining.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

FROZEN = "170c442d1dd7b878cdb4157139f7b4dc178f6d0e078534b6dc291d3c7091222f"

STEPS = [
    ("S1–S3b ablation (current split)",
     [sys.executable, "-m", "src.run",
      "--base-config", "configs/base_config_night.yaml",
      "--experiments", "configs/experiments_night_ablation_cur.yaml"]),
    ("classifier candidates ConvNeXt / EffNet / ResNeXt",
     [sys.executable, "-m", "src.classify", "--all",
      "--base-config", "configs/base_config_cls.yaml",
      "--experiments", "configs/experiments_cls_candidates_cur.yaml"]),
    ("gated AttnUNet init from new-split s42",
     [sys.executable, "-m", "src.run",
      "--base-config", "configs/base_config_cascade.yaml",
      "--experiments", "configs/experiments_cascade_cur.yaml"]),
    ("frozen scan head on s42",
     [sys.executable, "-m", "src.frozen_cls",
      "--base-config", "configs/base_config_frozen_cls.yaml",
      "--experiments", "configs/experiments_frozen_cls_cur.yaml",
      "--name", "frozen_s0_cls_cur"]),
    ("TransUNet resume epoch 14/50",
     [sys.executable, "-m", "src.run",
      "--base-config", "configs/base_config_gated.yaml",
      "--experiments", "configs/experiments_transunet_resume.yaml"]),
]


def _hash() -> str:
    p = ROOT / "artifacts_night" / "manifest.sha256"
    if not p.exists():
        raise SystemExit("[stop] artifacts_night/manifest.sha256 missing — pull, do not rebuild")
    return p.read_text(encoding="utf-8").strip()


def main() -> None:
    if "--fresh" in sys.argv:
        raise SystemExit("[stop] --fresh is forbidden: it rebuilds the frozen split")
    got = _hash()
    if got != FROZEN:
        raise SystemExit(f"[stop] MANIFEST MISMATCH\n  local  {got}\n  frozen {FROZEN}")
    print(f"[ok] frozen split {got}", flush=True)
    py = sys.executable
    rc_all = 0
    for title, argv in STEPS:
        print(f"\n=== {title} ===", flush=True)
        print(" ".join(argv), flush=True)
        rc = subprocess.run(argv).returncode
        if rc != 0:
            print(f"[error] {title}: exit {rc}", flush=True)
            rc_all = rc
            # continue so later short jobs still run if a long one failed
    sys.exit(rc_all)


if __name__ == "__main__":
    main()
