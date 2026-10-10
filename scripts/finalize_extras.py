"""Score frozen extra segmenters on TEST using each run's saved training config.

Does not retrain or refit thresholds. Uses outputs/*/*_config.json so checkpoints
are found in night_split / night_cascade / night_gated.

    .venv\\Scripts\\python.exe scripts\\finalize_extras.py --name transunet_night
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data_prep
from src.finalize import finalize_one

CONFIGS = {
    "night_bal_attunet9_cur": ROOT / "outputs/night_split/experiments/night_bal_attunet9_cur_config.json",
    "night_mtl_attunet9_cur": ROOT / "outputs/night_split/experiments/night_mtl_attunet9_cur_config.json",
    "night_tstack_attunet9_cur": ROOT / "outputs/night_split/experiments/night_tstack_attunet9_cur_config.json",
    "night_ltae_attunet9_cur": ROOT / "outputs/night_split/experiments/night_ltae_attunet9_cur_config.json",
    "gated_attn_unet_cur": ROOT / "outputs/night_cascade/experiments/gated_attn_unet_cur_config.json",
    "transunet_night": ROOT / "outputs/night_gated/experiments/transunet_night_config.json",
    "gated_swin_attn_night": ROOT / "outputs/night_gated/experiments/gated_swin_attn_night_config.json",
    "swin_unet_night": ROOT / "outputs/night_gated/experiments/swin_unet_night_config.json",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(CONFIGS))
    args = ap.parse_args()
    cfg_path = CONFIGS[args.name]
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} {args.name}", flush=True)
    manifest, norm_stats = data_prep.load_artifacts(cfg)
    finalize_one(cfg, manifest, norm_stats, device, verbose=True)
    print(f"DONE extras {args.name}", flush=True)


if __name__ == "__main__":
    main()
