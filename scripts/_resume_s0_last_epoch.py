"""One-shot: resume night_base_attunet9 at epoch 99 from final.pt so it can run 100/100."""
from __future__ import annotations

import json

import _bootstrap  # noqa: F401

import pandas as pd
import torch

from src import checkpoint as ckpt
from src import config as cfgmod
from src import engine, paths
from src.experiment import _save_resume
from src.models import create_model

NAME = "night_base_attunet9"
base = cfgmod.load_base_config("configs/base_config_night.yaml")
exp = next(e for e in cfgmod.load_experiments("configs/experiments_night.yaml") if e["name"] == NAME)
cfg = cfgmod.resolve_experiment(base, exp)
ckpt_dir = paths.checkpoint_dir(cfg)
exp_dir = paths.experiments_dir(cfg)

model = create_model(cfg)
final = ckpt.load_checkpoint(ckpt.final_path(ckpt_dir, NAME), "cpu")
if final is None:
    raise SystemExit("no final checkpoint")
model.load_state_dict(final["model"])
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=float(cfg["train"]["lr"]),
    weight_decay=float(cfg["train"].get("weight_decay", 1e-5)),
)
scheduler = engine.build_scheduler(optimizer, cfg)
for _ in range(99):
    scheduler.step()

hist_path = exp_dir / f"{NAME}_history.csv"
history = pd.read_csv(hist_path).to_dict("records")
result = json.loads((exp_dir / f"{NAME}_result.json").read_text(encoding="utf-8"))
_save_resume(
    ckpt_dir, NAME, model, optimizer, scheduler, None,
    99, float(result["best_val_dice_patch"]), int(result["best_epoch"]),
    50, False, history, cfg,
)
(exp_dir / f"{NAME}_result.json").unlink()
print(f"wrote resume at epoch 99; removed result.json; lr={optimizer.param_groups[0]['lr']:.3e}")
