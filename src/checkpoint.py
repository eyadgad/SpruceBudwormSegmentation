"""Checkpoint & resume helpers.

Per experiment ``<name>``:
  <name>_resume.pt   full training state, overwritten periodically DURING training
  <name>_best.pt     model at highest val Dice, written the moment it's beaten
  <name>_final.pt    model after the last epoch
  <name>_result.json final metrics; its presence means "done, skip retrain"

On clean completion the resume file is deleted. On restart, if a resume file
exists training continues from it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def _p(directory: Path, name: str, suffix: str) -> Path:
    return directory / f"{name}{suffix}"


def resume_path(ckpt_dir: Path, name: str) -> Path:
    return _p(ckpt_dir, name, "_resume.pt")


def best_path(ckpt_dir: Path, name: str) -> Path:
    return _p(ckpt_dir, name, "_best.pt")


def final_path(ckpt_dir: Path, name: str) -> Path:
    return _p(ckpt_dir, name, "_final.pt")


def result_path(exp_dir: Path, name: str) -> Path:
    return _p(exp_dir, name, "_result.json")


def is_done(exp_dir: Path, name: str) -> bool:
    return result_path(exp_dir, name).exists()


def load_result(exp_dir: Path, name: str) -> Dict:
    with open(result_path(exp_dir, name), "r", encoding="utf-8") as f:
        return json.load(f)


def save_result(exp_dir: Path, name: str, result: Dict) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(result_path(exp_dir, name), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)


def rng_state() -> Dict:
    state = {
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict) -> None:
    """Restore RNG state. torch.set_rng_state requires a CPU ByteTensor, but
    load_checkpoint's map_location may have moved it (and the cuda states) onto
    the training device along with the model weights -> move back explicitly.
    """
    if not state:
        return
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])


def save_model(path: Path, *, model, epoch, val_dice, cfg, threshold=0.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "epoch": epoch,
                "val_dice": val_dice, "threshold": threshold, "cfg": cfg}, path)


def load_checkpoint(path: Path, map_location) -> Optional[Dict]:
    if not path.exists():
        return None
    return torch.load(path, map_location=map_location, weights_only=False)
