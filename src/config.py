"""Configuration loading, deep-merge, and validation.

Configs are plain nested dicts loaded from YAML. A run's effective config is
``base.yaml`` deep-merged with a single experiment's overrides. Keeping configs
as dicts (rather than dataclasses) makes the override/merge logic trivial and
keeps every knob in one place; validation happens once at load time.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Channel specs the framework knows how to build (see channels.py). ``th_e{i}``
# and ``height_e{i}`` accept any elevation index 0..23; ``bh_e{i}`` any 0..25.
_STATIC_CHANNELS = {"dem", "valid_mask"}
# statistical-summary channels over TH elevations 0..5 (see channels.STAT_FUNCS)
_STAT_CHANNELS = {"th_max", "th_med", "th_mean"}
_KNOWN_MODELS = {
    "unet", "attention_unet", "nnunet", "smp_unetpp", "smp_deeplabv3p", "smp_segformer",
    "gated_swin_attn", "swin_unet", "transunet",
}
_KNOWN_CLS_MODELS = {"convnext_tiny", "efficientnetv2_s", "swin_tiny", "resnext50"}
_KNOWN_LOSSES = {"dice", "dice_bce", "focal", "tversky", "focal_tversky", "bce", "dice_boundary"}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` onto a copy of ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_base_config(path: str | Path) -> Dict[str, Any]:
    cfg = load_yaml(path)
    validate_config(cfg)
    return cfg


def load_experiments(path: str | Path) -> List[Dict[str, Any]]:
    """Load the experiment list. File is ``{"experiments": [ {...}, ... ]}``."""
    doc = load_yaml(path)
    exps = doc.get("experiments", [])
    if not isinstance(exps, list) or not exps:
        raise ValueError(f"{path}: 'experiments' must be a non-empty list")
    names = [e.get("name") for e in exps]
    if any(n is None for n in names):
        raise ValueError(f"{path}: every experiment needs a 'name'")
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate experiment names: {names}")
    return exps


def _validate_channel(spec: str) -> None:
    if spec in _STATIC_CHANNELS or spec in _STAT_CHANNELS:
        return
    for prefix in ("th_e", "height_e", "bh_e"):
        if spec.startswith(prefix):
            idx = spec[len(prefix):]
            if idx.isdigit():
                return
    raise ValueError(
        f"Unknown channel spec '{spec}'. Valid: th_e<i>, height_e<i>, bh_e<i>, "
        f"dem, valid_mask. (Note: 'ETA_raw'/log_eta channels are NOT available "
        f"in the local dataset — see progress.md.)"
    )


def validate_config(cfg: Dict[str, Any]) -> None:
    """Fail fast on structural/domain errors at config-parse time."""
    for key in ("data", "split", "channels", "patch", "train", "eval", "model", "loss"):
        if key not in cfg:
            raise ValueError(f"Config missing required top-level key: '{key}'")

    channels = cfg["channels"]
    if not isinstance(channels, list) or not channels:
        raise ValueError("config['channels'] must be a non-empty list")
    for spec in channels:
        _validate_channel(spec)

    model_name = cfg["model"]["name"]
    if cfg.get("task") == "classify":
        if model_name not in _KNOWN_CLS_MODELS:
            raise ValueError(
                f"Unknown classifier '{model_name}'. Known: {sorted(_KNOWN_CLS_MODELS)}"
            )
        return

    if model_name not in _KNOWN_MODELS:
        raise ValueError(f"Unknown model '{model_name}'. Known: {sorted(_KNOWN_MODELS)}")

    loss_name = cfg["loss"]["name"]
    if loss_name not in _KNOWN_LOSSES:
        raise ValueError(f"Unknown loss '{loss_name}'. Known: {sorted(_KNOWN_LOSSES)}")

    mode = cfg["target"]["mode"]
    if mode not in ("isfinite", "threshold"):
        raise ValueError(f"config['target']['mode'] must be 'isfinite' or 'threshold', got {mode}")

    ps = cfg["patch"]["size"]
    if ps % 32 != 0:
        raise ValueError(f"patch.size must be a multiple of 32 (encoder-friendly), got {ps}")


def resolve_experiment(base: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    """Merge one experiment's overrides onto the base config and validate."""
    merged = deep_merge(base, {k: v for k, v in exp.items() if k != "name"})
    merged["name"] = exp["name"]
    validate_config(merged)
    return merged


def save_config(cfg: Dict[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, default=str)
