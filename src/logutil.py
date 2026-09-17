"""Lightweight per-experiment logging: writes to console and a log file.

Each experiment logs its config summary, an organized per-epoch table
(train loss + val metrics), and the final full-scene test metrics to
``outputs/experiments/<name>_train.log``.
"""
from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, log_file: str | Path | None = None, console: bool = True,
               level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(f"radar.{name}")
    logger.setLevel(level)
    logger.handlers.clear()        # avoid duplicate handlers when called repeatedly
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# Column layout for the per-epoch training table.
EPOCH_HEADER = (
    f"{'epoch':>9} | {'train_loss':>10} | {'val_dice':>8} {'val_iou':>8} "
    f"{'val_prec':>8} {'val_rec':>8} | {'best':>15} | {'lr':>8} | {'time':>6}"
)


def format_epoch(epoch: int, epochs: int, train_loss: float, val: dict,
                 best_dice: float, best_epoch: int, lr: float, secs: float,
                 improved: bool) -> str:
    star = " *" if improved else "  "
    return (f"{epoch + 1:4d}/{epochs:<4d} | {train_loss:10.4f} | "
            f"{val['dice']:8.4f} {val['iou']:8.4f} {val['precision']:8.4f} {val['recall']:8.4f} | "
            f"{best_dice:8.4f}@{best_epoch:<3d}{star} | {lr:8.2e} | {secs:5.0f}s")
