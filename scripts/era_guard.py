"""Fail if a main-paper run was trained on the superseded (unbalanced) split.

    .venv\\Scripts\\python.exe scripts\\era_guard.py

A config reaches the main paper only if negatives.balanced is true and
negatives.ratio is 1.0. Development-screening runs (S0–S3b, λ, gated hybrids)
are ignored unless listed with --names.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_PREFIXES = (
    "night_base_attunet9_s",
    "unet_night_s",
    "cls_swin_tiny_bal",
)


def _is_main(name: str) -> bool:
    return any(name.startswith(p) for p in MAIN_PREFIXES) or name == "cls_swin_tiny_bal"


def _configs(root: Path):
    for path in root.glob("outputs/**/experiments/*_config.json"):
        yield path


def _parent_config(init_ckpt: str, root: Path) -> Path | None:
    """outputs/<g>/checkpoints/<name>_best.pt -> outputs/<g>/experiments/<name>_config.json."""
    p = Path(str(init_ckpt).replace("\\", "/"))
    stem = p.stem
    for suffix in ("_best", "_final", "_resume"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if p.parent.name != "checkpoints":
        return None
    return root / p.parent.parent / "experiments" / f"{stem}_config.json"


def era_violations(cfg: dict, root: Path | None = None, _seen=None) -> list[str]:
    """Why this config is not the current publication era (empty == it is).

    Also follows ``model.init_checkpoint``: a run fine-tuned from a superseded
    model inherits that model's training data, which no config field records.
    """
    bad = []
    neg = cfg.get("negatives") or {}
    if not bool(neg.get("balanced", False)):
        bad.append(f"negatives.balanced={neg.get('balanced')}")
    try:
        ratio = float(neg.get("ratio", 0.0))
    except (TypeError, ValueError):
        ratio = 0.0
    if abs(ratio - 1.0) > 1e-9:
        bad.append(f"negatives.ratio={neg.get('ratio')}")
    artifacts = str((cfg.get("data") or {}).get("artifacts_dir", ""))
    if artifacts != "artifacts_night":
        bad.append(f"data.artifacts_dir={artifacts!r}")

    init_ckpt = (cfg.get("model") or {}).get("init_checkpoint")
    if init_ckpt and root is not None:
        seen = set() if _seen is None else _seen
        parent = _parent_config(init_ckpt, root)
        if parent is None:
            bad.append(f"init_checkpoint={init_ckpt!r} (cannot resolve provenance)")
        elif str(parent) in seen:
            pass
        elif not parent.exists():
            bad.append(f"init_checkpoint={init_ckpt!r} (no config for the source run)")
        else:
            seen.add(str(parent))
            try:
                pcfg = json.loads(parent.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                bad.append(f"init_checkpoint={init_ckpt!r} (source config unreadable)")
            else:
                inherited = era_violations(pcfg, root, seen)
                if inherited:
                    bad.append(
                        f"inherits from {parent.stem.replace('_config','')} "
                        f"which is superseded ({'; '.join(inherited)})")
    return bad


def is_current_era(config_path, root: Path | None = None) -> bool:
    """True if the run behind ``config_path`` may supply main-paper numbers."""
    path = Path(config_path)
    if not path.exists():
        return False
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return not era_violations(cfg, root if root is not None else ROOT)


def check(root: Path, names=None) -> list[str]:
    errors = []
    wanted = set(names) if names else None
    for path in _configs(root):
        name = path.name.replace("_config.json", "")
        if wanted is not None:
            if name not in wanted:
                continue
        elif not _is_main(name):
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        bad = era_violations(cfg, root)
        if bad:
            errors.append(
                f"{name}: {', '.join(bad)} "
                f"(need balanced=true, ratio=1.0, artifacts_night) [{path}]"
            )
    return errors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--names", default=None, help="comma-separated; default = main-paper runs")
    args = ap.parse_args()
    names = [n.strip() for n in args.names.split(",")] if args.names else None
    root = Path(args.root)
    errors = check(root, names)
    if errors:
        print("ERA GUARD FAILED — superseded split in a main-paper config:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        raise SystemExit(1)
    checked = [p.name.replace("_config.json", "") for p in _configs(root)
               if (names and p.name.replace("_config.json", "") in set(names))
               or (not names and _is_main(p.name.replace("_config.json", "")))]
    if not checked:
        print("era guard: PASSED VACUOUSLY — no main-paper runs exist yet. "
              "This is not evidence of anything until the seed runs are trained.")
        return
    print(f"era guard ok ({len(checked)} main-paper runs checked: {', '.join(sorted(checked))})")


if __name__ == "__main__":
    main()
