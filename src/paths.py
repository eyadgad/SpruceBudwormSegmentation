"""Local path resolution.

The reference ``matched_samples.csv`` stores absolute paths under
``D:\\radar_test\\updated_Data\\...`` which do not exist on this machine (Phase 1
finding). We therefore NEVER trust the CSV's absolute paths: every file is
resolved from the 12-digit timestamp + local data root.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

TS_RE = re.compile(r"(\d{12})")


def project_root() -> Path:
    # src/paths.py -> project root is two levels up.
    return Path(__file__).resolve().parents[1]


def data_root(cfg: Dict) -> Path:
    root = Path(cfg["data"]["root"])
    if not root.is_absolute():
        root = project_root() / root
    return root


def cleaned_dir(cfg: Dict) -> Path:
    return data_root(cfg) / cfg["data"]["cleaned_subdir"]


def matched_csv(cfg: Dict) -> Path:
    return data_root(cfg) / cfg["data"]["matched_csv"]


def dem_file(cfg: Dict) -> Path:
    return cleaned_dir(cfg) / "xam_dem.nc"


def beam_height_file(cfg: Dict) -> Path:
    return cleaned_dir(cfg) / "xam_beam_height_asl.nc"


def mask_file(cfg: Dict, year: int) -> Path:
    return cleaned_dir(cfg) / f"Cleaned_dispersal_DBZH_{year}.nc"


def ppi_file(cfg: Dict, year: int, label: int, timestamp: int) -> Path:
    cls = "positives" if label == 1 else "negatives"
    return data_root(cfg) / str(year) / cls / f"XAM_{timestamp}_filtered_ppi.nc"


def artifacts_dir(cfg: Dict) -> Path:
    d = Path(cfg["data"]["artifacts_dir"])
    if not d.is_absolute():
        d = project_root() / d
    return d


def targets_dir(cfg: Dict) -> Path:
    return artifacts_dir(cfg) / "targets"


def output_dir(cfg: Dict) -> Path:
    d = Path(cfg.get("output_dir", "outputs"))
    if not d.is_absolute():
        d = project_root() / d
    return d


def checkpoint_dir(cfg: Dict) -> Path:
    return output_dir(cfg) / "checkpoints"


def experiments_dir(cfg: Dict) -> Path:
    return output_dir(cfg) / "experiments"


def timestamp_from_name(name: str) -> int:
    m = TS_RE.search(name)
    if not m:
        raise ValueError(f"No 12-digit timestamp found in '{name}'")
    return int(m.group(1))
