"""Build model input channels from a radar scene + static terrain grids.

Every layer (radar TH sweeps, per-scan beam height, static DEM, static
beam-height-ASL, valid-pixel mask) lives on the identical 960x960 grid
(Phase 1: byte-identical easting/northing across all files), so channels stack
directly with no reprojection or resampling.

Channel specs (config-driven):
  th_e<i>      TH reflectivity (dBZ) at elevation index i (0..23), NaN->0
  height_e<i>  per-scan beam height at elevation index i (0..23), NaN->0
  bh_e<i>      static beam-height-ASL at level i (0..25), NaN->0
  dem          static digital elevation model (m)
  valid_mask   binary: 1 where TH[valid_elev] is finite, else 0 (NOT normalized)
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import netCDF4 as nc

from . import paths

IMG_SIZE = 960
VALID_MASK_ELEV = 0  # elevation index used to define the valid-pixel mask

# module-level cache for the static grids (same for every scene)
_STATIC: Dict[str, np.ndarray] = {}


def _to_nan(arr) -> np.ndarray:
    """netCDF4 returns masked arrays; turn masked/fill values into NaN float32."""
    a = np.ma.filled(arr.astype(np.float32), np.nan) if np.ma.isMaskedArray(arr) else np.asarray(arr, np.float32)
    return a


def load_static_grids(cfg: Dict) -> Dict[str, np.ndarray]:
    """Load DEM and beam-height-ASL once, cached for the process."""
    if _STATIC:
        return _STATIC
    with nc.Dataset(paths.dem_file(cfg)) as ds:
        dem = _to_nan(ds.variables["xam_dem"][:])
    dem = np.where(np.isnan(dem), 0.0, dem).astype(np.float32)  # DEM has no NaN, but be safe
    with nc.Dataset(paths.beam_height_file(cfg)) as ds:
        bh = _to_nan(ds.variables["beam_height_asl"][:])  # (26, 960, 960)
    _STATIC["dem"] = dem
    _STATIC["bh"] = bh
    return _STATIC


def parse_spec(spec: str) -> Tuple[str, Optional[int]]:
    """('th_e2') -> ('th', 2); ('dem') -> ('dem', None)."""
    for prefix, kind in (("th_e", "th"), ("height_e", "height"), ("bh_e", "bh")):
        if spec.startswith(prefix):
            return kind, int(spec[len(prefix):])
    return spec, None


def norm_key(spec: str) -> Optional[str]:
    """Normalization-stats key for a spec, or None if the channel isn't normalized."""
    if spec == "valid_mask":
        return None
    return spec


def channel_elevations(channels: List[str]) -> Dict[str, set]:
    """Which TH / height elevation indices must be read from the PPI file."""
    need = {"th": set(), "height": set()}
    need["th"].add(VALID_MASK_ELEV)  # valid_mask + any th_e need TH
    for spec in channels:
        kind, idx = parse_spec(spec)
        if kind in ("th", "height"):
            need[kind].add(idx)
    return need


def _read_elevation(var, i: int, ppi_path) -> np.ndarray:
    """var[i], or all-NaN (no data) if this volume scan has fewer tilts than ``i``.

    Elevation count varies slightly across scans (VCP strategy differences), so a
    handful of scenes lack a high-index tilt entirely. Treat that the same as a
    missing/no-echo pixel (NaN -> 0 downstream) rather than crashing full-scene
    eval over one scene's absent elevation.
    """
    if i >= var.shape[0]:
        warnings.warn(f"{ppi_path}: elevation {i} missing (only {var.shape[0]} tilts); using NaN")
        return np.full(var.shape[-2:], np.nan, dtype=np.float32)
    return _to_nan(var[i])


def read_scene_layers(cfg: Dict, ppi_path, channels: List[str]) -> Dict[str, np.ndarray]:
    """Read only the TH/height elevation slices required by ``channels``."""
    need = channel_elevations(channels)
    layers: Dict[str, np.ndarray] = {}
    with nc.Dataset(ppi_path) as ds:
        th = ds.variables["TH"]
        for i in need["th"]:
            layers[f"th_e{i}"] = _read_elevation(th, i, ppi_path)
        if need["height"]:
            h = ds.variables["height"]
            for i in need["height"]:
                layers[f"height_e{i}"] = _read_elevation(h, i, ppi_path)
    return layers


def normalize(data: np.ndarray, key: str, norm_stats: Dict) -> np.ndarray:
    p = norm_stats[key]
    out = (data - p["mean"]) / (p["std"] + 1e-8)
    return np.clip(out, -5.0, 5.0)


def build_stack(cfg: Dict, ppi_path, channels: List[str],
                norm_stats: Optional[Dict]) -> np.ndarray:
    """Return a (C, 960, 960) float32 array in ``channels`` order.

    If ``norm_stats`` is given, normalizable channels are z-scored + clipped;
    valid_mask is always left as 0/1. If None, raw (NaN-filled) values are
    returned (used when accumulating normalization statistics).
    """
    static = load_static_grids(cfg)
    layers = read_scene_layers(cfg, ppi_path, channels)
    valid = np.isfinite(layers[f"th_e{VALID_MASK_ELEV}"]).astype(np.float32)

    out = np.empty((len(channels), IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for c, spec in enumerate(channels):
        kind, idx = parse_spec(spec)
        if spec == "valid_mask":
            out[c] = valid
            continue
        if kind == "th":
            ch = layers[f"th_e{idx}"].copy()
            ch[np.isnan(ch)] = 0.0
        elif kind == "height":
            ch = layers[f"height_e{idx}"].copy()
            ch[np.isnan(ch)] = 0.0
        elif kind == "bh":
            ch = static["bh"][idx].copy()
            ch[np.isnan(ch)] = 0.0
        elif spec == "dem":
            ch = static["dem"].copy()
        else:  # pragma: no cover - validated upstream
            raise ValueError(f"Unhandled channel spec: {spec}")
        key = norm_key(spec)
        out[c] = normalize(ch, key, norm_stats) if (norm_stats is not None and key) else ch
    return out


def raw_channel_values(cfg: Dict, ppi_path, spec: str) -> np.ndarray:
    """Flat finite raw values for one channel (for normalization statistics)."""
    kind, idx = parse_spec(spec)
    if spec == "dem":
        return load_static_grids(cfg)["dem"].ravel()
    if kind == "bh":
        return load_static_grids(cfg)["bh"][idx].ravel()
    layers = read_scene_layers(cfg, ppi_path, [spec])
    arr = layers[f"{kind}_e{idx}"]
    arr = arr[np.isfinite(arr)]
    return arr
