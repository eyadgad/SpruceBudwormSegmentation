"""Torch datasets for patch training / validation and full-scene evaluation.

Unlike the reference notebook (which preloaded ~70 scenes into RAM), the local
dataset is far too large to preload (1579 positives + negatives x up to 6
channels x 960x960). Scenes are therefore loaded lazily per access, with a
small bounded LRU cache per DataLoader worker so repeated patch draws from the
same scene don't re-read from disk.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
import torch
from torch.utils.data import Dataset, Sampler

from . import channels

IMG_SIZE = channels.IMG_SIZE


def temporal_cfg(cfg: Dict) -> Dict:
    """Normalized temporal settings: ``enabled, radius, max_gap_minutes, mode``.

    ``mode`` is ``attention`` (U-TAE style L-TAE fusion, the primary design) or
    ``stack`` (neighbour frames concatenated as extra input channels -- the naive
    2.5D control).
    """
    tc = dict(cfg.get("temporal") or {})
    return {
        "enabled": bool(tc.get("enabled", False)),
        "radius": int(tc.get("radius", 1)),
        "max_gap_minutes": float(tc.get("max_gap_minutes", 35.0)),
        "mode": str(tc.get("mode", "attention")),
    }


def sequence_length(cfg: Dict) -> int:
    tc = temporal_cfg(cfg)
    return 2 * tc["radius"] + 1 if tc["enabled"] else 1


def build_sequence_index(rows: List[dict], radius: int = 1,
                         max_gap_minutes: float = 35.0) -> Dict[int, List[int | None]]:
    """Map each row index to its chronological in-night neighbour row indices.

    Slots with no qualifying neighbour are ``None``. Neighbours are restricted to
    the same operational night AND the same split; because whole nights live in
    one split the second condition is normally implied, but it is enforced so the
    guarantee never silently depends on that.
    """
    by_night: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_night[(str(r.get("night", "")), str(r.get("split", "")))].append(i)

    out: Dict[int, List[int | None]] = {}
    for key, idxs in by_night.items():
        idxs = sorted(idxs, key=lambda i: int(rows[i]["timestamp"]))
        times = [pd.to_datetime(str(rows[i]["timestamp"]), format="%Y%m%d%H%M") for i in idxs]
        for pos, i in enumerate(idxs):
            window: List[int | None] = []
            for offset in range(-radius, radius + 1):
                j = pos + offset
                if offset == 0:
                    window.append(i)
                elif 0 <= j < len(idxs) and \
                        abs((times[j] - times[pos]).total_seconds()) / 60.0 <= max_gap_minutes:
                    window.append(idxs[j])
                else:
                    window.append(None)
            out[i] = window
    return out


def build_target(row, mode: str, dbz_threshold: float) -> np.ndarray:
    """Binary (H,W) float32 target for a manifest row.

    Positives: derived from the cached raw dBZ slice. Negatives: all zeros.
    """
    if int(row["label"]) == 0 or not row["target_path"]:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    dbz = np.load(row["target_path"])["dbz"]
    if mode == "isfinite":
        return np.isfinite(dbz).astype(np.float32)
    return np.where(np.isnan(dbz), 0.0, np.where(dbz >= dbz_threshold, 1.0, 0.0)).astype(np.float32)


class SceneStore:
    """Bounded LRU cache mapping manifest-row index -> (x_stack, y) numpy pair."""

    def __init__(self, cfg: Dict, rows: List[dict], norm_stats: Dict, maxsize: int = 16):
        self.cfg = cfg
        self.rows = rows
        self.norm_stats = norm_stats
        self.channels = cfg["channels"]
        self.mode = cfg["target"]["mode"]
        self.dbz_threshold = float(cfg["target"].get("dbz_threshold", 0.0))
        self.maxsize = maxsize
        self._cache: "OrderedDict[int, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

    def get(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        row = self.rows[idx]
        x = channels.build_stack(self.cfg, row["x_path"], self.channels, self.norm_stats)
        y = build_target(row, self.mode, self.dbz_threshold)
        self._cache[idx] = (x, y)
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)
        return x, y


class RadarPatchDataset(Dataset):
    """Patch dataset. mode='train' -> random positive-biased patches;
    mode='eval' -> deterministic non-overlapping grid patches (cheap validation)."""

    def __init__(self, cfg: Dict, manifest, split: str, norm_stats: Dict, mode: str = "train"):
        rows = manifest[manifest["split"] == split].reset_index(drop=True)
        self.rows = rows.to_dict("records")
        self.mode = mode
        self.split = split
        self.patch = cfg["patch"]
        # Auxiliary presence classification: the label is the PATCH's own, taken
        # from its ground-truth crop. Propagating the scan label to every patch
        # would tell a plume-free crop of a positive scan that it is positive.
        self.cls_head = bool(cfg.get("model", {}).get("cls_head", False))
        self.temporal = temporal_cfg(cfg)
        self.seq_index = (
            build_sequence_index(self.rows, self.temporal["radius"],
                                 self.temporal["max_gap_minutes"])
            if self.temporal["enabled"] else None
        )
        self.ps = int(self.patch["size"])
        self.patches_per_image = int(self.patch["patches_per_image"])
        self.pos_sample_rate = float(self.patch["pos_sample_rate"])
        self.hard_pos_rate = float(self.patch["hard_pos_rate"])
        self.aug_cfg = cfg.get("augment", {"enabled": False})
        self.augment = bool(self.aug_cfg.get("enabled", False)) and mode == "train"
        # SVLS-style soft boundary labels (train only): Gaussian-blur the binary
        # target so plume edges become soft probabilities -> better boundary
        # calibration. 0 disables. (Islam & Glocker, MICCAI 2021.)
        self.soft_sigma = float(cfg["target"].get("soft_sigma", 0.0)) if mode == "train" else 0.0
        # indices of TH channels (photometric aug only touches these)
        self.th_idx = [i for i, s in enumerate(cfg["channels"]) if channels.parse_spec(s)[0] == "th"]
        cache = int(cfg["train"].get("scene_cache", 16))
        self.store = SceneStore(cfg, self.rows, norm_stats, maxsize=cache)
        if mode != "train":
            self._grid = self._build_grid()

    # -- grid for deterministic eval --
    def _build_grid(self):
        ps = self.ps
        rows = list(range(0, IMG_SIZE - ps + 1, ps))
        cols = list(range(0, IMG_SIZE - ps + 1, ps))
        if rows[-1] + ps < IMG_SIZE:
            rows.append(IMG_SIZE - ps)
        if cols[-1] + ps < IMG_SIZE:
            cols.append(IMG_SIZE - ps)
        return [(r, c) for r in rows for c in cols]

    def __len__(self):
        if self.mode == "train":
            return len(self.rows) * self.patches_per_image
        return len(self.rows) * len(self._grid)

    # -- patch extraction --
    def _extract_patch(self, x, y, ensure_positive):
        ps = self.ps
        if ensure_positive and y.sum() > 0:
            pos = np.argwhere(y > 0)
            if np.random.random() < self.hard_pos_rate:
                density = ndimage.uniform_filter(y, size=max(ps // 2, 1))
                pd = density[pos[:, 0], pos[:, 1]]
                top_k = max(1, len(pd) // 10)
                top = np.argpartition(pd, -top_k)[-top_k:]
                center = pos[top[np.random.randint(len(top))]]
            else:
                center = pos[np.random.randint(len(pos))]
            r = int(max(0, min(center[0] - ps // 2 + np.random.randint(-ps // 4, ps // 4), IMG_SIZE - ps)))
            c = int(max(0, min(center[1] - ps // 2 + np.random.randint(-ps // 4, ps // 4), IMG_SIZE - ps)))
        else:
            r = np.random.randint(0, IMG_SIZE - ps + 1)
            c = np.random.randint(0, IMG_SIZE - ps + 1)
        # Ellipsis: x is (C,H,W) normally and (T,C,H,W) for the temporal variants.
        return x[..., r:r + ps, c:c + ps], y[r:r + ps, c:c + ps]

    def _augment(self, x, y):
        # Spatial ops address the trailing (H,W) axes so they apply unchanged to
        # both (C,H,W) and (T,C,H,W); the same transform is applied to every
        # frame of a sequence, which is what keeps the frames registered.
        a = self.aug_cfg
        if np.random.random() > 0.5:
            x = np.flip(x, axis=-1).copy(); y = np.flip(y, axis=1).copy()
        if np.random.random() > 0.5:
            x = np.flip(x, axis=-2).copy(); y = np.flip(y, axis=0).copy()
        k = np.random.randint(4)
        if k:
            x = np.rot90(x, k, axes=(-2, -1)).copy(); y = np.rot90(y, k, axes=(0, 1)).copy()
        if self.th_idx and np.random.random() < a.get("noise_prob", 0):
            noise = np.random.randn(len(self.th_idx), x.shape[-2], x.shape[-1]).astype(np.float32) * a.get("noise_std", 0.05)
            x[..., self.th_idx, :, :] += noise
        if np.random.random() < a.get("ch_drop_prob", 0):
            x[..., np.random.randint(x.shape[-3]), :, :] = 0.0
        if self.th_idx and np.random.random() < a.get("bright_prob", 0):
            x[..., self.th_idx, :, :] += np.random.uniform(-a.get("bright_range", 0.1), a.get("bright_range", 0.1))
        return x, y

    def _scene(self, img_idx: int):
        """Scene input and target, as a single frame or a temporal sequence.

        For the temporal variants the input is ``(T,C,H,W)``. A slot with no
        qualifying neighbour is filled with a COPY OF THE CENTRE FRAME and
        flagged in ``pad_mask``: the copy keeps the shared encoder's BatchNorm
        from ingesting all-zero images, while the mask removes the slot from the
        temporal attention softmax so it contributes nothing. A scan with no
        neighbours therefore degrades to the single-frame model exactly.
        """
        x, y = self.store.get(img_idx)
        if not self.temporal["enabled"]:
            return x, y, None
        window = self.seq_index[img_idx]
        frames, pad = [], []
        for j in window:
            if j is None:
                frames.append(x)
                pad.append(True)
            else:
                frames.append(self.store.get(j)[0])
                pad.append(False)
        return np.stack(frames, axis=0), y, np.asarray(pad, dtype=bool)

    def __getitem__(self, idx):
        if self.mode == "train":
            img_idx = idx // self.patches_per_image
            x, y, pad = self._scene(img_idx)
            ensure_pos = np.random.random() < self.pos_sample_rate
            xp, yp = self._extract_patch(x, y, ensure_pos)
            if self.augment:
                xp, yp = self._augment(xp, yp)
            # Presence label is read BEFORE soft-label blurring, so it stays an
            # exact "does this crop contain any ground-truth pixel".
            cls = float(yp.sum() > 0)
            if self.soft_sigma > 0:
                yp = np.clip(ndimage.gaussian_filter(yp.astype(np.float32), self.soft_sigma), 0.0, 1.0)
        else:
            img_idx = idx // len(self._grid)
            r, c = self._grid[idx % len(self._grid)]
            x, y, pad = self._scene(img_idx)
            xp = x[..., r:r + self.ps, c:c + self.ps]
            yp = y[r:r + self.ps, c:c + self.ps]
            cls = float(yp.sum() > 0)

        if self.temporal["enabled"] and self.temporal["mode"] == "stack":
            # 2.5D control: fold the time axis into channels. Padded slots are
            # already centre-frame copies, which is the standard imputation.
            xp = xp.reshape(-1, xp.shape[-2], xp.shape[-1])
            pad = None

        out = [torch.from_numpy(np.ascontiguousarray(xp)).float(),
               torch.from_numpy(np.ascontiguousarray(yp)).float().unsqueeze(0)]
        extra: Dict[str, torch.Tensor] = {}
        if self.cls_head:
            extra["cls"] = torch.tensor([cls], dtype=torch.float32)
        if pad is not None:
            extra["pad_mask"] = torch.from_numpy(np.ascontiguousarray(pad))
        if extra:
            out.append(extra)
        return tuple(out)


def load_full_scene(cfg: Dict, row: dict, norm_stats: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Full (C,960,960) input and (960,960) binary target for one scene."""
    x = channels.build_stack(cfg, row["x_path"], cfg["channels"], norm_stats)
    y = build_target(row, cfg["target"]["mode"], float(cfg["target"].get("dbz_threshold", 0.0)))
    return x, y


def load_full_scene_sequence(cfg: Dict, rows: List[dict], idx: int, norm_stats: Dict,
                             seq_index: Dict[int, List[int | None]]):
    """Temporal counterpart of ``load_full_scene`` for full-scene inference.

    Returns ``(x, y, pad_mask)`` with ``x`` shaped ``(T,C,960,960)`` -- or
    ``(T*C,960,960)`` in ``stack`` mode -- and ``pad_mask`` marking the slots
    filled by a centre-frame copy (``None`` in stack mode, which has no mask).
    """
    tc = temporal_cfg(cfg)
    centre, y = load_full_scene(cfg, rows[idx], norm_stats)
    frames, pad = [], []
    for j in seq_index[idx]:
        if j is None:
            frames.append(centre)
            pad.append(True)
        else:
            frames.append(centre if j == idx
                          else channels.build_stack(cfg, rows[j]["x_path"],
                                                    cfg["channels"], norm_stats))
            pad.append(False)
    x = np.stack(frames, axis=0)
    if tc["mode"] == "stack":
        return x.reshape(-1, x.shape[-2], x.shape[-1]), y, None
    return x, y, np.asarray(pad, dtype=bool)


class SceneGroupedSampler(Sampler):
    """Yields patch indices grouped by scene so the LRU SceneStore is effective.

    With the default DataLoader shuffle=True all 20 K+ patch indices are
    scattered randomly across 1301 scenes, causing near-zero cache hits and
    ~20 K netCDF reads per epoch.  This sampler shuffles the *scene* order
    each epoch but keeps all ``patches_per_image`` indices for the same scene
    consecutive, reducing reads to one per unique scene (~1301 per epoch).
    """

    def __init__(self, dataset: RadarPatchDataset, shuffle: bool = True, seed: int = 0):
        self.n_scenes = len(dataset.rows)
        self.ppi = dataset.patches_per_image
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        return self.n_scenes * self.ppi

    def set_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch so scene order differs every epoch."""
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        order = rng.permutation(self.n_scenes) if self.shuffle else np.arange(self.n_scenes)
        indices: List[int] = []
        for s in order.tolist():
            indices.extend(range(s * self.ppi, (s + 1) * self.ppi))
        return iter(indices)


class NightBalancedSceneSampler(Sampler):
    """Scene sampler that equalizes each night's contribution to an epoch.

    ``SceneGroupedSampler`` visits every scene once, so a night's influence is
    proportional to how many times the radar happened to scan it. On this corpus
    that is 1 to 28 scans, and migration nights average 13.5 against 3.6 for
    quiet nights -- so a handful of long migration nights dominate training and
    quiet nights are effectively rare.

    This sampler instead draws nights, then a scene within the night:

      * migration and quiet nights are drawn equally often, and
      * within a class every night is equally likely, regardless of its length.

    Epoch length is held at ``n_scenes * patches_per_image`` so results stay
    comparable with the unbalanced baseline, and a scene's patch indices are kept
    consecutive so the LRU ``SceneStore`` still hits.

    TRAIN ONLY. The validation and test loaders keep their natural distribution;
    reweighting them would change what is being measured.
    """

    def __init__(self, dataset: RadarPatchDataset, shuffle: bool = True, seed: int = 0):
        self.n_scenes = len(dataset.rows)
        self.ppi = dataset.patches_per_image
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        by_night: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(dataset.rows):
            by_night[str(r.get("night", ""))].append(i)
        # A night is a migration night if it holds any positive scan -- the same
        # rule presence.analyze_presence uses to derive night truth.
        self.migration, self.quiet = [], []
        for night, idxs in sorted(by_night.items()):
            target = self.migration if any(int(dataset.rows[i]["label"]) == 1 for i in idxs) \
                else self.quiet
            target.append(idxs)

    def __len__(self) -> int:
        return self.n_scenes * self.ppi

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        groups = [g for g in (self.migration, self.quiet) if g]
        indices: List[int] = []
        for _ in range(self.n_scenes):
            night_idxs = groups[rng.integers(len(groups))] if len(groups) > 1 else groups[0]
            night = night_idxs[rng.integers(len(night_idxs))]
            scene = night[rng.integers(len(night))]
            indices.extend(range(scene * self.ppi, (scene + 1) * self.ppi))
        return iter(indices)
