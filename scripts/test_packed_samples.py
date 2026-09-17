"""Validate every generated SBW1 sample pack against its source assets.

Run before deleting the legacy PNGs:

    python scripts/test_packed_samples.py --data-root ../Data \
        --site-dir ../sprucebudworm_progress.github.io

After migration, add ``--skip-legacy`` for structural/raw-reflectivity checks.
"""
from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import export_dashboard_data as export  # noqa: E402


def parse_pack(path: Path, model_order: list[str], width: int, height: int):
    raw = gzip.decompress(path.read_bytes())
    if len(raw) < export.SBW_HEADER.size:
        raise AssertionError(f"{path}: truncated header")
    magic, w, h, n_model, flags, header_size, reserved = export.SBW_HEADER.unpack_from(raw)
    assert magic == export.SBW_MAGIC, (path, magic)
    assert (w, h) == (width, height), (path, w, h)
    assert n_model == len(model_order), (path, n_model)
    assert flags == export.SBW_FLAGS, (path, flags)
    assert header_size == export.SBW_HEADER.size and reserved == 0, path
    n = width * height
    gt_bytes = (n + 7) // 8
    expected = header_size + n_model * n + gt_bytes + n
    assert len(raw) == expected, (path, len(raw), expected)
    pos = header_size
    probabilities = {}
    for key in model_order:
        probabilities[key] = np.frombuffer(raw, np.uint8, n, pos).reshape(height, width)
        pos += n
    gt = np.unpackbits(np.frombuffer(raw, np.uint8, gt_bytes, pos), bitorder="big")[:n].reshape(height, width)
    pos += gt_bytes
    reflectivity = np.frombuffer(raw, np.uint8, n, pos).reshape(height, width)
    assert int(reflectivity.max()) <= 6, (path, int(reflectivity.max()))
    return probabilities, gt, reflectivity


def metrics(probability: np.ndarray, gt: np.ndarray, threshold: float):
    pred = probability > threshold * 255.0
    truth = gt.astype(bool)
    tp = int((pred & truth).sum())
    fp = int((pred & ~truth).sum())
    fn = int((~pred & truth).sum())
    denom = 2 * tp + fp + fn
    dice = 2 * tp / denom if denom else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return tp, fp, fn, dice, precision, recall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=ROOT.parent / "Data")
    ap.add_argument("--site-dir", type=Path, default=ROOT.parent / "sprucebudworm_progress.github.io")
    ap.add_argument("--expected-scenes", type=int, default=615)
    ap.add_argument("--max-mib", type=float, default=21.0)
    ap.add_argument("--skip-legacy", action="store_true")
    args = ap.parse_args()

    site = args.site_dir.resolve()
    sample_dir = site / "data" / "samples"
    doc = json.loads((site / "data" / "samples.json").read_text(encoding="utf-8"))
    assets = doc.get("sample_assets") or {}
    assert assets.get("format") == "sbw1-gzip", assets
    assert assets.get("reflectivity_source") == "max_th_e0_th_e5", assets
    assert assets.get("reflectivity_elevations") == [0, 1, 2, 3, 4, 5], assets
    assert str(assets.get("version", "")).startswith(export.SBW_VERSION_PREFIX + "-"), assets
    width, height = int(assets["width"]), int(assets["height"])
    thumb_width = int(assets["thumbnail_width"])
    thumb_height = int(assets["thumbnail_height"])
    model_order = list(assets["model_order"])
    samples = doc["samples"]
    timestamps = [int(s["ts"]) for s in samples]
    assert len(samples) == len(set(timestamps)) == args.expected_scenes

    raw_index = export._raw_ppi_index(args.data_root.resolve())
    missing = sorted(set(timestamps) - set(raw_index))
    assert not missing, f"missing {len(missing)} raw PPI files: {missing[:20]}"
    packs = sorted(sample_dir.glob("*.sbw.gz"))
    thumbs = sorted(sample_dir.glob("*.webp"))
    assert len(packs) == len(thumbs) == args.expected_scenes, (len(packs), len(thumbs))
    total = sum(p.stat().st_size for p in packs + thumbs)
    assert total <= args.max_mib * 1024 * 1024, total

    thresholds = (0.02, 0.15, 0.50, 0.90)
    for i, ts in enumerate(timestamps, 1):
        probs, gt, reflectivity = parse_pack(sample_dir / f"{ts}.sbw.gz", model_order, width, height)
        expected_refl, _ = export._reflectivity_composite(raw_index[ts], width)
        assert np.array_equal(reflectivity, expected_refl), f"{ts}: reflectivity mismatch"
        with Image.open(sample_dir / f"{ts}.webp") as thumb:
            assert thumb.size == (thumb_width, thumb_height) and thumb.mode == "RGB", \
                (ts, thumb.size, thumb.mode)

        if not args.skip_legacy:
            legacy_gt = export._read_l_png(sample_dir / f"{ts}_gt.png", (height, width)) > 127
            assert np.array_equal(gt.astype(bool), legacy_gt), f"{ts}: GT mismatch"
            for key in model_order:
                legacy_prob = export._read_l_png(sample_dir / f"{ts}_prob_{key}.png", (height, width))
                assert np.array_equal(probs[key], legacy_prob), f"{ts}/{key}: probability mismatch"
                for threshold in thresholds:
                    assert metrics(probs[key], gt, threshold) == metrics(legacy_prob, legacy_gt, threshold), \
                        f"{ts}/{key}/{threshold}: metric mismatch"
        if i % 50 == 0 or i == len(timestamps):
            print(f"  {i}/{len(timestamps)}")

    # Independent category-boundary oracle. Values below the legend floor and
    # all-missing cells are black; high values remain in the last band.
    vals = np.asarray([np.nan, -10.001, -10.0, -1.0, -0.999, 2.0, 2.001, 7.0,
                       7.001, 12.0, 12.001, 19.0, 19.001, 70.0], np.float32)
    got = np.zeros(vals.shape, np.uint8)
    valid = np.isfinite(vals) & (vals >= -10.0)
    got[valid] = np.digitize(vals[valid], export.REFLECTIVITY_BINS, right=True) + 1
    assert got.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], got.tolist()

    # Missing elevations are ignored before the spatial block maximum; a cell
    # missing in all six elevations stays background. This uses a tiny array so
    # it does not share the NetCDF read path used by the exhaustive comparison.
    cube = np.full((6, 4, 4), np.nan, np.float32)
    cube[0, 0, 0] = -5.0
    cube[5, 0, 0] = 3.0       # maximum across available elevations -> code 3
    cube[2, 0, 1] = 8.0       # spatial 2x2 maximum -> code 4
    cube[1, 0, 2] = -10.01    # finite but below display floor -> code 0
    cube[4, 2, 2] = 19.001    # code 6
    tiny_codes, tiny_raw = export._reflectivity_from_array(cube, size=2)
    assert tiny_codes.tolist() == [[4, 0], [0, 6]], tiny_codes.tolist()
    assert tiny_raw[0, 0] == 8.0 and np.isclose(tiny_raw[0, 1], -10.01)
    assert np.isnan(tiny_raw[1, 0]) and np.isclose(tiny_raw[1, 1], 19.001)
    print(f"PASS: {len(timestamps)} packs + thumbnails, {total / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
