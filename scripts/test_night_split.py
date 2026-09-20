"""Tests for the leakage-free night split and the three ablation components.

Fixture-based and CPU-only, so this runs without a GPU or the radar archive.
Tests that need the generated artifacts skip cleanly when they are absent.

    .venv\\Scripts\\python.exe scripts\\test_night_split.py

Deliberately does NOT touch the old-split constants pinned in
scripts/test_presence.py and scripts/test_dashboard.py -- those describe the
historical artifacts and must keep describing them.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import nights  # noqa: E402
from src.channels import IMG_SIZE  # noqa: E402
from src.dataset import (  # noqa: E402
    NightBalancedSceneSampler, RadarPatchDataset, SceneGroupedSampler,
    build_sequence_index, temporal_cfg,
)
from src.losses import MultiTaskLoss, create_loss  # noqa: E402
from src.models import create_model, seg_logits  # noqa: E402
from src.presence import analyze_presence, operational_night_id  # noqa: E402

FRACTIONS = {"train": 0.70, "val": 0.20, "test": 0.10}
NIGHT_ARTIFACTS = ROOT / "artifacts_night"


def synthetic_manifest(n_nights: int = 40, years=(2013, 2014, 2015)) -> pd.DataFrame:
    """Manifest with the corpus's real shape: homogeneous nights, uneven lengths.

    Migration nights are long (8-14 scans) and all-positive; quiet nights are
    short (2-5) and all-negative -- which is how the real manifest looks.
    """
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_nights):
        year = years[i % len(years)]
        migration = (i % 2 == 0)
        n_scans = int(rng.integers(8, 15) if migration else rng.integers(2, 6))
        day = 10 + (i // len(years))
        for k in range(n_scans):
            hour, minute = divmod(22 * 60 + 30 * k, 60)
            day_off, hour = divmod(hour, 24)
            ts = int(f"{year}07{day + day_off:02d}{hour:02d}{minute:02d}")
            rows.append({"timestamp": ts, "year": year,
                         "label": 1 if migration else 0,
                         "night": "", "target_path": "", "x_path": f"/fake/{ts}.nc"})
    return pd.DataFrame(rows)


class NightDerivation(unittest.TestCase):
    def test_noon_boundary_and_rollovers(self):
        self.assertEqual(nights.night_id(201307122330), "2013-07-12")
        self.assertEqual(nights.night_id(201307130000), "2013-07-12")
        self.assertEqual(nights.night_id(201307131159), "2013-07-12")
        self.assertEqual(nights.night_id(201307131200), "2013-07-13")
        self.assertEqual(nights.night_id(202001010030), "2019-12-31")   # year rollover
        self.assertEqual(nights.night_id(202003010030), "2020-02-29")   # leap day

    def test_delegates_to_presence_so_split_and_dashboard_agree(self):
        for ts in (201307122330, 201908010300, 201607311730):
            self.assertEqual(nights.night_id(ts), operational_night_id(ts))

    def test_negatives_get_night_ids(self):
        df = synthetic_manifest()
        self.assertTrue((df["night"] == "").all(), "fixture starts with no night IDs")
        out = nights.add_night_ids(df)
        neg = out[out["label"] == 0]
        self.assertGreater(len(neg), 0)
        self.assertTrue((neg["night"] != "").all(),
                        "every negative scan must get a night ID; the manifest column is "
                        "empty for negatives, which is why IDs come from the timestamp")

    def test_original_night_string_preserved_as_mask_night(self):
        df = synthetic_manifest()
        df.loc[df["label"] == 1, "night"] = "2013 Jul 13_14"
        out = nights.add_night_ids(df)
        self.assertIn("mask_night", out.columns)
        self.assertEqual(set(out.loc[out["label"] == 1, "mask_night"]), {"2013 Jul 13_14"})
        self.assertNotIn("2013 Jul 13_14", set(out["night"]))


class SplitIntegrity(unittest.TestCase):
    def setUp(self):
        self.df = nights.add_night_ids(synthetic_manifest())
        self.df["split"] = nights.assign_night_split(self.df, FRACTIONS, seed=42)

    def test_zero_night_overlap(self):
        report = nights.night_split_report(self.df)
        self.assertEqual(report["overlap_nights"], 0, report["overlap_night_ids"])
        nights.verify_no_leakage(self.df)

    def test_every_scan_of_a_night_shares_one_split(self):
        for night, group in self.df.groupby("night"):
            self.assertEqual(group["split"].nunique(), 1, f"night {night} was split")

    def test_leakage_check_detects_a_deliberately_broken_split(self):
        broken = self.df.copy()
        first_night = broken["night"].iloc[0]
        idx = broken.index[broken["night"] == first_night]
        broken.loc[idx[0], "split"] = "train"
        broken.loc[idx[1:], "split"] = "test"
        with self.assertRaisesRegex(ValueError, "night split leaked"):
            nights.verify_no_leakage(broken)

    def test_scan_fractions_near_target(self):
        report = nights.night_split_report(self.df)
        for split, want in FRACTIONS.items():
            got = report["per_split"][split]["scan_fraction"]
            self.assertAlmostEqual(got, want, delta=0.06, msg=f"{split}: {got} vs {want}")

    def test_every_year_and_both_night_kinds_in_every_split(self):
        report = nights.night_split_report(self.df)
        all_years = {int(y) for y in self.df["year"].unique()}
        for split in ("train", "val", "test"):
            info = report["per_split"][split]
            self.assertEqual({int(y) for y in info["nights_per_year"]}, all_years,
                             f"{split} is missing years; per-stratum targets should "
                             f"put every year in every split")
            self.assertGreater(info["migration_nights"], 0, f"{split} has no migration nights")
            self.assertGreater(info["quiet_nights"], 0, f"{split} has no quiet nights")

    def test_validation_has_both_truth_classes(self):
        # presence.select_youden_cutoff raises without both classes on validation.
        val = self.df[self.df["split"] == "val"]
        self.assertEqual(set(val["label"].unique()), {0, 1})

    def test_reproducible_and_seed_sensitive(self):
        a = nights.assign_night_split(self.df, FRACTIONS, seed=42)
        b = nights.assign_night_split(self.df, FRACTIONS, seed=42)
        c = nights.assign_night_split(self.df, FRACTIONS, seed=7)
        self.assertTrue((a == b).all(), "same seed must give the same split")
        self.assertTrue((a != c).any(), "a different seed must give a different split")
        # ... and the alternative seed must still be a valid split.
        other = self.df.assign(split=c)
        nights.verify_no_leakage(other)

    def test_rejects_fractions_that_do_not_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            nights.assign_night_split(self.df, {"train": 0.7, "val": 0.2, "test": 0.2}, seed=42)


class GeneratedNightSplit(unittest.TestCase):
    """Assertions against the real generated artifacts_night/ split."""

    def setUp(self):
        summary = NIGHT_ARTIFACTS / "split_summary.json"
        if not summary.exists():
            self.skipTest("artifacts_night/ not generated")
        self.summary = json.loads(summary.read_text(encoding="utf-8"))

    def test_mode_and_zero_overlap(self):
        self.assertEqual(self.summary["mode"], "night")
        self.assertEqual(self.summary["night_split"]["overlap_nights"], 0)

    def test_scan_fractions_and_coverage(self):
        report = self.summary["night_split"]
        for split, want in FRACTIONS.items():
            self.assertAlmostEqual(report["per_split"][split]["scan_fraction"], want, delta=0.03)
        for split in ("train", "val", "test"):
            info = report["per_split"][split]
            self.assertEqual(len(info["nights_per_year"]), 7, f"{split} lacks all 7 years")
            self.assertGreater(info["migration_nights"], 0)
            self.assertGreater(info["quiet_nights"], 0)
            self.assertGreater(info["positives"], 0)
            self.assertGreater(info["negatives"], 0)

    def test_manifest_matches_the_report(self):
        manifest = NIGHT_ARTIFACTS / "manifest.csv"
        if not manifest.exists():
            self.skipTest("artifacts_night/manifest.csv not generated")
        df = pd.read_csv(manifest)
        df["night"] = df["night"].astype(str)
        self.assertEqual(df.groupby("night")["split"].nunique().max(), 1,
                         "a night spans multiple splits in the written manifest")
        self.assertTrue((df["night"] != "").all(), "some scan has no night ID")
        # Night truth is homogeneous on this corpus; the split must not change that.
        kinds = df.groupby("night")["label"].agg(["min", "max"])
        self.assertEqual(int(((kinds["min"] == 0) & (kinds["max"] == 1)).sum()), 0)


class PresenceCohortLeakage(unittest.TestCase):
    """The night split must show as zero exposure in the dashboard's own analysis."""

    def _docs(self):
        df = nights.add_night_ids(synthetic_manifest())
        df["split"] = nights.assign_night_split(df, FRACTIONS, seed=42)
        grid = {"h": 960, "w": 960, "pixel_m": 500}
        scenes = [{"ts": int(r.timestamp), "split": r.split, "label": int(r.label),
                   "area": (100 if r.label == 1 else None)} for r in df.itertuples()]
        rng = np.random.default_rng(1)
        samples = []
        for r in df[df["split"].isin(["val", "test"])].itertuples():
            gt = 100 if r.label == 1 else 0
            pred = int(rng.integers(80, 200)) if r.label == 1 else int(rng.integers(0, 40))
            samples.append({"ts": int(r.timestamp), "split": r.split, "gt_area": gt,
                            "models": {"m": {"pred_area": pred}}})
        return ({"selected": "model", "models": [{"key": "m", "name": "model",
                                                  "disp": "M", "thr": 0.15}],
                 "samples": samples},
                {"grid": grid, "scenes": scenes})

    def test_cohort_reports_no_night_overlap_or_training_exposure(self):
        samples, dataset = self._docs()
        out = analyze_presence(samples, dataset, generated="fixture")
        cohort = out["cohort"]
        self.assertEqual(cohort["night_overlap"]["validation_test"], 0)
        self.assertEqual(cohort["night_overlap"]["train_validation"], 0)
        self.assertEqual(cohort["night_overlap"]["train_test"], 0)
        self.assertEqual(cohort["night_overlap"]["nights_in_multiple_splits"], 0)
        for split in ("val", "test"):
            self.assertEqual(cohort["training_exposure"][split]["nights_seen_in_train"], 0)
            self.assertEqual(cohort["training_exposure"][split]["scans_on_nights_seen_in_train"], 0)

    def test_all_evaluated_nights_are_complete(self):
        # Whole nights per split means no night is only partially evaluated.
        samples, dataset = self._docs()
        out = analyze_presence(samples, dataset, generated="fixture")
        for split in ("val", "test"):
            coverage = out["cohort"]["night_coverage"][split]
            self.assertEqual(coverage["partial_nights"], 0)
            self.assertEqual(coverage["complete_nights"], coverage["nights_total"])


class _FakeStore:
    """Stands in for SceneStore so dataset tests need no netCDF.

    Scenes are full IMG_SIZE squares because _extract_patch crops against the
    real grid size. Each scene is filled with a distinct NON-ZERO constant so a
    padded temporal slot (a copy of the centre frame) is distinguishable from a
    zero-filled one.
    """

    def __init__(self, rows, n_channels=2):
        self.rows = rows
        self.n_channels = n_channels

    def get(self, idx):
        row = self.rows[idx]
        x = np.full((self.n_channels, IMG_SIZE, IMG_SIZE), float(idx) + 1.0, dtype=np.float32)
        y = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        if int(row["label"]) == 1:
            y[:64, :64] = 1.0      # a small plume in one corner only
        return x, y


def _patch_dataset(rows, **cfg_over):
    """RadarPatchDataset wired to a fake store (no disk access)."""
    cfg = {
        "channels": ["th_e0", "valid_mask"],
        "patch": {"size": 64, "patches_per_image": 4,
                  "pos_sample_rate": 0.5, "hard_pos_rate": 0.3},
        "target": {"mode": "threshold", "dbz_threshold": 0.0},
        "train": {"scene_cache": 8},
        "augment": {"enabled": False},
        "model": {"name": "attention_unet"},
    }
    cfg.update(cfg_over)
    manifest = pd.DataFrame(rows)
    ds = RadarPatchDataset(cfg, manifest, "train", {}, mode="train")
    ds.store = _FakeStore(ds.rows)
    return ds


def _night_rows():
    df = nights.add_night_ids(synthetic_manifest(n_nights=12, years=(2013, 2014)))
    df["split"] = "train"
    return df.to_dict("records")


class PatchPresenceLabels(unittest.TestCase):
    """The label must come from the patch, never from the parent scan."""

    def test_label_is_the_patch_s_own_not_the_scan_s(self):
        rows = _night_rows()
        ds = _patch_dataset(rows, model={"name": "attention_unet", "cls_head": True})
        positives = [i for i, r in enumerate(ds.rows) if int(r["label"]) == 1]
        self.assertTrue(positives)
        _x, y = ds.store.get(positives[0])
        self.assertEqual(int(ds.rows[positives[0]]["label"]), 1, "scan-level label is positive")

        # A crop far from the plume belongs to a POSITIVE scan but is itself empty.
        empty = y[500:564, 500:564]
        self.assertEqual(empty.sum(), 0)
        self.assertEqual(float(empty.sum() > 0), 0.0,
                         "a plume-free crop of a positive scan must be labelled negative; "
                         "stamping the scan label onto every patch is the noisy "
                         "bag-label propagation this design exists to avoid")
        # A crop over the plume is positive.
        self.assertEqual(float(y[:64, :64].sum() > 0), 1.0)

    def test_dataset_emits_cls_only_when_the_head_is_enabled(self):
        rows = _night_rows()
        off = _patch_dataset(rows)
        self.assertEqual(len(off[0]), 2, "plain runs must keep the (x, y) contract")

        on = _patch_dataset(rows, model={"name": "attention_unet", "cls_head": True})
        item = on[0]
        self.assertEqual(len(item), 3)
        self.assertIn("cls", item[2])
        self.assertEqual(tuple(item[2]["cls"].shape), (1,))
        self.assertIn(float(item[2]["cls"]), (0.0, 1.0))

    def test_label_matches_its_own_target_crop(self):
        rows = _night_rows()
        ds = _patch_dataset(rows, model={"name": "attention_unet", "cls_head": True})
        for i in range(0, len(ds), 7):
            _x, y, extra = ds[i]
            self.assertEqual(float(extra["cls"]), float(y.sum() > 0))


class MultiTask(unittest.TestCase):
    def _cfg(self, lambda_cls):
        return {
            "channels": ["th_e%d" % i for i in range(9)] + ["valid_mask"],
            "model": {"name": "attention_unet", "base_filters": 8, "cls_head": True},
            "loss": {"name": "focal_tversky", "alpha": 0.3, "beta": 0.7,
                     "gamma": 1.333, "lambda_cls": lambda_cls},
        }

    def test_output_shapes(self):
        model = create_model(self._cfg(0.3))
        seg, cls = model(torch.randn(2, 10, 32, 32))
        self.assertEqual(tuple(seg.shape), (2, 1, 32, 32))
        self.assertEqual(tuple(cls.shape), (2, 1))

    def test_lambda_zero_reproduces_pure_segmentation_loss(self):
        cfg = self._cfg(0.0)
        model = create_model(cfg)
        out = model(torch.randn(2, 10, 32, 32))
        y = torch.rand(2, 1, 32, 32).round()
        cls_t = torch.tensor([[1.0], [0.0]])
        multi = create_loss(cfg)
        plain = create_loss({**cfg, "model": {"name": "attention_unet", "base_filters": 8}})
        self.assertAlmostEqual(float(multi(out, y, cls_t)),
                               float(plain(out[0], y)), places=6)

    def test_nonzero_lambda_changes_the_loss(self):
        cfg = self._cfg(0.5)
        model = create_model(cfg)
        out = model(torch.randn(2, 10, 32, 32))
        y = torch.rand(2, 1, 32, 32).round()
        cls_t = torch.tensor([[1.0], [0.0]])
        self.assertNotAlmostEqual(float(create_loss(cfg)(out, y, cls_t)),
                                  float(create_loss(self._cfg(0.0))(out, y, cls_t)))

    def test_gradients_reach_both_heads(self):
        cfg = self._cfg(0.5)
        model = create_model(cfg)
        out = model(torch.randn(2, 10, 32, 32))
        create_loss(cfg)(out, torch.rand(2, 1, 32, 32).round(),
                         torch.tensor([[1.0], [0.0]])).backward()
        grads = {n for n, p in model.named_parameters()
                 if p.grad is not None and float(p.grad.abs().sum()) > 0}
        self.assertTrue(any(n.startswith("cls_") for n in grads), "classifier got no gradient")
        self.assertTrue(any(n.startswith("enc1") for n in grads), "shared encoder got no gradient")

    def test_plain_model_still_returns_a_bare_tensor(self):
        cfg = self._cfg(0.3)
        cfg["model"] = {"name": "attention_unet", "base_filters": 8}
        out = create_model(cfg)(torch.randn(1, 10, 32, 32))
        self.assertIsInstance(out, torch.Tensor)
        self.assertIs(seg_logits(out), out)
        self.assertFalse(getattr(create_loss(cfg), "is_multitask", False))


class TemporalSequences(unittest.TestCase):
    def setUp(self):
        df = nights.add_night_ids(synthetic_manifest(n_nights=12, years=(2013, 2014)))
        df["split"] = nights.assign_night_split(df, FRACTIONS, seed=42)
        self.rows = df.to_dict("records")
        self.index = build_sequence_index(self.rows, radius=1, max_gap_minutes=35.0)

    def test_never_crosses_night_or_split(self):
        for i, window in self.index.items():
            for j in window:
                if j is None:
                    continue
                self.assertEqual(self.rows[j]["night"], self.rows[i]["night"])
                self.assertEqual(self.rows[j]["split"], self.rows[i]["split"])

    def test_centre_slot_is_always_the_scan_itself(self):
        for i, window in self.index.items():
            self.assertEqual(window[1], i, "the centre frame must never be padding")

    def test_chronological_order(self):
        for i, window in self.index.items():
            stamps = [int(self.rows[j]["timestamp"]) for j in window if j is not None]
            self.assertEqual(stamps, sorted(stamps))

    def test_gap_tolerance_is_enforced(self):
        for i, window in self.index.items():
            centre = pd.to_datetime(str(self.rows[i]["timestamp"]), format="%Y%m%d%H%M")
            for j in window:
                if j is None or j == i:
                    continue
                other = pd.to_datetime(str(self.rows[j]["timestamp"]), format="%Y%m%d%H%M")
                self.assertLessEqual(abs((other - centre).total_seconds()) / 60.0, 35.0)

        tight = build_sequence_index(self.rows, radius=1, max_gap_minutes=1.0)
        self.assertTrue(all(w[0] is None and w[2] is None for w in tight.values()),
                        "a 1-minute tolerance must reject every 30-minute neighbour")

    def test_night_boundaries_produce_padding(self):
        # The first and last scan of a night can have at most one neighbour.
        by_night = {}
        for i, r in enumerate(self.rows):
            by_night.setdefault(r["night"], []).append(i)
        for idxs in by_night.values():
            ordered = sorted(idxs, key=lambda i: int(self.rows[i]["timestamp"]))
            self.assertIsNone(self.index[ordered[0]][0], "night's first scan has no predecessor")
            self.assertIsNone(self.index[ordered[-1]][2], "night's last scan has no successor")

    def test_pad_mask_matches_missing_neighbours(self):
        ds = _patch_dataset(
            self.rows, model={"name": "attention_unet", "cls_head": True},
            temporal={"enabled": True, "radius": 1, "max_gap_minutes": 35.0,
                      "mode": "attention"})
        ds.store = _FakeStore(ds.rows)
        for img_idx in range(0, len(ds.rows), 5):
            x, _y, pad = ds._scene(img_idx)
            self.assertEqual(x.shape[0], 3)
            expected = [j is None for j in ds.seq_index[img_idx]]
            self.assertEqual(list(pad), expected)
            # Padded slots hold a COPY OF THE CENTRE FRAME, not zeros: that keeps
            # BatchNorm statistics real while attention masks the slot out.
            for t, is_pad in enumerate(pad):
                if is_pad:
                    np.testing.assert_array_equal(x[t], x[1])
                    self.assertNotEqual(float(np.abs(x[t]).sum()), 0.0)

    def test_isolated_scan_degrades_to_the_single_frame_model(self):
        cfg = {"channels": ["th_e%d" % i for i in range(9)] + ["valid_mask"],
               "model": {"name": "attention_unet", "base_filters": 8},
               "temporal": {"enabled": True, "radius": 1, "mode": "attention",
                            "n_head": 8, "d_k": 8},
               "loss": {"name": "focal_tversky"}}
        model = create_model(cfg).eval()
        frame = torch.randn(1, 1, 10, 32, 32)
        seq = frame.repeat(1, 3, 1, 1, 1)
        with torch.no_grad():
            all_pad = model(seq, torch.tensor([[True, False, True]]))
            none_pad = model(seq, torch.tensor([[False, False, False]]))
        # With identical frames, masking must not change the aggregate: attention
        # over copies of one frame is that frame either way.
        self.assertTrue(torch.allclose(all_pad, none_pad, atol=1e-5),
                        "an isolated scan must behave exactly like the single-frame model")
        self.assertTrue(torch.isfinite(all_pad).all())

    def test_stack_mode_folds_time_into_channels_without_a_mask(self):
        ds = _patch_dataset(
            self.rows,
            temporal={"enabled": True, "radius": 1, "max_gap_minutes": 35.0, "mode": "stack"})
        ds.store = _FakeStore(ds.rows)
        item = ds[0]
        self.assertEqual(item[0].shape[0], 3 * 2, "stack mode must widen the channel axis")
        extra = item[2] if len(item) > 2 else {}
        self.assertNotIn("pad_mask", extra, "the 2.5D control has no masking mechanism")

    def test_stack_mode_widens_the_first_conv(self):
        cfg = {"channels": ["th_e%d" % i for i in range(9)] + ["valid_mask"],
               "model": {"name": "attention_unet", "base_filters": 8},
               "temporal": {"enabled": True, "radius": 1, "mode": "stack"},
               "loss": {"name": "focal_tversky"}}
        out = create_model(cfg)(torch.randn(1, 30, 32, 32))
        self.assertEqual(tuple(out.shape), (1, 1, 32, 32))

    def test_temporal_cfg_defaults_are_off(self):
        self.assertFalse(temporal_cfg({})["enabled"])
        self.assertEqual(temporal_cfg({})["mode"], "attention")


class NightBalancedSampling(unittest.TestCase):
    def setUp(self):
        self.rows = _night_rows()
        self.ds = _patch_dataset(self.rows)

    def _scene_counts(self, sampler):
        indices = list(iter(sampler))
        scenes = [i // self.ds.patches_per_image for i in indices]
        return scenes, indices

    def test_epoch_length_matches_the_unbalanced_sampler(self):
        balanced = NightBalancedSceneSampler(self.ds, seed=0)
        plain = SceneGroupedSampler(self.ds, seed=0)
        self.assertEqual(len(balanced), len(plain))
        self.assertEqual(len(list(iter(balanced))), len(balanced))

    def test_patch_indices_stay_contiguous_per_scene(self):
        _scenes, indices = self._scene_counts(NightBalancedSceneSampler(self.ds, seed=0))
        ppi = self.ds.patches_per_image
        for start in range(0, len(indices), ppi):
            block = indices[start:start + ppi]
            self.assertEqual(block, list(range(block[0], block[0] + ppi)),
                             "a scene's patches must stay together or the LRU cache thrashes")

    def test_migration_and_quiet_nights_are_balanced(self):
        sampler = NightBalancedSceneSampler(self.ds, seed=0)
        scenes, _ = self._scene_counts(sampler)
        labels = [int(self.ds.rows[s]["label"]) for s in scenes]
        share = sum(labels) / len(labels)
        self.assertAlmostEqual(share, 0.5, delta=0.12,
                               msg=f"migration share {share:.3f} should be near balanced")

    def test_balancing_actually_changes_the_natural_distribution(self):
        natural = [int(r["label"]) for r in self.ds.rows]
        natural_share = sum(natural) / len(natural)
        sampler = NightBalancedSceneSampler(self.ds, seed=0)
        scenes, _ = self._scene_counts(sampler)
        balanced_share = sum(int(self.ds.rows[s]["label"]) for s in scenes) / len(scenes)
        self.assertGreater(abs(balanced_share - natural_share), 0.05,
                           "balanced sampling should differ from the natural distribution")

    def test_per_night_contribution_is_more_even_than_baseline(self):
        def spread(sampler):
            scenes, _ = self._scene_counts(sampler)
            per_night = {}
            for s in scenes:
                night = self.ds.rows[s]["night"]
                per_night[night] = per_night.get(night, 0) + 1
            counts = np.array(list(per_night.values()), dtype=float)
            return counts.std() / counts.mean()
        self.assertLess(spread(NightBalancedSceneSampler(self.ds, seed=0)),
                        spread(SceneGroupedSampler(self.ds, seed=0)),
                        "night-balanced sampling should even out per-night contribution")

    def test_deterministic_per_seed_and_varies_by_epoch(self):
        a = NightBalancedSceneSampler(self.ds, seed=3)
        b = NightBalancedSceneSampler(self.ds, seed=3)
        self.assertEqual(list(iter(a)), list(iter(b)))
        b.set_epoch(1)
        self.assertNotEqual(list(iter(a)), list(iter(b)))


class PublicationMetrics(unittest.TestCase):
    def test_empty_pred_on_positive_is_nsd_zero_not_nan(self):
        from src.metrics import surface_metrics
        true = np.zeros((32, 32), dtype=np.uint8)
        true[8:16, 8:16] = 1
        pred = np.zeros((32, 32), dtype=np.uint8)
        m = surface_metrics(pred, true, tau=2.0)
        self.assertEqual(m["nsd"], 0.0)
        self.assertTrue(np.isnan(m["hd95"]))
        self.assertTrue(np.isnan(m["assd"]))
        self.assertEqual(m["nsd_curve"]["nsd"][0], 0.0)

    def test_both_empty_is_perfect(self):
        from src.metrics import surface_metrics
        z = np.zeros((16, 16), dtype=np.uint8)
        m = surface_metrics(z, z, tau=2.0)
        self.assertEqual(m["nsd"], 1.0)
        self.assertEqual(m["hd95"], 0.0)

    def test_existing_nsd_unchanged_when_both_surfaces_exist(self):
        from src.metrics import surface_metrics
        true = np.zeros((32, 32), dtype=np.uint8)
        pred = np.zeros((32, 32), dtype=np.uint8)
        true[8:20, 8:20] = 1
        pred[9:21, 9:21] = 1
        m = surface_metrics(pred, true, tau=2.0)
        self.assertGreater(m["nsd"], 0.5)
        self.assertTrue(np.isfinite(m["hd95"]))
        self.assertIn("bf1_fuzzy", m)

    def test_cluster_bootstrap_shared_draws(self):
        from src.stats import cluster_bootstrap_many
        rec = [{"dice": 0.5, "iou": 0.4, "night": "a"},
               {"dice": 0.7, "iou": 0.6, "night": "a"},
               {"dice": 0.2, "iou": 0.1, "night": "b"}]
        out = cluster_bootstrap_many(rec, [r["night"] for r in rec],
                                     ["dice", "iou"], n_boot=50, seed=0)
        self.assertIn("lo", out["dice"])
        self.assertEqual(out["dice"]["n_nights"], 2)

    def test_night_config_gate_off_and_balanced(self):
        from src import config as cfgmod
        cfg = cfgmod.load_base_config(ROOT / "configs" / "base_config_night.yaml")
        self.assertIn(str(cfg["eval"].get("gate", "off")), ("off", "", "None"))
        self.assertTrue(cfg["negatives"]["balanced"])
        self.assertEqual(float(cfg["negatives"]["ratio"]), 1.0)
        self.assertAlmostEqual(float(cfg["eval"]["cls_r_min"]), 0.98)
        self.assertLess(min(cfg["eval"]["cls_threshold_range"]), 1e-5)

    def test_seed_experiments_load(self):
        from src import config as cfgmod
        base = cfgmod.load_base_config(ROOT / "configs" / "base_config_night.yaml")
        exps = cfgmod.load_experiments(ROOT / "configs" / "experiments_night_seeds.yaml")
        names = [e["name"] for e in exps]
        self.assertEqual(len(names), 8)
        cfg = cfgmod.resolve_experiment(base, exps[0])
        self.assertEqual(cfg["train"]["seed"], 42)
        self.assertEqual(cfg["split"]["seed"], 42)
        self.assertEqual(cfg["eval"].get("gate"), "off")

    def test_calibrate_cls_uses_sensitivity_floor(self):
        from src.engine import calibrate_cls_threshold, metrics_from_cache
        rng = np.random.default_rng(0)
        items = []
        for i in range(40):
            label = 1 if i < 20 else 0
            y = np.zeros((8, 8), np.float32)
            if label:
                y[:4, :4] = 1
            p_cls = 0.9 if label else 1e-6
            if i == 0:
                p_cls = 1e-7  # one hard positive
            items.append({"label": label, "y": y, "prob": y * 0.8 + 0.1,
                          "p_cls": p_cls, "ts": i, "night": str(i // 5)})
        t = calibrate_cls_threshold(items, 0.5, [1e-6, 1e-4, 0.5, 0.9], r_min=0.9)
        m = metrics_from_cache(items, 0.5, t, gate="hard")
        self.assertGreaterEqual(m["scan_recall"], 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
