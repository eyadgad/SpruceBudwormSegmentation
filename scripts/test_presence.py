"""Focused tests for the GPU-free presence-analysis core."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import export_dashboard_data as export  # noqa: E402

from src.presence import (  # noqa: E402
    SCHEMA_VERSION,
    analyze_presence,
    classification_metrics,
    mann_whitney_analysis,
    operational_night_id,
    pr_analysis,
    roc_analysis,
    select_high_sensitivity_cutoff,
    select_youden_cutoff,
)


def _fixture_documents():
    models = [{"key": "m", "name": "model", "disp": "Model", "thr": 0.15}]
    dataset = {
        "grid": {"h": 960, "w": 960, "pixel_m": 500},
        "scenes": [
            {"ts": 201307012300, "split": "train", "label": 1, "area": 10},
            {"ts": 201307020100, "split": "test", "label": 1, "area": 8},
            {"ts": 201307032300, "split": "val", "label": 1, "area": 6},
            {"ts": 201307042300, "split": "val", "label": 0, "area": None},
            {"ts": 201307052300, "split": "test", "label": 0, "area": None},
        ],
    }
    scores = {
        201307020100: 7,
        201307032300: 8,
        201307042300: 2,
        201307052300: 1,
    }
    samples = {
        "selected": "model",
        "models": models,
        "samples": [
            {"ts": ts, "split": next(r["split"] for r in dataset["scenes"] if r["ts"] == ts),
             "gt_area": next((r["area"] or 0) for r in dataset["scenes"] if r["ts"] == ts),
             "models": {"m": {"pred_area": score}}}
            for ts, score in scores.items()
        ],
    }
    return samples, dataset


class PresenceCoreTests(unittest.TestCase):
    def test_pr_analysis_step_ap_and_baseline(self):
        # Perfect ranking of 2 pos / 2 neg: AP = 1, baseline = 0.5
        got = pr_analysis([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1])
        self.assertAlmostEqual(got["ap"], 1.0)
        self.assertAlmostEqual(got["baseline_precision"], 0.5)
        self.assertEqual(len(got["points"]), 101)
        self.assertEqual(got["points"][0]["recall"], 0.0)
        self.assertEqual(got["points"][-1]["recall"], 1.0)

    def test_analyze_presence_schema_and_score_field(self):
        samples, dataset = _fixture_documents()
        doc = analyze_presence(samples, dataset, generated="fixture")
        self.assertEqual(doc["schema_version"], SCHEMA_VERSION)
        self.assertEqual(SCHEMA_VERSION, 2)
        self.assertIn("pr", doc["models"][0]["scan"]["splits"]["validation"])
        self.assertIn("high_sensitivity",
                      doc["models"][0]["scan"]["splits"]["validation"]["operating_points"])
        for s in samples["samples"]:
            s["models"]["m"]["p_cls"] = 0.99 if s["gt_area"] else 0.01
        pcls = analyze_presence(samples, dataset, generated="fixture", score_field="p_cls")
        self.assertEqual(pcls["definitions"]["score_field"], "p_cls")
        self.assertNotIn("any_cell",
                         pcls["models"][0]["scan"]["splits"]["validation"]["operating_points"])

    def test_high_sensitivity_cutoff_respects_r_min(self):
        # Scores well separated; r_min=1.0 should pick a cutoff that keeps all positives.
        got = select_high_sensitivity_cutoff([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.05], r_min=1.0)
        self.assertGreaterEqual(got["validation_sensitivity"], 1.0)
        self.assertTrue(got["met_constraint"])

    def test_operational_night_noon_boundaries_and_rollovers(self):
        self.assertEqual(operational_night_id(201307122330), "2013-07-12")
        self.assertEqual(operational_night_id(201307130000), "2013-07-12")
        self.assertEqual(operational_night_id(201307131159), "2013-07-12")
        self.assertEqual(operational_night_id(201307131200), "2013-07-13")
        self.assertEqual(operational_night_id(202001010030), "2019-12-31")
        self.assertEqual(operational_night_id(202003010030), "2020-02-29")

    def test_classification_metrics_and_undefined_mcc(self):
        got = classification_metrics([1, 1, 0, 0], [9, 2, 3, 0], 2)
        self.assertEqual(got["confusion"], {"tp": 2, "fp": 1, "tn": 1, "fn": 0})
        self.assertAlmostEqual(got["sensitivity"], 1.0)
        self.assertAlmostEqual(got["specificity"], 0.5)
        self.assertAlmostEqual(got["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(got["mcc"], 1 / 3 ** 0.5)
        all_positive = classification_metrics([1, 1], [2, 3], 1)
        self.assertIsNone(all_positive["specificity"])
        self.assertIsNone(all_positive["balanced_accuracy"])
        self.assertIsNone(all_positive["mcc"])

    def test_roc_auc_perfect_reversed_and_ties(self):
        self.assertAlmostEqual(roc_analysis([1, 1, 0, 0], [4, 3, 2, 1])["auc"], 1.0)
        self.assertAlmostEqual(roc_analysis([1, 1, 0, 0], [1, 2, 3, 4])["auc"], 0.0)
        tied = roc_analysis([1, 0], [5, 5])
        self.assertAlmostEqual(tied["auc"], 0.5)
        fpr = [p["false_positive_rate"] for p in tied["points"]]
        tpr = [p["true_positive_rate"] for p in tied["points"]]
        self.assertEqual(fpr, sorted(fpr))
        self.assertEqual(tpr, sorted(tpr))

    def test_youden_tie_prefers_specificity_then_higher_cutoff(self):
        # Both all-negative and all-positive have J=0. The declared tie break
        # must choose all-negative because it has higher specificity.
        chosen = select_youden_cutoff([1, 0], [0, 0])
        self.assertEqual(chosen["cutoff"], 1.0)
        self.assertEqual(chosen["validation_specificity"], 1.0)

        # J is mathematically 1/6 at both cutoffs 2 and 1. Binary floating-point
        # gives those values slightly different representations, so selection
        # must compare the exact confusion-count numerator before specificity.
        chosen = select_youden_cutoff(
            [1, 1, 0, 0, 0, 0, 0, 0],
            [2, 1, 2, 2, 1, 1, 1, 0],
        )
        self.assertEqual(chosen["cutoff"], 2.0)
        self.assertAlmostEqual(chosen["validation_specificity"], 2 / 3)

    def test_mann_whitney_matches_auc_effect_size(self):
        got = mann_whitney_analysis([1, 1, 0, 0], [4, 3, 2, 1])
        self.assertEqual(got["alternative"], "two-sided")
        self.assertEqual(got["u"], 4.0)
        self.assertEqual(got["common_language_auc"], 1.0)
        self.assertEqual(got["rank_biserial"], 1.0)

    def test_end_to_end_fixture_uses_full_night_truth_and_val_cutoff(self):
        samples, dataset = _fixture_documents()
        out = analyze_presence(samples, dataset, generated="fixture")
        self.assertEqual(out["generated"], "fixture")
        self.assertEqual(out["cohort"]["training_exposure"]["test"]["nights_seen_in_train"], 1)
        model = out["models"][0]
        self.assertEqual(model["scan"]["selected_cutoff"]["cells"], 8.0)
        test_selected = model["scan"]["splits"]["test"]["operating_points"]["validation_selected"]
        self.assertEqual(test_selected["cutoff"], 8.0)
        self.assertEqual(test_selected["confusion"], {"tp": 0, "fp": 0, "tn": 1, "fn": 1})
        night_row = model["night"]["max"]["splits"]["test"]["records"][0]
        self.assertIn("evaluated_scan_count", night_row)
        self.assertIn("manifest_scan_count", night_row)
        self.assertNotIn("mann_whitney", model["scan"]["splits"]["test"])
        self.assertIn("mann_whitney", model["night"]["max"]["splits"]["test"])

    def test_schema_validation_rejects_invalid_units_and_models(self):
        cases = []

        samples, dataset = _fixture_documents()
        dataset["grid"]["pixel_m"] = 0
        cases.append((samples, dataset, "pixel_m must be finite and positive"))

        samples, dataset = _fixture_documents()
        dataset["scenes"][0]["area"] = 1.5
        cases.append((samples, dataset, "ground-truth area.*must be an integer"))

        samples, dataset = _fixture_documents()
        samples["samples"][0]["gt_area"] = 1.5
        cases.append((samples, dataset, "sample ground-truth area.*must be an integer"))

        samples, dataset = _fixture_documents()
        samples["samples"][0]["models"]["m"]["pred_area"] = 960 * 960 + 1
        cases.append((samples, dataset, "predicted area.*must be an integer"))

        samples, dataset = _fixture_documents()
        samples["models"].append(copy.deepcopy(samples["models"][0]))
        cases.append((samples, dataset, "model keys must be non-empty and unique"))

        samples, dataset = _fixture_documents()
        samples["models"][0]["thr"] = 1.1
        cases.append((samples, dataset, "pixel threshold must be finite in \\[0, 1\\]"))

        samples, dataset = _fixture_documents()
        dataset["scenes"][0]["split"] = "trian"
        cases.append((samples, dataset, "split must be train, val, or test"))

        for samples, dataset, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    analyze_presence(samples, dataset, generated="fixture")

    def test_partial_predict_plus_presence_is_rejected_before_writes(self):
        with self.assertRaisesRegex(SystemExit, "--limit cannot be combined"):
            export._validate_stage_request(["predict", "presence"], 10)
        export._validate_stage_request(["predict"], 10)
        export._validate_stage_request(["presence"], 10)

    def test_preserved_metric_lineage_must_match_version_name_and_threshold(self):
        spec = {"key": "m", "name": "model", "disp": "Model"}
        previous = {
            "sample_assets": {"model_artifact_version": "models-abc"},
            "models": [{"key": "m", "name": "model", "thr": 0.15}],
        }
        export._validate_preserved_model_lineage(previous, [(spec, 0.15)], "models-abc")
        mutations = [
            ("version", lambda p: p["sample_assets"].update(model_artifact_version="models-old")),
            ("name", lambda p: p["models"][0].update(name="other")),
            ("threshold", lambda p: p["models"][0].update(thr=0.2)),
            ("missing key", lambda p: p.update(models=[])),
        ]
        for label, mutate in mutations:
            candidate = copy.deepcopy(previous)
            mutate(candidate)
            with self.subTest(label=label):
                with self.assertRaisesRegex(SystemExit, "cannot reuse missing-checkpoint"):
                    export._validate_preserved_model_lineage(
                        candidate, [(spec, 0.15)], "models-abc")

    def test_stage_images_rejects_stale_pack_lineage_before_png_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_out = Path(tmp)
            stale = {
                "sample_assets": {"model_artifact_version": "models-stale"},
                "models": [
                    {"key": spec["key"], "name": spec["name"], "thr": 0.15}
                    for spec in export.VIEWER_MODELS
                ],
            }
            (data_out / "samples.json").write_text(
                json.dumps(stale), encoding="utf-8")

            fake_torch = types.ModuleType("torch")
            fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
            fake_torch.device = lambda name: name
            fake_data_prep = types.ModuleType("src.data_prep")
            fake_data_prep.load_artifacts = lambda _base: (object(), object())
            fake_dataset = types.ModuleType("src.dataset")
            fake_engine = types.ModuleType("src.engine")
            fake_config = types.ModuleType("src.config")
            fake_config.load_base_config = lambda _path: {}
            fake_modules = {
                "torch": fake_torch,
                "src.data_prep": fake_data_prep,
                "src.dataset": fake_dataset,
                "src.engine": fake_engine,
                "src.config": fake_config,
            }
            png_writer = mock.Mock()
            with mock.patch.dict(sys.modules, fake_modules), \
                    mock.patch.object(export, "DATA_OUT", data_out), \
                    mock.patch.object(export, "_load_model",
                                      side_effect=lambda _name, _device, required=False:
                                      (None, {}, 0.15)), \
                    mock.patch.object(export, "_model_artifact_version",
                                      return_value="models-current"), \
                    mock.patch.object(export, "_to_png", png_writer):
                with self.assertRaisesRegex(
                        SystemExit, "cannot reuse missing-checkpoint scene metrics"):
                    export.stage_images(limit=1)
            png_writer.assert_not_called()

    def test_model_artifact_fingerprint_tracks_inference_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [
                "configs/base_config.yaml",
                "configs/experiments_elev.yaml",
                "artifacts/norm_stats.json",
                "src/channels.py",
                "src/config.py",
                "src/checkpoint.py",
                "src/dataset.py",
                "src/engine.py",
                "src/models/__init__.py",
                "src/models/architecture.py",
                "scripts/export_dashboard_data.py",
                "outputs/experiments/model_result.json",
                "outputs/checkpoints/model_best.pt",
            ]
            for rel in files:
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"fixture:{rel}\n", encoding="utf-8")
            models = ({"key": "m", "name": "model", "disp": "Model"},)
            exporter_path = root / "scripts" / "export_dashboard_data.py"

            def version():
                return export._model_artifact_version(
                    root=root, exporter_path=exporter_path, viewer_models=models)

            baseline = version()
            dependencies = [
                "artifacts/norm_stats.json",
                "src/channels.py",
                "src/models/__init__.py",
                "src/models/architecture.py",
                "scripts/export_dashboard_data.py",
            ]
            for rel in dependencies:
                path = root / rel
                original = path.read_bytes()
                path.write_bytes(original + b"changed\n")
                with self.subTest(dependency=rel):
                    self.assertNotEqual(version(), baseline)
                path.write_bytes(original)
                self.assertEqual(version(), baseline)

            added_model = root / "src" / "models" / "new_architecture.py"
            added_model.write_text("new model\n", encoding="utf-8")
            self.assertNotEqual(version(), baseline)

    def test_subset_that_hides_full_night_presence_is_rejected(self):
        # These two scans share one noon-to-noon night. Seeing only the negative
        # test scan must not let the exporter relabel the full positive night.
        dataset = {
            "grid": {"h": 960, "w": 960, "pixel_m": 500},
            "scenes": [
                {"ts": 201307012300, "split": "train", "label": 1, "area": 10},
                {"ts": 201307020100, "split": "test", "label": 0, "area": None},
            ],
        }
        samples = {
            "selected": "model",
            "models": [{"key": "m", "name": "model", "disp": "Model", "thr": 0.15}],
            "samples": [
                {"ts": 201307020100, "split": "test", "gt_area": 0,
                 "models": {"m": {"pred_area": 0}}},
            ],
        }
        with self.assertRaisesRegex(ValueError, "evaluated subset does not preserve"):
            analyze_presence(samples, dataset, generated="fixture")


class GeneratedPresenceTests(unittest.TestCase):
    def test_generated_document_if_present(self):
        site = ROOT.parent / "sprucebudworm_progress.github.io" / "data"
        path = site / "presence.json"
        if not path.exists():
            self.skipTest("presence.json has not been generated")
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema_version"], 1)
        self.assertEqual(doc["selected_model_key"], "attunet9")
        self.assertEqual(doc["defaults"]["model_key"], "attunet9")
        self.assertEqual(doc["defaults"]["night_aggregation"], "max")
        self.assertEqual(doc["defaults"]["scan_operating_point"], "any_cell")
        self.assertEqual(doc["defaults"]["night_operating_point"], "validation_selected")
        self.assertTrue(any(
            "maximum scores are especially sensitive to unequal evaluated scan counts"
            in caveat for caveat in doc["caveats"]))
        self.assertEqual(len(doc["models"]), 4)
        self.assertEqual(sum(bool(m["selected"]) for m in doc["models"]), 1)
        self.assertEqual(doc["cohort"]["evaluation_scans"], 615)
        cohort = doc["cohort"]
        self.assertEqual(cohort["training_exposure"]["test"]["nights_seen_in_train"], 110)
        self.assertEqual(cohort["training_exposure"]["test"]["nights_total"], 113)
        self.assertEqual(cohort["night_overlap"]["validation_test"], 93)
        self.assertEqual(cohort["night_coverage"]["val"]["partial_nights"], 154)
        self.assertEqual(cohort["night_coverage"]["test"]["partial_nights"], 111)
        self.assertEqual(cohort["subset_full_night_truth_disagreements"], 0)
        summary = doc["models"][0]["night"]["max"]["splits"]["test"]["score_summary"]
        self.assertTrue(all("p05" in s and "p95" in s for s in summary.values()))

        expected = {
            "attunet9": {
                "scan": (18081.0, 0.8989529498624065, 0.9184256055363321,
                         {"tp": 127, "fp": 3, "tn": 31, "fn": 43}),
                "scan_any": {"tp": 169, "fp": 27, "tn": 7, "fn": 1},
                "max": (26207.0, 0.9329937927741525, 0.9236453201970442,
                        {"tp": 66, "fp": 3, "tn": 26, "fn": 18}),
                "mean": (12956.0, 0.9267865669266275, 0.9142036124794743,
                         {"tp": 73, "fp": 5, "tn": 24, "fn": 11}),
            },
            "unetpp9": {
                "scan": (16359.0, 0.8953621048392506, 0.9304498269896193,
                         {"tp": 127, "fp": 3, "tn": 31, "fn": 43}),
                "scan_any": {"tp": 168, "fp": 25, "tn": 9, "fn": 2},
                "max": (16359.0, 0.936495304790705, 0.9339080459770116,
                        {"tp": 69, "fp": 3, "tn": 26, "fn": 15}),
                "mean": (11873.5, 0.9309247174916441, 0.9252873563218391,
                         {"tp": 72, "fp": 4, "tn": 25, "fn": 12}),
            },
            "attunet7": {
                "scan": (16924.0, 0.9033492180683269, 0.9139273356401383,
                         {"tp": 126, "fp": 3, "tn": 31, "fn": 44}),
                "scan_any": {"tp": 167, "fp": 21, "tn": 13, "fn": 3},
                "max": (17316.0, 0.9345853891453128, 0.9207717569786535,
                        {"tp": 69, "fp": 3, "tn": 26, "fn": 15}),
                "mean": (13792.0, 0.9304472385802961, 0.9072249589490968,
                         {"tp": 72, "fp": 5, "tn": 24, "fn": 12}),
            },
            "attunet8": {
                "scan": (17505.0, 0.8958319350291959, 0.9160899653979239,
                         {"tp": 125, "fp": 3, "tn": 31, "fn": 45}),
                "scan_any": {"tp": 170, "fp": 29, "tn": 5, "fn": 0},
                "max": (16027.0, 0.9315613560401081, 0.9222085385878489,
                        {"tp": 69, "fp": 5, "tn": 24, "fn": 15}),
                "mean": (13590.5, 0.924399172369887, 0.9090722495894907,
                         {"tp": 72, "fp": 5, "tn": 24, "fn": 12}),
            },
        }
        self.assertEqual({m["key"] for m in doc["models"]}, set(expected))
        for model in doc["models"]:
            self.assertNotIn("mann_whitney", model["scan"]["splits"]["validation"])
            self.assertNotIn("mann_whitney", model["scan"]["splits"]["test"])
            cutoff, val_auc, test_auc, confusion = expected[model["key"]]["scan"]
            self.assertEqual(model["scan"]["selected_cutoff"]["cells"], cutoff)
            self.assertAlmostEqual(model["scan"]["splits"]["validation"]["roc"]["auc"], val_auc)
            self.assertAlmostEqual(model["scan"]["splits"]["test"]["roc"]["auc"], test_auc)
            self.assertEqual(
                model["scan"]["splits"]["test"]["operating_points"]["validation_selected"]["cutoff"],
                cutoff,
            )
            self.assertEqual(
                model["scan"]["splits"]["test"]["operating_points"]["validation_selected"]["confusion"],
                confusion,
            )
            self.assertEqual(
                model["scan"]["splits"]["test"]["operating_points"]["any_cell"]["confusion"],
                expected[model["key"]]["scan_any"],
            )
            for aggregation in ("max", "mean"):
                cutoff, val_auc, test_auc, confusion = expected[model["key"]][aggregation]
                block = model["night"][aggregation]
                self.assertEqual(block["selected_cutoff"]["cells"], cutoff)
                self.assertAlmostEqual(block["splits"]["validation"]["roc"]["auc"], val_auc)
                self.assertAlmostEqual(block["splits"]["test"]["roc"]["auc"], test_auc)
                self.assertEqual(
                    block["splits"]["test"]["operating_points"]["validation_selected"]["confusion"],
                    confusion,
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
