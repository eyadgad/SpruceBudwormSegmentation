"""Scan- and night-level SBW presence evaluation.

The segmentation dashboard already stores the authoritative full-resolution
predicted area for every viewer model in ``samples.json``.  This module turns
those areas into binary-detection analyses without loading a checkpoint or a
packed preview image.

Two thresholds are deliberately kept separate:

* the model's locked *pixel probability* threshold creates ``pred_area``;
* an *area cutoff* converts that continuous scene/night score to presence.

Area cutoffs are selected on validation data only.  The test split is evaluated
at the unchanged validation cutoff.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from math import sqrt
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
from scipy import stats


SCHEMA_VERSION = 2
NIGHT_BOUNDARY_UTC_HOUR = 12
DEFAULT_GT_MIN_CELLS = 1
DEFAULT_PRED_MIN_CELLS = 1


def operational_night_id(timestamp: int | str,
                         boundary_hour_utc: int = NIGHT_BOUNDARY_UTC_HOUR) -> str:
    """Return the UTC operational-night start date for a 12-digit timestamp.

    With the default noon boundary, scans from 12:00 through 23:59 retain their
    calendar date, while scans from 00:00 through 11:59 belong to the preceding
    operational night.  Subtracting the boundary hour also handles month, year,
    and leap-day transitions without special cases.
    """
    if not 0 <= int(boundary_hour_utc) <= 23:
        raise ValueError("boundary_hour_utc must be between 0 and 23")
    text = str(timestamp).strip()
    if len(text) != 12 or not text.isdigit():
        raise ValueError(f"timestamp must contain 12 digits, got {timestamp!r}")
    dt = datetime.strptime(text, "%Y%m%d%H%M")
    return (dt - timedelta(hours=int(boundary_hour_utc))).date().isoformat()


def _finite_scores(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(arr).all():
        raise ValueError("scores must all be finite")
    return arr


def _binary_truth(values: Sequence[int | bool]) -> np.ndarray:
    arr = np.asarray(values).reshape(-1)
    if not np.isin(arr, (0, 1, False, True)).all():
        raise ValueError("truth labels must be binary")
    return arr.astype(bool)


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _integer_area(value, label: str, max_cells: int) -> int:
    """Validate a full-resolution cell count without silently truncating it."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer cell count") from exc
    if (not np.isfinite(number) or not number.is_integer() or
            number < 0 or number > max_cells):
        raise ValueError(f"{label} must be an integer in [0, {max_cells}], got {value!r}")
    return int(number)


def classification_metrics(truth: Sequence[int | bool], scores: Sequence[float],
                           cutoff: float) -> Dict:
    """Binary metrics for the rule ``score >= cutoff``.

    Undefined ratios are emitted as ``None`` so the result is strict JSON and
    cannot silently turn an absent class into a perfect or zero score.
    """
    y = _binary_truth(truth)
    s = _finite_scores(scores)
    if y.size != s.size:
        raise ValueError("truth and scores must have the same length")
    if not np.isfinite(float(cutoff)):
        raise ValueError("cutoff must be finite")

    p = s >= float(cutoff)
    tp = int(np.count_nonzero(p & y))
    fp = int(np.count_nonzero(p & ~y))
    tn = int(np.count_nonzero(~p & ~y))
    fn = int(np.count_nonzero(~p & y))
    n = int(y.size)

    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    precision = _ratio(tp, tp + fp)
    npv = _ratio(tn, tn + fn)
    f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    balanced = (float((sensitivity + specificity) / 2)
                if sensitivity is not None and specificity is not None else None)
    youden = (float(sensitivity + specificity - 1)
              if sensitivity is not None and specificity is not None else None)
    mcc_den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc = ((tp * tn - fp * fn) / sqrt(mcc_den)) if mcc_den else None

    return {
        "cutoff": float(cutoff),
        "n": n,
        "n_positive": int(y.sum()),
        "n_negative": int((~y).sum()),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": _ratio(tp + tn, n),
        "sensitivity": sensitivity,
        "recall": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "negative_predictive_value": npv,
        "f1": f1,
        "balanced_accuracy": balanced,
        "mcc": (float(mcc) if mcc is not None else None),
        "youden_j": youden,
    }


def roc_analysis(truth: Sequence[int | bool], scores: Sequence[float]) -> Dict:
    """Return a tie-safe empirical ROC curve and trapezoidal ROC-AUC."""
    y = _binary_truth(truth)
    s = _finite_scores(scores)
    if y.size != s.size:
        raise ValueError("truth and scores must have the same length")
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not s.size:
        return {"auc": None, "points": [], "n_positive": 0, "n_negative": 0}

    unique = sorted((float(v) for v in np.unique(s)), reverse=True)
    # A finite sentinel keeps the generated JSON standards-compliant.
    thresholds = [float(unique[0] + 1.0), *unique]
    points = []
    for cutoff in thresholds:
        m = classification_metrics(y, s, cutoff)
        points.append({
            "cutoff": cutoff,
            "false_positive_rate": (None if m["specificity"] is None
                                     else float(1.0 - m["specificity"])),
            "true_positive_rate": m["sensitivity"],
            "specificity": m["specificity"],
            "youden_j": m["youden_j"],
        })

    auc = None
    if n_pos and n_neg:
        auc_value = 0.0
        for left, right in zip(points, points[1:]):
            dx = right["false_positive_rate"] - left["false_positive_rate"]
            auc_value += dx * (right["true_positive_rate"] + left["true_positive_rate"]) / 2.0
        auc = float(auc_value)
    return {"auc": auc, "points": points,
            "n_positive": n_pos, "n_negative": n_neg}


def pr_analysis(truth: Sequence[int | bool], scores: Sequence[float],
                n_recall_grid: int = 101) -> Dict:
    """Step-function average precision and a fixed recall-grid PR curve.

    ``AP = Σ (R_k − R_{k−1}) · P_k`` (not trapezoidal). ``baseline_precision``
    is the positive prevalence in this cohort — negatives are subsampled, so
    AUPRC is not an operational rate.
    """
    y = _binary_truth(truth)
    s = _finite_scores(scores)
    if y.size != s.size:
        raise ValueError("truth and scores must have the same length")
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    baseline = _ratio(n_pos, n_pos + n_neg)
    if not s.size:
        return {"ap": None, "baseline_precision": baseline, "points": [],
                "n_positive": 0, "n_negative": 0}
    unique = sorted((float(v) for v in np.unique(s)), reverse=True)
    thresholds = [float(unique[0] + 1.0), *unique]
    raw = []
    for cutoff in thresholds:
        m = classification_metrics(y, s, cutoff)
        raw.append({
            "cutoff": cutoff,
            "precision": m["precision"],
            "recall": m["sensitivity"],
        })
    ap = None
    if n_pos:
        ap_value = 0.0
        prev_r = 0.0
        for pt in raw:
            rec = pt["recall"]
            prec = pt["precision"]
            if rec is None or prec is None:
                continue
            if rec > prev_r:
                ap_value += (rec - prev_r) * prec
                prev_r = rec
        ap = float(ap_value)
    grid = []
    recalls = np.linspace(0.0, 1.0, int(n_recall_grid))
    rec_arr = np.asarray([0.0 if p["recall"] is None else p["recall"] for p in raw],
                         dtype=np.float64)
    prec_arr = np.asarray([0.0 if p["precision"] is None else p["precision"] for p in raw],
                          dtype=np.float64)
    for r in recalls:
        # Precision at the first operating point that reaches this recall.
        hit = np.where(rec_arr >= r)[0]
        prec = float(prec_arr[hit[0]]) if hit.size else (
            float(prec_arr[-1]) if prec_arr.size else None)
        grid.append({"recall": float(r), "precision": prec})
    return {"ap": ap, "baseline_precision": baseline, "points": grid,
            "n_positive": n_pos, "n_negative": n_neg}


def select_high_sensitivity_cutoff(truth: Sequence[int | bool],
                                   scores: Sequence[float],
                                   r_min: float = 0.98) -> Dict:
    """Maximise specificity subject to sensitivity ≥ ``r_min`` on validation."""
    y = _binary_truth(truth)
    s = _finite_scores(scores)
    if not s.size:
        raise ValueError("cannot select a cutoff from an empty cohort")
    if not y.any() or y.all():
        raise ValueError("high-sensitivity selection requires both truth classes")
    candidates = [float(s.max() + 1.0), *sorted((float(v) for v in np.unique(s)), reverse=True)]
    evaluated = [classification_metrics(y, s, cutoff) for cutoff in candidates]
    eligible = [m for m in evaluated
                if m["sensitivity"] is not None and m["sensitivity"] >= float(r_min)]
    pool = eligible if eligible else evaluated
    best = max(pool, key=lambda m: (
        m["specificity"] if m["specificity"] is not None else -1.0,
        m["sensitivity"] if m["sensitivity"] is not None else -1.0,
        m["cutoff"],
    ))
    return {
        "cutoff": best["cutoff"],
        "criterion": f"max_specificity_subject_to_sensitivity_ge_{r_min}",
        "r_min": float(r_min),
        "met_constraint": bool(eligible),
        "validation_specificity": best["specificity"],
        "validation_sensitivity": best["sensitivity"],
    }


def select_youden_cutoff(truth: Sequence[int | bool], scores: Sequence[float]) -> Dict:
    """Select a validation cutoff by Youden J with deterministic tie breaks.

    Ties first prefer higher specificity, then the higher cutoff, exactly as
    declared in the exported analysis metadata.
    """
    y = _binary_truth(truth)
    s = _finite_scores(scores)
    if not s.size:
        raise ValueError("cannot select a cutoff from an empty cohort")
    if not y.any() or y.all():
        raise ValueError("Youden selection requires both truth classes")
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    candidates = [float(s.max() + 1.0), *sorted((float(v) for v in np.unique(s)), reverse=True)]
    evaluated = [classification_metrics(y, s, cutoff) for cutoff in candidates]
    # J = TP/P - FP/N. Compare its integer numerator first so mathematically
    # tied cutoffs cannot be misordered by floating-point roundoff. Specificity
    # has a constant denominator N, so TN is its exact tie-break equivalent.
    best = max(evaluated, key=lambda m: (
        m["confusion"]["tp"] * n_neg - m["confusion"]["fp"] * n_pos,
        m["confusion"]["tn"],
        m["cutoff"],
    ))
    return {
        "cutoff": best["cutoff"],
        "criterion": "maximum_youden_j_on_validation",
        "tie_break": "higher_specificity_then_higher_cutoff",
        "validation_youden_j": best["youden_j"],
        "validation_specificity": best["specificity"],
        "validation_sensitivity": best["sensitivity"],
    }


def mann_whitney_analysis(truth: Sequence[int | bool], scores: Sequence[float]) -> Dict:
    """Two-sided Mann-Whitney U test, positive nights/scans versus negative."""
    y = _binary_truth(truth)
    s = _finite_scores(scores)
    if y.size != s.size:
        raise ValueError("truth and scores must have the same length")
    positive, negative = s[y], s[~y]
    out = {
        "alternative": "two-sided",
        "n_positive": int(positive.size),
        "n_negative": int(negative.size),
        "u": None,
        "p_value": None,
        "common_language_auc": None,
        "rank_biserial": None,
    }
    if not positive.size or not negative.size:
        return out
    result = stats.mannwhitneyu(positive, negative, alternative="two-sided", method="auto")
    common = float(result.statistic / (positive.size * negative.size))
    out.update({
        "u": float(result.statistic),
        "p_value": float(result.pvalue),
        "common_language_auc": common,
        "rank_biserial": float(2.0 * common - 1.0),
    })
    return out


def _score_summary(values: Sequence[float]) -> Dict:
    a = _finite_scores(values)
    if not a.size:
        return {"n": 0, "min": None, "p05": None, "q1": None,
                "median": None, "q3": None, "p95": None, "max": None,
                "mean": None}
    p05, q1, median, q3, p95 = np.quantile(a, (0.05, 0.25, 0.5, 0.75, 0.95))
    return {
        "n": int(a.size), "min": float(a.min()), "p05": float(p05),
        "q1": float(q1), "median": float(median), "q3": float(q3),
        "p95": float(p95), "max": float(a.max()), "mean": float(a.mean()),
    }


def _split_analysis(rows: Sequence[Mapping], selected_cutoff: float,
                    include_records: bool = False,
                    include_mann_whitney: bool = True,
                    include_any_cell: bool = True,
                    high_sens_cutoff: float | None = None) -> Dict:
    truth = [int(r["truth"]) for r in rows]
    scores = [float(r["score"]) for r in rows]
    positive = sorted(float(r["score"]) for r in rows if int(r["truth"]) == 1)
    negative = sorted(float(r["score"]) for r in rows if int(r["truth"]) == 0)
    ops = {
        "validation_selected": classification_metrics(truth, scores, selected_cutoff),
    }
    if include_any_cell:
        ops["any_cell"] = classification_metrics(truth, scores, DEFAULT_PRED_MIN_CELLS)
    if high_sens_cutoff is not None:
        ops["high_sensitivity"] = classification_metrics(truth, scores, high_sens_cutoff)
    result = {
        "n": len(rows),
        "n_positive": len(positive),
        "n_negative": len(negative),
        "score_summary": {
            "positive": _score_summary(positive),
            "negative": _score_summary(negative),
        },
        "distributions": {"positive": positive, "negative": negative},
        "roc": roc_analysis(truth, scores),
        "pr": pr_analysis(truth, scores),
        "operating_points": ops,
    }
    if include_mann_whitney:
        result["mann_whitney"] = mann_whitney_analysis(truth, scores)
    if include_records:
        result["records"] = list(rows)
    return result


def _dataset_scan_area(scene: Mapping, max_cells: int) -> int:
    label = scene.get("label")
    if label not in (0, 1, False, True):
        raise ValueError(f"dataset scene {scene.get('ts')} has non-binary label {label!r}")
    area = scene.get("area")
    if area is None:
        if int(label) == 0:
            return 0
        raise ValueError(f"positive dataset scene {scene.get('ts')} has no dbz0 area")
    return _integer_area(area, f"dataset ground-truth area for {scene.get('ts')}", max_cells)


def _count_truth(rows: Iterable[Mapping]) -> Dict[str, int]:
    rows = list(rows)
    pos = sum(int(r["truth"]) for r in rows)
    return {"total": len(rows), "positive": pos, "negative": len(rows) - pos}


def analyze_presence(samples_doc: Mapping, dataset_doc: Mapping,
                     generated: str | None = None,
                     score_field: str = "pred_area",
                     r_min: float = 0.98) -> Dict:
    """Build the complete GPU-free presence-analysis JSON document.

    ``score_field`` is ``pred_area`` (segmentation-area baseline) or ``p_cls``
    (classifier probability). Both scan-level ROC/PR and night aggregation
    use the same field.
    """
    if score_field not in {"pred_area", "p_cls"}:
        raise ValueError(f"score_field must be pred_area or p_cls, got {score_field!r}")
    use_area = score_field == "pred_area"
    sample_rows = list(samples_doc.get("samples") or [])
    dataset_rows = list(dataset_doc.get("scenes") or [])
    model_specs = list(samples_doc.get("models") or [])
    if not sample_rows or not dataset_rows or not model_specs:
        raise ValueError("samples.json and dataset.json must contain samples, scenes, and models")

    grid = dataset_doc.get("grid") or {}
    try:
        grid_h = int(grid["h"])
        grid_w = int(grid["w"])
        pixel_m = float(grid["pixel_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("dataset grid must declare numeric h, w, and pixel_m") from exc
    if grid_h <= 0 or grid_w <= 0:
        raise ValueError(f"dataset grid dimensions must be positive, got {(grid_h, grid_w)}")
    if not np.isfinite(pixel_m) or pixel_m <= 0:
        raise ValueError(f"dataset pixel_m must be finite and positive, got {grid.get('pixel_m')!r}")
    grid_cells = grid_h * grid_w
    pixel_area_km2 = (pixel_m / 1000.0) ** 2

    model_keys = [str(spec.get("key", "")) for spec in model_specs]
    if any(not key for key in model_keys) or len(set(model_keys)) != len(model_keys):
        raise ValueError(f"viewer model keys must be non-empty and unique, got {model_keys}")
    for spec, key in zip(model_specs, model_keys):
        try:
            threshold = float(spec["thr"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"viewer model {key} must declare a numeric pixel threshold") from exc
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"viewer model {key} pixel threshold must be finite in [0, 1], "
                             f"got {spec.get('thr')!r}")

    dataset_by_ts: Dict[int, Mapping] = {}
    manifest_by_night: Dict[str, List[Dict]] = defaultdict(list)
    for raw in dataset_rows:
        ts = int(raw["ts"])
        if ts in dataset_by_ts:
            raise ValueError(f"duplicate dataset timestamp {ts}")
        split = str(raw.get("split", ""))
        if split not in {"train", "val", "test"}:
            raise ValueError(f"dataset scene {ts} split must be train, val, or test, "
                             f"got {raw.get('split')!r}")
        dataset_by_ts[ts] = raw
        night_id = operational_night_id(ts)
        area = _dataset_scan_area(raw, grid_cells)
        manifest_by_night[night_id].append({
            "ts": ts,
            "split": split,
            "truth": int(area >= DEFAULT_GT_MIN_CELLS),
            "gt_area": area,
        })

    sample_by_ts: Dict[int, Mapping] = {}
    evaluation_base: List[Dict] = []
    for raw in sample_rows:
        ts = int(raw["ts"])
        if ts in sample_by_ts:
            raise ValueError(f"duplicate sample timestamp {ts}")
        sample_by_ts[ts] = raw
        if ts not in dataset_by_ts:
            raise ValueError(f"sample timestamp {ts} is absent from dataset.json")
        scene = dataset_by_ts[ts]
        if str(raw["split"]) != str(scene["split"]):
            raise ValueError(f"split mismatch for {ts}: samples={raw['split']} dataset={scene['split']}")
        sample_area = _integer_area(raw.get("gt_area"),
                                    f"sample ground-truth area for {ts}", grid_cells)
        expected_area = _dataset_scan_area(scene, grid_cells)
        if sample_area != expected_area:
            raise ValueError(f"ground-truth area mismatch for {ts}: "
                             f"samples={raw['gt_area']} dataset={expected_area}")
        evaluation_base.append({
            "ts": ts,
            "split": str(raw["split"]),
            "night_id": operational_night_id(ts),
            "truth": int(sample_area >= DEFAULT_GT_MIN_CELLS),
            "gt_area": sample_area,
        })

    expected_eval = {ts for ts, row in dataset_by_ts.items()
                     if str(row["split"]) in {"val", "test"}}
    if set(sample_by_ts) != expected_eval:
        missing = sorted(expected_eval - set(sample_by_ts))
        extra = sorted(set(sample_by_ts) - expected_eval)
        raise ValueError(f"samples.json must exactly cover dataset val/test scenes; "
                         f"missing={missing[:5]} extra={extra[:5]}")

    full_night_truth = {
        night_id: int(any(r["truth"] for r in rows))
        for night_id, rows in manifest_by_night.items()
    }
    # The current curated corpus has homogeneous scan labels within a night.
    mixed_truth_nights = sorted(
        night_id for night_id, rows in manifest_by_night.items()
        if len({r["truth"] for r in rows}) > 1
    )

    eval_groups: Dict[tuple[str, str], List[Dict]] = defaultdict(list)
    for row in evaluation_base:
        eval_groups[(row["split"], row["night_id"])].append(row)
    disagreements = []
    for (split, night_id), rows in eval_groups.items():
        subset_truth = int(any(r["truth"] for r in rows))
        if subset_truth != full_night_truth[night_id]:
            disagreements.append({"split": split, "night_id": night_id,
                                  "subset_truth": subset_truth,
                                  "full_truth": full_night_truth[night_id]})
    if disagreements:
        raise ValueError("evaluated subset does not preserve full-manifest night truth: "
                         f"{disagreements[:5]}")

    split_nights: Dict[str, set[str]] = {sp: set() for sp in ("train", "val", "test")}
    for night_id, rows in manifest_by_night.items():
        for split in {r["split"] for r in rows}:
            split_nights.setdefault(split, set()).add(night_id)
    train_nights = split_nights.get("train", set())

    def exposure(split: str) -> Dict:
        scan_rows = [r for r in evaluation_base if r["split"] == split]
        nights = {r["night_id"] for r in scan_rows}
        exposed = nights & train_nights
        exposed_rows = [r for r in scan_rows if r["night_id"] in train_nights]
        return {
            "nights_total": len(nights),
            "nights_seen_in_train": len(exposed),
            "positive_nights_seen_in_train": sum(full_night_truth[n] for n in exposed),
            "negative_nights_seen_in_train": sum(1 - full_night_truth[n] for n in exposed),
            "scans_total": len(scan_rows),
            "scans_on_nights_seen_in_train": len(exposed_rows),
        }

    def night_coverage(split: str) -> Dict:
        groups = [rows for (row_split, _night_id), rows in eval_groups.items()
                  if row_split == split]
        fractions = [len(rows) / len(manifest_by_night[rows[0]["night_id"]])
                     for rows in groups]
        complete = sum(
            len(rows) == len(manifest_by_night[rows[0]["night_id"]]) for rows in groups
        )
        return {
            "nights_total": len(groups),
            "complete_nights": complete,
            "partial_nights": len(groups) - complete,
            "evaluated_scans": sum(len(rows) for rows in groups),
            "manifest_scans_across_those_nights": sum(
                len(manifest_by_night[rows[0]["night_id"]]) for rows in groups
            ),
            "coverage_fraction_summary": _score_summary(fractions),
        }

    all_night_split_counts = {
        night_id: len({r["split"] for r in rows})
        for night_id, rows in manifest_by_night.items()
    }
    cohort = {
        "manifest_scans": len(dataset_rows),
        "evaluation_scans": len(sample_rows),
        "manifest_scan_truth": _count_truth(
            {"truth": int(_dataset_scan_area(r, grid_cells) >= DEFAULT_GT_MIN_CELLS)}
            for r in dataset_rows
        ),
        "manifest_nights": {
            "total": len(manifest_by_night),
            "positive": sum(full_night_truth.values()),
            "negative": len(full_night_truth) - sum(full_night_truth.values()),
            "mixed_scan_truth": len(mixed_truth_nights),
        },
        "evaluation": {
            split: {
                "scans": _count_truth(r for r in evaluation_base if r["split"] == split),
                "nights": _count_truth(
                    {"truth": full_night_truth[n]}
                    for n in sorted({r["night_id"] for r in evaluation_base if r["split"] == split})
                ),
            } for split in ("val", "test")
        },
        "night_overlap": {
            "nights_in_multiple_splits": sum(v > 1 for v in all_night_split_counts.values()),
            "nights_in_all_three_splits": sum(v == 3 for v in all_night_split_counts.values()),
            "train_validation": len(split_nights.get("train", set()) & split_nights.get("val", set())),
            "train_test": len(split_nights.get("train", set()) & split_nights.get("test", set())),
            "validation_test": len(split_nights.get("val", set()) & split_nights.get("test", set())),
        },
        "training_exposure": {split: exposure(split) for split in ("val", "test")},
        "night_coverage": {split: night_coverage(split) for split in ("val", "test")},
        "subset_full_night_truth_disagreements": 0,
    }

    model_results = []
    selected_name = str(samples_doc.get("selected", ""))
    selected_matches = [str(spec["key"]) for spec in model_specs
                        if str(spec.get("name", "")) == selected_name]
    if len(selected_matches) != 1:
        raise ValueError("samples.json selected experiment must match exactly one viewer model; "
                         f"selected={selected_name!r} matches={selected_matches}")
    default_model_key = selected_matches[0]
    for spec in model_specs:
        key = str(spec["key"])
        pixel_threshold = float(spec["thr"])
        scene_rows: List[Dict] = []
        for base in evaluation_base:
            raw = sample_by_ts[base["ts"]]
            values = (raw.get("models") or {}).get(key) or {}
            if use_area:
                if values.get("pred_area") is None:
                    raise ValueError(f"missing pred_area for {key}/{base['ts']}")
                score = float(_integer_area(values["pred_area"],
                                           f"predicted area for {key}/{base['ts']}",
                                           grid_cells))
            else:
                raw_p = values.get("p_cls", raw.get("p_cls"))
                if raw_p is None:
                    raise ValueError(f"missing p_cls for {key}/{base['ts']}")
                try:
                    score = float(raw_p)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"p_cls for {key}/{base['ts']} must be numeric") from exc
                if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                    raise ValueError(f"p_cls for {key}/{base['ts']} must be in [0, 1], got {raw_p!r}")
            scene_rows.append({**base, "score": score})

        val_scene = [r for r in scene_rows if r["split"] == "val"]
        test_scene = [r for r in scene_rows if r["split"] == "test"]
        scan_selection = select_youden_cutoff(
            [r["truth"] for r in val_scene], [r["score"] for r in val_scene])
        scan_cutoff = float(scan_selection["cutoff"])
        scan_high = select_high_sensitivity_cutoff(
            [r["truth"] for r in val_scene], [r["score"] for r in val_scene],
            r_min=r_min)
        scan_high_cutoff = float(scan_high["cutoff"])

        night_results = {}
        for aggregation in ("max", "mean"):
            night_rows_by_split: Dict[str, List[Dict]] = {"val": [], "test": []}
            grouped: Dict[tuple[str, str], List[Dict]] = defaultdict(list)
            for row in scene_rows:
                grouped[(row["split"], row["night_id"])].append(row)
            for (split, night_id), rows in sorted(grouped.items()):
                scores = [r["score"] for r in rows]
                score = max(scores) if aggregation == "max" else float(np.mean(scores))
                manifest_rows = manifest_by_night[night_id]
                record = {
                    "night_id": night_id,
                    "split": split,
                    "truth": full_night_truth[night_id],
                    "score": float(score),
                    "evaluated_scan_count": len(rows),
                    "manifest_scan_count": len(manifest_rows),
                    "manifest_present_scan_count": sum(r["truth"] for r in manifest_rows),
                    "coverage_fraction": float(len(rows) / len(manifest_rows)),
                    "seen_in_train": night_id in train_nights,
                }
                night_rows_by_split[split].append(record)

            val_night = night_rows_by_split["val"]
            test_night = night_rows_by_split["test"]
            selection = select_youden_cutoff(
                [r["truth"] for r in val_night], [r["score"] for r in val_night])
            cutoff = float(selection["cutoff"])
            night_high = select_high_sensitivity_cutoff(
                [r["truth"] for r in val_night], [r["score"] for r in val_night],
                r_min=r_min)
            if use_area:
                score_def = ("maximum predicted SBW-cell count among evaluated scans"
                             if aggregation == "max" else
                             "mean predicted SBW-cell count across evaluated scans")
                cutoff_block = {**selection, "cells": cutoff,
                                "km2": float(cutoff * pixel_area_km2)}
            else:
                score_def = (f"{aggregation} classifier probability among evaluated scans")
                cutoff_block = {**selection, "probability": cutoff}
            night_results[aggregation] = {
                "score_definition": score_def,
                "selected_cutoff": cutoff_block,
                "high_sensitivity_cutoff": night_high,
                "splits": {
                    "validation": _split_analysis(
                        val_night, cutoff, include_records=True,
                        include_any_cell=use_area,
                        high_sens_cutoff=float(night_high["cutoff"])),
                    "test": _split_analysis(
                        test_night, cutoff, include_records=True,
                        include_any_cell=use_area,
                        high_sens_cutoff=float(night_high["cutoff"])),
                },
            }

        model_results.append({
            "key": key,
            "name": str(spec.get("name", key)),
            "display_name": str(spec.get("disp", spec.get("name", key))),
            "selected": key == default_model_key,
            "pixel_probability_threshold": pixel_threshold,
            "scan": {
                "score_definition": (
                    "full-resolution count of pixels with probability above the locked pixel threshold"
                    if use_area else
                    "scan-level classifier probability p_cls"
                ),
                "score_field": score_field,
                "selected_cutoff": (
                    {**scan_selection, "cells": scan_cutoff,
                     "km2": float(scan_cutoff * pixel_area_km2)}
                    if use_area else
                    {**scan_selection, "probability": scan_cutoff}
                ),
                "high_sensitivity_cutoff": scan_high,
                "splits": {
                    "validation": _split_analysis(
                        val_scene, scan_cutoff, include_mann_whitney=False,
                        include_any_cell=use_area,
                        high_sens_cutoff=scan_high_cutoff),
                    "test": _split_analysis(
                        test_scene, scan_cutoff, include_mann_whitney=False,
                        include_any_cell=use_area,
                        high_sens_cutoff=scan_high_cutoff),
                },
            },
            "night": night_results,
        })

    generated = generated or datetime.now().isoformat(timespec="seconds")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated": generated,
        "selected_model_key": default_model_key,
        "defaults": {
            "model_key": default_model_key,
            "split": "test",
            "night_aggregation": "max",
            "scan_operating_point": "any_cell",
            "night_operating_point": "validation_selected",
        },
        "definitions": {
            "ground_truth_scan_presence": "gt_area >= 1 full-resolution cell",
            "ground_truth_min_cells": DEFAULT_GT_MIN_CELLS,
            "score_field": score_field,
            "prediction_score": (
                "pred_area at each model's locked pixel probability threshold"
                if use_area else
                "scan-level classifier probability p_cls"
            ),
            "prediction_any_cell_cutoff": DEFAULT_PRED_MIN_CELLS,
            "area_rule": "score >= cutoff",
            "threshold_selection": "maximize validation Youden J; ties prefer higher specificity, then higher cutoff",
            "night_boundary": "UTC noon-to-noon",
            "night_boundary_utc_hour": NIGHT_BOUNDARY_UTC_HOUR,
            "night_truth": "presence in any full-manifest scan assigned to the operational night",
            "night_scores": ["max", "mean"],
            "pixel_size_m": pixel_m,
            "pixel_area_km2": pixel_area_km2,
        },
        "cohort": cohort,
        "caveats": [
            "The manifest is split by scan, so many validation and test nights also occur in training; night-level results are exploratory rather than night-independent generalization estimates.",
            f"Validation and test share {cohort['night_overlap']['validation_test']} operational nights; validation-selected area cutoffs are therefore applied to different scan fragments of many of the same nights in test.",
            "Night max and mean use only the scans assigned to that evaluation split; per-night evaluated and manifest scan counts expose partial-night coverage, and maximum scores are especially sensitive to unequal evaluated scan counts per night.",
            "The training/evaluation manifest subsamples negative scans, so prevalence-dependent accuracy and precision describe this curated cohort rather than operational prevalence.",
            "Negative-scene ground-truth masks are synthesized as all-zero arrays by dataset construction; they are not independent pixel-level annotations.",
            "Predicted area depends on each model's locked segmentation probability threshold; the validation-selected area cutoff is a second, separate threshold.",
            "The current manifest has homogeneous scan truth within each operational night; future mixed nights must continue to derive night truth from the full manifest.",
        ],
        "models": model_results,
    }
