"""Operational-night identity and leakage-free whole-night splitting.

The manifest is a time series: the radar samples the same moth exodus roughly
every 30 minutes, so consecutive scans of one night are strongly correlated. A
scan-level split puts those correlated scans in train *and* test, which inflates
the measured score (Kattenborn et al. 2022, ISPRS Open J. Photogramm. Remote
Sens.: >90% of reviewed remote-sensing CNN studies did not ensure independence
between training and validation data). The standard remedy is grouped/blocked
splitting: pick the correlation unit and assign whole groups, never individual
samples (cf. scikit-learn ``StratifiedGroupKFold``).

Here the correlation unit is the *operational night*. Night identity comes from
``presence.operational_night_id`` (UTC noon-to-noon), which is already used by
the dashboard's presence analysis and is tested for the noon boundary and for
month/year/leap-day rollovers. Deriving it from the timestamp -- rather than the
manifest's ``night`` column -- is what makes it usable here at all: that column
is populated only for positives and is empty for every negative scan.

Two properties of the current corpus shape the assignment algorithm:

* No night mixes positive and negative scans, so each night is wholly a
  *migration* night or a *quiet* night.
* Migration nights carry ~13.5 scans and quiet nights ~3.6, so assigning nights
  in a 70/20/10 ratio would NOT produce a 70/20/10 ratio of scans. The greedy
  pass below therefore targets the scan deficit, not the night count.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .presence import operational_night_id

SPLITS = ("train", "val", "test")


def night_id(timestamp: int | str) -> str:
    """UTC operational-night ID for a 12-digit timestamp.

    Delegates to the presence module so the training split and the dashboard's
    night-level analysis can never disagree about which night a scan belongs to.
    """
    return operational_night_id(timestamp)


def add_night_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``night`` column derived from timestamps.

    Any pre-existing ``night`` value (the mask-file variable name, present for
    positives only) is preserved as ``mask_night`` because ``cache_targets``
    uses it to index the netCDF mask.
    """
    out = df.copy()
    if "night" in out.columns and "mask_night" not in out.columns:
        out["mask_night"] = out["night"].fillna("").astype(str)
    out["night"] = out["timestamp"].map(night_id)
    return out


def night_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per operational night: scan count, positives, year, class.

    ``kind`` is ``migration`` if the night contains any positive scan, else
    ``quiet`` -- matching how ``presence.analyze_presence`` derives night truth
    from the full manifest.
    """
    if "night" not in df.columns:
        df = add_night_ids(df)
    grouped = df.groupby("night", sort=True)
    table = grouped.agg(
        scans=("timestamp", "size"),
        positives=("label", "sum"),
        year=("year", "min"),
    )
    table["negatives"] = table["scans"] - table["positives"]
    table["kind"] = np.where(table["positives"] > 0, "migration", "quiet")
    return table.reset_index()


def assign_night_split(df: pd.DataFrame, fractions: Dict[str, float],
                       seed: int) -> pd.Series:
    """Assign whole nights to train/val/test, stratified by year x night kind.

    Within each (year, migration|quiet) stratum, nights are taken largest-first
    and given to whichever split is furthest below its target *scan* count. This
    is the same greedy rule ``build_year_split`` uses for whole years, lifted to
    nights and driven by scans rather than night count -- necessary because
    migration nights hold ~3.7x the scans of quiet nights.

    Deficits are tracked PER STRATUM, not globally. With a single global running
    total the greedy pass sends every early year to train (whose 70% target is
    the last to be satisfied) and val/test end up drawn only from the final
    years. Per-stratum targets make each year x kind split ~70/20/10 on its own,
    so every year and both night kinds are represented in every split.

    Returns a per-row split label aligned to ``df``'s index. Every scan of a
    night lands in exactly one split, so no night is ever shared.
    """
    for split in SPLITS:
        if split not in fractions:
            raise ValueError(f"fractions must define {SPLITS}, missing {split!r}")
    total_fraction = sum(float(fractions[s]) for s in SPLITS)
    if not np.isclose(total_fraction, 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {total_fraction}")

    if "night" not in df.columns:
        df = add_night_ids(df)
    nights = night_table(df)
    assignment: Dict[str, str] = {}

    rng = np.random.default_rng(seed)
    # Seeded tie-break key. The greedy deficit rule below is deterministic on its
    # own (exact float ties never occur), so without this the seed would be a
    # no-op. Many nights DO share a scan count, and this is what lets a different
    # seed produce a genuinely different -- but still valid -- split, which is
    # what makes split-sensitivity checkable.
    nights = nights.assign(_key=rng.random(len(nights)))

    # Deterministic stratum order; largest strata first keeps the greedy pass
    # from spending its whole budget on a long tail of tiny strata.
    strata = sorted(
        nights.groupby(["year", "kind"]).groups.keys(),
        key=lambda k: (int(k[0]), str(k[1])),
    )
    for year, kind in strata:
        block = nights[(nights["year"] == year) & (nights["kind"] == kind)]
        stratum_scans = float(block["scans"].sum())
        targets = {s: float(fractions[s]) * stratum_scans for s in SPLITS}
        running = {s: 0.0 for s in SPLITS}
        # Largest-first; equal-sized nights ordered by the seeded key.
        ordered = block.sort_values(["scans", "_key"], ascending=[False, True])
        for row in ordered.itertuples():
            deficits = {s: targets[s] - running[s] for s in SPLITS}
            best = max(deficits, key=lambda s: deficits[s])
            assignment[row.night] = best
            running[best] += float(row.scans)

    return df["night"].map(assignment)


def night_split_report(df: pd.DataFrame) -> Dict:
    """Per-split night/scan counts plus the zero-overlap verification.

    ``overlap_nights`` is the whole point: it must be 0 for a leakage-free
    split, and is reported rather than assumed.
    """
    if "night" not in df.columns:
        df = add_night_ids(df)
    nights = night_table(df)
    night_split = df.groupby("night")["split"].agg(lambda s: sorted(set(s)))
    overlap = sorted(n for n, splits in night_split.items() if len(splits) > 1)

    kind_by_night = dict(zip(nights["night"], nights["kind"]))
    per_split: Dict[str, Dict] = {}
    for split in SPLITS:
        rows = df[df["split"] == split]
        split_nights = sorted(set(rows["night"]))
        per_split[split] = {
            "nights": len(split_nights),
            "migration_nights": sum(kind_by_night[n] == "migration" for n in split_nights),
            "quiet_nights": sum(kind_by_night[n] == "quiet" for n in split_nights),
            "scans": int(len(rows)),
            "positives": int((rows["label"] == 1).sum()),
            "negatives": int((rows["label"] == 0).sum()),
            "scan_fraction": (round(len(rows) / len(df), 4) if len(df) else None),
            "nights_per_year": {
                int(y): int(g["night"].nunique())
                for y, g in rows.groupby("year")
            },
        }

    return {
        "unit": "operational_night",
        "night_boundary": "UTC noon-to-noon",
        "total_nights": int(len(nights)),
        "total_scans": int(len(df)),
        "migration_nights": int((nights["kind"] == "migration").sum()),
        "quiet_nights": int((nights["kind"] == "quiet").sum()),
        "overlap_nights": len(overlap),
        "overlap_night_ids": overlap[:10],
        "per_split": per_split,
    }


def verify_no_leakage(df: pd.DataFrame) -> None:
    """Raise if any operational night appears in more than one split."""
    report = night_split_report(df)
    if report["overlap_nights"]:
        raise ValueError(
            f"night split leaked: {report['overlap_nights']} night(s) span multiple "
            f"splits, e.g. {report['overlap_night_ids']}"
        )


def temporal_neighbours(df: pd.DataFrame, radius: int = 1,
                        max_gap_minutes: float = 35.0) -> Dict[int, List[int | None]]:
    """Chronological in-night neighbour timestamps for every scan.

    Returns ``{timestamp: [t-radius, ..., t, ..., t+radius]}`` where each slot is
    a timestamp or ``None`` when no neighbour exists within ``max_gap_minutes``
    **in the same operational night**. Because whole nights live in one split,
    same-night is automatically same-split -- but the split is checked anyway so
    the guarantee does not silently depend on that.

    ``None`` slots are the padding the temporal model masks out. They are common:
    on the current corpus only ~54% of scans have both +/-1 neighbours and ~17%
    have none at all.
    """
    if "night" not in df.columns:
        df = add_night_ids(df)
    stamps = pd.to_datetime(df["timestamp"].astype(str), format="%Y%m%d%H%M")
    work = df.assign(_dt=stamps).sort_values("_dt")

    out: Dict[int, List[int | None]] = {}
    has_split = "split" in work.columns
    for _night, group in work.groupby("night", sort=False):
        times = group["_dt"].tolist()
        tss = [int(t) for t in group["timestamp"].tolist()]
        splits = group["split"].tolist() if has_split else [None] * len(tss)
        for i, ts in enumerate(tss):
            window: List[int | None] = []
            for offset in range(-radius, radius + 1):
                j = i + offset
                if offset == 0:
                    window.append(ts)
                elif 0 <= j < len(tss) and splits[j] == splits[i] and \
                        abs((times[j] - times[i]).total_seconds()) / 60.0 <= max_gap_minutes:
                    window.append(tss[j])
                else:
                    window.append(None)
            out[ts] = window
    return out
