"""Night-clustered bootstrap and paired tests for publication metrics.

Resample *nights* with replacement and take every scan belonging to a drawn
night. Percentile intervals only. HD95 is not bootstrapped (undefined on empty
predictions; report median + IQR instead).
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from scipy import stats


def _night_index(clusters: Sequence) -> Dict:
    groups: Dict = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    nights = list(groups.keys())
    return {"nights": nights, "groups": groups}


def cluster_bootstrap(values: Sequence[float], clusters: Sequence,
                      stat: Callable[[np.ndarray], float] = np.nanmean,
                      n_boot: int = 2000, alpha: float = 0.05,
                      seed: int = 0) -> Dict[str, float]:
    """Percentile CI for ``stat(values)`` resampling nights with replacement."""
    v = np.asarray(values, dtype=np.float64)
    idx = _night_index(clusters)
    nights = idx["nights"]
    rng = np.random.default_rng(seed)
    point = float(stat(v))
    if not nights:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_boot": 0, "n_nights": 0}
    draws = np.empty(n_boot, dtype=np.float64)
    n = len(nights)
    for b in range(n_boot):
        chosen = rng.integers(0, n, size=n)
        take = [j for k in chosen for j in idx["groups"][nights[k]]]
        draws[b] = float(stat(v[take]))
    lo, hi = np.nanpercentile(draws, (100 * alpha / 2, 100 * (1 - alpha / 2)))
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "n_boot": int(n_boot), "n_nights": int(n)}


def cluster_bootstrap_many(records: Sequence[Mapping], clusters: Sequence,
                           keys: Sequence[str],
                           stat: Callable[[np.ndarray], float] = np.nanmean,
                           n_boot: int = 2000, alpha: float = 0.05,
                           seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Shared night draws across metric keys so intervals are comparable."""
    idx = _night_index(clusters)
    nights = idx["nights"]
    rng = np.random.default_rng(seed)
    arrays = {k: np.asarray([r.get(k, np.nan) for r in records], dtype=np.float64)
              for k in keys}
    out = {k: {"point": float(stat(arrays[k])), "lo": float("nan"),
               "hi": float("nan"), "n_boot": 0, "n_nights": len(nights)}
           for k in keys}
    if not nights:
        return out
    n = len(nights)
    draws = {k: np.empty(n_boot, dtype=np.float64) for k in keys}
    for b in range(n_boot):
        chosen = rng.integers(0, n, size=n)
        take = [j for k in chosen for j in idx["groups"][nights[k]]]
        for key in keys:
            draws[key][b] = float(stat(arrays[key][take]))
    lo_p, hi_p = 100 * alpha / 2, 100 * (1 - alpha / 2)
    for key in keys:
        lo, hi = np.nanpercentile(draws[key], (lo_p, hi_p))
        out[key]["lo"] = float(lo)
        out[key]["hi"] = float(hi)
        out[key]["n_boot"] = int(n_boot)
    return out


def cluster_bootstrap_curve(curves: Sequence[Sequence[float]], clusters: Sequence,
                            n_boot: int = 2000, alpha: float = 0.05,
                            seed: int = 0) -> Dict:
    """Night-clustered band for a vector-valued statistic (NSD vs distance)."""
    arr = np.asarray(curves, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("curves must be (n_scenes, n_taus)")
    idx = _night_index(clusters)
    nights = idx["nights"]
    point = np.nanmean(arr, axis=0)
    out = {"point": point.tolist(), "lo": [float("nan")] * arr.shape[1],
           "hi": [float("nan")] * arr.shape[1], "n_boot": 0,
           "n_nights": len(nights)}
    if not nights:
        return out
    rng = np.random.default_rng(seed)
    n = len(nights)
    draws = np.empty((n_boot, arr.shape[1]), dtype=np.float64)
    for b in range(n_boot):
        chosen = rng.integers(0, n, size=n)
        take = [j for k in chosen for j in idx["groups"][nights[k]]]
        draws[b] = np.nanmean(arr[take], axis=0)
    lo, hi = np.nanpercentile(draws, (100 * alpha / 2, 100 * (1 - alpha / 2)), axis=0)
    out["lo"] = [float(x) for x in lo]
    out["hi"] = [float(x) for x in hi]
    out["n_boot"] = int(n_boot)
    return out


def paired_cluster_bootstrap(a: Sequence[float], b: Sequence[float],
                             clusters: Sequence,
                             stat: Callable[[np.ndarray], float] = np.nanmean,
                             n_boot: int = 2000, alpha: float = 0.05,
                             seed: int = 0) -> Dict[str, float]:
    """CI for ``stat(a) - stat(b)`` with shared night draws."""
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    if av.shape != bv.shape:
        raise ValueError("paired series must have the same length")
    idx = _night_index(clusters)
    nights = idx["nights"]
    point = float(stat(av) - stat(bv))
    if not nights:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_boot": 0, "n_nights": 0}
    rng = np.random.default_rng(seed)
    n = len(nights)
    draws = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        chosen = rng.integers(0, n, size=n)
        take = [j for k in chosen for j in idx["groups"][nights[k]]]
        draws[i] = float(stat(av[take]) - stat(bv[take]))
    lo, hi = np.nanpercentile(draws, (100 * alpha / 2, 100 * (1 - alpha / 2)))
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "n_boot": int(n_boot), "n_nights": int(n)}


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Two-sided Wilcoxon signed-rank on paired scan-level values.

    Anticconservative if scans within a night are correlated; the night-
    clustered bootstrap is the primary interval.
    """
    av = np.asarray(a, dtype=np.float64)
    bv = np.asarray(b, dtype=np.float64)
    d = av - bv
    finite = np.isfinite(d)
    d = d[finite]
    out = {"n": int(d.size), "statistic": None, "p_value": None}
    if d.size < 1 or np.allclose(d, 0.0):
        return out
    try:
        res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided",
                             correction=False)
    except ValueError:
        return out
    out["statistic"] = float(res.statistic)
    out["p_value"] = float(res.pvalue)
    return out


def median_iqr(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if not v.size:
        return {"median": float("nan"), "q1": float("nan"), "q3": float("nan")}
    q1, med, q3 = np.nanpercentile(v, (25, 50, 75))
    return {"median": float(med), "q1": float(q1), "q3": float(q3)}
