"""Statistical tests for the HCAD-RAG benchmark (reviewer R2 / meta-review).

All functions are pure (lists in, dicts out) and deterministic given a
seed, so every number in a results file can be recomputed from the
per-question records in the checkpoint.

Contents
--------
- ``bootstrap_ci``          percentile bootstrap CI of a mean
- ``paired_bootstrap_ci``   percentile bootstrap CI of a mean paired difference
- ``wilcoxon_signed_rank``  two-sided Wilcoxon signed-rank test (Pratt zeros)
- ``paired_t_test``         two-sided paired t-test
- ``mcnemar_exact``         exact two-sided McNemar (binomial on discordant pairs)
- ``cohen_kappa``           Cohen's kappa for two raters (nominal labels)
- ``fleiss_kappa``          Fleiss' kappa for >= 2 raters (nominal labels)
- ``kappa_interpretation``  Landis & Koch (1977) verbal band
"""
from __future__ import annotations

import math
from collections import Counter
from math import comb
from typing import Hashable, Sequence

import numpy as np

from src.config import BOOTSTRAP_RESAMPLES, CONFIDENCE_LEVEL, RANDOM_SEED


def _finite(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError("statistics require finite values")
    return arr


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = RANDOM_SEED,
) -> dict:
    """Percentile bootstrap confidence interval for the mean."""
    arr = _finite(values)
    n = int(arr.size)
    if n == 0:
        return {"n": 0, "mean": None, "ci_low": None, "ci_high": None,
                "confidence": confidence, "n_resamples": n_resamples}
    mean = float(arr.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "ci_low": mean, "ci_high": mean,
                "confidence": confidence, "n_resamples": n_resamples}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return {"n": n, "mean": mean, "ci_low": float(lo), "ci_high": float(hi),
            "std": float(arr.std(ddof=1)),
            "confidence": confidence, "n_resamples": n_resamples}


def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = RANDOM_SEED,
) -> dict:
    """Bootstrap CI of mean(a - b) over paired observations (questions)."""
    a_arr, b_arr = _finite(a), _finite(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired bootstrap requires equal-length samples")
    out = bootstrap_ci(a_arr - b_arr, n_resamples, confidence, seed)
    out["mean_difference"] = out.pop("mean")
    if out["ci_low"] is not None:
        out["excludes_zero"] = bool(out["ci_low"] > 0 or out["ci_high"] < 0)
    else:
        out["excludes_zero"] = False
    return out


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sided Wilcoxon signed-rank test on paired samples.

    Uses scipy's implementation with the Pratt zero-handling and the exact
    null distribution for small, tie-free samples (normal approximation
    otherwise). All-zero differences return p = 1.0.
    """
    a_arr, b_arr = _finite(a), _finite(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError("Wilcoxon requires equal-length paired samples")
    diff = a_arr - b_arr
    n_nonzero = int(np.count_nonzero(diff))
    base = {"n": int(diff.size), "n_nonzero": n_nonzero,
            "median_difference": float(np.median(diff)) if diff.size else None}
    if n_nonzero == 0:
        return {**base, "statistic": 0.0, "p_value": 1.0, "significant_at_05": False}
    from scipy.stats import wilcoxon

    res = wilcoxon(a_arr, b_arr, zero_method="pratt", alternative="two-sided")
    p = float(res.pvalue)
    return {**base, "statistic": float(res.statistic), "p_value": p,
            "significant_at_05": p < 0.05}


def paired_t_test(a: Sequence[float], b: Sequence[float]) -> dict:
    """Two-sided paired t-test (reported alongside Wilcoxon)."""
    a_arr, b_arr = _finite(a), _finite(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired t-test requires equal-length samples")
    n = int(a_arr.size)
    diff = a_arr - b_arr
    if n < 2:
        return {"n": n, "t": None, "df": None, "p_value": None,
                "significant_at_05": False}
    sd = float(diff.std(ddof=1))
    if sd == 0.0:
        p = 1.0 if float(diff.mean()) == 0.0 else 0.0
        return {"n": n, "t": None, "df": n - 1, "p_value": p,
                "mean_difference": float(diff.mean()),
                "significant_at_05": p < 0.05}
    from scipy.stats import ttest_rel

    res = ttest_rel(a_arr, b_arr)
    p = float(res.pvalue)
    return {"n": n, "t": float(res.statistic), "df": n - 1, "p_value": p,
            "mean_difference": float(diff.mean()),
            # Cohen's d_z for paired designs
            "effect_size_dz": float(diff.mean() / sd),
            "significant_at_05": p < 0.05}


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test.

    n01 = a correct & b wrong, n10 = a wrong & b correct. Under H0 the
    discordant pairs are Binomial(n01 + n10, 0.5); the two-sided p-value
    is 2 * P(X <= min(n01, n10)), capped at 1.
    """
    if len(a_correct) != len(b_correct) or not len(a_correct):
        raise ValueError("McNemar requires paired, non-empty outcomes")
    n01 = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    n10 = sum(1 for x, y in zip(a_correct, b_correct) if not x and y)
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "n_discordant": 0, "p_value": 1.0,
                "significant_at_05": False}
    tail = sum(comb(n, i) for i in range(min(n01, n10) + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
    return {"n01": n01, "n10": n10, "n_discordant": n, "p_value": p,
            "significant_at_05": p < 0.05}


def cohen_kappa(r1: Sequence[Hashable], r2: Sequence[Hashable]) -> dict:
    """Cohen's kappa for two raters over nominal labels.

    kappa = (p_o - p_e) / (1 - p_e). When both raters use a single
    identical label throughout (p_e = 1) agreement is perfect but kappa is
    undefined; it is reported as None with ``degenerate=True``.
    """
    if len(r1) != len(r2):
        raise ValueError("Cohen's kappa requires equal-length ratings")
    n = len(r1)
    if n == 0:
        return {"n": 0, "kappa": None, "observed_agreement": None,
                "expected_agreement": None, "degenerate": True}
    p_o = sum(1 for x, y in zip(r1, r2) if x == y) / n
    c1, c2 = Counter(r1), Counter(r2)
    p_e = sum((c1[k] / n) * (c2[k] / n) for k in set(c1) | set(c2))
    if math.isclose(p_e, 1.0):
        return {"n": n, "kappa": None, "observed_agreement": p_o,
                "expected_agreement": p_e, "degenerate": True}
    kappa = (p_o - p_e) / (1.0 - p_e)
    return {"n": n, "kappa": kappa, "observed_agreement": p_o,
            "expected_agreement": p_e, "degenerate": False,
            "interpretation": kappa_interpretation(kappa)}


def fleiss_kappa(ratings: Sequence[Sequence[Hashable]]) -> dict:
    """Fleiss' kappa. ``ratings[i]`` holds every rater's label for item i.

    All items must have the same number of raters (>= 2).
    """
    items = [list(r) for r in ratings]
    if not items:
        return {"n_items": 0, "n_raters": 0, "kappa": None, "degenerate": True}
    m = len(items[0])
    if m < 2 or any(len(r) != m for r in items):
        raise ValueError("Fleiss' kappa needs >= 2 raters and a constant rater count")
    categories = sorted({lab for r in items for lab in r}, key=str)
    n = len(items)
    counts = np.array([[Counter(r)[c] for c in categories] for r in items], dtype=float)
    p_j = counts.sum(axis=0) / (n * m)
    p_i = ((counts ** 2).sum(axis=1) - m) / (m * (m - 1))
    p_bar = float(p_i.mean())
    p_e = float((p_j ** 2).sum())
    if math.isclose(p_e, 1.0):
        return {"n_items": n, "n_raters": m, "kappa": None,
                "observed_agreement": p_bar, "expected_agreement": p_e,
                "degenerate": True}
    kappa = (p_bar - p_e) / (1.0 - p_e)
    return {"n_items": n, "n_raters": m, "kappa": kappa,
            "observed_agreement": p_bar, "expected_agreement": p_e,
            "degenerate": False, "interpretation": kappa_interpretation(kappa)}


def kappa_interpretation(kappa: float | None) -> str:
    """Landis & Koch (1977) agreement band."""
    if kappa is None:
        return "undefined"
    if kappa < 0:
        return "poor"
    for upper, label in ((0.20, "slight"), (0.40, "fair"), (0.60, "moderate"),
                         (0.80, "substantial")):
        if kappa <= upper:
            return label
    return "almost perfect"


def mean_std(values: Sequence[float]) -> dict:
    """mean +- sample std (ddof=1) summary used in latency tables."""
    arr = _finite(values)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None}
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            "median": float(np.median(arr))}
