"""Robustness metrics beyond the base-paper protocol.

The base paper  reports the 5-dimension scoring, the
8-criteria selection win-rate, and retrieval latency - with no significance
testing and no retrieval-quality measurement. These four metrics make the
evaluation strictly more rigorous while remaining directly comparable.
"""
from __future__ import annotations

import math
from math import comb

import numpy as np

def ndcg_at_k(ranked_relevance: list[int], k: int = 8) -> float:
    """NDCG@K over binary relevance judgments of the ranked retrieval list."""
    if not ranked_relevance:
        return 0.0
    rel = np.asarray(ranked_relevance[:k], dtype=np.float64)
    dcg = float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))
    ideal = np.sort(np.asarray(ranked_relevance, dtype=np.float64))[::-1][:k]
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(ranked_relevance: list[int], k: int = 8) -> float:
    total_relevant = sum(ranked_relevance)
    if total_relevant == 0:
        return 0.0
    return sum(ranked_relevance[:k]) / total_relevant

def mcnemar_test(b_correct: list[bool], a_correct: list[bool]) -> dict:
    """Exact McNemar test on paired binary outcomes (same questions).

    b = system B (e.g., HCAD), a = baseline. Returns the exact two-sided
    p-value - the significance test the base paper never runs.
    """
    if len(b_correct) != len(a_correct) or not b_correct:
        raise ValueError("McNemar requires paired, non-empty outcomes")
    n01 = sum(1 for b, a in zip(b_correct, a_correct) if b and not a)
    n10 = sum(1 for b, a in zip(b_correct, a_correct) if not b and a)
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0, "significant_at_05": False}
    # exact two-sided binomial test on the discordant pairs
    p = sum(comb(n, i) for i in range(min(n01, n10) + 1)) / (2 ** (n - 1))
    p = min(1.0, 2 * p * 0.5)
    return {"n01": n01, "n10": n10, "p_value": round(p, 5),
            "significant_at_05": p < 0.05}

def citation_precision(answer: str, evidence_texts: list[str]) -> dict:
    """Citation precision: cited tags that point at real retrieved items.

    Directly measures the paper's stated goal (hallucination mitigation):
    what fraction of [Doc i]/[Evidence i] citations in the answer resolve to
    retrieved evidence items?
    """
    import re
    tags = re.findall(r"\[(?:Doc|Evidence|Context)\s*(\d+)\]", answer or "")
    if not tags:
        return {"n_citations": 0, "n_valid": 0, "precision": None}
    valid = sum(1 for t in tags if 1 <= int(t) <= len(evidence_texts))
    n = len(tags)
    return {"n_citations": n, "n_valid": valid,
            "precision": round(valid / n, 4) if n else None}


def cohens_kappa(rater1: list[str], rater2: list[str], categories: Optional[list[str]] = None) -> float:
    """Compute Cohen's Kappa for inter-annotator agreement between two judges."""
    if len(rater1) != len(rater2) or not rater1:
        return 0.0

    cats = categories or sorted(list(set(rater1) | set(rater2)))
    if len(cats) <= 1:
        return 1.0

    n = len(rater1)
    cat_to_idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    cm = np.zeros((k, k), dtype=np.float64)

    for r1, r2 in zip(rater1, rater2):
        if r1 in cat_to_idx and r2 in cat_to_idx:
            cm[cat_to_idx[r1], cat_to_idx[r2]] += 1.0

    po = np.trace(cm) / n
    pe = sum(np.sum(cm[i, :]) * np.sum(cm[:, i]) for i in range(k)) / (n * n)

    if pe >= 1.0:
        return 1.0
    kappa = (po - pe) / (1.0 - pe)
    return float(round(kappa, 4))


def fleiss_kappa(matrix: np.ndarray) -> float:
    """Compute Fleiss' Kappa for inter-annotator agreement across N subjects and k categories.

    Args:
        matrix: 2D numpy array of shape (N, k) where matrix[i, j] is the count of raters
                who assigned subject i to category j. Each row must sum to the number of raters m.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0:
        return 0.0

    n_subjects, n_categories = matrix.shape
    if n_subjects == 0 or n_categories <= 1:
        return 1.0

    # m = number of raters per subject
    m = np.sum(matrix[0, :])
    if m <= 1:
        return 1.0

    # Proportion of all assignments to each category
    p_j = np.sum(matrix, axis=0) / (n_subjects * m)

    # Extent of agreement for each subject i
    # P_i = (sum_j(n_ij^2) - m) / (m * (m - 1))
    p_i = (np.sum(matrix ** 2, axis=1) - m) / (m * (m - 1.0))
    p_bar = np.mean(p_i)
    p_bar_e = np.sum(p_j ** 2)

    if p_bar_e >= 1.0:
        return 1.0

    kappa = (p_bar - p_bar_e) / (1.0 - p_bar_e)
    return float(round(kappa, 4))


def bootstrap_ci(
    data: list[float],
    num_resamples: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float]:
    """Compute non-parametric 95% bootstrap confidence interval of the mean."""
    if not data:
        return 0.0, 0.0
    if len(data) == 1:
        return float(data[0]), float(data[0])

    arr = np.asarray(data, dtype=np.float64)
    rng = np.random.RandomState(random_seed)
    n = len(arr)
    boot_means = np.empty(num_resamples, dtype=np.float64)

    for i in range(num_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    low = float(np.percentile(boot_means, lower_pct))
    high = float(np.percentile(boot_means, upper_pct))
    return round(low, 2), round(high, 2)


def wilcoxon_signed_rank_test(x: list[float], y: list[float]) -> dict:
    """Non-parametric paired Wilcoxon signed-rank test for paired continuous scores."""
    if len(x) != len(y) or not x:
        return {"statistic": 0.0, "p_value": 1.0, "significant_at_05": False}

    diffs = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    nonzero_diffs = diffs[diffs != 0.0]

    n = len(nonzero_diffs)
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant_at_05": False}

    abs_diffs = np.abs(nonzero_diffs)
    ranks = np.argsort(np.argsort(abs_diffs)) + 1.0

    w_pos = np.sum(ranks[nonzero_diffs > 0])
    w_neg = np.sum(ranks[nonzero_diffs < 0])
    w_stat = min(w_pos, w_neg)

    # Normal approximation for p-value when n >= 10
    mean_w = n * (n + 1.0) / 4.0
    sigma_w = math.sqrt(n * (n + 1.0) * (2.0 * n + 1.0) / 24.0)

    if sigma_w > 0:
        z = (w_stat - mean_w) / sigma_w
        # Two-sided p-value via normal CDF
        p_val = 2.0 * 0.5 * math.erfc(-z / math.sqrt(2.0))
        p_val = max(0.0, min(1.0, p_val))
    else:
        p_val = 1.0

    return {
        "statistic": float(round(w_stat, 2)),
        "p_value": float(round(p_val, 5)),
        "significant_at_05": p_val < 0.05,
    }

