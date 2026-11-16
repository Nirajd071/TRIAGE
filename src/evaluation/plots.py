"""Result figures derived ONLY from the results JSON.

Rules (reviewer R3 / meta-review: Table 4 vs Figure 4 disagreed):
  - Every figure is drawn from the same row objects that the tables in
    ``generate_figures.py`` are written from - never recomputed here.
  - Win-rate figures draw complete win / tie / loss stacks for exactly the
    comparisons present in the JSON. No hard-coded panel list, so no
    ghost series (e.g. a "vs. Ablation" panel with no data behind it).
  - System names come from the JSON (``config.ARM_LABELS``); the retired
    generic "Baseline LLM" label is mapped to
    "Llama-3.1-8B-Instruct (Zero-Retrieval / Closed-Book)" on load.
  - Core-arm gate: the closed-book baseline, the dense retriever and
    HCAD-RAG must all be present; a figure never silently degrades into
    a two-way comparison.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Must be set BEFORE importing matplotlib: the import itself validates
# MPLBACKEND. Notebook kernels export the inline backend here, which crashes
# script-mode runs, so force Agg rather than setdefault.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import (  # noqa: E402
    ARM_LABELS,
    ARM_SHORT_LABELS,
    BASELINE_LABEL,
    FIGURES_DIR,
    canonical_system_label,
)

DIMS = ["comprehensiveness", "diversity", "empowerment", "logical", "readability", "overall_score"]

#: Canonical display order of arms (labels).
SYSTEM_ORDER = [ARM_LABELS[a] for a in ("closed_book", "bm25", "dense", "hybrid", "hcad",
                                        "hcad_no_decoupling", "hcad_no_gs", "hcad_no_gate")]
CORE_SYSTEMS = [ARM_LABELS["closed_book"], ARM_LABELS["dense"], ARM_LABELS["hcad"]]
SHORT = {ARM_LABELS[a]: ARM_SHORT_LABELS[a] for a in ARM_LABELS}

# Colour-blind-safe palette (Okabe-Ito) with fixed per-arm identity.
SYSTEM_STYLE = {
    ARM_LABELS["closed_book"]: {"color": "#7f7f7f", "marker": "o", "linestyle": "--"},
    ARM_LABELS["bm25"]: {"color": "#E69F00", "marker": "v", "linestyle": ":"},
    ARM_LABELS["dense"]: {"color": "#0072B2", "marker": "s", "linestyle": "-."},
    ARM_LABELS["hybrid"]: {"color": "#56B4E9", "marker": "D", "linestyle": ":"},
    ARM_LABELS["hcad"]: {"color": "#009E73", "marker": "^", "linestyle": "-"},
    ARM_LABELS["hcad_no_decoupling"]: {"color": "#CC79A7", "marker": "P", "linestyle": "-"},
    ARM_LABELS["hcad_no_gs"]: {"color": "#D55E00", "marker": "X", "linestyle": "-"},
    ARM_LABELS["hcad_no_gate"]: {"color": "#F0E442", "marker": "*", "linestyle": "-"},
}
OUTCOME_COLORS = {"win": "#009E73", "tie": "#BBBBBB", "loss": "#D55E00"}


# ---------------------------------------------------------------------------
# Loading / normalisation
# ---------------------------------------------------------------------------

def _relabel(d: Dict[str, Any]) -> Dict[str, Any]:
    return {canonical_system_label(k): v for k, v in (d or {}).items()}


def normalize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Map legacy labels / legacy win-rate entries onto the current schema.

    Legacy win-rate entries only carried criterion WIN percentages (ties
    were never shown in the figure - the root of the Table 4 / Figure 4
    disagreement). For those, the overall stack is rebuilt from
    ``overall_win_rate_pct`` + ``tie_rate_pct`` and per-criterion stacks
    are marked unavailable rather than guessed.
    """
    out = dict(results)
    for key in ("scoring_based_assessment", "retrieval_latency_seconds", "citation_precision",
                "latency_breakdown", "scoring_confidence_intervals"):
        if key in out:
            out[key] = _relabel(out[key])
    if "scoring_by_difficulty_tier" in out:
        out["scoring_by_difficulty_tier"] = {t: _relabel(m) for t, m in
                                             out["scoring_by_difficulty_tier"].items()}
    legacy_keys = {"HCAD_vs_Vanilla": ("hcad", "dense"),
                   "HCAD_vs_Baseline": ("hcad", "closed_book"),
                   "Vanilla_vs_Baseline": ("dense", "closed_book")}
    sel = {}
    for key, entry in (out.get("selection_based_assessment") or {}).items():
        if not entry:
            continue
        entry = dict(entry)
        if key in legacy_keys and "overall" not in entry:
            a, b = legacy_keys[key]
            win = float(entry.get("overall_win_rate_pct", 0.0))
            tie = float(entry.get("tie_rate_pct", 0.0))
            entry = {"system_a": ARM_LABELS[a], "system_b": ARM_LABELS[b],
                     "n": entry.get("n"),
                     "overall": {"win_pct": win, "tie_pct": tie,
                                 "loss_pct": max(0.0, 100.0 - win - tie)},
                     "criteria": None, "legacy": True}
            key = f"{a}_vs_{b}"
        sel[key] = entry
    out["selection_based_assessment"] = sel
    return out


def load_results(path: Path) -> Dict[str, Any]:
    return normalize_results(json.loads(Path(path).read_text(encoding="utf-8")))


def _systems_present(scores: Dict[str, Any], context: str) -> List[str]:
    """Core-arm gate + canonical ordering of the arms present."""
    missing = [s for s in CORE_SYSTEMS if s not in scores]
    if missing:
        raise ValueError(
            f"Core-arm protocol violated in {context}: missing systems {missing}. "
            f"Every figure must include {CORE_SYSTEMS}; refusing to draw a "
            "reduced comparison."
        )
    ordered = [s for s in SYSTEM_ORDER if s in scores]
    return ordered + [s for s in scores if s not in ordered]


def _style(label: str) -> Dict[str, str]:
    return SYSTEM_STYLE.get(label, {"color": "#444444", "marker": "o", "linestyle": "-"})


def _short(label: str) -> str:
    return SHORT.get(label, label)


def _save(fig, name: str, figures_dir: Optional[Path] = None) -> Path:
    out_dir = Path(figures_dir or FIGURES_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / name
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_dimension_scores(results: dict, figures_dir: Optional[Path] = None):
    """Grouped bars: 5 dimensions + Overall per arm, with bootstrap 95% CIs."""
    scores = results.get("scoring_based_assessment", {})
    if not scores:
        return None
    systems = _systems_present(scores, "dimension scores")
    cis = results.get("scoring_confidence_intervals") or {}
    fig, ax = plt.subplots(figsize=(max(11, 1.6 * len(systems) + 6), 5.5))
    n = len(systems)
    width = 0.8 / n
    for i, sys_name in enumerate(systems):
        xs = [d + (i - (n - 1) / 2) * width for d in range(len(DIMS))]
        ys = [scores[sys_name].get(d, 0) for d in DIMS]
        yerr = None
        if sys_name in cis:
            lo = [max(0.0, y - (cis[sys_name].get(d, {}).get("ci_low") or y)) for d, y in zip(DIMS, ys)]
            hi = [max(0.0, (cis[sys_name].get(d, {}).get("ci_high") or y) - y) for d, y in zip(DIMS, ys)]
            yerr = [lo, hi]
        bars = ax.bar(xs, ys, width=width, label=_short(sys_name), yerr=yerr, capsize=2,
                      color=_style(sys_name)["color"], edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.1f", fontsize=6, padding=2)
    ax.set_xticks(range(len(DIMS)))
    ax.set_xticklabels([d.replace("_score", "").capitalize() for d in DIMS], rotation=20)
    ax.set_ylabel("Score (0-100, consensus of judges)")
    ax.set_ylim(0, 105)
    ax.set_title("Rubric scores by arm (error bars: bootstrap 95% CI)")
    ax.legend(fontsize=8, ncol=min(4, n))
    return _save(fig, "dimension_scores.png", figures_dir)


def plot_difficulty_tiers(results: dict, figures_dir: Optional[Path] = None):
    """Overall score vs 1/2/3-stage difficulty per arm (base-paper Fig. 5)."""
    by_tier = results.get("scoring_by_difficulty_tier", {})
    if not by_tier:
        return None
    systems = _systems_present(next(iter(by_tier.values()), {}), "difficulty tiers")
    tiers = sorted(by_tier, key=int)
    fig, ax = plt.subplots(figsize=(8, 5))
    for sys_name in systems:
        style = _style(sys_name)
        ys = [by_tier.get(t, {}).get(sys_name, 0.0) for t in tiers]
        ax.plot(range(len(tiers)), ys, marker=style["marker"], linestyle=style["linestyle"],
                color=style["color"], label=_short(sys_name), linewidth=2, markersize=7)
    ax.set_xticks(range(len(tiers)))
    ax.set_xticklabels([f"{t}-stage" for t in tiers])
    ax.set_ylabel("Overall score (0-100)")
    ax.set_title("Robustness across question difficulty")
    ax.legend(fontsize=8)
    return _save(fig, "difficulty_tiers.png", figures_dir)


def win_rate_rows(results: dict) -> List[Dict[str, Any]]:
    """Rows for Table 4 AND Figure 4 (single source of truth)."""
    rows = []
    for key, entry in (results.get("selection_based_assessment") or {}).items():
        ov = entry.get("overall")
        if not ov:
            continue
        rows.append({
            "comparison": key,
            "system_a": entry.get("system_a"),
            "system_b": entry.get("system_b"),
            "n": entry.get("n"),
            "win_pct": float(ov["win_pct"]),
            "tie_pct": float(ov["tie_pct"]),
            "loss_pct": float(ov["loss_pct"]),
            "decisive_win_rate_pct": entry.get("decisive_win_rate_pct"),
            "criteria": entry.get("criteria"),
        })
    return rows


def plot_win_rates(results: dict, figures_dir: Optional[Path] = None,
                   rows: Optional[List[Dict[str, Any]]] = None):
    """Figure 4: overall win / tie / loss stacks, one bar per comparison
    present in the results (plus per-criterion stacks when available)."""
    rows = rows if rows is not None else win_rate_rows(results)
    if not rows:
        return None
    with_criteria = [r for r in rows if r.get("criteria")]
    ncols = 1 + (1 if with_criteria else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(7.5 * ncols, 0.7 * len(rows) + 2.5),
                             squeeze=False)
    ax = axes[0][0]
    labels = [f"{_short(r['system_a'])} vs {_short(r['system_b'])} (n={r['n']})" for r in rows]
    left = [0.0] * len(rows)
    for outcome in ("win", "tie", "loss"):
        vals = [r[f"{outcome}_pct"] for r in rows]
        bars = ax.barh(labels, vals, left=left, color=OUTCOME_COLORS[outcome],
                       label=outcome.capitalize(), edgecolor="white")
        ax.bar_label(bars, labels=[f"{v:.0f}%" if v >= 6 else "" for v in vals],
                     label_type="center", fontsize=8)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_xlabel("% of questions (system A perspective)")
    ax.set_title("Overall pairwise outcome (judge consensus)")
    ax.legend(loc="lower right", fontsize=8)

    if with_criteria:
        ax2 = axes[0][1]
        crit_names = list(with_criteria[0]["criteria"].keys())
        n_rows = len(with_criteria)
        height = 0.8 / n_rows
        for i, r in enumerate(with_criteria):
            ys = [c + (i - (n_rows - 1) / 2) * height for c in range(len(crit_names))]
            left = [0.0] * len(crit_names)
            for outcome in ("win", "tie", "loss"):
                vals = [float(r["criteria"][c][f"{outcome}_pct"]) for c in crit_names]
                ax2.barh(ys, vals, left=left, height=height * 0.92,
                         color=OUTCOME_COLORS[outcome], edgecolor="white",
                         alpha=1.0 if i == 0 else 0.75)
                left = [a + b for a, b in zip(left, vals)]
        ax2.set_yticks(range(len(crit_names)))
        ax2.set_yticklabels([c.capitalize() for c in crit_names], fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlim(0, 100)
        ax2.set_xlabel("% of questions")
        ax2.set_title("Per-criterion win/tie/loss\n(bars in comparison order: "
                      + "; ".join(f"{_short(r['system_a'])} vs {_short(r['system_b'])}"
                                  for r in with_criteria) + ")", fontsize=8)
    return _save(fig, "win_rates.png", figures_dir)


def plot_latency_performance(results: dict, figures_dir: Optional[Path] = None):
    """Retrieval-only latency (mean +- std) vs quality; generation excluded."""
    scores = results.get("scoring_based_assessment", {})
    if not scores:
        return None
    systems = _systems_present(scores, "latency trade-off")
    breakdown = results.get("latency_breakdown") or {}
    lat = dict(results.get("retrieval_latency_seconds", {}))
    lat.setdefault(BASELINE_LABEL, 0.0)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for sys_name in systems:
        if sys_name not in lat:
            continue
        x = lat[sys_name] or 0.0
        std = ((breakdown.get(sys_name) or {}).get("retrieval_seconds_per_question") or {}).get("std")
        y = scores.get(sys_name, {}).get("overall_score", 0)
        style = _style(sys_name)
        ax.errorbar(x, y, xerr=std if std else None, fmt=style["marker"], ms=11,
                    color=style["color"], capsize=3, label=_short(sys_name))
        ax.annotate(f"{x:.2f}s", (x, y), textcoords="offset points", xytext=(8, -4), fontsize=7)
    ax.set_xlabel("Retrieval-only latency per question (s, mean +- std; generation excluded)")
    ax.set_ylabel("Overall score (0-100)")
    ax.set_title("Retrieval cost vs answer quality")
    ax.legend(fontsize=8)
    return _save(fig, "latency_performance.png", figures_dir)


def plot_latency_breakdown(results: dict, figures_dir: Optional[Path] = None):
    """Stacked retrieval vs generation seconds per arm (disentangled latency)."""
    breakdown = results.get("latency_breakdown") or {}
    if not breakdown:
        return None
    systems = [s for s in SYSTEM_ORDER if s in breakdown] + \
        [s for s in breakdown if s not in SYSTEM_ORDER]
    ret = [(breakdown[s]["retrieval_seconds_per_question"] or {}).get("mean") or 0.0 for s in systems]
    gen = [(breakdown[s]["generation_seconds"] or {}).get("mean") or 0.0 for s in systems]
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(systems) + 3), 4.8))
    names = [_short(s) for s in systems]
    b1 = ax.bar(names, ret, color="#0072B2", label="Retrieval (embed + search + graph + GS + gate)")
    b2 = ax.bar(names, gen, bottom=ret, color="#E69F00", label="Generation (prefill + decode)")
    ax.bar_label(b1, fmt="%.2f", label_type="center", fontsize=7, color="white")
    ax.bar_label(b2, fmt="%.2f", label_type="center", fontsize=7)
    ax.set_ylabel("Seconds per question (mean)")
    ax.set_title("Latency decomposition: retrieval vs generation")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, "latency_breakdown.png", figures_dir)


def plot_citation_grounding(results: dict, figures_dir: Optional[Path] = None):
    """Citation coverage / count / precision for every arm."""
    citations = _relabel(results.get("citation_precision", {}))
    scores = results.get("scoring_based_assessment", {})
    if not citations:
        return None
    systems = _systems_present(scores, "citation grounding")
    if BASELINE_LABEL not in citations:
        n_baseline = (next(iter(citations.values()), {}) or {}).get("n_answers", 0)
        citations[BASELINE_LABEL] = {"n_answers_with_citations": 0, "n_answers": n_baseline,
                                     "total_citations": 0, "mean_precision": None}
    systems = [s for s in systems if s in citations]
    coverage, totals, precisions = [], [], []
    for sys_name in systems:
        c = citations.get(sys_name, {})
        n_ans = c.get("n_answers", 0) or 1
        coverage.append((c.get("n_answers_with_citations", 0) / n_ans) * 100.0)
        totals.append(c.get("total_citations", 0))
        mp = c.get("mean_precision")
        precisions.append(mp * 100.0 if mp is not None else 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    specs = [("Citation coverage (% of answers)", coverage, 100),
             ("Total [Evidence i] citations", totals, None),
             ("Mean citation precision (%)", precisions, 100)]
    names = [_short(s) for s in systems]
    for ax, (title, ys, ymax) in zip(axes, specs):
        bars = ax.bar(names, ys, color=[_style(s)["color"] for s in systems])
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        ax.set_title(title, fontsize=10)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8)
        if ymax:
            ax.set_ylim(0, ymax * 1.08)
    fig.suptitle("Citation & evidence grounding (closed-book arm cites nothing by construction)",
                 y=1.03)
    return _save(fig, "citation_grounding.png", figures_dir)


def plot_hallucination_tiers(hallu_path: Path | None = None,
                             figures_dir: Optional[Path] = None):
    """MedHallu by difficulty tier.

    detector mode   -> F1 of the hallucination classifier;
    system_answers  -> hallucination RATE of the system's own answers
                       (F1 is undefined there: there is no ground-truth label
                       per generated answer).
    """
    if hallu_path is None or not Path(hallu_path).exists():
        return None
    data = json.loads(Path(hallu_path).read_text(encoding="utf-8"))
    tiers = ["easy", "medium", "hard"]
    mode = data.get("evaluation_mode", "detector")
    key, ylabel = ("hallucination_rate", "Hallucination rate") if mode == "system_answers" \
        else ("f1", "F1")
    ys = [float((data.get(t) or {}).get(key) or 0.0) for t in tiers]
    ns = [(data.get(t) or {}).get("n", "") for t in tiers]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar([f"{t}\n(n={n})" if n != "" else t for t, n in zip(tiers, ns)], ys,
                  color="#009E73")
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(f"MedHallu ({mode}) - {data.get('system', 'system')}")
    return _save(fig, f"hallucination_tiers_{data.get('system', 'system').replace(' ', '_').lower()}.png",
                 figures_dir)


def plot_all(results: dict, hallu_path: Path | None = None,
             figures_dir: Optional[Path] = None) -> list:
    results = normalize_results(results)
    return [p for p in [plot_dimension_scores(results, figures_dir),
                        plot_difficulty_tiers(results, figures_dir),
                        plot_win_rates(results, figures_dir),
                        plot_latency_performance(results, figures_dir),
                        plot_latency_breakdown(results, figures_dir),
                        plot_citation_grounding(results, figures_dir),
                        plot_hallucination_tiers(hallu_path, figures_dir)] if p]
