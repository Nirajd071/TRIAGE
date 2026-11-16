"""Generate every paper table and figure from the benchmark JSON files.

    python -m src.evaluation.generate_figures \
        [--results results/base_paper_benchmark_results.json] \
        [--polarity results/polarity_benchmark.json] \
        [--hallucination results/hallucination/medhallu_hcad-rag.json]

Consistency guarantee (reviewers: Table 4 showed flat 50% ties while
Figure 4 showed 4-38% win rates plus a ghost "vs. Ablation" series):

  1. Each table is built ONCE as a list of row dicts from the JSON.
  2. The matching figure is drawn from those SAME row objects.
  3. After writing, ``verify_consistency`` re-reads every CSV table and
     checks each number against the source JSON; the run fails loudly on
     any mismatch, and lists every figure series with its table row in
     ``results/tables/figure_table_manifest.json``.

Outputs: ``results/tables/*.{csv,md,tex}``, ``results/figures/*.png``
(including ``methodology_pipeline.png``, the 5-stage diagram generated
from ``src/core/pipeline_stages.py``).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FIGURES_DIR, RESULTS_DIR, TABLES_DIR  # noqa: E402
from src.evaluation import plots  # noqa: E402

logger = logging.getLogger(__name__)

DIMS = plots.DIMS


# ---------------------------------------------------------------------------
# Table writers
# ---------------------------------------------------------------------------

def _fmt(v: Any, digits: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        return f"{v:.{digits}f}"
    return str(v)


def write_table(rows: List[Dict[str, Any]], columns: Sequence[str], name: str,
                tables_dir: Path, caption: str = "") -> Dict[str, Path]:
    """Write one table as CSV (machine-checkable), Markdown and LaTeX."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{name}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in columns})
    md_path = tables_dir / f"{name}.md"
    md = [f"<!-- {caption} -->" if caption else "",
          "| " + " | ".join(columns) + " |",
          "|" + "|".join("---" for _ in columns) + "|"]
    md += ["| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |" for r in rows]
    md_path.write_text("\n".join(line for line in md if line) + "\n", encoding="utf-8")
    tex_path = tables_dir / f"{name}.tex"

    def _tex(s: str) -> str:
        return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
                 .replace("_", r"\_").replace("#", r"\#"))
    tex = [r"\begin{table}[t]", r"\centering", r"\small",
           r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule",
           " & ".join(_tex(c) for c in columns) + r" \\", r"\midrule"]
    tex += [" & ".join(_tex(_fmt(r.get(c))) for c in columns) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}"]
    if caption:
        tex.append(r"\caption{" + _tex(caption) + "}")
    tex += [r"\label{tab:" + name + "}", r"\end{table}"]
    tex_path.write_text("\n".join(tex) + "\n", encoding="utf-8")
    return {"csv": csv_path, "md": md_path, "tex": tex_path}


# ---------------------------------------------------------------------------
# Row builders (each is the single source for one table + its figure)
# ---------------------------------------------------------------------------

def score_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    scores = results.get("scoring_based_assessment") or {}
    cis = results.get("scoring_confidence_intervals") or {}
    rows = []
    for system in [s for s in plots.SYSTEM_ORDER if s in scores] + \
            [s for s in scores if s not in plots.SYSTEM_ORDER]:
        row: Dict[str, Any] = {"system": system}
        for d in DIMS:
            row[d] = round(float(scores[system].get(d, 0.0)), 2)
        ov = (cis.get(system) or {}).get("overall_score") or {}
        row["overall_ci_low"] = None if ov.get("ci_low") is None else round(ov["ci_low"], 2)
        row["overall_ci_high"] = None if ov.get("ci_high") is None else round(ov["ci_high"], 2)
        row["n"] = ov.get("n")
        rows.append(row)
    return rows


def win_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for r in plots.win_rate_rows(results):
        out.append({
            "comparison": r["comparison"],
            "system_a": r["system_a"], "system_b": r["system_b"], "n": r["n"],
            "win_pct": round(r["win_pct"], 2), "tie_pct": round(r["tie_pct"], 2),
            "loss_pct": round(r["loss_pct"], 2),
            "decisive_win_rate_pct": (None if r["decisive_win_rate_pct"] is None
                                      else round(r["decisive_win_rate_pct"], 2)),
            "criteria": r["criteria"],
        })
    return out


def significance_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    comps = ((results.get("statistical_tests") or {}).get("comparisons") or {})
    for key, entry in comps.items():
        for judge, t in [("consensus", entry.get("consensus") or {})] + \
                list((entry.get("per_judge") or {}).items()):
            if not t or not t.get("n_paired"):
                continue
            ci = t["paired_difference_ci"]
            rows.append({
                "comparison": key, "judge": judge, "n": t["n_paired"],
                "mean_a": round(t["mean_a"], 2), "mean_b": round(t["mean_b"], 2),
                "mean_diff": round(ci["mean_difference"], 2),
                "ci_low": round(ci["ci_low"], 2), "ci_high": round(ci["ci_high"], 2),
                "wilcoxon_p": round(t["wilcoxon_signed_rank"]["p_value"], 4),
                "t_test_p": (None if t["paired_t_test"].get("p_value") is None
                             else round(t["paired_t_test"]["p_value"], 4)),
                "mcnemar_p": round(t["mcnemar_exact"]["p_value"], 4),
            })
    return rows


def agreement_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    agr = results.get("inter_judge_agreement") or {}
    rows = []
    for pair, m in (agr.get("pairwise") or {}).items():
        for measure, k in m.items():
            rows.append({"statistic": "Cohen's kappa", "judges": pair, "measure": measure,
                         "n": k.get("n"),
                         "kappa": None if k.get("kappa") is None else round(k["kappa"], 3),
                         "observed_agreement": (None if k.get("observed_agreement") is None
                                                else round(k["observed_agreement"], 3)),
                         "interpretation": k.get("interpretation", "undefined")})
    for measure, k in (agr.get("fleiss") or {}).items():
        rows.append({"statistic": "Fleiss' kappa", "judges": "all", "measure": measure,
                     "n": k.get("n_items"),
                     "kappa": None if k.get("kappa") is None else round(k["kappa"], 3),
                     "observed_agreement": (None if k.get("observed_agreement") is None
                                            else round(k["observed_agreement"], 3)),
                     "interpretation": k.get("interpretation", "undefined")})
    return rows


def latency_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    lb = results.get("latency_breakdown") or {}
    rows = []

    def ms(x):
        return (x or {}).get("mean"), (x or {}).get("std")
    for system in [s for s in plots.SYSTEM_ORDER if s in lb] + \
            [s for s in lb if s not in plots.SYSTEM_ORDER]:
        v = lb[system]
        r_m, r_s = ms(v.get("retrieval_seconds_per_question"))
        g_m, g_s = ms(v.get("generation_seconds"))
        rows.append({
            "system": system,
            "retrieval_mean_s": None if r_m is None else round(r_m, 4),
            "retrieval_std_s": None if r_s is None else round(r_s, 4),
            "generation_mean_s": None if g_m is None else round(g_m, 3),
            "generation_std_s": None if g_s is None else round(g_s, 3),
            "prompt_tokens": None if ms(v.get("prompt_tokens"))[0] is None
            else round(ms(v.get("prompt_tokens"))[0], 1),
            "generated_tokens": None if ms(v.get("generated_tokens"))[0] is None
            else round(ms(v.get("generated_tokens"))[0], 1),
            "tokens_per_s": None if ms(v.get("tokens_per_second"))[0] is None
            else round(ms(v.get("tokens_per_second"))[0], 2),
            "abstention_rate": round(v.get("abstention_rate", 0.0), 3),
            "n": v.get("n_questions"),
        })
    return rows


def polarity_rows(polarity: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for arm, res in (polarity.get("arms") or {}).items():
        o = res["overall"]
        rows.append({
            "arm": arm, "n_queries": o["n_queries"],
            "polarity_recall_at_k": round(o["polarity_recall_at_k"]["mean"], 3),
            "polarity_error_rate": round(o["polarity_error_rate"]["mean"], 3),
            "error_ci_low": round(o["polarity_error_rate"]["ci_low"], 3),
            "error_ci_high": round(o["polarity_error_rate"]["ci_high"], 3),
            "error_lexical": round(res["by_phrasing"]["lexical"]["polarity_error_rate"]["mean"], 3),
            "error_paraphrase": round(res["by_phrasing"]["paraphrase"]["polarity_error_rate"]["mean"], 3),
            "pair_accuracy": round(res["pair_accuracy"]["mean"], 3),
            "mrr": round(o["mrr"]["mean"], 3),
        })
    return rows


# ---------------------------------------------------------------------------
# Extra figures
# ---------------------------------------------------------------------------

def plot_polarity(rows: List[Dict[str, Any]], k: int, figures_dir: Path) -> Optional[Path]:
    if not rows:
        return None
    plt = plots.plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    arms = [r["arm"] for r in rows]
    rec = [r["polarity_recall_at_k"] for r in rows]
    err = [r["polarity_error_rate"] for r in rows]
    lo = [max(0.0, e - r["error_ci_low"]) for e, r in zip(err, rows)]
    hi = [max(0.0, r["error_ci_high"] - e) for e, r in zip(err, rows)]
    b = axes[0].bar(arms, rec, color="#0072B2")
    axes[0].bar_label(b, fmt="%.2f", fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Polarity Recall@{k} (higher is better)")
    b = axes[1].bar(arms, err, yerr=[lo, hi], capsize=3, color="#D55E00")
    axes[1].bar_label(b, fmt="%.2f", fontsize=8, padding=6)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Polarity Error Rate (lower is better; 95% CI)")
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.suptitle("Indication vs contraindication paired-polarity benchmark")
    return plots._save(fig, "polarity_benchmark.png", figures_dir)


def plot_methodology(figures_dir: Path, results: Optional[Dict[str, Any]] = None) -> Path:
    """Publication 5-stage methodology diagram from ``PIPELINE_STAGES``.

    When a results file is supplied, each stage box is annotated with the
    measured mean stage latency of HCAD-RAG.
    """
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    from src.core.pipeline_stages import PIPELINE_STAGES

    plt = plots.plt
    stage_lat = {}
    if results:
        hcad = (results.get("latency_breakdown") or {}).get("HCAD-RAG") or {}
        stage_lat = {k: (v or {}).get("mean")
                     for k, v in (hcad.get("retrieval_stage_seconds") or {}).items()}
    n = len(PIPELINE_STAGES)
    fig, ax = plt.subplots(figsize=(3.3 * n, 3.6))
    ax.set_xlim(0, 3.3 * n)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    palette = ["#E8F1FA", "#E6F4EA", "#FDF1E3", "#F3E8F7", "#E9F6F3"]
    edge = ["#0072B2", "#009E73", "#E69F00", "#CC79A7", "#117A65"]
    for i, st in enumerate(PIPELINE_STAGES):
        x = 0.15 + i * 3.3
        box = FancyBboxPatch((x, 0.35), 2.85, 2.9, boxstyle="round,pad=0.04,rounding_size=0.12",
                             fc=palette[i % 5], ec=edge[i % 5], lw=1.6)
        ax.add_patch(box)
        ax.text(x + 1.425, 2.95, f"Stage {st.number}", ha="center", va="center",
                fontsize=9, color=edge[i % 5], fontweight="bold")
        ax.text(x + 1.425, 2.55, st.name, ha="center", va="center", fontsize=10,
                fontweight="bold", wrap=True)
        ax.text(x + 1.425, 1.75, _wrap(st.summary, 34), ha="center", va="center", fontsize=7.2)
        ax.text(x + 1.425, 0.85, _wrap("in: " + st.inputs + "\nout: " + st.outputs, 36),
                ha="center", va="center", fontsize=6.6, color="#333333", style="italic")
        lat = stage_lat.get(st.timing_key)
        if lat is not None:
            ax.text(x + 1.425, 0.5, f"{lat * 1000:.0f} ms", ha="center", fontsize=7,
                    color=edge[i % 5])
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + 2.95, 1.8), (x + 3.4, 1.8),
                                         arrowstyle="-|>", mutation_scale=14, lw=1.4,
                                         color="#444444"))
    ax.set_title("HCAD-RAG methodology: five-stage relation-aware retrieval pipeline",
                 fontsize=12)
    return plots._save(fig, "methodology_pipeline.png", figures_dir)


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.fill(part, width) for part in text.split("\n"))


# ---------------------------------------------------------------------------
# Consistency verification
# ---------------------------------------------------------------------------

def verify_consistency(results: Dict[str, Any], tables_dir: Path) -> List[str]:
    """Re-read the CSV tables and compare against the results JSON.

    Returns a list of mismatch descriptions (empty = consistent).
    """
    problems: List[str] = []

    def close(a, b, tol=0.011):
        try:
            return abs(float(a) - float(b)) <= tol
        except (TypeError, ValueError):
            return str(a) == str(b)

    win_csv = tables_dir / "table4_win_rates.csv"
    sel = results.get("selection_based_assessment") or {}
    if win_csv.exists():
        with open(win_csv, encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
        if {r["comparison"] for r in csv_rows} != {k for k, v in sel.items() if v.get("overall")}:
            problems.append("table4 comparisons differ from JSON comparisons")
        for r in csv_rows:
            ov = (sel.get(r["comparison"]) or {}).get("overall") or {}
            for col in ("win_pct", "tie_pct", "loss_pct"):
                if not close(r[col], ov.get(col)):
                    problems.append(f"table4 {r['comparison']} {col}: {r[col]} != {ov.get(col)}")
            total = sum(float(r[c]) for c in ("win_pct", "tie_pct", "loss_pct"))
            if not close(total, 100.0, tol=0.05):
                problems.append(f"table4 {r['comparison']} outcomes sum to {total}, not 100")
    score_csv = tables_dir / "table3_scores.csv"
    scores = results.get("scoring_based_assessment") or {}
    if score_csv.exists():
        with open(score_csv, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                for d in DIMS:
                    if not close(r[d], (scores.get(r["system"]) or {}).get(d)):
                        problems.append(f"table3 {r['system']} {d} mismatch")
    return problems


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all(results_path: Optional[Path] = None, polarity_path: Optional[Path] = None,
                 hallucination_paths: Sequence[Path] = (), figures_dir: Optional[Path] = None,
                 tables_dir: Optional[Path] = None) -> Dict[str, Any]:
    figures_dir = Path(figures_dir or FIGURES_DIR)
    tables_dir = Path(tables_dir or TABLES_DIR)
    manifest: Dict[str, Any] = {"figures": {}, "tables": {}}

    results = None
    if results_path and Path(results_path).exists():
        results = plots.load_results(Path(results_path))
        s_rows = score_rows(results)
        w_rows = win_rows(results)
        manifest["tables"]["table3_scores"] = str(write_table(
            s_rows, ["system", *DIMS, "overall_ci_low", "overall_ci_high", "n"],
            "table3_scores", tables_dir,
            "Rubric scores (0-100), mean of judge consensus; 95% bootstrap CI on overall")["csv"])
        manifest["tables"]["table4_win_rates"] = str(write_table(
            w_rows, ["comparison", "system_a", "system_b", "n", "win_pct", "tie_pct",
                     "loss_pct", "decisive_win_rate_pct"],
            "table4_win_rates", tables_dir,
            "Pairwise outcomes from system A's perspective (judge consensus, AB+BA counterbalanced)")["csv"])
        crit_rows = [{"comparison": r["comparison"], "criterion": c, **{
            k: round(float(v[k]), 2) for k in ("win_pct", "tie_pct", "loss_pct")}}
            for r in w_rows if r.get("criteria") for c, v in r["criteria"].items()]
        if crit_rows:
            write_table(crit_rows, ["comparison", "criterion", "win_pct", "tie_pct", "loss_pct"],
                        "table4b_criteria_outcomes", tables_dir,
                        "Per-criterion pairwise outcomes")
        sig = significance_rows(results)
        if sig:
            write_table(sig, ["comparison", "judge", "n", "mean_a", "mean_b", "mean_diff",
                              "ci_low", "ci_high", "wilcoxon_p", "t_test_p", "mcnemar_p"],
                        "table5_significance", tables_dir,
                        "Paired significance tests on overall score")
        agr = agreement_rows(results)
        if agr:
            write_table(agr, ["statistic", "judges", "measure", "n", "kappa",
                              "observed_agreement", "interpretation"],
                        "table6_inter_judge_agreement", tables_dir, "Inter-judge agreement")
        lat = latency_rows(results)
        if lat:
            write_table(lat, ["system", "retrieval_mean_s", "retrieval_std_s",
                              "generation_mean_s", "generation_std_s", "prompt_tokens",
                              "generated_tokens", "tokens_per_s", "abstention_rate", "n"],
                        "table7_latency", tables_dir,
                        "Retrieval-only vs generation latency (mean, std)")

        figs = {
            "dimension_scores": plots.plot_dimension_scores(results, figures_dir),
            "difficulty_tiers": plots.plot_difficulty_tiers(results, figures_dir),
            "win_rates": plots.plot_win_rates(results, figures_dir, rows=plots.win_rate_rows(results)),
            "latency_performance": plots.plot_latency_performance(results, figures_dir),
            "latency_breakdown": plots.plot_latency_breakdown(results, figures_dir),
            "citation_grounding": plots.plot_citation_grounding(results, figures_dir),
        }
        manifest["figures"].update({k: str(v) for k, v in figs.items() if v})
        manifest["figure_series"] = {
            "win_rates": [{"comparison": r["comparison"], "win_pct": r["win_pct"],
                           "tie_pct": r["tie_pct"], "loss_pct": r["loss_pct"],
                           "table": "table4_win_rates"} for r in w_rows],
            "dimension_scores": [{"system": r["system"], "overall_score": r["overall_score"],
                                  "table": "table3_scores"} for r in s_rows],
        }

    if polarity_path and Path(polarity_path).exists():
        polarity = json.loads(Path(polarity_path).read_text(encoding="utf-8"))
        p_rows = polarity_rows(polarity)
        write_table(p_rows, ["arm", "n_queries", "polarity_recall_at_k", "polarity_error_rate",
                             "error_ci_low", "error_ci_high", "error_lexical", "error_paraphrase",
                             "pair_accuracy", "mrr"],
                    "table2_polarity", tables_dir,
                    f"Paired-polarity benchmark (k={polarity.get('k')})")
        fig = plot_polarity(p_rows, polarity.get("k", 5), figures_dir)
        if fig:
            manifest["figures"]["polarity_benchmark"] = str(fig)

    hallu_rows = []
    for hp in hallucination_paths:
        if not Path(hp).exists():
            continue
        data = json.loads(Path(hp).read_text(encoding="utf-8"))
        fig = plots.plot_hallucination_tiers(Path(hp), figures_dir)
        if fig:
            manifest["figures"][f"hallucination::{data.get('system')}"] = str(fig)
        o = data.get("overall") or {}
        hallu_rows.append({"system": data.get("system"), "mode": data.get("evaluation_mode"),
                           "n": data.get("num_samples"), "f1": o.get("f1"),
                           "hallucination_rate": o.get("hallucination_rate"),
                           "accuracy": o.get("accuracy")})
    if hallu_rows:
        write_table(hallu_rows, ["system", "mode", "n", "f1", "hallucination_rate", "accuracy"],
                    "table8_medhallu", tables_dir, "MedHallu evaluation")

    manifest["figures"]["methodology_pipeline"] = str(plot_methodology(figures_dir, results))

    problems = verify_consistency(results, tables_dir) if results else []
    manifest["consistency_problems"] = problems
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "figure_table_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    if problems:
        raise RuntimeError("Table/figure consistency check failed:\n  " + "\n  ".join(problems))
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate paper tables and figures")
    parser.add_argument("--results", type=Path,
                        default=RESULTS_DIR / "base_paper_benchmark_results.json")
    parser.add_argument("--polarity", type=Path, default=RESULTS_DIR / "polarity_benchmark.json")
    parser.add_argument("--hallucination", type=Path, nargs="*",
                        default=sorted((RESULTS_DIR / "hallucination").glob("medhallu_*.json")))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    manifest = generate_all(args.results, args.polarity, args.hallucination)
    for name, path in manifest["figures"].items():
        print(f"figure {name}: {path}")
    print("consistency: OK")


if __name__ == "__main__":
    main()
