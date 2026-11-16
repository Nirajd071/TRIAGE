import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib
matplotlib.use("Agg")


def _triarm_results():
    """Complete tri-arm fixture: three systems, three selection panels,
    baseline latency/citation arms, three difficulty tiers."""
    return {
        "scoring_based_assessment": {
            "Baseline LLM": {d: 70 for d in
                             ["comprehensiveness", "diversity", "empowerment",
                              "logical", "readability", "overall_score"]},
            "Vanilla RAG": {d: 75 for d in
                            ["comprehensiveness", "diversity", "empowerment",
                             "logical", "readability", "overall_score"]},
            "HCAD-RAG": {d: 82 for d in
                         ["comprehensiveness", "diversity", "empowerment",
                          "logical", "readability", "overall_score"]},
        },
        "scoring_by_difficulty_tier": {
            "1": {"Baseline LLM": 72, "Vanilla RAG": 76, "HCAD-RAG": 80},
            "2": {"Baseline LLM": 70, "Vanilla RAG": 75, "HCAD-RAG": 84},
            "3": {"Baseline LLM": 69, "Vanilla RAG": 73, "HCAD-RAG": 86},
        },
        "selection_based_assessment": {
            "HCAD_vs_Baseline": {
                "target_system": "HCAD-RAG", "overall_win_rate_pct": 55.0,
                "criteria_win_rates_pct": {c: 60 for c in
                    ["comprehensiveness", "empowerment", "accuracy", "relevance",
                     "coherence", "clarity", "logical", "flexibility"]}},
            "HCAD_vs_Vanilla": {
                "target_system": "HCAD-RAG", "overall_win_rate_pct": 52.0,
                "criteria_win_rates_pct": {c: 58 for c in
                    ["comprehensiveness", "empowerment", "accuracy", "relevance",
                     "coherence", "clarity", "logical", "flexibility"]}},
            "Vanilla_vs_Baseline": {
                "target_system": "Vanilla RAG", "overall_win_rate_pct": 61.0,
                "criteria_win_rates_pct": {c: 62 for c in
                    ["comprehensiveness", "empowerment", "accuracy", "relevance",
                     "coherence", "clarity", "logical", "flexibility"]}},
        },
        "retrieval_latency_seconds": {
            "Baseline LLM": 0.0, "Vanilla RAG": 0.2, "HCAD-RAG": 0.9,
        },
        "citation_precision": {
            "Baseline LLM": {"n_answers_with_citations": 0, "n_answers": 50,
                             "total_citations": 0, "mean_precision": None},
            "Vanilla RAG": {"n_answers_with_citations": 1, "n_answers": 50,
                            "total_citations": 2, "mean_precision": 1.0},
            "HCAD-RAG": {"n_answers_with_citations": 30, "n_answers": 50,
                         "total_citations": 98, "mean_precision": 0.9667},
        },
    }


def test_plots_render_from_results_json(tmp_path, monkeypatch):
    from src.evaluation import plots
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    out = plots.plot_all(_triarm_results(), hallu_path=None)
    assert len(out) == 5, out
    for p in out:
        assert Path(p).exists() and Path(p).stat().st_size > 5000


def test_triarm_gate_refuses_two_way_comparison(tmp_path, monkeypatch):
    """Tri-arm protocol: a figure missing any of the three systems must be
    refused, never silently rendered as a 2-way comparison."""
    from src.evaluation import plots
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    results = _triarm_results()
    # remove the baseline arm from EVERY data section the renderers read
    del results["scoring_based_assessment"]["Baseline LLM"]
    for tier in results["scoring_by_difficulty_tier"].values():
        del tier["Baseline LLM"]
    results["retrieval_latency_seconds"].pop("Baseline LLM", None)
    results["citation_precision"].pop("Baseline LLM", None)
    for renderer in (plots.plot_dimension_scores, plots.plot_difficulty_tiers,
                     plots.plot_latency_performance, plots.plot_citation_grounding):
        try:
            renderer(results)
        except ValueError as exc:
            assert "Core-arm protocol" in str(exc), renderer.__name__
        else:
            raise AssertionError(
                f"{renderer.__name__} rendered a 2-way comparison")


def _current_results():
    """Current-schema fixture: new labels, win/tie/loss stacks, latency breakdown."""
    from src.config import ARM_LABELS, BASELINE_LABEL
    dims = ["comprehensiveness", "diversity", "empowerment", "logical",
            "readability", "overall_score"]
    crit = ["comprehensiveness", "empowerment", "accuracy", "relevance",
            "coherence", "clarity", "logical", "flexibility"]
    systems = [ARM_LABELS[a] for a in ("closed_book", "bm25", "dense", "hybrid", "hcad")]

    def ms(m, s=0.1):
        return {"n": 10, "mean": m, "std": s, "median": m}

    def outcome(w, t):
        n = 20
        return {"n": n, "wins": int(w * n / 100), "ties": int(t * n / 100),
                "losses": n - int(w * n / 100) - int(t * n / 100),
                "win_pct": w, "tie_pct": t, "loss_pct": 100.0 - w - t}
    return {
        "scoring_based_assessment": {s: {d: 70 + i for d in dims} for i, s in enumerate(systems)},
        "scoring_by_difficulty_tier": {"1": {s: 70.0 for s in systems}},
        "selection_based_assessment": {
            f"hcad_vs_{a}": {"system_a": ARM_LABELS["hcad"], "system_b": ARM_LABELS[a], "n": 20,
                             "overall": outcome(40.0, 35.0),
                             "criteria": {c: outcome(30.0, 50.0) for c in crit},
                             "decisive_win_rate_pct": 61.5}
            for a in ("closed_book", "bm25", "dense", "hybrid")
        },
        "retrieval_latency_seconds": {s: 0.2 for s in systems},
        "latency_breakdown": {
            s: {"retrieval_seconds_per_question": ms(0.0 if s == BASELINE_LABEL else 0.3),
                "generation_seconds": ms(4.0)} for s in systems},
        "citation_precision": {s: {"n_answers_with_citations": 5, "n_answers": 10,
                                   "total_citations": 9, "mean_precision": 0.9}
                               for s in systems},
    }


def test_current_schema_renders_all_arms(tmp_path, monkeypatch):
    from src.evaluation import plots
    monkeypatch.setattr(plots, "FIGURES_DIR", tmp_path)
    out = plots.plot_all(_current_results(), hallu_path=None)
    names = {Path(p).name for p in out}
    assert {"dimension_scores.png", "win_rates.png", "latency_breakdown.png",
            "latency_performance.png", "citation_grounding.png"} <= names


def test_legacy_labels_are_renamed_on_load():
    """The retired generic 'Baseline LLM' label never reaches a figure."""
    from src.config import BASELINE_LABEL
    from src.evaluation import plots
    norm = plots.normalize_results(_triarm_results())
    assert "Baseline LLM" not in norm["scoring_based_assessment"]
    assert BASELINE_LABEL in norm["scoring_based_assessment"]
    # legacy criteria win% (ties hidden) are NOT re-plotted as full stacks
    for entry in norm["selection_based_assessment"].values():
        ov = entry["overall"]
        assert abs(ov["win_pct"] + ov["tie_pct"] + ov["loss_pct"] - 100.0) < 1e-9
        assert entry["criteria"] is None


def test_win_rate_figure_has_no_ghost_series():
    """Figure 4 rows == comparisons present in the JSON, nothing else."""
    from src.evaluation import plots
    res = _current_results()
    rows = plots.win_rate_rows(res)
    assert {r["comparison"] for r in rows} == set(res["selection_based_assessment"])
    assert not any("ablation" in r["comparison"].lower() for r in rows)
