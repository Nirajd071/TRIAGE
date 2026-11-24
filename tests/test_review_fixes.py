"""Tests for the reviewer / advisor fixes.

Covers: statistics (bootstrap, Wilcoxon, t-test, McNemar, Cohen/Fleiss
kappa), the multi-judge ensemble (json_mode, strict clients, no neutral
fallbacks), the paired-polarity benchmark, BM25 / RRF baselines, ablation
toggles, explicit baseline naming, the end-to-end multi-arm runner with
latency disentanglement, and table/figure consistency.
"""
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config import (
    ABLATION_ENV,
    ARM_LABELS,
    BASELINE_LABEL,
    canonical_system_label,
    decision_gate_enabled,
    judge_ensemble_specs,
    ortho_strength,
)
from src.evaluation import stats


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    def test_mcnemar_exact_known_value(self):
        # 7 discordant pairs favour A, 1 favours B: p = 2 * (1 + 8) / 2^8
        a = [True] * 7 + [False] * 1 + [True] * 5
        b = [False] * 7 + [True] * 1 + [True] * 5
        res = stats.mcnemar_exact(a, b)
        assert (res["n01"], res["n10"]) == (7, 1)
        assert res["p_value"] == pytest.approx(18 / 256)

    def test_mcnemar_no_discordance(self):
        assert stats.mcnemar_exact([True, False], [True, False])["p_value"] == 1.0

    def test_cohen_kappa_known_value(self):
        # Classic 2x2 example: agree-yes 20, agree-no 15, disagree 10 + 5.
        r1 = ["y"] * 20 + ["y"] * 5 + ["n"] * 10 + ["n"] * 15
        r2 = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
        k = stats.cohen_kappa(r1, r2)
        assert k["observed_agreement"] == pytest.approx(0.7)
        assert k["kappa"] == pytest.approx(0.4)
        assert k["interpretation"] == "fair"

    def test_cohen_kappa_degenerate_is_undefined_not_crash(self):
        k = stats.cohen_kappa(["Tie"] * 5, ["Tie"] * 5)
        assert k["kappa"] is None and k["degenerate"]

    def test_fleiss_kappa_wikipedia_example(self):
        # Fleiss (1971) / Wikipedia worked example: 10 items, 14 raters, kappa = 0.210
        table = [[0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
                 [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
                 [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]]
        ratings = [[c for c, n in enumerate(row) for _ in range(n)] for row in table]
        assert stats.fleiss_kappa(ratings)["kappa"] == pytest.approx(0.210, abs=1e-3)

    def test_bootstrap_ci_brackets_mean_and_is_deterministic(self):
        x = [60, 72, 81, 55, 90, 77, 68, 74]
        a, b = stats.bootstrap_ci(x), stats.bootstrap_ci(x)
        assert a == b
        assert a["ci_low"] <= a["mean"] <= a["ci_high"]
        assert a["mean"] == pytest.approx(np.mean(x))

    def test_paired_bootstrap_flags_zero_crossing(self):
        a = [80, 70, 75, 72, 79, 71]
        b = [79, 72, 74, 73, 78, 72]
        assert not stats.paired_bootstrap_ci(a, b)["excludes_zero"]
        assert stats.paired_bootstrap_ci([x + 20 for x in a], a)["excludes_zero"]

    def test_wilcoxon_and_t_test(self):
        a = [80, 82, 85, 88, 90, 79, 84, 86, 91, 83]
        b = [x - 5 for x in a]
        b[0] += 2
        w = stats.wilcoxon_signed_rank(a, b)
        t = stats.paired_t_test(a, b)
        assert w["p_value"] < 0.01 and w["significant_at_05"]
        assert t["p_value"] < 0.01 and t["df"] == 9
        same = stats.wilcoxon_signed_rank(a, a)
        assert same["p_value"] == 1.0


# ---------------------------------------------------------------------------
# Config: naming, judges, ablation toggles
# ---------------------------------------------------------------------------

class TestConfig:
    def test_explicit_baseline_name(self):
        assert BASELINE_LABEL == "Llama-3.1-8B-Instruct (Zero-Retrieval / Closed-Book)"
        assert canonical_system_label("Baseline LLM") == BASELINE_LABEL
        assert canonical_system_label("Vanilla RAG") == ARM_LABELS["dense"]
        assert "Baseline LLM" not in ARM_LABELS.values()

    def test_default_judge_ensemble(self, monkeypatch):
        monkeypatch.delenv("JUDGE_ENSEMBLE", raising=False)
        monkeypatch.delenv("JUDGE_ENABLE_API_JUDGE", raising=False)
        monkeypatch.delenv("JUDGE_MODEL_INDEX", raising=False)
        assert judge_ensemble_specs() == [("ollama", "gemma2:9b-instruct-q4_K_M"),
                                          ("ollama", "qwen2.5:14b-instruct-q4_K_M")]

    def test_optional_api_judge(self, monkeypatch):
        monkeypatch.delenv("JUDGE_ENSEMBLE", raising=False)
        monkeypatch.delenv("JUDGE_ENABLE_GEMINI_JUDGE", raising=False)
        monkeypatch.delenv("JUDGE_ENABLE_GROQ_JUDGE", raising=False)
        monkeypatch.setenv("JUDGE_ENABLE_API_JUDGE", "1")
        assert judge_ensemble_specs()[-1] == ("nvidia", "meta/llama-3.3-70b-instruct")

    def test_invalid_judge_spec(self, monkeypatch):
        monkeypatch.setenv("JUDGE_ENSEMBLE", "gemma2")
        with pytest.raises(ValueError):
            judge_ensemble_specs()

    def test_ablation_toggles(self, monkeypatch):
        monkeypatch.delenv("HCAD_ORTHO_STRENGTH", raising=False)
        monkeypatch.delenv("HCAD_DECISION_GATE", raising=False)
        assert ortho_strength() == pytest.approx(0.70)
        assert decision_gate_enabled()
        for key, value in ABLATION_ENV["hcad_no_gs"].items():
            monkeypatch.setenv(key, value)
        assert ortho_strength() == 0.0
        for key, value in ABLATION_ENV["hcad_no_gate"].items():
            monkeypatch.setenv(key, value)
        assert not decision_gate_enabled()
        assert ABLATION_ENV["hcad_no_decoupling"] == {"HCAD_DENSE_WEIGHT": "1.0"}


# ---------------------------------------------------------------------------
# Multi-judge evaluator
# ---------------------------------------------------------------------------

def _judge(name, score_json=None, winner="Answer 1", fail=False):
    from src.evaluation.base_paper_evaluator import BasePaperEvaluator

    client = MagicMock()
    client.provider, client.model = "fake", name

    def chat(messages, temperature=None, json_mode=False, **_):
        assert json_mode is True, "judge calls must use json_mode=True"
        if fail:
            return "not json at all"
        if "eight criteria" in messages[0]["content"]:
            crit = {c: {"winner": winner, "reason": "r"} for c in BasePaperEvaluator.CRITERIA_LIST}
            return json.dumps({**crit, "overall_winner": winner, "summary": "s"})
        return json.dumps(score_json)

    client.chat.side_effect = chat
    return BasePaperEvaluator(judge_llm=client, name=name)


SCORES_A = {"comprehensiveness": 80, "diversity": 70, "empowerment": 90, "logical": 85,
            "readability": 75, "rationale": "x"}
SCORES_B = {"comprehensiveness": 60, "diversity": 50, "empowerment": 70, "logical": 65,
            "readability": 55, "rationale": "y"}


class TestMultiJudge:
    def test_consensus_is_mean_over_valid_judges(self):
        from src.evaluation.base_paper_evaluator import MultiJudgeEvaluator

        mj = MultiJudgeEvaluator([_judge("j1", SCORES_A), _judge("j2", SCORES_B),
                                  _judge("j3", fail=True)])
        res = mj.evaluate_scoring("q", "a", "ref", retries=0)
        assert res["n_valid_judges"] == 2
        assert res["per_judge"]["j3"] is None          # recorded missing, NOT 50.0
        assert res["consensus"]["overall_score"] == pytest.approx((80.0 + 60.0) / 2)

    def test_all_judges_failing_gives_no_consensus(self):
        from src.evaluation.base_paper_evaluator import MultiJudgeEvaluator

        mj = MultiJudgeEvaluator([_judge("j1", fail=True)])
        assert mj.evaluate_scoring("q", "a", "ref", retries=0)["consensus"] is None

    def test_selection_counterbalanced_and_voted(self):
        from src.evaluation.base_paper_evaluator import MultiJudgeEvaluator

        # "Answer 1" judges are pure position-bias: they pick whichever answer
        # is shown first, so AB and BA disagree -> Tie for that judge.
        biased = _judge("biased", winner="Answer 1")
        mj = MultiJudgeEvaluator([biased])
        res = mj.evaluate_selection("q", "A", "B", "ref", label_a="HCAD-RAG", label_b="Dense")
        assert res["consensus"]["overall_winner"] == "Tie"
        assert res["per_judge"]["biased"]["order_consistent"] is False

    def test_majority_vote(self):
        from src.evaluation.base_paper_evaluator import majority_vote

        assert majority_vote(["A", "A", "B"]) == "A"
        assert majority_vote(["A", "B"]) == "Tie"
        assert majority_vote([]) == "Tie"

    def test_strict_client_refuses_silent_provider_swap(self, monkeypatch):
        import src.config as cfg
        from src.core.llm_client import UnifiedLLMClient

        monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
        monkeypatch.setattr(cfg, "NVIDIA_API_KEY", "")
        with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
            UnifiedLLMClient(model="meta/llama-3.3-70b-instruct", provider="nvidia", strict=True)

    def test_chat_with_stats_reads_ollama_counters(self, monkeypatch):
        import src.core.llm_client as mod
        from src.core.llm_client import UnifiedLLMClient

        resp = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "message": {"content": "answer"}, "prompt_eval_count": 120, "eval_count": 50,
            "prompt_eval_duration": 0.5e9, "eval_duration": 2.0e9})
        monkeypatch.setattr(mod.requests, "post", lambda *a, **k: resp)
        client = UnifiedLLMClient(model="gemma2:9b", provider="ollama")
        text, st = client.chat_with_stats([{"role": "user", "content": "hi"}])
        assert text == "answer"
        assert (st["prompt_tokens"], st["completion_tokens"]) == (120, 50)
        assert st["decode_s"] == pytest.approx(2.0)
        assert st["tokens_per_second"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# BM25 / hybrid baselines
# ---------------------------------------------------------------------------

class TestSparseBaselines:
    DOCS = ["Metformin is indicated for type 2 diabetes.",
            "Metformin is contraindicated in severe renal impairment.",
            "Aspirin inhibits platelet aggregation."]

    def test_bm25_ranks_lexical_match_first(self):
        from src.baselines.sparse_retrievers import BM25Index

        idx = BM25Index(self.DOCS, ["ind", "contra", "asp"])
        hits = idx.search("metformin contraindicated renal", top_k=2)
        assert hits[0]["id"] == "contra"
        assert set(hits[0]) == {"id", "text", "score", "metadata"}

    def test_bm25_keeps_polarity_words(self):
        from src.baselines.sparse_retrievers import bm25_tokenize

        assert {"not", "contraindicated"} <= set(bm25_tokenize("is not contraindicated"))

    def test_rrf_fusion(self):
        from src.baselines.sparse_retrievers import reciprocal_rank_fusion

        a = [{"id": "x"}, {"id": "y"}, {"id": "z"}]
        b = [{"id": "y"}, {"id": "x"}, {"id": "w"}]
        fused = reciprocal_rank_fusion([a, b], top_k=2)
        assert [d["id"] for d in fused] == ["x", "y"]  # tie broken by first appearance


# ---------------------------------------------------------------------------
# Paired-polarity benchmark
# ---------------------------------------------------------------------------

class TestPolarityBenchmark:
    def test_benchmark_size_and_structure(self):
        from src.evaluation import polarity_benchmark as pb

        assert len(pb.PAIRS) >= 30
        corpus, queries = pb.build_corpus(), pb.build_queries()
        assert len(corpus) == 2 * len(pb.PAIRS)
        assert len(queries) == 4 * len(pb.PAIRS)
        ids = {d["id"] for d in corpus}
        assert all(q["gold_id"] in ids and q["opposite_id"] in ids for q in queries)
        assert {p.polarity_type for p in pb.PAIRS} == {"contraindication", "caution"}
        # paraphrase queries must not contain the evidence polarity keyword
        for q in queries:
            if q["phrasing"] == "paraphrase":
                assert "contraindicated" not in q["query"].lower()
                assert "indicated" not in q["query"].lower()

    def test_score_query(self):
        from src.evaluation.polarity_benchmark import score_query

        ok = score_query(["g", "o", "x"], "g", "o", k=1)
        assert ok["recall_at_k"] == 1 and ok["polarity_error"] == 0
        bad = score_query(["o", "x", "g"], "g", "o", k=2)
        assert bad["recall_at_k"] == 0 and bad["polarity_error"] == 1
        assert bad["reciprocal_rank"] == pytest.approx(1 / 3)

    def test_oracle_and_adversarial_rankers(self):
        from src.evaluation import polarity_benchmark as pb

        queries = pb.build_queries()
        by_text = {q["query"]: q for q in queries}
        oracle = lambda text: [by_text[text]["gold_id"], by_text[text]["opposite_id"]]
        flipped = lambda text: [by_text[text]["opposite_id"], by_text[text]["gold_id"]]
        res = pb.evaluate_rankers({"hcad": oracle, "flipped": flipped}, queries, k=1)
        assert res["arms"]["hcad"]["overall"]["polarity_error_rate"]["mean"] == 0.0
        assert res["arms"]["flipped"]["overall"]["polarity_error_rate"]["mean"] == 1.0
        assert res["arms"]["hcad"]["pair_accuracy"]["mean"] == 1.0
        sig = res["significance_mcnemar_polarity_error"]["hcad_vs_flipped"]
        assert sig["p_value"] < 1e-6

    def test_bm25_and_hybrid_run_on_corpus(self, tmp_path):
        from src.evaluation import polarity_benchmark as pb

        rng = np.random.default_rng(0)
        vocab = {}

        def fake_encode(texts):
            # bag-of-words hashing encoder (deterministic, normalised)
            out = np.zeros((len(texts), 64), dtype=np.float32)
            for i, t in enumerate(texts):
                for w in t.lower().split():
                    out[i, vocab.setdefault(w, rng.integers(0, 64))] += 1.0
            return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)

        payload = pb.run_polarity_benchmark(arms=("bm25", "dense", "hybrid"), k=5,
                                            output_path=tmp_path / "pol.json",
                                            encoder=fake_encode)
        assert set(payload["arms"]) == {"bm25", "dense", "hybrid"}
        assert (tmp_path / "pol.json").exists()
        assert (tmp_path / "pol_per_query.jsonl").exists()
        o = payload["arms"]["bm25"]["overall"]
        assert 0.0 <= o["polarity_error_rate"]["mean"] <= 1.0
        assert o["n_queries"] == 4 * len(pb.PAIRS)

    def test_hcad_ranker_uses_pre_gate_scores_over_whole_corpus(self):
        from src.evaluation import polarity_benchmark as pb

        corpus = pb.build_corpus()

        class FakeRetriever:
            def _get_candidate_contexts(self, *a, **k):
                raise AssertionError("must be replaced by the corpus")

            def retrieve(self, query):
                pool = self._get_candidate_contexts(query, {})
                self._last_candidate_pool = pool
                # score: prefer statements whose polarity word matches the query
                want = "contraindicated" if "contraindicated" in query else "indicated as"
                self._last_phi_scores = np.array(
                    [1.0 if want in c["document"] else 0.0 for c in pool])
                return pool[:1]

        rank = pb.make_hcad_ranker(corpus, FakeRetriever())
        ranking = rank("In which condition is Metformin contraindicated?")
        assert len(ranking) == len(corpus)


# ---------------------------------------------------------------------------
# End-to-end runner + figures/tables consistency
# ---------------------------------------------------------------------------

def _fake_arms():
    from src.evaluation.run_base_paper_benchmark import BenchmarkArm

    def retr(tag):
        return lambda q: [{"id": f"{tag}1", "text": f"{tag} evidence for {q}. More.",
                           "document": f"{tag} evidence for {q}. More."}]

    msgs = lambda q, ctx: [{"role": "user", "content": q}]
    arms = {
        "closed_book": BenchmarkArm("closed_book", ARM_LABELS["closed_book"], None, msgs),
        "bm25": BenchmarkArm("bm25", ARM_LABELS["bm25"], retr("bm25"), msgs),
        "dense": BenchmarkArm("dense", ARM_LABELS["dense"], retr("dense"), msgs),
        "hybrid": BenchmarkArm("hybrid", ARM_LABELS["hybrid"], retr("hybrid"), msgs),
        "hcad": BenchmarkArm("hcad", ARM_LABELS["hcad"], retr("hcad"), msgs,
                             stage_timings=lambda: {"candidate_pool": 0.01,
                                                    "decision_gate": 0.002}),
        "hcad_no_gate": BenchmarkArm("hcad_no_gate", ARM_LABELS["hcad_no_gate"], retr("nogate"),
                                     msgs, env={"HCAD_DECISION_GATE": "0"}),
    }
    return arms


def _scripted_judge():
    """Two judges: scores depend on the answer text; HCAD answers win."""
    from src.evaluation.base_paper_evaluator import BasePaperEvaluator, MultiJudgeEvaluator

    def make(name, offset):
        client = MagicMock()
        client.provider, client.model = "fake", name

        def chat(messages, temperature=None, json_mode=False, **_):
            assert json_mode
            user = messages[1]["content"]
            if "eight criteria" in messages[0]["content"]:
                a1 = user.split("\nAnswer 1:")[1].split("Retrieved Evidence")[0]
                w = "Answer 1" if "HCADANS" in a1 else "Answer 2"
                if "HCADANS" not in user:
                    w = "Tie"
                crit = {c: {"winner": w} for c in BasePaperEvaluator.CRITERIA_LIST}
                return json.dumps({**crit, "overall_winner": w})
            ans = user.split("System Answer:")[1]
            base = 85 if "HCADANS" in ans else 66
            s = base + offset
            return json.dumps({d: s for d in ("comprehensiveness", "diversity", "empowerment",
                                              "logical", "readability")})
        client.chat.side_effect = chat
        return BasePaperEvaluator(judge_llm=client, name=name)

    return MultiJudgeEvaluator([make("gemma2:9b", 0), make("qwen2.5:14b", 3)])


def test_end_to_end_runner_and_consistent_figures(tmp_path, monkeypatch):
    import src.evaluation.run_base_paper_benchmark as runner
    import src.wandb_tracking as wt
    from src.evaluation.generate_figures import generate_all, verify_consistency

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(wt, "WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_CHECKPOINT_EVERY", "0")
    monkeypatch.delenv("HCAD_DECISION_GATE", raising=False)
    seen_env = []

    def fake_generate(arm, q, contexts):
        seen_env.append((arm.arm_id, os.environ.get("HCAD_DECISION_GATE")))
        text = ("HCADANS " if arm.arm_id == "hcad" else "other ") + q + " [Doc 1]"
        return text, {"t_generation": 1.5, "prompt_tokens": 100, "completion_tokens": 60,
                      "prefill_s": 0.2, "decode_s": 1.2, "tokens_per_second": 50.0}

    monkeypatch.setattr(runner, "generate_answer", fake_generate)
    # env overrides are applied during retrieval only
    orig_timed = runner.timed_retrieval

    def spy_timed(arm, q, runs):
        if arm.arm_id == "hcad_no_gate":
            arm.retrieve = (lambda f: (lambda qq: (seen_env.append(
                ("retrieve_no_gate", os.environ.get("HCAD_DECISION_GATE"))), f(qq))[1]))(arm.retrieve)
        return orig_timed(arm, q, runs)
    monkeypatch.setattr(runner, "timed_retrieval", spy_timed)

    samples = [{"question": f"Clinical question number {i}?", "reference": f"ref {i}",
                "source": "paper_protocol", "difficulty_tier": 1 + i % 3, "pool": "test",
                "source_chunk_id": str(i)} for i in range(6)]
    arms = list(_fake_arms())
    results = runner.run_benchmark(num_samples=6, arms=arms, include_ablations=False,
                                   judge=_scripted_judge(), latency_runs=2,
                                   built_arms=_fake_arms(), samples=samples)

    # naming
    assert BASELINE_LABEL in results["scoring_based_assessment"]
    assert "Baseline LLM" not in json.dumps(results)
    # win/tie/loss consistency
    for key, entry in results["selection_based_assessment"].items():
        ov = entry["overall"]
        assert ov["win_pct"] + ov["tie_pct"] + ov["loss_pct"] == pytest.approx(100.0)
        for c in entry["criteria"].values():
            assert c["win_pct"] + c["tie_pct"] + c["loss_pct"] == pytest.approx(100.0)
    assert results["selection_based_assessment"]["hcad_vs_dense"]["overall"]["win_pct"] == 100.0
    assert "dense_vs_closed_book" in results["selection_based_assessment"]
    # statistics present for every HCAD comparison, consensus + per judge
    comp = results["statistical_tests"]["comparisons"]["hcad_vs_dense"]
    assert comp["consensus"]["n_paired"] == 6
    assert comp["consensus"]["wilcoxon_signed_rank"]["p_value"] < 0.05
    assert set(comp["per_judge"]) == {"gemma2:9b", "qwen2.5:14b"}
    assert comp["consensus"]["paired_difference_ci"]["excludes_zero"]
    agr = results["inter_judge_agreement"]
    assert "gemma2:9b | qwen2.5:14b" in agr["pairwise"]
    assert agr["fleiss"]["selection_overall_winner"]["n_raters"] == 2
    # latency disentangled
    lat = json.loads((tmp_path / "latency_breakdown.json").read_text())["latency_breakdown"]
    hcad_lat = lat["HCAD-RAG"]
    assert hcad_lat["generation_seconds"]["mean"] == pytest.approx(1.5)
    assert hcad_lat["retrieval_seconds_all_runs"]["n"] == 12   # 6 questions x 2 warm runs
    assert set(hcad_lat["retrieval_stage_seconds"]) == {"candidate_pool", "decision_gate"}
    assert lat[BASELINE_LABEL]["retrieval_seconds_per_question"]["mean"] == 0.0
    assert hcad_lat["tokens_per_second"]["mean"] == pytest.approx(50.0)
    # ablation env applied only around that arm's retrieval
    assert ("retrieve_no_gate", "0") in seen_env
    assert all(v is None for a, v in seen_env if a != "retrieve_no_gate")
    # checkpoint is schema 2 with per-judge records
    rec = json.loads((tmp_path / "base_paper_checkpoint.jsonl").read_text().splitlines()[0])
    assert rec["schema"] == 2 and set(rec["judges"]) == {"gemma2:9b", "qwen2.5:14b"}

    # tables + figures from the saved JSON are mutually consistent
    figs, tabs = tmp_path / "figures", tmp_path / "tables"
    manifest = generate_all(tmp_path / "base_paper_benchmark_results.json",
                            figures_dir=figs, tables_dir=tabs)
    assert manifest["consistency_problems"] == []
    assert (figs / "methodology_pipeline.png").exists()
    assert (figs / "win_rates.png").exists()
    assert (tabs / "table4_win_rates.csv").exists()
    assert (tabs / "table7_latency.csv").exists()
    series = {s["comparison"] for s in manifest["figure_series"]["win_rates"]}
    assert series == set(results["selection_based_assessment"])

    # tampering with the table is detected
    csv_path = tabs / "table4_win_rates.csv"
    csv_path.write_text(csv_path.read_text().replace("100.0", "50.0", 1))
    assert verify_consistency(json.loads((tmp_path / "base_paper_benchmark_results.json")
                                         .read_text()), tabs)


def test_legacy_checkpoint_records_are_normalised(tmp_path):
    from src.evaluation.run_base_paper_benchmark import _records_from_checkpoint

    legacy = {"question": "q", "scores": {"Baseline LLM": {"overall_score": 40.0},
                                          "Vanilla RAG": {"overall_score": 60.0}},
              "selection": {"HCAD_vs_Baseline": {"overall_winner": "Baseline LLM",
                                                 "criteria_winners": {}}},
              "latency": {"Vanilla RAG": 0.4}}
    ckpt = tmp_path / "c.jsonl"
    ckpt.write_text(json.dumps(legacy) + "\n")
    rec = _records_from_checkpoint(ckpt, {"q"})
    assert rec["scoring"][BASELINE_LABEL][0]["overall_score"] == 40.0
    sel = rec["selection"]["hcad_vs_closed_book"][0]
    assert sel["consensus"]["overall_winner"] == BASELINE_LABEL
    assert rec["latency"][ARM_LABELS["dense"]][0]["t_retrieval"] == 0.4


def test_pipeline_stage_keys_match_retriever_timings():
    """The 5-stage diagram, the latency breakdown and the retriever share keys."""
    import inspect

    from src.core import hcad_retriever
    from src.core.pipeline_stages import PIPELINE_STAGES, STAGE_TIMING_KEYS

    assert len(PIPELINE_STAGES) == 5
    src = inspect.getsource(hcad_retriever.HCADRetriever.retrieve)
    for key in STAGE_TIMING_KEYS:
        assert f'timings["{key}"]' in src, key
    assert "decision_gate_enabled()" in src


def test_store_mode_polarity_labels_by_kg_relation():
    from src.evaluation import polarity_benchmark as pb

    def kg(drug, rel):
        return {"metadata": {"head_name": drug, "relation": rel, "tail_name": "x"}}

    def good(query):
        q = next(x for x in pb.build_queries() if x["query"] == query)
        first = "indication" if q["target"] == "indication" else "contraindication"
        second = "contraindication" if first == "indication" else "indication"
        return [kg(q["drug"], first), kg(q["drug"], second)]

    res = pb.evaluate_store_retrievers({"hcad": good}, pb.build_queries(), k=5)
    o = res["hcad"]["overall"]
    assert o["polarity_error_rate"]["mean"] == 0.0
    assert o["polarity_recall_at_k"]["mean"] == 1.0
    # a context about a different drug never counts
    assert pb.context_polarity(kg("Aspirin", "indication"), "Metformin") is None


def test_variant_runs_write_their_own_latency_file(tmp_path, monkeypatch):
    import src.evaluation.run_base_paper_benchmark as runner
    import src.wandb_tracking as wt

    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(wt, "WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_CHECKPOINT_EVERY", "0")
    monkeypatch.setattr(runner, "generate_answer", lambda arm, q, c: (
        ("HCADANS " if arm.arm_id == "hcad" else "x ") + q,
        {"t_generation": 1.0, "prompt_tokens": 1, "completion_tokens": 1,
         "tokens_per_second": 1.0}))
    arms = {a: v for a, v in _fake_arms().items() if a in ("closed_book", "dense", "hcad")}
    samples = [{"question": "Q one?", "reference": "r", "source": "curated"}]
    runner.run_benchmark(num_samples=1, arms=list(arms), judge=_scripted_judge(),
                         latency_runs=1, built_arms=arms, samples=samples,
                         output_path=tmp_path / "base_paper_benchmark_results_learned.json")
    assert (tmp_path / "latency_breakdown_learned.json").exists()
    assert not (tmp_path / "latency_breakdown.json").exists()


def test_cli_exposes_revision_commands():
    from src.main import build_parser

    p = build_parser()
    a = p.parse_args(["polarity", "--mode", "store"])
    assert a.mode == "store"
    b = p.parse_args(["base-paper", "--samples", "5", "--ablations", "--arms", "bm25,hcad"])
    assert b.ablations and b.arms == "bm25,hcad"
    assert p.parse_args(["medhallu", "--system", "hcad"]).system == "hcad"
    assert p.parse_args(["figures"]).func.__name__ == "cmd_figures"


def test_resume_reruns_legacy_or_incompatible_records(tmp_path):
    from src.evaluation.run_base_paper_benchmark import (
        _completed_questions_from_checkpoint, _hcad_config_id)

    ckpt = tmp_path / "c.jsonl"
    base = {"question_pool": "test", "question_set_id": "s",
            "hcad_config_id": _hcad_config_id()}
    rows = [
        {**base, "question": "legacy"},                                   # schema 1
        {**base, "question": "ok", "schema": 2,
         "arms": {"closed_book": "", "dense": "", "hcad": ""}, "judges": ["j1", "j2"]},
        {"question_pool": "test", "question_set_id": "s",                 # old evidence logic
         "question": "old_hcad_config", "schema": 2,
         "arms": {"closed_book": "", "dense": "", "hcad": ""}, "judges": ["j1", "j2"]},
        {**base, "question": "fewer_arms", "schema": 2,
         "arms": {"hcad": ""}, "judges": ["j1", "j2"]},
        {**base, "question": "other_judges", "schema": 2,
         "arms": {"closed_book": "", "dense": "", "hcad": ""}, "judges": ["j1"]},
    ]
    ckpt.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    done = _completed_questions_from_checkpoint(
        ckpt, "test", "s", required_arms=["closed_book", "dense", "hcad"],
        required_judges=["j1", "j2"])
    assert done == {"ok"}
    # legacy call signature keeps its old behaviour
    assert len(_completed_questions_from_checkpoint(ckpt, "test", "s")) == 5
