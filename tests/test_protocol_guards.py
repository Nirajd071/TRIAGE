"""Protocol guards: GO/STOP thresholds and the question-cache pool-drift guard."""

import pytest


def test_go_stop_defaults_match_review_protocol(monkeypatch):
    monkeypatch.delenv("HCAD_GO_DELTA_RECALL", raising=False)
    monkeypatch.delenv("HCAD_GO_DELTA_NDCG", raising=False)
    import importlib
    import src.evaluation.dev_pool_retrieval as mod

    importlib.reload(mod)
    assert mod.GO_DELTA_RECALL == pytest.approx(0.08)
    assert mod.GO_DELTA_NDCG == pytest.approx(0.06)


def test_go_stop_env_override(monkeypatch):
    monkeypatch.setenv("HCAD_GO_DELTA_RECALL", "0.5")
    import importlib
    import src.evaluation.dev_pool_retrieval as mod

    importlib.reload(mod)
    assert mod.GO_DELTA_RECALL == pytest.approx(0.5)
    importlib.reload(mod)


def test_missing_question_cache_fails_loud_when_required(tmp_path, monkeypatch):
    monkeypatch.setenv("HCAD_REQUIRE_QUESTION_CACHE", "1")
    monkeypatch.delenv("HCAD_ALLOW_QUESTION_REGEN", raising=False)
    import src.evaluation.run_base_paper_benchmark as bench

    monkeypatch.setattr(
        bench, "_paper_question_cache_path",
        lambda n, pool: tmp_path / f"nope_n{n}_{pool}.json",
    )
    monkeypatch.setattr(
        bench, "RESULTS_DIR", tmp_path,
    )
    with pytest.raises(RuntimeError, match="question_set_id"):
        bench.generate_paper_protocol_questions(10, pool="test")


def test_missing_question_cache_warns_but_allows_when_explicitly_permitted(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HCAD_ALLOW_QUESTION_REGEN", "1")
    import src.evaluation.run_base_paper_benchmark as bench

    monkeypatch.setattr(
        bench, "_paper_question_cache_path",
        lambda n, pool: tmp_path / f"nope_n{n}_{pool}.json",
    )
    monkeypatch.setattr(bench, "RESULTS_DIR", tmp_path)

    class _Coll:
        def count(self):
            return 10

    class _Gen:
        def chat(self, messages, **_):
            return "What is the contraindication profile of drug X in renal failure?"

    import src.data.build_vector_store as bvs

    monkeypatch.setattr(bvs, "load_vector_store", lambda: _Coll())
    monkeypatch.setattr(
        bench, "_fetch_stratified_chunks",
        lambda collection, n, pool, rng: [
            {"doc": f"chunk {i}", "kind": "primekg", "id": f"c{i}", "relation": "contraindication"}
            for i in range(n)
        ],
    )
    out = bench.generate_paper_protocol_questions(2, generator=_Gen(), pool="test")
    assert len(out) == 2
    assert all(s["question"] for s in out)
