"""Judge-health guards: a judge that cannot parse its rubric must not be
allowed to produce a full benchmark of neutral fallback scores.

Regression for the 2026-11-07 run: judge defaulted to meditron:7b, every
scoring call fell back to 50.0 and every selection call to Tie, and the
run shipped a flat results file that looked like a measurement.
"""

import pytest

from src.evaluation.base_paper_evaluator import BasePaperEvaluator
from src.evaluation.run_base_paper_benchmark import _judge_self_check


class _BrokenJudge:
    """Judge stub whose chat always fails to produce parseable JSON."""

    provider = "ollama"
    model = "meditron:7b"

    def chat(self, messages, temperature=0.0, **_):
        raise RuntimeError("model returned non-JSON garbage")


class _OkJudge:
    provider = "ollama"
    model = "qwen2.5:14b"

    def chat(self, messages, temperature=0.0, **_):
        # minimal parseable scoring rubric
        import json

        return json.dumps({
            "comprehensiveness": 80, "diversity": 70, "empowerment": 75,
            "logical": 85, "readability": 90, "overall_score": 82,
            "rationale": "ok",
        })


def test_self_check_passes_with_parseable_judge(monkeypatch):
    monkeypatch.delenv("HCAD_EXPECTED_JUDGE", raising=False)
    ev = BasePaperEvaluator(judge_llm=_OkJudge())
    _judge_self_check(ev)  # must not raise


def test_self_check_fails_fast_with_broken_judge(monkeypatch):
    monkeypatch.delenv("HCAD_EXPECTED_JUDGE", raising=False)
    ev = BasePaperEvaluator(judge_llm=_BrokenJudge())
    with pytest.raises(RuntimeError, match="Judge self-check failed"):
        _judge_self_check(ev)


def test_self_check_enforces_expected_judge(monkeypatch):
    monkeypatch.setenv("HCAD_EXPECTED_JUDGE", "qwen2.5:14b")
    ev = BasePaperEvaluator(judge_llm=_OkJudge())
    _judge_self_check(ev)  # matches, passes
    monkeypatch.setenv("HCAD_EXPECTED_JUDGE", "qwen2.5:14b")
    ev2 = BasePaperEvaluator(judge_llm=_BrokenJudge())
    with pytest.raises(RuntimeError, match="Judge model mismatch"):
        _judge_self_check(ev2)


def test_fallback_counters_scoring():
    ev = BasePaperEvaluator(judge_llm=_BrokenJudge())
    result = ev.evaluate_scoring(
        "q", "a", "ref", retries=0, fallback_on_error=True
    )
    assert result.overall_score == 50.0
    assert ev.judge_calls == 1
    assert ev.judge_fallbacks == 1
    assert ev.judge_fallback_rate == pytest.approx(1.0)


def test_fallback_counters_selection():
    ev = BasePaperEvaluator(judge_llm=_BrokenJudge())
    result = ev.evaluate_selection(
        "q", "a1", "a2", "ref", label_a="A", label_b="B",
        fallback_on_error=True,
    )
    assert result["overall_winner"] == "Tie"
    assert ev.judge_fallbacks >= 1


def test_fallback_rate_zero_calls():
    ev = BasePaperEvaluator(judge_llm=_OkJudge())
    assert ev.judge_fallback_rate == 0.0


def test_ok_judge_scores_are_counted_not_fallback():
    ev = BasePaperEvaluator(judge_llm=_OkJudge())
    result = ev.evaluate_scoring("q", "a", "ref", retries=0, fallback_on_error=True)
    # overall_score is the mean of the five dimension scores
    assert result.overall_score == pytest.approx(80.0)
    assert ev.judge_calls == 1
    assert ev.judge_fallbacks == 0
    assert ev.judge_fallback_rate == 0.0
