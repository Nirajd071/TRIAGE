"""HCAD evidence assembly: dense anchor, hyperedge grouping, budget packing."""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.decision_gate import DecisionGate, group_kg_facts


def _triple(head, rel, tail, score=0.6, source="graph_hyperedge"):
    from src.data.triple_text import triple_to_sentence
    return {"document": triple_to_sentence(head, rel, tail),
            "metadata": {"head_name": head, "relation": rel, "tail_name": tail},
            "phi_final": score, "source": source}


def _prose(n_words, score=0.5, tag="p"):
    sentence = " ".join(f"{tag}{i}" for i in range(10)) + "."   # 10-word sentences
    words = (sentence + " ") * (n_words // 10)
    return {"document": words.strip(), "metadata": {"source": "pmc"},
            "phi_final": score, "source": "vector"}


def _gate():
    return DecisionGate(use_jev=False, min_relevance_score=0.25,
                        max_evidence_items=8, token_budget_words=350)


def test_triples_sharing_head_and_relation_become_one_hyperedge_line():
    items = [_triple("Metformin", "indication", "type 2 diabetes mellitus", 0.9),
             _triple("Metformin", "contraindication", "lactic acidosis", 0.8),
             _triple("Metformin", "indication", "polycystic ovary syndrome", 0.7),
             _prose(50)]
    out = group_kg_facts(items)
    assert len(out) == 3
    assert out[0]["document"] == (
        "Metformin is indicated for: type 2 diabetes mellitus; polycystic ovary syndrome.")
    assert out[0]["metadata"]["grouped_tails"] == [
        "type 2 diabetes mellitus", "polycystic ovary syndrome"]
    assert len(out[0]["member_documents"]) == 2
    # singleton groups keep their original sentence and carry no member list
    assert out[1]["document"] == "Metformin is contraindicated in lactic acidosis."
    assert "member_documents" not in out[1]
    assert out[2]["source"] == "vector" and out[2]["metadata"] == {"source": "pmc"}


def test_group_caps_tails_and_skips_duplicates():
    items = [_triple("Aspirin", "indication", f"disease {i}") for i in range(8)]
    items.append(_triple("Aspirin", "indication", "DISEASE 0"))
    out = group_kg_facts(items, max_tails=5)
    assert len(out) == 1 and len(out[0]["metadata"]["grouped_tails"]) == 5


def test_relevant_passage_behind_triples_is_clipped_not_dropped():
    # 3 triples (~18 words) then a 400-word passage: the old gate dropped the
    # passage entirely because only the FIRST item could be clipped.
    items = [_triple("Drug A", "indication", f"condition {i}", 0.9 - i / 100)
             for i in range(3)]
    items[1]["metadata"]["head_name"] = "Drug B"   # keep them ungrouped
    items[1]["document"] = "Drug B is indicated for condition 1."
    items[2]["metadata"]["head_name"] = "Drug C"
    items[2]["document"] = "Drug C is indicated for condition 2."
    items.append(_prose(400, score=0.5))
    out = _gate().verify_and_prune_candidates(items, "q")
    assert len(out) == 4
    assert out[-1]["source"] == "vector"
    total = sum(len(c["document"].split()) for c in out)
    assert 300 <= total <= 350
    assert out[-1]["document"].endswith(".")   # sentence-boundary clip


def test_anchor_bypasses_relevance_floor_and_relation_cap():
    anchor = {**_prose(120, score=0.12), "anchor": True}
    low = _prose(30, score=0.12, tag="x")
    out = _gate().verify_and_prune_candidates([anchor, low], "q")
    assert out == [anchor]


def test_anchor_keeps_score_position_but_is_guaranteed_slot_and_budget():
    # 10 high-scoring prose items (40 words each = 400 words, over budget and
    # over the 8-item cap) followed by a low-scoring anchor.
    items = [_prose(40, score=0.9 - i / 100, tag=f"h{i}_") for i in range(10)]
    anchor = {**_prose(200, score=0.3, tag="a"), "anchor": True}
    out = _gate().verify_and_prune_candidates(items + [anchor], "q")
    assert out[0] is items[0]                       # score order kept
    assert out[-1]["document"].startswith("a0")     # anchor admitted last, in place
    assert len(out[-1]["document"].split()) == 200  # with its full reserved budget
    assert sum(len(c["document"].split()) for c in out) <= 350
    assert len(out) <= 8


def test_grouping_toggle_off_restores_one_line_per_triple(monkeypatch):
    monkeypatch.setenv("HCAD_GROUP_FACTS", "0")
    items = [_triple("Metformin", "indication", "A"), _triple("Metformin", "indication", "B")]
    out = _gate().verify_and_prune_candidates(items, "q")
    assert [c["document"] for c in out] == [
        "Metformin is indicated for A.", "Metformin is indicated for B."]


def test_dense_anchor_selects_best_full_text_vector_hit(monkeypatch):
    from src.core.hcad_retriever import HCADRetriever

    monkeypatch.delenv("HCAD_DENSE_ANCHOR_K", raising=False)
    monkeypatch.delenv("HCAD_DENSE_ANCHOR_MIN", raising=False)

    cands = [
        {"document": "triple", "source": "graph_hyperedge"},
        {"document": "weak passage", "source": "vector"},
        {"document": "gold passage", "source": "vector"},
        {"document": "also good passage", "source": "vector"},
    ]
    dense = np.array([0.95, 0.30, 0.82, 0.70], dtype=np.float32)
    phi = np.array([0.7, 0.2, 0.31, 0.3], dtype=np.float32)
    anchors = HCADRetriever._dense_anchors(cands, dense, phi)
    # graph candidates never anchor, even with a high cosine
    assert [a["document"] for a in anchors] == ["gold passage"]
    assert anchors[0]["anchor"] is True and abs(anchors[0]["phi_final"] - 0.31) < 1e-6

    monkeypatch.setenv("HCAD_DENSE_ANCHOR_K", "2")
    assert len(HCADRetriever._dense_anchors(cands, dense, phi)) == 2
    monkeypatch.setenv("HCAD_DENSE_ANCHOR_MIN", "0.9")
    assert HCADRetriever._dense_anchors(cands, dense, phi) == []
    monkeypatch.setenv("HCAD_DENSE_ANCHOR_K", "0")
    monkeypatch.setenv("HCAD_DENSE_ANCHOR_MIN", "0.0")
    assert HCADRetriever._dense_anchors(cands, dense, phi) == []
    # dense channel off (HCAD_DENSE_WEIGHT=0 -> all-zero dense) -> no anchor
    monkeypatch.setenv("HCAD_DENSE_ANCHOR_K", "1")
    assert HCADRetriever._dense_anchors(cands, np.zeros(4), phi) == []


def test_grouped_line_still_matches_gold_triple():
    from src.evaluation.run_base_paper_benchmark import _evidence_stats, _ranked_relevance

    items = group_kg_facts([_triple("Metformin", "indication", t) for t in
                            ("type 2 diabetes mellitus", "polycystic ovary syndrome",
                             "prediabetes", "gestational diabetes", "obesity")])
    ref = "Metformin is indicated for polycystic ovary syndrome."
    rel = _ranked_relevance(ref, items)
    assert rel == [1]
    stats = _evidence_stats(items, rel)
    assert stats["n_kg"] == 1 and stats["n_kg_facts"] == 5 and stats["gold_rank"] == 1


def test_changed_hcad_config_is_not_resumed(tmp_path, monkeypatch):
    import json
    from src.evaluation.run_base_paper_benchmark import (
        _completed_questions_from_checkpoint, _hcad_config_id)

    ckpt = tmp_path / "c.jsonl"
    ckpt.write_text(json.dumps({"schema": 2, "question": "q1", "question_pool": "test",
                                "question_set_id": "s", "arms": {"hcad": "HCAD-RAG"},
                                "judges": ["j"],
                                "hcad_config_id": _hcad_config_id()}), encoding="utf-8")
    kw = {"required_arms": ["hcad"], "required_judges": ["j"]}
    assert _completed_questions_from_checkpoint(ckpt, "test", "s", **kw) == {"q1"}
    monkeypatch.setenv("HCAD_DENSE_ANCHOR_K", "0")
    assert _completed_questions_from_checkpoint(ckpt, "test", "s", **kw) == set()
