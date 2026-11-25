"""Unit tests for Meta-Review gap fixes:
- Non-constant graph-degree IDF in AnisotropicTransformer
- Cross-Encoder Reranker (MedCPT) and source-matched GraphRAG baselines
- NevIR, ExcluIR, and polarity rel_prefix/reranker evaluators
"""
from __future__ import annotations

import numpy as np
import networkx as nx
from unittest.mock import MagicMock


def test_anisotropic_idf_is_not_constant_across_degrees():
    from src.core.anisotropic_transformer import AnisotropicTransformer

    g = nx.MultiDiGraph()
    g.add_node("hub")
    g.add_node("rare")
    for i in range(40):
        g.add_node(f"n{i}")
        g.add_edge("hub", f"n{i}", relation="rel", weight=0.8)
    g.add_edge("rare", "n0", relation="rel", weight=0.8)

    tr = AnisotropicTransformer()
    idf_hub = tr._compute_idf("hub", None, g)
    idf_rare = tr._compute_idf("rare", None, g)
    assert idf_rare > idf_hub, f"Expected rare IDF ({idf_rare}) > hub IDF ({idf_hub})"


def test_cross_encoder_rerank_rag_orders_by_rerank_score():
    from src.baselines.cross_encoder_rag import CrossEncoderRerankRAG

    mock_ce = MagicMock()
    mock_ce.score_pairs.return_value = np.array([0.1, 2.5, -0.4], dtype=np.float32)

    cands = [
        {"text": "doc0", "score": 0.9},
        {"text": "doc1", "score": 0.5},
        {"text": "doc2", "score": 0.7},
    ]
    rag = CrossEncoderRerankRAG(candidate_fn=lambda q, n: cands, cross_encoder=mock_ce, top_k=2)
    out = rag.retrieve("test query")
    assert [x["text"] for x in out] == ["doc1", "doc0"]
    assert out[0]["rerank_score"] == 2.5


def test_graph_rag_baseline_combines_dense_and_kg():
    from src.baselines.graph_rag import GraphRAGBaseline
    from src.core.hypergraph_builder import HypergraphBuilder

    g = nx.MultiDiGraph()
    g.add_edge("Metformin", "Type 2 Diabetes", relation="indication", weight=0.9)
    g.add_edge("Metformin", "Renal Failure", relation="contraindication", weight=0.9)
    hg = HypergraphBuilder(g)

    ner = MagicMock()
    ner.extract_entities.return_value = {"CHEMICAL": [{"text": "Metformin"}]}
    linker = MagicMock()
    linker.link_entities.return_value = {"CHEMICAL": [{"text": "Metformin", "canonical_name": "Metformin"}]}

    def fake_encode(texts):
        vecs = []
        for t in texts:
            if "contraindicated" in t.lower() or "renal" in t.lower():
                vecs.append([0.9, 0.1])
            else:
                vecs.append([0.1, 0.9])
        return np.asarray(vecs, dtype=np.float32)

    dense_hits = [{"text": "Metformin textbook overview.", "score": 0.8}]
    rag = GraphRAGBaseline(
        dense_retrieve_fn=lambda q, k: dense_hits,
        encode_fn=fake_encode,
        ner=ner,
        entity_linker=linker,
        hypergraph=hg,
        top_k=3,
        dense_keep=1,
    )
    res = rag.retrieve("Is Metformin contraindicated in renal failure?")
    assert len(res) == 3
    assert res[0]["text"] == "Metformin textbook overview."
    assert any(r.get("source") == "primekg_graphrag" for r in res)


def test_nevir_and_excluir_evaluators(tmp_path):
    from src.evaluation.external_benchmarks import evaluate_excluir_pairs, evaluate_nevir_pairs

    def fake_encode(texts):
        out = []
        for t in texts:
            tl = t.lower()
            if any(w in tl for w in ("not ", "avoid", "excluding")):
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        arr = np.asarray(out, dtype=np.float32)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)

    mock_ce = MagicMock()
    mock_ce.score_pairs.side_effect = lambda q, docs: np.array(
        [1.0 if ("not " in q.lower()) == ("not " in d.lower()) else 0.0 for d in docs],
        dtype=np.float32,
    )

    nevir_pairs = [
        {
            "q1": "Which drug treats diabetes?",
            "q2": "Which drug should not be given in renal failure?",
            "doc1": "Metformin treats diabetes.",
            "doc2": "Metformin is not given in renal failure.",
        }
    ]
    nev_res = evaluate_nevir_pairs(
        nevir_pairs,
        arms=("bm25", "dense", "reranker"),
        encoder=fake_encode,
        cross_encoder=mock_ce,
        output_path=tmp_path / "nevir.json",
    )
    assert nev_res["arms"]["reranker"]["pair_accuracy"]["mean"] == 1.0
    assert (tmp_path / "nevir.json").exists()

    excluir_pairs = [
        {
            "query": "Treatments excluding NSAIDs",
            "doc_pos": "Acetaminophen is not an NSAID.",
            "doc_neg": "Ibuprofen is an NSAID.",
        }
    ]
    exc_res = evaluate_excluir_pairs(
        excluir_pairs,
        arms=("bm25", "dense", "reranker"),
        encoder=fake_encode,
        cross_encoder=mock_ce,
        output_path=tmp_path / "excluir.json",
    )
    assert exc_res["arms"]["dense"]["exclusion_accuracy"]["mean"] == 1.0
    assert (tmp_path / "excluir.json").exists()


def test_extended_arms_and_aniso_ablation_config():
    from src.config import ABLATION_ENV, EXTENDED_ARMS
    from src.evaluation.run_base_paper_benchmark import resolve_arms

    assert "reranker" in EXTENDED_ARMS and "graphrag" in EXTENDED_ARMS
    assert "hcad_no_aniso" in ABLATION_ENV
    resolved = resolve_arms(["closed_book", "reranker", "graphrag", "hcad"], include_ablations=True)
    assert "hcad_no_aniso" in resolved
    assert "reranker" in resolved
    assert "graphrag" in resolved


def test_polarity_rel_prefix_and_reranker_arms(tmp_path):
    from src.evaluation.polarity_benchmark import PAIRS, run_polarity_benchmark

    def fake_encode(texts):
        vecs = []
        for t in texts:
            tl = t.lower()
            is_contra = ("contraindic" in tl or "avoid" in tl)
            vecs.append([1.0, 0.0] if is_contra else [0.0, 1.0])
        arr = np.asarray(vecs, dtype=np.float32)
        return arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12)

    mock_ce = MagicMock()
    mock_ce.score_pairs.side_effect = lambda q, docs: np.array(
        [1.0 if ("avoid" in q.lower() or "contraindic" in q.lower()) == ("contraindic" in d.lower() or "avoid" in d.lower()) else 0.0 for d in docs],
        dtype=np.float32,
    )

    res = run_polarity_benchmark(
        arms=("bm25", "dense", "hybrid", "reranker", "rel_prefix"),
        encoder=fake_encode,
        cross_encoder=mock_ce,
        output_path=tmp_path / "pol.json",
    )
    assert len(res["arms"]) == 5
    assert res["n_pairs"] == len(PAIRS)
    assert (tmp_path / "pol.json").exists()


def test_graph_rag_resolves_integer_node_ids_and_neighbor_names():
    from src.baselines.graph_rag import GraphRAGBaseline
    from src.core.hypergraph_builder import HypergraphBuilder

    g = nx.MultiDiGraph()
    g.add_node(10, name="Metformin", node_type="drug")
    g.add_node(20, name="Lactic acidosis", node_type="effect/phenotype")
    g.add_edge(10, 20, relation="contraindication", weight=0.9)
    g.name_to_node = {"Metformin": 10, "Lactic acidosis": 20}
    hg = HypergraphBuilder(g)

    ner = MagicMock()
    ner.extract_entities.return_value = {"CHEMICAL": [{"text": "Metformin"}]}
    linker = MagicMock()
    linker.link_entities.return_value = {"CHEMICAL": [{"text": "Metformin", "canonical_name": "Metformin"}]}

    rag = GraphRAGBaseline(
        dense_retrieve_fn=lambda q, k: [{"text": "Dense overview.", "score": 0.5}],
        encode_fn=lambda texts: np.tile(np.array([1.0, 0.0], dtype=np.float32), (len(texts), 1)),
        ner=ner,
        entity_linker=linker,
        hypergraph=hg,
        top_k=2,
        dense_keep=1,
    )
    out = rag.retrieve("Metformin lactic acidosis risk")
    kg_hits = [r for r in out if r.get("source") == "primekg_graphrag"]
    assert kg_hits
    assert "Metformin" in kg_hits[0]["text"] and "Lactic acidosis" in kg_hits[0]["text"]
    assert "10" not in kg_hits[0]["text"]


def test_mirage_benchmark_uses_env_overrides_cleanly(tmp_path):
    from unittest.mock import patch
    from src.evaluation.external_benchmarks import run_mirage_benchmark
    from src.evaluation.run_base_paper_benchmark import BenchmarkArm

    samples = [
        {"id": "medqa_0", "dataset": "MedQA", "question": "Q?", "retrieval_query": "Q?", "gold_letter": "A"},
    ]
    arm = BenchmarkArm(
        arm_id="hcad",
        label="HCAD-RAG",
        retrieve=lambda q: [{"document": "Answer is A.", "score": 0.9}],
        build_messages=lambda q, ctx: [{"role": "user", "content": q}],
        env={"HCAD_TEST_FLAG": "1"},
    )
    fake_llm = MagicMock()
    fake_llm.chat.return_value = "Final Answer: A"

    with patch("src.evaluation.external_benchmarks._load_mirage_samples", return_value=samples), \
         patch("src.evaluation.run_base_paper_benchmark.build_arms", return_value={"hcad": arm}):
        res = run_mirage_benchmark(
            n_per_dataset=1,
            arms=("hcad",),
            llm_client=fake_llm,
            output_path=tmp_path / "mirage.json",
            checkpoint_path=tmp_path / "mirage_ckpt.jsonl",
        )
    assert res["arms"]["hcad"]["overall_accuracy"]["mean"] == 1.0
    assert (tmp_path / "mirage.json").exists()
