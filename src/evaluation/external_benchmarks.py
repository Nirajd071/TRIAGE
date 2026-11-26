"""External polarity, exclusion, and MIRAGE-style biomedical QA benchmarks.

Addresses reviewer requests for:
1. NevIR (orionweller/NevIR): pairwise negation-in-retrieval evaluation.
2. ExcluIR (joey234/ExcluIR): pairwise exclusion-retrieval evaluation.
3. MIRAGE-style QA accuracy (MedQA-USMLE + PubMedQA pqa_labeled): closed-book
   vs BM25 vs Dense vs Hybrid vs Cross-Encoder Reranker vs GraphRAG vs HCAD-RAG.
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.config import LLM_MODEL, RANDOM_SEED, RESULTS_DIR
from src.evaluation.stats import bootstrap_ci, mcnemar_exact

logger = logging.getLogger(__name__)

_ABSTENTION_PAT = re.compile(
    r"\b(?:insufficient evidence|cannot determine|not enough information|"
    r"abstain|no supporting evidence)\b",
    re.IGNORECASE,
)


def _sample_indices(total: int, n: int, seed: int = RANDOM_SEED) -> List[int]:
    if n <= 0 or n >= total:
        return list(range(total))
    rng = random.Random(seed)
    return sorted(rng.sample(range(total), n))


def _default_dense_encoder():
    from sentence_transformers import SentenceTransformer
    from src.config import EMBEDDING_MODEL_NAME

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return lambda texts: model.encode(
        list(texts), convert_to_numpy=True, normalize_embeddings=True
    )


def _make_pair_scorers(
    arms: Sequence[str],
    encoder=None,
    cross_encoder=None,
    hcad_retriever=None,
) -> Dict[str, Callable[[str, str, str], Tuple[float, float]]]:
    """Build per-arm pair scorer fn(query, doc_a, doc_b) -> (score_a, score_b)."""
    from src.baselines.sparse_retrievers import BM25Index

    enc = encoder
    if ({"dense", "hybrid", "rel_prefix"} & set(arms)) and enc is None:
        enc = _default_dense_encoder()

    ce = cross_encoder
    if "reranker" in arms and ce is None:
        from src.baselines.cross_encoder_rag import MedCPTCrossEncoder
        ce = MedCPTCrossEncoder()

    rel_ext = None
    if "rel_prefix" in arms:
        from src.core.relation_extractor import RelationExtractor
        rel_ext = RelationExtractor()

    hcad = hcad_retriever
    if "hcad" in arms and hcad is None:
        import networkx as nx
        from src.core.hcad_retriever import HCADRetriever
        hcad = HCADRetriever(graph=nx.Graph(), collection=None)

    def _bm25_pair(q: str, da: str, db: str) -> Tuple[float, float]:
        idx = BM25Index([da, db], ["a", "b"])
        sc = idx.scores(q)
        return float(sc[0]), float(sc[1])

    def _dense_pair(q: str, da: str, db: str) -> Tuple[float, float]:
        vecs = np.asarray(enc([q, da, db]), dtype=np.float32)
        return float(vecs[0] @ vecs[1]), float(vecs[0] @ vecs[2])

    def _hybrid_pair(q: str, da: str, db: str) -> Tuple[float, float]:
        b0, b1 = _bm25_pair(q, da, db)
        d0, d1 = _dense_pair(q, da, db)
        # On a 2-doc pair, RRF reduces to rank agreement; blend normalized BM25 + cosine.
        bm = np.array([b0, b1], dtype=np.float64)
        bm_norm = (bm - bm.min()) / (bm.max() - bm.min() + 1e-12) if bm.max() > bm.min() else np.zeros(2)
        return float(0.5 * bm_norm[0] + 0.5 * d0), float(0.5 * bm_norm[1] + 0.5 * d1)

    def _rerank_pair(q: str, da: str, db: str) -> Tuple[float, float]:
        sc = ce.score_pairs(q, [da, db])
        return float(sc[0]), float(sc[1])

    def _rel_prefix_pair(q: str, da: str, db: str) -> Tuple[float, float]:
        from src.evaluation.polarity_benchmark import prefix_relation_text
        return _dense_pair(
            prefix_relation_text(rel_ext, q),
            prefix_relation_text(rel_ext, da),
            prefix_relation_text(rel_ext, db),
        )

    def _hcad_pair(q: str, da: str, db: str) -> Tuple[float, float]:
        cands = [
            {"id": "a", "document": da, "metadata": {}, "source": "pair_eval"},
            {"id": "b", "document": db, "metadata": {}, "source": "pair_eval"},
        ]
        original = hcad._get_candidate_contexts
        hcad._get_candidate_contexts = lambda *a, **kw: cands
        hcad._last_query_embedding = None
        prev_gate = os.environ.get("HCAD_DECISION_GATE")
        os.environ["HCAD_DECISION_GATE"] = "0"
        try:
            hcad.retrieve(q)
        finally:
            hcad._get_candidate_contexts = original
            if prev_gate is None:
                os.environ.pop("HCAD_DECISION_GATE", None)
            else:
                os.environ["HCAD_DECISION_GATE"] = prev_gate
        phi = hcad._last_phi_scores
        if phi is None or len(phi) < 2:
            return 0.0, 0.0
        return float(phi[0]), float(phi[1])

    factory = {
        "bm25": _bm25_pair,
        "dense": _dense_pair,
        "hybrid": _hybrid_pair,
        "reranker": _rerank_pair,
        "rel_prefix": _rel_prefix_pair,
        "hcad": _hcad_pair,
    }
    scorers: Dict[str, Callable[[str, str, str], Tuple[float, float]]] = {}
    for arm in arms:
        if arm not in factory:
            raise ValueError(f"Unknown retrieval pair arm {arm!r}")
        scorers[arm] = factory[arm]
    return scorers


# ---------------------------------------------------------------------------
# 1. NevIR (orionweller/NevIR)
# ---------------------------------------------------------------------------

def evaluate_nevir_pairs(
    pairs: Sequence[Dict[str, str]],
    arms: Sequence[str] = ("bm25", "dense", "hybrid", "reranker", "rel_prefix", "hcad"),
    encoder=None,
    cross_encoder=None,
    hcad_retriever=None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate pairwise negation discrimination on NevIR minimal pairs."""
    scorers = _make_pair_scorers(arms, encoder=encoder, cross_encoder=cross_encoder,
                                 hcad_retriever=hcad_retriever)
    arm_results: Dict[str, Any] = {}
    pair_flags_by_arm: Dict[str, List[bool]] = {}

    for arm, score_fn in scorers.items():
        pair_ok: List[int] = []
        query_ok: List[int] = []
        mrr_vals: List[float] = []
        t0 = time.perf_counter()
        for row in pairs:
            s1_a, s1_b = score_fn(row["q1"], row["doc1"], row["doc2"])
            s2_a, s2_b = score_fn(row["q2"], row["doc1"], row["doc2"])
            q1_hit = int(s1_a > s1_b)
            q2_hit = int(s2_b > s2_a)
            query_ok.extend([q1_hit, q2_hit])
            pair_ok.append(int(q1_hit == 1 and q2_hit == 1))
            mrr_vals.append(1.0 if q1_hit else 0.5)
            mrr_vals.append(1.0 if q2_hit else 0.5)
        elapsed = time.perf_counter() - t0
        pair_flags_by_arm[arm] = [bool(x) for x in pair_ok]
        err_vals = [1 - x for x in query_ok]
        arm_results[arm] = {
            "n_pairs": len(pairs),
            "n_queries": len(query_ok),
            "pair_accuracy": bootstrap_ci(pair_ok),
            "query_accuracy": bootstrap_ci(query_ok),
            "polarity_error_rate": bootstrap_ci(err_vals),
            "mrr": bootstrap_ci(mrr_vals),
            "seconds": round(elapsed, 2),
        }
        logger.info(
            "NevIR %-10s pair_acc=%.3f query_acc=%.3f err=%.3f",
            arm,
            arm_results[arm]["pair_accuracy"]["mean"] or 0.0,
            arm_results[arm]["query_accuracy"]["mean"] or 0.0,
            arm_results[arm]["polarity_error_rate"]["mean"] or 0.0,
        )

    sig: Dict[str, Any] = {}
    if "hcad" in pair_flags_by_arm:
        ref = pair_flags_by_arm["hcad"]
        for arm, flags in pair_flags_by_arm.items():
            if arm != "hcad":
                sig[f"hcad_vs_{arm}"] = mcnemar_exact(ref, flags)

    payload = {
        "benchmark": "NevIR",
        "dataset": "orionweller/NevIR",
        "split": "test",
        "n_pairs": len(pairs),
        "arms": arm_results,
        "significance_mcnemar_pair_accuracy": sig,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = output_path or (RESULTS_DIR / "nevir_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("NevIR benchmark saved to %s", out)
    return payload


def run_nevir_benchmark(
    n: int = 300,
    arms: Sequence[str] = ("bm25", "dense", "hybrid", "reranker", "rel_prefix", "hcad"),
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset("orionweller/NevIR", split="test")
    idxs = _sample_indices(len(ds), n)
    pairs = [
        {"q1": ds[i]["q1"], "q2": ds[i]["q2"], "doc1": ds[i]["doc1"], "doc2": ds[i]["doc2"]}
        for i in idxs
    ]
    return evaluate_nevir_pairs(pairs, arms=arms, output_path=output_path)


# ---------------------------------------------------------------------------
# 2. ExcluIR (joey234/ExcluIR)
# ---------------------------------------------------------------------------

def evaluate_excluir_pairs(
    pairs: Sequence[Dict[str, str]],
    arms: Sequence[str] = ("bm25", "dense", "hybrid", "reranker", "rel_prefix", "hcad"),
    encoder=None,
    cross_encoder=None,
    hcad_retriever=None,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate exclusion-query discrimination on ExcluIR pairs (doc_pos vs doc_neg)."""
    scorers = _make_pair_scorers(arms, encoder=encoder, cross_encoder=cross_encoder,
                                 hcad_retriever=hcad_retriever)
    arm_results: Dict[str, Any] = {}
    ok_flags_by_arm: Dict[str, List[bool]] = {}

    for arm, score_fn in scorers.items():
        hits: List[int] = []
        t0 = time.perf_counter()
        for row in pairs:
            s_pos, s_neg = score_fn(row["query"], row["doc_pos"], row["doc_neg"])
            hits.append(int(s_pos > s_neg))
        elapsed = time.perf_counter() - t0
        ok_flags_by_arm[arm] = [bool(x) for x in hits]
        errs = [1 - x for x in hits]
        arm_results[arm] = {
            "n_queries": len(pairs),
            "exclusion_accuracy": bootstrap_ci(hits),
            "polarity_error_rate": bootstrap_ci(errs),
            "seconds": round(elapsed, 2),
        }
        logger.info(
            "ExcluIR %-10s acc=%.3f err=%.3f",
            arm,
            arm_results[arm]["exclusion_accuracy"]["mean"] or 0.0,
            arm_results[arm]["polarity_error_rate"]["mean"] or 0.0,
        )

    sig: Dict[str, Any] = {}
    if "hcad" in ok_flags_by_arm:
        ref = ok_flags_by_arm["hcad"]
        for arm, flags in ok_flags_by_arm.items():
            if arm != "hcad":
                sig[f"hcad_vs_{arm}"] = mcnemar_exact(ref, flags)

    payload = {
        "benchmark": "ExcluIR",
        "dataset": "joey234/ExcluIR",
        "split": "train",
        "n_queries": len(pairs),
        "arms": arm_results,
        "significance_mcnemar_accuracy": sig,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = output_path or (RESULTS_DIR / "excluir_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("ExcluIR benchmark saved to %s", out)
    return payload


def run_excluir_benchmark(
    n: int = 300,
    arms: Sequence[str] = ("bm25", "dense", "hybrid", "reranker", "rel_prefix", "hcad"),
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    from datasets import load_dataset

    ds = load_dataset("joey234/ExcluIR", split="train")
    idxs = _sample_indices(len(ds), n)
    pairs = [
        {"query": ds[i]["query"], "doc_pos": ds[i]["doc_pos"], "doc_neg": ds[i]["doc_neg"]}
        for i in idxs
    ]
    return evaluate_excluir_pairs(pairs, arms=arms, output_path=output_path)


# ---------------------------------------------------------------------------
# 3. MIRAGE-style Clinical MCQ Accuracy (MedQA-USMLE + PubMedQA)
# ---------------------------------------------------------------------------

def _load_mirage_samples(n_per_dataset: int = 50) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    items: List[Dict[str, Any]] = []
    medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    for idx in _sample_indices(len(medqa), n_per_dataset):
        row = medqa[idx]
        opts = row["options"]
        opt_str = "\n".join(f"{k}. {opts[k]}" for k in sorted(opts))
        items.append({
            "id": f"medqa_{idx}",
            "dataset": "MedQA-USMLE",
            "question": f"{row['question']}\n\nOptions:\n{opt_str}\nAnswer with the single option letter (A, B, C, or D).",
            "retrieval_query": row["question"],
            "gold_letter": str(row["answer_idx"]).strip().upper(),
        })

    pqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    label_map = {"yes": "A", "no": "B", "maybe": "C"}
    for idx in _sample_indices(len(pqa), n_per_dataset):
        row = pqa[idx]
        gold = label_map.get(str(row["final_decision"]).strip().lower(), "C")
        items.append({
            "id": f"pubmedqa_{idx}",
            "dataset": "PubMedQA",
            "question": (
                f"{row['question']}\n\nOptions:\nA. yes\nB. no\nC. maybe\n"
                "Answer with the single option letter (A, B, or C)."
            ),
            "retrieval_query": row["question"],
            "gold_letter": gold,
        })
    return items


def run_mirage_benchmark(
    n_per_dataset: int = 50,
    arms: Sequence[str] = ("closed_book", "bm25", "dense", "hybrid", "reranker", "graphrag", "hcad"),
    output_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    llm_client=None,
) -> Dict[str, Any]:
    """Evaluate end-to-end MCQ accuracy on MedQA-USMLE + PubMedQA."""
    from src.baselines.baseline_llm import _extract_answer_letter
    from src.core.llm_client import UnifiedLLMClient
    from src.evaluation.run_base_paper_benchmark import (
        _env_overrides,
        _hcad_config_id,
        build_arms,
    )

    samples = _load_mirage_samples(n_per_dataset=n_per_dataset)
    built = build_arms(arms)
    llm = llm_client or UnifiedLLMClient(model=LLM_MODEL, temperature=0.0, max_tokens=96)
    cfg_id = _hcad_config_id()

    ckpt = checkpoint_path or (RESULTS_DIR / "mirage_checkpoint.jsonl")
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    completed: Dict[str, Dict[str, Any]] = {}
    if ckpt.exists():
        for line in ckpt.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (list(rec.get("arms", [])) == list(arms)
                    and rec.get("model", LLM_MODEL) == LLM_MODEL
                    and rec.get("n_per_dataset", n_per_dataset) == n_per_dataset
                    and rec.get("hcad_config_id", cfg_id) == cfg_id):
                completed[rec["id"]] = rec

    with open(ckpt, "a", encoding="utf-8") as cf:
        for idx, sample in enumerate(samples, 1):
            if sample["id"] in completed:
                continue
            per_arm: Dict[str, Any] = {}
            for arm_id in arms:
                arm = built[arm_id]
                with _env_overrides(arm.env):
                    contexts = arm.retrieve(sample["retrieval_query"]) if arm.retrieve else []
                    msgs = arm.build_messages(sample["question"], contexts)
                raw = llm.chat(msgs, temperature=0.0, max_tokens=96)
                ret_abstained = bool(arm.retrieve is not None and len(contexts) == 0)
                abstained = bool(_ABSTENTION_PAT.search(raw or ""))
                pred = "X" if abstained else _extract_answer_letter(raw)
                correct = int(pred == sample["gold_letter"])
                per_arm[arm_id] = {
                    "pred": pred,
                    "correct": correct,
                    "abstained": int(abstained),
                    "retrieval_abstained": int(ret_abstained),
                }
            rec = {
                "id": sample["id"],
                "dataset": sample["dataset"],
                "gold_letter": sample["gold_letter"],
                "arms": list(arms),
                "model": LLM_MODEL,
                "n_per_dataset": n_per_dataset,
                "hcad_config_id": cfg_id,
                "results": per_arm,
            }
            completed[sample["id"]] = rec
            cf.write(json.dumps(rec) + "\n")
            cf.flush()
            if idx % 10 == 0 or idx == len(samples):
                logger.info("MIRAGE progress: %d/%d", idx, len(samples))

    rows = [completed[s["id"]] for s in samples]
    summary_arms: Dict[str, Any] = {}
    for arm_id in arms:
        acc_all = [r["results"][arm_id]["correct"] for r in rows]
        abst_all = [r["results"][arm_id]["abstained"] for r in rows]
        ret_abst_all = [r["results"][arm_id].get("retrieval_abstained", 0) for r in rows]
        by_ds = {}
        for ds_name in ("MedQA-USMLE", "PubMedQA"):
            sub = [r["results"][arm_id]["correct"] for r in rows if r["dataset"] == ds_name]
            if sub:
                by_ds[ds_name] = bootstrap_ci(sub)
        summary_arms[arm_id] = {
            "overall_accuracy": bootstrap_ci(acc_all),
            "abstention_rate": bootstrap_ci(abst_all),
            "retrieval_abstention_rate": bootstrap_ci(ret_abst_all),
            "by_dataset": by_ds,
        }

    sig: Dict[str, Any] = {}
    if "hcad" in arms:
        ref = [bool(r["results"]["hcad"]["correct"]) for r in rows]
        for arm_id in arms:
            if arm_id != "hcad":
                sig[f"hcad_vs_{arm_id}"] = mcnemar_exact(
                    ref, [bool(r["results"][arm_id]["correct"]) for r in rows]
                )

    payload = {
        "benchmark": "MIRAGE_MedQA_PubMedQA",
        "n_per_dataset": n_per_dataset,
        "n_total": len(rows),
        "generator": LLM_MODEL,
        "arms": summary_arms,
        "significance_mcnemar_accuracy": sig,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = output_path or (RESULTS_DIR / "mirage_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("MIRAGE benchmark saved to %s", out)
    return payload
