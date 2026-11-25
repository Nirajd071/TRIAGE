"""Indication vs. Contraindication Paired-Polarity Benchmark.

Directly tests the paper's titular claim ("Does your retriever know what
'contraindicated' means?"), which the reviewers noted was never measured.

Design
------
Each of the curated drugs contributes a MINIMAL PAIR of evidence
statements that share the drug entity and differ in clinical polarity:

    +  "Metformin is indicated as first-line therapy for type 2 diabetes ..."
    -  "Metformin is contraindicated in severe renal impairment ..."

and four queries (2 polarities x 2 phrasings):

    indication / lexical      "What is metformin indicated for?"
    indication / paraphrase   "Which condition is metformin used to treat?"
    opposite   / lexical      "In which condition is metformin contraindicated?"
    opposite   / paraphrase   "In which patients must metformin be avoided?"

"Paraphrase" queries avoid the polarity keyword that appears in the
evidence, so a lexical matcher cannot solve them by string overlap. The
opposite polarity is either an absolute contraindication or a caution /
avoid-in statement (``polarity_type``).

All 2 x N statements form ONE shared corpus, so every query is ranked
against its own opposite-polarity twin (the hard distractor) plus every
other drug's statements.

Metrics (per retriever, per query, then aggregated with bootstrap CIs)
---------------------------------------------------------------------
- ``polarity_recall_at_k``  gold (same drug, requested polarity) in top-k.
- ``polarity_error``        the opposite-polarity twin is ranked ABOVE the
                            gold statement (full ranking over the corpus).
- ``polarity_error_rate``   mean of ``polarity_error``.
- ``pair_accuracy``         both polarities of a drug (same phrasing) are
                            resolved without a polarity error.
- ``mrr``                   reciprocal rank of the gold statement.
- McNemar exact test of HCAD-RAG vs every other arm on per-query errors.

Modes
-----
``corpus`` (default, controlled): every arm ranks the same curated corpus.
    HCAD-RAG runs its stage 1/3/4 scoring (query understanding, semantic
    decoupling with soft Gram-Schmidt, hybrid Phi) over the corpus as the
    candidate pool. Two HCAD variants are reported:
      ``hcad``     text-only candidates (same information as BM25/dense);
      ``hcad_kg``  candidates carry KG relation metadata, as PrimeKG triples
                   do in the deployed system (an information advantage,
                   reported separately for transparency).
``store``: full retrieval against the built vector store / knowledge graph;
    a retrieved context counts as gold/opposite by its PrimeKG relation
    metadata and head-entity match.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import POLARITY_TOP_K, RESULTS_DIR  # noqa: E402
from src.evaluation.stats import bootstrap_ci, mcnemar_exact  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Curated minimal pairs (established prescribing-information facts)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PolarityPair:
    drug: str
    indication: str
    indication_evidence: str
    opposite: str
    opposite_evidence: str
    polarity_type: str  # "contraindication" | "caution"


PAIRS: tuple[PolarityPair, ...] = (
    PolarityPair("Metformin", "type 2 diabetes mellitus",
                 "Metformin is indicated as first-line pharmacotherapy for type 2 diabetes mellitus to improve glycemic control.",
                 "severe renal impairment",
                 "Metformin is contraindicated in severe renal impairment (eGFR below 30 mL/min/1.73 m2) because of the risk of lactic acidosis.",
                 "contraindication"),
    PolarityPair("Warfarin", "atrial fibrillation",
                 "Warfarin is indicated for prevention of stroke and systemic embolism in patients with atrial fibrillation.",
                 "pregnancy",
                 "Warfarin is contraindicated in pregnancy because it crosses the placenta and causes embryopathy and fetal hemorrhage.",
                 "contraindication"),
    PolarityPair("Propranolol", "hypertension",
                 "Propranolol is indicated for the management of hypertension, alone or combined with other antihypertensive agents.",
                 "bronchial asthma",
                 "Propranolol is contraindicated in bronchial asthma because non-selective beta blockade can precipitate severe bronchospasm.",
                 "contraindication"),
    PolarityPair("Isotretinoin", "severe recalcitrant nodular acne",
                 "Isotretinoin is indicated for severe recalcitrant nodular acne unresponsive to conventional therapy including systemic antibiotics.",
                 "pregnancy",
                 "Isotretinoin is contraindicated in pregnancy because it is a potent teratogen causing severe birth defects.",
                 "contraindication"),
    PolarityPair("Sildenafil", "erectile dysfunction",
                 "Sildenafil is indicated for the treatment of erectile dysfunction.",
                 "concomitant nitrate therapy",
                 "Sildenafil is contraindicated in patients taking organic nitrates because the combination causes profound hypotension.",
                 "contraindication"),
    PolarityPair("Aspirin", "secondary prevention of myocardial infarction",
                 "Low-dose aspirin is indicated for secondary prevention of myocardial infarction and ischemic stroke.",
                 "children with viral illness",
                 "Aspirin should be avoided in children and teenagers with viral illness such as influenza or varicella because of the risk of Reye syndrome.",
                 "caution"),
    PolarityPair("Lisinopril", "hypertension",
                 "Lisinopril is indicated for the treatment of hypertension and of symptomatic heart failure.",
                 "history of angioedema",
                 "Lisinopril is contraindicated in patients with a history of angioedema related to previous ACE inhibitor treatment.",
                 "contraindication"),
    PolarityPair("Methotrexate", "rheumatoid arthritis",
                 "Methotrexate is indicated as disease-modifying therapy for active rheumatoid arthritis.",
                 "pregnancy",
                 "Methotrexate is contraindicated in pregnant women with non-neoplastic disease because it causes fetal death and congenital anomalies.",
                 "contraindication"),
    PolarityPair("Lithium", "bipolar disorder",
                 "Lithium is indicated for the treatment and maintenance of bipolar disorder to reduce manic episodes.",
                 "renal impairment and dehydration",
                 "Lithium requires great caution in renal impairment, dehydration or sodium depletion because lithium toxicity develops rapidly.",
                 "caution"),
    PolarityPair("Ibuprofen", "osteoarthritis pain",
                 "Ibuprofen is indicated for relief of the signs and symptoms of osteoarthritis and mild to moderate pain.",
                 "active peptic ulcer disease",
                 "Ibuprofen is contraindicated in active peptic ulcer disease or gastrointestinal bleeding because NSAIDs worsen mucosal injury.",
                 "contraindication"),
    PolarityPair("Verapamil", "supraventricular tachycardia",
                 "Verapamil is indicated for rate control and termination of paroxysmal supraventricular tachycardia.",
                 "Wolff-Parkinson-White syndrome with atrial fibrillation",
                 "Verapamil is contraindicated in atrial fibrillation with an accessory pathway such as Wolff-Parkinson-White syndrome because it can accelerate ventricular response.",
                 "contraindication"),
    PolarityPair("Clozapine", "treatment-resistant schizophrenia",
                 "Clozapine is indicated for treatment-resistant schizophrenia after failure of other antipsychotics.",
                 "history of clozapine-induced agranulocytosis",
                 "Clozapine must not be restarted in patients with a history of clozapine-induced agranulocytosis or severe neutropenia.",
                 "contraindication"),
    PolarityPair("Tamoxifen", "estrogen receptor-positive breast cancer",
                 "Tamoxifen is indicated for the treatment of estrogen receptor-positive breast cancer.",
                 "history of deep vein thrombosis or pulmonary embolism",
                 "Tamoxifen is contraindicated for breast cancer risk reduction in women with a history of deep vein thrombosis or pulmonary embolism.",
                 "contraindication"),
    PolarityPair("Ciprofloxacin", "complicated urinary tract infection",
                 "Ciprofloxacin is indicated for complicated urinary tract infections and pyelonephritis caused by susceptible bacteria.",
                 "myasthenia gravis",
                 "Ciprofloxacin should be avoided in patients with myasthenia gravis because fluoroquinolones can exacerbate muscle weakness.",
                 "caution"),
    PolarityPair("Succinylcholine", "rapid sequence intubation",
                 "Succinylcholine is indicated as a short-acting neuromuscular blocker to facilitate rapid sequence intubation.",
                 "personal or family history of malignant hyperthermia",
                 "Succinylcholine is contraindicated in patients with a personal or family history of malignant hyperthermia.",
                 "contraindication"),
    PolarityPair("Levothyroxine", "hypothyroidism",
                 "Levothyroxine is indicated as replacement therapy for primary hypothyroidism.",
                 "uncorrected adrenal insufficiency",
                 "Levothyroxine is contraindicated in uncorrected adrenal insufficiency because it can precipitate adrenal crisis.",
                 "contraindication"),
    PolarityPair("Nitroglycerin", "angina pectoris",
                 "Nitroglycerin is indicated for the acute relief and prophylaxis of angina pectoris.",
                 "concomitant phosphodiesterase-5 inhibitor use",
                 "Nitroglycerin is contraindicated with phosphodiesterase-5 inhibitors such as sildenafil because of severe hypotension.",
                 "contraindication"),
    PolarityPair("Amiodarone", "ventricular arrhythmias",
                 "Amiodarone is indicated for life-threatening recurrent ventricular arrhythmias such as ventricular tachycardia.",
                 "second- or third-degree atrioventricular block without a pacemaker",
                 "Amiodarone is contraindicated in second- or third-degree atrioventricular block unless a functioning pacemaker is present.",
                 "contraindication"),
    PolarityPair("Spironolactone", "heart failure with reduced ejection fraction",
                 "Spironolactone is indicated to reduce mortality in heart failure with reduced ejection fraction.",
                 "hyperkalemia",
                 "Spironolactone is contraindicated in hyperkalemia because it further raises serum potassium.",
                 "contraindication"),
    PolarityPair("Pioglitazone", "type 2 diabetes mellitus",
                 "Pioglitazone is indicated as an adjunct to diet and exercise to improve glycemic control in type 2 diabetes mellitus.",
                 "NYHA class III or IV heart failure",
                 "Pioglitazone is contraindicated in NYHA class III or IV heart failure because it causes fluid retention.",
                 "contraindication"),
    PolarityPair("Valproate", "epilepsy",
                 "Valproate is indicated for generalized and focal epilepsy including absence seizures.",
                 "significant hepatic dysfunction",
                 "Valproate is contraindicated in significant hepatic dysfunction because of the risk of fatal hepatotoxicity.",
                 "contraindication"),
    PolarityPair("Carbamazepine", "trigeminal neuralgia",
                 "Carbamazepine is indicated for the treatment of the pain of trigeminal neuralgia.",
                 "history of bone marrow depression",
                 "Carbamazepine is contraindicated in patients with a history of bone marrow depression.",
                 "contraindication"),
    PolarityPair("Metoclopramide", "diabetic gastroparesis",
                 "Metoclopramide is indicated for the relief of symptoms of diabetic gastroparesis.",
                 "gastrointestinal obstruction",
                 "Metoclopramide is contraindicated when stimulation of motility is dangerous, as in mechanical gastrointestinal obstruction or perforation.",
                 "contraindication"),
    PolarityPair("Codeine", "mild to moderate pain",
                 "Codeine is indicated for the management of mild to moderate pain in adults.",
                 "children younger than 12 years",
                 "Codeine is contraindicated in children younger than 12 years because ultra-rapid metabolism can cause fatal respiratory depression.",
                 "contraindication"),
    PolarityPair("Tetracycline", "acne vulgaris",
                 "Tetracycline is indicated as adjunctive therapy for severe acne vulgaris.",
                 "children under 8 years",
                 "Tetracycline should be avoided in children under 8 years because it causes permanent tooth discoloration.",
                 "caution"),
    PolarityPair("Allopurinol", "chronic gout",
                 "Allopurinol is indicated as urate-lowering therapy for chronic gout.",
                 "HLA-B*58:01 carriers",
                 "Allopurinol should be used with caution or avoided in HLA-B*58:01 carriers because of the high risk of severe cutaneous reactions.",
                 "caution"),
    PolarityPair("Abacavir", "HIV-1 infection",
                 "Abacavir is indicated in combination antiretroviral therapy for HIV-1 infection.",
                 "HLA-B*57:01 positive patients",
                 "Abacavir is contraindicated in HLA-B*57:01 positive patients because of the risk of hypersensitivity reactions.",
                 "contraindication"),
    PolarityPair("Bupropion", "major depressive disorder",
                 "Bupropion is indicated for the treatment of major depressive disorder and for smoking cessation.",
                 "seizure disorder",
                 "Bupropion is contraindicated in patients with a seizure disorder because it lowers the seizure threshold.",
                 "contraindication"),
    PolarityPair("Sumatriptan", "acute migraine",
                 "Sumatriptan is indicated for the acute treatment of migraine with or without aura.",
                 "coronary artery disease",
                 "Sumatriptan is contraindicated in ischemic coronary artery disease because it causes coronary vasoconstriction.",
                 "contraindication"),
    PolarityPair("Ethinyl estradiol", "contraception",
                 "Combined oral contraceptives containing ethinyl estradiol are indicated for the prevention of pregnancy.",
                 "smokers over 35 years of age",
                 "Combined oral contraceptives containing ethinyl estradiol are contraindicated in women over 35 years who smoke because of cardiovascular risk.",
                 "contraindication"),
    PolarityPair("Heparin", "venous thromboembolism",
                 "Heparin is indicated for the prophylaxis and treatment of venous thromboembolism.",
                 "history of heparin-induced thrombocytopenia",
                 "Heparin is contraindicated in patients with a history of heparin-induced thrombocytopenia.",
                 "contraindication"),
    PolarityPair("Ceftriaxone", "community-acquired pneumonia",
                 "Ceftriaxone is indicated for community-acquired pneumonia and other serious bacterial infections.",
                 "neonates receiving intravenous calcium",
                 "Ceftriaxone is contraindicated in neonates receiving calcium-containing intravenous solutions because of fatal precipitates.",
                 "contraindication"),
    PolarityPair("Digoxin", "atrial fibrillation rate control",
                 "Digoxin is indicated for ventricular rate control in chronic atrial fibrillation.",
                 "ventricular fibrillation",
                 "Digoxin is contraindicated in ventricular fibrillation.",
                 "contraindication"),
    PolarityPair("Hydrochlorothiazide", "hypertension",
                 "Hydrochlorothiazide is indicated for the management of hypertension and edema.",
                 "anuria",
                 "Hydrochlorothiazide is contraindicated in patients with anuria.",
                 "contraindication"),
    PolarityPair("Atorvastatin", "hypercholesterolemia",
                 "Atorvastatin is indicated to reduce LDL cholesterol in primary hypercholesterolemia.",
                 "acute liver failure or decompensated cirrhosis",
                 "Atorvastatin is contraindicated in acute liver failure or decompensated cirrhosis.",
                 "contraindication"),
)


PAIRED_CLINICAL_POLARITY_CASES: List[Dict[str, Any]] = [
    {
        "pair_id": f"POL_{i+1:02d}",
        "entity": p.drug,
        "positive": {
            "query": f"Under what clinical conditions is {p.drug.lower()} indicated for {p.indication}?",
            "expected_polarity": "indicated",
            "safety_keywords": [p.drug.lower(), "indicated", p.indication.lower()],
        },
        "negative": {
            "query": f"Why is {p.drug.lower()} contraindicated in patients with {p.opposite}?",
            "expected_polarity": "contraindicated",
            "safety_keywords": [p.drug.lower(), "contraindicated", p.opposite.lower(), "avoid"],
        },
    }
    for i, p in enumerate(PAIRS)
]


class PolarityBenchmark:
    """Benchmark runner / classifier evaluating RAG systems on clinical polarity discrimination."""

    @staticmethod
    def _evaluate_polarity_fidelity(
        answer: str,
        expected_polarity: str,
        safety_keywords: List[str],
    ) -> Dict[str, Any]:
        ans_lower = (answer or "").lower()
        warning_terms = [
            "contraindicated", "avoid", "caution", "danger", "harm", "toxic",
            "adverse", "black box", "not recommended", "should not", "risk",
            "fatal", "lethal", "discontinued", "prohibited", "stop", "withhold"
        ]
        has_warning = any(w in ans_lower for w in warning_terms)
        has_safety_keywords = any(kw.lower() in ans_lower for kw in safety_keywords)

        if expected_polarity == "contraindicated":
            is_inverted_error = (not has_warning) and ("indicated" in ans_lower or "recommended" in ans_lower)
            correct_polarity = has_warning and not is_inverted_error
        else:
            is_inverted_error = False
            correct_polarity = "indicated" in ans_lower or "effective" in ans_lower or "benefit" in ans_lower or has_safety_keywords

        return {
            "correct_polarity": bool(correct_polarity),
            "inversion_error": bool(is_inverted_error),
            "has_warning_language": bool(has_warning),
            "safety_keyword_coverage": (
                sum(1 for kw in safety_keywords if kw.lower() in ans_lower) / max(1, len(safety_keywords))
            ),
        }

    @staticmethod
    def _check_safety_recall(contexts: List[Dict[str, Any]], expected_keywords: List[str]) -> float:
        if not contexts or not expected_keywords:
            return 0.0
        all_text = " ".join([
            str(c.get("document", "")) + " " + str(c.get("metadata", ""))
            for c in contexts
        ]).lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in all_text)
        return hits / len(expected_keywords)

    def run_benchmark(self, num_pairs: Optional[int] = None) -> Dict[str, Any]:
        return run_polarity_benchmark()

QUERY_TEMPLATES = {
    ("indication", "lexical"): "What is {drug} indicated for?",
    ("indication", "paraphrase"): "Which condition is {drug} used to treat?",
    ("contraindication", "lexical"): "In which condition is {drug} contraindicated?",
    ("contraindication", "paraphrase"): "In which patients must {drug} never be given?",
    ("caution", "lexical"): "In which patients should {drug} be used with caution or avoided?",
    ("caution", "paraphrase"): "Which patients face a special danger when given {drug}?",
}

PHRASINGS = ("lexical", "paraphrase")


def build_corpus(pairs: Sequence[PolarityPair] = PAIRS) -> List[Dict]:
    """Two statements per drug; ids encode drug index and polarity."""
    corpus = []
    for i, p in enumerate(pairs):
        corpus.append({"id": f"p{i:02d}_ind", "pair": i, "polarity": "indication",
                       "text": p.indication_evidence,
                       "metadata": {"head_name": p.drug, "relation": "indication",
                                    "tail_name": p.indication}})
        corpus.append({"id": f"p{i:02d}_opp", "pair": i, "polarity": "opposite",
                       "text": p.opposite_evidence,
                       "metadata": {"head_name": p.drug, "relation": "contraindication",
                                    "tail_name": p.opposite}})
    return corpus


def build_queries(pairs: Sequence[PolarityPair] = PAIRS) -> List[Dict]:
    queries = []
    for i, p in enumerate(pairs):
        for phrasing in PHRASINGS:
            queries.append({
                "qid": f"p{i:02d}_ind_{phrasing}", "pair": i, "drug": p.drug,
                "target": "indication", "phrasing": phrasing,
                "polarity_type": "indication",
                "query": QUERY_TEMPLATES[("indication", phrasing)].format(drug=p.drug),
                "gold_id": f"p{i:02d}_ind", "opposite_id": f"p{i:02d}_opp",
            })
            queries.append({
                "qid": f"p{i:02d}_opp_{phrasing}", "pair": i, "drug": p.drug,
                "target": "opposite", "phrasing": phrasing,
                "polarity_type": p.polarity_type,
                "query": QUERY_TEMPLATES[(p.polarity_type, phrasing)].format(drug=p.drug),
                "gold_id": f"p{i:02d}_opp", "opposite_id": f"p{i:02d}_ind",
            })
    return queries


# ---------------------------------------------------------------------------
# Rankers: fn(query: str) -> list of corpus ids, best first (full ranking)
# ---------------------------------------------------------------------------

Ranker = Callable[[str], List[str]]


def _rank_from_scores(corpus: Sequence[Dict], scores: np.ndarray) -> List[str]:
    order = sorted(range(len(corpus)), key=lambda i: (-float(scores[i]), i))
    return [corpus[i]["id"] for i in order]


def make_bm25_ranker(corpus: Sequence[Dict]) -> Ranker:
    from src.baselines.sparse_retrievers import BM25Index

    index = BM25Index([d["text"] for d in corpus], [d["id"] for d in corpus])
    return lambda q: _rank_from_scores(corpus, index.scores(q))


def make_dense_ranker(corpus: Sequence[Dict],
                      encode: Callable[[List[str]], np.ndarray]) -> Ranker:
    """Cosine ranking with an L2-normalising encoder (S-PubMedBERT by default)."""
    doc_vecs = np.asarray(encode([d["text"] for d in corpus]), dtype=np.float32)

    def rank(q: str) -> List[str]:
        qv = np.asarray(encode([q]), dtype=np.float32)[0]
        return _rank_from_scores(corpus, doc_vecs @ qv)
    return rank


def make_hybrid_ranker(bm25: Ranker, dense: Ranker, k: int = 60) -> Ranker:
    """Reciprocal-rank fusion of two full rankings."""
    def rank(q: str) -> List[str]:
        fused: Dict[str, float] = {}
        for ranking in (bm25(q), dense(q)):
            for r, doc_id in enumerate(ranking, 1):
                fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + r)
        return sorted(fused, key=lambda d: (-fused[d], d))
    return rank


def prefix_relation_text(extractor, text: str) -> str:
    """Prepend extracted relation/negation label for the simple rel-prefix baseline."""
    kws = extractor.extract_keywords(text) or []
    neg, cue = extractor.detect_negation(text)
    tags = []
    if kws:
        tags.append(kws[0])
    if neg and cue and cue not in tags:
        tags.append(f"neg:{cue}")
    return f"[{' | '.join(tags)}] {text}" if tags else text


def make_rel_prefix_ranker(corpus: Sequence[Dict],
                           encode: Callable[[List[str]], np.ndarray]) -> Ranker:
    """Simple relation-phrase-prefixed dense baseline (no Gram-Schmidt)."""
    from src.core.relation_extractor import RelationExtractor

    extractor = RelationExtractor()
    doc_vecs = np.asarray(
        encode([prefix_relation_text(extractor, d["text"]) for d in corpus]),
        dtype=np.float32,
    )

    def rank(q: str) -> List[str]:
        qv = np.asarray(encode([prefix_relation_text(extractor, q)]), dtype=np.float32)[0]
        return _rank_from_scores(corpus, doc_vecs @ qv)
    return rank


def make_reranker_ranker(corpus: Sequence[Dict], cross_encoder=None) -> Ranker:
    """Biomedical Cross-Encoder (ncbi/MedCPT-Cross-Encoder) full corpus ranking."""
    from src.baselines.cross_encoder_rag import MedCPTCrossEncoder

    ce = cross_encoder or MedCPTCrossEncoder()
    texts = [d["text"] for d in corpus]

    def rank(q: str) -> List[str]:
        scores = ce.score_pairs(q, texts)
        return _rank_from_scores(corpus, scores)
    return rank


def make_hcad_ranker(corpus: Sequence[Dict], retriever=None,
                     with_kg_metadata: bool = False) -> Ranker:
    """HCAD-RAG ranking of the curated corpus (stages 1, 3, 4).

    The corpus is injected as the candidate pool and the PRE-GATE hybrid
    Phi scores recorded by ``HCADRetriever.retrieve`` rank every
    statement, so polarity errors are measured over the full ranking
    rather than only over items that survive the filter.
    """
    if retriever is None:
        import networkx as nx
        from src.core.hcad_retriever import HCADRetriever

        # Empty graph: no PrimeKG needed for the controlled corpus mode.
        retriever = HCADRetriever(graph=nx.Graph(), collection=None)

    def candidates() -> List[Dict]:
        out = []
        for d in corpus:
            meta = dict(d["metadata"]) if with_kg_metadata else {}
            out.append({"id": d["id"], "document": d["text"], "metadata": meta,
                        "source": "polarity_corpus"})
        return out

    def rank(q: str) -> List[str]:
        import os

        original = retriever._get_candidate_contexts
        retriever._get_candidate_contexts = lambda *a, **kw: candidates()
        # The ranking uses PRE-gate scores, so the decision gate (a live
        # TypeSafe API call when TYPESAFE_API_KEY is set) is skipped here.
        previous_gate = os.environ.get("HCAD_DECISION_GATE")
        os.environ["HCAD_DECISION_GATE"] = "0"
        try:
            retriever.retrieve(q)
        finally:
            retriever._get_candidate_contexts = original
            if previous_gate is None:
                os.environ.pop("HCAD_DECISION_GATE", None)
            else:
                os.environ["HCAD_DECISION_GATE"] = previous_gate
        pool = retriever._last_candidate_pool or []
        phi = retriever._last_phi_scores
        if not pool or phi is None:
            raise RuntimeError("HCAD retriever produced no candidate scores")
        scores = np.asarray(phi, dtype=np.float32)
        order = sorted(range(len(pool)), key=lambda i: (-float(scores[i]), i))
        return [pool[i]["id"] for i in order]
    return rank


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def score_query(ranking: Sequence[str], gold_id: str, opposite_id: str, k: int) -> Dict:
    """Per-query polarity metrics from a ranking (ids best-first)."""
    pos = {doc_id: r for r, doc_id in enumerate(ranking, 1)}
    inf = len(ranking) + 1
    r_gold = pos.get(gold_id, inf)
    r_opp = pos.get(opposite_id, inf)
    return {
        "rank_gold": r_gold if r_gold != inf else None,
        "rank_opposite": r_opp if r_opp != inf else None,
        "recall_at_k": int(r_gold <= k),
        # opposite-polarity twin ranked above the requested polarity
        "polarity_error": int(r_opp < r_gold),
        "opposite_in_top_k": int(r_opp <= k),
        "reciprocal_rank": (1.0 / r_gold) if r_gold != inf else 0.0,
    }


def _summary(rows: Sequence[Dict]) -> Dict:
    def ci(key):
        return bootstrap_ci([r[key] for r in rows])
    return {
        "n_queries": len(rows),
        "polarity_recall_at_k": ci("recall_at_k"),
        "polarity_error_rate": ci("polarity_error"),
        "opposite_in_top_k_rate": ci("opposite_in_top_k"),
        "mrr": ci("reciprocal_rank"),
    }


def _pair_accuracy(rows: Sequence[Dict]) -> Dict:
    by_key: Dict[tuple, List[int]] = {}
    for r in rows:
        by_key.setdefault((r["pair"], r["phrasing"]), []).append(r["polarity_error"])
    flags = [int(len(v) == 2 and sum(v) == 0) for v in by_key.values()]
    return bootstrap_ci(flags)


def evaluate_rankers(rankers: Dict[str, Ranker], queries: Sequence[Dict],
                     k: int = POLARITY_TOP_K, reference_arm: str = "hcad") -> Dict:
    per_query: Dict[str, List[Dict]] = {}
    timing: Dict[str, float] = {}
    for arm, ranker in rankers.items():
        rows = []
        t0 = time.perf_counter()
        for q in queries:
            ranking = ranker(q["query"])
            rows.append({**q, **score_query(ranking, q["gold_id"], q["opposite_id"], k),
                         "top_k": list(ranking[:k])})
        timing[arm] = (time.perf_counter() - t0) / max(1, len(queries))
        per_query[arm] = rows
        logger.info("Polarity arm %-10s error=%.3f recall@%d=%.3f", arm,
                    np.mean([r["polarity_error"] for r in rows]), k,
                    np.mean([r["recall_at_k"] for r in rows]))

    arms = {}
    for arm, rows in per_query.items():
        arms[arm] = {
            "overall": _summary(rows),
            "pair_accuracy": _pair_accuracy(rows),
            "by_target_polarity": {
                t: _summary([r for r in rows if r["target"] == t])
                for t in ("indication", "opposite")
            },
            "by_phrasing": {
                ph: _summary([r for r in rows if r["phrasing"] == ph]) for ph in PHRASINGS
            },
            "mean_seconds_per_query": timing[arm],
        }

    significance = {}
    if reference_arm in per_query:
        ref_ok = [not r["polarity_error"] for r in per_query[reference_arm]]
        for arm, rows in per_query.items():
            if arm == reference_arm:
                continue
            significance[f"{reference_arm}_vs_{arm}"] = mcnemar_exact(
                ref_ok, [not r["polarity_error"] for r in rows])
    return {"arms": arms, "significance_mcnemar_polarity_error": significance,
            "per_query": per_query}


# ---------------------------------------------------------------------------
# Store mode: full pipeline over the built vector store / PrimeKG
# ---------------------------------------------------------------------------

_INDICATION_RELATIONS = {"indication", "off-label use"}
_OPPOSITE_RELATIONS = {"contraindication"}


def context_polarity(ctx: Dict, drug: str) -> Optional[str]:
    """'indication' / 'opposite' / None for a retrieved store context."""
    meta = ctx.get("metadata") or {}
    rel = str(meta.get("relation") or "").lower()
    head = str(meta.get("head_name") or "").lower()
    tail = str(meta.get("tail_name") or "").lower()
    if drug.lower() not in (head, tail):
        return None
    if rel in _INDICATION_RELATIONS:
        return "indication"
    if rel in _OPPOSITE_RELATIONS:
        return "opposite"
    return None


def evaluate_store_retrievers(retrievers: Dict[str, Callable[[str], List[Dict]]],
                              queries: Sequence[Dict], k: int = POLARITY_TOP_K) -> Dict:
    """Polarity metrics when arms retrieve from the real store (top-k only).

    Gold = a context of the queried drug with the requested polarity;
    error = an opposite-polarity context of that drug ranks above the first
    gold context (or appears in top-k with no gold present).
    """
    out: Dict[str, Dict] = {}
    for arm, retrieve in retrievers.items():
        rows = []
        for q in queries:
            target = q["target"]
            labels = [context_polarity(c, q["drug"]) for c in retrieve(q["query"])[:k]]
            r_gold = next((i for i, lab in enumerate(labels, 1) if lab == target), None)
            other = "opposite" if target == "indication" else "indication"
            r_opp = next((i for i, lab in enumerate(labels, 1) if lab == other), None)
            rows.append({**q, "rank_gold": r_gold, "rank_opposite": r_opp,
                         "recall_at_k": int(r_gold is not None),
                         "polarity_error": int(r_opp is not None and (r_gold is None or r_opp < r_gold)),
                         "opposite_in_top_k": int(r_opp is not None),
                         "reciprocal_rank": 1.0 / r_gold if r_gold else 0.0,
                         "retrieved_labels": labels})
        out[arm] = {"overall": _summary(rows), "pair_accuracy": _pair_accuracy(rows),
                    "per_query": rows}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_encoder():
    from sentence_transformers import SentenceTransformer
    from src.config import EMBEDDING_MODEL_NAME

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return lambda texts: model.encode(list(texts), convert_to_numpy=True,
                                      normalize_embeddings=True)


def run_polarity_benchmark(arms: Sequence[str] = ("bm25", "dense", "hybrid", "reranker", "rel_prefix", "hcad", "hcad_kg"),
                           k: int = POLARITY_TOP_K, output_path: Optional[Path] = None,
                           encoder=None, hcad_retriever=None, cross_encoder=None) -> Dict:
    corpus, queries = build_corpus(), build_queries()
    rankers: Dict[str, Ranker] = {}
    bm25 = dense = None
    enc = encoder if ({"dense", "hybrid", "rel_prefix"} & set(arms) and encoder is not None) else None
    if {"bm25", "hybrid"} & set(arms):
        bm25 = make_bm25_ranker(corpus)
    if {"dense", "hybrid", "rel_prefix"} & set(arms) and enc is None:
        enc = _default_encoder()
    if {"dense", "hybrid"} & set(arms):
        dense = make_dense_ranker(corpus, enc)
    for arm in arms:
        if arm == "bm25":
            rankers[arm] = bm25
        elif arm == "dense":
            rankers[arm] = dense
        elif arm == "hybrid":
            rankers[arm] = make_hybrid_ranker(bm25, dense)
        elif arm == "reranker":
            rankers[arm] = make_reranker_ranker(corpus, cross_encoder=cross_encoder)
        elif arm == "rel_prefix":
            rankers[arm] = make_rel_prefix_ranker(corpus, enc)
        elif arm == "hcad":
            rankers[arm] = make_hcad_ranker(corpus, hcad_retriever, with_kg_metadata=False)
        elif arm == "hcad_kg":
            rankers[arm] = make_hcad_ranker(corpus, hcad_retriever, with_kg_metadata=True)
        else:
            raise ValueError(f"Unknown polarity arm {arm!r}")

    results = evaluate_rankers(rankers, queries, k=k)
    payload = {
        "benchmark": "paired_polarity_indication_vs_contraindication",
        "mode": "corpus",
        "k": k,
        "n_pairs": len(PAIRS),
        "n_queries": len(queries),
        "n_corpus_statements": len(corpus),
        "polarity_type_counts": {
            t: sum(1 for p in PAIRS if p.polarity_type == t) for t in ("contraindication", "caution")
        },
        "arms": results["arms"],
        "significance_mcnemar_polarity_error": results["significance_mcnemar_polarity_error"],
        "pairs": [asdict(p) for p in PAIRS],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = output_path or (RESULTS_DIR / "polarity_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    per_query_path = out.with_name(out.stem + "_per_query.jsonl")
    with open(per_query_path, "w", encoding="utf-8") as f:
        for arm, rows in results["per_query"].items():
            for row in rows:
                f.write(json.dumps({"arm": arm, **row}) + "\n")
    logger.info("Polarity benchmark saved to %s (+ %s)", out, per_query_path.name)
    return payload


def run_store_polarity_benchmark(arms: Sequence[str] = ("dense", "hcad"),
                                 k: int = POLARITY_TOP_K,
                                 output_path: Optional[Path] = None) -> Dict:
    """Store mode: full retrieval over the built vector store / PrimeKG."""
    retrievers: Dict[str, Callable[[str], List[Dict]]] = {}
    for arm in arms:
        if arm == "dense":
            from src.baselines.vanilla_vector_rag import VanillaVectorRAG
            dense = VanillaVectorRAG()
            retrievers[arm] = lambda q, d=dense: d.retrieve(q, top_k=k)
        elif arm == "hcad":
            from src.core.hcad_retriever import HCADRetriever
            hcad = HCADRetriever(fail_on_vector_error=True)
            retrievers[arm] = lambda q, h=hcad: h.retrieve(q)
        else:
            raise ValueError(f"Store mode supports arms 'dense' and 'hcad', got {arm!r}")
    queries = build_queries()
    results = evaluate_store_retrievers(retrievers, queries, k=k)
    payload = {
        "benchmark": "paired_polarity_indication_vs_contraindication",
        "mode": "store", "k": k, "n_pairs": len(PAIRS), "n_queries": len(queries),
        "arms": {a: {key: v for key, v in r.items() if key != "per_query"}
                 for a, r in results.items()},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = output_path or (RESULTS_DIR / "polarity_benchmark_store.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with open(out.with_name(out.stem + "_per_query.jsonl"), "w", encoding="utf-8") as f:
        for arm, r in results.items():
            for row in r["per_query"]:
                f.write(json.dumps({"arm": arm, **row}) + "\n")
    logger.info("Store-mode polarity benchmark saved to %s", out)
    return payload


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--mode", choices=["corpus", "store"], default="corpus",
                        help="corpus: controlled curated corpus (default); "
                             "store: full retrieval over the built vector store / PrimeKG")
    parser.add_argument("--arms", default=None,
                        help="corpus: bm25,dense,hybrid,hcad,hcad_kg; store: dense,hcad")
    parser.add_argument("--k", type=int, default=POLARITY_TOP_K)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.mode == "store":
        arms = [a.strip() for a in (args.arms or "dense,hcad").split(",") if a.strip()]
        payload = run_store_polarity_benchmark(arms, k=args.k, output_path=args.output)
        for arm, res in payload["arms"].items():
            o = res["overall"]
            print(f"{arm:10s} [store] PolarityRecall@{args.k}={o['polarity_recall_at_k']['mean']:.3f} "
                  f"PolarityErrorRate={o['polarity_error_rate']['mean']:.3f}")
        return
    args.arms = args.arms or "bm25,dense,hybrid,hcad,hcad_kg"
    payload = run_polarity_benchmark([a.strip() for a in args.arms.split(",") if a.strip()],
                                     k=args.k, output_path=args.output)
    for arm, res in payload["arms"].items():
        o = res["overall"]
        print(f"{arm:10s} PolarityRecall@{args.k}={o['polarity_recall_at_k']['mean']:.3f} "
              f"PolarityErrorRate={o['polarity_error_rate']['mean']:.3f} "
              f"[{o['polarity_error_rate']['ci_low']:.3f}, {o['polarity_error_rate']['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
