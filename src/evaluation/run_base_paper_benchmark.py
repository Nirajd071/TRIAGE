"""
Benchmark Runner - multi-judge, N-arm evaluation of HCAD-RAG.

Arms (config.ARM_LABELS):
  1. Llama-3.1-8B-Instruct (Zero-Retrieval / Closed-Book)
  2. BM25 RAG (top-5)
  3. Dense RAG (S-PubMedBERT, top-5)
  4. Hybrid BM25+Dense RAG (RRF, top-5)
  5. HCAD-RAG
  + optional component ablations: w/o Decoupling, w/o Soft Gram-Schmidt,
    w/o Decision Gate (``--ablations`` / BENCHMARK_ABLATIONS=1).

Every arm uses the same generator (llama3.1:8b), prompt template and
evidence budget; arms differ only in retrieval.

Outputs (results/base_paper_benchmark_results.json):
- 5-dimension rubric scores (0-100), consensus over a 4-judge local Ollama
  ensemble (Qwen3-14B, Gemma3-12B, Phi4-14B, Mistral-7B), with bootstrap 95% CIs
  per dimension.
- 8-criteria pairwise selection (AB + BA counterbalanced per judge,
  plurality vote across judges) as full win / tie / loss counts.
- Significance: paired bootstrap CI of the score difference, two-sided
  Wilcoxon signed-rank, paired t-test and exact McNemar - for consensus
  and for every judge.
- Inter-judge agreement: Cohen's kappa per judge pair, Fleiss' kappa.
- Retrieval-only latency (1 cold + N warm runs, per-stage for HCAD)
  separated from generation latency (prefill/decode, token counts,
  tokens/s) -> results/latency_breakdown.json.
- NDCG@5 / Recall@5 vs the paper-protocol gold chunk; citation precision.
"""

import json
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    ABLATION_ARMS,
    ABLATION_ENV,
    ARM_LABELS,
    ARM_SHORT_LABELS,
    BASELINE_LABEL,
    BOOTSTRAP_RESAMPLES,
    EVIDENCE_WORD_BUDGET,
    LATENCY_TIMED_RUNS,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RANDOM_SEED,
    RESULTS_DIR,
    STANDARD_ARMS,
    canonical_system_label,
)
from src.core.prompt_builder import context_document
from src.evaluation.base_paper_evaluator import (
    BasePaperEvaluator,
    MultiJudgeEvaluator,
    merge_counterbalanced,
)
from src.evaluation.metrics import citation_precision, ndcg_at_k, recall_at_k
from src.evaluation.stats import (
    bootstrap_ci,
    cohen_kappa,
    fleiss_kappa,
    mcnemar_exact,
    mean_std,
    paired_bootstrap_ci,
    paired_t_test,
    wilcoxon_signed_rank,
)
from src.wandb_tracking import log_base_paper_results, tracked_run, upload_checkpoint_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_clinical_benchmark_samples(num_samples: int = 5) -> List[Dict[str, str]]:
    """Load gold-standard clinical benchmark cases matching the base paper  protocol.

    Attempts to load from dataset files first, falls back to hardcoded samples if not available.
    """
    import json
    from pathlib import Path

    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer")

    # Try to load from processed dataset files
    try:
        # Look for MedQA or other clinical datasets
        medqa_path = Path(__file__).parent.parent.parent / "data" / "processed" / "medqa_samples.json"
        if medqa_path.exists():
            with open(medqa_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
            if isinstance(samples, list) and len(samples) >= num_samples:
                for sample in samples:
                    if isinstance(sample, dict):
                        sample.setdefault("source", "medqa_processed")
                logger.info(f"Loaded {len(samples)} clinical samples from {medqa_path}")
                return samples[:num_samples]
    except Exception as e:
        logger.debug(f"Could not load clinical samples from file: {e}")

    # Fallback to hardcoded gold-standard clinical cases
    clinical_cases = [
        {
            "question": "How should long-term treatment and motor symptom recovery of patients with Parkinson's disease and tremor be managed?",
            "reference": "Parkinson's disease (PD) motor management is rooted in restoring striatal dopaminergic signaling. First-line pharmacotherapy uses carbidopa/levodopa as the most effective symptomatic agent, supplemented by dopamine receptor agonists (pramipexole, ropinirole) and MAO-B inhibitors (rasagiline) or COMT inhibitors (entacapone) to smooth out motor fluctuations and wearing-off phenomena. For medication-refractory resting tremors and disabling motor fluctuations, surgical intervention with Deep Brain Stimulation (DBS) targeting the subthalamic nucleus (STN) or globus pallidus internus (GPi) yields significant motor improvement and reduces medication requirements. Comprehensive care also incorporates physical therapy, fall risk mitigation, and neurorehabilitation.",
            "ground_truth": "Levodopa pharmacotherapy combined with deep brain stimulation (DBS) for refractory motor tremors."
        },
        {
            "question": "What are the emergency management and risk stratification guidelines for acute ST-elevation myocardial infarction (STEMI)?",
            "reference": "Acute ST-elevation myocardial infarction (STEMI) represents complete transmural myocardial ischemia typically caused by acute coronary atherothrombotic occlusion. Emergency management mandates immediate reperfusion therapy: primary percutaneous coronary intervention (PCI) is preferred if door-to-balloon time is under 90 minutes (or under 120 minutes if transferred); intravenous fibrinolysis is indicated within 30 minutes if PCI is unavailable within 120 minutes. Concomitant pharmacotherapy includes immediate dual antiplatelet therapy (aspirin plus a potent P2Y12 inhibitor such as ticagrelor or prasugrel), unfractionated heparin or bivalirudin anticoagulation, sublingual nitroglycerin for ischemic pain, and high-intensity statin therapy. Risk stratification incorporates TIMI or GRACE risk scores and assessment for cardiogenic shock or mechanical complications.",
            "ground_truth": "Immediate primary PCI within 90 minutes or fibrinolysis within 30 minutes, combined with dual antiplatelet therapy, anticoagulation, and statins."
        },
        {
            "question": "What is the stepwise diagnostic and therapeutic algorithm for acute severe exacerbation of bronchial asthma?",
            "reference": "Acute severe asthma exacerbation is marked by acute bronchospasm, airway inflammation, and ventilation-perfusion mismatch. Emergency therapeutic protocol begins with high-concentration supplemental oxygen (titrated to SpO2 93-95%), repetitive or continuous nebulized short-acting beta-2 agonists (albuterol/salbutamol) combined with inhaled anticholinergics (ipratropium bromide), and early administration of systemic corticosteroids (oral prednisone or IV methylprednisolone) to reduce airway edema. For refractory severe bronchospasm, intravenous magnesium sulfate is administered. Clinical monitoring includes serial peak expiratory flow (PEF) measurements, arterial blood gas analysis to detect hypercapnia and impending respiratory muscle fatigue, and preparation for non-invasive positive pressure ventilation or endotracheal intubation if respiratory arrest is imminent.",
            "ground_truth": "Continuous inhaled beta-agonists plus ipratropium, early systemic corticosteroids, supplemental oxygen, and IV magnesium sulfate for refractory bronchospasm."
        },
        {
            "question": "How should cognitive and behavioral symptoms in moderate-to-severe Alzheimer's dementia be evaluated and managed?",
            "reference": "Alzheimer's disease management encompasses cognitive stabilization, neuropsychiatric symptom control, and caregiver support. Pharmacological therapy for moderate-to-severe dementia combines a cholinesterase inhibitor (donepezil, rivastigmine, or galantamine) with the NMDA receptor antagonist memantine, which provides neuroprotection against excitotoxicity. Behavioral and psychological symptoms of dementia (BPSD) including agitation, psychosis, and sleep disruption require initial non-pharmacological behavioral modifications and environmental optimization. Low-dose atypical antipsychotics are reserved strictly for severe distressing agitation or psychosis unresponsive to non-pharmacological measures, with close cardiac and metabolic monitoring due to black box mortality warnings.",
            "ground_truth": "Combination of cholinesterase inhibitors and memantine with non-pharmacological behavioral management for neuropsychiatric symptoms."
        },
        {
            "question": "What are the diagnostic criteria and management strategies for type 2 diabetes mellitus?",
            "reference": "Type 2 diabetes mellitus is characterized by insulin resistance and relative insulin deficiency. Diagnostic criteria include HbA1c ≥6.5%, fasting plasma glucose ≥126 mg/dL, 2-hour plasma glucose ≥200 mg/dL during OGTT, or random plasma glucose ≥200 mg/dL with classic symptoms. Management focuses on lifestyle modification with medical nutrition therapy, physical activity, and pharmacotherapy. First-line therapy is metformin, with addition of other agents (SGLT2 inhibitors, GLP-1 receptor agonists, DPP-4 inhibitors, sulfonylureas, or insulin) based on HbA1c targets, patient comorbidities, and cost considerations. Comprehensive management includes cardiovascular risk reduction, blood pressure control, lipid management, and regular monitoring for complications.",
            "ground_truth": "Metformin as first-line therapy with comprehensive cardiovascular risk reduction and regular complication monitoring."
        }
    ]
    for sample in clinical_cases:
        sample.setdefault("source", "curated")

    # the MedQA fallback was removed - the base paper never
    # uses MCQ benchmarks, and mixing question formats made aggregate scores
    # heterogeneous. Oversize requests are truncated to the curated pool.
    return clinical_cases[:num_samples]

def _merge_selection_results(ab_result: Dict, ba_result: Dict, target_label: str = "") -> Dict:
    """Merge AB and BA pairwise evaluations to counterbalance position bias."""
    return merge_counterbalanced(ab_result, ba_result)

def _validate_generated_response(system_name: str, response: str) -> str:
    """Stop the benchmark instead of scoring an error message as an answer."""
    if not isinstance(response, str) or not response.strip():
        raise RuntimeError(f"{system_name} returned an empty or non-text response")
    cleaned = response.strip()
    if cleaned.startswith("[Error:"):
        raise RuntimeError(f"{system_name} generation failed: {cleaned}")
    return cleaned

_ARM_CITATION_TAG = re.compile(r"\[(?:Evidence|Context)(\s*\d[\d,\s\-–]*)\]")


def _blind_citation_tags(response: str) -> str:
    """Rewrite arm-specific citation tags to the judge's neutral [Doc i] tags.

    Dense/BM25/hybrid answers cite [Context i] and HCAD answers cite
    [Evidence i]; the judge sees every evidence block tagged [Doc i]. Left
    as-is, the tag word reveals which system wrote an answer (unblinding the
    pairwise judge). Indices are unchanged, so [Doc i] still points at the
    same item. Applied only to judge inputs; citation precision is computed
    on the raw response.
    """
    return _ARM_CITATION_TAG.sub(r"[Doc\1]", response)


def _format_retrieved_evidence(contexts: List[Dict[str, Any]], tag: str) -> str:
    """Expose answer-specific retrieval evidence to the judge with a word cap."""
    if not contexts:
        return "No retrieved evidence was supplied for this answer."

    lines = []
    remaining_words = EVIDENCE_WORD_BUDGET
    for index, context in enumerate(contexts, 1):
        metadata = context.get("metadata") or {}
        document = context.get("document") or context.get("text") or ""
        if not document:
            head = metadata.get("head_name", "")
            relation = metadata.get("display_relation") or metadata.get("relation", "")
            tail = metadata.get("tail_name", "")
            document = f"{head} {relation} {tail}".strip()
        words = str(document).split()
        if not words or remaining_words <= 0:
            break
        # clip at the last sentence boundary within the
        # budget - mid-sentence clipping can sever negations/qualifiers.
        budget_text = " ".join(words[:remaining_words])
        cut = max(budget_text.rfind(". "), budget_text.rfind("! "), budget_text.rfind("? "))
        selected = budget_text if cut == -1 else budget_text[: cut + 1]
        lines.append(f"[{tag} {index}] {selected.strip()}")
        used = len(selected.split())
        remaining_words -= used
    return "\n".join(lines) or "Retrieved contexts contained no readable evidence."

# Both systems retrieve top-5; all retrieval metrics use the same cut.
RETRIEVAL_K = 5
# Paired correctness threshold for McNemar: overall_score >= 70 of 100.
CORRECTNESS_THRESHOLD = 70.0

_STOPWORDS = frozenset(
    "the a an and or of to in for with on by is are was were be been this that "
    "these those it its as at from may can which who whom what when where how "
    "not no than then also such into during before after above below over under "
    "between within without about against each other more most less least any "
    "all some both few many much very own same so if while because through "
    "their there here they them we you he she his her our your my me us him".split()
)

_context_document = context_document

def _content_tokens(text: str) -> set:
    """Lowercased content tokens; stop- and short tokens dropped."""
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(token) >= 3 and token not in _STOPWORDS
    }

def _ranked_relevance(
    reference: str,
    contexts: List[Dict[str, Any]],
    overlap_threshold: float = 0.3,
) -> List[int]:
    """Binary relevance of retrieved contexts against the gold source chunk.

    The paper protocol generates each question FROM a source chunk, so that
    chunk is the free gold relevance label. A retrieved context is relevant
    when its content-token Jaccard overlap with the reference reaches the
    threshold (1.0 for the chunk itself; paraphrases stay well above it
    while short fragments sharing a word or two do not).
    """
    ref_tokens = _content_tokens(reference)
    relevance: List[int] = []
    if not ref_tokens:
        return [0] * len(contexts)
    for context in contexts:
        # A grouped hyperedge line ("X is indicated for: A; B; C.") contains
        # several triples; it is relevant when any member fact matches.
        texts = [_context_document(context)] + list(context.get("member_documents") or [])
        best = 0.0
        for text in texts:
            doc_tokens = _content_tokens(text)
            union = len(ref_tokens | doc_tokens)
            best = max(best, len(ref_tokens & doc_tokens) / union if union else 0.0)
        relevance.append(1 if best >= overlap_threshold else 0)
    return relevance

#: Legacy (schema-1, three-arm) selection keys -> (arm_a, arm_b).
_LEGACY_COMPARISONS = {
    "HCAD_vs_Vanilla": ("hcad", "dense"),
    "HCAD_vs_Baseline": ("hcad", "closed_book"),
    "Vanilla_vs_Baseline": ("dense", "closed_book"),
}


def comparison_key(arm_a: str, arm_b: str) -> str:
    return f"{arm_a}_vs_{arm_b}"


def _normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade a checkpoint record to schema 2 (multi-judge, N-arm).

    Schema-1 records (single judge, labels "Baseline LLM"/"Vanilla RAG",
    flat score dicts, float latencies) are mapped onto the current labels
    so old checkpoints aggregate with the same code - and can never
    re-introduce the retired generic baseline name into a table.
    """
    if record.get("schema") == 2:
        return record
    rec = dict(record)
    scores: Dict[str, Any] = {}
    for system, sc in (record.get("scores") or {}).items():
        label = canonical_system_label(system)
        if isinstance(sc, dict) and "consensus" in sc:
            scores[label] = sc
        else:
            scores[label] = {"consensus": sc, "per_judge": {"legacy_single_judge": sc}}
    rec["scores"] = scores

    selection: Dict[str, Any] = {}
    for key, sel in (record.get("selection") or {}).items():
        if key in _LEGACY_COMPARISONS and isinstance(sel, dict) and "consensus" not in sel:
            a, b = _LEGACY_COMPARISONS[key]
            verdict = {
                "criteria_winners": {c: canonical_system_label(w) for c, w in
                                     (sel.get("criteria_winners") or {}).items()},
                "overall_winner": canonical_system_label(sel.get("overall_winner", "Tie")),
            }
            selection[comparison_key(a, b)] = {
                "system_a": ARM_LABELS[a], "system_b": ARM_LABELS[b],
                "consensus": verdict, "per_judge": {"legacy_single_judge": verdict},
            }
        else:
            selection[key] = sel
    rec["selection"] = selection

    latency: Dict[str, Any] = {}
    for system, lat in (record.get("latency") or {}).items():
        label = canonical_system_label(system)
        if isinstance(lat, dict):
            latency[label] = lat
        else:
            latency[label] = {"t_retrieval": float(lat), "t_retrieval_runs": [float(lat)]}
    rec["latency"] = latency
    rec["retrieval"] = {canonical_system_label(k): v
                        for k, v in (record.get("retrieval") or {}).items()}
    rec["citations"] = {canonical_system_label(k): v
                        for k, v in (record.get("citations") or {}).items()}
    rec["schema"] = 2
    return rec


def _evidence_stats(contexts: List[Dict[str, Any]], relevance: List[int]) -> Dict[str, Any]:
    """What an arm actually handed the generator (diagnostics only).

    Counts words within the shared EVIDENCE_WORD_BUDGET, prose vs KG items,
    and the rank of the first gold-matching item - the numbers needed to
    tell an evidence-selection loss from a generation/judging effect.
    """
    words, n_prose, n_kg = 0, 0, 0
    for context in contexts:
        text = _context_document(context)
        if words < EVIDENCE_WORD_BUDGET:
            words += min(len(text.split()), EVIDENCE_WORD_BUDGET - words)
        meta = context.get("metadata") or {}
        if meta.get("relation") or context.get("source") in ("graph_hyperedge", "nary_fact"):
            n_kg += 1
        else:
            n_prose += 1
    gold_rank = next((i + 1 for i, r in enumerate(relevance) if r), None)
    return {"n_items": len(contexts), "n_prose": n_prose, "n_kg": n_kg,
            "words": words, "gold_rank": gold_rank,
            "n_kg_facts": sum(len((c.get("metadata") or {}).get("grouped_tails") or [None])
                              for c in contexts
                              if (c.get("metadata") or {}).get("relation"))}


def _hcad_config_id() -> str:
    """Fingerprint of the HCAD evidence-assembly and model settings.

    HCAD answers produced under different evidence or generator logic must not be
    resumed into (or averaged with) the current run. Ablation arms are
    expressed through the same env toggles inside their own scope, so the
    fingerprint is taken from the process-level (full-system) settings.
    """
    from src.config import (
        LLM_MODEL,
        dense_anchor_k,
        dense_anchor_min,
        dense_weight,
        expansion_rule,
        gate_min_relevance,
        group_kg_facts_enabled,
        lex_weight,
        ortho_strength,
        decision_gate_enabled,
        polarity_scoring_enabled,
        presence_masking_enabled,
        relation_vocab_enabled,
        tau_sat,
    )
    t_mode = os.getenv("HCAD_TENSOR_MODE", "static").strip().lower()
    return (
        f"evidence-v3|gen={LLM_MODEL}|dw={dense_weight():g}"
        f"|lw={lex_weight():g}|pol={int(polarity_scoring_enabled())}"
        f"|anchor={dense_anchor_k()}@{dense_anchor_min():g}"
        f"|group={int(group_kg_facts_enabled())}|tm={t_mode}"
        f"|tau={tau_sat()}|exp={expansion_rule()}"
        f"|mask={int(presence_masking_enabled())}|rv={int(relation_vocab_enabled())}"
        f"|ortho={ortho_strength():g}|gate={int(decision_gate_enabled())}"
        f"|gmin={gate_min_relevance():g}"
    )


def _config_matches(record: Dict[str, Any],
                    required_judges: Optional[Sequence[str]]) -> bool:
    """A record counts only for the judge panel AND HCAD config that made it.

    Switching JUDGE_MODEL_INDEX (or the HCAD evidence settings) and
    re-running must NOT resume or aggregate another configuration's scores.
    Inactive (always True) when ``required_judges`` is None.
    """
    if required_judges is None:
        return True
    return (list(record.get("judges") or []) == list(required_judges)
            and record.get("hcad_config_id") == _hcad_config_id())


def _records_from_checkpoint(
    checkpoint_path: Path, sample_questions: set, question_pool: Optional[str] = None,
    question_set_id: Optional[str] = None,
    required_judges: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Load the newest per-question record for every question in this run.

    Aggregating from the checkpoint (not session memory) keeps resumed runs
    correct: questions evaluated before the interruption keep their records,
    and stale questions from older runs with different sample sets are
    filtered out. With ``required_judges``, only records of exactly that
    judge panel and the current HCAD config are used. Records are
    normalised to schema 2 on read.
    """
    by_question: Dict[str, Dict[str, Any]] = {}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (record.get("question") in sample_questions
                        and (question_pool is None
                             or record.get("question_pool") == question_pool)):
                    if ((question_set_id is None
                            or record.get("question_set_id") == question_set_id)
                            and _config_matches(record, required_judges)):
                        by_question[record["question"]] = _normalize_record(record)  # later lines win

    labels = list(ARM_LABELS.values())
    scoring: Dict[str, List[Dict[str, Any]]] = {label: [] for label in labels}
    selection: Dict[str, List[Dict[str, Any]]] = {}
    latency: Dict[str, List[Dict[str, Any]]] = {label: [] for label in labels}
    retrieval: Dict[str, List[Dict[str, Any]]] = {label: [] for label in labels}
    citations: Dict[str, List[Dict[str, Any]]] = {label: [] for label in labels}
    scoring_by_tier: Dict[int, Dict[str, List[Dict[str, Any]]]] = {}
    scoring_by_source: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    selection_by_source: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for record in by_question.values():
        for system, score in (record.get("scores") or {}).items():
            consensus = (score or {}).get("consensus")
            if not consensus:
                continue  # every judge failed: missing, never imputed
            scoring.setdefault(system, []).append(consensus)
            scoring_by_tier.setdefault(
                int(record.get("difficulty_tier", 1) or 1), {}
            ).setdefault(system, []).append(consensus)
            scoring_by_source.setdefault(
                record.get("source", "curated"), {}
            ).setdefault(system, []).append(consensus)
        for comparison, sel in (record.get("selection") or {}).items():
            selection.setdefault(comparison, []).append(sel)
            selection_by_source.setdefault(
                record.get("source", "curated"), {}
            ).setdefault(comparison, []).append(sel)
        for system, lat in (record.get("latency") or {}).items():
            latency.setdefault(system, []).append(lat)
        for system, ret in (record.get("retrieval") or {}).items():
            if ret:
                retrieval.setdefault(system, []).append(ret)
        for system, cit in (record.get("citations") or {}).items():
            citations.setdefault(system, []).append(cit)

    return {
        "scoring": scoring,
        "selection": selection,
        "latency": latency,
        "retrieval": retrieval,
        "citations": citations,
        "scoring_by_tier": scoring_by_tier,
        "scoring_by_source": scoring_by_source,
        "selection_by_source": selection_by_source,
        "per_question": list(by_question.values()),
    }

QUESTION_GEN_PROMPT = (
    "You are given a passage from a clinical knowledge base. Generate one "
    "examination-style, open-ended question that this passage fully answers.\n"
    "{tier}\n"
    "Return ONLY the question text.\n\nPassage:\n{chunk}"
)
QUESTION_GEN_PROMPT_ASSOC = (
    "You are given a single factual statement from a biomedical knowledge graph "
    "describing one association (for example, between a drug and a protein, a "
    "disease and a phenotype, or two drugs). Generate one examination-style, "
    "open-ended question that this statement fully answers.\n"
    "Rules: ask only about what the statement itself supports; do NOT ask about "
    "mechanisms, processes, or clinical management that the statement does not "
    "describe. Any follow-up aspect must remain answerable from the statement.\n"
    "{tier}\n"
    "Return ONLY the question text.\n\nStatement:\n{chunk}"
)
TIERS = {
    1: "Ask a single focused question.",
    2: "Ask a two-stage question: first one aspect, then 'and specifically' a second, dependent aspect.",
    3: "Ask a three-stage nested question chaining three dependent aspects with connecting phrases.",
}

# Stratified sampling: question sources are drawn across these PrimeKG relation
# types plus the PMC clinical-text corpus, instead of the first N documents of
# the vector store (which are all contiguous PrimeKG drug_protein triples -
# the root cause of the degenerate 8-fact pilot question set).
# Relation vocabulary verified against the built store's sqlite
# (data/vectordb/chroma.sqlite3): contraindication, disease_disease,
# disease_phenotype_negative, disease_phenotype_positive, disease_protein,
# drug_effect, drug_protein, indication, off-label use. Strata absent from
# the store would simply be skipped, but keeping the list exact avoids
# wasted fetch attempts and guarantees every stratum contributes.
TRIPLE_RELATION_STRATA = [
    "drug_protein", "contraindication", "indication", "drug_effect",
    "disease_protein", "disease_phenotype_positive",
    "disease_phenotype_negative", "disease_disease", "off-label use",
]

def _question_pool_of(doc_id: str, pool: str) -> bool:
    """Deterministic dev/test pool assignment by article/entity key hash.

    The key is the SAME one used for per-chunk dedup: the parent article
    (doc_id) for PMC prose and the head entity (head_name) for PrimeKG
    triples. This keeps the pools disjoint at the article/entity level, so
    no single clinical study or drug contributes questions to both the
    pool used to train/tune HCAD components and the held-out evaluation
    pool (no data leakage, Critical Review V3).
    """
    import hashlib
    h = int(hashlib.sha1(str(doc_id).encode("utf-8")).hexdigest()[:8], 16)
    want = 0 if pool == "test" else 1
    return h % 2 == want

def _stratum_size(collection, where: dict, upper_bound: int) -> Optional[int]:
    """Find the filtered row count using offset probes, or None if unsupported."""
    low, high = 0, upper_bound
    while low < high:
        mid = (low + high) // 2
        try:
            got = collection.get(where=where, limit=1, offset=mid,
                                 include=["documents", "metadatas"])
        except TypeError:
            return None
        if (got or {}).get("documents"):
            low = mid + 1
        else:
            high = mid
    return low

def _paper_question_cache_path(num_samples: int, pool: str) -> Path:
    if pool not in ("test", "dev"):
        raise ValueError(f"pool must be 'test' or 'dev', got {pool!r}")
    return RESULTS_DIR / f"paper_questions_n{num_samples}_seed{RANDOM_SEED}_{pool}.json"

def _question_set_id(samples: list[dict]) -> str:
    import hashlib

    contract = [(sample["question"], sample.get("source_chunk_id"), sample.get("pool"))
                for sample in samples]
    payload = json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def _completed_questions_from_checkpoint(
    checkpoint_path: Path, pool: str, question_set_id: str,
    required_arms: Optional[Sequence[str]] = None,
    required_judges: Optional[Sequence[str]] = None,
) -> set[str]:
    """Questions that need no re-evaluation in this run.

    When ``required_arms`` / ``required_judges`` are given, only schema-2
    records that cover every requested arm and were scored by exactly the
    configured judges count as complete. Older records (single-judge,
    three-arm checkpoints restored from a previous session) are re-run
    instead of being silently mixed into the new protocol.
    """
    completed: set[str] = set()
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A killed session can leave a truncated final line; the
                    # aggregation path already skips those - resume must not
                    # crash on the exact failures it exists to recover from.
                    continue
                if (record.get("question_pool") == pool
                        and record.get("question_set_id") == question_set_id):
                    if required_arms is not None or required_judges is not None:
                        if record.get("schema") != 2:
                            continue
                        if required_arms is not None and not set(required_arms) <= set(
                                (record.get("arms") or {}).keys()):
                            continue
                        if required_judges is not None and not _config_matches(
                                record, required_judges):
                            continue
                        scores = record.get("scores") or {}
                        if scores and not any(
                            (v or {}).get("n_valid_judges", 1) > 0 for v in scores.values()
                        ):
                            continue
                    completed.add(record["question"])
    return completed

def _grounded_mode_counts(
    checkpoint_path: Path, pool: str, question_set_id: str
) -> Dict[str, int]:
    """P2d provenance: grounded vs parametric HCAD answers for this sample set.

    Read from the checkpoint so resumed runs aggregate the full set, not
    just the questions evaluated in the current session.
    """
    counts = {"grounded": 0, "parametric": 0}
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (record.get("question_pool") == pool
                        and record.get("question_set_id") == question_set_id):
                    mode = record.get("grounded_mode")
                    if mode in counts:
                        counts[mode] += 1
    return counts

def _fetch_stratified_chunks(collection, num_docs: int, pool: str, rng) -> list[dict]:
    """Fetch candidate source chunks stratified by corpus segment.

    PMC clinical-text chunks and each PrimeKG relation type form separate
    strata. Within each stratum MULTIPLE random windows are drawn,
    round-robin across strata, until enough distinct pool-qualified chunks
    are collected: PrimeKG's kg.csv is ordered by head drug, so a single
    contiguous window is almost one entity and the old one-window-per-
    stratum draw produced only ~50 usable chunks from the 817k store.
    Each stratum's offset span shrinks adaptively when a window lands
    past the end of that stratum. Chunks are deduplicated per entity
    (head_name) / article (doc_id) so a single drug cannot dominate, and
    filtered to the requested dev/test pool.
    """
    strata = [{"where": {"source": "pmc"}, "kind": "pmc",
               "span": None, "no_offset": False}]
    strata += [{"where": {"relation": rel}, "kind": "primekg",
                "span": None, "no_offset": False}
               for rel in TRIPLE_RELATION_STRATA]
    total = collection.count()
    for s in strata:
        s["span"] = _stratum_size(collection, s["where"], total)
        if s["span"] is None:
            s["no_offset"] = True
    target = num_docs * 2
    per = 64
    out: list[dict] = []
    seen: set[str] = set()
    for _round in range(24):
        if len(out) >= target:
            break
        for s in strata:
            if len(out) >= target:
                break
            if s["no_offset"]:
                try:
                    got = collection.get(where=s["where"], limit=per,
                                         include=["documents", "metadatas"])
                except TypeError:
                    got = None
            else:
                span = s["span"]
                if not span:
                    continue
                offset = rng.randrange(0, max(1, span - per + 1))
                try:
                    got = collection.get(where=s["where"], limit=per,
                                         offset=offset,
                                         include=["documents", "metadatas"])
                except TypeError:
                    # chroma build without offset support: head window only
                    s["no_offset"] = True
                    try:
                        got = collection.get(where=s["where"], limit=per,
                                             include=["documents", "metadatas"])
                    except TypeError:
                        got = None
                else:
                    if not (got or {}).get("documents") and offset > 0:
                        # window landed past this stratum's end - shrink span
                        s["span"] = max(per, offset)
                        continue
            docs = (got or {}).get("documents") or []
            metas = (got or {}).get("metadatas") or [{}] * len(docs)
            ids = (got or {}).get("ids") or [str(i) for i in range(len(docs))]
            for d, m, i in zip(docs, metas, ids):
                if not d or len(d) < 30:
                    continue
                if s["kind"] == "pmc" and len(d) < 200:
                    continue
                if s["kind"] == "primekg" and len(d) > 300:
                    continue
                key = str(m.get("doc_id") or m.get("head_name") or i)
                if key in seen:
                    continue
                # Pool assignment must run on the SAME key as dedup (article
                # doc_id for PMC, head entity for PrimeKG): hashing raw chunk
                # ids let chunks of one article / one drug straddle the dev
                # and test pools (Critical Review V3 leakage vector).
                if not _question_pool_of(key, pool):
                    continue
                seen.add(key)
                out.append({"doc": d, "id": str(i), "kind": s["kind"],
                            "relation": str(m.get("relation") or ""),
                            # the article/entity key the pool was assigned by
                            # (provenance for leakage audits and for tests)
                            "pool_key": key})
    rng.shuffle(out)
    return out

def generate_paper_protocol_questions(num_samples: int, generator=None,
                                      pool: str = "test") -> list[dict]:
    """Base-paper protocol (Sec. 3.5): questions are GENERATED from source
    chunks by an LLM, and the SOURCE CHUNK is the scoring reference.

    Fairness properties:
    - chunks are drawn stratified across PMC text and PrimeKG relation types
      (not the head of the store), deduplicated per entity/article;
    - each question comes from a DISTINCT chunk (tiers get disjoint slices -
      previously every tier rephrased the same 8 facts);
    - chunks are assigned to the dev or test pool by id hash, so training/
      tuning on the dev pool can never leak into the evaluation pool;
    - questions from short KG-triple statements use an association-style
      prompt that forbids asking mechanisms the statement cannot answer.

    The generated set is CACHED to disk (keyed by sample count, seed, pool).
    Generation samples at temperature 0.3, so without the cache a resumed
    run would produce different questions and silently discard every
    checkpointed answer. With it, interrupted runs resume exactly.
    """
    from src.core.llm_client import UnifiedLLMClient
    from src.data.build_vector_store import load_vector_store
    import random as _random

    if pool not in ("test", "dev"):
        raise ValueError(f"pool must be 'test' or 'dev', got {pool!r}")
    cache_path = _paper_question_cache_path(num_samples, pool)
    legacy_path = RESULTS_DIR / f"paper_questions_n{num_samples}_seed{RANDOM_SEED}.json"
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if (len(cached) == num_samples and all(
                    "question" in s and "reference" in s
                    and s.get("pool") == pool
                    and s.get("source") == "paper_protocol"
                    and s.get("source_chunk_id")
                    for s in cached)):
                logger.info("Paper-protocol questions loaded from cache: %d (pool=%s).",
                            len(cached), pool)
                return cached
        except Exception as exc:
            logger.warning("Question cache unreadable (%s); regenerating.", exc)
    if legacy_path.exists():
        logger.warning("Ignoring unscoped legacy question cache %s; regenerating pool=%s.",
                       legacy_path.name, pool)

    if generator is None:
        generator = UnifiedLLMClient(temperature=0.3)
    collection = load_vector_store()
    total = collection.count()
    if total == 0:
        raise RuntimeError("Vector store is empty - run `python src/main.py prepare` first.")
    rng = _random.Random(RANDOM_SEED)
    chunks = _fetch_stratified_chunks(collection, num_samples * 2, pool, rng)
    if len(chunks) < num_samples:
        raise RuntimeError(
            f"Only {len(chunks)} distinct pool-{pool} source chunks available "
            f"for {num_samples} questions; regenerate with a different pool size."
        )

    per_tier = max(1, -(-num_samples // 3))
    samples, tier_sizes = [], {1: 0, 2: 0, 3: 0}
    cursor = 0
    for tier in (1, 2, 3):
        while tier_sizes[tier] < per_tier and len(samples) < num_samples:
            if cursor >= len(chunks):
                raise RuntimeError("Ran out of distinct source chunks during generation.")
            chunk = chunks[cursor]
            cursor += 1
            prompt = (QUESTION_GEN_PROMPT if chunk["kind"] == "pmc"
                     else QUESTION_GEN_PROMPT_ASSOC)
            try:
                q = generator.chat([{"role": "user", "content": prompt.format(
                    tier=TIERS[tier], chunk=chunk["doc"][:2500])}]).strip()
            except Exception:
                continue
            if len(q) < 25:
                continue
            samples.append({"question": q, "reference": chunk["doc"],
                            "ground_truth": "", "source": "paper_protocol",
                            "difficulty_tier": tier, "pool": pool,
                            "source_kind": chunk["kind"],
                            "source_chunk_id": chunk["id"],
                            "source_relation": chunk["relation"]})
            tier_sizes[tier] += 1
    logger.info("Paper-protocol questions: %s (pool=%s, kinds=%s)", tier_sizes, pool,
                {k: sum(1 for s in samples if s["source_kind"] == k)
                 for k in ("pmc", "primekg")})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    logger.info("Paper-protocol questions cached to %s (resume-safe).", cache_path)
    return samples

# ---------------------------------------------------------------------------
# Benchmark arms
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkArm:
    """One system under test. ``retrieve is None`` means closed-book."""
    arm_id: str
    label: str
    retrieve: Optional[Callable[[str], List[Dict[str, Any]]]]
    build_messages: Callable[[str, List[Dict[str, Any]]], List[Dict[str, str]]]
    env: Dict[str, str] = field(default_factory=dict)
    stage_timings: Optional[Callable[[], Dict[str, float]]] = None


@contextmanager
def _env_overrides(env: Dict[str, str]):
    """Apply an ablation's environment toggles for the duration of a call."""
    previous = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def resolve_arms(arms: Optional[Sequence[str]] = None,
                 include_ablations: Optional[bool] = None) -> List[str]:
    """Arms to run: explicit list > BENCHMARK_ARMS env > standard arms.

    HCAD-RAG is always included (every pairwise comparison is against it).
    """
    if arms is None:
        env_arms = os.getenv("BENCHMARK_ARMS", "").strip()
        arms = [a.strip() for a in env_arms.split(",") if a.strip()] or list(STANDARD_ARMS)
    if include_ablations is None:
        include_ablations = os.getenv("BENCHMARK_ABLATIONS", "0") == "1"
    out = list(dict.fromkeys(arms))
    if include_ablations:
        out += [a for a in ABLATION_ARMS if a not in out]
    unknown = [a for a in out if a not in ARM_LABELS]
    if unknown:
        raise ValueError(f"Unknown benchmark arms {unknown}; valid: {sorted(ARM_LABELS)}")
    if "hcad" not in out:
        out.append("hcad")
    return out


def selection_pairs(arm_ids: Sequence[str]) -> List[tuple]:
    """HCAD-RAG vs every other arm, plus Dense vs Closed-Book (retrieval gain)."""
    pairs = [("hcad", other) for other in arm_ids if other != "hcad"]
    if "dense" in arm_ids and "closed_book" in arm_ids:
        pairs.append(("dense", "closed_book"))
    return pairs


def build_arms(arm_ids: Sequence[str]) -> Dict[str, BenchmarkArm]:
    """Instantiate the requested systems (heavy resources loaded once)."""
    from src.baselines.baseline_llm import BaselineLLM
    from src.baselines.vanilla_vector_rag import VanillaVectorRAG, build_rag_messages

    arms: Dict[str, BenchmarkArm] = {}
    need_dense = bool({"dense", "hybrid", "bm25", "reranker", "graphrag"} & set(arm_ids))
    vanilla = VanillaVectorRAG() if need_dense else None
    bm25_index = None
    if {"bm25", "hybrid", "reranker"} & set(arm_ids):
        from src.baselines.sparse_retrievers import BM25Index
        logger.info("Building BM25 index over the vector-store corpus...")
        bm25_index = BM25Index.from_collection(vanilla.collection)

    if "closed_book" in arm_ids:
        arms["closed_book"] = BenchmarkArm(
            "closed_book", ARM_LABELS["closed_book"], None, BaselineLLM.build_messages)
    if "dense" in arm_ids:
        arms["dense"] = BenchmarkArm(
            "dense", ARM_LABELS["dense"],
            lambda q: vanilla.retrieve(q, top_k=RETRIEVAL_K), build_rag_messages)
    if "bm25" in arm_ids:
        from src.baselines.sparse_retrievers import BM25RAG
        bm25 = BM25RAG(bm25_index, top_k=RETRIEVAL_K)
        arms["bm25"] = BenchmarkArm("bm25", ARM_LABELS["bm25"], bm25.retrieve,
                                    build_rag_messages)
    hybrid = None
    if {"hybrid", "reranker"} & set(arm_ids):
        from src.baselines.sparse_retrievers import HybridRAG
        hybrid = HybridRAG(bm25_index, lambda q, n: vanilla.retrieve(q, top_k=n),
                           top_k=RETRIEVAL_K)
    if "hybrid" in arm_ids:
        arms["hybrid"] = BenchmarkArm("hybrid", ARM_LABELS["hybrid"], hybrid.retrieve,
                                      build_rag_messages)
    if "reranker" in arm_ids:
        from src.baselines.cross_encoder_rag import CrossEncoderRerankRAG
        reranker = CrossEncoderRerankRAG(
            candidate_fn=lambda q, n: hybrid.retrieve(q, top_k=n),
            candidate_k=max(20, RETRIEVAL_K * 4),
            top_k=RETRIEVAL_K,
        )
        arms["reranker"] = BenchmarkArm("reranker", ARM_LABELS["reranker"],
                                        reranker.retrieve, build_rag_messages)

    hcad_ids = [a for a in arm_ids if a == "hcad" or a in ABLATION_ARMS]
    if hcad_ids or "graphrag" in arm_ids:
        from src.core.hcad_retriever import HCADRetriever

        hcad = HCADRetriever(fail_on_vector_error=True)
        hcad._ensure_components()
        _ = hcad.graph
        _ = hcad.hypergraph
        if getattr(getattr(hcad, "_ner", None), "degraded", False):
            raise RuntimeError(
                "SciSpacy NER failed to load (BiomedicalNER in degraded regex "
                "mode). Refusing to run base paper benchmark on degraded features."
            )
        if "graphrag" in arm_ids:
            from src.baselines.graph_rag import GraphRAGBaseline
            graphrag = GraphRAGBaseline(
                dense_retrieve_fn=lambda q, n: vanilla.retrieve(q, top_k=n),
                encode_fn=lambda texts: vanilla.embedding_model.encode(texts, convert_to_numpy=True),
                ner=hcad._ner,
                entity_linker=hcad._entity_linker,
                hypergraph=hcad.hypergraph,
                top_k=RETRIEVAL_K,
            )
            arms["graphrag"] = BenchmarkArm("graphrag", ARM_LABELS["graphrag"],
                                            graphrag.retrieve, build_rag_messages)
        for arm_id in hcad_ids:
            arms[arm_id] = BenchmarkArm(
                arm_id, ARM_LABELS[arm_id], hcad.retrieve, hcad.build_messages,
                env=dict(ABLATION_ENV.get(arm_id, {})),
                stage_timings=lambda h=hcad: dict(h._last_stage_timings or {}))
        if "hcad" in arms:
            arms["hcad"]._retriever = hcad  # type: ignore[attr-defined]
    return {a: arms[a] for a in arm_ids}


def timed_retrieval(arm: BenchmarkArm, query: str, runs: int) -> tuple:
    """Retrieval-only latency: 1 cold run + ``runs`` warm timed runs.

    Returns (contexts_of_cold_run, latency_dict). Generation is NOT
    included (reviewers: the 4.18x claim mixed retrieval with decode
    length and abstention).
    """
    if arm.retrieve is None:
        return [], {"t_retrieval": 0.0, "t_retrieval_runs": [], "t_retrieval_cold": 0.0,
                    "stages": {}, "n_contexts": 0, "abstained": False}
    if runs <= 0:
        raise ValueError("latency runs must be positive")
    with _env_overrides(arm.env):
        t0 = time.perf_counter()
        contexts = arm.retrieve(query)
        cold = time.perf_counter() - t0
        warm, stage_runs = [], []
        for _ in range(runs):
            t0 = time.perf_counter()
            arm.retrieve(query)
            warm.append(time.perf_counter() - t0)
            if arm.stage_timings is not None:
                stage_runs.append(arm.stage_timings())
    stages = {}
    if stage_runs:
        keys = sorted({k for s in stage_runs for k in s})
        stages = {k: sum(s.get(k, 0.0) for s in stage_runs) / len(stage_runs) for k in keys}
    mean_warm = sum(warm) / len(warm)
    return contexts, {
        "t_retrieval": mean_warm,
        "t_retrieval_runs": warm,
        "t_retrieval_cold": cold,
        "stages": stages,
        "n_contexts": len(contexts or []),
        "abstained": not contexts,
    }


def generate_answer(arm: BenchmarkArm, query: str, contexts: List[Dict[str, Any]]) -> tuple:
    """Generation-only latency + token accounting (prefill + decode)."""
    from src.core.llm_client import UnifiedLLMClient

    client = UnifiedLLMClient(model=LLM_MODEL, temperature=LLM_TEMPERATURE,
                              max_tokens=LLM_MAX_TOKENS)
    text, stats = client.chat_with_stats(arm.build_messages(query, contexts))
    return _validate_generated_response(arm.label, text), stats


# ---------------------------------------------------------------------------
# Aggregation (pure functions over normalised checkpoint records)
# ---------------------------------------------------------------------------

SCORE_DIMS = ["comprehensiveness", "diversity", "empowerment", "logical",
              "readability", "overall_score"]


def calc_mean_scores(score_records: List[Dict[str, Any]]) -> Dict[str, float]:
    if not score_records:
        return {}
    return {dim: sum(r[dim] for r in score_records) / len(score_records) for dim in SCORE_DIMS}


def _outcome_counts(winners: List[str], target: str, opponent: str) -> Dict[str, Any]:
    n = len(winners)
    wins = sum(1 for w in winners if w == target)
    losses = sum(1 for w in winners if w == opponent)
    ties = n - wins - losses
    pct = (lambda c: (c / n) * 100.0 if n else 0.0)
    return {"n": n, "wins": wins, "ties": ties, "losses": losses,
            "win_pct": pct(wins), "tie_pct": pct(ties), "loss_pct": pct(losses)}


def calc_win_rate(win_records: List[Dict[str, Any]], target_label: str,
                  opponent_label: str, verdict_key: str = "consensus",
                  judge: Optional[str] = None) -> Dict[str, Any]:
    """Win / tie / loss counts for ``target_label`` - one table row.

    Overall AND per-criterion outcomes are reported as complete
    win/tie/loss triples (summing to 100%). Table 4 and Figure 4 are both
    rendered from these exact fields, so a figure can no longer show win
    rates while the table shows ties for the same comparison.
    """
    verdicts = []
    for r in win_records:
        v = (r.get("per_judge") or {}).get(judge) if judge else r.get(verdict_key)
        if v:
            verdicts.append(v)
    n_missing = len(win_records) - len(verdicts)
    if not verdicts:
        return {"system_a": target_label, "system_b": opponent_label, "n": 0,
                "n_missing_verdicts": n_missing}
    overall = _outcome_counts([v.get("overall_winner", "Tie") for v in verdicts],
                              target_label, opponent_label)
    criteria = {
        c: _outcome_counts([(v.get("criteria_winners") or {}).get(c, "Tie") for v in verdicts],
                           target_label, opponent_label)
        for c in BasePaperEvaluator.CRITERIA_LIST
    }
    decisive = overall["wins"] + overall["losses"]
    return {
        "system_a": target_label,
        "system_b": opponent_label,
        "n": overall["n"],
        "n_missing_verdicts": n_missing,
        "overall": overall,
        "criteria": criteria,
        "decisive_win_rate_pct": (overall["wins"] / decisive) * 100.0 if decisive else None,
        # Back-compatible flat fields (W&B logging, API server).
        "target_system": target_label,
        "overall_win_rate_pct": overall["win_pct"],
        "tie_rate_pct": overall["tie_pct"],
        "loss_rate_pct": overall["loss_pct"],
        "criteria_win_rates_pct": {c: o["win_pct"] for c, o in criteria.items()},
    }


def _paired_scores(per_question: List[Dict[str, Any]], label_a: str, label_b: str,
                   dim: str = "overall_score", judge: Optional[str] = None) -> tuple:
    a, b = [], []
    for rec in per_question:
        sa = (rec.get("scores") or {}).get(label_a) or {}
        sb = (rec.get("scores") or {}).get(label_b) or {}
        va = (sa.get("per_judge") or {}).get(judge) if judge else sa.get("consensus")
        vb = (sb.get("per_judge") or {}).get(judge) if judge else sb.get("consensus")
        if va and vb:
            a.append(float(va[dim]))
            b.append(float(vb[dim]))
    return a, b


def significance_tests(per_question: List[Dict[str, Any]], pairs: List[tuple],
                       judge_names: List[str]) -> Dict[str, Any]:
    """Paired tests of arm A vs arm B on the consensus overall score.

    For every pair: bootstrap 95% CI of the mean paired difference,
    two-sided Wilcoxon signed-rank, two-sided paired t-test, and exact
    McNemar on correctness (overall >= CORRECTNESS_THRESHOLD). The same
    tests are repeated per judge so a result can be checked for judge
    dependence.
    """
    out: Dict[str, Any] = {"correctness_threshold": CORRECTNESS_THRESHOLD,
                           "comparisons": {}}
    for arm_a, arm_b in pairs:
        la, lb = ARM_LABELS[arm_a], ARM_LABELS[arm_b]

        def _tests(a, b):
            if not a:
                return {"n_paired": 0}
            return {
                "n_paired": len(a),
                "mean_a": sum(a) / len(a),
                "mean_b": sum(b) / len(b),
                "paired_difference_ci": paired_bootstrap_ci(a, b),
                "wilcoxon_signed_rank": wilcoxon_signed_rank(a, b),
                "paired_t_test": paired_t_test(a, b),
                "mcnemar_exact": mcnemar_exact(
                    [x >= CORRECTNESS_THRESHOLD for x in a],
                    [y >= CORRECTNESS_THRESHOLD for y in b]),
            }

        a, b = _paired_scores(per_question, la, lb)
        entry = {"system_a": la, "system_b": lb, "consensus": _tests(a, b),
                 "per_judge": {}}
        for j in judge_names:
            ja, jb = _paired_scores(per_question, la, lb, judge=j)
            entry["per_judge"][j] = _tests(ja, jb)
        out["comparisons"][comparison_key(arm_a, arm_b)] = entry
    return out


def score_confidence_intervals(per_question: List[Dict[str, Any]],
                               labels: List[str]) -> Dict[str, Any]:
    """Bootstrap 95% CI of every rubric dimension, per arm (consensus)."""
    out: Dict[str, Any] = {}
    for label in labels:
        rows = [((rec.get("scores") or {}).get(label) or {}).get("consensus")
                for rec in per_question]
        rows = [r for r in rows if r]
        out[label] = {dim: bootstrap_ci([r[dim] for r in rows]) for dim in SCORE_DIMS}
    return out


def inter_judge_agreement(per_question: List[Dict[str, Any]],
                          judge_names: List[str]) -> Dict[str, Any]:
    """Cohen's kappa (each judge pair) and Fleiss' kappa (all judges).

    - Pairwise winner selection: nominal label of the overall winner
      (system A / system B / Tie) per (question, comparison).
    - Rubric scoring: binary correctness (overall >= threshold) and the
      5-level rubric band (floor(overall / 20)) per (question, arm).
    Only items rated validly by every judge involved are used.
    """
    from itertools import combinations

    sel_items: List[Dict[str, str]] = []
    score_items: List[Dict[str, float]] = []
    for rec in per_question:
        for sel in (rec.get("selection") or {}).values():
            pj = sel.get("per_judge") or {}
            sel_items.append({j: (pj.get(j) or {}).get("overall_winner")
                              for j in judge_names if pj.get(j)})
        for sc in (rec.get("scores") or {}).values():
            pj = (sc or {}).get("per_judge") or {}
            score_items.append({j: float(pj[j]["overall_score"])
                                for j in judge_names if pj.get(j)})

    def _band(x: float) -> int:
        return min(4, int(x // 20))

    result: Dict[str, Any] = {"judges": list(judge_names), "pairwise": {},
                              "fleiss": {}}
    if len(judge_names) < 2:
        result["note"] = "Agreement requires >= 2 judges."
        return result
    for j1, j2 in combinations(judge_names, 2):
        sel = [(it[j1], it[j2]) for it in sel_items if j1 in it and j2 in it]
        sco = [(it[j1], it[j2]) for it in score_items if j1 in it and j2 in it]
        result["pairwise"][f"{j1} | {j2}"] = {
            "selection_overall_winner": cohen_kappa([x for x, _ in sel], [y for _, y in sel]),
            "scoring_correctness": cohen_kappa(
                [x >= CORRECTNESS_THRESHOLD for x, _ in sco],
                [y >= CORRECTNESS_THRESHOLD for _, y in sco]),
            "scoring_rubric_band": cohen_kappa([_band(x) for x, _ in sco],
                                               [_band(y) for _, y in sco]),
        }
    full_sel = [[it[j] for j in judge_names] for it in sel_items
                if all(j in it for j in judge_names)]
    full_sco = [[it[j] for j in judge_names] for it in score_items
                if all(j in it for j in judge_names)]
    result["fleiss"] = {
        "selection_overall_winner": fleiss_kappa(full_sel),
        "scoring_correctness": fleiss_kappa(
            [[x >= CORRECTNESS_THRESHOLD for x in row] for row in full_sco]),
        "scoring_rubric_band": fleiss_kappa([[_band(x) for x in row] for row in full_sco]),
    }
    return result


def latency_breakdown(per_question: List[Dict[str, Any]], labels: List[str]) -> Dict[str, Any]:
    """Retrieval-only vs generation latency, mean +- std per arm."""
    out: Dict[str, Any] = {}
    for label in labels:
        lat = [((rec.get("latency") or {}).get(label)) for rec in per_question]
        lat = [x for x in lat if x]
        if not lat:
            continue
        runs = [t for x in lat for t in (x.get("t_retrieval_runs") or [])]
        stage_keys = sorted({k for x in lat for k in (x.get("stages") or {})})
        gen = [x.get("generation") or {} for x in lat]

        def _vals(key):
            return [g[key] for g in gen if g.get(key) is not None]

        out[label] = {
            "n_questions": len(lat),
            "retrieval_seconds_per_question": mean_std([x.get("t_retrieval", 0.0) for x in lat]),
            "retrieval_seconds_all_runs": mean_std(runs),
            "retrieval_cold_seconds": mean_std([x["t_retrieval_cold"] for x in lat
                                                if x.get("t_retrieval_cold") is not None]),
            "retrieval_stage_seconds": {
                k: mean_std([(x.get("stages") or {}).get(k, 0.0) for x in lat
                             if k in (x.get("stages") or {})])
                for k in stage_keys
            },
            "generation_seconds": mean_std(_vals("t_generation")),
            "prefill_seconds": mean_std(_vals("prefill_s")),
            "decode_seconds": mean_std(_vals("decode_s")),
            "prompt_tokens": mean_std(_vals("prompt_tokens")),
            "generated_tokens": mean_std(_vals("completion_tokens")),
            "tokens_per_second": mean_std(_vals("tokens_per_second")),
            "abstention_rate": (sum(1 for x in lat if x.get("abstained")) / len(lat)),
        }
    return out


def aggregate_results(records: Dict[str, Any], arm_ids: List[str],
                      judge_names: List[str]) -> Dict[str, Any]:
    """Every aggregate number of a run, computed from checkpoint records."""
    labels = [ARM_LABELS[a] for a in arm_ids]
    per_question = records["per_question"]
    pairs = selection_pairs(arm_ids)

    final_scores = {label: calc_mean_scores(records["scoring"].get(label, []))
                    for label in labels}
    final_scores = {k: v for k, v in final_scores.items() if v}

    win_rates: Dict[str, Any] = {}
    win_rates_per_judge: Dict[str, Any] = {}
    for arm_a, arm_b in pairs:
        key = comparison_key(arm_a, arm_b)
        recs = records["selection"].get(key, [])
        if not recs:
            continue
        la, lb = ARM_LABELS[arm_a], ARM_LABELS[arm_b]
        win_rates[key] = calc_win_rate(recs, la, lb)
        win_rates_per_judge[key] = {j: calc_win_rate(recs, la, lb, judge=j)
                                    for j in judge_names}

    win_rates_by_source = {
        source: {
            key: calc_win_rate(recs, ARM_LABELS[key.split("_vs_")[0]],
                               ARM_LABELS[key.split("_vs_")[1]])
            for key, recs in comps.items() if recs and key.split("_vs_")[0] in ARM_LABELS
            and key.split("_vs_")[1] in ARM_LABELS
        }
        for source, comps in records["selection_by_source"].items()
    }

    lat = latency_breakdown(per_question, labels)
    retrieval_latency = {label: v["retrieval_seconds_per_question"]["mean"]
                         for label, v in lat.items()}

    retrieval_quality: Dict[str, Any] = {"k": RETRIEVAL_K, "systems": {}}
    for label in labels:
        rows = records["retrieval"].get(label, [])
        if not rows:
            continue
        retrieval_quality["systems"][label] = {
            metric: bootstrap_ci([r[metric] for r in rows]) for metric in ("ndcg_at_k", "recall_at_k")
        }
    retrieval_quality["n_samples"] = len(records["retrieval"].get(ARM_LABELS["hcad"], []))

    citation_summary: Dict[str, Any] = {}
    for label in labels:
        cit = records["citations"].get(label, [])
        if not cit and label != BASELINE_LABEL:
            continue
        precisions = [r["precision"] for r in cit if r.get("precision") is not None]
        citation_summary[label] = {
            "n_answers_with_citations": len(precisions),
            "n_answers": len(cit) if cit else len(records["scoring"].get(label, [])),
            "total_citations": sum(r.get("n_citations", 0) for r in cit),
            "mean_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
        }

    judge_failures = {
        j: sum(1 for rec in per_question
               for blob in list((rec.get("scores") or {}).values())
               + list((rec.get("selection") or {}).values())
               if j in ((blob or {}).get("per_judge") or {})
               and not (blob or {}).get("per_judge", {}).get(j))
        for j in judge_names
    }

    return {
        "systems": {ARM_LABELS[a]: {"arm_id": a, "short_label": ARM_SHORT_LABELS[a]}
                    for a in arm_ids},
        "scoring_based_assessment": final_scores,
        "scoring_confidence_intervals": score_confidence_intervals(per_question, labels),
        "scoring_by_difficulty_tier": {
            str(t): {s_: (sum(r["overall_score"] for r in recs) / len(recs) if recs else 0.0)
                     for s_, recs in sys_map.items()}
            for t, sys_map in records["scoring_by_tier"].items()
        },
        "scoring_based_assessment_by_source": {
            source: {s_: calc_mean_scores(recs) for s_, recs in m.items()}
            for source, m in records["scoring_by_source"].items()
        },
        "selection_based_assessment": win_rates,
        "selection_based_assessment_per_judge": win_rates_per_judge,
        "selection_based_assessment_by_source": win_rates_by_source,
        "statistical_tests": significance_tests(per_question, pairs, judge_names),
        "inter_judge_agreement": inter_judge_agreement(per_question, judge_names),
        "retrieval_latency_seconds": retrieval_latency,
        "latency_breakdown": lat,
        "retrieval_quality": retrieval_quality,
        "citation_precision": citation_summary,
        "judge_failures": judge_failures,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@tracked_run(
    name="base-paper-benchmark",
    job_type="benchmark",
    config=lambda num_samples=10, output_path=None, question_pool="test", **kw: {
        "samples": num_samples, "question_pool": question_pool},
    log_results=log_base_paper_results,
)
def run_benchmark(num_samples: int = 10, output_path: Optional[Path] = None,
                  question_pool: str = "test", arms: Optional[Sequence[str]] = None,
                  include_ablations: Optional[bool] = None,
                  judge: Optional[MultiJudgeEvaluator] = None,
                  latency_runs: int = LATENCY_TIMED_RUNS,
                  built_arms: Optional[Dict[str, BenchmarkArm]] = None,
                  samples: Optional[List[Dict[str, Any]]] = None):
    """Multi-judge, N-arm benchmark (closed-book, BM25, dense, hybrid, HCAD
    + optional ablations) with significance tests and latency breakdown.

    ``judge``, ``built_arms`` and ``samples`` are injection points for
    tests; production runs build them from config.
    """
    arm_ids = resolve_arms(arms, include_ablations)
    judge = judge or MultiJudgeEvaluator.from_config()
    logger.info("Arms: %s", [ARM_LABELS[a] for a in arm_ids])
    # Fail before hours of retrieval/generation when a judge is not pulled,
    # does not fit in memory, or cannot produce JSON.
    if os.getenv("JUDGE_SKIP_PREFLIGHT", "0") != "1" and hasattr(judge, "check_ready"):
        judge.check_ready()
    judge_names = judge.judge_names
    logger.info("Judges: %s", judge_names)
    from src.config import JUDGE_MAX_FAILURE_RATE
    hcad_config_id = _hcad_config_id()
    logger.info("HCAD evidence config: %s", hcad_config_id)

    systems = built_arms or build_arms(arm_ids)
    hcad_retriever = getattr(systems.get("hcad"), "_retriever", None)
    gate_mode = getattr(hcad_retriever, "gate_mode", "n/a")
    expected_gate_mode = os.environ.get("HCAD_EXPECTED_GATE_MODE", "").strip() or None
    logger.info("JEV gate mode: %s (expected: %s)", gate_mode, expected_gate_mode or "any")
    if expected_gate_mode and gate_mode != expected_gate_mode:
        raise RuntimeError(
            f"Gate mode mismatch: expected HCAD_EXPECTED_GATE_MODE="
            f"{expected_gate_mode!r}, got {gate_mode!r}. Refusing to run: "
            f"benchmark results are only comparable within one gate mode."
        )

    if samples is None:
        # Paper protocol (Sec. 3.5): LLM-generated questions with source
        # chunks as references. Generation failure is FATAL.
        samples = generate_paper_protocol_questions(num_samples, pool=question_pool)
    if not samples:
        raise RuntimeError(
            "No benchmark questions were generated or loaded. Refusing to "
            "write empty results. Check that the LLM provider is reachable."
        )
    logger.info("Loaded %d clinical test cases.", len(samples))

    checkpoint_path = RESULTS_DIR / os.getenv(
        "BENCHMARK_CHECKPOINT", "base_paper_checkpoint.jsonl")
    question_cache_path = _paper_question_cache_path(num_samples, question_pool)
    question_set_id = _question_set_id(samples)
    checkpoint_upload_every = int(os.getenv("WANDB_CHECKPOINT_EVERY", "5"))
    evaluated_this_session = 0
    completed_questions = _completed_questions_from_checkpoint(
        checkpoint_path, question_pool, question_set_id,
        required_arms=arm_ids, required_judges=judge_names,
    )
    logger.info("Resuming: %d samples already checkpointed.", len(completed_questions))
    pairs = selection_pairs(arm_ids)

    for idx, sample in enumerate(samples):
        q = sample["question"]
        ref = sample["reference"]
        if q in completed_questions:
            logger.info("Skipping checkpointed sample %d/%d.", idx + 1, len(samples))
            continue
        logger.info("Evaluating Sample %d/%d: %.70s...", idx + 1, len(samples), q)

        answers, evidence, contexts_by_arm, latency = {}, {}, {}, {}
        for arm_id in arm_ids:
            arm = systems[arm_id]
            contexts, lat = timed_retrieval(arm, q, latency_runs)
            answer, gen_stats = generate_answer(arm, q, contexts)
            lat["generation"] = gen_stats
            answers[arm_id] = answer
            contexts_by_arm[arm_id] = contexts
            evidence[arm_id] = (_format_retrieved_evidence(contexts, "Doc")
                                if arm.retrieve is not None else
                                "No retrieved evidence was supplied for this answer.")
            latency[arm.label] = lat
            logger.info("  %-40s t_ret=%.3fs t_gen=%.2fs ctx=%d", arm.label,
                        lat["t_retrieval"], gen_stats["t_generation"], lat["n_contexts"])

        # Judge inputs use neutral [Doc i] citation tags for every arm (blind
        # judging); raw answers are kept for citation precision below.
        judge_answers = {a: _blind_citation_tags(answers[a]) for a in arm_ids}
        scores = {
            ARM_LABELS[a]: judge.evaluate_scoring(q, judge_answers[a], ref,
                                                  answer_evidence=evidence[a])
            for a in arm_ids
        }
        selection = {
            comparison_key(a, b): judge.evaluate_selection(
                q, judge_answers[a], judge_answers[b], ref,
                label_a=ARM_LABELS[a], label_b=ARM_LABELS[b],
                evidence_a=evidence[a], evidence_b=evidence[b])
            for a, b in pairs
        }

        sample_retrieval: Dict[str, Dict[str, float]] = {}
        sample_citations: Dict[str, Dict[str, Any]] = {}
        sample_evidence: Dict[str, Dict[str, Any]] = {}
        for arm_id in arm_ids:
            if systems[arm_id].retrieve is None:
                continue
            label, contexts = ARM_LABELS[arm_id], contexts_by_arm[arm_id]
            relevance = _ranked_relevance(ref, contexts)
            sample_evidence[label] = _evidence_stats(contexts, relevance)
            if sample.get("source") == "paper_protocol":
                sample_retrieval[label] = {
                    "ndcg_at_k": round(ndcg_at_k(relevance, k=RETRIEVAL_K), 4),
                    "recall_at_k": round(recall_at_k(relevance, k=RETRIEVAL_K), 4),
                }
            sample_citations[label] = citation_precision(
                answers[arm_id], [_context_document(c) for c in contexts])

        hcad_contexts = contexts_by_arm.get("hcad") or []
        with open(checkpoint_path, "a", encoding="utf-8") as ckpt:
            ckpt.write(json.dumps({
                "schema": 2,
                "question": q,
                "source": sample.get("source", "curated"),
                "difficulty_tier": sample.get("difficulty_tier", 1),
                "source_kind": sample.get("source_kind", ""),
                "source_chunk_id": sample.get("source_chunk_id", ""),
                "question_pool": sample.get("pool", question_pool),
                "question_set_id": question_set_id,
                "arms": {a: ARM_LABELS[a] for a in arm_ids},
                "judges": judge_names,
                "hcad_config_id": hcad_config_id,
                "grounded_mode": "grounded" if hcad_contexts else "parametric",
                "n_hcad_contexts": len(hcad_contexts),
                "answers": {ARM_LABELS[a]: answers[a] for a in arm_ids},
                "scores": scores,
                "selection": selection,
                "latency": latency,
                "retrieval": sample_retrieval,
                "evidence": sample_evidence,
                "citations": sample_citations,
            }) + "\n")

        evaluated_this_session += 1
        if checkpoint_upload_every > 0 and evaluated_this_session % checkpoint_upload_every == 0:
            upload_checkpoint_artifact([checkpoint_path, question_cache_path])

        # Abort when a judge is broken (OOM, timeouts, unparseable JSON): its
        # verdicts are missing, so the run would measure nothing for it. Two
        # questions are enough to tell that from an occasional bad reply.
        calls = getattr(judge, "calls", None) or {}
        for name in judge_names:
            if not calls.get(name):
                continue
            rate = judge.failure_rate(name)
            logger.info("  Judge %s failures this session: %d/%d (%.0f%%)", name,
                        judge.failures.get(name, 0), calls[name], 100 * rate)
            if evaluated_this_session >= 2 and rate > JUDGE_MAX_FAILURE_RATE:
                raise RuntimeError(
                    f"Judge {name} failed {judge.failures.get(name, 0)}/{calls[name]} "
                    f"evaluations (> {JUDGE_MAX_FAILURE_RATE:.0%}). Aborting: its "
                    "verdicts would be missing, not judgments. Check the WARNING "
                    "lines above (unparseable JSON -> pick another judge index; "
                    "timeouts / empty replies -> GPU memory, see `ollama ps`). "
                    "Checkpointed questions are kept and will resume."
                )

    if checkpoint_upload_every > 0 and evaluated_this_session > 0 and evaluated_this_session % checkpoint_upload_every != 0:
        upload_checkpoint_artifact([checkpoint_path, question_cache_path])

    records = _records_from_checkpoint(
        checkpoint_path, {s["question"] for s in samples}, question_pool=question_pool,
        question_set_id=question_set_id, required_judges=judge_names,
    )
    aggregates = aggregate_results(records, arm_ids, judge_names)

    # Evidence diagnostics: what each retrieval arm fed the generator,
    # averaged over the questions whose records carry it.
    evidence_diagnostics: Dict[str, Any] = {}
    for arm_id in arm_ids:
        label = ARM_LABELS[arm_id]
        rows = [rec["evidence"][label] for rec in records["per_question"]
                if isinstance(rec.get("evidence"), dict) and label in rec["evidence"]]
        if not rows:
            continue
        evidence_diagnostics[label] = {
            "n_questions": len(rows),
            **{f"mean_{key}": round(sum(r[key] for r in rows) / len(rows), 2)
               for key in ("n_items", "n_prose", "n_kg", "n_kg_facts", "words")},
            "gold_in_evidence_rate": round(
                sum(1 for r in rows if r["gold_rank"] is not None) / len(rows), 3),
            "gold_at_rank1_rate": round(
                sum(1 for r in rows if r["gold_rank"] == 1) / len(rows), 3),
        }
    from src.config import dense_anchor_k, dense_anchor_min, dense_weight, group_kg_facts_enabled
    from src.config import ENABLE_FEW_SHOT

    summary_results = {
        "run_manifest": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "num_samples": len(samples),
            "arms": {a: ARM_LABELS[a] for a in arm_ids},
            "ablation_env": {a: ABLATION_ENV[a] for a in arm_ids if a in ABLATION_ENV},
            "generator": {"model": LLM_MODEL, "label": BASELINE_LABEL},
            "judges": judge.manifest(),
            "judge_selection": {
                "model_index": os.environ.get("JUDGE_MODEL_INDEX") or None,
                "quant": os.environ.get("JUDGE_QUANT") or None,
                "ensemble_env": os.environ.get("JUDGE_ENSEMBLE") or None,
            },
            "hcad_evidence_config": {
                "config_id": hcad_config_id,
                "dense_weight": dense_weight(),
                "dense_anchor_k": dense_anchor_k(),
                "dense_anchor_min": dense_anchor_min(),
                "group_kg_facts": group_kg_facts_enabled(),
            },
            "judge_json_mode": True,
            "latency_timed_runs": latency_runs,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "prompt_mode": "few_shot" if ENABLE_FEW_SHOT else "zero_shot",
            "random_seed": RANDOM_SEED,
            "question_pool": question_pool,
            "question_set_id": question_set_id,
            "gate_mode": gate_mode,
            "tensor_mode": os.environ.get("HCAD_TENSOR_MODE", "static"),
        },
        **aggregates,
        "evidence_diagnostics": evidence_diagnostics,
        # counted from the de-duplicated newest record per question
        "grounded_mode": {
            mode: sum(1 for rec in records["per_question"] if rec.get("grounded_mode") == mode)
            for mode in ("grounded", "parametric")
        },
        "num_samples": len(samples),
        "sample_source_counts": {
            source: sum(1 for s in samples if s.get("source", "curated") == source)
            for source in sorted({s.get("source", "curated") for s in samples})
        },
        "judge": {"ensemble": judge_names},
    }

    out_file = output_path or (RESULTS_DIR / os.getenv(
        "BENCHMARK_RESULTS_FILE", "base_paper_benchmark_results.json"))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    # base_paper_benchmark_results[_suffix].json -> latency_breakdown[_suffix].json,
    # so variant runs (e.g. the learned-tensor ablation) never overwrite the main one.
    latency_file = out_file.with_name(
        out_file.stem.replace("base_paper_benchmark_results", "latency_breakdown", 1) + ".json"
        if out_file.stem.startswith("base_paper_benchmark_results")
        else out_file.stem + "_latency_breakdown.json")
    with open(latency_file, "w", encoding="utf-8") as f:
        json.dump({"run_manifest": summary_results["run_manifest"],
                   "latency_breakdown": summary_results["latency_breakdown"]}, f, indent=2)

    logger.info("Benchmark results saved to %s (latency: %s)", out_file, latency_file)
    return summary_results


def _cli(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="HCAD-RAG multi-judge benchmark")
    parser.add_argument("samples", nargs="?", type=int, default=5)
    parser.add_argument("--pool", default="test", choices=["test", "dev"])
    parser.add_argument("--arms", default=None,
                        help=f"comma list from {sorted(ARM_LABELS)}")
    parser.add_argument("--ablations", action="store_true",
                        help="add w/o Decoupling, w/o Soft GS, w/o Decision Gate arms")
    parser.add_argument("--latency-runs", type=int, default=LATENCY_TIMED_RUNS)
    args = parser.parse_args(argv)
    arms = [a.strip() for a in args.arms.split(",")] if args.arms else None
    run_benchmark(num_samples=args.samples, question_pool=args.pool, arms=arms,
                  include_ablations=args.ablations or None, latency_runs=args.latency_runs)


if __name__ == "__main__":
    _cli()
