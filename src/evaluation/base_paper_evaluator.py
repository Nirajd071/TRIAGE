"""
Base Paper Evaluation Suite for HCAD-RAG.

Directly implements the evaluation protocols from the foundational base paper:
"Combating LLM Hallucinations using Hypergraph-Driven Retrieval-Augmented Generation"
(Hyper-RAG, arXiv:2504.08758v1, Sections 4.1, 4.2, 1.5, and 5.2).

Protocols:
1. Scoring-Based Assessment (P_eval_score, Section 4.1 & 5.2):
   - 5 dimensions on a 0-100 scale using a 5-level hierarchical rubric:
     * Comprehensiveness (0-100): Covers all relevant aspects without omissions.
     * Diversity (0-100): Richness of domain context beyond direct query.
     * Empowerment (0-100): Factual credibility, hallucination suppression, and clinical confidence.
     * Logical (0-100): Coherence and absence of self-contradiction.
     * Readability (0-100): Structural clarity, accessibility, and medical formatting.
   - Overall Score = arithmetic mean of the 5 dimensions.

2. Selection-Based Assessment (P_eval_select, Section 4.2 & 5.2):
   - Blind pairwise evaluation (Model A vs. Model B) across 8 qualitative criteria:
     * Comprehensiveness, Empowerment, Accuracy, Relevance, Coherence,
       Clarity, Logical, Flexibility.
   - Head-to-head cumulative win rate (%).

3. Retrieval Latency Efficiency (Section 1.5):
   - Wall-clock retrieval time (seconds) isolating local DB search and tensor processing.
"""

import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """Quantitative evaluation across the base paper's 5 dimensions."""
    comprehensiveness: float
    diversity: float
    empowerment: float
    logical: float
    readability: float
    overall_score: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SelectionResult:
    """Pairwise selection evaluation across the base paper's 8 criteria."""
    comprehensiveness_winner: str
    empowerment_winner: str
    accuracy_winner: str
    relevance_winner: str
    coherence_winner: str
    clarity_winner: str
    logical_winner: str
    flexibility_winner: str
    overall_winner: str
    criterion_rationales: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ---------------------------------------------------------------------------
# Base Paper Prompts 
# ---------------------------------------------------------------------------

SCORING_PROMPT_SYSTEM = """You are an expert medical evaluator tasked with evaluating clinical answers by using the relevant documents based on five criteria: Comprehensiveness, Diversity, Empowerment, Logical, and Readability."""

SCORING_PROMPT_USER = """Evaluate the answer to the clinical question using the relevant reference document based on five criteria: Comprehensiveness, Diversity, Empowerment, Logical, and Readability.

- Comprehensiveness:
Measure whether the answer comprehensively covers all key aspects of the question and whether there are omissions.
Level 1 (0-20): The answer is extremely one-sided, leaving out key parts or important aspects of the question.
Level 2 (20-40): The answer has some content, but it misses many important aspects of the question and is not comprehensive enough.
Level 3 (40-60): The answer is more comprehensive, covering the main aspects of the question, but there are still some omissions.
Level 4 (60-80): The answer is comprehensive, covering most aspects of the question, with few omissions.
Level 5 (80-100): The answer is extremely comprehensive, covering all aspects of the question with no omissions, enabling the reader to gain a complete understanding.

- Diversity:
Evaluates the richness of the content, including additional related biomedical knowledge beyond the direct answer (Level 1: 0-20 to Level 5: 80-100).

- Empowerment:
Measures clinical credibility, factual accuracy, and appropriate uncertainty (Level 1: 0-20 to Level 5: 80-100). When retrieved evidence is supplied, verify cited claims against it. Give credit for citations only when the cited evidence directly supports the claim; citation tags alone earn no credit. Do not penalize uncited claims when they are supported by the reference document or established clinical knowledge. Penalize unsupported or contradicted high-impact claims regardless of citation style.

- Logical:
Determines the coherence and clarity of the response, ensuring that the arguments are logically structured, well-articulated, and free of contradictions (Level 1: 0-20 to Level 5: 80-100).

- Readability:
Examines the organization and formatting of the response, ensuring it is clear, professional, and easy to understand (Level 1: 0-20 to Level 5: 80-100).

--- Inputs ---
Question: {question}
Relevant Reference Document: {reference_document}
Retrieved Evidence Available to This System: {answer_evidence}
System Answer: {answer}

--- Output Instructions ---
Provide numeric scores strictly on a scale from 0 to 100 for each indicator based on the rubric.
Respond strictly in JSON format as follows:
{{
  "comprehensiveness": <score 0-100>,
  "diversity": <score 0-100>,
  "empowerment": <score 0-100>,
  "logical": <score 0-100>,
  "readability": <score 0-100>,
  "rationale": "<summary explanation>"
}}
"""

SELECTION_PROMPT_SYSTEM = """You are an expert clinical evaluator tasked with evaluating two answers to the same question based on eight criteria: Comprehensiveness, Empowerment, Accuracy, Relevance, Coherence, Clarity, Logical, and Flexibility."""

SELECTION_PROMPT_USER = """Evaluate two answers to the same clinical question by using the relevant documents based on eight criteria:
1. Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
2. Empowerment: How well does the answer help the reader make informed clinical judgments with accurate, appropriately supported claims? When evidence is supplied, verify citations against it. Do not reward citation tags by themselves or penalize uncited claims supported by the reference or established clinical knowledge.
3. Accuracy: How well does the answer align with factual truth and avoid hallucination based on the reference and any answer-specific retrieved evidence supplied below?
4. Relevance: How precisely does the answer address the core aspects of the question without including unnecessary information?
5. Coherence: How well does the system integrate and synthesize information from multiple sources into a logically flowing response?
6. Clarity: How well does the system provide complete information while avoiding unnecessary verbosity and redundancy?
7. Logical: How well does the system maintain consistent logical arguments without contradicting itself across the response?
8. Flexibility: How well does the system handle various question formats, tones, and levels of complexity?

For each criterion, choose the better answer (either "Answer 1" or "Answer 2") and explain why. Then, select an overall winner between Answer 1 and Answer 2.

--- Inputs ---
Question: {question}
Relevant Reference Document: {reference_document}
Retrieved Evidence Available to Answer 1: {evidence_1}
Answer 1: {answer_1}
Retrieved Evidence Available to Answer 2: {evidence_2}
Answer 2: {answer_2}

--- Output Instructions ---
Respond strictly in JSON format as follows:
{{
  "comprehensiveness": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "empowerment": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "accuracy": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "relevance": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "coherence": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "clarity": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "logical": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "flexibility": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "overall_winner": "Answer 1" | "Answer 2" | "Tie",
  "summary": "<overall comparative verdict>"
}}
"""

# ---------------------------------------------------------------------------
# Base Paper Evaluator Engine
# ---------------------------------------------------------------------------

class BasePaperEvaluator:
    """
    Evaluator strictly compliant with Hyper-RAG .
    Provides Scoring-Based Assessment (0-100), Selection-Based Assessment (Win-Rate),
    and Retrieval Latency profiling.
    """

    CRITERIA_LIST = [
        "comprehensiveness",
        "empowerment",
        "accuracy",
        "relevance",
        "coherence",
        "clarity",
        "logical",
        "flexibility",
    ]

    def __init__(self, judge_llm: Optional[UnifiedLLMClient] = None,
                 name: Optional[str] = None):
        """Initialize evaluator with a deterministic judge LLM (defaults to Qwen2.5-72B).

        Every judge call is issued with ``json_mode=True`` (Ollama
        ``format=json`` / OpenAI-style ``response_format=json_object``).
        """
        if judge_llm is None:
            from src.config import JUDGE_HF_MODEL
            self.judge = UnifiedLLMClient(model=JUDGE_HF_MODEL, temperature=0.0, max_tokens=768)
        else:
            self.judge = judge_llm
        self.name = name or (
            f"{getattr(self.judge, 'provider', 'judge')}:{getattr(self.judge, 'model', 'unknown')}"
        )

    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        """Robustly extract JSON payload from LLM markdown fences or raw strings."""
        clean_text = (raw_text or "").strip()
        # Strip chain-of-thought tags (e.g. DeepSeek-R1 <think>...</think>)
        clean_text = re.sub(r"<think>[\s\S]*?</think>", "", clean_text).strip()

        # Try multiple JSON extraction strategies

        # 1. Try to find JSON in markdown code fences first (most reliable)
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 2. Try to find the first complete JSON object
        idx = clean_text.find("{")
        if idx != -1:
            try:
                decoder = json.JSONDecoder()
                obj, end_idx = decoder.raw_decode(clean_text[idx:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

            # 3. Try to find balanced braces
            brace_count = 0
            for i, char in enumerate(clean_text[idx:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = clean_text[idx:idx+i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            break

        # 4. Try to find JSON without markdown fences - look for standalone JSON
        standalone_match = re.search(r'\{[^{}]*\}', clean_text, re.DOTALL)
        if standalone_match:
            try:
                return json.loads(standalone_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse LLM evaluation JSON directly. Falling back to heuristic recovery. Snippet: {clean_text[:200]}")
        return self._heuristic_json_recovery(clean_text)

    def _heuristic_json_recovery(self, text: str) -> Dict[str, Any]:
        """Fallback parser when LLM outputs markdown text or malformed JSON."""
        res: Dict[str, Any] = {}

        try:
            text_lower = (text or "").lower()

            # 1. Scoring dimensions - handle multiple pattern formats
            scoring_keys = ["comprehensiveness", "diversity", "empowerment", "logical", "readability", "overall_score"]
            for key in scoring_keys:
                patterns = [
                    # Nested object: "comprehensiveness": {"score": 85}
                    rf'(?:\"{key}\"|\'{key}\'|\b{key}\b)\s*:\s*\{{[\s\S]*?\"score\"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                    # JSON-style with quotes: "comprehensiveness": 85
                    rf'(?:\"{key}\"|\'{key}\')\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                    # Unquoted key: comprehensiveness: 85
                    rf'\b{key}\b\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                    # Markdown bold: **Comprehensiveness:** 85
                    rf'\*\*{key}\*\*\s*:?\s*([0-9]+(?:\.[0-9]+)?)',
                    # Score for key: Score for Comprehensiveness: 85
                    rf'(?:score|rating|level)\s*(?:for|of)?\s*{key}\s*:?\s*([0-9]+(?:\.[0-9]+)?)',
                ]

                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        try:
                            score = float(match.group(1))
                            res[key] = max(0.0, min(100.0, score))
                            break
                        except ValueError:
                            continue

            # 2. Pairwise Selection criteria - comprehensive pattern matching
            for crit in self.CRITERIA_LIST:
                # Pattern 1: JSON-style with nested object (supports multiline)
                crit_match = re.search(
                    rf'(?:\"{crit}\"|\'{crit}\'|\b{crit}\b)\s*:\s*\{{[\s\S]*?(?:\"winner\"|winner)\s*:\s*\"?(Answer 1|Answer 2|Tie)\"?',
                    text, re.IGNORECASE
                )
                if not crit_match:
                    # Pattern 2: Markdown-style with Winner mention
                    crit_match = re.search(
                        rf'(?:\*\*{crit}\*\*|\b{crit}\b)[\s\S]{0,200}?(?:Winner|winner|choice|answer)\s*:?\s*(Answer 1|Answer 2|Tie)',
                        text, re.IGNORECASE
                    )
                if not crit_match:
                    # Pattern 3: Simple mention of Answer 1 or Answer 2 after criterion
                    answer_1_match = re.search(rf'(?:\*\*{crit}\*\*|\b{crit}\b)[\s\S]{0,200}?\banswer\s*1\b', text, re.IGNORECASE)
                    answer_2_match = re.search(rf'(?:\*\*{crit}\*\*|\b{crit}\b)[\s\S]{0,200}?\banswer\s*2\b', text, re.IGNORECASE)
                    tie_match = re.search(rf'(?:\*\*{crit}\*\*|\b{crit}\b)[\s\S]{0,200}?\btie\b', text, re.IGNORECASE)

                    if answer_1_match:
                        res[crit] = {"winner": "Answer 1"}
                    elif answer_2_match:
                        res[crit] = {"winner": "Answer 2"}
                    elif tie_match:
                        res[crit] = {"winner": "Tie"}

            # 3. Overall winner - comprehensive patterns
            overall_patterns = [
                r'(?:Overall Winner|overall_winner|Final Winner|Best Answer|Chooses?)\s*:?\s*(Answer 1|Answer 2|Tie)',
                r'(?:Winner|Victor|Preferred|Selects?)\s*:?\s*(Answer 1|Answer 2|Tie)',
                r'Final\s+(?:decision|result|outcome|answer)\s*:?\s*(Answer 1|Answer 2|Tie)',
            ]

            overall_match = None
            for pattern in overall_patterns:
                overall_match = re.search(pattern, text, re.IGNORECASE)
                if overall_match:
                    break

            if overall_match:
                res["overall_winner"] = overall_match.group(1)
            elif any(crit in res for crit in self.CRITERIA_LIST):
                # Compute majority winner across criteria if overall_winner was not explicitly stated
                winners = [res[c].get("winner", "Tie") for c in self.CRITERIA_LIST if c in res and isinstance(res[c], dict)]
                count_1 = sum(1 for w in winners if "Answer 1" in w or "1" in w)
                count_2 = sum(1 for w in winners if "Answer 2" in w or "2" in w)
                if count_1 > count_2:
                    res["overall_winner"] = "Answer 1"
                elif count_2 > count_1:
                    res["overall_winner"] = "Answer 2"
                else:
                    res["overall_winner"] = "Tie"

            if "rationale" not in res and "summary" not in res:
                res["rationale"] = "Extracted via regex fallback."

        except Exception as e:
            logger.warning(f"Heuristic JSON recovery failed: {e}")
            # Return minimal valid structure
            res = {"rationale": f"Recovery failed: {e}"}

        return res

    # -----------------------------------------------------------------------
    # 1. Scoring-Based Assessment (0-100 Scale, 5 Dimensions)
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_dim_score(val: Any, default: float = 75.0) -> float:
        """Convert raw judge output to a standard 0-100 scale.

        Our scoring prompt explicitly asks for 'numeric scores strictly on a
        scale from 0 to 100', so we treat all numeric outputs as 0-100 values.
        """
        if isinstance(val, dict):
            val = val.get("score", val.get("level", default))
        try:
            score = float(val)
        except (ValueError, TypeError):
            return default

        return max(0.0, min(100.0, score))

    def evaluate_scoring(
        self,
        question: str,
        answer: str,
        reference_document: str,
        answer_evidence: str = "No retrieved evidence was supplied for this answer.",
        retries: int = 1,
        fallback_on_error: bool = False,
    ) -> ScoringResult:
        """
        Execute Scoring-Based Assessment (P_eval_scoring(q, R, T_0)).
        Returns scores across the 5 dimensions on a 0-100 scale.
        """
        messages = [
            {"role": "system", "content": SCORING_PROMPT_SYSTEM},
            {"role": "user", "content": SCORING_PROMPT_USER.format(
                question=question,
                reference_document=reference_document,
                answer=answer,
                answer_evidence=answer_evidence,
            )},
        ]

        score_fields = ("comprehensiveness", "diversity", "empowerment", "logical", "readability")
        last_exc: Optional[Exception] = None

        for attempt in range(max(1, retries + 1)):
            try:
                raw_response = self.judge.chat(messages, temperature=0.0, json_mode=True)
                data = self._extract_json(raw_response)
                logger.debug("Judge raw evaluation response: %s", raw_response)
                logger.debug("Judge parsed data: %s", data)

                missing = [field for field in score_fields if field not in data]
                if missing:
                    raise ValueError(f"Judge response is missing score fields: {', '.join(missing)}")

                normalized = {}
                for field in score_fields:
                    value = data[field]
                    if isinstance(value, dict):
                        value = value.get("score", value.get("level"))
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Judge returned a non-numeric {field} score") from exc
                    if not math.isfinite(numeric_value):
                        raise ValueError(f"Judge returned a non-finite {field} score")
                    normalized[field] = max(0.0, min(100.0, numeric_value))

                comp = normalized["comprehensiveness"]
                div = normalized["diversity"]
                emp = normalized["empowerment"]
                log = normalized["logical"]
                read = normalized["readability"]

                overall = (comp + div + emp + log + read) / 5.0
                rationale = str(data.get("rationale") or data.get("justification") or "")

                return ScoringResult(
                    comprehensiveness=comp,
                    diversity=div,
                    empowerment=emp,
                    logical=log,
                    readability=read,
                    overall_score=overall,
                    rationale=rationale,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Judge scoring attempt %d failed: %s", attempt + 1, exc
                )

        if fallback_on_error:
            logger.warning(
                "All scoring attempts failed (%s); returning neutral 50.0 fallback scores.",
                last_exc,
            )
            return ScoringResult(
                comprehensiveness=50.0,
                diversity=50.0,
                empowerment=50.0,
                logical=50.0,
                readability=50.0,
                overall_score=50.0,
                rationale=f"Fallback due to judge failure: {last_exc}",
            )
        assert last_exc is not None
        raise last_exc

    # -----------------------------------------------------------------------
    # 2. Selection-Based Assessment (Pairwise Win-Rate, 8 Criteria)
    # -----------------------------------------------------------------------

    def evaluate_selection(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        reference_document: str,
        label_a: str = "System A",
        label_b: str = "System B",
        evidence_a: str = "No retrieved evidence was supplied for this answer.",
        evidence_b: str = "No retrieved evidence was supplied for this answer.",
        retries: int = 1,
        fallback_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute Selection-Based Assessment (P_eval_select(q, R_a, R_b)).
        Compares Answer A vs. Answer B blindly across 8 criteria.
        """
        messages = [
            {"role": "system", "content": SELECTION_PROMPT_SYSTEM},
            {"role": "user", "content": SELECTION_PROMPT_USER.format(
                question=question,
                reference_document=reference_document,
                answer_1=answer_a,
                answer_2=answer_b,
                evidence_1=evidence_a,
                evidence_2=evidence_b,
            )},
        ]

        last_exc: Optional[Exception] = None

        for attempt in range(max(1, retries + 1)):
            try:
                raw_response = self.judge.chat(messages, temperature=0.0, json_mode=True)
                data = self._extract_json(raw_response)

                missing_criteria = [crit for crit in self.CRITERIA_LIST if crit not in data]
                if missing_criteria:
                    raise ValueError(
                        "Judge response is missing pairwise criteria: "
                        + ", ".join(missing_criteria)
                    )

                criteria_winners = {}
                for crit in self.CRITERIA_LIST:
                    crit_obj = data.get(crit, {})
                    if isinstance(crit_obj, dict):
                        w = crit_obj.get("winner")
                    elif isinstance(crit_obj, str):
                        w = crit_obj
                    else:
                        raise ValueError(f"Judge returned malformed result for {crit}")

                    # Map Answer 1 / Answer 2 back to caller labels
                    w_str = str(w or "").strip()
                    normalized_winner = w_str.casefold().replace(" ", "")
                    if normalized_winner in {"answer1", "1"}:
                        mapped_w = label_a
                    elif normalized_winner in {"answer2", "2"}:
                        mapped_w = label_b
                    elif normalized_winner == "tie":
                        mapped_w = "Tie"
                    else:
                        raise ValueError(f"Judge returned invalid winner for {crit}: {w_str!r}")
                    criteria_winners[crit] = mapped_w

                overall_raw = str(data.get("overall_winner", "Tie")).strip().casefold().replace(" ", "")
                if overall_raw in {"answer1", "1"}:
                    final_winner = label_a
                elif overall_raw in {"answer2", "2"}:
                    final_winner = label_b
                elif overall_raw == "tie":
                    final_winner = "Tie"
                else:
                    # Derive overall winner from majority of 8 qualitative criteria
                    count_a = sum(1 for w in criteria_winners.values() if w == label_a)
                    count_b = sum(1 for w in criteria_winners.values() if w == label_b)
                    if count_a > count_b:
                        final_winner = label_a
                    elif count_b > count_a:
                        final_winner = label_b
                    else:
                        final_winner = "Tie"

                return {
                    "criteria_winners": criteria_winners,
                    "overall_winner": final_winner,
                    "summary": data.get("summary", ""),
                }
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Judge selection attempt %d failed: %s", attempt + 1, exc
                )

        if fallback_on_error:
            logger.warning(
                "All selection attempts failed (%s); returning neutral Tie fallback.",
                last_exc,
            )
            return {
                "criteria_winners": {crit: "Tie" for crit in self.CRITERIA_LIST},
                "overall_winner": "Tie",
                "summary": f"Fallback due to judge failure: {last_exc}",
            }
        assert last_exc is not None
        raise last_exc

    # -----------------------------------------------------------------------
    # 3. Efficiency & Latency Benchmark
    # -----------------------------------------------------------------------

    @staticmethod
    def measure_retrieval_latency(retriever_fn, query: str, runs: int = 3) -> Tuple[float, Any]:
        """
        Measure pure wall-clock retrieval time (seconds) isolating local DB search,
        graph traversal, and tensor projection from LLM inference latency.
        """
        if runs <= 0:
            raise ValueError("runs must be a positive integer")
        # one untimed WARMUP so cold caches do not poison
        # the measurement; report the MEDIAN plus p95 instead of the mean
        # (a single cold sample previously dominated the committed numbers:
        # 14.0 s "average" for vanilla RAG vs 0.98 s for HCAD).
        retriever_fn(query)
        latencies = []
        result = None
        for _ in range(runs):
            t0 = time.perf_counter()
            result = retriever_fn(query)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
        ordered = sorted(latencies)
        median = ordered[len(ordered) // 2] if ordered else 0.0
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else 0.0
        logger.info("Latency median=%.3fs p95=%.3fs runs=%d", median, p95, runs)
        return median, result


# ---------------------------------------------------------------------------
# Multi-Judge Ensemble (advisor directive; reviewer R2)
# ---------------------------------------------------------------------------

SCORE_DIMENSIONS = ("comprehensiveness", "diversity", "empowerment", "logical",
                    "readability", "overall_score")


def merge_counterbalanced(ab_result: Dict, ba_result: Dict) -> Dict:
    """Merge AB and BA pairwise verdicts of ONE judge (position-bias control).

    A criterion keeps its winner only when both presentation orders agree;
    disagreement is a Tie. Labels are already mapped to system names.
    """
    merged = {}
    ab_c = ab_result.get("criteria_winners", {})
    ba_c = ba_result.get("criteria_winners", {})
    for crit in BasePaperEvaluator.CRITERIA_LIST:
        a, b = ab_c.get(crit, "Tie"), ba_c.get(crit, "Tie")
        merged[crit] = a if a == b else "Tie"
    ab_o, ba_o = ab_result.get("overall_winner", "Tie"), ba_result.get("overall_winner", "Tie")
    return {
        "criteria_winners": merged,
        "overall_winner": ab_o if ab_o == ba_o else "Tie",
        "order_consistent": ab_o == ba_o,
        "summary": f"AB: {ab_result.get('summary', '')} | BA: {ba_result.get('summary', '')}",
    }


def majority_vote(labels: List[str]) -> str:
    """Plurality label across judges; any tie for first place -> "Tie"."""
    if not labels:
        return "Tie"
    counts: Dict[str, int] = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    best = max(counts.values())
    leaders = [lab for lab, c in counts.items() if c == best]
    return leaders[0] if len(leaders) == 1 else "Tie"


class MultiJudgeEvaluator:
    """Ensemble of independent LLM judges (default: gemma2:9b + qwen2.5:14b).

    - Each judge scores / compares independently; per-judge verdicts are
      kept so inter-judge agreement (Cohen's / Fleiss' kappa) can be
      computed from the checkpoint.
    - Consensus rubric score = mean over judges that returned a valid JSON
      verdict; consensus pairwise verdict = plurality vote over judges'
      counterbalanced (AB+BA) verdicts.
    - A judge failure is recorded as missing (None) and counted. It is
      NEVER replaced by a neutral 50.0 / "Tie" fallback: those fallbacks
      previously leaked into Table 4 as artificial ties.
    """

    def __init__(self, evaluators: Optional[List[Any]] = None, judges: Optional[List[Any]] = None):
        items = evaluators if evaluators is not None else judges
        if items is None or len(items) == 0:
            raise ValueError("MultiJudgeEvaluator needs at least one judge")
        self.evaluators = []
        for e in items:
            name = getattr(e, "name", None) or getattr(getattr(e, "judge", None), "model", None) or "judge"
            e.name = name
            self.evaluators.append(e)
        names = [e.name for e in self.evaluators]
        self.calls: Dict[str, int] = {n: 0 for n in names}
        self.failures: Dict[str, int] = {n: 0 for n in names}

    @property
    def judge_names(self) -> List[str]:
        return [e.name for e in self.evaluators]

    def failure_rate(self, name: str) -> float:
        calls = self.calls.get(name, 0)
        return self.failures.get(name, 0) / calls if calls else 0.0

    def check_ready(self) -> None:
        """Preflight every judge (pulled, loadable, answers JSON).

        Raises RuntimeError naming the failing judge(s), so a missing or
        out-of-memory judge stops the run before any generation work.
        """
        problems = []
        healthy = []
        for ev in self.evaluators:
            check = getattr(ev.judge, "check_ready", None)
            if check is None:
                healthy.append(ev)
                continue
            try:
                check()
                logger.info("Judge %s preflight OK", ev.name)
                healthy.append(ev)
            except Exception as exc:
                problems.append(f"{ev.name}: {exc}")
                is_cloud = ev.name.startswith(("gemini:", "groq:", "nvidia:", "huggingface:", "openrouter:"))
                if is_cloud:
                    logger.warning("Cloud API judge %s preflight failed: %s", ev.name, exc)
                else:
                    logger.error("Local judge %s preflight failed: %s", ev.name, exc)

        local_failures = [p for p in problems if not any(p.startswith(pfx) for pfx in ("gemini:", "groq:", "nvidia:", "huggingface:", "openrouter:"))]
        if local_failures or not healthy:
            raise RuntimeError("Judge preflight failed:\n  " + "\n  ".join(problems))

        if len(healthy) < len(self.evaluators):
            dropped = [ev.name for ev in self.evaluators if ev not in healthy]
            logger.warning(
                "Proceeding with %d ready judge(s) (%s); dropped unreachable API judge(s): %s",
                len(healthy), ", ".join(ev.name for ev in healthy), ", ".join(dropped)
            )
            self.evaluators = healthy

    def _record(self, name: str, ok: bool) -> None:
        self.calls[name] = self.calls.get(name, 0) + 1
        if not ok:
            self.failures[name] = self.failures.get(name, 0) + 1

    @classmethod
    def from_config(cls) -> "MultiJudgeEvaluator":
        """Build the ensemble from ``config.judge_ensemble_specs()``.

        Judge clients are STRICT: no provider/model substitution, so a
        result labelled 'gemma2:9b' was scored by gemma2:9b.
        """
        from src.config import JUDGE_MAX_TOKENS, JUDGE_TEMPERATURE, judge_ensemble_specs

        evaluators = []
        for provider, model in judge_ensemble_specs():
            client = UnifiedLLMClient(model=model, provider=provider,
                                      temperature=JUDGE_TEMPERATURE,
                                      max_tokens=JUDGE_MAX_TOKENS, strict=True,
                                      is_judge=True)
            evaluators.append(BasePaperEvaluator(judge_llm=client,
                                                 name=f"{provider}:{model}"))
        return cls(evaluators)

    def manifest(self) -> List[Dict[str, str]]:
        return [{"name": e.name,
                 "provider": getattr(e.judge, "provider", ""),
                 "model": getattr(e.judge, "model", ""),
                 "session_evaluations": self.calls.get(e.name, 0),
                 "session_failures": self.failures.get(e.name, 0)}
                for e in self.evaluators]

    # -- scoring ---------------------------------------------------------------
    def evaluate_scoring(self, question: str, answer: str, reference_document: str,
                         answer_evidence: str = "No retrieved evidence was supplied for this answer.",
                         retries: int = 1, fallback_on_error: bool = False) -> Dict[str, Any]:
        per_judge: Dict[str, Optional[Dict[str, Any]]] = {}
        for ev in self.evaluators:
            name = getattr(ev, "name", None) or getattr(getattr(ev, "judge", None), "model", "judge")
            try:
                raw = ev.evaluate_scoring(
                    question, answer, reference_document,
                    answer_evidence=answer_evidence, retries=retries,
                    fallback_on_error=fallback_on_error)
                per_judge[name] = raw.to_dict() if hasattr(raw, "to_dict") else dict(raw)
                self._record(name, True)
            except Exception as exc:
                logger.warning("Judge %s scoring failed: %s", name, exc)
                per_judge[name] = None
                self._record(name, False)
        valid = [r for r in per_judge.values() if r]
        consensus = (
            {d: sum(r[d] for r in valid) / len(valid) for d in SCORE_DIMENSIONS}
            if valid else None
        )
        return {"per_judge": per_judge, "consensus": consensus,
                "n_valid_judges": len(valid), "num_judges": len(self.evaluators)}

    # -- pairwise selection ----------------------------------------------------
    def evaluate_selection(self, question: str, answer_a: str, answer_b: str,
                           reference_document: str, label_a: str = "System A", label_b: str = "System B",
                           evidence_a: str = "No retrieved evidence was supplied for this answer.",
                           evidence_b: str = "No retrieved evidence was supplied for this answer.",
                           retries: int = 1, fallback_on_error: bool = False) -> Dict[str, Any]:
        """Counterbalanced (AB + BA) comparison by every judge, then vote."""
        per_judge: Dict[str, Optional[Dict[str, Any]]] = {}
        for ev in self.evaluators:
            name = getattr(ev, "name", None) or getattr(getattr(ev, "judge", None), "model", "judge")
            try:
                ab = ev.evaluate_selection(question, answer_a, answer_b, reference_document,
                                           label_a=label_a, label_b=label_b,
                                           evidence_a=evidence_a, evidence_b=evidence_b,
                                           retries=retries, fallback_on_error=fallback_on_error)
                ba = ev.evaluate_selection(question, answer_b, answer_a, reference_document,
                                           label_a=label_b, label_b=label_a,
                                           evidence_a=evidence_b, evidence_b=evidence_a,
                                           retries=retries, fallback_on_error=fallback_on_error)
                if isinstance(ab, dict) and "criteria_winners" in ab:
                    per_judge[name] = merge_counterbalanced(ab, ba)
                else:
                    per_judge[name] = ab
                self._record(name, True)
            except Exception as exc:
                logger.warning("Judge %s selection failed: %s", name, exc)
                per_judge[name] = None
                self._record(name, False)
        valid = [v for v in per_judge.values() if v]
        if valid:
            consensus = {
                "criteria_winners": {
                    c: majority_vote([v["criteria_winners"][c] for v in valid if "criteria_winners" in v and c in v["criteria_winners"]])
                    for c in BasePaperEvaluator.CRITERIA_LIST
                },
                "overall_winner": majority_vote([v["overall_winner"] for v in valid if "overall_winner" in v]),
            }
        else:
            consensus = None

        overall_winner = consensus["overall_winner"] if consensus else "Tie"
        criteria_winners = consensus["criteria_winners"] if consensus else {}

        # Compute inter-judge agreement
        valid_keys = [k for k, v in per_judge.items() if v and "criteria_winners" in v]
        agreement = {"type": "none", "kappa": 1.0}
        if len(valid_keys) == 2:
            from src.evaluation.metrics import cohens_kappa
            j1, j2 = valid_keys[0], valid_keys[1]
            r1 = [per_judge[j1]["criteria_winners"].get(c, "Tie") for c in BasePaperEvaluator.CRITERIA_LIST]
            r2 = [per_judge[j2]["criteria_winners"].get(c, "Tie") for c in BasePaperEvaluator.CRITERIA_LIST]
            k_val = cohens_kappa(r1, r2, categories=[label_a, label_b, "Tie"])
            agreement = {"type": "cohens_kappa", "kappa": k_val}
        elif len(valid_keys) > 2:
            from src.evaluation.metrics import fleiss_kappa
            cats = [label_a, label_b, "Tie"]
            cat_map = {c: i for i, c in enumerate(cats)}
            matrix = np.zeros((len(BasePaperEvaluator.CRITERIA_LIST), len(cats)), dtype=np.float64)
            for i, crit in enumerate(BasePaperEvaluator.CRITERIA_LIST):
                for jname in valid_keys:
                    w = per_judge[jname]["criteria_winners"].get(crit, "Tie")
                    if w in cat_map:
                        matrix[i, cat_map[w]] += 1.0
            k_val = fleiss_kappa(matrix)
            agreement = {"type": "fleiss_kappa", "kappa": k_val}

        return {
            "system_a": label_a,
            "system_b": label_b,
            "per_judge": per_judge,
            "consensus": consensus,
            "n_valid_judges": len(valid),
            "num_judges": len(self.evaluators),
            "overall_winner": overall_winner,
            "criteria_winners": criteria_winners,
            "agreement": agreement,
        }
