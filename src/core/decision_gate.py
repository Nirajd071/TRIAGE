import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from src.config import (
    EVIDENCE_WORD_BUDGET,
    GROUP_MAX_TAILS,
    MAX_EVIDENCE_ITEMS,
    MAX_EVIDENCE_PER_RELATION,
    MIN_CLIP_WORDS,
    group_kg_facts_enabled,
)
from src.data.triple_text import TRIPLE_REL_MAP

logger = logging.getLogger(__name__)

# Sources whose items are single PrimeKG triples (n-ary facts have their own
# multi-role rendering and are never merged).
_TRIPLE_SOURCES = ("graph_hyperedge", "graph", "vector")
_MAX_TRIPLE_WORDS = 40  # longer "vector" documents are prose chunks, not triples


def _is_kg_triple(c: Dict[str, Any]) -> bool:
    meta = c.get("metadata") or {}
    if not (meta.get("head_name") and meta.get("tail_name") and meta.get("relation")):
        return False
    if c.get("source", "vector") not in _TRIPLE_SOURCES:
        return False
    return len(str(c.get("document") or "").split()) <= _MAX_TRIPLE_WORDS


def group_kg_facts(items: List[Dict[str, Any]], max_tails: int = GROUP_MAX_TAILS
                   ) -> List[Dict[str, Any]]:
    """Merge KG triples that share (head, relation) into one hyperedge line.

    "Metformin is indicated for A." + "Metformin is indicated for B." ->
    "Metformin is indicated for: A; B."  The merged item takes the position
    (and score) of the group's best member; prose and n-ary facts pass
    through untouched. `member_documents` keeps each original triple so
    gold matching and citation checks can still see individual facts.
    """
    out: List[Dict[str, Any]] = []
    groups: Dict[tuple, Dict[str, Any]] = {}
    for c in items:
        if not _is_kg_triple(c):
            out.append(c)
            continue
        meta = c["metadata"]
        key = (meta["head_name"].strip().lower(), meta["relation"].strip().lower())
        tail = meta["tail_name"].strip()
        doc = str(c.get("document") or "").strip()
        g = groups.get(key)
        if g is None:
            g = {**c, "metadata": dict(meta), "_tails": [tail], "member_documents": [doc]}
            groups[key] = g
            out.append(g)
        elif len(g["_tails"]) < max_tails and tail.lower() not in {t.lower() for t in g["_tails"]}:
            g["_tails"].append(tail)
            g["member_documents"].append(doc)
            if c.get("anchor"):
                g["anchor"] = True
    for g in groups.values():
        tails = g.pop("_tails")
        if len(tails) > 1:
            meta = g["metadata"]
            rel = TRIPLE_REL_MAP.get(meta["relation"], meta["relation"].replace("_", " "))
            g["document"] = f"{meta['head_name']} {rel}: {'; '.join(tails)}."
            g["metadata"]["grouped_tails"] = tails
        else:
            g.pop("member_documents", None)
    return out


def _clip_to_sentence(words: List[str], budget: int) -> str:
    """First `budget` words, cut back to the last sentence boundary."""
    text = " ".join(words[:budget])
    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    return (text if cut == -1 else text[: cut + 1]).strip()

try:
    from typesafe_sdk import TypeSafeClient, Noul, Score, Choice
    HAS_TYPESAFE = True
except ImportError:
    HAS_TYPESAFE = False

class DecisionGate:
    """
    JEV System One Decision Gate for token-efficient RAG.
    
    Uses TypeSafe AI's Jev (System One decision model) to evaluate candidate
    contexts against the query, returning structured probabilistic decisions (Noul, Choice, Score)
    to verify relevance and strictly constrain the token budget prior to prompt construction.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        min_relevance_score: float = 0.35,
        max_evidence_items: int = MAX_EVIDENCE_ITEMS,
        token_budget_words: int = EVIDENCE_WORD_BUDGET,
        use_jev: bool = True,
        max_per_relation: int = MAX_EVIDENCE_PER_RELATION,
    ):
        self.min_relevance_score = min_relevance_score
        self.max_evidence_items = max_evidence_items
        self.token_budget_words = token_budget_words
        self.max_per_relation = max_per_relation
        self.use_jev = use_jev

        self.client: Optional[Any] = None
        jev_key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("OPENROUTER_API_KEY")

        if self.use_jev and HAS_TYPESAFE and jev_key:
            try:
                base_url = "https://openrouter.ai/api/v1" if jev_key.startswith("sk-or-") else None
                model = "typesafe/jev-1.13" if base_url else None
                self.client = TypeSafeClient(api_key=jev_key, base_url=base_url, model=model, timeout=30.0)
                logger.info("JEV DecisionGate successfully initialized with TypeSafeClient.")
            except Exception as e:
                logger.warning(f"Failed to initialize TypeSafeClient for JEV: {e}")
        elif not HAS_TYPESAFE:
            logger.warning("typesafe-sdk is not installed. Run `pip install typesafe-sdk`.")
        elif not jev_key:
            logger.info("No TYPESAFE_API_KEY detected. JEV DecisionGate running in local calibrated mode.")

    @property
    def mode(self) -> str:
        """Active gate mode: 'live-typesafe' or 'local-calibrated'.

        Recorded in the benchmark run_manifest so silent mode drift between
        runs (e.g. a TYPESAFE_API_KEY appearing in the environment) is
        detectable in every results file.
        """
        return "live-typesafe" if self.client is not None else "local-calibrated"

    def verify_and_prune_candidates(
        self,
        candidates: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Decision Primitive 1: Score & Filter (Noul).
        Prunes candidate contexts that fall below the relevance quality floor
        or duplicate already verified biomedical relations.
        """
        if not candidates:
            return []

        # 1. Live JEV System One Decision (when API key is present)
        if self.client is not None:
            try:
                eval_slice = candidates[:min(8, len(candidates))]
                candidate_payload = [
                    {"index": i, "evidence": c.get("document", "")[:250]}
                    for i, c in enumerate(eval_slice)
                ]
                state = {
                    "clinical_query": query,
                    "candidates": candidate_payload,
                }
                choice_criteria = {str(i): f"Candidate context {i}" for i in range(len(candidate_payload))}
                choice_criteria["none"] = "None of the contexts provide verifiable evidence"

                questions = {
                    "is_factually_supported": Noul(
                        instructions="Does at least one of these candidate contexts directly provide verifiable biomedical evidence for the query?"
                    ),
                    "best_context_index": Choice(
                        instructions="Which candidate index provides the highest-quality clinical answer?",
                        criteria=choice_criteria,
                    ),
                }
                decision = self.client.system_one(state=state, questions=questions)
                answers = getattr(decision, "answers", {})
                noul_val = answers.get("is_factually_supported")
                choice_val = answers.get("best_context_index")

                best_choice = getattr(choice_val, "choice", "none") if choice_val else "none"
                is_supported = getattr(noul_val, "noul", 0.0) if noul_val else 0.0

                logger.info(
                    "JEV Decision: Supported (Noul)=%s, Best Index (Choice)=%s",
                    is_supported,
                    best_choice,
                )
                # OOD Gating: If JEV determines candidate evidence is ungrounded / unsupported,
                # suppress noisy context so the LLM relies purely on parametric domain knowledge
                # rather than being distracted by irrelevant biological triples.
                if is_supported < 0.10 and best_choice == "none":
                    logger.info("JEV OOD Gate triggered: candidates lack verifiable support. Suppressing noisy retrieval.")
                    return []

                if best_choice != "none" and str(best_choice).isdigit():
                    best_idx = int(best_choice)
                    if best_idx < len(candidates):
                        selected = candidates[best_idx]
                        candidates = [selected] + [c for i, c in enumerate(candidates) if i != best_idx]
            except Exception as e:
                logger.warning("JEV decision failed: %s. Continuing with calibrated gating.", e)

        # Relevance order is preserved (candidates arrive sorted by phi, with
        # the JEV-selected item promoted when the live gate ran). Diversity
        # is a soft cap: at most `max_per_relation` items of one relation
        # type are admitted before the gate moves on; over-cap items are
        # only used if budget remains. The previous rule put one item per
        # relation type FIRST, which pushed every PMC prose chunk (no
        # relation label) behind all KG triples regardless of relevance.
        primary, deferred = [], []
        seen_triples = set()
        seen_docs = set()
        relation_counts: Dict[str, int] = {}

        for c in candidates:
            score = c.get("phi_final", c.get("score", 0.0))
            # Dense anchors were admitted on their full-text cosine; the
            # hybrid floor would re-reject them for the missing-channel prior.
            if score < self.min_relevance_score and not c.get("anchor"):
                continue

            meta = c.get("metadata") or {}
            head = meta.get("head_name", "") or ""
            rel = meta.get("relation", "") or ""
            tail = meta.get("tail_name", "") or ""
            triple_key = (head.lower(), rel.lower(), tail.lower())
            doc_key = " ".join(str(c.get("document") or "").lower().split())

            # Skip duplicate facts to conserve prompt budget
            if any(triple_key) and triple_key in seen_triples:
                continue
            if doc_key and doc_key in seen_docs:
                continue
            if any(triple_key):
                seen_triples.add(triple_key)
            if doc_key:
                seen_docs.add(doc_key)

            relation = (meta.get("relation") or meta.get("display_relation") or "").strip().lower()
            if (relation and not c.get("anchor")
                    and relation_counts.get(relation, 0) >= self.max_per_relation):
                deferred.append(c)
                continue
            if relation:
                relation_counts[relation] = relation_counts.get(relation, 0) + 1
            primary.append(c)

        ordered = primary + deferred
        if group_kg_facts_enabled():
            ordered = group_kg_facts(ordered)

        # Keep the combined prompt evidence within the configured word budget
        # (the same EVIDENCE_WORD_BUDGET the vanilla arm gets). An item that
        # does not fit is clipped at a sentence boundary when a useful amount
        # of budget remains - the vanilla arm's rule - instead of being
        # dropped: previously only the FIRST item could be clipped, so a
        # relevant 200-word passage ranked behind a few triples vanished.
        # Short items further down may still fill what is left.
        #
        # Dense anchors are guaranteed a slot and their words (up to the
        # budget) are reserved up front, but they keep their score position:
        # a higher-scoring structured item still comes first.
        budget = self.token_budget_words
        reservation: Dict[int, int] = {}   # id(anchor item) -> reserved words
        for c in ordered:
            if c.get("anchor") and len(reservation) < self.max_evidence_items:
                n = len(str(c.get("document") or "").split())
                reservation[id(c)] = min(n, max(0, budget - sum(reservation.values())))
        reserved_words = sum(reservation.values())
        reserved_slots = len(reservation)

        bounded = []
        remaining_words = budget
        for candidate in ordered:
            if len(bounded) >= self.max_evidence_items or remaining_words <= 0:
                break
            doc = str(candidate.get("document") or "").strip()
            words = doc.split()
            if not words:
                continue
            is_anchor = id(candidate) in reservation
            if is_anchor:
                reserved_words -= reservation.pop(id(candidate))
                reserved_slots -= 1
                available, slots_free = remaining_words, True
            else:
                available = remaining_words - reserved_words
                slots_free = len(bounded) + reserved_slots < self.max_evidence_items
            if not slots_free or available <= 0:
                continue
            if len(words) <= available:
                bounded.append(candidate)
                remaining_words -= len(words)
            elif not bounded or is_anchor or available >= MIN_CLIP_WORDS:
                clipped = _clip_to_sentence(words, available)
                if not clipped:
                    continue
                bounded.append({**candidate, "document": clipped})
                remaining_words -= len(clipped.split())
        verified = bounded

        if not verified and candidates:
            top_score = candidates[0].get("phi_final", candidates[0].get("score", 0.0))
            if top_score >= self.min_relevance_score:
                verified = candidates[:1]

        logger.debug(
            "DecisionGate pruned %d candidates down to %d verified items.",
            len(candidates),
            len(verified),
        )
        return verified

    def format_constrained_evidence(
        self,
        verified_candidates: List[Dict[str, Any]],
    ) -> str:
        """
        Decision Primitive 2: Structured Token Economy.
        Formats evidence into dense, structured, noise-free lines
        rather than verbose repetitive text.
        """
        if not verified_candidates:
            return "(No high-confidence structured evidence found.)"

        lines = []
        for i, ctx in enumerate(verified_candidates, 1):
            meta = ctx.get("metadata") or {}
            head = meta.get("head_name", "")
            rel = meta.get("relation", meta.get("display_relation", ""))
            tail = meta.get("tail_name", "")
            doc = ctx.get("document", "")
            score = ctx.get("phi_final", ctx.get("score", 0.0))

            if head and rel and tail:
                evidence_str = (
                    f"[Evidence {i}] (Score: {score:.2f}) {head} --[{rel.replace('_', ' ')}]--> {tail}."
                )
            elif doc:
                clean_doc = " ".join(doc.split()[:45])
                evidence_str = f"[Evidence {i}] (Score: {score:.2f}) {clean_doc}"
            else:
                continue

            lines.append(evidence_str)

        return "\n".join(lines)
