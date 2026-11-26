"""Polarity, negation, exclusion, and in-pool lexical scoring for HCAD-RAG.

Addresses bi-encoder polarity blindness (S-PubMedBERT embeds "indicated for"
and "contraindicated in" close together and barely registers negation cues or
exclusion clauses) without altering offline/local execution.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np

from src.baselines.sparse_retrievers import BM25Index

# Clinical relation polarity (-1 contraindication/caution vs +1 indication/therapy).
# Negative patterns are checked BEFORE positive ones so "not recommended",
# "not indicated", and "should be avoided" never match as positive.
_POS_META_RELATIONS = frozenset({
    "indication", "treats", "therapeutic", "approved_for", "recommended_for",
    "manages", "alleviates", "ameliorates", "off-label use", "off_label_use",
})
_NEG_META_RELATIONS = frozenset({
    "contraindication", "contraindicated", "not_recommended", "avoid",
    "prohibited", "unsafe",
})

_NEG_CLINICAL_RE = re.compile(
    r"\b(?:"
    r"contraindicat(?:ed|ion|ions)"
    r"|(?:should|must|ought)\s+(?:not\s+be\s+(?:used|given|administered|prescribed)|be\s+avoided)"
    r"|avoid(?:ed|ing)?(?:\s+in|\s+for|\s+with)?"
    r"|not\s+(?:indicated|recommended|approved|safe|be\s+given|be\s+used)"
    r"|never\s+(?:be\s+)?(?:used?|given?|prescribed?|administered?)"
    r"|(?:special|serious|particular|high)\s+(?:danger|risk|hazard)\s+when\s+(?:given|taking|using|administered|prescribed)"
    r"|(?:requires?|with)\s+(?:great\s+|extreme\s+)?caution(?:\s+in)?"
    r"|cautioned\s+in|prohibited\s+in|unsafe\s+(?:in|for)|incompatible\s+with"
    r")\b",
    re.IGNORECASE,
)

_POS_CLINICAL_RE = re.compile(
    r"\b(?:"
    r"indication|indicated(?:\s+for|\s+in|\s+as)?"
    r"|used\s+(?:to\s+treat|in\s+the\s+treatment\s+of|for\s+(?:the\s+)?(?:treatment|management|prevention|relief)\s+of)"
    r"|first-?line(?:\s+(?:therapy|pharmacotherapy|treatment))?"
    r"|treats|therapeutic(?:\s+for)?|effective\s+against"
    r"|approved\s+for|recommended\s+for|prescribed\s+for|manages|alleviates|ameliorates"
    r"|(?:for\s+)?(?:prevention|relief|management|treatment)\s+of"
    r")\b",
    re.IGNORECASE,
)

# General clause-level negation cues for short claims (NevIR-style pairs).
_GENERAL_NEG_RE = re.compile(
    r"\b(?:"
    r"not|no|never|neither|nor|without|cannot|can't"
    r"|doesn't|does\s+not|don't|do\s+not|didn't|did\s+not"
    r"|isn't|is\s+not|aren't|are\s+not|wasn't|was\s+not|weren't|were\s+not|won't"
    r"|lacks?|lack\s+of|absence\s+of|fails?\s+to|ruled?\s+out|unable\s+to"
    r")\b",
    re.IGNORECASE,
)

# Explicit exclusion clauses for ExcluIR-style queries.
_EXCLUSION_CLAUSE_RE = re.compile(
    r"\b(?:without|excluding|except(?:\s+for)?|free\s+of|absence\s+of|"
    r"not\s+(?:involving|including|containing|with|mentioning))\s+"
    r"([a-z0-9][a-z0-9\- ]{1,45}?)"
    r"(?=[.,;?!\n]|\b(?:and|or|in|for|who|which|that)\b|$)",
    re.IGNORECASE,
)
_EXCL_STOPWORDS = frozenset(
    "a an the of and or to in for on by with as at from any all other than "
    "patients patient cases case studies study".split()
)
_LOCAL_NEG_PREFIX_RE = re.compile(
    r"\b(?:not|no|never|without|exclude[sd]?|excluding|except|free\s+of|absent)\b",
    re.IGNORECASE,
)

CLINICAL_MATCH_BONUS = 0.08
CLINICAL_CONFLICT_PENALTY = -0.35
EXCLUSION_MATCH_BONUS = 0.05
EXCLUSION_CONFLICT_PENALTY = -0.25
NEG_MATCH_BONUS = 0.04
NEG_CONFLICT_PENALTY = -0.12
_SHORT_CLAIM_MAX_WORDS = 120


def classify_clinical_polarity(
    text: str, metadata: dict[str, Any] | None = None
) -> int:
    """Return +1 (indication/therapy), -1 (contraindication/caution), or 0."""
    if metadata:
        rel = str(metadata.get("relation") or metadata.get("display_relation") or "").strip().lower()
        if rel in _NEG_META_RELATIONS:
            return -1
        if rel in _POS_META_RELATIONS:
            return 1

    txt = text or ""
    if _NEG_CLINICAL_RE.search(txt):
        return -1
    if _POS_CLINICAL_RE.search(txt):
        return 1
    return 0


def extract_exclusion_terms(query: str) -> list[str]:
    """Return non-stopword content terms targeted by exclusion phrases in *query*."""
    terms: list[str] = []
    seen: set[str] = set()
    for match in _EXCLUSION_CLAUSE_RE.finditer(query or ""):
        span = match.group(1)
        for tok in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", span.lower()):
            if len(tok) >= 3 and tok not in _EXCL_STOPWORDS and tok not in seen:
                seen.add(tok)
                terms.append(tok)
    return terms


def _has_unnegated_term(text: str, term: str) -> bool:
    """True if *term* appears in *text* without a local negation prefix."""
    pat = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    found_unnegated = False
    for m in pat.finditer(text):
        prefix = text[max(0, m.start() - 28):m.start()]
        if not _LOCAL_NEG_PREFIX_RE.search(prefix):
            found_unnegated = True
            break
    return found_unnegated


def polarity_adjustments(
    query: str,
    docs: Sequence[str],
    metas: Sequence[dict[str, Any] | None] | None = None,
) -> np.ndarray:
    """Per-candidate score adjustments for clinical polarity, exclusions, and negation."""
    n = len(docs)
    if n == 0:
        return np.empty((0,), dtype=np.float32)

    adj = np.zeros(n, dtype=np.float32)
    q_pol = classify_clinical_polarity(query)
    excl_terms = extract_exclusion_terms(query)
    q_short = len((query or "").split()) <= _SHORT_CLAIM_MAX_WORDS
    q_neg = bool(_GENERAL_NEG_RE.search(query or ""))

    for i, doc in enumerate(docs):
        doc_str = str(doc or "")
        meta = metas[i] if metas is not None and i < len(metas) else None
        c_pol = classify_clinical_polarity(doc_str, meta)

        # 1. Clinical polarity alignment / conflict (indication vs contraindication)
        clinical_resolved = False
        if q_pol != 0 and c_pol != 0:
            adj[i] += CLINICAL_MATCH_BONUS if q_pol == c_pol else CLINICAL_CONFLICT_PENALTY
            clinical_resolved = True

        # 2. Exclusion target check (query explicitly excludes terms)
        if excl_terms:
            violated = any(_has_unnegated_term(doc_str, t) for t in excl_terms)
            adj[i] += EXCLUSION_CONFLICT_PENALTY if violated else EXCLUSION_MATCH_BONUS
            continue

        # 3. Clause negation alignment for short claims/statements (NevIR-style)
        if not clinical_resolved and q_short and len(doc_str.split()) <= _SHORT_CLAIM_MAX_WORDS:
            c_neg = bool(_GENERAL_NEG_RE.search(doc_str))
            if q_neg != c_neg:
                adj[i] += NEG_CONFLICT_PENALTY
            elif q_neg and c_neg:
                adj[i] += NEG_MATCH_BONUS

    return adj


def pool_bm25_normalized(query: str, docs: Sequence[str]) -> np.ndarray:
    """Fast in-pool Okapi BM25 scores normalized to [0, 1] by max score."""
    n = len(docs)
    if n == 0:
        return np.empty((0,), dtype=np.float32)
    scores = BM25Index([str(d or "") for d in docs]).scores(query)
    peak = float(scores.max())
    if peak <= 0.0:
        return np.zeros(n, dtype=np.float32)
    return (scores / peak).astype(np.float32)
