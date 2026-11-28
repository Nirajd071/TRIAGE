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
    r"not|no|never|neither|nor|without|cannot|can't|unlike"
    r"|doesn't|does\s+not|don't|do\s+not|didn't|did\s+not"
    r"|isn't|is\s+not|aren't|are\s+not|wasn't|was\s+not|weren't|were\s+not|won't"
    r"|lacks?|lack\s+of|absence\s+of|fails?\s+to|ruled?\s+out|unable\s+to"
    r")\b",
    re.IGNORECASE,
)

_CLAUSE_SPLIT_RE = re.compile(
    r"[.;?!\n]+|\b(?:but|however|although|though|whereas|while|yet|conversely|instead)\b",
    re.IGNORECASE,
)

# Explicit exclusion clauses for ExcluIR-style queries.
_EXCLUSION_CLAUSE_RE = re.compile(
    r"\b(?:without|excluding|exclude|except(?:\s+for)?|free\s+of|absence\s+of|"
    r"other\s+than|rather\s+than|instead\s+of|aside\s+from|apart\s+from|but\s+not|"
    r"unrelated\s+to|independent\s+of|"
    r"does\s+not\s+(?:involve|include|contain|mention|use)|"
    r"do\s+not\s+(?:involve|include|contain|mention|use)|"
    r"not\s+(?:involving|including|containing|with|mentioning|related\s+to))\s+"
    r"([a-z0-9][a-z0-9\- ]{1,55}?)"
    r"(?=[.,;?!\n]|\b(?:and|or|in|for|who|which|that)\b|$)",
    re.IGNORECASE,
)
_CONTENT_STOPWORDS = frozenset(
    "a an the of and or to in for on by with as at from any all other than "
    "is are was were be been being have has had does do did will would could should "
    "this that these those it its they them their which who whom what where when why how "
    "not no never neither nor without cannot can unlike "
    "patients patient cases case studies study drug medication treatment".split()
)
_LOCAL_NEG_PREFIX_RE = re.compile(
    r"\b(?:not|no|never|without|exclude[sd]?|excluding|except|free\s+of|absent|other\s+than)\b",
    re.IGNORECASE,
)

CLINICAL_MATCH_BONUS = 0.08
CLINICAL_CONFLICT_PENALTY = -0.35
EXCLUSION_MATCH_BONUS = 0.06
EXCLUSION_CONFLICT_PENALTY = -0.28
NEG_MATCH_BONUS = 0.08
NEG_CONFLICT_PENALTY = -0.14
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


def has_general_negation(text: str) -> bool:
    """True when *text* contains an explicit general negation cue."""
    return bool(_GENERAL_NEG_RE.search(text or ""))


def extract_exclusion_terms(query: str) -> list[str]:
    """Return non-stopword content terms targeted by exclusion phrases in *query*."""
    terms: list[str] = []
    seen: set[str] = set()
    for match in _EXCLUSION_CLAUSE_RE.finditer(query or ""):
        span = match.group(1)
        for tok in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", span.lower()):
            if len(tok) >= 3 and tok not in _CONTENT_STOPWORDS and tok not in seen:
                seen.add(tok)
                terms.append(tok)
    return terms


def strip_exclusion_terms(query: str, excl_terms: Sequence[str]) -> str:
    """Remove excluded target words from *query* before lexical BM25 matching."""
    if not excl_terms or not query:
        return query
    cleaned = query
    for term in excl_terms:
        cleaned = re.sub(rf"\b{re.escape(term)}\b", " ", cleaned, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", cleaned).strip()


def _has_unnegated_term(text: str, term: str) -> bool:
    """True if *term* appears in *text* without a local negation prefix."""
    pat = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
    for m in pat.finditer(text):
        prefix = text[max(0, m.start() - 32):m.start()]
        if not _LOCAL_NEG_PREFIX_RE.search(prefix):
            return True
    return False


def _clause_focus_tokens(text: str) -> tuple[set[str], set[str]]:
    """Split text into clauses and return (affirmed_tokens, negated_tokens)."""
    aff: set[str] = set()
    neg: set[str] = set()
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        clause = clause.strip()
        if not clause:
            continue
        toks = {
            t for t in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", clause.lower())
            if len(t) >= 3 and t not in _CONTENT_STOPWORDS
        }
        if not toks:
            continue
        if _GENERAL_NEG_RE.search(clause):
            neg |= toks
        else:
            aff |= toks
    return aff, neg


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
    q_neg = has_general_negation(query)
    q_aff_toks, q_neg_toks = _clause_focus_tokens(query) if q_short else (set(), set())

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

        # 3. Clause-level negation alignment for short claims (NevIR-style pairs)
        if not clinical_resolved and q_short and len(doc_str.split()) <= _SHORT_CLAIM_MAX_WORDS:
            c_neg = has_general_negation(doc_str)
            c_aff_toks, c_neg_toks = _clause_focus_tokens(doc_str)
            if q_neg and q_neg_toks:
                if not c_neg:
                    adj[i] += NEG_CONFLICT_PENALTY
                else:
                    neg_hit = len(q_neg_toks & c_neg_toks) / len(q_neg_toks)
                    aff_only = c_aff_toks - c_neg_toks
                    aff_leak = len(q_neg_toks & aff_only) / len(q_neg_toks)
                    if neg_hit > aff_leak:
                        adj[i] += NEG_MATCH_BONUS * min(1.0, neg_hit + 0.35)
                    elif aff_leak > neg_hit:
                        adj[i] += NEG_CONFLICT_PENALTY * min(1.0, aff_leak + 0.35)
            elif not q_neg and q_aff_toks:
                neg_only = c_neg_toks - c_aff_toks
                if c_neg and (q_aff_toks & neg_only) and not (q_aff_toks & c_aff_toks):
                    adj[i] += NEG_CONFLICT_PENALTY

    return adj


def pool_bm25_normalized(query: str, docs: Sequence[str]) -> np.ndarray:
    """Fast in-pool Okapi BM25 scores normalized to [0, 1] by max score."""
    n = len(docs)
    if n == 0 or not (query or "").strip():
        return np.zeros(n, dtype=np.float32)
    scores = BM25Index([str(d or "") for d in docs]).scores(query)
    peak = float(scores.max())
    if peak <= 0.0:
        return np.zeros(n, dtype=np.float32)
    return (scores / peak).astype(np.float32)
