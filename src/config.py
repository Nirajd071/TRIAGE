import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
load_dotenv(PROJECT_ROOT / ".env", override=False)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORDB_DIR = DATA_DIR / "vectordb"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
TENSOR_MODEL_PATH = MODELS_DIR / "dynamic_tensor.pt"
TABLES_DIR = RESULTS_DIR / "tables"
CACHE_DIR = RESULTS_DIR / "cache"
FIGURES_DIR = RESULTS_DIR / "figures"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTORDB_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Datasets
PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
PRIMEKG_FILENAME = "kg.csv"

MEDHALLU_DATASET_ID = "UTAustin-AIHealth/MedHallu"
MEDHALLU_CONFIG_LABELED = "pqa_labeled"
MEDHALLU_CONFIG_ARTIFICIAL = "pqa_artificial"

# Embedding & Entity Linking Models
EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBEDDING_DIMENSION = 768
ENTITY_LINKING_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# HCAD Core Parameters
NUM_SEMANTIC_COMPONENTS = 3  # Entity, Relation, Modifier
HCAD_MATRIX_SHAPE = (NUM_SEMANTIC_COMPONENTS, EMBEDDING_DIMENSION)

# Anisotropic Transformation Weights
ANISO_ALPHA = 0.4   # Node degree
ANISO_BETA = 0.4    # Edge confidence
ANISO_GAMMA = 0.2   # Inverse document frequency

# Retrieval Score Weights
SCORE_WEIGHT_ENTITY = 0.3
SCORE_WEIGHT_RELATION = 0.5
SCORE_WEIGHT_MODIFIER = 0.2

# Statistical Filtering
FILTERING_K = 1.0
MIN_PHI_SCORE = 0.10

# Hybrid scoring: Phi = w_d * dense_cosine(query, document)
#                      + (1 - w_d) * Phi_struct(E/R/M channels).
# The structured channels compare short extracted fragments (entity names,
# relation phrase, modifiers); on their own they discard the full-text
# semantic match that dense retrieval ranks by. Fusing the two keeps the
# dense signal as a floor and lets the decoupled channels re-rank on top
# of it. HCAD_DENSE_WEIGHT=0 restores the legacy structure-only score.
HYBRID_DENSE_WEIGHT = 0.5


def dense_weight() -> float:
    """Active dense-score weight w_d in [0, 1] (env HCAD_DENSE_WEIGHT)."""
    raw = os.environ.get("HCAD_DENSE_WEIGHT")
    if raw is None or raw == "":
        return HYBRID_DENSE_WEIGHT
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return HYBRID_DENSE_WEIGHT


# Anisotropic per-dimension weighting. The topology statistics (degree,
# edge confidence, IDF) were mapped onto three contiguous 256-dim blocks of
# the sentence embedding, but embedding coordinates carry no such meaning,
# PrimeKG edges have no confidence attribute (constant 1.0) and IDF used a
# constant document frequency - the weights were effectively a fixed vector
# that suppressed the first third of every embedding. Default OFF
# (isotropic weighted cosine == plain cosine); HCAD_ANISO_DIM_WEIGHTING=1
# re-enables the legacy block weighting for ablations.
ANISO_DIM_WEIGHTING = False


def aniso_dim_weighting_enabled() -> bool:
    default = "1" if ANISO_DIM_WEIGHTING else "0"
    return os.environ.get("HCAD_ANISO_DIM_WEIGHTING", default) == "1"


# Entity linking: exact (case-insensitive) PrimeKG name match first, then
# SapBERT nearest neighbour above this cosine threshold.
ENTITY_LINK_THRESHOLD = 0.65

# Candidate pool composition.
CANDIDATE_POOL_MAX = 80          # total candidates scored per query
GRAPH_TRIPLES_PER_ENTITY = 12    # hyperedge triples emitted per linked entity
GRAPH_TRIPLES_PER_HYPEREDGE = 3  # spread over relation-typed hyperedges
NARY_FACTS_PER_QUERY = 10

# DecisionGate: at most this many evidence items may share one relation
# type before lower-scored items of other relations/prose are preferred.
MAX_EVIDENCE_PER_RELATION = 3

# ---------------------------------------------------------------------------
# Evidence assembly (HCAD-RAG final context, same 350-word budget as dense RAG)
# ---------------------------------------------------------------------------
# Dense anchor: the top-K full-text dense matches are always admitted (slot
# and words reserved) when their cosine clears DENSE_ANCHOR_MIN; they keep
# their hybrid-score position. The E/R/M channels compare short extracted
# fragments, so prose passages (no entity/relation metadata -> missing-
# channel prior) were systematically out-ranked by short KG triples, and the
# passage that best matches the whole question could be gated out. With the
# anchor, HCAD's evidence is "best dense passage + structure-selected
# complement", so the comparison with dense RAG isolates what the
# structured selection adds. HCAD_DENSE_ANCHOR_K=0 restores legacy selection.
DENSE_ANCHOR_K = 1
DENSE_ANCHOR_MIN = 0.40


def dense_anchor_k() -> int:
    try:
        return max(0, int(os.environ.get("HCAD_DENSE_ANCHOR_K", DENSE_ANCHOR_K)))
    except ValueError:
        return DENSE_ANCHOR_K


def dense_anchor_min() -> float:
    try:
        return float(os.environ.get("HCAD_DENSE_ANCHOR_MIN", DENSE_ANCHOR_MIN))
    except ValueError:
        return DENSE_ANCHOR_MIN


# Hyperedge rendering: KG triples sharing (head, relation) are emitted as ONE
# evidence line ("Metformin is indicated for: A; B; C.") instead of repeating
# head + verb per fact - the hyperedge as a unit, ~3x more facts per word of
# the shared budget. HCAD_GROUP_FACTS=0 restores one line per triple.
GROUP_KG_FACTS = True
GROUP_MAX_TAILS = 5


def group_kg_facts_enabled() -> bool:
    default = "1" if GROUP_KG_FACTS else "0"
    return os.environ.get("HCAD_GROUP_FACTS", default) == "1"


# Budget packing: an item that does not fit the remaining word budget is
# clipped at a sentence boundary (as the dense arm does) when at least this
# many words remain, instead of being dropped.
MIN_CLIP_WORDS = 25

# Semantic Decoupling 
ORTHO_STRENGTH = 0.70        # soft Gram-Schmidt blend weight (eta)
GS_EPS = 1e-8
GS_RESIDUAL_TOL = 1e-3      # below this, a component is marked absent (presence mask)
NEUTRAL_RELATION_TEXT = "biomedical relation"   # anchor, never equal to entity text
NEUTRAL_MODIFIER_TEXT = "no qualifier stated"

# Knowledge-graph relation exclusion policy
KG_EXCLUDED_RELATIONS = {"drug_drug"}

# ---------------------------------------------------------------------------
# Deliverable-B (Phase 3): query-conditioned channel masking, relation
# vocabulary mapping, and Gate-2 saturation cap.
# ---------------------------------------------------------------------------

# Calibrated prior similarity assigned to a candidate that is missing a
# channel the query actively uses (p_k^Q = 1, p_k^C = 0). Replaces both the
# old silent 1.0 (neutral-anchor cosine) and the proposed drop-from-
# denominator bug; range [0.15, 0.25] per the epistemic audit.
CHANNEL_MISSING_PRIOR = 0.20

# Query-conditioned presence masking: omega_k = w_k * p_k^Q / sum_j w_j * p_j^Q.
# Toggles default ON (the recommended experiment); HCAD_PRESENCE_MASKING=0
# disables for paired before/after measurement.
CHANNEL_PRESENCE_MASKING = True

# PrimeKG schema labels -> canonical clinical verb phrases. Raw schema
# strings ("drug protein") barely cosine-match natural query verbs
# ("targets") in S-PubMedBert; the mapped phrases close that gap. drug_drug
# stays excluded (KG_EXCLUDED_RELATIONS) to preserve the benchmark protocol
# and the memory footprint.
RELATION_VOCAB_MAPPING = True
RELATION_VOCAB_MAP = {
    "contraindication": "is contraindicated in and must not be used for",
    "indication": "is indicated for the treatment of",
    "off-label use": "is used off-label for",
    "drug_effect": "causes effect or adverse effect",
    "disease_disease": "is associated with",
    "drug_protein": "targets or binds protein",
    "disease_protein": "is associated with altered protein",
    "disease_phenotype_positive": "presents with phenotype",
    "disease_phenotype_negative": "excludes phenotype and is not characterized by",
    "biomarker": "is a biomarker for",
    "phenotype_protein": "phenotype associated with protein",
    "protein_protein": "interacts with protein",
}

# Gate-2 saturation ceiling: tau_eff = min(tau_sat, max(Phi_min, mu + k*sigma)).
# None disables the cap until it is calibrated on the development pool as the
# 25th percentile of gold development candidate scores; set via
# HCAD_TAU_SAT (e.g. "0.42") once calibrated.
TAU_SAT = None
TAU_SAT_PERCENTILE = 25.0


def presence_masking_enabled() -> bool:
    """Runtime toggle for query-conditioned channel masking."""
    default = "1" if CHANNEL_PRESENCE_MASKING else "0"
    return os.environ.get("HCAD_PRESENCE_MASKING", default) == "1"


def relation_vocab_enabled() -> bool:
    """Runtime toggle for PrimeKG relation-vocabulary mapping."""
    default = "1" if RELATION_VOCAB_MAPPING else "0"
    return os.environ.get("HCAD_RELATION_VOCAB", default) == "1"


# Query-side relation types (RelationExtractor rule labels) rendered with
# the same canonical verb phrases the candidates use, so the R channel
# compares like with like ("TREATS" vs "is indicated for the treatment of").
QUERY_RELATION_PHRASES = {
    "TREATS": RELATION_VOCAB_MAP["indication"],
    "INDICATED_FOR": RELATION_VOCAB_MAP["indication"],
    "CONTRAINDICATED": RELATION_VOCAB_MAP["contraindication"],
    "NOT_RECOMMENDED": RELATION_VOCAB_MAP["contraindication"],
    "CAUSES": RELATION_VOCAB_MAP["drug_effect"],
    "TARGETS": RELATION_VOCAB_MAP["drug_protein"],
    "INHIBITS": RELATION_VOCAB_MAP["drug_protein"],
    "ACTIVATES": RELATION_VOCAB_MAP["drug_protein"],
    "BIOMARKER_FOR": RELATION_VOCAB_MAP["biomarker"],
    "ASSOCIATED_WITH": RELATION_VOCAB_MAP["disease_disease"],
}

# Query relation label -> PrimeKG relations it asks about. Hyperedges of
# these relations are expanded first for the query's entities.
QUERY_RELATION_TO_PRIMEKG = {
    "TREATS": {"indication", "off-label use"},
    "INDICATED_FOR": {"indication", "off-label use"},
    "CONTRAINDICATED": {"contraindication"},
    "NOT_RECOMMENDED": {"contraindication"},
    "CAUSES": {"drug_effect", "disease_phenotype_positive"},
    "TARGETS": {"drug_protein"},
    "INHIBITS": {"drug_protein"},
    "ACTIVATES": {"drug_protein"},
    "BIOMARKER_FOR": {"biomarker"},
    "ASSOCIATED_WITH": {"disease_disease", "disease_protein"},
}

# PrimeKG stores every edge in both directions. For directed relations the
# head must be of this node type; reversed copies are swapped back so the
# evidence reads correctly ("Metformin is indicated for X", never
# "X is indicated for Metformin") and duplicates collapse.
RELATION_HEAD_TYPE = {
    "indication": "drug",
    "contraindication": "drug",
    "off-label use": "drug",
    "drug_effect": "drug",
    "drug_protein": "drug",
    "disease_protein": "disease",
    "disease_phenotype_positive": "disease",
    "disease_phenotype_negative": "disease",
    "phenotype_protein": "effect/phenotype",
}


def map_query_relation(raw_relation: str) -> str:
    """Canonical verb phrase for a query-side rule relation label."""
    if not raw_relation:
        return ""
    if relation_vocab_enabled():
        phrase = QUERY_RELATION_PHRASES.get(raw_relation.strip().upper())
        if phrase:
            return phrase
    return raw_relation.replace("_", " ").lower()


def map_relation_text(raw_relation: str) -> str:
    """Map a raw PrimeKG relation label to a canonical verb phrase.

    Unknown relations keep the legacy underscores-to-spaces rendering.
    """
    if not raw_relation:
        return ""
    key = raw_relation.strip().lower()
    if relation_vocab_enabled() and key in RELATION_VOCAB_MAP:
        return RELATION_VOCAB_MAP[key]
    return key.replace("_", " ")


def tau_sat() -> float | None:
    """Calibrated Gate-2 saturation ceiling, if set."""
    raw = os.environ.get("HCAD_TAU_SAT")
    if raw:
        try:
            return float(raw)
        except ValueError:
            return None
    return TAU_SAT


# ---------------------------------------------------------------------------
# Phase 4a: hub-degree bias correction (soft, no hard cap).
#
# PrimeKG is scale-free: promiscuous hubs (NR3C1, CYP3A4, ALB, TP53, major
# diseases) have degrees orders of magnitude above specific targets
# (SERPINA6, RBP4). Descending-degree neighbor truncation deterministically
# floods candidate slices with hubs; a HARD degree cap would blind the
# system to canonical receptors and comorbidities (deg > 500 is normal for
# them), so the deployed rule is a soft inverse-log-degree weighting:
# hubs stay reachable but specific neighbors win the truncation slice.
# ---------------------------------------------------------------------------

# "desc"     - legacy descending-degree truncation (ablation baseline)
# "soft"     - soft inverse-log-degree weighting, NO hard drop (default)
# "asc_cap"  - ascending degree with hard deg<=cap drop (ablation-only;
#              expected to lose canonical high-degree targets)
HUB_EXPANSION_RULE = "soft"
HUB_DEGREE_CAP = 500
HUB_SOFT_WEIGHT_EPS = 1.0


def expansion_rule() -> str:
    """Active neighborhood-expansion rule (env HCAD_EXPANSION_RULE)."""
    rule = os.environ.get("HCAD_EXPANSION_RULE", HUB_EXPANSION_RULE)
    if rule not in ("desc", "soft", "asc_cap"):
        raise ValueError(f"Unknown HCAD_EXPANSION_RULE: {rule!r}")
    return rule


def hub_degree_cap() -> int:
    return int(os.environ.get("HCAD_HUB_DEGREE_CAP", HUB_DEGREE_CAP))


def hub_weight(degree: int) -> float:
    """Soft inverse-log-degree specificity weight, in (0, 1].

    1 / (1 + ln(1 + deg)) - monotonically decreasing in degree, never zero:
    a 10^5-degree hub keeps ~12% weight (reachable), while a degree-3
    specific target keeps ~50%.
    """
    import math

    return 1.0 / (1.0 + math.log1p(max(0, int(degree))) * HUB_SOFT_WEIGHT_EPS)


# ---------------------------------------------------------------------------
# Phase 4b: latency precompute with a config-versioned decoupler cache.
# ---------------------------------------------------------------------------

DECOUPLER_CACHE_ENABLED = True
DECOUPLER_EMBED_CACHE_MAXSIZE = 20_000  # ~61 MB at d=768 float32


def decoupler_cache_enabled() -> bool:
    default = "1" if DECOUPLER_CACHE_ENABLED else "0"
    return os.environ.get("HCAD_DECOUPLER_CACHE", default) == "1"

# LLM
LLM_OLLAMA_TIMEOUT_S = 120.0

# Graph Expansion
GRAPH_HOP_DEPTH = 2
MAX_NEIGHBORS_PER_HOP = 50
MAX_EVIDENCE_ITEMS = 8
EVIDENCE_WORD_BUDGET = 350

# Vanilla Baseline
VANILLA_TOP_K = 5

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia")  # "nvidia", "huggingface", or "ollama"
LLM_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"  # Separate judge to avoid self-preference bias
# Local (ollama) judge. Resolved from the registry in src/judge_models.py:
#   JUDGE_OLLAMA_MODEL (exact tag) > JUDGE_MODEL_INDEX + JUDGE_QUANT > qwen2.5:72b.
# The Kaggle notebook sets JUDGE_MODEL_INDEX; never edit tags here.
from src.judge_models import resolve_judge_ollama_model as _resolve_judge

JUDGE_OLLAMA_MODEL = _resolve_judge()
# Judge context window. The pairwise prompt carries two answers (up to
# LLM_MAX_TOKENS each) plus two evidence blocks; Ollama's small default
# context silently truncates the front of the prompt (the instructions).
JUDGE_NUM_CTX = int(os.getenv("JUDGE_NUM_CTX", "8192"))
# Abort the benchmark when more than this fraction of judge evaluations in
# the current session failed (unparseable JSON / OOM / timeouts): such a run
# measures nothing. Checkpointed questions survive the abort.
JUDGE_MAX_FAILURE_RATE = float(os.getenv("JUDGE_MAX_FAILURE_RATE", "0.5"))
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
# Text-optimized NIM models (vision-instruct variants underperform on clinical
# prose and waste tokens on the image tower). Override via env if needed.
LLM_NVIDIA_MODEL = os.getenv("LLM_NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
JUDGE_NVIDIA_MODEL = os.getenv("JUDGE_NVIDIA_MODEL", "qwen/qwen2.5-72b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Hub downloads (SapBERT/S-PubMedBert) still need this
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "hcad-rag")
WANDB_ENTITY = os.getenv("WANDB_ENTITY") or None
WANDB_MODE = os.getenv("WANDB_MODE", "online")
# Generator: Llama-3.1-8B-Instruct, 4-bit (Q4_K_M, ~4.9 GB). Explicit quant
# tag (identical weights to Ollama's default "llama3.1:8b"), so the
# quantization is pinned instead of depending on Ollama's default tag.
LLM_MODEL = "llama3.1:8b-instruct-q4_K_M"
# Untagged spelling accepted as an alias of LLM_MODEL.
LLM_MODEL_ALIASES = ("llama3.1:8b",)
# Base-paper Fig. 3 parity: evaluate augmentation generality across generators.
BENCHMARK_GENERATORS = [LLM_MODEL]  # e.g. ["llama3.1:8b", "qwen3:14b", "deepseek-r1:8b"]
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024
ENABLE_FEW_SHOT = False  # zero-shot default; opt into few-shot explicitly
FEW_SHOT_COUNT = 4  # consumed by PromptBuilder.build_mcq_prompt

# Embeddings are always L2-normalized at encode time (cosine consistency)

LLM_SYSTEM_PROMPT = """You are a biomedical research assistant specializing in clinical medicine.
Use the provided context as primary evidence, and synthesize it with your expert clinical
domain knowledge to deliver comprehensive, accurate, and actionable answers.
Clearly cite specific evidence from the context where applicable, and supplement
with established medical knowledge when the context is incomplete.
Be precise, thorough, and clinically rigorous."""

# System prompt for any arm answering WITHOUT retrieved evidence: the
# closed-book arm and HCAD-RAG when its gate abstains. Shared so both
# no-evidence cases are prompted identically.
NO_CONTEXT_SYSTEM_PROMPT = (
    "You are a biomedical research assistant specializing in clinical medicine. "
    "Use your expert clinical domain knowledge to deliver comprehensive, accurate, "
    "and actionable answers. Be precise, thorough, and clinically rigorous."
)

# Vector Database
CHROMA_COLLECTION_NAME = "biomedical_contexts"
CHROMA_DISTANCE_METRIC = "cosine"

# NER Models
NER_MODELS = {
    "drug_disease": "en_ner_bc5cdr_md",
    "molecular": "en_ner_jnlpba_md",
}

# Relation Types & Negation
RELATION_TYPES = [
    "TREATS",
    "CAUSES",
    "CONTRAINDICATED",
    "INTERACTS_WITH",
    "TARGETS",
    "ASSOCIATED_WITH",
    "INHIBITS",
    "ACTIVATES",
    "UPREGULATES",
    "DOWNREGULATES",
    "BIOMARKER_FOR",
    "NOT_RECOMMENDED",
    "INDICATED_FOR",
]

NEGATION_CUES = [
    "not", "no", "never", "without", "avoid", "contraindicated",
    "should not", "must not", "do not", "doesn't", "does not",
    "isn't", "is not", "wasn't", "was not", "cannot", "can't",
    "not recommended", "discontinued", "withdrawn", "prohibited",
    "lack of", "absence of", "failure to",
    "rules out", "ruled out", "unlikely", "fails to improve", "cannot tolerate",
    "refractory to", "excluded",
]

# Evaluation Parameters
DEFAULT_EVAL_SAMPLES = 200
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Benchmark arms (reviewer request: explicit baseline naming + standard
# retrieval baselines + component ablations).
#
# Every arm has a stable machine id (used in checkpoint keys and CLI flags)
# and ONE display label used verbatim in result JSON, tables and figures,
# so a table and the figure drawn from it can never disagree on a name.
# ---------------------------------------------------------------------------

# The generic "Baseline LLM" term is retired: the closed-book arm is the
# generator itself with retrieval removed.
BASELINE_LABEL = "Llama-3.1-8B-Instruct (Zero-Retrieval / Closed-Book)"
BASELINE_SHORT_LABEL = "Llama-3.1-8B (Closed-Book)"

ARM_LABELS = {
    "closed_book": BASELINE_LABEL,
    "bm25": "BM25 RAG (top-5)",
    "dense": "Dense RAG (S-PubMedBERT, top-5)",
    "hybrid": "Hybrid BM25+Dense RAG (RRF, top-5)",
    "hcad": "HCAD-RAG",
    "hcad_no_decoupling": "HCAD-RAG w/o Decoupling",
    "hcad_no_gs": "HCAD-RAG w/o Soft Gram-Schmidt",
    "hcad_no_gate": "HCAD-RAG w/o Decision Gate",
}
ARM_SHORT_LABELS = {
    "closed_book": BASELINE_SHORT_LABEL,
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
    "hcad": "HCAD-RAG",
    "hcad_no_decoupling": "w/o Decoupling",
    "hcad_no_gs": "w/o Soft GS",
    "hcad_no_gate": "w/o Gate",
}
STANDARD_ARMS = ["closed_book", "bm25", "dense", "hybrid", "hcad"]
ABLATION_ARMS = ["hcad_no_decoupling", "hcad_no_gs", "hcad_no_gate"]

# Labels used by checkpoints / result files written before the renaming.
# Readers map them onto the current labels; writers never emit them.
LEGACY_SYSTEM_LABELS = {
    "Baseline LLM": BASELINE_LABEL,
    "Vanilla RAG": ARM_LABELS["dense"],
}


def canonical_system_label(label: str) -> str:
    """Map a legacy system label onto its current display label."""
    return LEGACY_SYSTEM_LABELS.get(label, label)


# Component ablations are expressed as environment overrides of the
# runtime toggles below, so an ablation arm runs the *same* code path as
# the full system with exactly one component switched off.
ABLATION_ENV = {
    # Phi = dense cosine only: the E/R/M structured channel score is removed.
    "hcad_no_decoupling": {"HCAD_DENSE_WEIGHT": "1.0"},
    # eta = 0: relation/modifier embeddings are used raw (no soft Gram-Schmidt).
    "hcad_no_gs": {"HCAD_ORTHO_STRENGTH": "0.0"},
    # The JEV decision gate is bypassed after the Heaviside filter.
    "hcad_no_gate": {"HCAD_DECISION_GATE": "0"},
}


def ortho_strength() -> float:
    """Active soft Gram-Schmidt blend eta in [0, 1] (env HCAD_ORTHO_STRENGTH)."""
    raw = os.environ.get("HCAD_ORTHO_STRENGTH")
    if raw is None or raw == "":
        return ORTHO_STRENGTH
    try:
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        return ORTHO_STRENGTH


def decision_gate_enabled() -> bool:
    """Runtime toggle for the JEV decision gate (env HCAD_DECISION_GATE)."""
    return os.environ.get("HCAD_DECISION_GATE", "1") != "0"


# ---------------------------------------------------------------------------
# Multi-judge evaluation (advisor directive). JUDGE_ENSEMBLE is a comma
# separated list of "provider:model" specs; the first entry is the primary
# judge. The 70B judge is an optional API judge (NVIDIA NIM by default).
# ---------------------------------------------------------------------------
# Explicit 4-bit tags (registry indices 5 and 3) - same weights as Ollama's
# default "gemma2:9b" / "qwen2.5:14b" tags, but the quantization is pinned.
DEFAULT_JUDGE_ENSEMBLE = ("ollama:gemma2:9b-instruct-q4_K_M,"
                          "ollama:qwen2.5:14b-instruct-q4_K_M")
OPTIONAL_API_JUDGE = "nvidia:meta/llama-3.3-70b-instruct"


def judge_ensemble_specs() -> list[tuple[str, str]]:
    """Parsed judge ensemble: [(provider, model), ...] (primary first).

    Precedence: JUDGE_ENSEMBLE (explicit "provider:model" list) >
    JUDGE_MODEL_INDEX (registry index/indices, quantized) > the default
    local pair. JUDGE_ENABLE_API_JUDGE=1 appends the optional
    Llama-3.3-70B API judge.
    """
    raw = os.environ.get("JUDGE_ENSEMBLE", "").strip()
    if not raw:
        # Notebook path: JUDGE_MODEL_INDEX ("2" or "0,3") -> quantized tags
        # from the registry in src/judge_models.py.
        from src.judge_models import resolve_judge_tags
        tags = resolve_judge_tags()
        raw = ",".join(f"ollama:{t}" for t in tags) if tags else DEFAULT_JUDGE_ENSEMBLE
    specs: list[tuple[str, str]] = []
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    if os.environ.get("JUDGE_ENABLE_API_JUDGE", "0") == "1":
        entries.append(OPTIONAL_API_JUDGE)
    for entry in entries:
        provider, sep, model = entry.partition(":")
        if not sep or not model:
            raise ValueError(
                f"Invalid JUDGE_ENSEMBLE entry {entry!r}: expected 'provider:model'"
            )
        provider = provider.strip().lower()
        if provider not in ("ollama", "nvidia", "huggingface"):
            raise ValueError(f"Unknown judge provider {provider!r} in {entry!r}")
        spec = (provider, model.strip())
        if spec not in specs:
            specs.append(spec)
    return specs


JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Statistics, latency and polarity benchmark parameters.
# ---------------------------------------------------------------------------
BOOTSTRAP_RESAMPLES = int(os.getenv("BOOTSTRAP_RESAMPLES", "10000"))
CONFIDENCE_LEVEL = 0.95
LATENCY_TIMED_RUNS = int(os.getenv("LATENCY_TIMED_RUNS", "3"))
POLARITY_TOP_K = int(os.getenv("POLARITY_TOP_K", "5"))
HYBRID_RRF_K = 60  # reciprocal-rank-fusion constant (Cormack et al., 2009)

