# HCAD-RAG
**Gated Evidence Selection and Context Decoupling for Clinical RAG**

A novel Retrieval-Augmented Generation framework that represents clinical semantics through Entity–Relation–Modifier (E/R/M) channels. Relation and modifier embeddings are partially decorrelated from the entity embedding using soft modified Gram–Schmidt orthogonalization (blend factor 0.70); the resulting channels are not guaranteed to be mutually orthogonal. It applies graph-topology anisotropic transformations, adaptive statistical Heaviside filtering, and a calibrated JEV System One decision gate to address context distraction and hallucination risk in medical LLMs.

---
## Mechanism

HCAD-RAG answers a clinical query in five stages. Stages 1–4 select the
evidence; stage 5 is generation. Every arm of the benchmark (closed-book,
BM25, dense, hybrid, HCAD) uses the same generator, the same synthesis
instructions and the same **350-word evidence budget**, so arms differ
only in *which* evidence they hand the generator.

### 1. Query understanding
- **Biomedical NER** (SciSpacy BC5CDR + JNLPBA, plus regex rules) finds drugs,
  diseases, genes and proteins.
- **Entity linking**: SapBERT embeddings + a FAISS cosine index map each
  mention to a PrimeKG node.
- **Relation and modifier extraction** (rule-based) detects the clinical
  predicate (treats, contraindicated, interacts with, …) and modifiers,
  including negation cues ("not recommended", "contraindicated", …).
  Query relations are mapped onto PrimeKG's canonical relation vocabulary
  (`HCAD_RELATION_VOCAB=1`).

### 2. Candidate pool (≤ 80 candidates)
Three channels are pooled:
- **Dense passages**: nearest neighbours of the full query in the Chroma
  vector store (S-PubMedBERT, cosine): PMC clinical text and rendered PrimeKG facts.
- **Hyperedge triples**: bidirectional, relation-typed hyperedges around every
  linked entity in the PrimeKG graph (up to 12 triples per entity, ≤ 3 per
  hyperedge, members ordered by the `soft` inverse-log-degree rule so hub
  nodes do not crowd out specific facts).
- **N-ary facts**: the fact hypergraph groups triples that share a tail into
  multi-entity clinical facts (≤ 10 per query).

### 3. Semantic decoupling and hybrid scoring
- Each statement is represented as a 3×d matrix $X_{\text{HCAD}} = [E; R; M]$
  (entity, relation, modifier) instead of one flat vector. Soft modified
  Gram–Schmidt (blend $\eta = 0.70$) partially decorrelates R and M from E.
  The channels are not guaranteed to be mutually orthogonal. A
  query-independent anchor replaces a degenerate residual, and channels
  absent from a candidate are masked (`HCAD_PRESENCE_MASKING=1`).
- Candidates are ranked by the hybrid score
  $$\Phi = w_d \cdot \cos(q, c) + (1 - w_d) \cdot \Phi_{\text{struct}}, \qquad w_d = 0.5$$
  This fuses the full-text dense match with the presence-masked E/R/M channel score.
  The legacy topology-block weighting $W_A$ (node degree, edge confidence,
  IDF) is an ablation (`HCAD_ANISO_DIM_WEIGHTING=1`). A query-adaptive
  channel-weight adapter can be trained on the dev pool (`train-tensor`,
  `HCAD_TENSOR_MODE=learned`).

### 4. Gated evidence selection
- **Adaptive Heaviside filter (two gates)**: an absolute floor
  $\Phi \ge 0.10$, then a dynamic threshold
  $$\mathcal{M}_{\text{HCAD}}(t_i) = \Theta(\Phi(t_i) - (\mu + k\sigma))$$
  with an optional saturation cap $\tau_{sat}$ calibrated on the dev pool.
  If nothing survives, HCAD **abstains** and answers with exactly the
  closed-book arm's no-evidence prompt.
- **Dense anchor** *(this branch)*: the best full-text dense passage
  (cosine ≥ 0.40) is always admitted. It keeps its Φ position, but the gate
  reserves a slot and its words. Reason: the E/R/M channels compare short
  extracted fragments, so long prose passages, which have no entity or
  relation metadata, were out-ranked by short KG triples. The passage that
  best matches the whole question could be gated out entirely while dense
  RAG kept it. HCAD's evidence is now "best dense passage +
  structure-selected complement", so the comparison against dense RAG
  isolates what structured selection adds. `HCAD_DENSE_ANCHOR_K=0` restores
  the old behaviour.
- **JEV System One decision gate** (local-calibrated mode):
  - drops candidates below relevance 0.25 (anchors are exempt);
  - allows at most 3 items per relation type, so one relation cannot fill the context;
  - keeps at most 8 items and 350 words.
- **Hyperedge rendering** *(this branch)*: KG triples that share
  (head, relation) are emitted as ONE line, e.g. "Metformin is indicated for: A; B;
  C." (≤ 5 tails), instead of repeating head + verb per triple. This fits about
  3× more facts into the same word budget. `HCAD_GROUP_FACTS=0` restores one line per triple.
- **Clip, don't drop** *(this branch)*: an item that does not fit the
  remaining budget is clipped at a sentence boundary, as the dense arm
  does, when ≥ 25 words remain. Previously only the first item could be
  clipped, and a relevant passage behind a few triples was discarded whole.

### 5. Generation
A structured prompt lists the evidence as `[Evidence i]` blocks with source
and relevance and is sent to Llama-3.1-8B-Instruct (4-bit,
`llama3.1:8b-instruct-q4_K_M`). The instructions are the same shared
`synthesis_instructions` every arm gets.

```
Clinical Query
      │
      ▼
Biomedical NER (SciSpacy) ──► SapBERT Entity Linking (PrimeKG)
      │                                │
      ▼                                ▼
Relation & Modifier Extraction   Dense passages + Hyperedge triples + N-ary facts
      │                                │
      ▼                                ▼
Semantic Decoupling [E;R;M]      Candidate Pool (≤ 80)
      └───────────────┬────────────────┘
                      ▼
        Hybrid score Φ = w_d·dense + (1−w_d)·E/R/M
                      ▼
        Heaviside filter (floor + μ+kσ)  ──►  abstain if empty
                      ▼
        + Dense anchor (best full-text passage, reserved slot)
                      ▼
        JEV Decision Gate (relevance, relation cap, 8 items)
                      ▼
        Evidence assembly: hyperedge grouping, 350-word budget, clip-not-drop
                      ▼
        Llama-3.1-8B-Instruct (4-bit) ──► cited clinical answer
```

### Evaluation fairness (judge side)
- The judge sees each answer's own evidence under neutral `[Doc i]` tags.
  The arm-specific citation tags in the answers (`[Evidence i]` /
  `[Context i]`) are rewritten to `[Doc i]` before scoring, so the judge
  cannot tell which system wrote an answer. Citation precision is still
  computed on the raw answers.
- The closed-book arm and an abstaining HCAD share one no-evidence system
  prompt (`NO_CONTEXT_SYSTEM_PROMPT`).
- Pairwise comparisons run in both orders (AB + BA); a criterion keeps its
  winner only when both orders agree.
- Every evidence block shown to the judge is clipped at a sentence boundary
  within the same 350-word budget.

### Evidence diagnostics
Each checkpoint record stores, per retrieval arm, what was actually handed
to the generator. The fields are item count, prose vs KG items, KG facts,
words used and the rank of the gold source chunk. The results JSON
aggregates them in `evidence_diagnostics`
(`gold_in_evidence_rate`, `gold_at_rank1_rate`, `mean_words`, …). If HCAD
trails dense RAG, these numbers tell an evidence-selection loss apart from
a generation or judging effect.

> The evidence-assembly changes target the observed HCAD < dense-RAG gap.
> They are not yet validated by a benchmark run. In particular, the 0.40
> anchor floor is a prior, not a tuned value. Compare `evidence_diagnostics`
> and the rubric scores against a `HCAD_DENSE_ANCHOR_K=0 HCAD_GROUP_FACTS=0`
> run before claiming an improvement.
---
## Quick Start

### 1. Prerequisites & Environment Setup

```bash
git clone https://github.com/ajheshbasnet/HCAD.git
cd HCAD

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Install SciSpacy biomedical NER models
# (--no-deps: their metadata pins spacy<3.8 but they run fine under 3.8)
pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_jnlpba_md-0.5.4.tar.gz
```

Use the repository-root `.env` as the single local environment file — all LLMs run locally via Ollama with zero cloud LLM API keys:
- `LLM_PROVIDER="ollama"` + `JUDGE_MODEL_INDEX=<n>` (see *Judge selection* below). The generator (`llama3.1:8b-instruct-q4_K_M`) and the judge(s) run locally via Ollama, all 4-bit quantized. W&B stays online if `WANDB_API_KEY` is set.
```bash
LLM_PROVIDER="ollama"
JUDGE_MODEL_INDEX="0,1,2,3"                             # Local 4-bit Ollama judge panel
WANDB_API_KEY="your_rotated_wandb_api_key"              # Optional, enables experiment tracking
WANDB_PROJECT="hcad-rag"
# WANDB_ENTITY="your_wandb_username_or_team"            # Optional
```
The root `.env` is git-ignored and is loaded centrally by `src.config`; add
settings there rather than creating another environment file. The W&B API key
is loaded automatically for each evaluation, benchmark, hallucination
evaluation, and weight-learning run. Do not put API keys in source code or
commit them. Set `WANDB_MODE="disabled"` (or pass `--no-wandb`) to turn
tracking off.
### 2. Prepare Knowledge Corpus & Indices

```bash
# Build knowledge graph indices and vector database with clinical relations
python src/main.py prepare --rebuild
```
### 3. Run Benchmark (Base Paper Protocol)

Replicate the evaluation protocol from Hyper-RAG ([arXiv:2504.08758](https://arxiv.org/abs/2504.08758)):

```bash
# Multi-judge benchmark: closed-book Llama-3.1-8B, BM25, dense, hybrid, HCAD-RAG
python src/main.py base-paper --samples 50 --plot
# + component ablations (w/o Decoupling, w/o Soft Gram-Schmidt, w/o Decision Gate)
python src/main.py base-paper --samples 50 --ablations
# Indication vs contraindication paired-polarity retrieval benchmark (35 drug pairs)
python src/main.py polarity                 # controlled curated corpus, all arms
python src/main.py polarity --mode store    # full retrieval over the built KG / vector store
# MedHallu: classifier-only (detector) and HCAD-RAG generated answers
python src/main.py medhallu --system detector --n 200
python src/main.py medhallu --system hcad --n 200
# Every table + figure (incl. 5-stage methodology diagram) from the saved JSON,
# with a table-vs-JSON consistency check
python src/main.py figures
```
Arms (one label each, used verbatim in JSON, tables and figures):
**Llama-3.1-8B-Instruct (Zero-Retrieval / Closed-Book)**, BM25 RAG (top-5),
Dense RAG (S-PubMedBERT, top-5), Hybrid BM25+Dense RAG (RRF, top-5), HCAD-RAG.
Judges: selected locally via `JUDGE_MODEL_INDEX` (below), or an explicit
`JUDGE_ENSEMBLE` list of `ollama:<model>` tags. The default panel is
`ollama:gemma2:9b-instruct-q4_K_M,ollama:qwen2.5:14b-instruct-q4_K_M`.
Every call runs in JSON mode, and a judge failure never becomes a neutral
fallback score.
Outputs:
- **5-Dimension Scoring Assessment** (0–100 scale: Comprehensiveness, Diversity, Empowerment, Logical, Readability)
- **8-Criteria Selection Assessment** (Pairwise Win-Rate with order counterbalancing)
- **Statistics** (`src/evaluation/stats.py`): bootstrap 95% CIs per rubric dimension, paired bootstrap CI of score differences, two-sided Wilcoxon signed-rank, paired t-test, exact McNemar - for the judge consensus and for each judge
- **Inter-judge agreement**: Cohen's kappa per judge pair and Fleiss' kappa (pairwise winners, correctness, rubric band)
- **Latency breakdown** (`results/latency_breakdown.json`): retrieval-only latency (1 cold + N warm runs, mean ± std, per pipeline stage for HCAD-RAG) separated from generation latency (prefill/decode seconds, prompt/generated tokens, tokens/s)
- **Retrieval Quality** (`src/evaluation/metrics.py`): NDCG@5 / Recall@5 over the paper-protocol gold relevance labels — the source chunk that generated each question
- **Citation Precision**: fraction of `[Evidence i]` tags in generated answers that resolve to actually retrieved items
- **MedHallu hallucination evaluation** via `src/evaluation/hallucination_evaluator.py`
- **Evidence diagnostics** (`evidence_diagnostics`): what each arm handed the generator (see *Mechanism*)

Per-question results are checkpointed to `results/base_paper_checkpoint.jsonl`
and all aggregates are recomputed from the checkpoint, so interrupted runs
resume without losing or double-counting samples. Every record is tagged
with its judge panel and an HCAD evidence-config fingerprint
(`hcad_config_id`). A question counts as done, and is aggregated, only for
the same judges and the same HCAD settings, so switching the judge index or
an evidence toggle re-evaluates rather than mixing scores. W&B logs the
aggregate and per-question numeric metrics when `WANDB_API_KEY` is
configured.

### Judge selection and quantization
`src/judge_models.py` is the single registry of judge models. Set an index
and every judge call in `src/` follows, with no source edits:

| `JUDGE_MODEL_INDEX` | Judge | `q4_K_M` tag | Size |
|---|---|---|---|
| 0 | Qwen3-14B | `qwen3:14b-q4_K_M` | 9.3 GB |
| 1 | DeepSeek-R1-Distill-Qwen-14B | `deepseek-r1:14b-qwen-distill-q4_K_M` | 9.0 GB |
| 2 | Gemma 3 12B IT | `gemma3:12b-it-q4_K_M` | 8.2 GB |
| 3 | Qwen2.5-14B-Instruct | `qwen2.5:14b-instruct-q4_K_M` | 9.0 GB |
| 4 | Phi-4 14B | `phi4:14b-q4_K_M` | 9.1 GB |
| 5 | Gemma 2 9B IT | `gemma2:9b-instruct-q4_K_M` | 5.8 GB |

- `JUDGE_MODEL_INDEX=2` selects one judge; `JUDGE_MODEL_INDEX=0,3` selects a two-judge
  consensus panel, with Cohen's/Fleiss' kappa reported.
- Precedence: `JUDGE_ENSEMBLE` > `JUDGE_MODEL_INDEX` > default panel.
- `JUDGE_QUANT` (default `q4_K_M`) picks the quantization; some models also
  offer `q3_K_M`, `qat`, `q8_0` and `fp16`. An unknown index or quant fails
  immediately with the menu.
- **All LLMs are 4-bit**: the generator and the default panel use explicit
  `q4_K_M` tags rather than Ollama's implicit default tag.
- Memory: judge calls send `num_ctx=8192`. Reasoning models (qwen3,
  deepseek-r1) get `think=False`, so chain-of-thought does not consume the
  token budget or break the JSON.
- On Kaggle the notebook exposes **only** `JUDGE_MODEL_INDEX`. The quant,
  30 questions, KV-cache `q8_0` and evidence settings are fixed so team
  members' runs stay comparable.

### Error handling
- **Judge preflight** (before any generation): every judge must be pulled
  in ollama and answer a JSON test call. Otherwise the run stops with
  `ollama pull <tag>` or an out-of-memory hint. `JUDGE_SKIP_PREFLIGHT=1`
  skips this check.
- **Broken-judge abort**: per-judge failure counters (stored in the
  manifest as `session_failures`). After 2 questions, a judge failing more than
  `JUDGE_MAX_FAILURE_RATE` (0.5) of its evaluations aborts the run, because its
  verdicts would be missing rather than judgments. Checkpointed questions are kept.
- **Kaggle notebook**:
  - `run_checked` turns a SIGKILL into an explicit out-of-memory error,
    printing `free -h`, `nvidia-smi`, `ollama ps` and the ollama log;
  - pulls are retried;
  - every model is warmed up and must be `100% GPU`, since CPU offload is what
    exhausts host RAM next to the PrimeKG graph;
  - models are swapped one at a time when they do not all fit in VRAM;
  - the cell checks that `src/` resolves exactly the tags the notebook pulled.
Weight learning tunes only `filter_k`, the learned parameter currently consumed by
the retriever. Anisotropic and component-scoring weights remain at their configured
values; retrieval and model architecture are unchanged.
### Retrieval scoring knobs (ablations)
| Variable | Default | Effect |
|----------|---------|--------|
| `HCAD_DENSE_WEIGHT` | `0.5` | Weight of the dense full-text cosine in Φ; `0` = legacy structure-only score |
| `HCAD_ANISO_DIM_WEIGHTING` | `0` | `1` re-enables the legacy topology block weighting of embedding dimensions |
| `HCAD_PRESENCE_MASKING` | `1` | Query-conditioned channel-presence masking |
| `HCAD_RELATION_VOCAB` | `1` | PrimeKG relation labels / query relations mapped to canonical verb phrases |
| `HCAD_EXPANSION_RULE` | `soft` | Member order inside relation hyperedges: `desc`, `soft`, `asc_cap` |
| `HCAD_TAU_SAT` | unset | Gate-2 saturation ceiling, calibrated by `retrieval-eval` on the dev pool |
| `HCAD_DENSE_ANCHOR_K` | `1` | Best full-text passages always admitted; `0` = legacy structure-only selection |
| `HCAD_DENSE_ANCHOR_MIN` | `0.40` | Cosine floor a dense anchor must clear |
| `HCAD_GROUP_FACTS` | `1` | One evidence line per (head, relation) hyperedge; `0` = one line per triple |
`python src/main.py retrieval-eval --pool dev` reports paired Recall@5/NDCG@5
for the legacy baseline, each added change, and the full configuration.

All benchmark arms receive the
same answer-synthesis instructions (`synthesis_instructions` in
`src/core/prompt_builder.py`); they differ only in the evidence supplied.

---
## Results

The paper reports a pilot evaluation on 50 clinical questions, using one
evaluation session and a single LLM judge.

| System | Overall rubric score (0–100) | End-to-end latency |
|---------|------------------------------:|-------------------:|
| Llama-3.1-8B-Instruct (Zero-Retrieval / Closed-Book) | 70.6 | Not reported |
| Dense RAG (S-PubMedBERT, top-5) | 74.7 | 2.09 s |
| HCAD-RAG | 75.5 | 0.50 s |

HCAD-RAG's overall rubric score was 0.8 points higher than dense RAG
(75.5 versus 74.7). The paper describes this difference as too small to
establish a meaningful advantage. The reported end-to-end latency was
0.50 s for HCAD-RAG and 2.09 s for dense RAG. These pilot latencies mix
retrieval with generation; the revised harness reports them separately.

These figures are preliminary pilot results, not verified benchmark claims.
See the Limitations section below and the paper for the evaluation details.

---
## Limitations

- **Preliminary evaluation:** The reported results come from one pilot run of
  50 clinical questions and one LLM judge. Inter-judge agreement, human scoring,
  confidence intervals, and paired significance testing were not established.
- **Evaluation protocol:** Questions were generated from passages in the
  retrieval corpus and evaluated against those same sources. This may favor
  retrieval-based systems over the no-retrieval baseline.
- **No measured hallucination result:** The paper reports that the hallucination
  evaluation modes did not produce saved output files. Therefore, the paper
  does not establish that HCAD-RAG reduces medical hallucinations.
- **Biomedical scope:** The models, knowledge graph, relation vocabulary,
  encoder, and evaluation questions are biomedical. Performance in other
  domains has not been established.
- **No clinical validation:** HCAD-RAG has not been validated in a clinical
  setting and must not be used for diagnosis, treatment decisions, or other
  clinical purposes.

---
## Project Structure
```
HCAD/
├── requirements.txt                   # Project dependencies
├── README.md                          # Documentation
├── .gitignore                         # Git exclusion rules
├── src/
│   ├── config.py                      # Global configurations, model weights & paths
│   ├── judge_models.py                # Judge registry: index + quant -> exact 4-bit ollama tag
│   ├── main.py                        # CLI entry point
│   ├── wandb_tracking.py              # Shared W&B helpers (auto-enabled when configured)
│   ├── api/
│   │   └── server.py                   # FastAPI workbench API (`python src/main.py serve`)
│   ├── data/
│   │   ├── download_datasets.py       # Dataset fetchers (PrimeKG, MedHallu)
│   │   ├── build_knowledge_graph.py   # PrimeKG MultiDiGraph & index construction
│   │   └── build_vector_store.py      # ChromaDB clinical context ingestion
│   ├── core/
│   │   ├── ner_extractor.py           # SciSpacy + Regex Biomedical NER
│   │   ├── entity_linker.py           # SapBERT FAISS cosine entity resolution
│   │   ├── relation_extractor.py      # Rule-based clinical predicate extraction
│   │   ├── semantic_decoupler.py      # Innovation 1: Entity-Relation-Modifier decoupling
│   │   ├── anisotropic_transformer.py # Innovation 2: Graph topology W_A transformation
│   │   ├── hypergraph_builder.py      # Innovation 3: Bidirectional hyperedge expansion
│   │   ├── fact_hypergraph.py         # N-ary clinical fact hypergraph from PrimeKG shared-tail contexts
│   │   ├── statistical_filter.py      # Innovation 4: Adaptive Heaviside cutoff
│   │   ├── decision_gate.py           # JEV gate + evidence assembly (anchors, hyperedge lines, budget)
│   │   ├── dynamic_tensor.py          # (P3, opt-in) trainable anisotropic tensor
│   │   ├── entropy_gate.py            # (P3, opt-in) entropy-based abstention gate
│   │   ├── weight_learner.py          # filter_k weight learning
│   │   ├── prompt_builder.py          # Standardized 4-section clinical prompt builder
│   │   ├── llm_client.py              # Local Ollama LLM interface
│   │   └── hcad_retriever.py          # End-to-end integrated HCAD-RAG pipeline
│   ├── baselines/
│   │   ├── baseline_llm.py            # Parametric LLM baseline (zero-retrieval)
│   │   └── vanilla_vector_rag.py      # Standard dense Top-K vector baseline
│   └── evaluation/
│       ├── base_paper_evaluator.py    # Scoring & selection rubric evaluator (LLM-as-a-judge)
│       ├── run_base_paper_benchmark.py# Base paper benchmark runner (checkpointed, resume-safe)
│       ├── hallucination_evaluator.py # MedHallu hallucination evaluation
│       ├── metrics.py                 # NDCG@K, Recall@K, McNemar, citation precision
│       └── plots.py                   # Benchmark figures rendered from results JSON
└── results/
    ├── figures/                       # Benchmark charts
    └── tables/                        # Summary metrics & logs
```
---
## Datasets

| Dataset | Source | Purpose |
|---------|--------|---------|
| [PrimeKG](https://doi.org/10.7910/DVN/IXA7BM) | Harvard Dataverse | Precision medicine knowledge graph (129K nodes, 4M+ edges) |
| [MedHallu](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu) | HuggingFace | Medical hallucination detection dataset |
| PMC Open Access | NCBI / PMC | Curated open-access clinical text corpus |

---
## References

- **Hyper-RAG**: *Combating LLM Hallucinations using Hypergraph-Driven Retrieval-Augmented Generation* ([arXiv:2504.08758](https://arxiv.org/abs/2504.08758))
- **SIRG**: *Structured Information Retrieval Gating* ([arXiv:2601.03052](https://arxiv.org/abs/2601.03052))
- **PrimeKG**: Chandak et al., *Building a knowledge graph to enable precision medicine*, Scientific Data (2023)
- **SapBERT**: Liu et al., *Self-Alignment Pretraining for Biomedical Entity Representations*, NAACL (2021)

---
## License

Research and academic use only. Individual components are governed by their respective dataset and model licenses.
