# Research Roadmap

## Mission

Determine when and how neural representations can be transferred between independently trained models to produce reliable, measurable downstream utility.

## Research gates

| Gate | Question | Exit criterion |
|---|---|---|
| G0 — Literature | What is already known? | Source-backed review and explicit claims ledger |
| G1 — Reproduction | Can we reproduce a minimal Procrustes transfer? | Deterministic toy test + one open-weight model pair |
| G2 — Validity | Is transfer causal and robust? | Held-out facts, controls, CIs, layer ablations |
| G3 — Generalization | Does it work across model families/tasks? | Multi-pair evaluation and per-pair reporting |
| G4 — Multi-way | Can a shared space serve 3+ models? | Compare pairwise, GPA, GCCA and GCPA |
| G5 — Communication | Is latent exchange useful operationally? | Utility/bandwidth/latency/security comparison with text |
| G6 — Research release | Can others audit and reproduce it? | Versioned data, code, configs, artifacts and limitations |

## Phase 0 — Research foundation (current)

Deliverables:
- literature review;
- mathematical definitions and conventions;
- claim/evidence separation;
- reproducibility and safety policy;
- experiment plan.

Exit: methods and evaluation protocol are unambiguous.

## Phase 1 — Minimal mathematical implementation

Build:
- activation dataset abstraction;
- centering and scale utilities;
- orthogonal Procrustes via SVD;
- rectangular dimension handling;
- serialization for fitted transforms;
- unit tests on synthetic rotations, noise, rank deficiency and unequal dimensions.

No large model download is needed for this phase.

Exit:
- synthetic recovery tests pass;
- numerical stability and shape contracts documented;
- no silent leakage between fit and evaluation data.

## Phase 2 — Single-pair LLM reproduction

Choose two permissively licensed, locally runnable open-weight decoder-only models with compatible intervention points and manageable resource requirements.

Pipeline:
1. select a factual-recall dataset and license;
2. construct entity-level train/validation/test splits;
3. extract activations from donor and recipient;
4. fit the map on train;
5. choose layers on validation;
6. inject mapped donor activation into recipient;
7. evaluate test and negative controls.

Exit: effect exceeds controls with uncertainty reported; otherwise document a negative or inconclusive result.

## Phase 3 — Robustness and ablation

Vary:
- prompt templates and paraphrases;
- entity/relation types;
- model pair and family;
- layer pair;
- calibration set size;
- activation precision and noise;
- PCA dimension / rank;
- scale and normalization;
- tokenizer and token-position alignment.

Exit: characterize where method succeeds and fails, not merely maximize a headline score.

## Phase 4 — Multi-model latent universe

Compare:
- independent pairwise Procrustes;
- Generalized Procrustes Analysis;
- GCCA;
- GCPA;
- direct versus chained maps;
- cycle consistency and drift.

Exit: determine whether a shared coordinate system improves any-to-any transfer at practical cost.

## Phase 5 — Latent communication prototype

Only after causal transfer is established, prototype a controlled message envelope:
- sender/receiver model and revision identifiers;
- source/target layer and tensor shape;
- transform ID and version;
- dtype, quantization, checksum and norm metadata;
- payload plus provenance;
- compatibility checks and rejection behavior.

Benchmark against text/JSON exchange on task quality, bytes, latency, compute, failure recovery and security.

## Initial priority

The highest-value next step is a small, dependency-light Procrustes library with synthetic tests. It validates the mathematical core before committing to large model downloads or GPU spend.

## Non-goals for early phases

- no production deployment;
- no arbitrary black-box API activation exchange;
- no claims of general intelligence transfer;
- no autonomous model-to-model trust;
- no weight modification or fine-tuning in the initial reproduction.
