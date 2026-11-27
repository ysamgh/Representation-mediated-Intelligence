# Technical Foundations

## 1. Problem formulation

Let donor model A produce an activation `x_A \in \mathbb{R}^{d_A}) at layer `l_A` and token position `t`. Let recipient model B expect an activation `x_B \in \mathbb{R}^{d_B}` at layer `l_B`.

We seek a map `T_{A\to B}:\mathbb{R}^{d_A}\to\mathbb{R}^{d_B}` such that injecting `T(x_A)` into B's forward computation preserves or induces a measurable task-relevant behavior.

The target is not merely low reconstruction error. The primary criterion is downstream functional utility under controlled intervention.

## 2. Centered orthogonal Procrustes with scale

Given paired calibration activations `X_A\in\mathbb{R}^{n\times d_A}` and `X_B\in\mathbb{R}^{n\times d_B}`, compute column means and center:

`\bar x_A=\frac1n\sum_i x_A^{(i)}, \quad \bar x_B=\frac1n\sum_i x_B^{(i)}`.

Using row-vector convention, solve

`R^*=\arg\min_{R^TR=I}\|\tilde X_A R-\tilde X_B\|_F^2`.

If `\tilde X_A^T\tilde X_B=U\Sigma V^T`, then `R^*=UV^T` (with the appropriate rectangular/semi-orthogonal convention). The scaled map is

`T(x_A) = (x_A-\bar x_A)R^*s+\bar x_B`.

The scale convention must be declared and tested. A simple global spread-matching scale is

`s=\|\tilde X_B\|_F/\|\tilde X_A\|_F`.

Implementations must state whether activations are stored as row or column vectors and use tests to prevent transposition errors.

## 3. Baselines

Minimum baseline suite:

- Identity / no intervention.
- Recipient self-swap (upper-bound-style reference, not a perfect ceiling).
- Random orthogonal map.
- Shuffled donor-to-recipient correspondence.
- Orthogonal Procrustes without scale.
- Procrustes with global scale.
- Ridge regression, with and without scale.
- CCA or GCCA for retrieval-oriented comparisons.
- Wasserstein Procrustes when paired calibration facts are unavailable.
- Low-rank/PCA-constrained maps.

Do not compare methods with different layer-selection privileges or different data budgets without explicitly reporting the difference.

## 4. Three levels of evaluation

### A. Geometry

- held-out paired alignment error;
- principal angles / subspace overlap;
- CKA, SVCCA, RSA;
- mutual-nearest-neighbor retrieval;
- singular-value spectrum and effective rank.

### B. Information and retrieval

- top-1 / top-k candidate retrieval;
- relation-conditioned factual accuracy;
- calibration and confidence;
- robustness to paraphrase and prompt templates;
- transfer accuracy relative to self-swap.

### C. Causal functional transfer

- activation injection versus no-injection;
- correct donor mapping versus shuffled/random controls;
- answer flip rate toward donor target;
- negative log-likelihood change for target and competing answers;
- paired bootstrap confidence intervals and suitable significance tests;
- per-model-pair and per-relation breakdowns.

A high geometry score does not imply causal transfer. A successful causal transfer does not alone prove informational equivalence.

## 5. Data split and leakage controls

- Split by subject/entity, not by individual prompt variant.
- Keep all paraphrases and relations for an entity in the same split where leakage is plausible.
- Fit the map only on training examples.
- Select layer pairs and hyperparameters on validation examples.
- Touch the test split once for final evaluation.
- Report facts known by both models separately from donor-only and recipient-miss cases.
- Record exact prompts, answer aliases, tokenizer behavior and token positions.

## 6. Dark-memory operational definition

A candidate is a recipient miss where donor A answers correctly and recipient B does not under a predefined direct-prompt protocol.

A candidate qualifies as a recoverable/latent-knowledge case only if it passes preregistered secondary checks such as:
1. correct answer after a fixed number of in-context demonstrations;
2. correct answer under held-out paraphrase templates;
3. correct answer ranked first among a fixed candidate set.

Report each criterion separately and report overlap. Do not collapse these into a claim that weights literally contain a discrete stored memory.

## 7. Reproducibility contract

Every experiment must save:
- experiment ID and timestamp;
- model IDs, immutable revisions, tokenizer IDs and software versions;
- layer indices, token-position rule and activation normalization;
- dataset version, license, hashes and split manifest;
- map family, rank, scale convention and fitted parameters;
- random seeds and deterministic settings;
- hardware, runtime, peak memory and cost;
- metrics, confidence intervals, raw predictions and control outcomes.

No result is accepted without a configuration file, logs, and a reproducible command.

## 8. Security and privacy

Activations are data. They may encode sensitive training or prompt information. Treat activation dumps and fitted maps as potentially sensitive artifacts:
- do not upload private prompts or user data to public storage;
- keep model and dataset licenses in scope;
- avoid assuming a representation is anonymized;
- test malicious or out-of-distribution activation injection;
- validate tensor shapes, dtypes, finite values and norm bounds before injection;
- isolate experimental model execution from external tools and credentials.
