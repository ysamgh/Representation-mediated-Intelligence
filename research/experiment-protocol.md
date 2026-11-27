# Experiment Protocol EXP-001: Synthetic Procrustes Recovery

**Status:** Planned  
**Purpose:** Verify the numerical implementation before any LLM activation experiment.

## Hypothesis

Given paired samples related by a known orthogonal transform and a global scale, the implementation recovers the transform and maps held-out samples with low error, subject to rank and sample-size constraints.

## Data generation

1. Sample a deterministic matrix `X_A\in\mathbb{R}^{n\times d}`.
2. Generate an orthogonal matrix `Q` using QR decomposition of a seeded Gaussian matrix.
3. Construct `X_B=sX_AQ+b+\epsilon`, where `s>0`, `b` is a fixed offset and `\epsilon` is controlled noise.
4. Split rows into fit and held-out sets before fitting.
5. Repeat for multiple seeds, dimensions, sample counts and noise levels.

## Conditions

- exact orthogonal transform, zero noise;
- non-unit global scale;
- non-zero mean offset;
- additive Gaussian noise;
- rank-deficient calibration data;
- insufficient calibration samples;
- unequal source and target dimensions (separate semi-orthogonal tests).

## Metrics

- held-out normalized Frobenius error;
- orthogonality residual `\|Q^TQ-I\|_F` where square;
- recovered scale error;
- recovered offset error;
- runtime and numerical warnings.

## Controls

- identity map;
- random orthogonal map;
- shuffled row correspondence;
- no centering;
- no scale correction.

## Acceptance criteria

- exact noiseless square full-rank cases recover to numerical tolerance;
- held-out error degrades monotonically as noise increases in aggregate;
- incorrect controls are measurably worse in non-degenerate cases;
- rank-deficient and underdetermined cases produce explicit diagnostics rather than silent success;
- test suite is deterministic under fixed seeds.

Numerical thresholds will be fixed in code tests after selecting dtype and dimension ranges. They must not be tuned against any future LLM test set.

## EXP-002 (planned): Single-pair factual activation transfer

Prerequisites: EXP-001 passes; two open-weight models and a licensed factual dataset are selected.

Primary endpoint: held-out recipient answer-target success after mapped donor activation injection.

Controls: no intervention, recipient self-swap, shuffled donor correspondence, random orthogonal map, and unmapped donor activation where dimensions permit.

Split policy: entity-level train/validation/test. Fit maps on train only; choose layer pairs on validation only; report test once.

Report: absolute accuracy, paired difference versus controls, bootstrap confidence intervals, relation and model-pair breakdowns, failure examples and compute cost.

A result is positive only if the preregistered primary endpoint exceeds the relevant controls with uncertainty quantified. Otherwise report null/inconclusive findings without changing the protocol post hoc.
