# Representation-mediated Intelligence

**An open research and experimental engineering lab for communication, transfer, and interoperability through neural representations.**

This project investigates whether independently trained neural networks can exchange useful internal representations—not only text—and under what mathematical, architectural, and empirical conditions such exchange is reliable.

## Research thesis

Neural networks trained independently may develop partially aligned representational geometries. If a transformation maps a donor model's activation into a recipient model's representation space, the recipient may use its own downstream computation and stored parameters to interpret that activation.

This is a research hypothesis, not an assumption of universal shared semantics.

## Scope

1. **Representation alignment:** Procrustes, CCA/GCCA, SVCCA, CKA, RSA, mutual-nearest-neighbor and Wasserstein-based methods.
2. **Functional transfer:** activation patching, model stitching, factual key transfer and causal interventions.
3. **Multi-model spaces:** pairwise maps, generalized Procrustes, shared latent universes, composition and cycle consistency.
4. **Latent communication:** protocols for transferring activations with measurable utility, bandwidth, latency and safety properties.
5. **Reproducible evaluation:** controlled baselines, held-out splits, confidence intervals, ablations and artifact provenance.

## What this project is not

- Not a claim that all models share one universal semantic space.
- Not a production agent communication protocol yet.
- Not evidence that ordinary hosted model APIs expose compatible hidden states.
- Not a claim that geometric similarity alone proves semantic equivalence.

## Research anchors

- Cheng et al., **Plato's Key** (Hazy Research, Stanford, 25 Nov 2026): cross-model factual representation transfer using an orthogonal Procrustes rotation and global scale. The report describes 12 base LLMs from eight families. Reported results are summarized in [the literature review](research/literature-review.md).
- Achara et al., **Multi-Way Representation Alignment** (ICML 2026): generalized Procrustes shared spaces and Geometry-Corrected Procrustes Alignment; highlights that retrieval agreement and geometry preservation are different objectives.
- Smith et al., **Functional Alignment Can Mislead** (ICML 2025): warns that successful stitching does not, by itself, establish informational similarity.
- Maiorca et al., **Latent Space Translation via Semantic Alignment** (NeurIPS 2023): earlier closed-form latent translation across pretrained networks.

## Research principles

- Separate representational similarity, information preservation and causal functional transfer.
- Fit transformations only on training data; choose layers and hyperparameters on validation data; report final results on held-out test data.
- Include random, shuffled, self-swap and no-intervention controls where applicable.
- Report per-model-pair results, not only pooled averages.
- Preserve exact model revisions, tokenizer versions, prompts, datasets, seeds and software environment.
- Do not describe a memory as “stored” solely because a model answers under one prompt and fails under another; use explicit operational criteria.
- “No retraining” does not mean “no calibration data” or “no compute”.

## Initial implementation target

A reproducible Python package that extracts selected hidden states from open-weight models, fits a centered orthogonal Procrustes map plus a documented scale, injects mapped activations at controlled layers, evaluates factual transfer and negative controls, and exports machine-readable results.

## Documentation

- [Literature review](research/literature-review.md)
- [Technical foundations](research/technical-foundations.md)
- [Research roadmap](research/roadmap.md)
- [Experiment protocol EXP-001](research/experiment-protocol.md)
- [Source registry](references/README.md)

## Status

**Phase:** Research foundation / pre-implementation  
**Default branch:** `main`  
**Working branch:** `research/foundation-v0.1`

No empirical result is claimed by this repository until a run is reproducible from its recorded configuration and artifacts.

## License

MIT. Third-party papers, datasets, model weights and code remain under their own licenses and terms.
