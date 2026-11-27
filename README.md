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

## Initial implementation

The core package provides a NumPy-only Procrustes engine. Optional Hugging Face support extracts hidden-state activations from open models without changing the core dependency footprint.

### Install

~~~bash
python -m pip install -e ".[test]"
python -m pytest
~~~

Hugging Face support:

~~~bash
python -m pip install -e ".[huggingface]"
~~~

### Paired activation alignment

Rows in `source_acts` and `target_acts` must represent corresponding inputs in the same order. Never fit on held-out evaluation examples.

~~~python
from rmi import extract_hf_activations, fit_procrustes

texts = ["A calibration sentence.", "A second calibration sentence."]
source_acts = extract_hf_activations("open-model-a", texts, layer=-1, pooling="mean")
target_acts = extract_hf_activations("open-model-b", texts, layer=-1, pooling="mean")

mapping = fit_procrustes(source_acts, target_acts)
mapped = mapping.transform(source_acts)
~~~

This snippet illustrates the API only. It does not establish that the two models' representations are aligned or functionally interchangeable. A valid experiment requires a sufficiently large calibration corpus, fixed model revisions, held-out prompts, and random/shuffled baselines.

### Extraction conventions

- `layer=-1` selects the final hidden-state output; `layer=0` selects embeddings.
- `pooling="mean"` averages non-padding token vectors.
- `pooling="cls"` uses the first token and requires it to be unmasked.
- `pooling="last_token"` selects the final non-padding token, including with left padding.
- Custom Hugging Face remote code is disabled (`trust_remote_code=False`).
- Extraction requires model weights and may need substantial RAM/VRAM. CPU fallback is supported but can be slow.

## Research principles

- Separate representational similarity, information preservation and causal functional transfer.
- Fit transformations only on training data; choose layers and hyperparameters on validation data; report final results on held-out test data.
- Include random, shuffled, self-swap and no-intervention controls where applicable.
- Report per-model-pair results, not only pooled averages.
- Preserve exact model revisions, tokenizer versions, prompts, datasets, seeds and software environment.
- Do not describe a memory as “stored” solely because a model answers under one prompt and fails under another; use explicit operational criteria.
- “No retraining” does not mean “no calibration data” or “no compute”.

## Documentation

- [Literature review](research/literature-review.md)
- [Technical foundations](research/technical-foundations.md)
- [Research roadmap](research/roadmap.md)
- [Experiment protocol EXP-001](research/experiment-protocol.md)
- [Source registry](references/README.md)

## Status

**Phase:** Alignment engine and first activation-extraction adapter  
**Default branch:** `main`  
**Working branch:** `research/foundation-v0.1`

No empirical result is claimed until a run is reproducible from its recorded configuration and artifacts.

## License

MIT. Third-party papers, datasets, model weights and code remain under their own licenses and terms.
