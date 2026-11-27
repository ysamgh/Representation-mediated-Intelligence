# Literature Review: Representation-Mediated Intelligence

**Review date:** 2026-11-28  
**Purpose:** Establish the research baseline, distinguish evidence classes, and identify testable gaps.

## Executive synthesis

The field contains several related but non-equivalent problems:

1. **Representational similarity:** do two models arrange corresponding inputs similarly?
2. **Geometric alignment:** can a transformation map one activation space onto another?
3. **Functional interchangeability:** can the recipient continue computation successfully after receiving a mapped donor activation?
4. **Informational equivalence:** does the mapped state preserve the same task-relevant information?
5. **Causal transfer:** does an intervention on the recipient's activation cause the intended output change?
6. **Communication utility:** is the transfer more useful, cheaper, faster or more bandwidth-efficient than text/API exchange?

Evidence for one level does not automatically establish the next. The project must evaluate them separately.

## 1. Plato's Key — direct LLM-to-LLM factual transfer

**Cheng, Liu, Yau & Ré, Hazy Research, Stanford (25 Nov 2026).**  
Primary source: https://hazyresearch.stanford.edu/blog/2026-11-25-platos-key

### Method

The authors treat factual recall as a key–value-like process: an early-to-middle-layer hidden state at the subject token acts as a key, while later computation retrieves an answer from the recipient's own weights.

For donor A and recipient B, centered activation matrices are aligned with orthogonal Procrustes. If

`\tilde Z_A=Z_A-\mu_A\mathbf{1}^T` and `\tilde Z_B=Z_B-\mu_B\mathbf{1}^T`,

compute

`\tilde Z_B\tilde Z_A^T=U\Sigma V^T, \quad R=UV^T`.

The mapped activation is

`z_B=\mu_B+sR(z_A-\mu_A)`,

where the authors choose a global scale to approximately match the recipient's overall spread. Their convention includes reflections and semi-orthogonal maps when dimensions differ.

The transformation is fit in closed form using fewer than 5,000 shared facts recalled correctly by both models. The blog reports 8,800 facts, 16 relations, 12 base models from 8 families, 20 model pairs, and a subject-level train/validation/test split.

### Reported findings

- Median cross-model transfer accuracy: 41%; self-swap reference: 55%.
- Median chained A→B→C accuracy retention relative to direct A→C: 99% across 47 chains.
- 256 principal components retain median 74% of full-map transfer accuracy; 16 PCs retain 4%.
- Of 2,600 donor-correct / recipient-incorrect misses, 1,463 qualify as “dark memories” under at least one of three operational checks; mapped donor keys recover 267 (18%). Controls recover no more than 1.1%.

### Interpretation and limitations

This is functional evidence that some factual representations are partly interchangeable under a specific intervention. It is not proof of a universal semantic coordinate system, general reasoning transfer, multimodal transfer, or direct communication through ordinary APIs.

Layer choice is critical: selecting layers by mutual nearest-neighbor similarity produced 4.5% median transfer accuracy on the validation set, versus 44% when layers were selected by transfer accuracy. This is a major methodological warning: representational similarity metrics are not substitutes for task-level causal evaluation.

The “dark memory” label is operational. The recipient's misses are checked using few-shot examples, paraphrased prompts, and candidate ranking. A miss passing one check is not a direct measurement of a literal memory object.

## 2. Multi-Way Representation Alignment — shared spaces for 3+ models

**Achara et al., ICML 2026, PMLR 306, pp. 310–331.**  
Paper: https://proceedings.mlr.press/v306/achara26a.html  
Code: https://github.com/acharaakshit/multiway-alignment  
arXiv: https://arxiv.org/abs/2602.06205

The work addresses the quadratic growth and inconsistency of independently learned pairwise maps. It studies Generalized Procrustes Analysis (GPA) for a shared orthogonal universe, GCCA as an agreement-maximizing baseline, and Geometry-Corrected Procrustes Alignment (GCPA).

Important distinction: preserving within-model geometry and maximizing cross-model retrieval agreement are different objectives. Strict isometry can be suboptimal for retrieval; CCA-like methods can improve agreement while sacrificing geometric structure. This motivates multi-objective evaluation rather than a single “alignment score”.

## 3. Functional Alignment Can Mislead — a necessary caution

**Smith, Mannering & Marcu, ICML 2025, PMLR 267.**  
Paper: https://proceedings.mlr.press/v267/smith25a.html

The authors show that models can be functionally stitched despite representing very different information, including settings involving different modalities and clustered random noise. Therefore, successful downstream behavior after stitching is not sufficient evidence that two models encode the same information.

For this project, every functional-transfer claim must be paired with controls and information-sensitive analyses.

## 4. Latent Space Translation via Semantic Alignment

**Maiorca et al., NeurIPS 2023.**  
Paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/ad5fa03c906ca15905144ca3fbf2a768-Abstract.html  
DOI: https://doi.org/10.52202/075280-2418

Shows that latent spaces of pretrained networks can be translated with closed-form algebraic procedures, enabling encoder/decoder stitching across architectures and modalities. This provides earlier methodological grounding for closed-form latent-space translation, although the tasks and architectures differ from modern LLM factual key transfer.

## 5. Related concepts and methods to track

| Topic | Why it matters | Role in this project |
|---|---|---|
| Platonic Representation Hypothesis | Hypothesis of convergent representational geometry | Theory to test, not an axiom |
| Orthogonal Procrustes | Closed-form geometry-preserving alignment | First baseline |
| Wasserstein Procrustes | Distribution alignment without paired examples | Important baseline |
| CCA / GCCA | Agreement-maximizing cross-view alignment | Retrieval baseline |
| Generalized Procrustes (GPA) | Joint alignment for multiple models | Shared-space baseline |
| GCPA | Geometry-corrected multi-way alignment | Candidate multi-model method |
| CKA / SVCCA / RSA | Similarity diagnostics | Descriptive, not functional proof |
| Mutual nearest neighbors | Local correspondence metric | Layer diagnostic; known mismatch risk |
| Activation patching / causal tracing | Tests causal effect of internal interventions | Core evaluation |
| Model stitching | Tests compatibility of network components | Related but not equivalent |

## 6. Open research questions

1. How stable is the mapping across random seeds, prompts, tokenizers, model revisions and datasets?
2. Does the method transfer beyond single-hop factual recall to multi-hop reasoning, instruction following, code, and multimodal tasks?
3. How much does success depend on architecture family, normalization, hidden dimension, tokenizer and layer selection?
4. Does a shared multi-model latent universe outperform pairwise maps on held-out model pairs?
5. What information is preserved or lost under compression, quantization and noisy transmission?
6. Can recipient-side calibration be reduced without sacrificing transfer?
7. What are the privacy and security implications of activation exchange, including unintended information leakage and adversarial activations?
8. Can transfer be made robust to post-training changes such as instruction tuning and preference optimization?

## Evidence policy

The Stanford Plato's Key source is a research-group technical report/blog in this review; cite it as such unless a peer-reviewed paper or preprint is separately identified. Numerical findings above are attributed to that report and must not be presented as independent replications. Re-check source versions before publication.
