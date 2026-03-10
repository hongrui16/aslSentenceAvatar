# Model Comparison: SignAvatar vs. Diffusion Variants

## Metric Definitions

| Metric | Description | Better |
|--------|-------------|--------|
| **Model-based FID** | Fréchet Inception Distance computed in a learned motion feature space. Measures distributional similarity between generated and ground-truth motions. | Lower ↓ |
| **Model-based Accuracy** | Classification accuracy of generated motions evaluated by a pretrained gloss classifier. Measures whether generated motions are recognizable as their intended sign. | Higher ↑ |
| **KNN Accuracy** | K-nearest-neighbor accuracy in feature space. Measures whether generated samples are close to ground-truth samples of the same class. Higher values indicate better class discriminability. | Higher ↑ |
| **Diversity** | Average pairwise distance among generated samples. Should be close to GT — too low means mode collapse, too high means noisy/uncontrolled generation. | Close to GT |
| **Multimodality** | Average pairwise distance among samples generated for the *same* class. Should be close to GT — too low means lack of variation per sign, too high means inconsistency. | Close to GT |
| **Variance Ratio** | Ratio of generated motion variance to ground-truth variance per body part. Ideal = 1.0. Values < 1 indicate under-articulation; values > 1 indicate over-articulation. | Close to 1.0 |

---

## Main Results (vs. Baseline)

| Metric | SignAvatar (ep5000) | Diff. Gloss-only CLIP (ep407) |
|--------|--------------------:|------------------------------:|
| **Model-based FID ↓** | 62.23 | **62.44** |
| **Model-based Accuracy ↑** | 0.758 | **0.838** ✓ |
| **KNN Accuracy ↑** | 0.271 | **0.378** ✓ |
| **Diversity** (GT=27.57) | 25.63 | **27.44** ≈ GT |
| **Multimodality** (GT=11.68) | 10.06 | 9.33 |

---

## Text Encoder × Conditioning Ablation (2×2)

| Encoder | Conditioning | FID ↓ | Accuracy ↑ | KNN Acc ↑ | Diversity (GT=27.57) | Multimodality (GT=11.68) |
|---------|-------------|------:|-----------:|----------:|---------------------:|-------------------------:|
| CLIP | Gloss-only | **62.44** | **0.838** | **0.378** | 27.44 | 9.33 |
| CLIP | Gloss+Attri | 127.95 | 0.201 | 0.097 | 26.61 | 12.67 |
| T5 | Gloss-only | 66.70 | 0.496 | 0.304 | 27.22 | 12.18 |
| T5 | Gloss+Attri | 63.66 | 0.525 | 0.310 | 27.31 | 12.23 |

**Key observations from the 2×2 ablation:**

- **CLIP excels at single-word gloss encoding but catastrophically fails on structured attributes.** Adding attributes causes accuracy to collapse from 0.838 → 0.201 (−76%). CLIP's visual-semantic pretraining is well-suited for common English words ("BOOK", "HOUSE") but maps similar attribute strings to nearby embeddings, destroying fine-grained categorical distinctions.
- **T5 is weaker for gloss-only but handles attributes gracefully.** T5 gloss-only underperforms CLIP (0.496 vs. 0.838) — single words are not T5's strength. But adding attributes slightly *improves* T5 (0.496 → 0.525), confirming T5 can extract useful signal from structured phonological descriptions.
- **No single text encoder solves both problems simultaneously.** CLIP wins on gloss identity; T5 wins on attribute integration. This motivates structured tensor conditioning: use an embedding lookup for gloss identity (no text encoder needed) and a separate structured pathway for phonological attributes, avoiding the bottleneck entirely.

---

## Variance Ratio (ideal = 1.0)

| Body Part | SignAvatar | CLIP Gloss-only | CLIP Gloss+Attri | T5 Gloss-only | T5 Gloss+Attri |
|-----------|----------:|----------------:|-----------------:|--------------:|---------------:|
| Arms      | 0.620     | 1.289           | **0.750**        | 1.337         | 1.286          |
| L-Hand    | 0.495     | 1.470           | **0.837**        | 1.401         | 1.346          |
| R-Hand    | 0.622     | 1.348           | **0.835**        | 1.262         | 1.283          |
| Torso     | 0.675     | **0.756**       | 0.579            | 0.774         | 0.712          |

**Variance Ratio observations:**

- SignAvatar consistently under-articulates across all body parts (ratios 0.49–0.68), especially hands.
- CLIP Gloss+Attri achieves the best hand/arm ratios (0.75–0.84), closest to ideal 1.0. Phonological attributes successfully modulate motion amplitude at the articulatory level — but at the cost of gloss identity.
- All other diffusion variants show mild over-articulation in hands/arms (1.26–1.47), suggesting this is a characteristic of the diffusion baseline rather than a conditioning effect.

---

## Key Findings

### 1. Gloss-only Diffusion Already Outperforms SignAvatar

The diffusion baseline with CLIP gloss conditioning achieves superior gloss discriminability: model-based accuracy of 0.838 vs. SignAvatar's 0.758, and KNN accuracy of 0.378 vs. 0.271. Model-based FID is on par (62.44 vs. 62.23). Diversity closely matches ground truth (27.44 vs. GT 27.57), indicating well-calibrated generation without mode collapse. This establishes a strong baseline before adding phonological conditioning.

### 2. Attribute Conditioning via Text Encoding Degrades Gloss Discriminability

Across both text encoders, adding phonological attributes as concatenated text fails to match the gloss-only CLIP baseline. With CLIP, the degradation is catastrophic (accuracy 0.201); with T5, it is more moderate but still substantial (accuracy 0.525 vs. gloss-only CLIP's 0.838). The condition is funneled through a single 512-d token, forcing gloss identity and phonological attributes to compete for the same representational bottleneck.

### 3. Paradox: CLIP Attribute Conditioning Produces Better Per-Body-Part Dynamics

Despite poor gloss-level metrics, the CLIP attribute-conditioned model achieves the best variance ratios for hands and arms (0.75–0.84 vs. gloss-only's 1.29–1.47 and SignAvatar's 0.49–0.62). This suggests phonological attributes successfully modulate motion amplitude at the articulatory level, but at the cost of gloss identity. This paradox also suggests that the CLIP model's "better" variance ratios may partly reflect mode collapse (reduced variation) rather than purely improved articulation control.

### 4. No Text Encoder Can Simultaneously Serve Both Roles

The 2×2 ablation reveals a fundamental tension: CLIP is optimal for encoding gloss identity (single common English words) but collapses structured attribute strings; T5 better preserves attribute distinctions but is weaker on single-word gloss encoding. No text encoder excels at both tasks, because they require fundamentally different representational properties — gloss identity needs a unique embedding per class, while attributes need compositional, dimension-independent encoding.

### 5. Root Causes of Attribute Conditioning Failure

**Noisy attribute–video alignment.** ASL-LEX 2.0 provides phonological attributes for one canonical citation form of each sign, but WLASL contains multiple signing variants per gloss (different handshapes, locations, or movement patterns performed by different signers). Only a subset of videos actually match the ASL-LEX attributes. This creates a label noise problem: the model receives the same attribute conditioning for videos with genuinely different phonological realizations, directly undermining the conditioning signal.

**Shared attributes collapse gloss identity.** Phonological attributes are shared across many glosses. When encoded as text, the model may over-rely on attribute signals while discarding gloss-specific information, causing different glosses with similar attribute profiles to produce nearly identical motions.

**Single-token conditioning bottleneck.** Both gloss identity and phonological attributes are compressed into a single 512-d condition token. This forces two qualitatively different types of information to compete for the same representational capacity.

---

## Potential Directions

- **Structured tensor conditioning:** Bypass text encoders entirely — use embedding lookup for gloss identity and a separate structured tensor for phonological attributes, each injected through independent pathways (e.g., gloss via token prepending, attributes via FiLM or cross-attention). This avoids the single-token bottleneck and the text encoder mismatch.
- **Filter or re-annotate WLASL videos** to identify the subset that matches ASL-LEX 2.0 canonical forms, or use soft/probabilistic attribute labels to account for variant diversity.
- **Add a gloss classification auxiliary loss** to enforce discriminability in the latent space when attribute conditioning is active.
- **Dual-pathway conditioning architecture:** Separate pathways for gloss and attributes, with gloss as the primary condition and attributes as auxiliary modulation.