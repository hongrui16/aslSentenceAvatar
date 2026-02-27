# Model Comparison: SignAvatar vs. Diffusion (Gloss-only) vs. Diffusion (Gloss + Attribute)

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

## Main Results

| Metric | SignAvatar (ep5000) | Diff. Gloss-only (ep407) | Diff. Gloss+Attri (ep457) |
|--------|--------------------:|-------------------------:|--------------------------:|
| **Model-based FID ↓** | 62.23 | **62.44** | 127.95 |
| **Model-based Accuracy ↑** | 0.758 | **0.838** ✓ | 0.201 ✗ |
| **KNN Accuracy ↑** | 0.271 | **0.378** ✓ | 0.097 ✗ |
| **Diversity** (GT=27.57) | 25.63 | **27.44** ≈ GT | 26.61 |
| **Multimodality** (GT=11.68) | 10.06 | 9.33 | **12.67** ≈ GT |

## Variance Ratio (ideal = 1.0)

| Body Part | SignAvatar | Diff. Gloss-only | Diff. Gloss+Attri |
|-----------|----------:|------------------:|-------------------:|
| Arms      | 0.620     | 1.289             | **0.750**          |
| L-Hand    | 0.495     | 1.470             | **0.837**          |
| R-Hand    | 0.622     | 1.348             | **0.835**          |
| Torso     | 0.675     | **0.756**         | 0.579              |

---

## Key Findings

### 1. Gloss-only Diffusion Already Outperforms SignAvatar

The diffusion baseline with gloss conditioning alone achieves superior gloss discriminability: model-based accuracy of 0.838 vs. SignAvatar's 0.758, and KNN accuracy of 0.378 vs. 0.271. Model-based FID is on par (62.44 vs. 62.23). Diversity closely matches ground truth (27.44 vs. GT 27.57), indicating well-calibrated generation without mode collapse. This establishes a strong baseline before adding phonological conditioning.

### 2. Attribute Conditioning Severely Degrades Gloss Discriminability

Adding phonological attribute conditioning causes model-based FID to double (127.95), accuracy to drop from 0.838 to 0.201, and KNN accuracy to fall from 0.378 to 0.097. The model has nearly lost the ability to distinguish between different glosses.

### 3. Paradox: Attribute Conditioning Produces Better Per-Body-Part Dynamics

Despite poor gloss-level metrics, the attribute-conditioned model achieves the best variance ratios for hands and arms (0.83–0.84 vs. gloss-only's 1.29–1.47 and SignAvatar's 0.49–0.62). This suggests that phonological attributes successfully modulate motion amplitude at the articulatory level, but at the cost of gloss identity.

### 4. Likely Cause: Shared Attributes Collapse Gloss Identity

Phonological attributes are shared across many glosses. The model likely over-relies on attribute signals while discarding gloss-specific information, causing different glosses with similar attribute profiles to produce nearly identical motions.

### Potential Directions

- Preserve gloss embedding as the primary condition; inject attributes as auxiliary/additive signals rather than replacements
- Add a gloss classification auxiliary loss to enforce discriminability in the latent space
- Investigate the conditioning fusion mechanism (e.g., cross-attention or FiLM for hierarchical injection)
- Check whether the attribute embedding dimensionality overwhelms the gloss signal
