---
name: Paper 1 method-contribution options (3 candidates, user likes all)
description: Three viable method angles on top of ASL-SignBank-3D dataset; user accepts all as worth considering, not yet committed
type: project
originSessionId: 2be47fb7-b67d-4054-9d23-8841007fd70f
---
Paper 1 can't be dataset-only. After ruling out (i) phonology-bottleneck ISLR (minimal-pair problem) and (ii) trivial per-attribute WLASL comparison (too weak), three method angles remain on the table — user said all three are "not bad":

**A. Phonology-Conditioned 3D Reconstruction (PhonoFit)**
Use SignBank's canonical phonology labels as constraints to refine HAMER+SMPLer-X 3D fitting for sign videos. On WLASL, look up canonical phonology by gloss and use as prior to clean noisy WLASL 3D. Evaluate via reprojection error / mesh quality / downstream ISLR with refined-vs-unrefined 3D. Novelty: phonology-guided 3D optimization. Dual appeal (3D human + sign language communities). Directly feeds Paper 2 with cleaner motion.

**B. Phonology-Driven Distribution-Level Anomaly Detection**
Learn `P(motion | gloss, phonology)` on SignBank via normalizing flow / diffusion score. Flag WLASL samples by likelihood. Separates: correct vs dialect-variant vs learner-error vs mislabel. Evaluation: AUROC on known WLASL mislabels + linguist-labeled 4-class split + downstream ISLR on cleaned WLASL. Novelty: structured density estimation in phonology-organized latent space (not classification).

**C. Gloss → Phonology → 3D Prototype Synthesis**
Compositional prototype-level 3D motion synthesis from gloss/phonology at the WORD level only (not sentence — reserved for Paper 2). Output canonical 3D motion for any ASL word. Novelty: phonology as compositional unit. Evaluation: prototype-vs-real distance, template matching for ISLR, augmentation for ISLR training. Risk: overlaps conceptually with Paper 2; requires explicit word-level vs sentence-level positioning.

**Recommendation on file: A** — clearest WLASL role (refinement, not trivial comparison), dataset's phonology is genuinely necessary, connects to existing HAMER+SMPLer-X pipeline, directly feeds Paper 2.

**Why:** User pushed back on trivial framings; these three are the non-trivial candidates. User wants to pick later — keep all three alive for now.

**How to apply:** Don't default to A in future conversations; user hasn't committed. If Paper 1 discussion resumes, present all three as live options and let user narrow down.
