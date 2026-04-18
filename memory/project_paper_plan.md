---
name: Two-paper research plan — SignBank attribute completion + phonological ASL generation
description: Paper 1 (AAAI 2027, 8/1 ddl) builds ASL-SignBank-3D dataset + attribute classifiers; Paper 2 uses its output for phonologically-conditioned generation
type: project
originSessionId: 2be47fb7-b67d-4054-9d23-8841007fd70f
---
**Paper 1 — ASL-SignBank 3D (AAAI 2027, deadline 2026-08-01):**
Pivot from WLASL (poor handshape classification due to data quality) to ASL-SignBank (4,479 gloss entries, ~70% fully annotated, ~30% with missing attribute fields marked "-"). Pipeline:
1. 3D fit all SignBank videos with HAMER + SMPLer-X → SMPL-X params.
2. Train attribute classifiers (handshape / location / flexion / selected fingers) on labeled 70%. Dual input: RGB hand crop + SMPL-X hand pose.
3. Predict missing "-" fields on remaining 30%.
4. HandMotionScript: extract kinematic features (distances, palm orientation) from 3D pose.
5. LLM (Gemini) generates natural-language hand-motion description from phonology + HandMotionScript.
6. Release ASL-SignBank-3D dataset.

**Paper 2 — Phonological Conditioning for ASL Sentence Generation (AAAI/CVPR 2027):**
Built on Paper 1's output. How2Sign sentences → gloss via existing word-matching pipeline → sentence-level phonological attribute sequence → injected into NSA-style generator as extra conditioning. Contributes How2Sign-SMPLX-Motion (sentence-level 3D motion + NL description).

**Why:** Handshape classification on WLASL failed due to noisy data/inconsistent annotation; SignBank is much cleaner. Attribute completion is a natural self-supervised framing that reuses that classifier work. Phonology conditioning is a credible novelty angle for sign language production.

**How to apply:**
- `--use_phono_attribute` flag on `train_NeuralSignActors.py` is the hook for Paper 2; keep it wired in even while NSA reproduction is the current focus.
- Paper 2 depends on Paper 1's output — start Paper 2 preliminary work on the 70% already-labeled SignBank subset to de-risk schedule.
- Timeline is tight: 3D fitting on 4479 videos alone is 1-2 weeks of compute.
- Bottleneck risk: evaluation of attribute predictions (no GT for the 30% missing) — need hold-out or linguist-annotated validation set.
