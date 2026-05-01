---
name: Planned additions for NeurIPS submission (4 items)
description: Four planned additions to strengthen the paper — (1) phonology attributes, (2) Phoenix14T dataset, (3) alignment quantitative eval, (4) more SLP baselines
type: project
---

User confirmed (2026-04-18) that all four additions below are planned and necessary for NeurIPS submission.

## 1. Phonological attribute conditioning (highest priority)

Add ASL-SignBank phonology attributes (handshape, location, movement, selected fingers, flexion) as structured conditioning on top of voting+fusion.

This creates a three-layer contribution: gloss selection → temporal reordering → phonological conditioning.

**Open design questions:**
1. How to map How2Sign glosses to SignBank entries (string match? fuzzy?)
2. Which attributes to use
3. Where to inject: concatenate to gloss embeddings before voting? Separate branch? Part of cross-attention keys?
4. How to handle glosses not in SignBank (fallback?)
5. Can Paper 1's attribute classifiers provide phonology embeddings?

**Existing hooks:** `--use_phono_attribute` flag in config.py

## 2. Second dataset — Phoenix14T

Add Phoenix-2014T (German Sign Language) to demonstrate cross-language generalization.
- Different sign language (DGS vs ASL)
- Well-established benchmark in SLT literature
- PGG-SLT reports results on Phoenix14T, so direct comparison possible

## 3. Cross-attention alignment quantitative analysis

Go beyond attention visualization — measure alignment quality quantitatively.
- Compare learned attention patterns against GT gloss timing (if available) or pseudo-GT from forced alignment
- Compute alignment accuracy / correlation metric
- Show that cross-attention genuinely learns reordering, not just uniform attention

## 4. More SLP baselines

Compare against established sign language production methods:
- SignAvatar (CVAE, already compared in hong2026phonologyguided)
- Progressive Transformers (Saunders et al., 2020)
- T2S-GPT (Xie et al., 2024)
- SignDiff (Fang et al., 2023)

**Why:** User confirmed these four additions are needed for NeurIPS-level contribution. Without them, the paper is "borderline" — with them, the method story (3-layer contribution) and experimental coverage (2 datasets, multiple baselines, quantitative alignment eval) are sufficient.

**How to apply:** Prioritize (1) phonology attributes first (biggest impact on contribution). (2) and (4) can be parallelized. (3) can be done after votingfusion training completes.
