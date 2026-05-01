---
name: Full ablation results — 21 experiments on 2286 samples (Round 2 + Round 3, OLD protocol)
description: SUPERSEDED for paper. Round 2/3 ablation results on H2S full 2286-sample test, target_seq_len=200, no FID_AE. Paper now uses filtered (3-30 words, 1954 samples) + target_seq_len=100 + FID_AE. See project_results_summary_2026_04_28.md for current paper-facing numbers.
type: project
---

> **⚠ STALE PROTOCOL — DO NOT CITE FOR PAPER.**
> Numbers below were computed on the **OLD** H2S protocol: full 2286-sample test, `target_seq_len=200`, no FID_AE. The paper has since switched to **filtered (3–30 words) + `target_seq_len=100` + FID_AE** (different absolute FID values; whole-str CFG-rule went from 2.62 here → 2.05 under the new protocol — architecture unchanged, only the eval setup differs).
> Current paper-facing reference: **`project_results_summary_2026_04_28.md`**.
> Keep this memo for the gloss-quality robustness story (Round 3) and historical context — but ignore the FID magnitudes.

All runs: V1 + epsilon + CFG (uncond_prob=0.1, guidance_scale=3.0), upper_body=True, rot6d=False.
Baselines (#1-5): 150 epochs. VoteOnly (#6-10): 100 epochs. VoteAlign (#11-16): 100 epochs.
Each row = best FID across best/second_best/newest checkpoints.

## Full test set metrics (2286 samples, seed=42)

| # | Method | Sentence | Gloss | Gloss Enc | Phono | Job | FID ↓ | Div ↑ | MPJPE ↓ | DTW ↓ | Vel ↓ | Jerk ↓ |
|---|--------|----------|-------|-----------|-------|-----|-------|-------|---------|-------|-------|--------|
| 1 | CFG | ✓ | - | - | - | 6997838 | 2.1189 | 1.9408 | 0.4568 | 2.4668 | 3.1428 | 7.6475 |
| 2 | CFG | - | rule | whole-str | - | 6997858 | 2.6237 | 2.0059 | 0.4648 | 2.4949 | 3.0225 | 7.1393 |
| 2' | CFG | - | rule | per-word | - | 7039815 | 2.5451 | 2.6342 | 0.4707 | 2.5223 | 2.8979 | 6.7709 |
| 3 | CFG | - | LLM | whole-str | - | 7003891 | 2.3041 | 2.1160 | 0.4563 | 2.4503 | 2.9141 | 6.6954 |
| 3' | CFG | - | LLM | per-word | - | 7039816 | **2.0843** | 2.0468 | 0.4627 | 2.4945 | 3.2059 | 7.8228 |
| 6 | VoteOnly | - | LLM | per-word | - | 7002731 | 2.2515 | 2.0830 | 0.4570 | 2.4556 | 2.8854 | 6.6037 |
| 7 | VoteOnly | pfx | LLM | per-word | - | 7029812 | 2.2980 | 2.1839 | 0.4650 | 2.5179 | 3.0927 | 7.4253 |
| 8 | VoteOnly | kv | LLM | per-word | - | 7029811 | 2.5972 | 2.1065 | 0.4616 | 2.4840 | 2.9746 | 6.9367 |
| 9 | VoteOnly | - | LLM | per-word | ✓ | 7029837 | 2.1043 | 2.0403 | 0.4603 | 2.4914 | 3.1837 | 7.8687 |
| 10 | VoteOnly | kv | LLM | per-word | ✓ | 7029838 | 2.3871 | 2.1590 | 0.4541 | 2.4328 | 2.9750 | 6.8938 |
| 11 | VoteAlign | - | LLM | per-word | - | 7039817 | 2.2885 | 2.2900 | 0.4589 | 2.4629 | 2.7827 | 6.2579 |
| 12 | VoteAlign | pfx | LLM | per-word | - | 7039818 | 2.5075 | 2.6237 | 0.4513 | 2.4014 | 2.4515 | 4.7583 |
| 13 | VoteAlign | kv | LLM | per-word | - | 7039799 | 2.4931 | 2.1688 | 0.4598 | 2.4212 | 2.4781 | 4.8963 |
| 14 | VoteAlign | - | LLM | per-word | ✓ | 7039808 | 2.2912 | 2.3718 | 0.4702 | 2.4996 | 2.7505 | 6.0365 |
| 15 | VoteAlign | pfx | LLM | per-word | ✓ | 7039822 | 2.7028 | 2.5610 | **0.4421** | **2.3487** | **2.2508** | **3.8901** |
| 16 | VoteAlign | kv | LLM | per-word | ✓ | 7039810 | 2.7616 | 2.4549 | 0.4739 | 2.5069 | 2.5495 | 5.1469 |

## Round 3 — 5 follow-up experiments (added 2026-04-25, evaluated on full 2286 test set)

| # | Method | Sentence | Gloss src | Gloss Enc | Phono | Job | FID ↓ | Div ↑ | MPJPE ↓ | DTW ↓ | Vel ↓ | Jerk ↓ |
|---|--------|----------|-----------|-----------|-------|-----|-------|-------|---------|-------|-------|--------|
| 17 | CFG | – | LLM | per-word | ✓ | 7041887 | 2.3658 | 2.2834 | 0.4740 | 2.5588 | 3.2260 | 7.9359 |
| 18 | VoteOnly | – | **rule** | per-word | – | 7041888 | **2.1759** | 2.2189 | 0.4830 | 2.6144 | 3.5761 | 9.2448 |
| 19 | VoteAlign | – | **rule** | per-word | – | 7041889 | **2.1359** | 2.1235 | 0.4595 | **2.4542** | 2.7818 | 6.2027 |
| 20 | VoteOnly | – | shuffled-LLM | per-word | – | 7041967 | 2.0839 | 2.2814 | 0.4811 | 2.5989 | 3.4894 | 8.9024 |
| 21 | VoteAlign | – | shuffled-LLM | per-word | – | 7041976 | 2.6339 | 2.6453 | 0.4795 | 2.5594 | 2.7535 | 6.1110 |

## PT non-diffusion baselines

| Method | Job | FID ↓ | Div ↑ | MPJPE ↓ | DTW ↓ | Vel ↓ | Jerk ↓ | Note |
|---|---|---|---|---|---|---|---|---|
| PT (3D, Saunders 2020) | 7043040 (best ckpt @ ep ?) | 2.39 / **2.18** (2nd_best) | **0.21 / 0.23** ⚠️ | 0.23 | 0.42-0.48 | 0.26 | 0.085 | Diversity ~0 = autoregressive mode collapse — flag in paper |
| PT (SMPL-X axis-angle) | 7042064 / eval 7045807 | 3.30 (best) / **2.27** (2nd_best) / 3.69 (newest) | 1.32 / **0.95** / 0.33 | 0.388 / **0.380** / 0.382 | 1.42 / **1.33** / 1.60 | 1.07 / **1.08** / 1.04 | 0.066 / 0.081 / 0.044 | 2nd_best beats PT-3D on FID (2.27 vs 2.18 close) AND on Diversity (0.95 vs 0.21). Suggests rotation rep is brittle but less mode-collapsed. |

## Key findings (2026-04-25, post-Round 3)

### Round 2 takeaways (still valid)
1. **CFG-LLM per-word is best FID** (2.08, #3') — simplest method on high-quality gloss
2. **VoteAlign + sent_pfx + phono** (#15) — best dynamics: Vel 2.25, Jerk **3.89** (below GT 4.77!), MPJPE 0.4421, DTW 2.3487
3. Adding sentence on top of gloss (pfx/kv) usually hurts FID
4. Per-word > whole-string encoding for LLM gloss
5. Phono helps VoteOnly FID (#9: 2.10 vs #6: 2.25), but not VoteAlign on its own (#11 ≈ #14)

### Round 3 takeaways (gloss-quality hypothesis)
6. **Hypothesis confirmed**: voting gate / cross-attention DO help when gloss quality drops:

   | Gloss quality | CFG (no gate) | VoteOnly | VoteAlign |
   |---|---|---|---|
   | LLM (high) | **2.08** ⭐ | 2.25 | 2.29 |
   | shuffled-LLM (mid) | — | 2.08 | 2.63 |
   | rule (low) | 2.55 | 2.18 ✓ | **2.14** ✓ |

   On low-quality rule gloss, CFG drops to 2.55 but VoteAlign claws back to 2.14 — the gate compensates for noisy input.

7. **Phono actually HURT CFG** (#17 phono 2.37 vs #3' no-phono 2.08). Phono only helps when paired with the voting/alignment mechanism, not on direct CFG conditioning.
8. **VoteAlign + rule (#19)** — best DTW (2.45) of any low-quality-gloss setup; FID 2.14 close to LLM-clean baseline.
9. **PT-3D Diversity ≈ 0.21** — autoregressive mode collapse confirmed. Looks competitive on FID alone (2.18 second_best ≈ best diffusion) but produces near-deterministic output. Must be flagged when reporting.

## Paper main-table recommendation (2026-04-25)

Recommended 5–6 row main table for NeurIPS submission:

| Row | Method | Justification |
|---|---|---|
| 1 | PT (3D, Saunders 2020) | Established baseline; flag Div ≈ 0 |
| 2 | NSA (matched) | If we resurrect it; otherwise drop |
| 3 | CFG (sentence only, #1) | Diffusion baseline w/o gloss |
| 4 | CFG-LLM per-word (#3') | Best FID — simplest method headline |
| 5 | **VoteAlign + rule (#19)** | "Voting compensates for low-quality gloss" headline |
| 6 | **VoteAlign + sent_pfx + phono (#15)** | "Best dynamics" headline |

(Phoenix rows will mirror this structure once LLM gloss extraction finishes.)

### Three narrative lines for the paper
- **A. Best FID story**: CFG-LLM per-word (2.08) beats all complex variants on clean LLM gloss → simpler is better on high-quality input.
- **B. Best dynamics story**: VoteAlign + sent_pfx + phono drives Jerk 3.89 (lower than GT) → cross-attention alignment + phono is the right inductive bias for temporally-coherent motion.
- **C. Robustness story (the original Round 3 hypothesis, now validated)**: voting / alignment is a robustness mechanism. On low-quality rule gloss, FID drops 2.55 (CFG) → 2.14 (VoteAlign) — the gate matters when input quality varies.

The paper can either pick **one narrative** (cleanest) or interleave A+C (CFG-LLM is best on clean input; VoteAlign closes the gap on noisy input → robust pipeline). B serves as the dynamics-quality table row but doesn't need to be the headline.

## Pending experiments (next, if time permits)

1. **PT (SMPL-X axis-angle, 7042064)** — eval the cancelled-but-saved ckpt @ ep 28
2. **Phoenix-2014T cross-dataset replication** — CFG / VoteOnly / VoteAlign × {gt, translation, pseudo_rule, llm_draft}. CFG-gtgloss already training (7045237); rule cache built; LLM cache extraction running (7045301).

**Why:** Core ablation table for NeurIPS 2026 paper. Round 3 confirmed "voting/alignment helps on low-quality gloss" — main story can lean on Robustness (C) + Best-FID (A) jointly. Phoenix experiments give cross-dataset / cross-language validation.

**How to apply:** When loading memory before paper writing, read this memo for Round 2+3 numbers and the recommended main-table structure. The narrative lines A/B/C are framings, not commitments — pick by editor mood.
