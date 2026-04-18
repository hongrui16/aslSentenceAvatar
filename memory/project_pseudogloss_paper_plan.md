---
name: Pseudo-Gloss Extraction — Paper Plan & Prior Work
description: Reference methods (PGG-SLT NeurIPS 2025 + Sign2GPT ICLR 2024) for LLM-based pseudo-gloss extraction, and the text→motion adaptation planned for Paper 2
type: project
---

User is planning a paper on sentence→pseudo-gloss extraction for text-to-sign motion synthesis, explicitly rejecting fusion-with-sentence and phonological-attribute paths. The gloss extractor itself is the contribution.

**Key prior work (verified via paper fetches 2026-04-16):**

1. **PGG-SLT** — "Bridging Sign and Spoken Languages: Pseudo Gloss Generation for Sign Language Translation" (Wong et al., NeurIPS 2025, arxiv 2505.15438). This is the "LLM + voting network" paper user was recalling. Two stages:
   - **Draft**: LLM (Gemini / GPT-4) with **in-context learning** — only ~30 text-gloss example pairs (0.4% of Phoenix14T) — produces draft gloss from spoken-language text. No trainable component.
   - **Reorder**: weakly-supervised multi-label classifier over **video frame features** predicts per-frame gloss presence (softmax over global gloss vocab). Loss = BCE + frequency-aware weighting + L1 temporal smoothness. Greedy reorder of LLM draft by video-inferred temporal structure.
   - Setting: video → gloss (reverse of ours).

2. **Sign2GPT** (ICLR 2024, arxiv 2405.04164). **Not** LLM-based — uses **spaCy + POS filter** to keep `{NOUN, NUM, ADV, PRON, PROPN, ADJ, VERB}` lemmas. Authors note pseudo-glosses are in spoken-language order, not sign order. This is the spaCy baseline we were considering.

**Planned adaptation for our text→motion setting (Paper 2):**

PGG-SLT's reordering depends on having *video* at training time. We have motion (SMPL-X) instead. Proposed pipeline:

```
1. sentence → [LLM few-shot, frozen, ~30 examples]  → draft gloss (spoken order)
2. motion (B, T, 159) → [learnable classifier head] → per-frame gloss probabilities
                          ↑ trained with BCE against LLM-draft gloss set
3. greedy reorder draft → sign-order pseudo-gloss
4. feed reordered gloss as condition to MotionDiffusionModel (cond_mode='gloss')
```

**Claimed contribution delta over PGG-SLT:**
- Transfer PGG-SLT's LLM-draft + voting-classifier design from video→text to **text→motion** synthesis
- Replace video features with 3D SMPL-X motion features (sparser, harder signal)
- Downstream evaluation is motion quality (FID / MPJPE), not BLEU — different ablation surface

**Why:** User rejected (a) rule-based alone (too brittle, weak story), (b) gloss+sentence fusion, (c) phonological attributes (saved for different paper). Wants focused gloss-extractor paper with LLM + trainable module.

**Paper framing — rule-based vs. learning-based (user's preferred terminology, 2026-04-16):**
- **Rule-based baseline**: hand-crafted ASL linguistic rules (keep pronouns, n't→NOT, drop articles/copulas/discourse markers), deterministic offline extraction. See [Pseudo-Gloss Extraction](project_pseudogloss_extraction.md).
- **Learning-based method (ours)**: LLM few-shot draft + motion-features voting classifier, trained jointly with diffusion loss.
- This binary frames the contribution cleanly and aligns with prior work: Sign2GPT/PGG-SLT are learning-based; classical SLT dictionary-projection pipelines are rule-based.
- Contribution sentence: "we replace the rule-based extraction stage with a learning-based voting network conditioned on motion."

**How to apply:** When writing the paper or discussing contribution, use "rule-based vs. learning-based" (not "hard-rule vs. soft" or "deterministic vs. neural"). When implementing, start with PGG-SLT Step 1 (LLM few-shot with 30 hand-curated How2Sign examples, cache output). Then add motion-features classifier for reordering. Do NOT silently fall back to our existing rule-based extractor — the rule-based code is now officially the baseline in the ablation table.
