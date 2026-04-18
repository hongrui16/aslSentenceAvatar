---
name: Voting Network Design — Candidates A & B (text-only)
description: Two candidate voting-network designs (Paper 2, learning-based pseudo-gloss). Voting network is TEXT-ONLY — motion does not enter the network. Discussion pending.
type: project
---

Voting network for Paper 2 (learning-based pseudo-gloss). Critical constraint clarified by user 2026-04-16: **the voting network takes text only, NOT motion**. Motion GT enters only through downstream diffusion reconstruction loss (gradients flow back through voting → LLM-draft embeddings).

User narrowed to two candidates below. Design C (fixed-vocab multi-label classifier) and the earlier motion-features cross-attention design are both rejected.

## Design A — Per-token keep/drop gate on single LLM draft

```
LLM draft (K tokens, variable per sample)
         ↓ embed + transformer (attention_mask handles variable K)
         ↓ per-position sigmoid head
gates of shape (B, K)
         ↓ soft-gate × gloss_embedding, OR hard select when gate > 0.5
refined gloss sequence (≤ K, variable length)
         ↓
condition → MotionDiffusionModel
```

- **Voting semantics**: token-level keep/drop on one LLM draft
- **Variable-length I/O**: input K tokens, output K gates. Standard transformer over variable sequence — same pattern as BERT NER. No special handling needed.
- **Supervision**: soft gate makes it differentiable; diffusion reconstruction loss (ε-prediction MSE vs noised motion GT) backprops through gates into transformer weights
- **Pros**: simplest, single LLM call per sample at train time, structure mirrors NLP token classification
- **Cons**: if the single LLM draft is bad, no second chance — no redundancy to vote over

## Design B — Ensemble voting across multiple LLM samples

```
sentence → LLM × N (different temperatures / prompts / models) → N candidate drafts
           ↓
each candidate token gets votes across the N drafts
           ↓
voting net scores / learns to weight candidates   (transformer or simple counting)
           ↓
consensus gloss set (variable length, driven by threshold or top-k)
           ↓
condition → MotionDiffusionModel
```

- **Voting semantics**: literal voting among N LLM samples for the same sentence
- **Variable-length I/O**: N×K_n candidate tokens in, variable-size consensus set out
- **Supervision**: can be (a) unsupervised counting + threshold (no training needed), (b) transformer learns voting weights via diffusion recon loss, (c) both
- **Pros**: addresses LLM hallucination by redundancy; transformer-free version works immediately; closer in spirit to the "voting" word
- **Cons**: N× LLM inference cost at training time (can cache offline); need to decide how to align same-meaning-different-spelling tokens across drafts (normalization / lemmatization / embedding distance)

## Hybrid option mentioned but not decided

B first (ensemble to reduce LLM noise) → A second (token-level keep/drop on the consensus set). Would combine redundancy with learnable refinement.

## Open decisions (discussion pending with user)

1. A, B, or A∘B hybrid
2. For B: how to align tokens across drafts (exact match / lemma / embedding similarity)
3. Supervision: end-to-end diffusion loss only, or also contrastive / distillation-from-LLM auxiliary loss
4. Whether inference-time pipeline needs a separate text-only student to mimic voting output (still open — may be moot if voting net itself is text-only, it just runs at inference too)

## How to apply

When user returns to this discussion, recall these two designs by letter. Do NOT re-propose motion-features cross-attention (rejected — the voting network is text-only by design).
