---
name: Reorder & Fusion — Candidate A chosen (cross-attention)
description: User chose Candidate A (implicit reorder via cross-attention). Implemented as VoteAlignModule + V1_votingfusion. Candidate B (explicit permutation) rejected.
type: project
---

User raised two issues (2026-04-18): (1) LLM draft glosses are in spoken-language order, not sign-language order — need reordering; (2) sentence+gloss fusion by element-wise addition is too simple.

## Decision: Candidate A (cross-attention) — chosen 2026-04-18

User chose implicit reorder via cross-attention over explicit learned permutation. Key reasons:
- Cross-attention naturally handles variable-length gated gloss (compatible with voting gate)
- End-to-end training, no separate reorder stage
- Explicit reorder (Candidate B) conflicts with voting gate (order breaks when tokens are dropped)
- Cross-attention is proven to learn reordering in machine translation (SVO→SOV)

## Implementation (new files, not modifications)

| File | Role |
|------|------|
| `network/VoteAlignModule.py` | Stage 1: voting gate (same as VotingConditionModule). Stage 2: TransformerDecoder cross-attention — motion queries attend to gated gloss keys. Output: per-frame condition (B, T, D). |
| `network/MotionDiffusionModelV1_votingfusion.py` | Extends V1_cfg. Overrides `denoise()` to use VoteAlignModule. CFG uncond path falls back to single null_cond_emb token. |
| `trainMotionDiffusion_votingfusion.py` | Training script. Uses `How2SignSMPLXVotingDataset` (LLM draft gloss). New args: `--fusion_n_layers`, `--fusion_n_heads`. |
| `trainMotionDiffusion_votingfusion.slurm` | SLURM job script. |

## Architecture difference vs V1_voting

```
V1_voting:       gloss → vote → weighted_mean_pool → single vector (B, D) → prepend as token
V1_votingfusion: gloss → vote → gated tokens (B, K, D) → cross-attn with motion (B, T, D) → per-frame condition
```

Key: V1_votingfusion's `denoise()` prepends timestep token to motion, then runs VoteAlignModule (vote + fuse), then runs transformer encoder. The cross-attention in the fusion decoder implicitly learns which gloss is relevant at which time step = soft temporal alignment.

## Log/checkpoint paths

- Log dir: `zlog/ASLSenAvatar_v1_votingfusion/`
- Checkpoint dir: `/scratch/rhong5/weights/.../ASLSenAvatar_v1_votingfusion/`

## Prior work context

- **PGG-SLT (NeurIPS 2025)**: explicit greedy reorder via video frame-level classifier. Our approach replaces this with implicit cross-attention alignment.
- **Select and Reorder (LREC-COLING 2024)**: learnable reordering for text→gloss only, no motion generation.
- **No existing work** does implicit temporal alignment via cross-attention for sign language production.

**Why:** Cross-attention fusion unifies selection + reorder + fusion in one module. Novelty claim: "implicit temporal alignment via cross-attention replaces explicit gloss reordering."

**How to apply:** This is the most advanced model variant. Compare V1_votingfusion vs V1_voting vs V1_cfg baselines for Paper 2 ablation.
