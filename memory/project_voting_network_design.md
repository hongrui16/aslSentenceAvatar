---
name: Gloss Voting Network — Prior Design (superseded, kept for reference)
description: Earlier decision on voting network (x_t-aware joint voting inside diffusion). SUPERSEDED 2026-04-16 — user clarified voting network should be text-only, motion does not enter. See project_voting_network_design_candidates.md for current candidates A and B.
type: project
---

**STATUS (2026-04-16):** The x_t-aware "joint voting inside diffusion" design below is no longer the active plan. User clarified: the voting network is **text-only**, motion (including x_t) does not enter the network. Current candidates are in [Voting Network Candidates A/B](project_voting_network_design_candidates.md). This file is retained for reference (the x_t-conditioning idea might still be reachable by feeding t as a scalar without motion).

Context: pseudo-gloss extractor is the paper contribution. LLM produces draft gloss (frozen, in-context learning with ~30 examples). A trainable module refines the draft. Two architectures were compared; **user chose Option B** at that time.

## Option A: Text-only Refiner (Claude's initial proposal, NOT chosen)

```
sentence → LLM → draft → [frozen RoBERTa + binary head] → keep/drop labels
                                     ↑
                       BCE vs LLM draft multi-hot (2-stage training)
```
- Pre-trained once, then cached output fed to diffusion
- Pure text pathway at inference
- Just knowledge-distillation from LLM

## Option B: Joint Voting inside Diffusion (user's choice)

```
sentence → LLM (frozen, few-shot) → draft gloss tokens (variable-length)
                                           ↓
                                [Voting Network, trainable] ← x_t (current denoising step)
                                           ↓
                              soft vote weights (B, N_tokens) via sigmoid
                                           ↓
                      weighted pool → condition embedding → MDM denoiser
                                           ↓
                                   diffusion MSE loss
                                           ↓
                         (gradient propagates through voting)
```

- Single-stage end-to-end training
- Voting module lives inside the diffusion model (drop-in replacement for `gloss_proj` in `MotionDiffusionModel{V1,V2}_cfg.py`)
- No explicit gloss labels — supervision is motion reconstruction

## Decisive advantages of Option B (why user's approach wins)

1. **Ceiling above LLM.** Option A imitates LLM → capped at LLM quality. Option B's voting can discover tokens LLM missed (via motion gradient) and zero out LLM hallucinations.
2. **Cleaner paper story.** "Gloss-aware conditional diffusion with learnable token voting" is a concrete architectural contribution. Option A reduces to "we did knowledge distillation."
3. **Robust to LLM quality.** Bad LLM tokens get voted down. Option A's supervision quality = LLM quality.

## Implementation contract

- Replace `self.gloss_proj` (current 3-layer MLP) in both `MotionDiffusionModelV1_cfg.py` and `MotionDiffusionModelV2_cfg.py` with a `VotingConditionModule`
- Module API: `forward(draft_gloss_tokens, x_t) → (condition_embedding, vote_weights)`
- `vote_weights` returned for (a) auxiliary loss, (b) visualization/interpretability plots in paper
- Soft voting via sigmoid (NOT hard selection) — keeps gradient flow. Could upgrade to Gumbel-Softmax later if hard selection desired.

## Required auxiliary losses

Without regularization, voting can degenerate (all 0.5, or all 1.0). Add to `compute_loss`:

```python
L_sparse  = vote_weights.mean()                    # most weights near 0
L_entropy = -H(vote_weights)                       # push toward binary decisions
L_total   = L_diffusion + λ_sparse*L_sparse + λ_entropy*L_entropy
```

Typical weights: `λ_sparse=0.01`, `λ_entropy=0.01` — tune by inspecting vote histograms.

## x_t awareness of voting (key design choice)

Voting is conditioned on current denoising state (`x_t`), allowing different tokens to dominate at different diffusion steps:
- Large t (heavy noise): coarse semantic tokens (verbs, subjects)
- Small t (near clean): fine-grained tokens (hand-specific descriptors)

At inference time `x_T` is pure noise but the voting still works — it just starts from the same distribution and adapts as the reverse chain progresses.

## Integration order (what to build)

1. `VotingConditionModule` class (add to `network/` or inline in model files)
2. Drop-in replace `gloss_proj` in both `_cfg` model files
3. Return `vote_weights` through `forward` signature; wire up aux losses in `trainMotionDiffusion_cfg.py::compute_loss`
4. Use existing rule-based pseudo-gloss cache as initial draft source; swap to LLM-generated draft later (Stage 1 of paper plan — see [Paper Plan](project_pseudogloss_paper_plan.md))
