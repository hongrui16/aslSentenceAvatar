---
name: CFG + Pseudo-Gloss Training Variant
description: Drop-in trainMotionDiffusion_cfg.py / MotionDiffusionModel{V1,V2}_cfg.py add epsilon prediction, CFG, and 3-way pseudo-gloss conditioning for Paper 2 ablations
type: project
---

`_cfg` suffix files extend the original MDM pipeline with three orthogonal features, all driven by CLI flags on `trainMotionDiffusion_cfg.py`.

**Files:**
- `trainMotionDiffusion_cfg.py` — training entrypoint
- `network/MotionDiffusionModelV1_cfg.py` — V1 (transformer denoiser) + CFG + gloss_proj
- `network/MotionDiffusionModelV2_cfg.py` — V2 (kinematic GNN) + CFG + gloss_proj

**`--cond_mode` ablations (for Paper 2):**
| mode | encoding |
|------|----------|
| `sentence` (default) | `condition_proj(CLIP(sentence))` — uses `How2SignSMPLXDataset` |
| `gloss` | `gloss_proj(CLIP(pseudo_gloss))` — uses `How2SignSMPLXPhonoDataset` |
| `sentence_gloss` | additive fusion of both |

`gloss_proj` is a separate 3-layer MLP; both projections share the frozen CLIP/T5 encoder. CFG null embedding replaces the *fused* condition.

**Non-obvious design decisions:**

1. **vel_loss disabled under epsilon prediction.** The original MDM design uses x_0 prediction specifically because aux losses (vel, foot-contact) need x_0 space. Recovering `x_0_pred = (x_t - √(1-ᾱ)·eps_pred) / √ᾱ` from eps explodes at large t (√ᾱ → 0), so `compute_loss` returns `vel_loss = 0` when `PREDICTION_TYPE == 'epsilon'`. Log will show `vel=0.0000` — that's intentional, not a bug.

2. **`--no_upper_body` (negative flag).** `store_true` cannot semantically default to True, so the flag was inverted: default behavior is upper-body-only, pass `--no_upper_body` to disable. `cfg.USE_UPPER_BODY = not args.no_upper_body`.

3. **dtype-match in bypass pad.** In both V{1,2}_cfg `forward`, the zero-pad tensor for non-tosave slices must use `torch.zeros_like(motion, dtype=output.dtype)` — under autocast, `motion` is fp32 but `output` is fp16/bf16, and index_put errors on dtype mismatch.

**Why:** Paper 2 compares sentence-only vs pseudo-gloss vs both conditioning to show phonological gloss improves generation. CFG gives inference-time adherence/diversity tradeoff.

**How to apply:** For new Paper 2 experiments, use `trainMotionDiffusion_cfg.py` not the original `trainMotionDiffusion.py`. On a 4GB GPU, V1 runs at `--batch_size 8` with bf16 (peak ~2.3 GB). Three comparison runs all use same cfg except `--cond_mode`.
