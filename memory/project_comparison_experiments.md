---
name: Gloss Comparison Experiments (rule-based + LLM-draft + voting)
description: Ablation runs — rule-based (done), LLM-draft (job 7003805, launched 2026-04-18), voting (job 7002731, ongoing). Paths and metrics for all runs.
type: project
---

Four experiments launched 2026-04-16/17, all V1 + epsilon + CFG (uncond_prob=0.1, guidance_scale=3.0), 150 epochs, upper_body=True, rot6d=False.

## Completed runs (cfg, 150 epochs each)

| Metric | sentence | gloss (rule-based) | sentence_gloss |
|--------|:--------:|:------------------:|:--------------:|
| Best Eval Loss | **0.4536** | 0.4584 | 0.4564 |
| FID ↓ | **2.2019** | 2.7198 | 2.5395 |
| Diversity | 1.9505 | 2.0524 | **2.2531** |
| MPJPE_all ↓ | **0.4538** | 0.4630 | 0.4580 |
| MPJPE_arms ↓ | **0.6365** | 0.6818 | 0.6622 |
| MPJPE_lhand ↓ | 0.5360 | **0.5321** | **0.5318** |
| MPJPE_rhand ↓ | **0.5664** | 0.5672 | 0.5645 |
| DTW ↓ | **2.4239** | 2.4606 | **2.4236** |
| Vel Error ↓ | 3.2003 | 3.0783 | **2.9531** |
| Jerk Gen ↓ | 7.7836 | 7.2545 | **6.7152** |

Checkpoint paths:
- sentence: `ASLSenAvatar_v1_cfg/.../20260416_145045_job6997838/`
- gloss: `ASLSenAvatar_v1_cfg/.../20260416_151940_job6997858/`
- sentence_gloss: `ASLSenAvatar_v1_cfg/.../20260416_152442_job6997864/`

## Voting run (Design A, still running as of 2026-04-18)

- Job 7002731, `cond_mode=voting`, bs=100, 2-layer voting transformer (4 heads, ff_mult=2, max_words=64)
- Uses LLM draft gloss (`cache/llm_draft_gloss_{train,test}.json`), NOT rule-based
- Trainable params: 19.3M (vs 15.1M for cfg runs)
- At epoch 66/150: best eval loss = **0.4514** (already lower than all cfg baselines)
- Gate stats trending down: mean 0.805→0.761, min 0.291→0.176 — learning to drop tokens

## Key takeaways

1. Rule-based gloss alone (gloss-only) is **worse** than sentence on FID and MPJPE — the extracted gloss loses too much information
2. sentence_gloss fusion helps dynamics (velocity/jerk) but hurts FID — suggests gloss adds complementary temporal info but introduces noise in distribution matching
3. Voting network at only 44% through training already has the best eval loss — strong early signal for the learning-based approach over rule-based

## LLM-draft 3-way comparison (launched 2026-04-18)

- Job **7003805**, sequential: sentence → gloss → sentence_gloss, all with `--gloss_source llm_draft`
- Script: `trainMotionDiffusion_cfg.py`, same hyperparams as rule-based runs (V1, epsilon, CFG, 150 epochs, bs=200)
- Gloss data: `cache/llm_draft_gloss_{train,test}.json` (via `How2SignSMPLXVotingDataset`)
- Log dir: `zlog/ASLSenAvatar_v1_cfg_llm/How2SignSMPLX/`
- Checkpoint dir: `/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/ASLSenAvatar_v1_cfg_llm/How2SignSMPLX/`
- Slurm script: `trainMotionDiffusion_llm3.slurm`
- Slurm output: `zlog/zslurm/cfgLLM-*.{out,err}`
- Code change: added `--gloss_source {rule, llm_draft}` to `trainMotionDiffusion_cfg.py`
- The `sentence` run does not use gloss data, so its result should match the rule-based sentence baseline (sanity check)
- Results: **pending** — update this section when runs complete

**Why:** These are the core ablation numbers for Paper 2. Rule-based gloss is the baseline; LLM-draft gloss is an intermediate; voting is the proposed method.

**How to apply:** When discussing Paper 2 results, cite these numbers. Compare rule-based vs LLM-draft vs voting to show progression. The voting run needs to finish 150 epochs before final comparison.
