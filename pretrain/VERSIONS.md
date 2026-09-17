# 4.1 pretraining versions

| | v1 (2026-09-13) | v2 (2026-09-17, user-requested) |
|---|---|---|
| VQ trainer | `train_part_vqvae.py` defaults (`--w_hand 1 --w_face 1 --fk_w_finger 1`) | same file, `--w_hand 0.5 --w_face 0.25 --fk_w_finger 0.3` (fingers and face fitted from low-res video are unreliable; face below hands) |
| VQ ckpt | `MotionVQ_Pooled/20260913_012619_job27263_vq_K512_w64` | tag `vq_v2_K512_w64_hw` |
| tokens | `pooled_tokens/vq_K512_w64/` | `pooled_tokens/vq_v2_K512_w64/` |
| MMM trainer | `train_mmm.py` (whole-time-step masking 30%) | `train_mmm_v2.py` (mixed: 15% time steps + 15% single cells) |
| MMM ckpt | `MotionMMM_Pooled/20260913_122233_job86413_mmm_d512_L8` | tag `mmm_v2_d512_L8` |
| downstream | aligner v1..v5, seg_bank_v1..v5 | to be redone on v2 tokens (aligner, SignBank tokens, spots, bank) |

Both versions stay runnable; v1 files are untouched except the VQ trainer's new flags (defaults reproduce v1).

**File policy (user 09-17: never overwrite earlier code):** v1 files are byte-identical to what trained v1. Every change lives in a new file:
`train_part_vqvae_v2.py`, `train_mmm_v2.py`, `align/align_dataset_v2.py`, `align/extract_gloss_pooled.py`. Note: job vqV2 376861 was launched from
`train_part_vqvae.py` while it temporarily carried the v2 flags (identical code to train_part_vqvae_v2.py); its config records the flags.
