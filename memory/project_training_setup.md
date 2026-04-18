---
name: NSA training run configuration
description: Current NSA training run settings on 80GB A100 — 500 epochs, bs=128
type: project
originSessionId: 2be47fb7-b67d-4054-9d23-8841007fd70f
---
Launched 2026-04-13: NSA reproduction training run on A100 80GB, batch_size=128, epochs=500 (truncated from paper's 2000 for efficiency; LinearLR auto-scales 1e-3→1e-6 over reduced step count). Only `newest_model.pt` and `best_model.pt` saved — no periodic checkpoints. Slurm script: `train_NeuralSignActors.slurm`.

**Why:** User wants faster turnaround for ablation comparisons; 2000 epochs is overkill given loss typically plateaus around 300-500. Full 2000-epoch run reserved for final paper-claim reproduction only.

**How to apply:**
- Batch size sweet spot: 128 (not larger). Paper LR 1e-3 is not scaled up; going beyond bs=192 risks divergence without LR tuning.
- `config.py` `MIXED_PRECISION = "bf16"` (switched from fp16 for diffusion numerical stability on A100).
- Dataset: 30687 train / 2286 test samples. 30687/128 = ~240 steps/epoch → ~120k total steps over 500 epochs.
- Decision point: if val loss still dropping at epoch 400-500, extend to 800-1000. If plateaued by ~300, current run is sufficient.
- GPU spec for this project: A100 80GB (not MIG slices) — fits comfortably, 15-20GB actual use.
