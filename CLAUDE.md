# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ASL sentence-level avatar generation: English text → SMPL-X motion sequences → rendered mesh video/GIF. Uses diffusion models (MDM-style) for motion synthesis and a back-translation model (motion → text) for quality assessment.

## Key Commands

### Training (SLURM cluster)

Motion diffusion model (V1 or V2):
```bash
sbatch train_v1.slurm
# Or directly:
accelerate launch --num_processes=1 --num_machines=1 --machine_rank=0 \
    --main_process_ip=localhost --main_process_port=$PORT \
    train_v1.py --dataset How2SignSMPLX --use_upper_body --use_rot6d --model_version v2
```

Back-translation model (motion → text):
```bash
sbatch train_backtrans.slurm
# Or directly:
accelerate launch --num_processes=1 --num_machines=1 --machine_rank=0 \
    --main_process_ip=localhost --main_process_port=$PORT \
    train_backtrans.py --use_upper_body --epochs 60 \
        --freeze_epochs 25 --encoder_lr 1e-4 --decoder_lr 1e-5 \
        --token_drop 0.3 --n_pool 24
```

### Generation

Sentence-level (How2Sign):
```bash
python generate_how2sign_smplx_param.py \
    --checkpoint /path/to/best_model.pt \
    --use_rot6d --use_upper_body --model_version v1
```

Single-word gloss:
```bash
python generate_smplx_param.py \
    --glosses drink before \
    --render_mesh \
    --checkpoint /path/to/best_model.pt \
    --use_upper_body --use_rot6d
```

### Evaluation

```bash
python eval_generated_smplx_param_v2.py \
    --checkpoint /path/to/best_model.pt \
    --render_mesh --gif --render_comparison \
    --use_upper_body --use_rot6d \
    --dataset_name ASL3DWord --evaluate
```

### Interactive GPU allocation

```bash
salloc -p contrib-gpuq -q gpu --nodes=1 --ntasks-per-node=1 --gres=gpu:2g.20gb:1 --mem=40gb -t 0-24:00:00
```

## Environment

- Python virtualenv: `/home/rhong5/py310Torch/bin/activate`
- Cluster modules: `gnu10`, `python`
- Logs: `./zlog/` (training logs), `./zlog/zslurm/` (SLURM output)
- Checkpoints: `/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/`
- SMPL-X model files: `./human_models/human_model_files/`

## Architecture

### Pipeline

```
English text → [CLIP/T5 encoder] → [Diffusion Transformer denoiser] → SMPL-X motion (T, 159)
                                                                            ↓
                                                                   [SMPL-X mesh renderer] → GIF
                                                                            ↓
                                                                   [BackTranslation T5] → reconstructed text
```

### Models

- **MotionDiffusionModelV1** (`MotionDiffusionModelV1.py`): Transformer encoder denoiser with cosine noise schedule (1000 steps), DDIM sampling (50 steps), x_0 prediction. Text conditioning via CLIP or T5.
- **MotionDiffusionModelV2** (`MotionDiffusionModelV2.py`): V1 + kinematic GNN encoder that processes joints along the SMPL-X skeleton tree before the transformer.
- **BackTranslationModel** (`BackTranslationModel.py`): Transformer encoder + Perceiver-style temporal pooling (24 queries) + T5 decoder. Two-stage training: freeze T5 for first N epochs, then unfreeze with differential LR.

### Data

- **How2SignSMPLXDataset** (`dataloader/How2SignSMPLXDataset.py`): Loads per-frame SMPL-X pkl files (53 joints × 3 axis-angle = 159-D). Supports rot6d (318-D) and upper-body-only modes. Sequences padded/sampled to 200 frames.
- Data root: `/scratch/rhong5/dataset/Neural-Sign-Actors`
- Metadata: `how2sign_realigned_{train,val,test}.xlsx`

### Config

`config.py` contains `BaseConfig` and dataset-specific configs (e.g., `How2Sign_SMPLX_Config`). Key flags: `--use_rot6d`, `--use_upper_body`, `--model_version {v1,v2}`, `--text_encoder_type {clip,t5}`.

### Loss weights (diffusion training)

Joint group weights in reconstruction loss: root=0, lower_body=0, torso=0.5, arms=5, left_hand=5, right_hand=5, jaw=0.1. Velocity loss added with configurable `VEL_WEIGHT`.

### Rotation utilities

`utils/rotation_conversion.py`: axis-angle ↔ 6D rotation conversions, joint group definitions, motion postprocessing.

## Memory location

All auto-memory files live in `./memory/` at the project root (NOT the default `~/.claude/projects/.../memory/`).

- Read: the index below auto-loads via `@import`; read individual entries on demand.
- Write: when saving new memories, write the `.md` file to `./memory/` and add its pointer to `./memory/MEMORY.md`.

@memory/MEMORY.md
