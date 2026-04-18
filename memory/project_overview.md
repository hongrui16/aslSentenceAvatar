---
name: ASL Sentence Avatar Project
description: Sign language avatar generation project — trains models to generate SMPL-X parameters from ASL sentence input, uses How2Sign dataset
type: project
---

ASL sentence-level avatar generation project. The goal is to generate realistic sign language motion (SMPL-X body parameters) from text/gloss input.

**Why:** Research project for sign language production — generating 3D avatar animations for ASL sentences.

**How to apply:** Key components include training scripts (train_v1.py), generation scripts (generate_smplx_param.py, generate_how2sign_smplx_param.py), motion diffusion models (MotionDiffusionModelV1/V2), back-translation model, and How2Sign SMPL-X dataset. Keep this context in mind when assisting with model training, evaluation, or data pipeline work.
