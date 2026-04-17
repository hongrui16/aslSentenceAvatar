"""
SignBank SMPL-X Inference Script — CFG version
================================================

Generate sign language motion from gloss with CFG,
save SMPL-X parameters (.npz) per frame.

Usage:
    python generate_smplx_param_cfg.py \
        --checkpoint path/to/best_model.pt \
        --glosses AMAZING HELLO THANK-YOU \
        --render_mesh --use_rot6d --use_upper_body

    # Override guidance scale:
    python generate_smplx_param_cfg.py \
        --checkpoint path/to/best_model.pt \
        --glosses drink before --guidance_scale 5.0
"""
import logging
import os
import sys
import argparse
import random
from typing import List, Dict, Optional
from accelerate.logging import get_logger

import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.MotionDiffusionModelV1_cfg import MotionDiffusionModelV1_CFG
from network.MotionDiffusionModelV2_cfg import MotionDiffusionModelV2_CFG
from utils.rotation_conversion import postprocess_motion

# ── reuse utilities from original generate_smplx_param ────────────────────────
from generate_smplx_param import (
    load_model_weight,
    load_smplx_model,
    params_to_mesh,
    render_smplx_frame,
    save_gif,
    save_frame_npz,
    PARAM_SLICES,
    split_params,
    get_glosses_from_dataset,
)


# =============================================================================
# Generation with CFG
# =============================================================================

@torch.no_grad()
def generate_from_gloss(model, gloss: str, seq_len: int, device: str, cfg=None):
    """Generate (T, 159) SMPL-X params from gloss with CFG."""
    guidance_scale = getattr(cfg, 'GUIDANCE_SCALE', 1.0)
    if cfg.USE_LABEL_INDEX_COND:
        gloss_index = cfg.GLOSS_NAME_LIST.index(gloss)
        label_indices = torch.tensor([gloss_index], dtype=torch.long).to(device)
        motion = model.generate(label_indices, seq_len=seq_len, device=device,
                                guidance_scale=guidance_scale)
    else:
        motion = model.generate([gloss], seq_len=seq_len, device=device,
                                guidance_scale=guidance_scale)

    diff = (motion[0, 1:] - motion[0, :-1]).abs().mean()
    print(f"Generated frame diff: {diff:.8f}")

    motion_raw = motion.squeeze(0).cpu().numpy()

    if cfg is not None:
        motion_raw = postprocess_motion(motion_raw, cfg)

    return motion_raw


# =============================================================================
# Process one gloss
# =============================================================================

def process_a_gloss(model, gloss, output_dir, seq_len, device,
                    smpl_x=None, img_size=512, dump_param=False,
                    cfg=None, make_gif=True, gif_fps=8, dataset=None):

    if cfg.USE_PHONO_ATTRIBUTE:
        gloss_str = dataset._gloss_with_phono(gloss)
    else:
        gloss_str = gloss

    motion = generate_from_gloss(model, gloss_str, seq_len, device, cfg)
    print('motion.shape', motion.shape, 'cfg.N_FEATS', cfg.N_FEATS)
    T = motion.shape[0]

    gloss_dir = os.path.join(output_dir, gloss)
    os.makedirs(gloss_dir, exist_ok=True)

    if smpl_x is not None:
        render_dir = os.path.join(gloss_dir, 'renders')
        os.makedirs(render_dir, exist_ok=True)

    gif_frames = []
    for t in range(T):
        params = split_params(motion[t])

        if dump_param:
            npz_path = os.path.join(gloss_dir, f"{gloss}_{t:06d}_p0.npz")
            save_frame_npz(params, npz_path)

        if smpl_x is not None:
            try:
                import cv2
                vertices, faces = params_to_mesh(smpl_x, params)

                img = render_smplx_frame(vertices, faces,
                                         img_w=img_size, img_h=img_size,
                                         debug=(t == 0), gloss=gloss)

                if make_gif:
                    gif_frames.append(img)

            except Exception as e:
                import traceback
                print(f"  Render error frame {t}: {e}")
                traceback.print_exc()

    if make_gif and gif_frames:
        gif_path = os.path.join(gloss_dir, f"{gloss}.gif")
        save_gif(gif_frames, gif_path, fps=gif_fps)

    return T
