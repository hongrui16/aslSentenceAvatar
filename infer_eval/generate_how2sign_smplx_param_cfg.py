"""
How2Sign Generation + Evaluation Script -- CFG version
=======================================================

For N test sentences:
  - Generate motion with CFG -> save as GIF
  - Render GT motion -> save as GIF
  - Comprehensive metrics: MPJPE (per group), velocity/accel error,
    FID, DTW, diversity, back-translation BLEU

Usage:
    python generate_how2sign_smplx_param_cfg.py \
        --checkpoint path/to/best_model.pt \
        --use_upper_body --compute_metrics \
        --bt_checkpoint path/to/backtrans_best.pt
"""

import os
import sys
import random
import argparse
import pickle
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network.MotionDiffusionModelV1_cfg import MotionDiffusionModelV1_CFG
from network.MotionDiffusionModelV2_cfg import MotionDiffusionModelV2_CFG
from config import How2Sign_SMPLX_Config
from dataloader.How2SignSMPLXDataset import How2SignSMPLXDataset
from utils.rotation_conversion import postprocess_motion

from generate_smplx_param import (
    load_model_weight,
    load_smplx_model,
    params_to_mesh,
    render_smplx_frame,
    save_gif,
    PARAM_SLICES,
)

from generate_how2sign_smplx_param import (
    _pool_sequence,
    compute_fid,
    compute_dtw,
    compute_mpjpe,
    gt_params_to_flat,
    sentence_to_slug,
    params_to_mesh_fixed,
    render_frame_fixed,
    render_to_gif,
    load_gt_params,
    sample_gt_indices,
)

from dataloader.How2SignSMPLXPhonoDataset import extract_gloss_string
from utils.rotation_conversion import (
    TORSO_INDICES, ARMS_INDICES, LHAND_INDICES, RHAND_INDICES,
    ROOT_INDICES, UPPER_BODY_INDICES,
)


# =============================================================================
# Helpers
# =============================================================================

def split_params(flat: np.ndarray):
    return {name: flat[s:e].copy() for name, (s, e) in PARAM_SLICES.items()}


# =============================================================================
# Comprehensive motion-quality metrics
# =============================================================================

def _reshape_to_joints(seq, n_feats=3):
    """(T, D) -> (T, J, n_feats)"""
    T, D = seq.shape
    return seq.reshape(T, -1, n_feats)


def _align_length(pred, gt):
    """Truncate both to the shorter length so frame-wise metrics work."""
    T = min(len(pred), len(gt))
    return pred[:T], gt[:T]


def compute_mpjpe_per_group(pred, gt, n_feats=3):
    """Per-joint-group MPJPE on the 53-joint layout."""
    pred, gt = _align_length(pred, gt)
    pred_j = _reshape_to_joints(pred, n_feats)
    gt_j   = _reshape_to_joints(gt,   n_feats)
    groups = {
        'torso': TORSO_INDICES,
        'arms':  ARMS_INDICES,
        'lhand': LHAND_INDICES,
        'rhand': RHAND_INDICES,
    }
    results = {}
    for name, indices in groups.items():
        valid = [i for i in indices if i < pred_j.shape[1]]
        if valid:
            err = np.linalg.norm(pred_j[:, valid] - gt_j[:, valid], axis=-1).mean()
            results[f'MPJPE_{name}'] = float(err)
    results['MPJPE_all'] = float(
        np.linalg.norm(pred_j - gt_j, axis=-1).mean()
    )
    return results


def compute_velocity_error(pred, gt):
    """L2 error between first-order finite differences."""
    pred, gt = _align_length(pred, gt)
    vel_pred = np.diff(pred, axis=0)
    vel_gt   = np.diff(gt,   axis=0)
    return float(np.linalg.norm(vel_pred - vel_gt, axis=-1).mean())


def compute_acceleration_error(pred, gt):
    """L2 error between second-order finite differences."""
    pred, gt = _align_length(pred, gt)
    acc_pred = np.diff(pred, n=2, axis=0)
    acc_gt   = np.diff(gt,   n=2, axis=0)
    return float(np.linalg.norm(acc_pred - acc_gt, axis=-1).mean())


def compute_jerk(seq):
    """Mean jerk magnitude (third derivative). Lower = smoother."""
    if seq.shape[0] < 4:
        return 0.0
    jerk = np.diff(seq, n=3, axis=0)
    return float(np.linalg.norm(jerk, axis=-1).mean())


def compute_diversity(all_gen_seqs):
    """Average pairwise L2 distance of mean-pooled features."""
    feats = np.stack([s.mean(axis=0) for s in all_gen_seqs])
    n = len(feats)
    if n < 2:
        return 0.0
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(np.linalg.norm(feats[i] - feats[j]))
    return float(np.mean(dists))


# =============================================================================
# Generation with CFG
# =============================================================================

@torch.no_grad()
def generate_motion(model, sentence: str, seq_len: int, device: str, cfg,
                    gloss_string: str = None):
    """Generate (T, 159) axis-angle params from sentence text with CFG."""
    guidance_scale = getattr(cfg, 'GUIDANCE_SCALE', 1.0)
    cond_mode = getattr(cfg, 'COND_MODE', 'sentence')

    gloss_input = None
    if cond_mode in ('gloss', 'sentence_gloss'):
        if gloss_string is None:
            gloss_string = extract_gloss_string(sentence)
        gloss_input = [gloss_string]

    motion = model.generate([sentence], seq_len=seq_len, device=device,
                            guidance_scale=guidance_scale,
                            gloss_input=gloss_input)
    motion_raw = motion.squeeze(0).cpu().numpy()
    motion_raw = postprocess_motion(motion_raw, cfg)
    return motion_raw


# =============================================================================
# Process one sample
# =============================================================================

def process_one_sample(
    idx, sentence, pkl_paths,
    model, smpl_x, output_dir,
    seq_len, device, cfg,
    img_size=384, gif_fps=8,
    gloss_string=None, render_gif=True,
):
    slug     = sentence_to_slug(sentence)
    save_dir = os.path.join(output_dir, f"{idx:02d}_{slug}")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'sentence.txt'), 'w') as f:
        f.write(sentence + '\n')

    print(f"\n[{idx}] {sentence}")
    print(f"     GT frames: {len(pkl_paths)}  ->  sample to {seq_len}")

    # 1. Generate motion
    motion = generate_motion(model, sentence, seq_len, device, cfg,
                             gloss_string=gloss_string)

    if render_gif:
        gen_params = [split_params(motion[t]) for t in range(motion.shape[0])]
        render_to_gif(gen_params, smpl_x,
                      os.path.join(save_dir, 'generated.gif'),
                      img_size=img_size, gif_fps=gif_fps,
                      flip_coords=False)

    # 2. GT
    gt_indices  = sample_gt_indices(len(pkl_paths), seq_len)
    selected    = [pkl_paths[i] for i in gt_indices]
    gt_params   = load_gt_params(selected)

    if render_gif:
        render_to_gif(gt_params, smpl_x,
                      os.path.join(save_dir, 'gt.gif'),
                      img_size=img_size, gif_fps=gif_fps,
                      flip_coords=False)

    # 3. Return flat sequences for metric computation
    gt_flat = gt_params_to_flat(gt_params)
    return motion, gt_flat


# =============================================================================
# Main
# =============================================================================

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    RENDER_GIF = args.render

    # -- config
    cfg = How2Sign_SMPLX_Config()
    cfg.USE_ROT6D       = args.use_rot6d
    cfg.USE_UPPER_BODY  = args.use_upper_body
    cfg.ROOT_NORMALIZE  = not args.no_root_normalize
    cfg.N_FEATS         = 6 if cfg.USE_ROT6D else 3
    cfg.TARGET_SEQ_LEN  = args.target_seq_len
    cfg.MODEL_VERSION   = args.model_version
    COMPUTE_METRICS     = args.compute_metrics

    # Load config from checkpoint
    ckpt_meta = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_cfg = ckpt_meta.get('config', {})
    cfg.PREDICTION_TYPE = ckpt_cfg.get('PREDICTION_TYPE', 'epsilon')
    cfg.UNCOND_PROB     = ckpt_cfg.get('UNCOND_PROB', 0.1)
    cfg.GUIDANCE_SCALE  = args.guidance_scale if args.guidance_scale is not None \
                          else ckpt_cfg.get('GUIDANCE_SCALE', 3.0)
    cfg.COND_MODE       = ckpt_cfg.get('COND_MODE', 'sentence')
    print(f"Prediction type: {cfg.PREDICTION_TYPE}, Guidance scale: {cfg.GUIDANCE_SCALE}, "
          f"Cond mode: {cfg.COND_MODE}")
    del ckpt_meta

    if args.poses_root is not None:
        cfg.ROOT_DIR = args.poses_root
    if args.xlsx is not None:
        cfg.XLSX_PATH = args.xlsx
    cfg.CAMERA = 'rgb_front'

    # -- output dir
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        logging_dir = checkpoint_dir.replace(
            '/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar',
            '/home/rhong5/research_pro/hand_modeling_pro/aslSentenceAvatar/zlog')
        os.makedirs(logging_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(logging_dir, f"test_{timestamp}", 'gen_images')
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(args.output_dir, f"test_{timestamp}", 'gen_images')

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # -- dataset
    test_dataset = How2SignSMPLXDataset(mode='test', cfg=cfg)
    cfg.INPUT_DIM = test_dataset.input_dim

    # -- model (CFG variants)
    print(f"Building MotionDiffusionModel {cfg.MODEL_VERSION} (CFG)...")
    if cfg.MODEL_VERSION == 'v1':
        model = MotionDiffusionModelV1_CFG(cfg)
    elif cfg.MODEL_VERSION == 'v2':
        model = MotionDiffusionModelV2_CFG(cfg)

    model = load_model_weight(model, args.checkpoint, device)

    # -- SMPL-X renderer (only needed for GIF rendering)
    smpl_x = load_smplx_model(cfg.HUMAN_MODELS_PATH) if RENDER_GIF else None

    # -- sample test sentences
    n_samples = min(args.num_samples, len(test_dataset))
    sample_indices = random.sample(range(len(test_dataset)), n_samples)
    print(f"\nSelected {n_samples} samples from {len(test_dataset)} test clips")

    # ── generate + render + collect ──────────────────────────────────────────
    feats_gen      = []
    feats_real     = []
    gen_seqs_all   = []
    gt_seqs_all    = []
    sentences_all  = []
    dtw_list       = []
    mpjpe_group_list = []
    vel_err_list   = []
    acc_err_list   = []
    jerk_gen_list  = []
    jerk_gt_list   = []

    need_gloss = cfg.COND_MODE in ('gloss', 'sentence_gloss')
    gloss_cache = {}
    if need_gloss:
        print(f"Pre-computing pseudo-gloss strings for {n_samples} test samples...")
        for ds_idx in sample_indices:
            sentence, _ = test_dataset.data_list[ds_idx]
            if sentence not in gloss_cache:
                gloss_cache[sentence] = extract_gloss_string(sentence)

    for rank, ds_idx in enumerate(tqdm(sample_indices, desc="Processing")):
        sentence, pkl_paths = test_dataset.data_list[ds_idx]
        gloss_str = gloss_cache.get(sentence) if need_gloss else None
        gen_seq, gt_seq = process_one_sample(
            idx          = rank,
            sentence     = sentence,
            pkl_paths    = pkl_paths,
            model        = model,
            smpl_x       = smpl_x,
            output_dir   = output_dir,
            seq_len      = cfg.TARGET_SEQ_LEN,
            device       = device,
            cfg          = cfg,
            img_size     = args.img_size,
            gif_fps      = args.gif_fps,
            gloss_string = gloss_str,
            render_gif   = RENDER_GIF,
        )

        if COMPUTE_METRICS:
            feats_gen.append(_pool_sequence(gen_seq))
            feats_real.append(_pool_sequence(gt_seq))
            gen_seqs_all.append(gen_seq)
            gt_seqs_all.append(gt_seq)
            sentences_all.append(sentence)
            dtw_list.append(compute_dtw(gen_seq, gt_seq))
            mpjpe_group_list.append(compute_mpjpe_per_group(gen_seq, gt_seq, n_feats=3))
            vel_err_list.append(compute_velocity_error(gen_seq, gt_seq))
            acc_err_list.append(compute_acceleration_error(gen_seq, gt_seq))
            jerk_gen_list.append(compute_jerk(gen_seq))
            jerk_gt_list.append(compute_jerk(gt_seq))

    # ── compute and save metrics ─────────────────────────────────────────────
    if COMPUTE_METRICS and len(feats_gen) >= 2:
        import json

        fid       = compute_fid(feats_real, feats_gen)
        dtw       = float(np.mean(dtw_list))
        diversity = compute_diversity(gen_seqs_all)

        # Aggregate per-group MPJPE
        group_keys = list(mpjpe_group_list[0].keys())
        mpjpe_agg = {}
        for k in group_keys:
            vals = [m[k] for m in mpjpe_group_list]
            mpjpe_agg[k] = round(float(np.mean(vals)), 4)

        vel_err  = float(np.mean(vel_err_list))
        acc_err  = float(np.mean(acc_err_list))
        jerk_gen = float(np.mean(jerk_gen_list))
        jerk_gt  = float(np.mean(jerk_gt_list))

        metrics = {
            'config': {
                'cond_mode':       cfg.COND_MODE,
                'prediction_type': cfg.PREDICTION_TYPE,
                'guidance_scale':  cfg.GUIDANCE_SCALE,
                'num_samples':     len(feats_gen),
                'seed':            args.seed,
            },
            'distribution': {
                'FID':       round(fid, 4),
                'Diversity': round(diversity, 4),
            },
            'reconstruction': {
                **mpjpe_agg,
                'DTW': round(dtw, 4),
            },
            'dynamics': {
                'velocity_error':     round(vel_err, 4),
                'acceleration_error': round(acc_err, 4),
                'jerk_generated':     round(jerk_gen, 4),
                'jerk_gt':            round(jerk_gt, 4),
            },
        }

        metrics_path = os.path.join(os.path.dirname(output_dir), 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print("\n" + "=" * 60)
        print(f"  cond_mode: {cfg.COND_MODE}")
        print(f"  -- Distribution --")
        print(f"  FID       : {fid:.4f}")
        print(f"  Diversity : {diversity:.4f}")
        print(f"  -- Reconstruction (vs GT) --")
        for k, v in mpjpe_agg.items():
            print(f"  {k:15s}: {v:.4f}")
        print(f"  DTW       : {dtw:.4f}")
        print(f"  -- Dynamics --")
        print(f"  Vel error : {vel_err:.4f}")
        print(f"  Acc error : {acc_err:.4f}")
        print(f"  Jerk (gen): {jerk_gen:.4f}  (GT: {jerk_gt:.4f})")
        print("=" * 60)
        print(f"  Metrics saved -> {metrics_path}")

    elif COMPUTE_METRICS:
        print("WARNING: need >= 2 samples to compute FID, skipping metrics.")

    print(f"\nDone! Results in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",        required=True)
    parser.add_argument("--xlsx",              default=None)
    parser.add_argument("--poses_root",        default=None)

    parser.add_argument("--output_dir",        default=None)
    parser.add_argument("--num_samples",       type=int, default=15)
    parser.add_argument("--target_seq_len",    type=int, default=200)
    parser.add_argument("--img_size",          type=int, default=384)
    parser.add_argument("--gif_fps",           type=int, default=5)
    parser.add_argument("--use_rot6d",         action="store_true")
    parser.add_argument("--use_upper_body",    action="store_true")
    parser.add_argument("--no_root_normalize", action="store_true")
    parser.add_argument("--compute_metrics",   action="store_true")

    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--model_version",     type=str, default='v1', choices=["v1", "v2"])
    parser.add_argument("--guidance_scale",    type=float, default=None,
                        help="Override guidance scale (default: from checkpoint)")
    parser.add_argument("--render",            action="store_true",
                        help="Render GIFs (skip by default for metric-only runs)")
    args = parser.parse_args()
    main(args)
