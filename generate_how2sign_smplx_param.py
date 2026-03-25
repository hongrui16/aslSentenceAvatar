"""
How2Sign Generation Script

For 10 randomly sampled test sentences:
  - Generate motion from sentence text → save as GIF
  - Render GT motion from pkl files    → save as GIF

Output:
    output_dir/
        {i}_{sentence_slug}/
            generated.gif
            gt.gif

Usage:
    python generate_how2sign.py \
        --checkpoint path/to/best_model.pt \
        --xlsx /path/to/how2sign_realigned_test.xlsx \
        --poses_root /path/to/test_poses/poses \
        --human_model_path /path/to/smplx_model
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

from aslAvatarModel_v5 import MotionDiffusionModel
from config import How2Sign_SMPLX_Config
from dataloader.How2SignSMPLXDataset import How2SignSMPLXDataset
from utils.rotation_conversion import postprocess_motion

# ── reuse rendering utilities from generate_smplx_param ─────────────────────
from generate_smplx_param import (
    load_model_weight,
    load_smplx_model,
    params_to_mesh,
    render_smplx_frame,
    save_gif,
    PARAM_SLICES,
)

# =============================================================================
# Helpers
# =============================================================================

def split_params(flat: np.ndarray):
    return {name: flat[s:e].copy() for name, (s, e) in PARAM_SLICES.items()}


def sentence_to_slug(sentence: str, max_len=40) -> str:
    """Convert sentence to a safe directory name."""
    slug = sentence.lower()
    slug = ''.join(c if c.isalnum() or c == ' ' else '' for c in slug)
    slug = '_'.join(slug.split())
    return slug[:max_len]


# =============================================================================
# Fixed params_to_mesh — also returns pelvis joint for stable centering
# =============================================================================

# Rotation matrix: flip Y and Z to convert SMPL-X → pyrender viewing frame
# (rotate 180° around X axis so avatar is upright and facing camera)
_ROT_X_180 = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=np.float32)


def params_to_mesh_fixed(smpl_x, frame_params, flip_coords=False):
    """
    Run SMPL-X forward pass.
    Returns (vertices, faces, pelvis_pos).

    Args:
        flip_coords: apply 180° X-rotation to fix GT raw pkl coordinate mismatch.
                     Generated motion has already been postprocessed, so False.
    """
    def _t(arr):
        return torch.tensor(arr.reshape(1, -1), dtype=torch.float32)

    zeros3 = torch.zeros(1, 3, dtype=torch.float32)
    output = smpl_x.layer['neutral'](
        global_orient=_t(frame_params['smplx_root_pose']),
        body_pose=_t(frame_params['smplx_body_pose']),
        left_hand_pose=_t(frame_params['smplx_lhand_pose']),
        right_hand_pose=_t(frame_params['smplx_rhand_pose']),
        jaw_pose=_t(frame_params['smplx_jaw_pose']),
        leye_pose=zeros3,
        reye_pose=zeros3,
        betas=torch.zeros(1, 10, dtype=torch.float32),
        expression=torch.zeros(1, 10, dtype=torch.float32),
    )

    vertices = output.vertices.cpu().numpy().squeeze(0)  # (N, 3)
    joints   = output.joints.cpu().numpy().squeeze(0)    # (J, 3)
    faces    = smpl_x.face.astype(np.int32)

    if flip_coords:
        vertices = vertices @ _ROT_X_180.T
        pelvis   = (joints[0:1] @ _ROT_X_180.T).squeeze(0)
    else:
        pelvis   = joints[0]

    return vertices, faces, pelvis


# =============================================================================
# Fixed render — centers on pelvis joint (no jitter)
# =============================================================================

def render_frame_fixed(vertices, faces, pelvis, img_w=384, img_h=512, distance=None, debug=False):
    """
    Render SMPL-X mesh, anchored on pelvis joint.

    Args:
        vertices:  (N, 3)
        faces:     (F, 3)
        pelvis:    (3,)  pelvis joint position — used as centering anchor
        distance:  camera distance (pass fixed value from first frame to avoid flickering)
    """
    import trimesh
    import pyrender

    # Center mesh on pelvis (XYZ all axes) → stable across frames
    verts_centered = vertices - pelvis

    fov_y = np.radians(50.0)
    # Only compute distance if not provided (first frame)
    if distance is None:
        extent   = (vertices.max(axis=0) - vertices.min(axis=0)).max()
        distance = (extent / 2.0) / np.tan(fov_y / 2.0) * 1.4

    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1,
        roughnessFactor=0.85,
        alphaMode="OPAQUE",
        baseColorFactor=(0.6, 0.9, 0.65, 1.0),
    )

    if debug:
        print(f"  render: pelvis={pelvis}, distance={distance:.4f}")

    body_trimesh = trimesh.Trimesh(verts_centered, faces, process=False)
    body_mesh    = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    camera   = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=img_w / img_h)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = distance

    light      = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_pose = cam_pose.copy()

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0], ambient_light=(0.4, 0.4, 0.4))
    scene.add(body_mesh, "mesh")
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=light_pose)

    r = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h, point_size=1.0)
    color_img, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()

    return color_img[:, :, :3]  # (H, W, 3) uint8


# =============================================================================
# Generation
# =============================================================================

@torch.no_grad()
def generate_motion(model, sentence: str, seq_len: int, device: str, cfg):
    """Generate (T, 159) axis-angle params from sentence text."""
    motion = model.generate([sentence], seq_len=seq_len, device=device)
    motion_raw = motion.squeeze(0).cpu().numpy()          # (T, input_dim)
    motion_raw = postprocess_motion(motion_raw, cfg)      # (T, 159)
    return motion_raw


# =============================================================================
# GT loading from pkl files
# =============================================================================

def load_gt_params(pkl_paths, zero_root=True):
    """
    Load GT SMPL-X params from per-frame pkl files.

    Args:
        zero_root: if True, zero out smplx_root_pose (removes global
                   rotation / sitting tilt so avatar stands upright)
    """
    frame_params = []
    for p in pkl_paths:
        with open(p, 'rb') as f:
            d = pickle.load(f)
        params = {
            'smplx_root_pose':  np.zeros(3, dtype=np.float32) if zero_root
                                else np.array(d['smplx_root_pose']).reshape(3,).astype(np.float32),
            'smplx_body_pose':  np.array(d['smplx_body_pose']).reshape(63,).astype(np.float32),
            'smplx_lhand_pose': np.array(d['smplx_lhand_pose']).reshape(45,).astype(np.float32),
            'smplx_rhand_pose': np.array(d['smplx_rhand_pose']).reshape(45,).astype(np.float32),
            'smplx_jaw_pose':   np.array(d['smplx_jaw_pose']).reshape(3,).astype(np.float32),
        }
        frame_params.append(params)
    return frame_params


def sample_gt_indices(total_len, target_len):
    """Uniform linspace sampling for GT (deterministic)."""
    if total_len <= target_len:
        return list(range(total_len))
    return np.round(np.linspace(0, total_len - 1, target_len)).astype(int).tolist()


# =============================================================================
# Render one motion sequence → GIF
# =============================================================================

# =============================================================================
# Render one motion sequence → GIF
# =============================================================================

def render_to_gif(frame_params_list, smpl_x, gif_path, img_size=384, gif_fps=8, flip_coords=False):
    """
    Render a list of per-frame param dicts → animated GIF.
    Uses pelvis-anchored centering and fixed camera distance for stable rendering.

    Args:
        flip_coords: apply 180 X-rotation. Do NOT use for GT with zero_root=True,
                     SMPL-X canonical pose already renders correctly in pyrender.
    """
    img_w = img_size
    img_h = int(img_size * 512 / 384)   # portrait ratio

    frames = []
    fixed_distance = None   # computed from first frame, reused for all frames

    for t, params in enumerate(frame_params_list):
        try:
            vertices, faces, pelvis = params_to_mesh_fixed(smpl_x, params, flip_coords=flip_coords)

            # Compute camera distance from first frame only -> no zoom flicker
            if fixed_distance is None:
                extent = (vertices.max(axis=0) - vertices.min(axis=0)).max()
                fov_y  = np.radians(50.0)
                fixed_distance = (extent / 2.0) / np.tan(fov_y / 2.0) * 1.4

            img = render_frame_fixed(vertices, faces, pelvis,
                                     img_w=img_w, img_h=img_h,
                                     distance=fixed_distance,
                                     debug=(t == 0))
            frames.append(img)
        except Exception as e:
            print(f"  Render error frame {t}: {e}")

    if frames:
        save_gif(frames, gif_path, fps=gif_fps)
    else:
        print(f"  WARNING: no frames rendered for {gif_path}")


# =============================================================================
# Process one sample
# =============================================================================

def process_one_sample(
    idx, sentence, pkl_paths,
    model, smpl_x, output_dir,
    seq_len, device, cfg,
    img_size=384, gif_fps=8,
):
    slug     = sentence_to_slug(sentence)
    save_dir = os.path.join(output_dir, f"{idx:02d}_{slug}")
    os.makedirs(save_dir, exist_ok=True)

    # save sentence text for reference
    with open(os.path.join(save_dir, 'sentence.txt'), 'w') as f:
        f.write(sentence + '\n')

    print(f"\n[{idx}] {sentence}")
    print(f"     GT frames: {len(pkl_paths)}  →  sample to {seq_len}")

    # ── 1. Generated GIF ──────────────────────────────────────────────────────
    print(f"     Generating motion...")
    motion = generate_motion(model, sentence, seq_len, device, cfg)  # (T, 159)
    gen_params = [split_params(motion[t]) for t in range(motion.shape[0])]
    render_to_gif(gen_params, smpl_x,
                  os.path.join(save_dir, 'generated.gif'),
                  img_size=img_size, gif_fps=gif_fps,
                  flip_coords=False)   # postprocessed → no flip needed

    # ── 2. GT GIF ─────────────────────────────────────────────────────────────
    print(f"     Rendering GT...")
    gt_indices  = sample_gt_indices(len(pkl_paths), seq_len)
    selected    = [pkl_paths[i] for i in gt_indices]
    gt_params   = load_gt_params(selected)
    render_to_gif(gt_params, smpl_x,
                  os.path.join(save_dir, 'gt.gif'),
                  img_size=img_size, gif_fps=gif_fps,
                  flip_coords=False)  # zero_root=True → SMPL-X canonical, no flip needed


# =============================================================================
# Main
# =============================================================================

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # ── output dir ────────────────────────────────────────────────────────────
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(args.checkpoint),
        f"how2sign_test_{timestamp}"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # ── config ────────────────────────────────────────────────────────────────
    cfg = How2Sign_SMPLX_Config()
    cfg.USE_ROT6D       = args.use_rot6d
    cfg.USE_UPPER_BODY  = args.use_upper_body
    cfg.ROOT_NORMALIZE  = not args.no_root_normalize
    cfg.N_FEATS         = 6 if cfg.USE_ROT6D else 3
    cfg.TARGET_SEQ_LEN  = args.target_seq_len
    
    if not args.poses_root is None:
        cfg.ROOT_DIR        = args.poses_root 
    if not args.xlsx is None:            
        cfg.XLSX_PATH       = args.xlsx 
    cfg.CAMERA          = 'rgb_front'


    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)        
        logging_dir = checkpoint_dir.replace('/scratch/rhong5/weights/temp_training_weights/aslAvatar', '/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/zlog')
        os.makedirs(logging_dir, exist_ok=True)
        
        # Output dir
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # test_log_dir = os.path.join(logging_dir, f"test_{timestamp}")
        output_dir = os.path.join(logging_dir, f"test_{timestamp}", 'gen_images')        
    else:
        logging_dir = args.output_dir
        # test_log_dir = os.path.join(logging_dir, f"test_{timestamp}")
        output_dir = os.path.join(logging_dir, f"test_{timestamp}", 'gen_images')        

    os.makedirs(output_dir, exist_ok=True)
    
    # ── dataset ───────────────────────────────────────────────────────────────
    test_dataset = How2SignSMPLXDataset(mode='test', cfg=cfg)
    cfg.INPUT_DIM = test_dataset.input_dim

    # ── model ─────────────────────────────────────────────────────────────────
    model = MotionDiffusionModel(cfg)
    model = load_model_weight(model, args.checkpoint, device)

    # ── SMPL-X renderer ───────────────────────────────────────────────────────
    smpl_x = load_smplx_model(cfg.HUMAN_MODELS_PATH)

    # ── sample 10 test sentences ──────────────────────────────────────────────
    n_samples = min(args.num_samples, len(test_dataset))
    sample_indices = random.sample(range(len(test_dataset)), n_samples)
    print(f"\nSelected {n_samples} samples from {len(test_dataset)} test clips")

    # ── generate + render ─────────────────────────────────────────────────────
    for rank, ds_idx in enumerate(tqdm(sample_indices, desc="Processing")):
        sentence, pkl_paths = test_dataset.data_list[ds_idx]
        process_one_sample(
            idx        = rank,
            sentence   = sentence,
            pkl_paths  = pkl_paths,
            model      = model,
            smpl_x     = smpl_x,
            output_dir = output_dir,
            seq_len    = cfg.TARGET_SEQ_LEN,
            device     = device,
            cfg        = cfg,
            img_size   = args.img_size,
            gif_fps    = args.gif_fps,
        )

    print(f"\nDone! Results in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",        required=True,  help="Path to trained .pt checkpoint")
    parser.add_argument("--xlsx",              default=None,  help="how2sign_realigned_test.xlsx")
    parser.add_argument("--poses_root",        default=None,  help="test_poses/poses/ directory")

    parser.add_argument("--output_dir",        default=None)
    parser.add_argument("--num_samples",       type=int, default=15)
    parser.add_argument("--target_seq_len",    type=int, default=200)
    parser.add_argument("--img_size",          type=int, default=384)
    parser.add_argument("--gif_fps",           type=int, default=5)
    parser.add_argument("--use_rot6d",         action="store_true")
    parser.add_argument("--use_upper_body",    action="store_true")
    parser.add_argument("--no_root_normalize", action="store_true")
    parser.add_argument("--seed",              type=int, default=42)
    args = parser.parse_args()
    main(args)