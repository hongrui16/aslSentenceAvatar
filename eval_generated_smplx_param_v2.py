"""
eval_generated_smplx_param_v2.py
=================================
Changes from v1:
  [Fix-1] Rendering: roughnessFactor 0.4→0.85 (matte), reduced light intensity,
          lower emissive, extra soft fill light — eliminates chest specular hotspot.
  [Fix-2] Camera: distance multiplier 1.4→1.1 so body fills the frame.
          Camera IS already truly frontal (confirmed). The "tilted" appearance
          in prior renders comes from the body's root_pose, not the camera.
  [Fix-3] Per-gloss cosine-similarity + L2 metrics; prints top-10 by cos-sim;
          saves per_gloss_metrics_*.csv alongside the aggregate YAML.
  [Fix-4] --render_comparison flag: saves gt_{gloss}_{t:06d}.png,
          ours_{gloss}_{t:06d}.png, gt_{gloss}.gif, ours_{gloss}.gif
          for paper figures.

Usage:
    # Generate + render both GT and ours for comparison
    python eval_generated_smplx_param_v2.py \
        --checkpoint path/to/best_model.pt \
        --render_mesh --gif --render_comparison

    # Full pipeline
    python eval_generated_smplx_param_v2.py \
        --checkpoint path/to/best_model.pt \
        --render_mesh --gif --evaluate --render_comparison
"""
import os
import csv
import argparse
import random
import yaml
from typing import List, Dict, Optional
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from aslAvatarModel import ASLAvatarModel
from aslAvatarModel_v2 import ASLAvatarModelV2
from aslAvatarModel_v4 import ASLAvatarModelV4
from aslAvatarModel_v5 import MotionDiffusionModel

from config import SignBank_SMPLX_Config
from config import WLASL_SMPLX_Config
from config import ASL3DWord_SMPLX_Config

from dataloader.SignBankSMPLXDataset import SignBankSMPLXDataset
from dataloader.WLASLSMPLXDataset import WLASLSMPLXDataset
from dataloader.WLASLSMPLXDatasetV2 import WLASLSMPLXDatasetV2
from dataloader.ASL3DWordDataset import ASL3DWordDataset

from utils.rotation_conversion import postprocess_motion


# =============================================================================
# SMPL-X Parameter Layout (159 dims)
#   root_pose(3) + body_pose(63) + lhand(45) + rhand(45) + jaw(3)
# =============================================================================

PARAM_SLICES = {
    'smplx_root_pose':  (0,   3),
    'smplx_body_pose':  (3,   66),
    'smplx_lhand_pose': (66,  111),
    'smplx_rhand_pose': (111, 156),
    'smplx_jaw_pose':   (156, 159),
}


def split_params(flat: np.ndarray) -> Dict[str, np.ndarray]:
    """Split (159,) vector into named SMPL-X components."""
    return {name: flat[s:e].copy() for name, (s, e) in PARAM_SLICES.items()}


# =============================================================================
# Model Loading
# =============================================================================

def load_model_weight(model, checkpoint_path: str, device: str = 'cuda'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = ckpt.get('model_state_dict', ckpt)
    cur = model.state_dict()
    loaded = 0
    for k in model_state:
        if k in cur and cur[k].shape == model_state[k].shape:
            cur[k] = model_state[k]
            loaded += 1
    model.load_state_dict(cur, strict=False)
    model.to(device).eval()
    print(f"Loaded: {checkpoint_path}  (epoch {ckpt.get('epoch','?')}, {loaded} keys)")
    return model, ckpt


# =============================================================================
# Generation
# =============================================================================

@torch.no_grad()
def generate_from_gloss(model, gloss: str, seq_len: int, device: str, cfg=None):
    """Generate (T, 159) SMPL-X axis-angle params from gloss."""
    motion = model.generate([gloss], seq_len=seq_len, device=device)

    diff = (motion[0, 1:] - motion[0, :-1]).abs().mean()
    print(f"Generated frame diff: {diff:.8f}")

    motion_raw = motion.squeeze(0).cpu().numpy()  # (T, input_dim)

    if cfg is not None:
        motion_raw = postprocess_motion(motion_raw, cfg)  # (T, 159)

    return motion_raw


@torch.no_grad()
def generate_motion_raw(model, gloss: str, seq_len: int, device: str):
    """
    Generate motion in model's native representation (no postprocessing).
    GT and Gen are directly comparable for metrics.
    """
    motion = model.generate([gloss], seq_len=seq_len, device=device)
    return motion.squeeze(0).cpu()  # (T, input_dim)


# =============================================================================
# Save .npz
# =============================================================================

def save_frame_npz(frame_params: Dict[str, np.ndarray], save_path: str):
    dump = {
        'smplx_root_pose':  frame_params['smplx_root_pose'].reshape(3,).astype(np.float32),
        'smplx_body_pose':  frame_params['smplx_body_pose'].reshape(21, 3).astype(np.float32),
        'smplx_lhand_pose': frame_params['smplx_lhand_pose'].reshape(15, 3).astype(np.float32),
        'smplx_rhand_pose': frame_params['smplx_rhand_pose'].reshape(15, 3).astype(np.float32),
        'smplx_jaw_pose':   frame_params['smplx_jaw_pose'].reshape(3,).astype(np.float32),
        'smplx_shape':      np.zeros(10, dtype=np.float32),
        'smplx_expr':       np.zeros(10, dtype=np.float32),
        'cam_trans':         np.zeros(3, dtype=np.float32),
    }
    np.savez(save_path, **dump)


# =============================================================================
# [Fix-1 & Fix-2] SMPL-X Mesh Rendering  — improved material + camera
# =============================================================================

def load_smplx_model(human_model_path):
    from human_models.human_models import SMPLX
    smpl_x = SMPLX(human_model_path)
    print(f"Loaded SMPL-X model from {human_model_path}")
    return smpl_x


def params_to_mesh(smpl_x, frame_params):
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
    vertices = output.vertices.cpu().numpy().squeeze(0)
    faces    = smpl_x.face.astype(np.int32)
    # joint 0 = pelvis — stable anchor for camera centering (unaffected by hand pose)
    pelvis   = output.joints.cpu().numpy()[0, 0]   # (3,)
    return vertices, faces, pelvis


def render_smplx_frame(vertices, faces, pelvis, img_w=384, img_h=512,
                       debug=False, gloss=""):
    """
    Render SMPL-X mesh to a (H, W, 3) uint8 image.

    Fix-1: Material changes
      - roughnessFactor: 0.4 → 0.85  (matte surface, eliminates chest hotspot)
      - metallicFactor:  0.1 → 0.0   (fully dielectric, no metallic sheen)
      - emissiveFactor:  (0.15,0.2,0.15) → (0.04,0.06,0.04) (minimal self-glow)
      - Main light intensity: 3.0 → 2.0
      - Ambient: 0.4 → 0.30
      - Added soft fill light from upper-left for depth without extra glare

    Fix-2: Camera
      - distance multiplier: 1.4 → 1.1  (body fills frame better)
      - Camera pose is identity rotation + Z translation → truly frontal view.
        The "top-down" appearance in previous renders came from the body's
        root_pose leaning the figure backward, NOT from the camera angle.
    """
    import trimesh
    import pyrender

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    extent = (vmax - vmin).max()
    # Fix-5: anchor to pelvis so body stays still when hands move
    center = pelvis.copy()   # all 3 axes anchored to pelvis — hands cannot shift the view

    # [Fix-2] Tighter FOV + reduced distance so body fills the frame
    fov_y = np.radians(45.0)                                       # was 50°
    distance = (extent / 2.0) / np.tan(fov_y / 2.0) * 1.1        # was 1.4

    verts_centered = vertices - center

    # [Fix-1] More matte/diffuse material — kills specular hotspot on chest
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,                      # was 0.1
        roughnessFactor=0.85,                    # was 0.4  ← KEY FIX
        alphaMode="OPAQUE",
        emissiveFactor=(0.04, 0.06, 0.04),       # was (0.15, 0.2, 0.15)
        baseColorFactor=(0.62, 0.88, 0.67, 1.0),
    )

    body_trimesh = trimesh.Trimesh(verts_centered, faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    # [Fix-2] Camera: at (0, 0, +distance) with identity rotation.
    #   In pyrender the camera looks in its local -Z direction.
    #   With identity rotation, local -Z = world -Z → camera looks toward
    #   the body at the origin.  This is a TRUE FRONTAL VIEW.
    camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=img_w / img_h)
    cam_pose = np.eye(4)
    cam_pose[2, 3] = distance

    # [Fix-1] Key light from slightly above (25° pitch-down) — avoids flat flash look
    key_angle = np.radians(25.0)
    key_pose = np.eye(4)
    key_pose[:3, :3] = np.array([
        [1.0,  0.0,              0.0            ],
        [0.0,  np.cos(key_angle), -np.sin(key_angle)],
        [0.0,  np.sin(key_angle),  np.cos(key_angle)],
    ])
    key_pose[2, 3] = distance
    light_key = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)  # was 3.0

    # [Fix-1] Soft fill light from upper-left (45° yaw, weak) for gentle depth
    fill_angle = np.radians(45.0)
    fill_pose = np.eye(4)
    fill_pose[:3, :3] = np.array([
        [ np.cos(fill_angle), 0.0, np.sin(fill_angle)],
        [ 0.0,                1.0, 0.0               ],
        [-np.sin(fill_angle), 0.0, np.cos(fill_angle)],
    ])
    fill_pose[2, 3] = distance
    light_fill = pyrender.DirectionalLight(color=np.ones(3), intensity=0.7)

    # [Fix-1] Reduced ambient
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0],
                           ambient_light=(0.30, 0.30, 0.30))  # was 0.4
    scene.add(body_mesh, "mesh")
    scene.add(camera,     pose=cam_pose)
    scene.add(light_key,  pose=key_pose)
    scene.add(light_fill, pose=fill_pose)

    r = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h,
                                   point_size=1.0)
    color_img, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()

    return color_img[:, :, :3]


# =============================================================================
# [Fix-4] GT lookup helper
# =============================================================================

def build_gloss_gt_lookup(dataset, cfg) -> Dict[str, np.ndarray]:
    """
    Scan dataset and return {gloss: motion_axis_angle (T, 159)}.
    Keeps the FIRST sample found per gloss (sufficient for paper figures).
    Applies postprocess_motion to convert from model representation.
    """
    lookup: Dict[str, np.ndarray] = {}
    print("Building GT lookup for comparison rendering ...")
    for i in tqdm(range(len(dataset)), desc="GT lookup"):
        item = dataset[i]
        # Dataset items: (seq_tensor, gloss_str, gloss_with_attrs, ...)
        seq   = item[0]
        gloss = item[1] if isinstance(item, (tuple, list)) else item.get('gloss', '')

        if gloss in lookup:
            continue  # keep first occurrence only

        motion = seq.numpy() if isinstance(seq, torch.Tensor) else np.array(seq)
        if cfg is not None:
            motion = postprocess_motion(motion, cfg)   # → (T, 159) axis-angle
        lookup[gloss] = motion

    print(f"  GT lookup: {len(lookup)} unique glosses")
    return lookup


# =============================================================================
# Process one gloss (generate + optional render/save)
# =============================================================================

def process_a_gloss(model, gloss, output_dir, seq_len, device,
                    smpl_x=None, img_w=384, img_h=512, dump_param=False,
                    cfg=None, make_gif=True, gif_fps=8, dataset=None,
                    gt_motion: Optional[np.ndarray] = None,   # [Fix-4]
                    render_comparison: bool = False):          # [Fix-4]
    """
    Generate motion for one gloss, optionally render meshes / GIFs.

    If render_comparison=True and gt_motion is provided:
      - saves ours_{gloss}_{t:06d}.png  and  gt_{gloss}_{t:06d}.png
      - saves ours_{gloss}.gif          and  gt_{gloss}.gif
    Otherwise falls back to original naming (no prefix).
    """
    if cfg.USE_PHONO_ATTRIBUTE:
        gloss_with_attributes = dataset._gloss_with_phono(gloss)
    else:
        gloss_with_attributes = gloss

    motion = generate_from_gloss(model, gloss_with_attributes, seq_len, device, cfg)
    T = motion.shape[0]

    gloss_dir = os.path.join(output_dir, gloss)
    os.makedirs(gloss_dir, exist_ok=True)

    render_dir = None
    if smpl_x is not None:
        render_dir = os.path.join(gloss_dir, 'renders')
        os.makedirs(render_dir, exist_ok=True)

    # ── Render generated frames ──────────────────────────────────────────────
    ours_gif_frames = []
    for t in range(T):
        params = split_params(motion[t])

        if dump_param:
            npz_path = os.path.join(gloss_dir, f"{gloss}_{t:06d}_p0.npz")
            save_frame_npz(params, npz_path)

        if smpl_x is not None:
            try:
                vertices, faces, pelvis = params_to_mesh(smpl_x, params)
                img = render_smplx_frame(vertices, faces, pelvis,
                                         img_w=img_w, img_h=img_h,
                                         debug=(t == 0), gloss=gloss)
                ours_gif_frames.append(img)

                if render_comparison:
                    # [Fix-4] Save individual frames with "ours_" prefix for paper
                    _save_png(img, os.path.join(render_dir,
                                                f"ours_{gloss}_{t:06d}.png"))
            except Exception as e:
                import traceback
                print(f"  Render error frame {t}: {e}")
                traceback.print_exc()

    # ── GIF for generated ────────────────────────────────────────────────────
    if make_gif and ours_gif_frames:
        if render_comparison:
            gif_name = f"ours_{gloss}.gif"
        else:
            gif_name = f"{gloss}.gif"
        save_gif(ours_gif_frames, os.path.join(gloss_dir, gif_name), fps=gif_fps)

    # ── [Fix-4] Render GT frames ─────────────────────────────────────────────
    if render_comparison and smpl_x is not None and gt_motion is not None:
        T_gt = gt_motion.shape[0]
        gt_gif_frames = []
        for t in range(T_gt):
            params = split_params(gt_motion[t])
            try:
                vertices, faces, pelvis = params_to_mesh(smpl_x, params)
                img = render_smplx_frame(vertices, faces, pelvis,
                                         img_w=img_w, img_h=img_h)
                gt_gif_frames.append(img)
                _save_png(img, os.path.join(render_dir,
                                             f"gt_{gloss}_{t:06d}.png"))
            except Exception as e:
                print(f"  GT render error frame {t}: {e}")

        if make_gif and gt_gif_frames:
            save_gif(gt_gif_frames,
                     os.path.join(gloss_dir, f"gt_{gloss}.gif"),
                     fps=gif_fps)
    elif render_comparison and gt_motion is None:
        print(f"  WARNING: render_comparison=True but no GT motion found for '{gloss}'")

    return T


def _save_png(img_array: np.ndarray, path: str):
    """Save uint8 (H, W, 3) array as PNG."""
    from PIL import Image
    Image.fromarray(img_array).save(path)


# =============================================================================
# Helpers
# =============================================================================

def get_glosses_from_dataset(root_dir: str, num_glosses: Optional[int] = None) -> List[str]:
    if not os.path.isdir(root_dir):
        print(f"WARNING: not found: {root_dir}")
        return []
    glosses = sorted(d for d in os.listdir(root_dir)
                     if os.path.isdir(os.path.join(root_dir, d)))
    if num_glosses and num_glosses < len(glosses):
        glosses = random.sample(glosses, num_glosses)
    return glosses


def save_gif(frames, gif_path, fps=10):
    import imageio
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"  GIF saved: {gif_path} ({len(frames)} frames, {fps} fps)")


def make_collate_fn(gloss_to_idx):
    def collate_fn(batch_list):
        seqs   = [item[0] for item in batch_list]
        glosses = [item[1] for item in batch_list]
        gloss_with_attributes = [item[2] for item in batch_list]
        lengths = len(seqs)
        x = torch.stack(seqs, dim=0)
        y = torch.tensor([gloss_to_idx[g] for g in glosses], dtype=torch.long)
        return {"x": x, "y": y,
                "lengths": torch.tensor(lengths, dtype=torch.long),
                "glosses": glosses}
    return collate_fn


# =============================================================================
# Evaluation: collect GT and Gen motions, run metrics
# =============================================================================

@torch.no_grad()
def collect_gt_motions(dataset, batch_size=32):
    from torch.utils.data import DataLoader
    collate_fn = make_collate_fn(dataset.gloss_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0,
                        collate_fn=collate_fn)
    all_x, all_y = [], []
    for batch in tqdm(loader, desc="Collecting GT"):
        all_x.append(batch["x"])
        all_y.append(batch["y"])
    return torch.cat(all_x, 0), torch.cat(all_y, 0)


@torch.no_grad()
def collect_gen_motions(model, dataset, cfg, device, batch_size=32):
    from torch.utils.data import DataLoader
    collate_fn = make_collate_fn(dataset.gloss_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0,
                        collate_fn=collate_fn)
    all_x, all_y = [], []
    for batch in tqdm(loader, desc="Generating"):
        y      = batch["y"]
        glosses = batch["glosses"]
        B = y.shape[0]
        gen_list = []
        for i in range(B):
            motion_t = generate_motion_raw(
                model, glosses[i], cfg.TARGET_SEQ_LEN, device)
            gen_list.append(motion_t)
        all_x.append(torch.stack(gen_list, 0))
        all_y.append(y)
    return torch.cat(all_x, 0), torch.cat(all_y, 0)


# =============================================================================
# [Fix-3] Per-gloss cosine similarity
# =============================================================================

def compute_per_gloss_similarity(gt_x, gt_y, gen_x, gen_y,
                                  gloss_name_list) -> Dict[str, dict]:
    """
    For each gloss compute:
      - mean_cos_sim : mean cosine similarity between all Gen and all GT motions
                       (flattened T*D vectors, higher = more similar to GT)
      - mean_l2_dist : mean L2 distance from each Gen to the GT mean
      - n_gt / n_gen : sample counts

    Returns:
        {gloss_name: {'mean_cos_sim', 'mean_l2_dist', 'n_gt', 'n_gen'}}
    """
    import torch.nn.functional as F

    per_gloss: Dict[str, dict] = {}

    for cls_idx, gloss in enumerate(gloss_name_list):
        gt_mask  = (gt_y  == cls_idx)
        gen_mask = (gen_y == cls_idx)
        n_gt  = int(gt_mask.sum())
        n_gen = int(gen_mask.sum())

        if n_gt == 0 or n_gen == 0:
            continue

        gt_vecs  = gt_x[gt_mask].reshape(n_gt,  -1).float()
        gen_vecs = gen_x[gen_mask].reshape(n_gen, -1).float()

        # Cosine similarity: (n_gen, n_gt) → mean scalar
        gt_norm  = F.normalize(gt_vecs,  dim=1)
        gen_norm = F.normalize(gen_vecs, dim=1)
        sim_mat  = gen_norm @ gt_norm.T       # (n_gen, n_gt)
        mean_cos = float(sim_mat.mean())

        # L2 distance to GT centroid
        gt_mean  = gt_vecs.mean(dim=0, keepdim=True)
        l2_dists = torch.norm(gen_vecs - gt_mean, dim=1)
        mean_l2  = float(l2_dists.mean())

        per_gloss[gloss] = {
            'mean_cos_sim': round(mean_cos, 6),
            'mean_l2_dist': round(mean_l2, 6),
            'n_gt':  n_gt,
            'n_gen': n_gen,
        }

    return per_gloss


def print_top_k_glosses(per_gloss: Dict[str, dict], k: int = 10):
    """Print top-k glosses ranked by cosine similarity to GT."""
    ranked = sorted(per_gloss.items(),
                    key=lambda x: x[1]['mean_cos_sim'], reverse=True)
    print(f"\n{'='*65}")
    print(f"  Top-{k} Glosses by Cosine Similarity to GT")
    print(f"{'='*65}")
    print(f"  {'Rank':<5} {'Gloss':<22} {'cos_sim':>8} {'l2_dist':>9} "
          f"{'n_gt':>6} {'n_gen':>6}")
    print(f"  {'-'*5} {'-'*22} {'-'*8} {'-'*9} {'-'*6} {'-'*6}")
    for rank, (gloss, m) in enumerate(ranked[:k], 1):
        print(f"  {rank:<5} {gloss:<22} {m['mean_cos_sim']:>8.4f} "
              f"{m['mean_l2_dist']:>9.4f} {m['n_gt']:>6} {m['n_gen']:>6}")
    print()
    return [g for g, _ in ranked[:k]]


def save_per_gloss_csv(per_gloss: Dict[str, dict], save_path: str):
    """Save per-gloss metrics to CSV, sorted by descending cos-sim."""
    ranked = sorted(per_gloss.items(),
                    key=lambda x: x[1]['mean_cos_sim'], reverse=True)
    fieldnames = ['rank', 'gloss', 'mean_cos_sim', 'mean_l2_dist', 'n_gt', 'n_gen']
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, (gloss, m) in enumerate(ranked, 1):
            writer.writerow({'rank': rank, 'gloss': gloss, **m})
    print(f"  Per-gloss CSV saved: {save_path}")


# =============================================================================
# Full evaluation pipeline
# =============================================================================

def run_evaluation(model, cfg, device, train_dataset, test_dataset,
                   batch_size=32, seed=42):
    from model_free_metrics import ModelFreeEvaluator
    from model_based_metrics import ModelBasedEvaluator

    n_feats     = cfg.N_FEATS
    num_classes = cfg.NUM_CLASSES

    # Align label indices between train and test
    train_g2i = train_dataset.gloss_to_idx
    test_g2i  = test_dataset.gloss_to_idx

    test_to_train_label = {}
    missing_glosses = set()
    for gloss, test_idx in test_g2i.items():
        if gloss in train_g2i:
            test_to_train_label[test_idx] = train_g2i[gloss]
        else:
            missing_glosses.add(gloss)
    if missing_glosses:
        print(f"  WARNING: {len(missing_glosses)} test glosses not in train")

    # Collect GT train
    print("\n>>> Collecting GT train motions ...")
    gt_train_x, gt_train_y = collect_gt_motions(train_dataset, batch_size)

    # Collect GT test
    print("\n>>> Collecting GT test motions ...")
    gt_test_x, gt_test_y_raw = collect_gt_motions(test_dataset, batch_size)

    def _remap_labels(y_raw, mask_ref=None):
        y = y_raw.clone()
        valid = torch.ones(len(y), dtype=torch.bool)
        for i in range(len(y)):
            old = int(y_raw[i])
            if old in test_to_train_label:
                y[i] = test_to_train_label[old]
            else:
                valid[i] = False
        return y, valid

    gt_test_y, valid_mask = _remap_labels(gt_test_y_raw)
    if not valid_mask.all():
        n_rm = int((~valid_mask).sum())
        print(f"    Removing {n_rm} test samples with unknown glosses")
        gt_test_x = gt_test_x[valid_mask]
        gt_test_y = gt_test_y[valid_mask]

    # Generate test set
    print("\n>>> Generating motions for test set ...")
    gen_test_x, gen_test_y_raw = collect_gen_motions(
        model, test_dataset, cfg, device, batch_size)

    gen_test_y, valid_mask_gen = _remap_labels(gen_test_y_raw)
    if not valid_mask_gen.all():
        gen_test_x = gen_test_x[valid_mask_gen]
        gen_test_y = gen_test_y[valid_mask_gen]

    # [Fix-3] Per-gloss similarity
    print("\n>>> Computing per-gloss cosine similarity ...")
    per_gloss_sim = compute_per_gloss_similarity(
        gt_test_x, gt_test_y, gen_test_x, gen_test_y,
        cfg.GLOSS_NAME_LIST)
    top10 = print_top_k_glosses(per_gloss_sim, k=10)
    print(f"  Top-10 gloss names: {top10}")

    # Model-free metrics
    print("\n" + "="*65)
    print("  Running Model-Free Evaluation")
    print("="*65)
    mf_eval = ModelFreeEvaluator(
        n_feats=n_feats, num_classes=num_classes, seed=seed)
    mf_results = mf_eval.evaluate(gt_test_x, gt_test_y,
                                   gen_test_x, gen_test_y)
    mf_eval.print_results(mf_results, title="test")

    # Model-based metrics
    print("\n" + "="*65)
    print("  Running Model-Based Evaluation")
    print("="*65)
    mb_eval = ModelBasedEvaluator(
        n_feats=n_feats, num_classes=num_classes,
        device=device, seed=seed)
    mb_eval.train(gt_train_x, gt_train_y)
    mb_results = mb_eval.evaluate(gt_test_x, gt_test_y,
                                   gen_test_x, gen_test_y)
    mb_eval.print_results(mb_results, title="test")

    return {
        "model_free":    mf_results,
        "model_based":   mb_results,
        "per_gloss_sim": per_gloss_sim,   # [Fix-3]
        "top10_glosses": top10,            # [Fix-3]
    }


# =============================================================================
# Dataset / Model helpers
# =============================================================================

def load_dataset(dataset_name, cfg, mode='train', logger=None):
    if dataset_name == "SignBank_SMPLX":
        return SignBankSMPLXDataset(mode=mode, cfg=cfg)
    elif dataset_name == "WLASL_SMPLX":
        ver = getattr(cfg, 'DATASET_VERSION', 'v1').lower()
        if ver == 'v2':
            return WLASLSMPLXDatasetV2(mode=mode, cfg=cfg, logger=logger)
        else:
            return WLASLSMPLXDataset(mode=mode, cfg=cfg, logger=logger)
    elif dataset_name == "ASL3DWord":
        return ASL3DWordDataset(mode=mode, cfg=cfg, logger=logger)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_model(cfg):
    ver = cfg.MODEL_VERSION.lower()
    if ver == 'v1':
        return ASLAvatarModel(cfg)
    elif ver == 'v2':
        return ASLAvatarModelV2(cfg)
    elif ver == 'v4':
        return ASLAvatarModelV4(cfg)
    elif ver == 'v5':
        return MotionDiffusionModel(cfg)
    else:
        raise ValueError(f"Unknown model version: {cfg.MODEL_VERSION}")


# =============================================================================
# Main
# =============================================================================

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Config
    config_map = {
        "SignBank_SMPLX": SignBank_SMPLX_Config,
        "WLASL_SMPLX":   WLASL_SMPLX_Config,
        "ASL3DWord":      ASL3DWord_SMPLX_Config,
    }
    cfg = config_map[args.dataset_name]()
    cfg.USE_UPPER_BODY      = args.use_upper_body
    cfg.USE_ROT6D           = args.use_rot6d
    cfg.USE_MINI_DATASET    = args.use_mini_dataset
    cfg.ROOT_NORMALIZE      = not args.no_root_normalize
    cfg.N_FEATS             = 6 if cfg.USE_ROT6D else 3
    cfg.USE_PHONO_ATTRIBUTE = args.use_phono_attribute
    cfg.TEXT_ENCODER_TYPE   = args.text_encoder_type

    # Dataset
    train_dataset = load_dataset(args.dataset_name, cfg, mode='train')
    test_dataset  = None
    if args.evaluate or args.dataset_name == "ASL3DWord":
        try:
            test_dataset = load_dataset(args.dataset_name, cfg, mode='test')
        except Exception as e:
            print(f"WARNING: Could not load test dataset: {e}")

    cfg.INPUT_DIM       = train_dataset.input_dim
    cfg.GLOSS_NAME_LIST = train_dataset.gloss_name_list
    cfg.NUM_CLASSES     = len(cfg.GLOSS_NAME_LIST)

    print(f"Dataset: {args.dataset_name}, {cfg.NUM_CLASSES} classes, "
          f"n_feats={cfg.N_FEATS}")
    print(f"  train: {len(train_dataset)} samples")
    if test_dataset:
        print(f"  test:  {len(test_dataset)} samples")

    # Model
    model = create_model(cfg)
    model, ckpt = load_model_weight(model, args.checkpoint, device)
    epoch_str = ckpt.get('epoch', 'unknown')

    seq_len = cfg.TARGET_SEQ_LEN

    # Output dir
    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        logging_dir = checkpoint_dir.replace(
            '/scratch/rhong5/weights/temp_training_weights/aslAvatar',
            '/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/zlog')
        os.makedirs(logging_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_log_dir = os.path.join(logging_dir, f"test_{timestamp}")
        output_dir   = os.path.join(test_log_dir, 'gen_images')
    else:
        timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_log_dir = os.path.join(args.output_dir, f"test_{timestamp}")
        output_dir   = os.path.join(test_log_dir, 'gen_images')

    os.makedirs(output_dir, exist_ok=True)

    # ==========================================================================
    # Phase 1: Generation + Rendering
    # ==========================================================================
    if not args.eval_only:
        glosses = args.glosses if args.glosses else cfg.GLOSS_NAME_LIST

        smpl_x = None
        if args.render_mesh:
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            smpl_x = load_smplx_model(cfg.HUMAN_MODELS_PATH)

        # [Fix-4] Build GT lookup when comparison rendering is requested
        gt_lookup: Dict[str, np.ndarray] = {}
        if args.render_comparison and smpl_x is not None:
            gt_lookup = build_gloss_gt_lookup(train_dataset, cfg)

        print(f"\nGenerating {len(glosses)} glosses, {seq_len} frames each")
        if smpl_x:
            print(f"Mesh rendering enabled ({args.img_width}x{args.img_height})")
        if args.render_comparison:
            print("Comparison rendering enabled "
                  "(ours_*.png + gt_*.png + both GIFs)")

        total = 0
        for gloss in tqdm(glosses, desc="Generating"):
            gt_motion = gt_lookup.get(gloss, None)  # None if not found
            total += process_a_gloss(
                model, gloss, output_dir, seq_len, device,
                smpl_x=smpl_x,
                img_w=args.img_width,
                img_h=args.img_height,
                dump_param=args.dump_param,
                cfg=cfg,
                make_gif=args.gif,
                gif_fps=args.gif_fps,
                dataset=train_dataset,
                gt_motion=gt_motion,
                render_comparison=args.render_comparison,
            )
        print(f"\nDone! {len(glosses)} glosses, {total} frames -> {output_dir}")

    # ==========================================================================
    # Phase 2: Evaluation
    # ==========================================================================
    if args.evaluate:
        if test_dataset is None:
            print("\nERROR: Cannot evaluate without a test dataset.")
            return

        print("\n" + "#"*65)
        print("  EVALUATION")
        print("#"*65)

        all_results = run_evaluation(
            model, cfg, device,
            train_dataset, test_dataset,
            batch_size=args.batch_size,
            seed=args.seed,
        )

        # Save YAML (aggregate metrics)
        save_dir = test_log_dir
        os.makedirs(save_dir, exist_ok=True)

        def _serialize(d):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out[k] = _serialize(v)
                elif isinstance(v, (np.floating, np.integer)):
                    out[k] = float(v)
                elif isinstance(v, float):
                    out[k] = round(v, 6)
                else:
                    out[k] = v
            return out

        save_data = {
            "config": {
                "checkpoint":     args.checkpoint,
                "epoch":          str(epoch_str),
                "dataset":        args.dataset_name,
                "num_classes":    cfg.NUM_CLASSES,
                "n_feats":        cfg.N_FEATS,
                "use_upper_body": args.use_upper_body,
                "use_rot6d":      args.use_rot6d,
                "root_normalize": cfg.ROOT_NORMALIZE,
                "seed":           args.seed,
            },
            "model_free":    _serialize(all_results["model_free"]),
            "model_based":   _serialize(all_results["model_based"]),
            # [Fix-3] top-10 summary in YAML
            "top10_by_cos_sim": all_results["top10_glosses"],
        }

        fname     = f"eval_metrics_ep{epoch_str}_{timestamp}.yaml"
        save_path = os.path.join(save_dir, fname)
        with open(save_path, 'w') as f:
            yaml.dump(save_data, f, default_flow_style=False, sort_keys=False)
        print(f"\n>>> Aggregate metrics saved: {save_path}")

        # [Fix-3] Save per-gloss CSV
        csv_fname = f"per_gloss_metrics_ep{epoch_str}_{timestamp}.csv"
        csv_path  = os.path.join(save_dir, csv_fname)
        save_per_gloss_csv(all_results["per_gloss_sim"], csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SMPL-X sign motion + evaluate quality")

    # Model / Data
    parser.add_argument("--checkpoint",    type=str, required=True)
    parser.add_argument("--dataset_name",  type=str, default="WLASL_SMPLX",
                        choices=["SignBank_SMPLX", "WLASL_SMPLX", "ASL3DWord"])
    parser.add_argument("--glosses",       type=str, nargs='+', default=None,
                        help="Specific glosses to generate (default: all)")

    # Model config
    parser.add_argument("--use_upper_body",     action="store_true")
    parser.add_argument("--use_rot6d",          action="store_true")
    parser.add_argument("--no_root_normalize",  action="store_true", default=False)
    parser.add_argument("--use_mini_dataset",   action="store_true")

    # Generation
    parser.add_argument("--output_dir",   type=str, default=None)
    parser.add_argument("--render_mesh",  action="store_true",
                        help="Render SMPL-X meshes")
    parser.add_argument("--dump_param",   action="store_true",
                        help="Save per-frame .npz parameters")
    parser.add_argument("--gif",          action="store_true",
                        help="Generate animated GIF per gloss")
    parser.add_argument("--gif_fps",      type=int, default=8)
    parser.add_argument("--img_width",   type=int, default=384,
                        help="Render width in pixels (default: 384, portrait 3:4)")
    parser.add_argument("--img_height",  type=int, default=512,
                        help="Render height in pixels (default: 512, portrait 3:4)")

    # [Fix-4] Comparison rendering
    parser.add_argument("--render_comparison", action="store_true",
                        help="Also render GT frames alongside generated ones; "
                             "saves gt_{gloss}_{t}.png + ours_{gloss}_{t}.png "
                             "and both GIFs for paper figures. "
                             "Requires --render_mesh to be set.")

    # Evaluation
    parser.add_argument("--evaluate",    action="store_true",
                        help="Run model-free + model-based evaluation")
    parser.add_argument("--eval_only",   action="store_true",
                        help="Skip generation, only run evaluation")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--use_phono_attribute",  action="store_true", default=False)
    parser.add_argument("--text_encoder_type",    type=str, default='clip',
                        choices=["clip", "t5"])

    # General
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.eval_only:
        args.evaluate = True

    if args.render_comparison and not args.render_mesh:
        parser.error("--render_comparison requires --render_mesh")

    main(args)