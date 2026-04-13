"""
SignBank SMPL-X Inference Script (params only)

Generate sign language motion from gloss, save SMPL-X parameters (.npz) per frame.

Output:
    output_dir/
        GLOSS_NAME/
            GLOSS_NAME_000000_p0.npz
            GLOSS_NAME_000001_p0.npz
            ...

Usage:
    python generate_smplx_param.py \
        --checkpoint path/to/best_model.pt \
        --glosses AMAZING HELLO THANK-YOU

    python generate_smplx_param.py \
        --checkpoint path/to/best_model.pt \
        --from_dataset --num_glosses 20
"""
import logging

import os
import argparse
import random
from typing import List, Dict, Optional
from accelerate.logging import get_logger

import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime




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
    return model


# =============================================================================
# Generation — uses model.generate() (sample from prior N(0,I))
# =============================================================================

@torch.no_grad()
def generate_from_gloss(model, gloss: str, seq_len: int, device: str, cfg=None):
    """Generate (T, 159) SMPL-X params from gloss via prior sampling."""
    if cfg.USE_LABEL_INDEX_COND:
        gloss_index = cfg.GLOSS_NAME_LIST.index(gloss)
        label_indices = torch.tensor([gloss_index], dtype=torch.long).to(device)
        motion = model.generate(label_indices, seq_len=seq_len, device=device)
    else:
        motion = model.generate([gloss], seq_len=seq_len, device=device)

    diff = (motion[0, 1:] - motion[0, :-1]).abs().mean()
    print(f"Generated frame diff: {diff:.8f}")

    motion_raw = motion.squeeze(0).cpu().numpy()  # (T, input_dim)

    if cfg is not None:
        motion_raw = postprocess_motion(motion_raw, cfg)  # (T, 159)

    return motion_raw



# =============================================================================
# Save .npz (matching your extraction pipeline format)
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
# SMPL-X Mesh Rendering
# =============================================================================

def load_smplx_model(human_model_path):
    """Load SMPL-X model for mesh generation."""
    from human_models.human_models import SMPLX
    smpl_x = SMPLX(human_model_path)
    print(f"Loaded SMPL-X model from {human_model_path}")
    return smpl_x


def params_to_mesh(smpl_x, frame_params):
    """Run SMPL-X forward pass, return (vertices, faces)."""
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
    faces = smpl_x.face.astype(np.int32)
    return vertices, faces

def render_smplx_frame(vertices, faces, img_w=512, img_h=512, 
                       debug=False, gloss=""):
    """
    Render SMPL-X mesh to an image. Self-contained, no external camera functions.
    
    Args:
        vertices: (N, 3) mesh vertices
        faces: (F, 3) face indices
        img_w, img_h: output image size
        debug: print camera/scene info
        gloss: for debug prints
    
    Returns:
        img: (H, W, 3) uint8 RGB image
    """
    import trimesh
    import pyrender

    # --- 1. Compute camera to frame the mesh ---
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2.0
    extent = (vmax - vmin).max()

    # Camera distance so mesh fills ~60% of frame
    fov_y = np.radians(50.0)
    distance = (extent / 2.0) / np.tan(fov_y / 2.0) * 1.4

    if debug:
        print(f"  [{gloss}] center={center}, extent={extent:.4f}, distance={distance:.4f}")

    # --- 2. Center the mesh at origin ---
    verts_centered = vertices - center

    if debug:
        vmin2 = verts_centered.min(axis=0)
        vmax2 = verts_centered.max(axis=0)
        print(f"  [{gloss}] centered verts: min={vmin2}, max={vmax2}")

    # --- 3. Build scene ---
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.1,
        roughnessFactor=0.4,
        alphaMode="OPAQUE",
        emissiveFactor=(0.15, 0.2, 0.15),
        baseColorFactor=(0.6, 0.9, 0.65, 1.0),
    )

    body_trimesh = trimesh.Trimesh(verts_centered, faces, process=False)
    body_mesh = pyrender.Mesh.from_trimesh(body_trimesh, material=material)

    # Perspective camera
    camera = pyrender.PerspectiveCamera(yfov=fov_y, aspectRatio=img_w / img_h)

    # Camera pose: placed at (0, 0, distance) looking at origin
    # pyrender camera looks along -Z in its local frame
    cam_pose = np.eye(4)
    cam_pose[2, 3] = distance  # move camera back along +Z

    if debug:
        print(f"  [{gloss}] cam_pose:\n{cam_pose}")

    # Lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_pose = cam_pose.copy()  # same position as camera

    scene = pyrender.Scene(
        bg_color=[1.0, 1.0, 1.0, 1.0],  # white background
        ambient_light=(0.4, 0.4, 0.4),
    )
    scene.add(body_mesh, "mesh")
    scene.add(camera, pose=cam_pose)
    scene.add(light, pose=light_pose)

    # --- 4. Render ---
    r = pyrender.OffscreenRenderer(
        viewport_width=img_w,
        viewport_height=img_h,
        point_size=1.0,
    )
    color_img, depth_img = r.render(scene, flags=pyrender.RenderFlags.RGBA)
    r.delete()

    if debug:
        depth_valid = depth_img[depth_img > 0]
        print(f"  [{gloss}] render RGBA: shape={color_img.shape}, "
              f"dtype={color_img.dtype}, min={color_img.min()}, max={color_img.max()}")
        print(f"  [{gloss}] depth: valid_pixels={len(depth_valid)}, "
              f"range=[{depth_valid.min():.3f}, {depth_valid.max():.3f}]" 
              if len(depth_valid) > 0 else f"  [{gloss}] depth: NO valid pixels (mesh not visible!)")
        alpha_nonzero = (color_img[:, :, 3] > 0).sum()
        print(f"  [{gloss}] alpha > 0 pixels: {alpha_nonzero} / {img_w * img_h}")

    return color_img[:, :, :3]  # (H, W, 3) uint8



# =============================================================================
# Process one gloss
# =============================================================================
def process_a_gloss(model, gloss, output_dir, seq_len, device,
                  smpl_x=None, img_size=512, dump_param = False, 
                        cfg=None, make_gif=True, gif_fps=8, dataset = None):

    if cfg.USE_PHONO_ATTRIBUTE:
        gloss_str = dataset._gloss_with_phono(gloss)
    else:
        gloss_str = gloss
        
    motion = generate_from_gloss(model, gloss_str, seq_len, device, cfg)  # (T, 159)
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
            # Save .npz
            npz_path = os.path.join(gloss_dir, f"{gloss}_{t:06d}_p0.npz")
            save_frame_npz(params, npz_path)

        # Render mesh
        if smpl_x is not None:
            try:
                import cv2
                vertices, faces = params_to_mesh(smpl_x, params)

                # Debug: print mesh stats for first frame
                # if t == 0:
                #     print(f"  [{gloss}] vertices: shape={vertices.shape}, "
                #           f"min={vertices.min(axis=0)}, max={vertices.max(axis=0)}")
                #     print(f"  [{gloss}] faces: shape={faces.shape}, "
                #           f"min={faces.min()}, max={faces.max()}")
                #     has_nan = np.any(np.isnan(vertices))
                #     has_inf = np.any(np.isinf(vertices))
                #     print(f"  [{gloss}] NaN={has_nan}, Inf={has_inf}")

                img = render_smplx_frame(vertices, faces, 
                                         img_w=img_size, img_h=img_size,
                                         debug=(t == 0), gloss=gloss)
                img_path = os.path.join(render_dir, f"{gloss}_{t:06d}.png")
                # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                
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




# =============================================================================
# Gloss discovery from dataset dir
# =============================================================================

def get_glosses_from_dataset(root_dir: str, num_glosses: Optional[int] = None) -> List[str]:
    if not os.path.isdir(root_dir):
        print(f"WARNING: not found: {root_dir}")
        return []
    glosses = sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)))
    if num_glosses and num_glosses < len(glosses):
        glosses = random.sample(glosses, num_glosses)
    return glosses

def save_gif(frames, gif_path, fps=10):
    """
    Save a list of RGB numpy arrays as an animated GIF.
    
    Args:
        frames: list of (H, W, 3) uint8 numpy arrays
        gif_path: output .gif path
        fps: frames per second
    """
    import imageio
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"  GIF saved: {gif_path} ({len(frames)} frames, {fps} fps)")


# =============================================================================
# Main
# =============================================================================

