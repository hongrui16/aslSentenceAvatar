"""
Render Synthetic SMPL-X Data
==============================

Uses:
  - human_models.human_models.SMPLX for mesh generation
  - utils.renders.render_mesh for rendering

Usage:
  python render_synthetic_smplx.py --root ./synthetic_smplx_data --gif --html
  python render_synthetic_smplx.py --root ./synthetic_smplx_data --gloss wave nod --gif
  python render_synthetic_smplx.py --root ./synthetic_smplx_data --all_samples --gif
"""

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import sys
import argparse
import numpy as np
import torch
import cv2
import imageio


if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
from utils.renders import render_mesh
from human_models.human_models import SMPLX


# ============================================================================
# SMPL-X (using your custom loader)
# ============================================================================

def load_smplx_model(human_model_path):
    """Load SMPL-X model for mesh generation."""
    smpl_x = SMPLX(human_model_path)
    print(f"[OK] Loaded SMPL-X model from {human_model_path}")
    return smpl_x


def params_to_mesh(smpl_x, frame_params):
    """Run SMPL-X forward pass for a single frame, return (vertices, faces)."""
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


# ============================================================================
# Pose Loading
# ============================================================================

def load_video_frames(video_dir):
    """
    Load all .npz frames from a video directory.
    Returns list of dicts, each with smplx_root_pose, smplx_body_pose, etc.
    """
    npz_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.npz')])
    if not npz_files:
        return None

    frames = []
    for fname in npz_files:
        data = np.load(os.path.join(video_dir, fname), allow_pickle=True)
        frames.append({
            'smplx_root_pose':  data['smplx_root_pose'],
            'smplx_body_pose':  data['smplx_body_pose'],
            'smplx_lhand_pose': data['smplx_lhand_pose'],
            'smplx_rhand_pose': data['smplx_rhand_pose'],
            'smplx_jaw_pose':   data['smplx_jaw_pose'],
        })
    return frames


# ============================================================================
# Camera parameter estimation
# ============================================================================

def estimate_cam_params(all_vertices, img_w=960, img_h=540):
    """
    Estimate focal, princpt, cam_trans from mesh bounding box.
    
    Args:
        all_vertices: list of (V, 3) arrays — vertices from all frames
    Returns:
        cam dict compatible with render_mesh
    """
    all_v = np.concatenate(all_vertices, axis=0)  # (T*V, 3)
    center = all_v.mean(axis=0)
    extent = all_v.max(axis=0) - all_v.min(axis=0)
    max_extent = extent.max()

    tz = max_extent * 2.5
    focal = 0.6 * img_h * tz / max_extent

    cam = {
        "focal":    np.array([focal, focal], dtype=np.float32),
        "princpt":  np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32),
        "cam_trans": np.array([-center[0], -center[1], tz - center[2]], dtype=np.float32),
    }
    return cam


# ============================================================================
# Vertex translation into camera space
# ============================================================================

def translate_verts_to_camera_space(all_verts, img_w=960, img_h=540, focal=5000.0):
    """
    render_mesh's camera setup:
      - Camera at origin
      - pyrender2opencv flips Y and Z: camera looks in +Z, Y is flipped
      - So vertices need: positive Z (in front), and Y-flip to look right

    This function computes a translation [tx, ty, tz] to apply to all vertices
    so the mesh is centered and properly sized in the rendered image.

    Returns:
        translation: (3,) array to ADD to all vertices
    """
    # Stack all verts to find global bounds
    all_v = np.concatenate(all_verts, axis=0)  # (T*V, 3)
    center = all_v.mean(axis=0)
    extent = all_v.max(axis=0) - all_v.min(axis=0)
    max_extent = max(extent[0], extent[1])  # X and Y extent

    # Choose tz so mesh fills ~60% of image height
    # projection: pixel_size = focal * world_size / depth
    # We want: max_extent * focal / tz ≈ 0.6 * img_h
    tz = focal * max_extent / (0.6 * img_h)
    tz = max(tz, 1.0)  # safety minimum

    # Translation:
    # - X: center mesh horizontally (no flip needed)
    # - Y: center mesh vertically, BUT Y is flipped by pyrender2opencv
    #       so we negate the Y center
    # - Z: push mesh to +Z (in front of camera)
    tx = -center[0]
    ty = -center[1]   # Y-flip handled by pyrender2opencv, just center it
    tz_final = tz - center[2]

    return np.array([tx, ty, tz_final], dtype=np.float32)



# ============================================================================
# Render one video sample
# ============================================================================
def render_video_sample(smpl_x, video_dir, out_dir,
                        img_w=960, img_h=540, alpha=0.92, make_gif=False):
    """Render all frames of one video sample."""
    frame_params_list = load_video_frames(video_dir)
    if frame_params_list is None:
        print(f"  [SKIP] No data in {video_dir}")
        return 0

    T = len(frame_params_list)
    os.makedirs(out_dir, exist_ok=True)

    # Forward pass all frames
    all_verts = []
    all_faces = None
    for fp in frame_params_list:
        with torch.no_grad():
            verts, faces = params_to_mesh(smpl_x, fp)
        all_verts.append(verts)
        if all_faces is None:
            all_faces = faces

    # Compute translation to place mesh in camera view
    focal = 5000.0
    all_verts = [v * np.array([1, -1, -1], dtype=np.float32) for v in all_verts]

    translation = translate_verts_to_camera_space(all_verts, img_w, img_h, focal)

    # cam_param for render_mesh (camera intrinsics only, no extrinsic translation)
    cam = {
        "focal":    np.array([focal, focal], dtype=np.float32),
        "princpt":  np.array([img_w / 2.0, img_h / 2.0], dtype=np.float32),
    }

    # Debug: print once per sample
    v0 = all_verts[0]
    print(f"    raw verts: x=[{v0[:,0].min():.2f},{v0[:,0].max():.2f}] "
          f"y=[{v0[:,1].min():.2f},{v0[:,1].max():.2f}] "
          f"z=[{v0[:,2].min():.2f},{v0[:,2].max():.2f}]")
    print(f"    translation: {translation}")
    v0t = v0 + translation
    print(f"    translated:  x=[{v0t[:,0].min():.2f},{v0t[:,0].max():.2f}] "
          f"y=[{v0t[:,1].min():.2f},{v0t[:,1].max():.2f}] "
          f"z=[{v0t[:,2].min():.2f},{v0t[:,2].max():.2f}]")

    # Render each frame
    frames_rgb = []
    for t in range(T):
        bg = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

        # Apply translation to put vertices in camera space
        translated_verts = all_verts[t] + translation

        rendered = render_mesh(bg, translated_verts, all_faces, cam, alpha=alpha)
        frames_rgb.append(rendered)

        out_path = os.path.join(out_dir, f"frame_{t:04d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))

    if make_gif and frames_rgb:
        gif_path = os.path.join(out_dir, "animation.gif")
        imageio.mimsave(gif_path, frames_rgb, fps=8, loop=0)

    return T


# ============================================================================
# HTML Comparison
# ============================================================================

def generate_comparison_html(out_root, glosses_rendered):
    """Side-by-side GIF comparison, base64-embedded."""
    import base64

    html = ["""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Synthetic SMPL-X Motion Comparison</title>
<style>
  body { font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; margin: 20px; }
  h1 { color: #e94560; }
  .gloss-row { display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 30px;
               padding: 15px; background: #16213e; border-radius: 10px; }
  .sample-card { text-align: center; }
  .sample-card img { border: 2px solid #0f3460; border-radius: 5px; max-width: 320px; }
  .sample-card p { margin: 5px 0; font-size: 12px; color: #a8a8a8; }
  h2 { color: #e94560; border-bottom: 1px solid #0f3460; padding-bottom: 5px; }
</style></head><body>
<h1>Synthetic SMPL-X Motion Comparison</h1>
<p>Each row = one gloss. Motion should be visually distinct across rows.</p>
"""]

    for gloss, sample_dirs in glosses_rendered.items():
        html.append(f'<h2>{gloss}</h2>\n<div class="gloss-row">\n')
        for sd in sample_dirs[:5]:
            gif_path = os.path.join(sd, "animation.gif")
            if os.path.exists(gif_path):
                with open(gif_path, 'rb') as f:
                    b64 = base64.b64encode(f.read()).decode()
                name = os.path.basename(sd)
                html.append(
                    f'<div class="sample-card">'
                    f'<img src="data:image/gif;base64,{b64}"/>'
                    f'<p>{name}</p></div>\n'
                )
        html.append('</div>\n')

    html.append('</body></html>')
    html_path = os.path.join(out_root, "comparison.html")
    with open(html_path, 'w') as f:
        f.write(''.join(html))
    print(f"\n[HTML] Comparison page saved: {html_path}")


# ============================================================================
# Main
# ============================================================================

def main(args):

    if args.html:
        args.gif = True

    smplx_dir = os.path.join(args.root, args.mode, "smplx_params")
    out_root = args.out_dir or os.path.join(args.root, "rendered", args.mode)

    if not os.path.isdir(smplx_dir):
        print(f"[ERROR] Not found: {smplx_dir}")
        return

    glosses = args.gloss or sorted(
        [d for d in os.listdir(smplx_dir) if os.path.isdir(os.path.join(smplx_dir, d))]
    )

    print(f"Glosses: {glosses}")
    print(f"Output:  {out_root}\n")

    # Load model once
    smpl_x = load_smplx_model('/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/human_models/human_model_files')

    glosses_rendered = {}
    for gloss in glosses:
        gloss_path = os.path.join(smplx_dir, gloss)
        if not os.path.isdir(gloss_path):
            print(f"[SKIP] {gloss_path}")
            continue

        videos = sorted(os.listdir(gloss_path))
        if not args.all_samples:
            videos = videos[:args.max_samples]

        print(f"=== {gloss} ({len(videos)} samples) ===")
        sample_dirs = []
        for i, vid in enumerate(videos):
            vdir = os.path.join(gloss_path, vid)
            if not os.path.isdir(vdir):
                continue
            sout = os.path.join(out_root, gloss, vid)
            n = render_video_sample(
                smpl_x, vdir, sout,
                img_w=args.img_w, img_h=args.img_h,
                alpha=args.alpha, make_gif=args.gif,
            )
            print(f"  {vid}: {n} frames")
            sample_dirs.append(sout)
            

        glosses_rendered[gloss] = sample_dirs

    if args.html:
        generate_comparison_html(out_root, glosses_rendered)

    print(f"\n[DONE] Output: {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render synthetic SMPL-X data")
    parser.add_argument("--root", type=str, default="./synthetic_smplx_data")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--gloss", type=str, nargs="+", default=None)
    parser.add_argument("--all_samples", action="store_true")
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--html", action="store_true", help="Implies --gif")
    parser.add_argument("--out_dir", type=str, default=None)

    parser.add_argument("--img_w", type=int, default=540)
    parser.add_argument("--img_h", type=int, default=540)
    parser.add_argument("--alpha", type=float, default=0.92)
    args = parser.parse_args()


    import numpy as np
    npz = np.load("/scratch/rhong5/dataset/wlasl/train/smplx_params/good/25066/good_25066_000000_p0.npz", allow_pickle=True)
    print("cam_trans:", npz["cam_trans"])
    print("focal:", npz["focal"])
    print("princpt:", npz["princpt"])

    main(args)