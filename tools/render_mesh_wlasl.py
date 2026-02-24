"""
Local renderer for ASL Signbank SMPL-X outputs.

Directory structure expected:

    File structure:
    wlasl/
        train/
            smplx_params/
                book/
                    06297/
                        book_06297_000000_p0.npz
                        book_06297_000012_p0.npz
                        ...
                    06301/
                        ...
                drink/
                    07102/
                        ...
        videos/
                06297.mp4
                06301.avi
                07102.mp4

Usage:
  # Render one gloss (auto-extracts video frames as background)
  python render_local.py --root ./asl_signbank --gloss AMAZING

  # Render multiple glosses
  python render_local.py --root ./asl_signbank --gloss AMAZING HELLO THANK-YOU

  # Render ALL glosses found in smplx_params/
  python render_local.py --root ./asl_signbank --all

  # White background (skip video frame extraction)
  python render_local.py --root ./asl_signbank --gloss AMAZING --no_bg

  # Custom output dir
  python render_local.py --root ./asl_signbank --gloss AMAZING --out_dir ./my_renders

  # With GIF generation
  python render_local.py --root ./asl_signbank --gloss AMAZING --gif --gif_fps 10

  # 视频帧背景（默认）
python render_local.py --root ./asl_signbank --gloss AMAZING

# 白色背景
python render_local.py --root ./asl_signbank --gloss AMAZING --no_bg

# 手动调透明度（1.0 = 完全不透明）
python render_local.py --root ./asl_signbank --gloss AMAZING --alpha 0.95

"""

import os

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import re
import argparse
import glob
import numpy as np
import cv2
import trimesh
import pyrender
import sys
import random

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from utils.renders import render_mesh

# ─── I/O helpers ──────────────────────────────────────────────────────

def load_obj(obj_path):
    mesh = trimesh.load(obj_path, process=False)
    return np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces, dtype=np.int32)


def load_npz(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    return {
        "focal": data["focal"],        # [fx, fy]
        "princpt": data["princpt"],    # [cx, cy]
        "cam_trans": data.get("cam_trans", None),
    }


def parse_frame_idx(filename, gloss):
    """
    Extract frame index from filename like AMAZING_000004_p0.obj
    Returns int or None.
    """
    stem = os.path.splitext(filename)[0]
    pattern = re.escape(gloss) + r"_(\d+)_p\d+"
    m = re.match(pattern, stem)
    if m:
        return int(m.group(1))
    return None


def save_gif(frames, gif_path, fps=10):
    """Save a list of RGB numpy arrays as an animated GIF."""
    import imageio
    imageio.mimsave(gif_path, frames, fps=fps, loop=0)
    print(f"  GIF saved: {gif_path} ({len(frames)} frames, {fps} fps)")


# ─── Per-gloss pipeline ──────────────────────────────────────────────

def render_a_gloss_instance(video_path, gloss_param_dir, out_dir, use_bg=True, alpha=0.8,
                            make_gif=False, gif_fps=10):
    """Render all keyframes for a single gloss."""

    obj_files = sorted(glob.glob(os.path.join(gloss_param_dir, "*.obj")))
    if not obj_files:
        print(f"[SKIP] No .obj files in {gloss_param_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    out_img_dir = os.path.join(out_dir, 'img')
    os.makedirs(out_img_dir, exist_ok=True)
    

    # Open video once, seek per frame
    has_video = os.path.isfile(video_path) and use_bg
    if use_bg and not has_video:
        print(f"[WARN] Video not found: {video_path}, using white background")

    cap = None
    if has_video:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] Cannot open {video_path}, using white background")
            cap = None

    gif_frames = []
    rendered_count = 0
    for obj_path in obj_files:
        base_name = os.path.splitext(os.path.basename(obj_path))[0]
        npz_path = os.path.join(gloss_param_dir, base_name + ".npz")

        if not os.path.isfile(npz_path):
            print(f"[WARN] Missing .npz for {obj_path}, skipping")
            continue

        frame_idx = int(base_name.split('_')[2]) # wlasl/train/smplx_params/bad/04708/bad_04708_0000/bad_04708_000042_p0.obj

        # Load mesh + camera params
        vertices, faces = load_obj(obj_path)
        cam = load_npz(npz_path)

        # Get background frame from video
        bg = None
        if cap is not None and frame_idx is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame_bgr = cap.read()
            if ret:
                bg = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if bg is None:
            # Fallback: white canvas sized from camera principal point
            cx, cy = cam["princpt"]
            w = int(cx * 2) if cx > 100 else 960
            h = int(cy * 2) if cy > 100 else 540
            bg = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Render overlay
        rendered = render_mesh(bg, vertices, faces, cam, alpha=alpha)

        out_path = os.path.join(out_img_dir, f"{base_name}_render.png")
        cv2.imwrite(out_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        rendered_count += 1

        if make_gif:
            gif_frames.append(rendered)

    if cap is not None:
        cap.release()

    # Save GIF after all frames
    if make_gif and gif_frames:
        # Name GIF after the directory (gloss_videoID)
        gif_name = os.path.basename(out_dir) + ".gif"
        gif_path = os.path.join(out_dir, gif_name)
        save_gif(gif_frames, gif_path, fps=gif_fps)


def process_all_gloss_categories(gloss_parent_dir, video_source_dir, out_dir, use_bg=True, alpha=0.8,
                                 gloss_name_list=None, make_gif=False, gif_fps=10, max_video_num = 6):
    gloss_names = os.listdir(gloss_parent_dir) if gloss_name_list is None else gloss_name_list
    for gloss in gloss_names:
        video_parent_dir = os.path.join(gloss_parent_dir, gloss)
        video_names = os.listdir(video_parent_dir)
        random.shuffle(video_names)
        for vi_id, video_na in enumerate(video_names):
            video_filepath = os.path.join(video_source_dir, f"{video_na}.mp4")
            save_img_dir = os.path.join(out_dir, gloss, video_na)
            gloss_param_dir = os.path.join(gloss_parent_dir, gloss, video_na)
            os.makedirs(save_img_dir, exist_ok=True)
            render_a_gloss_instance(video_filepath, gloss_param_dir, save_img_dir,
                                    use_bg=use_bg, alpha=alpha,
                                    make_gif=make_gif, 
                                    gif_fps=gif_fps)
            if vi_id >= (max_video_num -1):
                break
    




# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ASL Signbank SMPL-X local renderer")
    parser.add_argument("--root", type=str, default=None,
                        help="Root dir containing smplx_params/ and videos/")
    parser.add_argument("--gloss", type=str, nargs="+", default=None,
                        help="One or more gloss names to render")
    parser.add_argument("--all", action="store_true",
                        help="Render all glosses in smplx_params/")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (default: {root}/rendered)")
    parser.add_argument("--no_bg", action="store_true",
                        help="White background instead of video frames")
    parser.add_argument("--alpha", type=float, default=0.92,
                        help="Mesh overlay transparency [0-1]")
    parser.add_argument("--gif", action="store_true",
                        help="Generate animated GIF per video instance")
    parser.add_argument("--gif_fps", type=int, default=8,
                        help="GIF frame rate (default: 8)")
    args = parser.parse_args()

    out_dir = args.out_dir
    args.gif  = True

    gloss_name_list = ['before', 'cool', 'drink', 'go', 'thin']
    gloss_parent_dir = '/scratch/rhong5/dataset/wlasl/train/smplx_params/'
    video_source_dir = '/scratch/rhong5/dataset/wlasl/videos'
    out_dir = '/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/output_wlasl_render'
    process_all_gloss_categories(gloss_parent_dir, video_source_dir, out_dir,
                                 gloss_name_list=gloss_name_list,
                                 make_gif=args.gif, gif_fps=args.gif_fps)

if __name__ == "__main__":
    main()