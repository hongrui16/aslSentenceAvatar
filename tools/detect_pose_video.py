"""
MediaPipe Holistic 3D Detection: Pose (33) + Left Hand (21) + Right Hand (21).

Directory structure:
  input:  {root}/images/{category}/{video_id}/{frame}.jpg
  output: {root}/keypoints/{category}/{video_id}/{frame}.json

JSON output fields:
  - pose_world_landmarks_3d   (33 landmarks)
  - pose_landmarks_2d         (33 landmarks)
  - left_hand_landmarks_3d    (21 landmarks)  ← NEW
  - left_hand_landmarks_2d    (21 landmarks)  ← NEW
  - right_hand_landmarks_3d   (21 landmarks)  ← NEW
  - right_hand_landmarks_2d   (21 landmarks)  ← NEW

Usage:
  python detect_pose_hand.py --root /path/to/video_frame_fitting
  python detect_pose_hand.py --root /path/to/video_frame_fitting --visualize
"""

import argparse
import json
import os
import glob
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import csv

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# ============================================================
# Landmark name lists
# ============================================================

# 33 MediaPipe pose landmarks
POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# 21 MediaPipe hand landmarks (same for left & right)
HAND_LANDMARK_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


# ============================================================
# Extraction helpers
# ============================================================

def extract_pose_landmarks(results, h, w):
    """Extract 33 pose landmarks (3D world + 2D pixel)."""
    landmarks_3d = []
    landmarks_2d = []

    if results.pose_world_landmarks:
        for i, lm in enumerate(results.pose_world_landmarks.landmark):
            landmarks_3d.append({
                "id": i,
                "name": POSE_LANDMARK_NAMES[i],
                "x": round(lm.x, 6),
                "y": round(lm.y, 6),
                "z": round(lm.z, 6),
                "visibility": round(lm.visibility, 4),
            })

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmarks_2d.append({
                "id": i,
                "name": POSE_LANDMARK_NAMES[i],
                "x": round(lm.x * w, 2),
                "y": round(lm.y * h, 2),
                "z": round(lm.z, 6),
                "visibility": round(lm.visibility, 4),
            })

    return landmarks_3d, landmarks_2d


def extract_hand_landmarks(hand_landmarks, h, w, side="left"):
    """
    Extract 21 hand landmarks.
    MediaPipe Holistic hand output only has normalized 2D+z (no world coords).
    We treat the normalized coords as a local 3D space:
      - landmarks_3d:  (x, y, z) in normalized image coords (z = depth relative to wrist)
      - landmarks_2d:  (x*w, y*h) in pixel coords, z kept as-is
    """
    landmarks_3d = []
    landmarks_2d = []

    if hand_landmarks is None:
        return landmarks_3d, landmarks_2d

    for i, lm in enumerate(hand_landmarks.landmark):
        name = HAND_LANDMARK_NAMES[i]
        landmarks_3d.append({
            "id": i,
            "name": name,
            "x": round(lm.x, 6),
            "y": round(lm.y, 6),
            "z": round(lm.z, 6),
        })
        landmarks_2d.append({
            "id": i,
            "name": name,
            "x": round(lm.x * w, 2),
            "y": round(lm.y * h, 2),
            "z": round(lm.z, 6),
        })

    return landmarks_3d, landmarks_2d


# ============================================================
# ASL-LEX helper
# ============================================================

def load_asl_lex_glosses(csv_path):
    """Load all gloss names (Entry ID) from ASL-LEX CSV. Returns a set of lowercase glosses."""
    glosses = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            gloss = row['Entry ID'].strip().lower()
            if gloss:
                glosses.add(gloss)
    return glosses


# ============================================================
# Single-image processing
# ============================================================

def process_single_image(img_path, out_json_path, images_dir, holistic,
                         visualize=False, vis_dir=None):
    """
    Process one image with Holistic model.
    Returns: (success, has_pose, has_lh, has_rh)
    """
    rel_path = os.path.relpath(img_path, images_dir)
    stem = Path(rel_path).stem
    rel_dir = os.path.dirname(rel_path)

    image = cv2.imread(img_path)
    if image is None:
        return 'unreadable', False, False, False

    h, w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    has_pose = results.pose_world_landmarks is not None
    has_lh = results.left_hand_landmarks is not None
    has_rh = results.right_hand_landmarks is not None

    if not has_pose:
        empty_data = {
            "image": rel_path,
            "image_size": {"width": w, "height": h},
            "pose_world_landmarks_3d": [],
            "pose_landmarks_2d": [],
            "left_hand_landmarks_3d": [],
            "left_hand_landmarks_2d": [],
            "right_hand_landmarks_3d": [],
            "right_hand_landmarks_2d": [],
            "detected": False,
            "detected_left_hand": False,
            "detected_right_hand": False,
        }
        with open(out_json_path, "w") as f:
            json.dump(empty_data, f, indent=2)
        return 'no_pose', False, False, False

    # ---- Extract all landmarks ----
    pose_3d, pose_2d = extract_pose_landmarks(results, h, w)
    lh_3d, lh_2d = extract_hand_landmarks(results.left_hand_landmarks, h, w, "left")
    rh_3d, rh_2d = extract_hand_landmarks(results.right_hand_landmarks, h, w, "right")

    keypoint_data = {
        "image": rel_path,
        "image_size": {"width": w, "height": h},
        "pose_world_landmarks_3d": pose_3d,
        "pose_landmarks_2d": pose_2d,
        "left_hand_landmarks_3d": lh_3d,
        "left_hand_landmarks_2d": lh_2d,
        "right_hand_landmarks_3d": rh_3d,
        "right_hand_landmarks_2d": rh_2d,
        "detected": True,
        "detected_left_hand": has_lh,
        "detected_right_hand": has_rh,
    }

    with open(out_json_path, "w") as f:
        json.dump(keypoint_data, f, indent=2)

    # ---- Visualization ----
    if visualize and vis_dir:
        vis_out_dir = os.path.join(vis_dir, rel_dir)
        os.makedirs(vis_out_dir, exist_ok=True)
        vis_path = os.path.join(vis_out_dir, f"{stem}.jpg")

        annotated = image.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )
        cv2.imwrite(vis_path, annotated)

    return 'ok', has_pose, has_lh, has_rh


# ============================================================
# Per-video processing  (static_image_mode=False → tracking)
# ============================================================

def process_video(gloss, video_id, frame_paths, images_dir, keypoints_dir,
                  model_complexity, visualize=False, vis_dir=None, debug=False):
    """
    Process one video's frames in sorted order using a fresh Holistic instance
    (static_image_mode=False) so MediaPipe can track hands across frames.
    A new Holistic is created per video to prevent tracking state leaking
    between different videos.

    Skip logic: if a frame's JSON already has at least one hand → keep it.
                if both hands are empty → re-detect.
    """
    stats = dict(total=0, detected_pose=0, detected_lh=0,
                 detected_rh=0, failed=0, skipped=0)

    # Frames must be in temporal order for tracking to work
    frame_paths = sorted(frame_paths)
    stats['total'] = len(frame_paths)

    # Collect frames that actually need (re-)detection
    frames_to_run = []
    for img_path in frame_paths:
        rel_path  = os.path.relpath(img_path, images_dir)
        stem      = Path(rel_path).stem
        rel_dir   = os.path.dirname(rel_path)
        out_dir   = os.path.join(keypoints_dir, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_json  = os.path.join(out_dir, f"{stem}.json")

        if os.path.exists(out_json):
            try:
                with open(out_json, 'r') as _f:
                    _d = json.load(_f)
                _lh_ok = len(_d.get('left_hand_landmarks_3d',  [])) > 0
                _rh_ok = len(_d.get('right_hand_landmarks_3d', [])) > 0
                if _lh_ok or _rh_ok:
                    stats['skipped'] += 1
                    continue          # at least one hand present → keep
                # both hands empty → queue for re-detection
            except Exception:
                pass                  # corrupted JSON → re-detect

        frames_to_run.append((img_path, out_json))

    if not frames_to_run:
        return stats

    # One fresh Holistic per video → clean tracking state
    with mp_holistic.Holistic(
        static_image_mode=False,          # tracking across consecutive frames
        model_complexity=model_complexity,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as holistic:

        for frame_idx, (img_path, out_json_path) in enumerate(frames_to_run):
            status, has_pose, has_lh, has_rh = process_single_image(
                img_path, out_json_path, images_dir, holistic,
                visualize=visualize, vis_dir=vis_dir,
            )

            if status == 'unreadable':
                stats['failed'] += 1
                print(f"  [{gloss}/{video_id}] Cannot read: {os.path.basename(img_path)}")
                continue
            elif status == 'no_pose':
                stats['failed'] += 1
                continue

            stats['detected_pose'] += 1
            if has_lh: stats['detected_lh'] += 1
            if has_rh: stats['detected_rh'] += 1

            if debug and frame_idx >= 20:
                break

    return stats


# ============================================================
# Main processing
# ============================================================

def collect_videos(images_dir):
    """
    Walk images_dir/{gloss}/{video_id}/ and return a list of
    (gloss, video_id, [frame_paths...]) tuples, sorted by (gloss, video_id).
    """
    videos = []
    for gloss in sorted(os.listdir(images_dir)):
        gloss_dir = os.path.join(images_dir, gloss)
        if not os.path.isdir(gloss_dir):
            continue
        for video_id in sorted(os.listdir(gloss_dir)):
            vid_dir = os.path.join(gloss_dir, video_id)
            if not os.path.isdir(vid_dir):
                continue
            frames = sorted(
                os.path.join(vid_dir, f)
                for f in os.listdir(vid_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            if frames:
                videos.append((gloss, video_id, frames))
    return videos


def process_directory(root_dir, asl_lex_csv=None, visualize=False,
                      model_complexity=2, debug=False):
    """
    Process all videos under {root_dir}/images/{gloss}/{video_id}/.
    Each video is processed with its own Holistic instance (static_image_mode=False)
    so tracking works within a video without leaking across videos.
    If asl_lex_csv is provided, ASL-LEX matched glosses are processed first.
    """
    images_dir    = os.path.join(root_dir, "images")
    keypoints_dir = os.path.join(root_dir, "keypoints_v")
    vis_dir       = os.path.join(root_dir, "visualizations") if visualize else None

    if not os.path.isdir(images_dir):
        print(f"[ERROR] Images directory not found: {images_dir}")
        return

    # ---- Load ASL-LEX glosses (optional priority split) ----
    asl_lex_glosses = set()
    if asl_lex_csv and os.path.exists(asl_lex_csv):
        asl_lex_glosses = load_asl_lex_glosses(asl_lex_csv)
        print(f"Loaded {len(asl_lex_glosses)} glosses from ASL-LEX")

    # ---- Collect all videos ----
    all_videos = collect_videos(images_dir)
    total_videos = len(all_videos)
    total_images = sum(len(f) for _, _, f in all_videos)
    print(f"Found {total_videos} videos  ({total_images} frames)  in {images_dir}")
    if not all_videos:
        return

    # ---- Priority split: ASL-LEX matched first, then others ----
    matched_videos = [(g, v, f) for g, v, f in all_videos if g.lower() in asl_lex_glosses]
    other_videos   = [(g, v, f) for g, v, f in all_videos if g.lower() not in asl_lex_glosses]

    if asl_lex_glosses:
        print(f"{'='*60}")
        print(f"ASL-LEX matched : {len(matched_videos):>5} videos")
        print(f"Other           : {len(other_videos):>5} videos")
        print(f"{'='*60}")

    phases = []
    if matched_videos:
        phases.append(("PHASE 1 [ASL-LEX]", matched_videos))
    if other_videos and not debug:
        phases.append(("PHASE 2 [OTHER]",   other_videos))
    if not asl_lex_glosses:
        phases = [("ALL", all_videos)]

    # ---- Accumulate stats ----
    grand = dict(total=0, detected_pose=0, detected_lh=0,
                 detected_rh=0, failed=0, skipped=0)

    for phase_label, video_list in phases:
        n_vids = len(video_list)
        print(f"\n{'='*60}")
        print(f"  {phase_label}: {n_vids} videos")
        print(f"{'='*60}")

        phase_stats = dict(total=0, detected_pose=0, detected_lh=0,
                           detected_rh=0, failed=0, skipped=0)

        for vid_idx, (gloss, video_id, frame_paths) in enumerate(video_list):
            vstats = process_video(
                gloss, video_id, frame_paths,
                images_dir, keypoints_dir, model_complexity,
                visualize=visualize, vis_dir=vis_dir, debug=debug,
            )
            for k in phase_stats:
                phase_stats[k] += vstats[k]
                grand[k]       += vstats[k]

            # Progress line per video
            lh_rate = vstats['detected_lh'] / max(vstats['total'], 1) * 100
            rh_rate = vstats['detected_rh'] / max(vstats['total'], 1) * 100
            print(f"  [{vid_idx+1:>5}/{n_vids}] {gloss}/{video_id}  "
                  f"frames={vstats['total']}  "
                  f"skip={vstats['skipped']}  "
                  f"LH={lh_rate:.0f}%  RH={rh_rate:.0f}%")

            if debug and vid_idx >= 3:
                break

        print(f"  --- {phase_label} done: "
              f"pose={phase_stats['detected_pose']}  "
              f"LH={phase_stats['detected_lh']}  "
              f"RH={phase_stats['detected_rh']}  "
              f"skipped={phase_stats['skipped']}  "
              f"failed={phase_stats['failed']}")

    # ---- Grand summary ----
    print(f"\n{'='*60}")
    print(f"GRAND SUMMARY")
    print(f"{'='*60}")
    print(f"  Total frames  : {grand['total']}")
    print(f"  Skipped       : {grand['skipped']}")
    print(f"  Pose detected : {grand['detected_pose']}")
    print(f"  Left hand     : {grand['detected_lh']}")
    print(f"  Right hand    : {grand['detected_rh']}")
    print(f"  Failed        : {grand['failed']}")
    print(f"\nKeypoints saved to: {keypoints_dir}")
    if visualize:
        print(f"Visualizations  to: {vis_dir}")

    print(f"\nKeypoints saved to: {keypoints_dir}")
    if visualize:
        print(f"Visualizations saved to: {vis_dir}")





def check_hand_landmarks(folder_path):
    HAND_KEYS = [
        'left_hand_landmarks_3d',
        'left_hand_landmarks_2d',
        'right_hand_landmarks_3d',
        'right_hand_landmarks_2d',
    ]

    json_files = sorted(f for f in os.listdir(folder_path) if f.endswith('.json'))
    if not json_files:
        print(f"No JSON files found in: {folder_path}")
        return

    found_any = False
    for jf in json_files:
        path = os.path.join(folder_path, jf)
        with open(path, 'r') as f:
            data = json.load(f)

        non_empty = {k: data[k] for k in HAND_KEYS if k in data and len(data[k]) > 0}
        if non_empty:
            found_any = True
            print(f"\n[{jf}]")
            for k, v in non_empty.items():
                print(f"  {k}: {len(v)} landmarks")

    if not found_any:
        print("All JSON files have empty hand landmark fields.")



    
def main():
    parser = argparse.ArgumentParser(description="MediaPipe Holistic: Pose + Hands Detection")
    parser.add_argument("--root", type=str,
                        default="/scratch/rhong5/dataset/wlasl/video_frame_fitting",
                        help="Root directory containing 'images/' subfolder")
    parser.add_argument("--asl_lex_csv", type=str,
                        default='/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/data/ASL_LEX2.0/ASL-LEX_View_Data.csv',
                        help="Path to ASL-LEX_View_Data.csv (prioritize matched glosses)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization images with skeleton + hand overlay")
    parser.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2],
                        help="Model complexity: 0=lite, 1=full, 2=heavy (default: 2)")
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()
    process_directory(args.root, asl_lex_csv=args.asl_lex_csv,
                      visualize=args.visualize,
                      model_complexity=args.model_complexity, debug=args.debug)


if __name__ == "__main__":
    main()
    # folder_path = '/scratch/rhong5/dataset/wlasl/video_frame_fitting/keypoints/apple/02999'
    # check_hand_landmarks(folder_path)
    
    
    