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
# Batch processing helper
# ============================================================

def process_batch(image_paths, images_dir, keypoints_dir, holistic,
                  phase_label, visualize=False, vis_dir=None, debug=False):
    """
    Process a list of image paths. Returns stats dict.
    phase_label: string like "[ASL-LEX]" or "[OTHER]" for printing.
    """
    total = len(image_paths)
    detected_pose = 0
    detected_lh = 0
    detected_rh = 0
    failed = 0
    skipped = 0

    for idx, img_path in enumerate(image_paths):
        rel_path = os.path.relpath(img_path, images_dir)
        stem = Path(rel_path).stem
        rel_dir = os.path.dirname(rel_path)

        out_json_dir = os.path.join(keypoints_dir, rel_dir)
        os.makedirs(out_json_dir, exist_ok=True)
        out_json_path = os.path.join(out_json_dir, f"{stem}.json")

        # Skip if already processed
        if os.path.exists(out_json_path):
            # skipped += 1
            # if skipped % 200 == 0:
            #     print(f"  {phase_label} [{idx+1}/{total}] Skipping (exists): {rel_path}")
            # continue
            os.remove(out_json_path)
            
        status, has_pose, has_lh, has_rh = process_single_image(
            img_path, out_json_path, images_dir, holistic,
            visualize=visualize, vis_dir=vis_dir,
        )

        if status == 'unreadable':
            failed += 1
            print(f"  {phase_label} [{idx+1}/{total}] Cannot read: {rel_path}")
            continue
        elif status == 'no_pose':
            failed += 1
            if (idx + 1) % 100 == 0 or failed < 5:
                print(f"  {phase_label} [{idx+1}/{total}] No pose: {rel_path}")
            continue

        detected_pose += 1
        if has_lh:
            detected_lh += 1
        if has_rh:
            detected_rh += 1

        if (idx + 1) % 30 == 0:
            print(f"  {phase_label} [{idx+1}/{total}] Processed: {rel_path}  "
                  f"(LH={'Y' if has_lh else 'N'}, RH={'Y' if has_rh else 'N'})")

        if debug and (idx + 1) > 20:
            break

    return {
        'total': total,
        'detected_pose': detected_pose,
        'detected_lh': detected_lh,
        'detected_rh': detected_rh,
        'failed': failed,
        'skipped': skipped,
    }


# ============================================================
# Main processing
# ============================================================

def process_directory(root_dir, asl_lex_csv=None, visualize=False,
                      model_complexity=2, debug=False):
    """
    Process all images under {root_dir}/images/.
    If asl_lex_csv is provided, process ASL-LEX matched glosses first,
    then process remaining glosses.
    """
    images_dir = os.path.join(root_dir, "images")
    keypoints_dir = os.path.join(root_dir, "keypoints")

    if not os.path.isdir(images_dir):
        print(f"[ERROR] Images directory not found: {images_dir}")
        return

    # ---- Load ASL-LEX glosses ----
    asl_lex_glosses = set()
    if asl_lex_csv and os.path.exists(asl_lex_csv):
        asl_lex_glosses = load_asl_lex_glosses(asl_lex_csv)
        print(f"Loaded {len(asl_lex_glosses)} glosses from ASL-LEX")

    # ---- Collect all images ----
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    all_image_paths = []
    for pat in patterns:
        all_image_paths.extend(glob.glob(os.path.join(images_dir, pat), recursive=True))
    all_image_paths.sort()

    print(f"Found {len(all_image_paths)} total images in {images_dir}")
    if len(all_image_paths) == 0:
        return

    # ---- Split into ASL-LEX matched vs other ----
    # Image path structure: images_dir/<gloss>/<video_id>/<frame>.jpg
    # The gloss is the first component of the relative path.
    matched_paths = []
    other_paths = []

    if asl_lex_glosses:
        for img_path in all_image_paths:
            rel_path = os.path.relpath(img_path, images_dir)
            parts = rel_path.split(os.sep)
            gloss = parts[0].strip().lower() if len(parts) >= 1 else ""
            if gloss in asl_lex_glosses:
                matched_paths.append(img_path)
            else:
                other_paths.append(img_path)

        # Count unique glosses
        matched_glosses = set()
        other_glosses = set()
        for p in matched_paths:
            rel = os.path.relpath(p, images_dir)
            matched_glosses.add(rel.split(os.sep)[0].lower())
        for p in other_paths:
            rel = os.path.relpath(p, images_dir)
            other_glosses.add(rel.split(os.sep)[0].lower())

        print(f"\n{'='*60}")
        print(f"ASL-LEX matched : {len(matched_paths):>7} images  ({len(matched_glosses)} glosses)")
        print(f"Other           : {len(other_paths):>7} images  ({len(other_glosses)} glosses)")
        print(f"{'='*60}")
    else:
        # No ASL-LEX CSV → treat everything as "other"
        other_paths = all_image_paths
        print("No ASL-LEX CSV provided, processing all images without priority.")

    vis_dir = os.path.join(root_dir, "visualizations") if visualize else None

    # ---- Process ----
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=model_complexity,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
    ) as holistic:

        # Phase 1: ASL-LEX matched glosses
        stats_matched = None
        if matched_paths:
            print(f"\n{'='*60}")
            print(f"  PHASE 1: Processing ASL-LEX matched glosses ({len(matched_paths)} images)")
            print(f"{'='*60}")
            stats_matched = process_batch(
                matched_paths, images_dir, keypoints_dir, holistic,
                phase_label="[ASL-LEX]",
                visualize=visualize, vis_dir=vis_dir, debug=debug,
            )

        # Phase 2: Other glosses
        stats_other = None
        if other_paths and not debug:
            print(f"\n{'='*60}")
            print(f"  PHASE 2: Processing other glosses ({len(other_paths)} images)")
            print(f"{'='*60}")
            stats_other = process_batch(
                other_paths, images_dir, keypoints_dir, holistic,
                phase_label="[OTHER]",
                visualize=visualize, vis_dir=vis_dir, debug=False,
            )

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    def print_stats(label, stats):
        if stats is None:
            return
        print(f"\n  {label}:")
        print(f"    Total images  : {stats['total']}")
        print(f"    Skipped       : {stats['skipped']}")
        print(f"    Pose detected : {stats['detected_pose']}")
        print(f"    Left hand     : {stats['detected_lh']}")
        print(f"    Right hand    : {stats['detected_rh']}")
        print(f"    Failed        : {stats['failed']}")

    print_stats("[ASL-LEX]", stats_matched)
    print_stats("[OTHER]", stats_other)

    print(f"\nKeypoints saved to: {keypoints_dir}")
    if visualize:
        print(f"Visualizations saved to: {vis_dir}")


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
    
    
    