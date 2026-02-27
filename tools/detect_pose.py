"""
MediaPipe 3D Pose Detection for video frame images.

Directory structure:
  input:  {split}/images/{category}/{video_id}/{frame}.jpg
  output: {split}/keypoints/{category}/{video_id}/{frame}.json

Usage:
  python detect_pose.py --root test
  python detect_pose.py --root train
  python detect_pose.py --root test --visualize
"""

import argparse
import json
import os
import glob
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 33 MediaPipe pose landmarks
LANDMARK_NAMES = [
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


def detect_pose_3d(image_path, pose_model):
    """Detect 3D pose landmarks from a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"  [WARN] Cannot read: {image_path}")
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_model.process(image_rgb)

    if not results.pose_world_landmarks:
        print(f"  [WARN] No pose detected: {image_path}")
        return None, image

    landmarks_3d = []
    landmarks_2d = []
    h, w = image.shape[:2]

    for i, lm in enumerate(results.pose_world_landmarks.landmark):
        landmarks_3d.append({
            "id": i,
            "name": LANDMARK_NAMES[i],
            "x": round(lm.x, 6),
            "y": round(lm.y, 6),
            "z": round(lm.z, 6),
            "visibility": round(lm.visibility, 4),
        })

    for i, lm in enumerate(results.pose_landmarks.landmark):
        landmarks_2d.append({
            "id": i,
            "name": LANDMARK_NAMES[i],
            "x": round(lm.x * w, 2),
            "y": round(lm.y * h, 2),
            "z": round(lm.z, 6),
            "visibility": round(lm.visibility, 4),
        })

    keypoint_data = {
        "image": image_path,
        "image_size": {"width": w, "height": h},
        "pose_world_landmarks_3d": landmarks_3d,
        "pose_landmarks_2d": landmarks_2d,
    }

    return keypoint_data, image


def save_visualization(image, results, save_path):
    """Draw pose on image and save."""
    annotated = image.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    cv2.imwrite(save_path, annotated)


def process_directory(root_dir, visualize=False, model_complexity=2, debug = False):
    """
    Process all images under {root_dir}/images/ and save keypoints
    to {root_dir}/keypoints/ with the same sub-directory structure.
    """
    images_dir = os.path.join(root_dir, "images")
    keypoints_dir = os.path.join(root_dir, "keypoints")

    if not os.path.isdir(images_dir):
        print(f"[ERROR] Images directory not found: {images_dir}")
        return

    # Collect all jpg/png images
    patterns = ["**/*.jpg", "**/*.jpeg", "**/*.png"]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(os.path.join(images_dir, pat), recursive=True))
    image_paths.sort()

    print(f"Found {len(image_paths)} images in {images_dir}")

    if len(image_paths) == 0:
        return

    # Optional visualization output dir
    vis_dir = os.path.join(root_dir, "visualizations") if visualize else None

    total = len(image_paths)
    detected = 0
    failed = 0

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=model_complexity,  # 0, 1, or 2 (2 = most accurate)
        enable_segmentation=False,
        min_detection_confidence=0.5,
    ) as pose:

        for idx, img_path in enumerate(image_paths):
            # Compute relative path: e.g. accident/00639/frame.jpg
            rel_path = os.path.relpath(img_path, images_dir)
            stem = Path(rel_path).stem
            rel_dir = os.path.dirname(rel_path)

            # Output json path
            out_json_dir = os.path.join(keypoints_dir, rel_dir)
            os.makedirs(out_json_dir, exist_ok=True)
            out_json_path = os.path.join(out_json_dir, f"{stem}.json")

            # Skip if already processed
            if os.path.exists(out_json_path):
                if (idx + 1) % 200 == 0:
                    print(f"  [{idx+1}/{total}] Skipping (exists): {rel_path}")
                continue

            # Detect
            image = cv2.imread(img_path)
            if image is None:
                print(f"  [{idx+1}/{total}] Cannot read: {rel_path}")
                failed += 1
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if not results.pose_world_landmarks:
                failed += 1
                # Save empty result so we know it was processed
                empty_data = {
                    "image": rel_path,
                    "image_size": {"width": image.shape[1], "height": image.shape[0]},
                    "pose_world_landmarks_3d": [],
                    "pose_landmarks_2d": [],
                    "detected": False,
                }
                with open(out_json_path, "w") as f:
                    json.dump(empty_data, f, indent=2)

                if (idx + 1) % 100 == 0 or failed < 5:
                    print(f"  [{idx+1}/{total}] No pose: {rel_path}")
                continue

            # Extract landmarks
            h, w = image.shape[:2]
            landmarks_3d = []
            landmarks_2d = []

            for i, lm in enumerate(results.pose_world_landmarks.landmark):
                landmarks_3d.append({
                    "id": i,
                    "name": LANDMARK_NAMES[i],
                    "x": round(lm.x, 6),
                    "y": round(lm.y, 6),
                    "z": round(lm.z, 6),
                    "visibility": round(lm.visibility, 4),
                })

            for i, lm in enumerate(results.pose_landmarks.landmark):
                landmarks_2d.append({
                    "id": i,
                    "name": LANDMARK_NAMES[i],
                    "x": round(lm.x * w, 2),
                    "y": round(lm.y * h, 2),
                    "z": round(lm.z, 6),
                    "visibility": round(lm.visibility, 4),
                })

            keypoint_data = {
                "image": rel_path,
                "image_size": {"width": w, "height": h},
                "pose_world_landmarks_3d": landmarks_3d,
                "pose_landmarks_2d": landmarks_2d,
                "detected": True,
            }

            with open(out_json_path, "w") as f:
                json.dump(keypoint_data, f, indent=2)

            detected += 1

            # Visualization
            if visualize and vis_dir:
                vis_out_dir = os.path.join(vis_dir, rel_dir)
                os.makedirs(vis_out_dir, exist_ok=True)
                vis_path = os.path.join(vis_out_dir, f"{stem}.jpg")
                annotated = image.copy()
                mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )
                cv2.imwrite(vis_path, annotated)

            if (idx + 1) % 100 == 0:
                print(f"  [{idx+1}/{total}] Processed: {rel_path}")
                
            if debug and idx > 20:
                break

    print(f"\nDone! Detected: {detected}, Failed: {failed}, Total: {total}")
    print(f"Keypoints saved to: {keypoints_dir}")
    if visualize:
        print(f"Visualizations saved to: {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description="MediaPipe 3D Pose Detection")
    parser.add_argument("--root", type=str, default="/scratch/rhong5/dataset/wlasl/video_frame_fitting",
                        help="Root directory containing 'images/' subfolder (e.g., 'test' or 'train')")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization images with skeleton overlay")
    parser.add_argument("--model_complexity", type=int, default=2, choices=[0, 1, 2],
                        help="MediaPipe model complexity: 0=lite, 1=full, 2=heavy (default: 2)")
    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    process_directory(args.root, visualize=args.visualize, model_complexity=args.model_complexity, debug = args.debug)


if __name__ == "__main__":
    main()
