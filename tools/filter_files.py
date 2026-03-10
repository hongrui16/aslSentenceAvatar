import os
import shutil
from pathlib import Path

# ===== path config =====
src_root = Path("/scratch/rhong5/dataset/wlasl/video_frame_fitting")
src_images = src_root / "images"
src_keypoints = src_root / "keypoints"
video_root = Path("/scratch/rhong5/dataset/wlasl/videos")

dst_root = Path("/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/tools/data")
dst_images = dst_root / "images"
dst_keypoints = dst_root / "keypoints"

# ===== gloss list =====
gloss_list = [
    "accident",
    "basketball",
    "bowling",
    "check",
    "cry",
    "dog"
]

for gloss in gloss_list:

    img_gloss_dir = src_images / gloss
    kp_gloss_dir = src_keypoints / gloss

    if not img_gloss_dir.exists():
        continue

    for video_id in os.listdir(img_gloss_dir):

        src_img_dir = img_gloss_dir / video_id
        src_kp_dir = kp_gloss_dir / video_id

        dst_img_dir = dst_images / gloss / video_id
        dst_kp_dir = dst_keypoints / gloss / video_id

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_kp_dir.mkdir(parents=True, exist_ok=True)

        # copy images
        for f in src_img_dir.glob("*.jpg"):
            shutil.copy2(f, dst_img_dir)

        # copy keypoints
        if src_kp_dir.exists():
            for f in src_kp_dir.glob("*.json"):
                shutil.copy2(f, dst_kp_dir)

        # copy video
        video_file = video_root / f"{video_id}.mp4"
        if video_file.exists():
            shutil.copy2(video_file, dst_images / gloss)

print("Done.")