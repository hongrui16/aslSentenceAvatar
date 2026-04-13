"""
Pre-extract SignBank hand crops using MediaPipe Hands
=====================================================
For each video:
  1. Uniformly sample n_frames
  2. Run MediaPipe Hands to detect both hands (left & right)
  3. Crop each hand region (with padding), resize to img_size
  4. Horizontally concatenate: [Right hand | Left hand] → (3, H, 2W)
     Missing hand is filled with black (zeros)
  5. Save as (n_frames, 3, img_size, img_size*2) float16 tensor

Usage:
    python tools/preextract_signbank_handcrops.py --n_frames 4 --img_size 224
    python tools/preextract_signbank_handcrops.py --n_frames 1 --img_size 224

Output:
    /scratch/rhong5/dataset/asl_signbank/handcrops_{n_frames}x{img_size}/
"""

import os
import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms
from tqdm import tqdm

import mediapipe as mp

mp_hands = mp.solutions.hands


def get_hand_bbox(hand_landmarks, h, w, padding=0.3):
    """Get bounding box from MediaPipe hand landmarks with padding."""
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # Make it square based on the larger dimension
    bw = x_max - x_min
    bh = y_max - y_min
    side = max(bw, bh)
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    half = side * (1 + padding) / 2
    x_min = int(max(0, cx - half))
    y_min = int(max(0, cy - half))
    x_max = int(min(w, cx + half))
    y_max = int(min(h, cy + half))

    return x_min, y_min, x_max, y_max


def center_crop_square(frame):
    """Fallback: center crop to square."""
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0:y0+side, x0:x0+side]


def crop_and_resize(frame, hand_lm, img_size, normalize):
    """Crop a single hand from frame and return normalized tensor."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = get_hand_bbox(hand_lm, h, w, padding=0.3)
    if x2 > x1 and y2 > y1:
        crop = frame[y1:y2, x1:x2]
    else:
        crop = center_crop_square(frame)
    return normalize(crop)


def extract_one(video_path, output_path, n_frames, img_size, hands_detector):
    """Extract hand crops from one video. Saves [Right | Left] concatenated."""
    normalize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    black = torch.zeros(3, img_size, img_size)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        tensor = torch.zeros(n_frames, 3, img_size, img_size * 2, dtype=torch.float16)
        torch.save(tensor, output_path)
        return 'empty', 0, 0

    # Uniform sampling
    if total >= n_frames:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
    else:
        indices = list(range(total)) + [total - 1] * (n_frames - total)

    # Read all needed frames first
    raw_frames = {}
    for fi in indices:
        if fi in raw_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if ret:
            raw_frames[fi] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

    crops = []
    n_detected = 0

    for fi in indices:
        if fi not in raw_frames:
            crops.append(torch.cat([black, black], dim=2))  # (3, H, 2W)
            continue

        frame = raw_frames[fi]

        # Detect hands
        result = hands_detector.process(frame)

        right_crop = None
        left_crop = None

        if result.multi_hand_landmarks and result.multi_handedness:
            n_detected += 1
            for i, handedness in enumerate(result.multi_handedness):
                label = handedness.classification[0].label  # "Left" or "Right"
                hand_lm = result.multi_hand_landmarks[i]
                if label == "Right" and right_crop is None:
                    right_crop = crop_and_resize(frame, hand_lm, img_size, normalize)
                elif label == "Left" and left_crop is None:
                    left_crop = crop_and_resize(frame, hand_lm, img_size, normalize)

        if right_crop is None:
            right_crop = black
        if left_crop is None:
            left_crop = black

        # Concatenate: [Right | Left] → (3, img_size, img_size*2)
        crops.append(torch.cat([right_crop, left_crop], dim=2))

    tensor = torch.stack(crops).half()
    torch.save(tensor, output_path)
    return 'ok', n_detected, len(indices)


def main(args):
    if not args.output_dir:
        args.output_dir = os.path.join(
            os.path.dirname(args.video_dir.rstrip('/')),
            f'handcrops_{args.n_frames}x{args.img_size}x{args.img_size * 2}'
        )
    os.makedirs(args.output_dir, exist_ok=True)

    video_files = sorted([f for f in os.listdir(args.video_dir)
                          if f.lower().endswith(('.mp4', '.mov', '.avi'))])
    print(f'Found {len(video_files)} videos')
    print(f'Output: {args.output_dir}')

    # Skip already extracted
    existing = set(os.listdir(args.output_dir))
    tasks = []
    for vf in video_files:
        stem = os.path.splitext(vf)[0]
        out_name = f'{stem}.pt'
        if out_name in existing:
            continue
        tasks.append((vf, stem))

    print(f'To extract: {len(tasks)} (skipping {len(video_files) - len(tasks)} existing)')

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3,
    ) as hands:
        total_detected = 0
        total_frames = 0
        failed = []

        for vf, stem in tqdm(tasks, desc='Extracting hand crops'):
            video_path = os.path.join(args.video_dir, vf)
            output_path = os.path.join(args.output_dir, f'{stem}.pt')

            try:
                status, n_det, n_total = extract_one(
                    video_path, output_path, args.n_frames, args.img_size, hands
                )
                if status == 'empty':
                    failed.append(vf)
                total_detected += n_det
                total_frames += n_total
            except Exception as e:
                failed.append(vf)
                print(f'Error: {vf}: {e}')

    det_rate = 100 * total_detected / max(total_frames, 1)
    print(f'\nDone. Extracted: {len(tasks) - len(failed)}, Failed: {len(failed)}')
    print(f'Hand detection rate: {total_detected}/{total_frames} ({det_rate:.1f}%)')
    if failed:
        print(f'Failed files ({len(failed)}):')
        for f in failed[:20]:
            print(f'  {f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video_dir', type=str,
                   default='/scratch/rhong5/dataset/asl_signbank/videos/')
    p.add_argument('--output_dir', type=str, default='')
    p.add_argument('--n_frames', type=int, default=4,
                   help='Number of frames to sample (default 4)')
    p.add_argument('--img_size', type=int, default=224)
    args = p.parse_args()
    main(args)
