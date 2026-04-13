"""
Pre-extract SignBank video frames to .pt files
================================================
For each video, uniformly sample n_frames, resize to img_size, normalize,
and save as a (n_frames, 3, H, W) float16 tensor.

Usage:
    python tools/preextract_signbank_frames.py [--n_frames 16] [--img_size 224] [--num_workers 8]

Output:
    /scratch/rhong5/dataset/asl_signbank/frames_16x224/
        ABHOR.pt
        ABOUTb.pt
        ...
"""

import os
import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm




def extract_one(video_path, output_path, n_frames, img_size):
    """Extract frames from one video and save as .pt tensor."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        tensor = torch.zeros(n_frames, 3, img_size, img_size, dtype=torch.float16)
        torch.save(tensor, output_path)
        return os.path.basename(video_path), 0, 'empty'

    # Uniform sampling
    if total >= n_frames:
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
    else:
        indices = list(range(total)) + [total - 1] * (n_frames - total)

    frames = []
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            if frames:
                frames.append(frames[-1].clone())
            else:
                frames.append(torch.zeros(3, img_size, img_size))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame))

    cap.release()

    tensor = torch.stack(frames).half()  # float16 to save space
    torch.save(tensor, output_path)
    return os.path.basename(video_path), total, 'ok'


def main(args):

    if not args.output_dir:
        args.output_dir = os.path.join(
            os.path.dirname(args.video_dir.rstrip('/')),
            f'frames_{args.n_frames}x{args.img_size}'
        )
    os.makedirs(args.output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(args.video_dir)
                   if f.lower().endswith(('.mp4', '.mov', '.avi'))]
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
        tasks.append((
            os.path.join(args.video_dir, vf),
            os.path.join(args.output_dir, out_name),
            args.n_frames,
            args.img_size,
        ))

    print(f'To extract: {len(tasks)} (skipping {len(video_files) - len(tasks)} existing)')

    failed = []
    done = 0
    total_tasks = len(tasks)
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(extract_one, *t): t[0] for t in tasks}
        for future in as_completed(futures):
            done += 1
            try:
                name, n_total, status = future.result()
                if status == 'empty':
                    failed.append(name)
            except Exception as e:
                failed.append(futures[future])
                print(f'Error: {futures[future]}: {e}', flush=True)
            if done % 100 == 0 or done == total_tasks:
                print(f'Progress: {done}/{total_tasks} ({100*done/total_tasks:.1f}%)', flush=True)

    print(f'Done. Extracted: {len(tasks) - len(failed)}, Failed: {len(failed)}')
    if failed:
        print('Failed files:')
        for f in failed:
            print(f'  {f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video_dir', type=str,
                   default='/scratch/rhong5/dataset/asl_signbank/videos/')
    p.add_argument('--output_dir', type=str, default='')
    p.add_argument('--n_frames', type=int, default=16)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--num_workers', type=int, default=8)
    args = p.parse_args()

    main(args)
