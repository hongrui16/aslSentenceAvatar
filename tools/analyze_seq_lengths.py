"""
Analyze sequence length distribution of Neural Sign Actors dataset.
Usage: python analyze_seq_lengths.py --poses_dir /path/to/train_poses/poses
"""

import os
import argparse
import numpy as np
from collections import Counter

def analyze_seq_lengths(poses_dir):
    clip_dirs = sorted([
        d for d in os.listdir(poses_dir)
        if os.path.isdir(os.path.join(poses_dir, d))
    ])

    print(f"Total clips found: {len(clip_dirs)}")

    lengths = []
    for clip in clip_dirs:
        clip_path = os.path.join(poses_dir, clip)
        pkls = [f for f in os.listdir(clip_path) if f.endswith('.pkl')]
        lengths.append(len(pkls))

    lengths = np.array(lengths)

    print(f"\n=== Sequence Length Stats ===")
    print(f"  Min    : {lengths.min()}")
    print(f"  Max    : {lengths.max()}")
    print(f"  Mean   : {lengths.mean():.1f}")
    print(f"  Median : {np.median(lengths):.1f}")
    print(f"  Std    : {lengths.std():.1f}")

    print(f"\n=== Percentiles ===")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  p{p:>2d}: {np.percentile(lengths, p):.1f}")

    print(f"\n=== Clips by length bucket ===")
    buckets = [0, 30, 60, 90, 120, 150, 200, 300, float('inf')]
    labels  = ["<30", "30-60", "60-90", "90-120", "120-150", "150-200", "200-300", ">300"]
    for lo, hi, label in zip(buckets[:-1], buckets[1:], labels):
        count = ((lengths >= lo) & (lengths < hi)).sum()
        pct   = 100 * count / len(lengths)
        print(f"  {label:>10s}: {count:5d} clips  ({pct:.1f}%)")

    # Your current fixed length
    target = 60
    within = (lengths <= target).sum()
    print(f"\n=== Fit for your current T={target} ===")
    print(f"  Clips with len <= {target}: {within} / {len(lengths)}  ({100*within/len(lengths):.1f}%)")
    print(f"  Clips with len >  {target}: {len(lengths)-within} — need padding/truncation or longer T")

    return lengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses_dir", type=str, required=True,
                        help="Path to train_poses/poses directory")
    args = parser.parse_args()
    analyze_seq_lengths(args.poses_dir)
