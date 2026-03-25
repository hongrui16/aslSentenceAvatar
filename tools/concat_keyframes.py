"""
concat_keyframes.py
===================
Crop keyframe PNGs and concatenate horizontally per method (gt / ours / signavatar).

Usage:
    python concat_keyframes.py --input_dir path/to/renders/cool
"""
import os
import argparse
import glob
from PIL import Image

CROP_BOX = (82, 27, 295, 300)   # xmin, ymin, xmax, ymax


def load_sorted(input_dir, prefix):
    """Load and sort all PNGs matching prefix_*.png"""
    pattern = os.path.join(input_dir, f"{prefix}_*.png")
    files   = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")
    print(f"  {prefix}: {len(files)} frames")
    return [Image.open(f).convert("RGB") for f in files]


def crop_and_concat(images, crop_box):
    """Crop each image and concatenate horizontally."""
    cropped = [img.crop(crop_box) for img in images]
    w, h    = cropped[0].size
    canvas  = Image.new("RGB", (w * len(cropped), h), (255, 255, 255))
    for i, img in enumerate(cropped):
        canvas.paste(img, (i * w, 0))
    return canvas


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    gloss = args.gloss

    for prefix in [f"gt_{gloss}", f"ours_{gloss}", f"signavatar_{gloss}"]:
        images = load_sorted(args.input_dir, prefix)
        strip  = crop_and_concat(images, CROP_BOX)
        out    = os.path.join(args.output_dir, f"{prefix}_strip.png")
        strip.save(out)
        print(f"  Saved: {out}  ({strip.width}x{strip.height})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  type=str, default='/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/comparision/frames',
                        help="Directory containing gt_*/ours_*/signavatar_* PNGs")
    parser.add_argument("--gloss",      type=str, default='cool',
                        help="Gloss name, e.g. cool")
    parser.add_argument("--output_dir", type=str, default="/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/comparision",
                        help="Where to save the strips")
    main(parser.parse_args())
