"""
Compare fitted (GT) vs generated SMPL-X render GIFs.

Per gloss group:
  Row 1 (Fitted): all video GIFs from output_wlasl_render/[gloss]/[vid_id]/[vid_id].gif
  Row 2 (Gen):    single GIF from gen_dir/[gloss]/[gloss].gif

Groups are stacked vertically for each gloss.

Usage:
  python visualize_comparison.py \
    --fit_dir output_wlasl_render \
    --gen_dir test_20260219_143646/gen_images \
    --glosses before cool drink go thin \
    --output comparison.gif
"""

import argparse
import os
import glob
from PIL import Image, ImageDraw, ImageFont
import imageio


def load_gif_frames(path, max_frames=None):
    img = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(img.convert("RGBA").copy())
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    if max_frames and len(frames) > max_frames:
        frames = frames[:max_frames]
    return frames


def find_all_fit_gifs(fit_dir, gloss):
    """Find ALL fitted GIFs under fit_dir/gloss/[video_id]/."""
    gloss_dir = os.path.join(fit_dir, gloss)
    results = []
    if not os.path.isdir(gloss_dir):
        return results
    for vid_id in sorted(os.listdir(gloss_dir)):
        vid_path = os.path.join(gloss_dir, vid_id)
        if not os.path.isdir(vid_path):
            continue
        # Try [video_id].gif first
        gif = os.path.join(vid_path, f"{vid_id}.gif")
        if os.path.exists(gif):
            results.append((vid_id, gif))
            continue
        # Try any .gif in the folder
        gifs = glob.glob(os.path.join(vid_path, "*.gif"))
        if gifs:
            results.append((vid_id, gifs[0]))
    return results


def find_gen_gif(gen_dir, gloss):
    for candidate in [
        os.path.join(gen_dir, gloss, f"{gloss}.gif"),
        os.path.join(gen_dir, gloss, "renders", f"{gloss}.gif"),
    ]:
        if os.path.exists(candidate):
            return candidate
    gifs = glob.glob(os.path.join(gen_dir, gloss, "**", "*.gif"), recursive=True)
    return gifs[0] if gifs else None


def resize_frames(frames, size):
    return [f.resize(size, Image.LANCZOS) for f in frames]


def get_font(size=16):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit_dir", required=True)
    parser.add_argument("--gen_dir", required=True)
    parser.add_argument("--glosses", nargs="+", default=["before", "cool", "drink", "go", "thin"])
    parser.add_argument("--output", default="comparison.gif")
    parser.add_argument("--cell_size", type=int, default=200)
    parser.add_argument("--max_frames", type=int, default=80)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    cell = args.cell_size
    title_h = 32   # gloss title bar height
    label_h = 20   # "Fitted" / "Generated" label height
    gap = 16       # vertical gap between groups

    font_title = get_font(18)
    font_label = get_font(13)

    # ── Load data ──
    data = {}
    max_fit_count = 0

    for g in args.glosses:
        fit_list = find_all_fit_gifs(args.fit_dir, g)
        gen_path = find_gen_gif(args.gen_dir, g)

        fit_data = []
        for vid_id, fp in fit_list:
            frames = load_gif_frames(fp, args.max_frames)
            fit_data.append((vid_id, resize_frames(frames, (cell, cell))))
            print(f"  [{g}] fit {vid_id}: {len(frames)} frames")

        gen_data = None
        if gen_path:
            frames = load_gif_frames(gen_path, args.max_frames)
            gen_data = resize_frames(frames, (cell, cell))
            print(f"  [{g}] gen: {len(frames)} frames")
        else:
            print(f"  [{g}] gen: NOT FOUND")

        data[g] = {"fit": fit_data, "gen": gen_data}
        max_fit_count = max(max_fit_count, len(fit_data))

    # ── Layout calculation ──
    fit_cols = 3
    fit_rows = 2  # 2 rows x 3 cols for fitted
    n_cols = fit_cols
    W = n_cols * cell + 10

    group_h = title_h + label_h + fit_rows * cell + label_h + cell
    H = len(args.glosses) * group_h + (len(args.glosses) - 1) * gap

    # Total frames (longest GIF determines loop length)
    all_frame_counts = []
    for g in args.glosses:
        for _, fr in data[g]["fit"]:
            all_frame_counts.append(len(fr))
        if data[g]["gen"]:
            all_frame_counts.append(len(data[g]["gen"]))
    n_frames = max(all_frame_counts) if all_frame_counts else 1

    # Placeholder
    placeholder = Image.new("RGBA", (cell, cell), (50, 50, 50, 255))
    d = ImageDraw.Draw(placeholder)
    d.text((cell // 2 - 15, cell // 2 - 8), "N/A", fill=(120, 120, 120, 255))

    # ── Build frames ──
    output_frames = []
    for fi in range(n_frames):
        canvas = Image.new("RGBA", (W, H), (20, 20, 20, 255))
        draw = ImageDraw.Draw(canvas)

        y_offset = 0
        for gi, g in enumerate(args.glosses):
            # Title bar
            draw.rectangle([0, y_offset, W, y_offset + title_h], fill=(60, 60, 60, 255))
            bbox = draw.textbbox((0, 0), g.upper(), font=font_title)
            tw = bbox[2] - bbox[0]
            draw.text(((W - tw) // 2, y_offset + 6), g.upper(), fill=(255, 255, 255), font=font_title)
            y_offset += title_h

            # Row 1: Fitted (2 rows x 3 cols grid)
            draw.text((4, y_offset + 2), "Fitted", fill=(180, 180, 180), font=font_label)
            y_offset += label_h

            fit_data = data[g]["fit"]
            if fit_data:
                for ci, (vid_id, frames) in enumerate(fit_data):
                    row_i = ci // fit_cols
                    col_i = ci % fit_cols
                    x = col_i * cell
                    y = y_offset + row_i * cell
                    idx = fi % len(frames)
                    canvas.paste(frames[idx], (x, y))
                    # Video ID label
                    vid_label = str(vid_id)
                    lb = draw.textbbox((0, 0), vid_label, font=font_label)
                    lw = lb[2] - lb[0]
                    draw.rectangle(
                        [x, y + cell - 18, x + lw + 8, y + cell],
                        fill=(0, 0, 0, 160)
                    )
                    draw.text((x + 4, y + cell - 17), vid_label,
                              fill=(200, 200, 200), font=font_label)
            else:
                canvas.paste(placeholder, (0, y_offset))
            y_offset += fit_rows * cell

            # Row 2: Generated
            draw.text((4, y_offset + 2), "Generated", fill=(180, 180, 180), font=font_label)
            y_offset += label_h

            gen_frames = data[g]["gen"]
            if gen_frames:
                idx = fi % len(gen_frames)
                canvas.paste(gen_frames[idx], (0, y_offset))
            else:
                canvas.paste(placeholder, (0, y_offset))
            y_offset += cell

            y_offset += gap

        output_frames.append(canvas.convert("RGB"))

    imageio.mimsave(args.output, output_frames, fps=args.fps, loop=0)
    print(f"\nSaved {args.output} ({len(output_frames)} frames, {W}x{H}px)")


if __name__ == "__main__":
    main()