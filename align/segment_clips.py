"""Changepoint motion segmentation for roadmap 4.2 (velocity minima + hand-drop), pure numpy on fused SMPL-X npz.

Boundaries = local minima of the smoothed faster-hand speed; runs where both hands rest near the hips are cut out
entirely (rest segments are not signs). Segments shorter than --min_dur are merged into the weaker-boundary side.
FAST is the fallback segmenter if this proves too coarse (needs raw WiLoR, not these fused fits).

  python align/segment_clips.py --list <clip ids> --out <tsv> [--shard 0 --nshards 1]
Output tsv: clip_id, T, fps, boundaries (comma-joined frame idx incl. 0 and T), rest_mask (comma-joined 0/1 per segment).
"""
import argparse, csv, os, sys
import numpy as np

INDEX = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/curation/clip_index.tsv'
WRIST_L, WRIST_R, PELVIS, NECK = 20, 21, 0, 12
FING_L, FING_R = slice(25, 40), slice(40, 55)


def gauss_smooth(x, sigma=1.5):
    r = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    return np.convolve(np.pad(x, r, mode='edge'), k, mode='valid')


def segment(j3d, fps, min_dur=0.32, rest_band=0.25, rest_speed=0.15):
    """j3d (T,144,3) camera frame (y down). Returns boundaries [0..T] and per-segment rest flag."""
    T = len(j3d)
    handL = 0.5 * j3d[:, WRIST_L] + 0.5 * j3d[:, FING_L].mean(1); handR = 0.5 * j3d[:, WRIST_R] + 0.5 * j3d[:, FING_R].mean(1)
    sL = np.linalg.norm(np.diff(handL, axis=0), axis=-1) * fps; sR = np.linalg.norm(np.diff(handR, axis=0), axis=-1) * fps
    s = gauss_smooth(np.maximum(np.append(sL, sL[-1:]), np.append(sR, sR[-1:])))  # m/s, faster hand
    # rest = both hands in the lower band between neck and pelvis (y down: bigger y = lower) AND slow
    lo = j3d[:, NECK, 1] + (1 - rest_band) * (j3d[:, PELVIS, 1] - j3d[:, NECK, 1])
    rest = (handL[:, 1] > lo) & (handR[:, 1] > lo) & (s < rest_speed)
    # boundary candidates: interior local minima of s
    b = [t for t in range(1, T - 1) if s[t] <= s[t - 1] and s[t] <= s[t + 1]]
    bounds = sorted(set([0, T] + b + [t for t in range(1, T) if rest[t] != rest[t - 1]]))
    # merge segments shorter than min_dur into the neighbour across the weaker boundary (rest transitions are strong)
    min_f = max(2, int(round(min_dur * fps))); strong = set(t for t in range(1, T) if rest[t] != rest[t - 1])
    changed = True
    while changed and len(bounds) > 2:
        changed = False
        for i in range(len(bounds) - 1):
            if bounds[i + 1] - bounds[i] < min_f:
                l, r = bounds[i], bounds[i + 1]
                if l == 0: drop = r
                elif r == T: drop = l
                else:
                    wl = np.inf if l in strong else s[l]; wr = np.inf if r in strong else s[r]
                    drop = l if wl >= wr else r
                if drop in (0, T): break
                bounds.remove(drop); changed = True; break
    seg_rest = [int(rest[a:c].mean() > 0.5) for a, c in zip(bounds[:-1], bounds[1:])]
    return bounds, seg_rest


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--list', required=True); ap.add_argument('--index', default=INDEX); ap.add_argument('--out', required=True)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1); ap.add_argument('--min_dur', type=float, default=0.32)
    a = ap.parse_args()
    wanted = [l.strip() for l in open(a.list) if l.strip()][a.shard::a.nshards]; wset = set(wanted); rows = {}
    with open(a.index) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            if r['clip_id'] in wset: rows[r['clip_id']] = (r['npz'], float(r['fps']))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); n = 0
    with open(a.out, 'w') as fo:
        fo.write('clip_id\tT\tfps\tboundaries\trest\n')
        for cid in wanted:
            if cid not in rows: continue
            npz, fps = rows[cid]
            try:
                j = np.load(npz)['joints_3d']
                if len(j) < 4 or not np.isfinite(j).all(): raise ValueError('bad joints')
                bounds, rest = segment(j, fps, a.min_dur)
            except Exception as e:
                print('skip', cid, e, flush=True); continue
            fo.write(f"{cid}\t{len(j)}\t{fps:g}\t{','.join(map(str, bounds))}\t{','.join(map(str, rest))}\n"); n += 1
            if n % 20000 == 0: print(a.shard, n, flush=True)
    print(a.shard, 'DONE', n, flush=True)


if __name__ == '__main__': main()
