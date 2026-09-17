"""Segmenter validation on SignBank citation videos (replaces the ASLLRP boundary-F1 check, user 09-14).

Each citation video contains exactly one sign, giving two tests without manual annotation:
  1. single-sign check: the segmenter should yield exactly 1 non-rest segment per video.
  2. synthetic-concatenation boundary F1: concatenate K rest-trimmed signs; joins are ground-truth
     boundaries; score predicted boundaries at +-tol frames.
Caveats reported with the numbers: citation-form transitions (hand-drop between signs) are easier
than real co-articulation, and rest trimming reuses the segmenter's rest criterion (documented proxy).

  python align/validate_segmenter_signbank.py --fused <dir> --out <json> [--nseq 500 --k 8]
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segment_clips import segment, gauss_smooth, WRIST_L, WRIST_R, PELVIS, NECK, FING_L, FING_R


def hand_speed(j3d, fps):
    handL = 0.5 * j3d[:, WRIST_L] + 0.5 * j3d[:, FING_L].mean(1)
    handR = 0.5 * j3d[:, WRIST_R] + 0.5 * j3d[:, FING_R].mean(1)
    sL = np.linalg.norm(np.diff(handL, axis=0), axis=-1) * fps
    sR = np.linalg.norm(np.diff(handR, axis=0), axis=-1) * fps
    return gauss_smooth(np.maximum(np.append(sL, sL[-1:]), np.append(sR, sR[-1:])))


def trim_rest(j3d, fps, rest_band=0.25, rest_speed=0.15):
    """Cut leading/trailing frames where both hands are in the low band and slow (same criterion as segment())."""
    s = hand_speed(j3d, fps)
    handL = 0.5 * j3d[:, WRIST_L] + 0.5 * j3d[:, FING_L].mean(1)
    handR = 0.5 * j3d[:, WRIST_R] + 0.5 * j3d[:, FING_R].mean(1)
    lo = j3d[:, NECK, 1] + (1 - rest_band) * (j3d[:, PELVIS, 1] - j3d[:, NECK, 1])
    rest = (handL[:, 1] > lo) & (handR[:, 1] > lo) & (s < rest_speed)
    act = np.where(~rest)[0]
    if len(act) < 4: return None
    return j3d[act[0]:act[-1] + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fused', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--nseq', type=int, default=500); ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--tols', type=int, nargs='+', default=[2, 4]); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.fused, '*.npz')))
    clips, fps_all = {}, []
    n_bad = 0
    for f in files:
        d = np.load(f); j = d['joints_3d']; fps = float(d['fps']) if 'fps' in d else 25.0
        if len(j) < 4 or not np.isfinite(j).all(): n_bad += 1; continue
        clips[os.path.splitext(os.path.basename(f))[0]] = (j.astype(np.float32), fps); fps_all.append(fps)
    print(f'loaded {len(clips)} clips ({n_bad} bad), fps median {np.median(fps_all):g}', flush=True)

    # --- test 1: single-sign check ---
    counts = {}
    for g, (j, fps) in clips.items():
        bounds, rest = segment(j, fps)
        counts[g] = sum(1 for r in rest if r == 0)
    arr = np.array(list(counts.values()))
    t1 = {'n': int(len(arr)), 'exact1': float((arr == 1).mean()), 'zero': float((arr == 0).mean()),
          'over2': float((arr == 2).mean()), 'over3plus': float((arr >= 3).mean()),
          'mean_nonrest_segs': float(arr.mean())}
    print('single-sign:', json.dumps(t1), flush=True)
    worst = sorted(counts, key=counts.get)[:5] + sorted(counts, key=counts.get)[-5:]
    t1['examples_low_high'] = {g: int(counts[g]) for g in worst}

    # --- test 2: synthetic-concatenation boundary F1 ---
    rng = np.random.default_rng(a.seed)
    trimmed = {}
    for g, (j, fps) in clips.items():
        t = trim_rest(j, fps)
        if t is not None and len(t) >= 8: trimmed[g] = (t, fps)
    keys = sorted(trimmed)
    print(f'trimmed usable: {len(keys)}', flush=True)
    stats = {tol: [0, 0, 0] for tol in a.tols}  # tp, npred, ngt
    for _ in range(a.nseq):
        picks = rng.choice(len(keys), size=a.k, replace=False)
        parts, gt, off = [], [], 0
        anchor = None
        for i in picks:
            j, fps = trimmed[keys[i]]
            j = j.copy()
            if anchor is None: anchor = j[:, PELVIS].mean(0)
            else: j = j + (anchor - j[:, PELVIS].mean(0))  # kill inter-video framing offsets
            parts.append(j)
            off += len(j); gt.append(off)
        gt = gt[:-1]  # last offset is sequence end, not a join
        seq = np.concatenate(parts, 0)
        bounds, rest = segment(seq, 25.0)
        pred = [b for b in bounds if 0 < b < len(seq)]
        for tol in a.tols:
            used = set(); tp = 0
            for g0 in gt:
                cand = [p for p in pred if abs(p - g0) <= tol and p not in used]
                if cand:
                    p = min(cand, key=lambda x: abs(x - g0)); used.add(p); tp += 1
            stats[tol][0] += tp; stats[tol][1] += len(pred); stats[tol][2] += len(gt)
    t2 = {}
    for tol, (tp, npred, ngt) in stats.items():
        prec = tp / max(npred, 1); rec = tp / max(ngt, 1)
        t2[f'tol{tol}'] = {'precision': round(prec, 4), 'recall': round(rec, 4),
                           'f1': round(2 * prec * rec / max(prec + rec, 1e-9), 4),
                           'pred_per_seq': round(npred / a.nseq, 2), 'gt_per_seq': round(ngt / a.nseq, 2)}
    print('concat F1:', json.dumps(t2), flush=True)

    out = {'fused_dir': a.fused, 'n_clips': len(clips), 'n_bad': n_bad,
           'single_sign': t1, 'concat': {'nseq': a.nseq, 'k': a.k, **t2}}
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out, flush=True)


if __name__ == '__main__': main()
