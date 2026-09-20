"""Tokenize the ASL Citizen fits (gloss-level isolated signs, 83,399 webcam clips, 2,731 signs, 52 signers) into ONE npz, like
tokenize_signbank.py does for SignBank. LICENCE: Microsoft Research licence = in-house research only. The videos, the fits and this
token file must never be redistributed and must never enter a bank that produces output motion (encoder / judge training only).

Per clip:
  - label from the official split csv (Participant ID, Video file, Gloss, ASL-LEX Code); the official signer-disjoint split is kept;
  - text = query word(s) from the FILE NAME ('7974224982720846-NOT MIND.mp4' -> 'not mind', 'seedCORN 2' -> 'corn'), because the
    Gloss column glues words together (NOTMIND) and carries variant digits (SAIL1);
  - trim = the span in which WiLoR sees at least one hand (+- --margin frames at 25 fps). Signers start and end with the hands out of
    frame, so this span is the sign itself. Clips without any detection keep their full length and get has_hands = 0;
  - lr_energy = mean frame-to-frame pose change of the left / right arm+fingers inside the trim (handedness / mirrored-webcam check;
    nothing is mirrored here, the decision is left to the training code).
Output npz: stems, gloss, text, asllex, participant, split, tokens (sum n_tok, 4) int16, offsets (N+1), trim (N,2), T (N,), has_hands (N,),
lr_energy (N,2).
  python align/tokenize_asl_citizen.py --root <smplx_fits/asl_citizen> --splits <.../ASL_Citizen/splits> --vq_ckpt <pt> --out <npz>
"""
import argparse, csv, os, re, sys
import numpy as np, torch
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from pretrain.part_vqvae import PartVQVAE
from pretrain.motion_window_dataset import aa_to_rot6d_np
PARTS = ['body', 'lhand', 'rhand', 'face']
L_ARM, R_ARM = [16, 18, 20] + list(range(22, 37)), [17, 19, 21] + list(range(37, 52))  # 53-joint order: root 0, body 1-21, lhand 22-36, rhand 37-51, jaw 52


def text_from_file(fn):
    s = os.path.splitext(fn)[0].split('-', 1)[-1]
    s = re.sub(r'^seed', '', s); s = re.sub(r'\s*\d+$', '', s.strip())
    return re.sub(r'\s+', ' ', s.replace('_', ' ')).strip().lower()


def feats(aa, ex):
    aa = aa.astype(np.float32).copy(); ex = ex.astype(np.float32); aa[:, 0] = 0.0; T = len(aa); pad = (-T) % 4
    if pad: aa = np.concatenate([aa, np.repeat(aa[-1:], pad, 0)]); ex = np.concatenate([ex, np.repeat(ex[-1:], pad, 0)])
    r6 = aa_to_rot6d_np(aa); W = len(aa)
    return {'body': r6[:, 1:22].reshape(W, -1), 'lhand': r6[:, 22:37].reshape(W, -1), 'rhand': r6[:, 37:52].reshape(W, -1), 'face': np.concatenate([r6[:, 52], ex], -1)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True); ap.add_argument('--splits', required=True)
    ap.add_argument('--vq_ckpt', required=True); ap.add_argument('--out', required=True); ap.add_argument('--margin', type=int, default=3)
    ap.add_argument('--min_frames', type=int, default=8); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.vq_ckpt, map_location='cpu'); cfg = ck['config']
    vq = PartVQVAE(ck['part_dims'], cfg['K'], cfg['width'], cfg['down']).to(dev); vq.load_state_dict(ck['model']); vq.eval()
    rows = []
    for sp in ['train', 'val', 'test']:
        with open(os.path.join(a.splits, f'{sp}.csv'), newline='') as f:
            for r in csv.DictReader(f): rows.append((sp, r['Participant ID'], r['Video file'], r['Gloss'], r['ASL-LEX Code']))
    if a.limit: rows = [r for r in rows if os.path.exists(os.path.join(a.root, 'fused', os.path.splitext(r[2])[0] + '.npz'))][:a.limit]
    out = {k: [] for k in ['stems', 'gloss', 'text', 'asllex', 'participant', 'split', 'trim', 'T', 'has_hands', 'lr_energy']}; toks, offs = [], [0]; miss = short = 0
    for n, (sp, pid, fn, gl, lex) in enumerate(rows):
        stem = os.path.splitext(fn)[0]; fp = os.path.join(a.root, 'fused', stem + '.npz')
        if not os.path.exists(fp): miss += 1; continue
        d = np.load(fp); aa, ex, T = d['axis_angle'], d['expression'], int(d['T']); lo, hi, hh = 0, T, 0
        wp = os.path.join(a.root, 'wilor', stem + '.npz')
        if os.path.exists(wp):
            w = np.load(wp); det = np.where((w['left_score'] > 0) | (w['right_score'] > 0))[0]
            if len(det):
                k = float(d['fps']) / max(float(d['src_fps']), 1e-6); hh = 1
                lo = max(0, int(np.floor(det[0] * k)) - a.margin); hi = min(T, int(np.ceil((det[-1] + 1) * k)) + a.margin)
        if hi - lo < a.min_frames: lo, hi = 0, T
        if hi - lo < a.min_frames: short += 1; continue
        f = feats(aa[lo:hi], ex[lo:hi])
        with torch.no_grad(): idx = vq.encode({p: torch.from_numpy(v)[None].to(dev) for p, v in f.items()})
        t = torch.stack([idx[p][0] for p in PARTS], -1).cpu().numpy().astype(np.int16); toks.append(t); offs.append(offs[-1] + len(t))
        dv = np.abs(np.diff(aa[lo:hi], axis=0)).mean(axis=(0, 2))
        for k_, v in zip(out, [stem, gl, text_from_file(fn), lex, pid, sp, (lo, hi), T, hh, (float(dv[L_ARM].mean()), float(dv[R_ARM].mean()))]): out[k_].append(v)
        if (n + 1) % 5000 == 0: print(n + 1, len(toks), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or '.', exist_ok=True)
    np.savez(a.out, tokens=np.concatenate(toks), offsets=np.array(offs, np.int64), trim=np.array(out['trim'], np.int32), T=np.array(out['T'], np.int32),
             has_hands=np.array(out['has_hands'], np.int8), lr_energy=np.array(out['lr_energy'], np.float32),
             **{k: np.array(out[k]) for k in ['stems', 'gloss', 'text', 'asllex', 'participant', 'split']})
    le = np.array(out['lr_energy']); tl = np.diff(np.array(out['trim']), axis=1)[:, 0]
    print(f'clips {len(toks)} (missing fit {miss}, too short {short}) | texts {len(set(out["text"]))} glosses {len(set(out["gloss"]))} | split '
          f'{ {s: out["split"].count(s) for s in ["train", "val", "test"]} } | trim frames median {np.median(tl):.0f} | no-hand clips {int((np.array(out["has_hands"]) == 0).sum())} | '
          f'left-dominant share {float((le[:, 0] > le[:, 1]).mean()):.3f}', flush=True)


if __name__ == '__main__':
    main()
