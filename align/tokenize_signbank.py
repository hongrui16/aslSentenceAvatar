"""Tokenize the SignBank citation-form fits once (VQ ids per part) + gloss text + an 80/20 split by gloss text,
so the anchor loss (train_align.py --anchor) and the held-out probe share one file.
  python align/tokenize_signbank.py --fused <dir> --vq_ckpt <pt> --out <npz> --split <json>
"""
import argparse, glob, json, os, random, sys
import numpy as np, torch
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from pretrain.part_vqvae import PartVQVAE
from pretrain.tokenize_corpus import clip_feats
from align.probe_signbank import gloss_text, PARTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fused', required=True); ap.add_argument('--vq_ckpt', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--split', required=True); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.vq_ckpt, map_location='cpu'); cfg = ck['config']
    vq = PartVQVAE(ck['part_dims'], cfg['K'], cfg['width'], cfg['down']).to(dev); vq.load_state_dict(ck['model']); vq.eval()
    stems, toks, lens = [], [], []
    with torch.no_grad():
        for f in sorted(glob.glob(os.path.join(a.fused, '*.npz'))):
            try:
                feats, T = clip_feats(f); idx = vq.encode({p: torch.from_numpy(v)[None].to(dev) for p, v in feats.items()})
            except Exception: continue
            t = torch.stack([idx[p][0] for p in PARTS], -1).cpu().numpy().astype(np.int16)
            stems.append(os.path.splitext(os.path.basename(f))[0]); toks.append(t); lens.append(len(t))
    off = np.cumsum([0] + lens); flat = np.concatenate(toks)
    texts = [gloss_text(s) for s in stems]
    np.savez(a.out, stems=np.array(stems), texts=np.array(texts), tokens=flat, offsets=off)
    uniq = sorted(set(texts)); random.Random(a.seed).shuffle(uniq); n_ho = len(uniq) // 5
    held = set(uniq[:n_ho]); split = {'train_texts': sorted(set(uniq[n_ho:])), 'heldout_texts': sorted(held),
                                      'train_stems': [s for s, t in zip(stems, texts) if t not in held], 'heldout_stems': [s for s, t in zip(stems, texts) if t in held]}
    json.dump(split, open(a.split, 'w'), indent=1)
    print(f'{len(stems)} signs, {len(uniq)} unique texts, heldout {len(split["heldout_stems"])} stems / {n_ho} texts -> {a.out}, {a.split}')


if __name__ == '__main__': main()
