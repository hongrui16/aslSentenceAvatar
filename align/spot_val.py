"""Fixed-spotter pseudo labels on the val clips: embed every non-rest segment of --val_list with a FIXED aligner (the
v4 anchor model), nearest SignBank gloss text (all 2,820), accept when the gloss word(s) occur in the clip caption and
cos >= --min_cos. Output tsv (clip_id, seg_idx, fr_a, fr_b, gloss, cos) used by eval_seg_retrieval.py --spots so that
"positive" can include segments in OTHER clips carrying the same gloss as a chunk's words (cross-clip phrase positives).
  python align/spot_val.py --align_ckpt <anchor best> --val_list <txt> --out <tsv>
"""
import argparse, csv, os, sys
import numpy as np, torch, torch.nn.functional as F
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from torch.utils.data import DataLoader
from align.align_dataset import ClipAlignDataset, collate, INDEX
from align.train_align import AlignModel
from align.train_align_v4 import SignBank, DR
from align.spot_signs import caption_words, norm, STOP


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--align_ckpt', required=True); ap.add_argument('--val_list', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--tok_dir', default='/projects/kosecka/hongrui/dataset/smplx_fits/pooled_tokens/vq_K512_w64'); ap.add_argument('--chunk_dir', default=f'{DR}/chunks')
    ap.add_argument('--min_cos', type=float, default=0.4); a = ap.parse_args(); dev = 'cuda'
    ck = torch.load(a.align_ckpt, map_location='cpu'); c = ck['config']
    model = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']).to(dev); model.load_state_dict(ck['model']); model.eval()
    sb = SignBank(f'{DR}/signbank_tokens.npz', f'{DR}/signbank_split.json', model.mmm.pos.num_embeddings)
    idx = list(range(len(sb.stems))); Z = []
    with torch.no_grad():
        for i in range(0, len(idx), 128):
            x, m, spans, _ = sb.batch(idx[i:i + 128])
            with torch.autocast('cuda', dtype=torch.bfloat16): Z.append(model.embed_segments(x.to(dev), m.to(dev), spans).float())
    Z = torch.cat(Z); texts = [sb.texts[i] for i in idx]; uniq = sorted(set(texts)); ti = {t: k for k, t in enumerate(uniq)}
    G = torch.zeros(len(uniq), Z.shape[1], device=dev); cnt = torch.zeros(len(uniq), device=dev)
    for z, t in zip(Z, texts): G[ti[t]] += z; cnt[ti[t]] += 1
    G = F.normalize(G / cnt[:, None], dim=-1); gw = [[w for w in norm(t) if w not in STOP] for t in uniq]
    rows = {}
    with open(INDEX) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE): rows[r['clip_id']] = r['text'].strip()
    ds = ClipAlignDataset(a.tok_dir, a.val_list, a.chunk_dir, train=False, max_items=2000)
    cids = [os.path.basename(t)[:-4] for t, _, _ in ds.items]
    back = {l.strip().replace(':', '__'): l.strip() for l in open(a.val_list) if l.strip()}
    dl = DataLoader(ds, 32, shuffle=False, num_workers=4, collate_fn=collate); n_acc = 0; base = 0
    with open(a.out, 'w') as fo, torch.no_grad():
        fo.write('clip_id\tseg_idx\tfr_a\tfr_b\tgloss\tcos\n')
        for x, m, spans, seg_own, chunks, chk_own in dl:
            with torch.autocast('cuda', dtype=torch.bfloat16): zs = model.embed_segments(x.to(dev), m.to(dev), spans).float()
            sim = F.normalize(zs, dim=-1) @ G.T; cos, g = sim.max(1); cos = cos.cpu().numpy(); g = g.cpu().numpy()
            local = {}
            for k, (bi, ta, tb) in enumerate(spans):
                cid = back[cids[base + bi]]; j = local.get(bi, 0); local[bi] = j + 1
                if cos[k] < a.min_cos or not gw[g[k]]: continue
                if all(w in caption_words(rows.get(cid, '')) for w in gw[g[k]]):
                    fa, fb = ds.items[base + bi][2][j]; fo.write(f'{cid}\t{j}\t{fa}\t{fb}\t{uniq[g[k]]}\t{cos[k]:.4f}\n'); n_acc += 1
            base += int(m.shape[0])
    print('val spots accepted', n_acc, '->', a.out)


if __name__ == '__main__': main()
