"""Phrase -> segment retrieval on RUNNING signing (held-out clips), the metric between the SignBank probe (citation
form) and eval_retrieval_v2 (whole-clip stitching). For every chunk of every pool_val clip, rank all non-rest segments
of the val clips (same-clip segments = positives, MIL-style) by aligner similarity; report R@1/R@5/R@10 (any positive
in top-k), chance = mean(S_clip / N_segments), plus the same for the reverse direction (segment -> chunk).
  python align/eval_seg_retrieval.py --align_ckpt <pt> --val_list <txt> --out <json>
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn.functional as F
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from torch.utils.data import DataLoader
from align.align_dataset_v2 import ClipAlignDataset, collate
from align.train_align_v6 import AlignModel

DR = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/alignment'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--align_ckpt', required=True); ap.add_argument('--val_list', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--tok_dir', default='/projects/kosecka/hongrui/dataset/smplx_fits/pooled_tokens/vq_K512_w64'); ap.add_argument('--chunk_dir', default=f'{DR}/gloss'); ap.add_argument('--merge', type=int, default=1); ap.add_argument('--spots', default='')
    a = ap.parse_args(); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.align_ckpt, map_location='cpu'); c = ck['config']
    model = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']).to(dev); model.load_state_dict(ck['model']); model.eval()
    ds = ClipAlignDataset(a.tok_dir, a.val_list, a.chunk_dir, train=False, max_items=2000)
    if a.merge > 1:  # sliding windows of k adjacent segments (stride 1) as retrieval units, frame spans merged
        ds.items = [(tok, ch, [(ss[i][0], ss[min(i + a.merge, len(ss)) - 1][1]) for i in range(max(1, len(ss) - a.merge + 1))]) for tok, ch, ss in ds.items]
    dl = DataLoader(ds, 32, shuffle=False, num_workers=4, collate_fn=collate)
    ZT, ZM, TO, MO = [], [], [], []; base = 0
    with torch.no_grad():
        for x, m, spans, seg_own, chunks, chk_own in dl:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                zs = model.embed_segments(x.to(dev), m.to(dev), spans); zc = model.embed_chunks(chunks, dev)
            ZT.append(zc.float().cpu()); ZM.append(zs.float().cpu()); TO.append(chk_own + base); MO.append(seg_own + base); base += int(m.shape[0])
    zt = F.normalize(torch.cat(ZT), dim=-1).to(dev); zm = F.normalize(torch.cat(ZM), dim=-1).to(dev); to = torch.cat(TO).to(dev); mo = torch.cat(MO).to(dev)
    sim = zt @ zm.T  # (Nt, Nm)
    res = {'n_clips': base, 'n_chunks': int(len(zt)), 'n_segments': int(len(zm))}
    if a.spots:  # gloss-consistent positives: chunk i <-> segment j positive if segment j is spotted with a gloss whose words are in chunk i
        from align.spot_signs import caption_words, norm, STOP
        import csv as _csv
        cids = [os.path.basename(t)[:-4] for t, _, _ in ds.items]; back = {l.strip().replace(':', '__'): l.strip() for l in open(a.val_list) if l.strip()}
        cid2i = {back[c]: i for i, c in enumerate(cids)}; seg_gloss = {}
        for r in _csv.DictReader(open(a.spots), delimiter='\t'):
            if r['clip_id'] in cid2i: seg_gloss.setdefault((cid2i[r['clip_id']], int(r['seg_idx'])), set()).add(r['gloss'])
        seg_local = torch.cat([torch.arange(int((mo == i).sum())) for i in range(base)]) if len(mo) else mo
        glosses = sorted({g for v in seg_gloss.values() for g in v}); gi = {g: k for k, g in enumerate(glosses)}
        S = torch.zeros(len(zm), len(glosses), dtype=torch.bool)
        for (ci, j), gs in seg_gloss.items():
            rows_ = ((mo.cpu() == ci) & (seg_local == j)).nonzero().flatten()
            for r_ in rows_:
                for g in gs: S[r_, gi[g]] = True
        chunk_texts = [c for _, ch, _ in ds.items for c in ch]
        T = torch.zeros(len(zt), len(glosses), dtype=torch.bool)
        gw = [[w for w in norm(g) if w not in STOP] for g in glosses]
        for i, c in enumerate(chunk_texts):
            cw = caption_words(c)
            for k, ws in enumerate(gw):
                if ws and all(w in cw for w in ws): T[i, k] = True
        pos_g = (T.to(dev).float() @ S.to(dev).float().T) > 0  # (Nt, Nm) gloss-consistent
        has = pos_g.any(1); res['gloss_pos'] = {'n_glosses': len(glosses), 'n_spotted_segments': int(S.any(1).sum()), 'n_chunks_with_pos': int(has.sum())}
        first, ranks = [], []
        for i in range(0, len(zt), 1024):
            s_ = sim[i:i + 1024]; p_ = pos_g[i:i + 1024]
            best = torch.where(p_, s_, torch.full_like(s_, -2.0)).max(1).values; first.append((s_ > best[:, None]).sum(1))
        first = torch.cat(first)[has]
        res['chunk2seg_gloss'] = {f'R{k}': float((first < k).float().mean()) for k in (1, 5, 10, 50)} | {'medR': int(first.median()) + 1, 'chance_R1': float(pos_g[has].float().mean())}
        print('chunk2seg_gloss', json.dumps(res['chunk2seg_gloss']), res['gloss_pos'], flush=True)
    for name, S, qo, co in [('chunk2seg', sim, to, mo), ('seg2chunk', sim.T, mo, to)]:
        first, chance = [], []
        for i in range(0, len(qo), 1024):  # rank of the best positive = #candidates scoring above it (no full argsort)
            s = S[i:i + 1024]; pos = (qo[i:i + 1024, None] == co[None, :])
            best = torch.where(pos, s, torch.full_like(s, -2.0)).max(1).values
            first.append((s > best[:, None]).sum(1)); chance.append(pos.float().mean(1))
        first = torch.cat(first); chance = torch.cat(chance)
        res[name] = {f'R{k}': float((first < k).float().mean()) for k in (1, 5, 10, 50)} | {'medR': int(first.median()) + 1, 'chance_R1': float(chance.mean())}
        print(name, json.dumps(res[name]), flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1); print('wrote', a.out)


if __name__ == '__main__': main()
