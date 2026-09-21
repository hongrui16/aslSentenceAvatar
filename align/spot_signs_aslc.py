"""Sign spotting with ASL Citizen exemplars (2026-09-21). First test of "does ASL Citizen help".
Same recipe as spot_signs_v2.py (nearest labelled sign in the aligner's motion space + the gloss word must occur in the clip's caption),
but the labelled signs are the ASL Citizen clips (2,731 signs, ~30 clips per sign, 52 signers) instead of ONE SignBank citation video per
gloss. Each bank segment takes the label of its nearest ASL Citizen exemplar (exemplar-level NN, so left-handed or variant renditions
stay usable; no prototype averaging). Official train + val signers are the exemplars; the test signers are left untouched for a judge.
LICENCE: ASL Citizen is used in-house as supervision only. The output contains segments of OUR corpora, no ASL Citizen data.
  python align/spot_signs_aslc.py --align_ckpt <anchor pt> --bank_dir <seg_bank_v4_anchor> --out <tsv> --stats <json>
"""
import argparse, csv, glob, json, os, sys
import numpy as np, torch, torch.nn.functional as F
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from align.align_dataset import INDEX
from align.train_align import AlignModel
from align.spot_signs_v2 import STOP, norm, caption_words
ASLC = '/scratch/rhong5/dataset/pooled_tokens/asl_citizen_tokens_v1.npz'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--align_ckpt', required=True); ap.add_argument('--bank_dir', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--stats', required=True); ap.add_argument('--aslc', default=ASLC)
    ap.add_argument('--splits', nargs='+', default=['train', 'val']); ap.add_argument('--min_cos', type=float, default=0.4); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); dev = 'cuda'
    ck = torch.load(a.align_ckpt, map_location='cpu'); c = ck['config']
    model = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']).to(dev); model.load_state_dict(ck['model']); model.eval(); max_len = model.mmm.pos.num_embeddings
    d = np.load(a.aslc, allow_pickle=True); keep = np.where(np.isin(d['split'], a.splits) & (d['has_hands'] > 0))[0]; off = d['offsets']; tok = d['tokens']
    texts = [str(d['text'][i]) for i in keep]; uniq = sorted(set(texts)); ti = {t: k for k, t in enumerate(uniq)}
    lab = torch.tensor([ti[t] for t in texts], device=dev); Z = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(keep), 128):
            ts = [torch.from_numpy(tok[off[j]:off[j + 1]].astype(np.int64))[:max_len] for j in keep[i:i + 128]]; L = max(len(t) for t in ts)
            x = torch.zeros(len(ts), L, 4, dtype=torch.long); m = torch.zeros(len(ts), L, dtype=torch.bool); spans = []
            for j, t in enumerate(ts): x[j, :len(t)] = t; m[j, :len(t)] = True; spans.append((j, 0, len(t)))
            Z.append(model.embed_segments(x.to(dev), m.to(dev), spans).float())
    Z = F.normalize(torch.cat(Z), dim=-1).half(); gloss_words = [[w for w in norm(t) if w not in STOP] for t in uniq]
    print(f'{len(keep)} ASL Citizen exemplars, {len(uniq)} gloss texts', flush=True)
    rows = {}
    with open(INDEX) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE): rows[r['clip_id']] = r['text'].strip()
    rng = np.random.default_rng(a.seed); all_clips = list(rows); shuf = {cid: rows[all_clips[j]] for cid, j in zip(all_clips, rng.permutation(len(all_clips)))}
    ths = [0.3, 0.4, 0.5, 0.6, 0.7]; st = {str(t): {'n': 0, 'hit_true': 0, 'hit_shuf': 0} for t in ths}; n_all = n_acc = 0; capw = {}
    with open(a.out, 'w') as fo:
        fo.write('clip_id\tfr_a\tfr_b\tgloss\tcos\tmargin\n')
        for fp in sorted(glob.glob(os.path.join(a.bank_dir, 'bank_*.npz'))):
            b = np.load(fp); E = torch.from_numpy(b['emb']).to(dev); meta = b['meta']; cids = [str(x) for x in b['clip_ids']]
            for i in range(0, len(E), 20000):
                e = F.normalize(E[i:i + 20000].float(), dim=-1).half(); sim = e @ Z.T; top = sim.max(1); g = lab[top.indices]
                sim.scatter_(1, top.indices[:, None], -2.0); sim[lab[None, :] == g[:, None]] = -2.0  # margin = best exemplar of a DIFFERENT gloss
                cos = top.values.float().cpu().numpy(); mar = cos - sim.max(1).values.float().cpu().numpy(); g = g.cpu().numpy()
                for k in np.where(cos >= ths[0])[0]:
                    ci, _, _, fa, fb = meta[i + k]; cid = cids[ci]; n_all += 1
                    if cid not in rows: continue
                    gw = gloss_words[g[k]]
                    if not gw: continue
                    if cid not in capw: capw[cid] = (caption_words(rows[cid]), caption_words(shuf[cid]))
                    wt, ws = capw[cid]; ht = all(w in wt for w in gw); hs = all(w in ws for w in gw)
                    for t in ths:
                        if cos[k] >= t: s_ = st[str(t)]; s_['n'] += 1; s_['hit_true'] += ht; s_['hit_shuf'] += hs
                    if ht and cos[k] >= a.min_cos: fo.write(f'{cid}\t{fa}\t{fb}\t{uniq[g[k]]}\t{cos[k]:.4f}\t{mar[k]:.4f}\n'); n_acc += 1
            print(fp, 'accepted', n_acc, flush=True)
    for v in st.values(): v['hit_rate_true'] = v['hit_true'] / max(v['n'], 1); v['hit_rate_shuf'] = v['hit_shuf'] / max(v['n'], 1); v['lift'] = v['hit_rate_true'] / max(v['hit_rate_shuf'], 1e-9)
    json.dump({'n_exemplars': int(len(keep)), 'n_gloss_texts': len(uniq), 'n_segments_cos_ge_0.3': n_all, 'n_accepted': n_acc, 'thresholds': st}, open(a.stats, 'w'), indent=1)
    print(json.dumps(st, indent=1))


if __name__ == '__main__':
    main()
