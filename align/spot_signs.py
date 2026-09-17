"""Citation-NN sign spotting (BOBSL-style, with our SignBank anchors): for every bank segment, nearest SignBank citation
sign in the aligner's motion space; a spot is accepted when the gloss word appears in the clip's caption. Reports the
caption-hit rate for TRUE captions vs SHUFFLED captions at several cosine / margin thresholds (signal check), and writes the
accepted (clip, frame span, gloss word, cos, margin) pairs for phrase<->segment supervision (aligner round 2).
  python align/spot_signs.py --align_ckpt <anchor best> --bank_dir <seg_bank_v4_anchor> --out <tsv> --stats <json>
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np, torch, torch.nn.functional as F
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from align.align_dataset import INDEX
from align.train_align import AlignModel
from align.train_align_v4 import SignBank, DR

STOP = set('a an the to of in on at for with and or but is are was were be been am it its this that i you he she we they me him her us them my your his our their do does did have has had will would can could should not no yes so if then than there here what who how when where why which'.split())
norm = lambda s: re.sub(r"[^a-z0-9' ]+", ' ', s.lower()).split()


def caption_words(t):
    w = set(norm(t)); out = set(w)
    for x in w:  # crude lemmas so GLOSS 'walk' matches 'walked'/'walking'/'walks'
        for suf in ('s', 'es', 'ed', 'ing', 'd'):
            if x.endswith(suf) and len(x) - len(suf) >= 3: out.add(x[:-len(suf)])
        if x.endswith('ies'): out.add(x[:-3] + 'y')
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--align_ckpt', required=True); ap.add_argument('--bank_dir', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--stats', required=True)
    ap.add_argument('--sb_npz', default=f'{DR}/signbank_tokens.npz'); ap.add_argument('--sb_split', default=f'{DR}/signbank_split.json')
    ap.add_argument('--seed', type=int, default=0); a = ap.parse_args(); dev = 'cuda'
    ck = torch.load(a.align_ckpt, map_location='cpu'); c = ck['config']
    model = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']).to(dev); model.load_state_dict(ck['model']); model.eval()
    sb = SignBank(a.sb_npz, a.sb_split, model.mmm.pos.num_embeddings)
    # citation embeddings (ALL signs incl. held-out: spotting is a data step, not the probe), averaged over variants of a gloss text
    idx = list(range(len(sb.stems))); Z = []
    with torch.no_grad():
        for i in range(0, len(idx), 128):
            x, m, spans, _ = sb.batch(idx[i:i + 128])
            with torch.autocast('cuda', dtype=torch.bfloat16): Z.append(model.embed_segments(x.to(dev), m.to(dev), spans).float())
    Z = torch.cat(Z); texts = [sb.texts[i] for i in idx]; uniq = sorted(set(texts)); ti = {t: k for k, t in enumerate(uniq)}
    G = torch.zeros(len(uniq), Z.shape[1], device=dev); cnt = torch.zeros(len(uniq), device=dev)
    for z, t in zip(Z, texts): G[ti[t]] += z; cnt[ti[t]] += 1
    G = F.normalize(G / cnt[:, None], dim=-1); gloss_words = [[w for w in norm(t) if w not in STOP] for t in uniq]
    single = np.array([len(w) == 1 for w in gloss_words])  # multi-word glosses (e.g. 'next year') matched by all words present
    print(f'{len(uniq)} gloss texts ({single.sum()} single-word)', flush=True)
    rows = {}
    with open(INDEX) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE): rows[r['clip_id']] = r['text'].strip()
    rng = np.random.default_rng(a.seed); all_clips = list(rows); shuf = {cid: rows[all_clips[j]] for cid, j in zip(all_clips, rng.permutation(len(all_clips)))}
    ths = [(0.3, 0.0), (0.4, 0.0), (0.5, 0.0), (0.5, 0.05), (0.6, 0.05), (0.7, 0.05)]
    st = {f'{c}/{m}': {'n': 0, 'hit_true': 0, 'hit_shuf': 0} for c, m in ths}; n_all = 0; n_acc = 0
    with open(a.out, 'w') as fo:
        fo.write('clip_id\tfr_a\tfr_b\tgloss\tcos\tmargin\n')
        for fp in sorted(glob.glob(os.path.join(a.bank_dir, 'bank_*.npz'))):
            d = np.load(fp); E = torch.from_numpy(d['emb']).to(dev); meta = d['meta']; cids = [str(x) for x in d['clip_ids']]
            for i in range(0, len(E), 200000):
                e = F.normalize(E[i:i + 200000].float(), dim=-1); sim = e @ G.T; top2 = sim.topk(2, dim=1)
                cos = top2.values[:, 0].cpu().numpy(); mar = (top2.values[:, 0] - top2.values[:, 1]).cpu().numpy(); g = top2.indices[:, 0].cpu().numpy()
                for k in range(len(cos)):
                    ci, _, _, fa, fb = meta[i + k]; cid = cids[ci]; n_all += 1
                    if cid not in rows: continue
                    gw = gloss_words[g[k]]
                    if not gw: continue
                    wt = caption_words(rows[cid]); ws = caption_words(shuf[cid])
                    ht = all(w in wt for w in gw); hs = all(w in ws for w in gw)
                    for c_, m_ in ths:
                        if cos[k] >= c_ and mar[k] >= m_: s_ = st[f'{c_}/{m_}']; s_['n'] += 1; s_['hit_true'] += ht; s_['hit_shuf'] += hs
                    if ht and cos[k] >= 0.4: fo.write(f"{cid}\t{fa}\t{fb}\t{uniq[g[k]]}\t{cos[k]:.4f}\t{mar[k]:.4f}\n"); n_acc += 1
            print(fp, n_all, 'accepted', n_acc, flush=True)
    for k, v in st.items(): v['hit_rate_true'] = v['hit_true'] / max(v['n'], 1); v['hit_rate_shuf'] = v['hit_shuf'] / max(v['n'], 1); v['lift'] = v['hit_rate_true'] / max(v['hit_rate_shuf'], 1e-9)
    res = {'n_segments': n_all, 'n_accepted_cos0.4': n_acc, 'thresholds': st}; json.dump(res, open(a.stats, 'w'), indent=1); print(json.dumps(st, indent=1))


if __name__ == '__main__': main()
