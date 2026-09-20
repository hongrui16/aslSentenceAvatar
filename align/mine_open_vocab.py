"""Open-vocabulary sign mining (2026-09-19, dictionary route, option A). No citation video is needed for a word.
Idea: a gloss unit w occurs in the captions of thousands of clips. In a clip whose caption has w, the segment that best matches the
TEXT embedding of w should score higher than the best segment of clips WITHOUT w. The clips without w give a per-word null distribution,
so every candidate gets a calibrated tail probability instead of a global cosine threshold.
Per word w (all gloss-unit types with clip frequency >= --min_df):
  1. s(seg) = cos(text(w), seg) for every bank segment; clip score = max over the clip's segments.
  2. null = clip scores of clips whose gloss list does NOT contain w, taken within the same clip-length bin (a longer clip has more
     segments and a higher max by chance). null_q = share of null clips scoring at least as high. A positive clip is accepted when
     null_q <= --alpha.
  3. emitted instances = segments of an accepted clip within --delta of the clip max (adjacent ones are merged later by
     build_sign_dictionary.py).
  4. purity = among the k nearest motion neighbours (other clips only) of an emitted segment, the share that are emitted segments of
     the same word rather than best-matching segments of null clips (0.5 = no evidence, 1.0 = a recurring motion pattern specific to w).
Outputs: <out>.tsv  clip_id fr_a fr_b gloss cos margin(=purity)   (same 6 columns as spot_signs.py, readable by the dictionary builder)
         <out>_detail.tsv  adds null_q and n_seg;  <stats>.json per-word df / accepted share / mean null_q / mean purity.
  python align/mine_open_vocab.py --align_ckpt <g1 pt> --model_module align.train_align_v6 --bank_dir <seg_bank_g1> --out <tsv> --stats <json>
"""
import argparse, collections, csv, glob, hashlib, importlib, json, os, sys
import numpy as np, torch, torch.nn.functional as F
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
CUR = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation'


def load_bank(bank_dir, dev):
    E, clip_of, fa, fb, clip_ids, gid = [], [], [], [], [], {}
    for fp in sorted(glob.glob(os.path.join(bank_dir, 'bank_*.npz'))):
        d = np.load(fp, allow_pickle=True); cids = [str(c) for c in d['clip_ids']]; meta = d['meta']
        loc = np.array([gid.setdefault(c, len(gid)) for c in cids]); clip_ids = list(gid)
        E.append(d['emb']); clip_of.append(loc[meta[:, 0]]); fa.append(meta[:, 3]); fb.append(meta[:, 4])
    E = np.concatenate(E); clip_of = np.concatenate(clip_of); fa = np.concatenate(fa); fb = np.concatenate(fb)
    order = np.lexsort((fa, clip_of)); E, clip_of, fa, fb = E[order], clip_of[order], fa[order], fb[order]
    Eg = torch.empty(E.shape, dtype=torch.float16, device=dev)  # normalised in chunks: peak memory stays near 2 GB, so a 1g.10gb slice is enough
    for i in range(0, len(E), 500000): Eg[i:i + 500000] = F.normalize(torch.from_numpy(E[i:i + 500000]).to(dev).float(), dim=-1).half()
    return Eg, torch.from_numpy(clip_of).to(dev), fa, fb, list(gid)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--align_ckpt', required=True); ap.add_argument('--model_module', default='align.train_align_v6')
    ap.add_argument('--bank_dir', required=True); ap.add_argument('--out', required=True); ap.add_argument('--stats', required=True)
    ap.add_argument('--gloss_dir', default=f'{CUR}/data_records/alignment/gloss'); ap.add_argument('--index', default=f'{CUR}/data_records/curation/clip_index.tsv')
    ap.add_argument('--min_df', type=int, default=20); ap.add_argument('--alpha', type=float, default=0.05); ap.add_argument('--delta', type=float, default=0.03)
    ap.add_argument('--max_per_clip', type=int, default=3); ap.add_argument('--k', type=int, default=10); ap.add_argument('--max_purity_pos', type=int, default=4000)
    ap.add_argument('--word_batch', type=int, default=96); ap.add_argument('--n_len_bins', type=int, default=8); ap.add_argument('--max_words', type=int, default=0)
    ap.add_argument('--only_words', default='', help='optional file with one word per line (smoke / targeted runs)'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); dev = 'cuda'; rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    AlignModel = importlib.import_module(a.model_module).AlignModel
    ck = torch.load(a.align_ckpt, map_location='cpu'); c = ck['config']
    model = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']).to(dev); model.load_state_dict(ck['model']); model.eval()

    E, clip_of, fa, fb, clip_ids = load_bank(a.bank_dir, dev); N, C = len(E), len(clip_ids); cid_idx = {cid: i for i, cid in enumerate(clip_ids)}
    print(f'bank: {N} segments / {C} clips', flush=True)
    n_seg = torch.bincount(clip_of, minlength=C); edges = torch.quantile(n_seg.float(), torch.linspace(0, 1, a.n_len_bins + 1, device=dev)[1:-1])
    len_bin = torch.bucketize(n_seg.float(), edges)  # (C,) 0..n_len_bins-1

    # gloss units per bank clip
    g_of = {}
    for fp in glob.glob(os.path.join(a.gloss_dir, 'gloss_shard*.jsonl')):
        for ln in open(fp):
            d = json.loads(ln)
            if d.get('ok'): g_of[d['h']] = sorted({u.lower().strip() for u in d['gloss'] if u.strip()})
    clips_of_word = collections.defaultdict(list)
    with open(a.index) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            ci = cid_idx.get(r['clip_id'])
            if ci is None: continue
            for u in g_of.get(hashlib.md5(r['text'].strip().encode()).hexdigest(), ()): clips_of_word[u].append(ci)
    words = sorted((w for w, v in clips_of_word.items() if len(v) >= a.min_df), key=lambda w: -len(clips_of_word[w]))
    if a.only_words: keep = {l.strip().lower() for l in open(a.only_words) if l.strip()}; words = [w for w in words if w in keep]
    if a.max_words: words = words[:a.max_words]
    print(f'{len(words)} words with df >= {a.min_df}', flush=True)

    stats = {}; n_emit = 0; CH = 400000
    fo = open(a.out, 'w'); fo.write('clip_id\tfr_a\tfr_b\tgloss\tcos\tmargin\n')
    fd = open(a.out.replace('.tsv', '_detail.tsv'), 'w'); fd.write('clip_id\tfr_a\tfr_b\tgloss\tcos\tmargin\tnull_q\tn_seg\n')
    for b0 in range(0, len(words), a.word_batch):
        wb = words[b0:b0 + a.word_batch]; B = len(wb)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16): T = F.normalize(model.embed_chunks(wb, dev).float(), dim=-1)
        P = torch.zeros(C, B, dtype=torch.bool, device=dev)
        for j, w in enumerate(wb): P[torch.tensor(clips_of_word[w], device=dev), j] = True
        M = torch.full((C, B), -2.0, device=dev)
        for i in range(0, N, CH):  # pass 1: clip max
            s = E[i:i + CH].float() @ T.T; M.scatter_reduce_(0, clip_of[i:i + CH, None].expand(-1, B), s, 'amax')
        # calibrated tail probability of every positive clip, within its length bin
        Q = torch.ones(C, B, device=dev)
        for j in range(B):
            for lb in range(a.n_len_bins):
                inb = len_bin == lb; neg = M[inb & ~P[:, j], j]; pos_i = torch.where(inb & P[:, j])[0]
                if len(pos_i) == 0 or len(neg) < 50: continue
                ns = neg.sort().values; Q[pos_i, j] = 1.0 - torch.searchsorted(ns, M[pos_i, j].contiguous(), right=False).float() / len(ns)
        ACC = P & (Q <= a.alpha)
        NEG = torch.zeros_like(ACC)  # matched null sample for the purity test: as many null clips as accepted clips (capped)
        for j in range(B):
            na = int(ACC[:, j].sum()); ni = torch.where(~P[:, j] & (M[:, j] > -1.5))[0]
            if na and len(ni): NEG[ni[torch.randperm(len(ni), device=dev)[:min(na, a.max_purity_pos)]], j] = True
        rows_p, rows_n = [[] for _ in range(B)], [[] for _ in range(B)]
        for i in range(0, N, CH):  # pass 2: collect the segments
            s = E[i:i + CH].float() @ T.T; co = clip_of[i:i + CH]; mx = M[co]
            hit = (s >= mx - a.delta) & ACC[co]; r, j = torch.where(hit)
            for jj in range(B):
                m = j == jj
                if m.any(): rows_p[jj].append(torch.stack([r[m] + i, (s[r[m], jj] * 1e4).long()], 1))
            r, j = torch.where((s >= mx - 1e-6) & NEG[co])
            for jj in range(B):
                m = j == jj
                if m.any(): rows_n[jj].append(r[m] + i)
        for j, w in enumerate(wb):
            npos = int(P[:, j].sum()); qs = Q[P[:, j], j]
            st = {'df': npos, 'accepted_share': float(ACC[:, j].float().sum() / max(npos, 1)), 'mean_null_q': float(qs.mean()) if npos else None, 'n_instances': 0, 'mean_purity': None}
            if rows_p[j]:
                rp = torch.cat(rows_p[j]); ridx, sc = rp[:, 0], rp[:, 1].float() / 1e4
                # at most --max_per_clip segments per clip, highest score first
                o = torch.argsort(-sc); ridx, sc = ridx[o], sc[o]; cl = clip_of[ridx].cpu().numpy(); seen = collections.Counter(); keep, first = [], []
                for t, cc in enumerate(cl):
                    if seen[cc] < a.max_per_clip: first.append(seen[cc] == 0); seen[cc] += 1; keep.append(t)
                keep = torch.tensor(keep, device=dev); ridx, sc = ridx[keep], sc[keep]; first = torch.tensor(first, device=dev)
                pur = torch.full((len(ridx),), 0.5, device=dev)
                if rows_n[j]:
                    rn = torch.cat(rows_n[j]); sub = ridx[first][:a.max_purity_pos]  # one segment per accepted clip, so positives and nulls are balanced (baseline 0.5)
                    X = torch.cat([E[sub], E[rn]]).float(); lab = torch.cat([torch.ones(len(sub)), torch.zeros(len(rn))]).to(dev)
                    cx = torch.cat([clip_of[sub], clip_of[rn]]); k = min(a.k, len(X) - 1)
                    for q0 in range(0, len(ridx), 2048):
                        sim = E[ridx[q0:q0 + 2048]].float() @ X.T; sim[clip_of[ridx[q0:q0 + 2048], None] == cx[None]] = -2.0
                        pur[q0:q0 + 2048] = lab[sim.topk(k, dim=1).indices].mean(1)
                ri = ridx.cpu().numpy(); scn = sc.cpu().numpy(); pn = pur.cpu().numpy(); qn = Q[clip_of[ridx], j].cpu().numpy(); nsn = n_seg[clip_of[ridx]].cpu().numpy()
                for t in range(len(ri)):
                    cid = clip_ids[int(clip_of[ri[t]])]; line = f'{cid}\t{fa[ri[t]]}\t{fb[ri[t]]}\t{w}\t{scn[t]:.4f}\t{pn[t]:.4f}'
                    fo.write(line + '\n'); fd.write(f'{line}\t{qn[t]:.5f}\t{nsn[t]}\n')
                st['n_instances'] = int(len(ri)); st['mean_purity'] = float(pn.mean()); n_emit += len(ri)
            stats[w] = st
        print(f'words {b0 + B}/{len(words)} emitted {n_emit}', flush=True)
    fo.close(); fd.close()
    acc = np.array([v['accepted_share'] for v in stats.values()]); pur = np.array([v['mean_purity'] for v in stats.values() if v['mean_purity'] is not None])
    summ = {'n_words': len(stats), 'n_instances': n_emit, 'alpha': a.alpha, 'accepted_share_median': float(np.median(acc)), 'accepted_share_p25_p75': [float(np.percentile(acc, 25)), float(np.percentile(acc, 75))],
            'words_with_accepted_share_ge_2alpha': int((acc >= 2 * a.alpha).sum()), 'mean_purity_median': float(np.median(pur)) if len(pur) else None}
    json.dump({'summary': summ, 'args': vars(a), 'per_word': stats}, open(a.stats, 'w'), indent=1); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
