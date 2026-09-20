"""v2 (2026-09-20; v1 = refine_open_vocab.py untouched). Finding behind v2: the highest prototype scores of v1 came from ONE video per word
(STEM Wiki: a whole article about "computer" / "fish" signed by one interpreter -> hundreds of same-signer candidates form a tight cluster, and the
independent judge rejects it: fish 0.00, city 0.00, computer 0.11). "Recurs across clips" must mean "recurs across VIDEOS / signers".
Changes: (1) the density ignores candidates of the same VIDEO, not only the same clip; (2) the prototype is built from the top candidates with at
most --per_video of them per video; (3) new columns n_vid (distinct videos among the word's candidates) and proto_vid (distinct videos in the
prototype set) so that words carried by a single video can be filtered.
--- v1 docstring ---
Second stage of open-vocabulary sign mining (2026-09-19): re-rank the text-driven candidates of mine_open_vocab.py INSIDE each word
with motion-only evidence. Finding that motivates it: the text score, the null tail probability and the g1-space purity rank instances
well globally but not within a word (dictionary top-20 precision 0.21 vs 0.34 for citation-NN spotting), and purity saturates near 1.
Idea: the true sign of word w is the motion pattern that RECURS across the candidate segments of many different clips, while false
candidates are scattered. In a motion space that never saw text (MMM encoder bank by default):
  dens     = mean cosine to the k nearest candidates of the same word from OTHER clips
  bg       = mean cosine to the k nearest segments of a random background sample of the bank (how common this motion is anyway)
  contrast = dens - bg
  proto    = cosine to the word prototype = normalised mean of the top --proto_share candidates by contrast (one refinement round:
             the prototype is rebuilt from the top candidates by proto and the score recomputed). The prototype plays the role of a
             citation form, but it is built from running signing and needs no labelled video.
Output: <out> = input columns + dens bg contrast proto (tab separated, same header convention, readable by eval_mined_precision_v2.py).
  python align/refine_open_vocab.py --cands <..._detail.tsv> --bank_dir <seg_bank_judge_mmm> --out <tsv>
"""
import argparse, csv, glob, os, sys
import numpy as np, torch, torch.nn.functional as F


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cands', required=True); ap.add_argument('--bank_dir', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--k', type=int, default=10); ap.add_argument('--n_bg', type=int, default=200000); ap.add_argument('--proto_share', type=float, default=0.2)
    ap.add_argument('--per_video', type=int, default=2); ap.add_argument('--index', default='/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/curation/clip_index.tsv')
    ap.add_argument('--min_proto', type=int, default=5); ap.add_argument('--rounds', type=int, default=1); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); dev = 'cuda'; rng = np.random.default_rng(a.seed)
    lines = open(a.cands).read().split('\n'); hdr = lines[0].split('\t'); col = {h: i for i, h in enumerate(hdr)}; rows = [l.split('\t') for l in lines[1:] if l]
    need = {}
    for i, p in enumerate(rows): need.setdefault((p[col['clip_id']], int(p[col['fr_a']]), int(p[col['fr_b']])), []).append(i)
    need_clips = {p[col['clip_id']] for p in rows}; E = np.zeros((len(rows), 512), np.float32); found = np.zeros(len(rows), bool); bg = []
    for fp in sorted(glob.glob(os.path.join(a.bank_dir, 'bank_*.npz'))):
        b = np.load(fp, allow_pickle=True); cids = b['clip_ids']; meta = b['meta']; emb = b['emb']
        cmask = np.array([str(c) in need_clips for c in cids])
        for r in np.where(cmask[meta[:, 0]])[0]:
            for i in need.get((str(cids[meta[r, 0]]), int(meta[r, 3]), int(meta[r, 4])), ()): E[i] = emb[r]; found[i] = True
        bg.append(emb[rng.choice(len(emb), a.n_bg // 8 + 1, replace=False)].astype(np.float32))
    print(f'candidates {len(rows)} found in bank {int(found.sum())}', flush=True)
    Z = F.normalize(torch.from_numpy(E).to(dev), dim=-1); BG = F.normalize(torch.from_numpy(np.concatenate(bg)[:a.n_bg]).to(dev), dim=-1)
    vid_of = {}
    with open(a.index) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            if r['clip_id'] in need_clips: vid_of[r['clip_id']] = r['corpus'] + ':' + r['video']
    word = np.array([p[col['gloss']] for p in rows]); clip = np.array([vid_of.get(p[col['clip_id']], p[col['clip_id']]) for p in rows])  # grouping key = VIDEO
    nvid = np.zeros(len(rows), np.int32); pvid = np.zeros(len(rows), np.int32)
    dens = np.zeros(len(rows), np.float32); bgs = np.zeros(len(rows), np.float32); proto = np.zeros(len(rows), np.float32)
    for w in sorted(set(word.tolist())):
        ix = np.where((word == w) & found)[0]
        if len(ix) < 3: continue
        X = Z[torch.from_numpy(ix).to(dev)]; cl = clip[ix]; _, cinv = np.unique(cl, return_inverse=True); cinv = torch.from_numpy(cinv).to(dev)
        k = min(a.k, max(1, len(ix) - 1)); d_ = torch.zeros(len(ix), device=dev); b_ = torch.zeros(len(ix), device=dev)
        for q0 in range(0, len(ix), 4096):
            sim = X[q0:q0 + 4096] @ X.T; sim[cinv[q0:q0 + 4096, None] == cinv[None]] = -2.0
            tk = sim.topk(k, dim=1).values; d_[q0:q0 + 4096] = torch.where(tk > -1.5, tk, torch.zeros_like(tk)).sum(1) / (tk > -1.5).sum(1).clamp_min(1)
            b_[q0:q0 + 4096] = (X[q0:q0 + 4096] @ BG.T).topk(a.k, dim=1).values.mean(1)
        score = d_ - b_; pr = score
        for _ in range(a.rounds + 1):
            n_top = max(a.min_proto, int(len(ix) * a.proto_share)); order = torch.argsort(-pr).cpu().numpy(); cnt = {}; top = []
            for t in order:  # best candidates first, at most --per_video per video
                v = cl[t]
                if cnt.get(v, 0) < a.per_video: cnt[v] = cnt.get(v, 0) + 1; top.append(t)
                if len(top) >= n_top: break
            pv = len(cnt); top = torch.tensor(top, device=dev)
            c = F.normalize(X[top].mean(0), dim=-1); pr = X @ c
        nvid[ix] = len(set(cl.tolist())); pvid[ix] = pv
        dens[ix] = d_.cpu().numpy(); bgs[ix] = b_.cpu().numpy(); proto[ix] = pr.cpu().numpy()
    with open(a.out, 'w') as f:
        f.write('\t'.join(hdr + ['dens', 'bg', 'contrast', 'proto', 'n_vid', 'proto_vid']) + '\n')
        for i, p in enumerate(rows):
            if found[i]: f.write('\t'.join(p + [f'{dens[i]:.4f}', f'{bgs[i]:.4f}', f'{dens[i] - bgs[i]:.4f}', f'{proto[i]:.4f}', str(nvid[i]), str(pvid[i])]) + '\n')
    print('saved', a.out, flush=True)


if __name__ == '__main__':
    main()
