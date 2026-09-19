"""Independent precision of MINED sign instances (2026-09-19, dictionary route).
Question: are the running-signing segments that the spotter labelled with gloss w really the sign w?
Judge = ASL3DWord (103 words, 1,539 clips, real signers, GT labels, never used by any aligner / spotter).
Each mined instance is classified by motion->motion similarity against the ASL3DWord clips, in the motion space of a
JUDGE model that has no lineage with the spotter (g1 = MIL-NCE on gloss units only, or the motion-only MMM encoder).
Mined embeddings are read from the judge's segment bank (same segmentation as the spotter's bank), nothing is re-embedded
except the ASL3DWord clips and the SignBank citation forms.
Reported per judge:
  ceiling_isolated : leave-one-out word accuracy inside ASL3DWord (isolated -> isolated)
  signbank_citation: SignBank citation form of the overlap words classified the same way (citation -> isolated gap)
  mined            : precision R1/R5 of spotted instances, by spotter-cos bin, single segments and merged adjacent spans
  dict_topN        : per word, the N highest-cos instances (what a dictionary would keep), mean precision over words
  control          : NON-spotted segments of the same clips, labelled with the clip's spotted word (removes caption-word leakage
                     in a text-trained judge: only the gap mined - control is evidence that the spotter found the sign)
  random           : random bank segments given the same labels (chance reference)
  python align/eval_mined_precision.py --align_ckpt <pt> --model_module align.train_align_v6 --bank_dir <dir> --out <json>
"""
import argparse, glob, importlib, json, os, pickle, sys
import numpy as np, torch, torch.nn.functional as F
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from pretrain.part_vqvae import PartVQVAE
from align.probe_asl3dword import feats_from_aa, D as A3D, PARTS
CUR = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation'
VQ = '/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/MotionVQ_Pooled/20260913_012619_job27263_vq_K512_w64/best_model.pt'


@torch.no_grad()
def embed_token_seqs(model, toks, dev):
    out = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(toks), 64):
            b = toks[i:i + 64]; L = max(len(t) for t in b); x = torch.zeros(len(b), L, 4, dtype=torch.long); m = torch.zeros(len(b), L, dtype=torch.bool); spans = []
            for j, t in enumerate(b): x[j, :len(t)] = t; m[j, :len(t)] = True; spans.append((j, 0, len(t)))
            out.append(model.embed_segments(x.to(dev), m.to(dev), spans).float())
    return F.normalize(torch.cat(out), dim=-1)


def word_scores(z, ref, ref_y, n_words, k=3, exclude_self=False):
    """(N, W) score = mean of the top-k cosine sims to the reference clips of each word."""
    sim = z @ ref.T
    if exclude_self: sim.fill_diagonal_(-2.0)
    S = torch.full((len(z), n_words), -2.0, device=z.device)
    for w in range(n_words):
        c = sim[:, ref_y == w]
        if c.shape[1]: S[:, w] = c.topk(min(k, c.shape[1]), dim=1).values.clamp_min(-1).mean(1)
    return S


def prec(S, y):
    rank = (S > S.gather(1, y[:, None])).sum(1)
    return {'n': int(len(y)), 'R1': float((rank == 0).float().mean()), 'R5': float((rank < 5).float().mean()), 'medR': int(rank.median()) + 1}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--align_ckpt', required=True); ap.add_argument('--model_module', default='align.train_align_v6')
    ap.add_argument('--bank_dir', required=True); ap.add_argument('--spots', default=f'{CUR}/data_records/alignment/spots_v4_anchor.tsv')
    ap.add_argument('--signbank', default=f'{CUR}/data_records/alignment/signbank_tokens.npz'); ap.add_argument('--vq_ckpt', default=VQ)
    ap.add_argument('--out', required=True); ap.add_argument('--n_random', type=int, default=20000); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); dev = 'cuda'; rng = np.random.default_rng(a.seed)
    AlignModel = importlib.import_module(a.model_module).AlignModel
    ck = torch.load(a.vq_ckpt, map_location='cpu'); cfg = ck['config']
    vq = PartVQVAE(ck['part_dims'], cfg['K'], cfg['width'], cfg['down']).to(dev); vq.load_state_dict(ck['model']); vq.eval()
    ack = torch.load(a.align_ckpt, map_location='cpu'); ac = ack['config']
    model = AlignModel(ac['mmm_ckpt'], ac['text_encoder'], ac['e']).to(dev); model.load_state_dict(ack['model']); model.eval(); max_len = model.mmm.pos.num_embeddings

    # ---- reference: ASL3DWord clips
    toks, labels = [], []
    for sp in ['train', 'test']:
        P = pickle.load(open(f'{A3D}/{sp}/samples_pose.pkl', 'rb')); L = pickle.load(open(f'{A3D}/{sp}/samples_label.pkl', 'rb'))
        for p, l in zip(P, L):
            if len(p) < 5: continue
            f = feats_from_aa(np.asarray(p))
            with torch.no_grad(): idx = vq.encode({k: torch.from_numpy(v)[None].to(dev) for k, v in f.items()})
            toks.append(torch.stack([idx[k][0] for k in PARTS], -1)[:max_len].cpu()); labels.append(str(l).strip().lower().replace('_', ' '))
    words = sorted(set(labels)); wi = {w: i for i, w in enumerate(words)}; nW = len(words)
    ref_y = torch.tensor([wi[l] for l in labels], device=dev); ref = embed_token_seqs(model, toks, dev)
    res = {'judge_ckpt': a.align_ckpt, 'bank_dir': a.bank_dir, 'spots': a.spots, 'n_ref_clips': len(labels), 'n_words': nW, 'chance_R1': 1 / nW}
    res['ceiling_isolated'] = prec(word_scores(ref, ref, ref_y, nW, exclude_self=True), ref_y)

    # ---- SignBank citation forms of the overlap words
    sb = np.load(a.signbank, allow_pickle=True); st, sy = [], []
    for i, t in enumerate(sb['texts']):
        t = str(t).strip().lower()
        if t in wi: st.append(torch.from_numpy(sb['tokens'][sb['offsets'][i]:sb['offsets'][i + 1]].astype(np.int64))[:max_len]); sy.append(wi[t])
    if st: res['signbank_citation'] = prec(word_scores(embed_token_seqs(model, st, dev), ref, ref_y, nW), torch.tensor(sy, device=dev))

    # ---- mined instances: join spots with the judge bank on (clip_id, fr_a, fr_b)
    spots = []
    for ln in open(a.spots).read().split('\n')[1:]:
        if not ln: continue
        c, fa, fb, g, cs, mg = ln.split('\t')
        if g in wi: spots.append((c, int(fa), int(fb), wi[g], float(cs)))
    need = {(c, fa, fb): i for i, (c, fa, fb, _, _) in enumerate(spots)}
    need_clips = {s[0] for s in spots}; clip_word = {}
    for s_ in spots: clip_word.setdefault(s_[0], s_[3])
    ctrl_E, ctrl_y = [], []  # control: NON-spotted segments of the same clips (caption has the word, spotter did not pick them)
    E = np.zeros((len(spots), 512), np.float32); found = np.zeros(len(spots), bool); rand = []
    for fp in sorted(glob.glob(os.path.join(a.bank_dir, 'bank_*.npz'))):
        b = np.load(fp, allow_pickle=True); cids = b['clip_ids']; meta = b['meta']; emb = b['emb']
        cmask = np.array([str(c) in need_clips for c in cids])
        for r in np.where(cmask[meta[:, 0]])[0]:
            i = need.get((str(cids[meta[r, 0]]), int(meta[r, 3]), int(meta[r, 4])))
            if i is not None: E[i] = emb[r]; found[i] = True
            elif rng.random() < 0.25: ctrl_E.append(emb[r].astype(np.float32)); ctrl_y.append(clip_word[str(cids[meta[r, 0]])])
        rand.append(emb[rng.choice(len(emb), a.n_random // 8 + 1, replace=False)].astype(np.float32))
    res['n_spots_overlap_words'] = len(spots); res['n_found_in_bank'] = int(found.sum()); res['n_overlap_words'] = len({s[3] for s in spots})
    keep = np.where(found)[0]; spots = [spots[i] for i in keep]; E = E[keep]
    y = torch.tensor([s[3] for s in spots], device=dev); cs = np.array([s[4] for s in spots]); Z = F.normalize(torch.from_numpy(E).to(dev), dim=-1)
    S = word_scores(Z, ref, ref_y, nW)
    res['mined_all'] = prec(S, y); res['mined_by_cos'] = {}
    for lo, hi in [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 9)]:
        m = torch.from_numpy((cs >= lo) & (cs < hi)).to(dev)
        if m.sum() > 0: res['mined_by_cos'][f'{lo}-{hi if hi < 9 else "up"}'] = prec(S[m], y[m])

    # merged spans: adjacent segments of the same clip with the same gloss -> one instance (mean embedding, max cos)
    order = sorted(range(len(spots)), key=lambda i: (spots[i][0], spots[i][1])); groups, cur = [], [order[0]]
    for i in order[1:]:
        p = spots[cur[-1]]; q = spots[i]
        if q[0] == p[0] and q[3] == p[3] and q[1] == p[2]: cur.append(i)
        else: groups.append(cur); cur = [i]
    groups.append(cur)
    Zm = F.normalize(torch.stack([Z[g].mean(0) for g in groups]), dim=-1); ym = torch.tensor([spots[g[0]][3] for g in groups], device=dev)
    csm = np.array([max(cs[i] for i in g) for g in groups]); nseg = np.array([len(g) for g in groups]); Sm = word_scores(Zm, ref, ref_y, nW)
    res['merged_all'] = prec(Sm, ym); res['merged_mean_segments'] = float(nseg.mean())
    mm = torch.from_numpy(nseg >= 2).to(dev)
    if mm.sum() > 0: res['merged_multi_segment_only'] = prec(Sm[mm], ym[mm])

    # dictionary view: per word keep the N highest-cos merged instances
    rankm = (Sm > Sm.gather(1, ym[:, None])).sum(1).cpu().numpy(); ymn = ym.cpu().numpy(); res['dict_topN'] = {}
    for N in [5, 20, 100]:
        p1, p5, nw = [], [], 0
        for w in sorted(set(ymn.tolist())):
            ix = np.where(ymn == w)[0]; ix = ix[np.argsort(-csm[ix])][:N]
            if len(ix) == 0: continue
            nw += 1; p1.append(float((rankm[ix] == 0).mean())); p5.append(float((rankm[ix] < 5).mean()))
        res['dict_topN'][str(N)] = {'n_words': nw, 'P1_mean_over_words': float(np.mean(p1)), 'P5_mean_over_words': float(np.mean(p5)),
                                    'words_with_P1_ge_0.5': int(np.sum(np.array(p1) >= 0.5))}
    # per-word table (top-20 view) for inspection
    per = {}
    for w in sorted(set(ymn.tolist())):
        ix = np.where(ymn == w)[0]; top = ix[np.argsort(-csm[ix])][:20]
        per[words[w]] = {'n_instances': int(len(ix)), 'P1_all': float((rankm[ix] == 0).mean()), 'P1_top20': float((rankm[top] == 0).mean())}
    res['per_word'] = per

    # chance reference: random bank segments with the mined label distribution
    R = F.normalize(torch.from_numpy(np.concatenate(rand)[:a.n_random]).to(dev), dim=-1)
    yr = y[torch.from_numpy(rng.integers(0, len(y), len(R))).to(dev)]
    res['random_segments'] = prec(word_scores(R, ref, ref_y, nW), yr)
    if ctrl_E:
        C = F.normalize(torch.from_numpy(np.stack(ctrl_E)).to(dev), dim=-1)
        res['control_same_clip_not_spotted'] = prec(word_scores(C, ref, ref_y, nW), torch.tensor(ctrl_y, device=dev))
    print(json.dumps({k: v for k, v in res.items() if k != 'per_word'}, indent=1))
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1); print('saved', a.out)


if __name__ == '__main__':
    main()
