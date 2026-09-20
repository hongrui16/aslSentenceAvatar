"""Independent precision of mined sign instances, v2 (2026-09-19; v1 = eval_mined_precision.py, untouched).
Same judge protocol as v1 (each mined segment is classified among the 103 ASL3DWord words by motion->motion similarity to the 1,539
ground-truth ASL3DWord clips, in the motion space of a judge model). v2 makes miners with DIFFERENT score scales comparable:
  - any spots tsv with a header (clip_id fr_a fr_b gloss + confidence columns) is accepted;
  - for every confidence column (prefix '-' = lower is better, e.g. -null_q) precision is reported on the top 10 / 25 / 50 / 100 % of
    the instances and in the dictionary view (per word, the N most confident instances);
  - --words_file restricts the evaluation to a fixed word set, so two miners are compared on exactly the same words.
  python align/eval_mined_precision_v2.py --align_ckpt <judge pt> --model_module align.train_align --bank_dir <judge bank> \
         --spots <tsv> --conf_cols cos,margin,-null_q --out <json> [--words_file <txt>]
"""
import argparse, glob, importlib, json, os, pickle, sys
import numpy as np, torch, torch.nn.functional as F
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from pretrain.part_vqvae import PartVQVAE
from align.probe_asl3dword import feats_from_aa, D as A3D, PARTS
from align.eval_mined_precision import embed_token_seqs, word_scores, prec, VQ


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--align_ckpt', required=True); ap.add_argument('--model_module', default='align.train_align')
    ap.add_argument('--bank_dir', required=True); ap.add_argument('--spots', required=True); ap.add_argument('--conf_cols', default='cos')
    ap.add_argument('--words_file', default=''); ap.add_argument('--vq_ckpt', default=VQ); ap.add_argument('--out', required=True)
    ap.add_argument('--n_random', type=int, default=20000); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); dev = 'cuda'; rng = np.random.default_rng(a.seed)
    AlignModel = importlib.import_module(a.model_module).AlignModel
    ck = torch.load(a.vq_ckpt, map_location='cpu'); cfg = ck['config']
    vq = PartVQVAE(ck['part_dims'], cfg['K'], cfg['width'], cfg['down']).to(dev); vq.load_state_dict(ck['model']); vq.eval()
    ack = torch.load(a.align_ckpt, map_location='cpu'); ac = ack['config']
    model = AlignModel(ac['mmm_ckpt'], ac['text_encoder'], ac['e']).to(dev); model.load_state_dict(ack['model']); model.eval(); max_len = model.mmm.pos.num_embeddings

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
    allow = {l.strip().lower() for l in open(a.words_file) if l.strip()} if a.words_file else None

    lines = open(a.spots).read().split('\n'); hdr = lines[0].split('\t'); col = {h: i for i, h in enumerate(hdr)}
    cols = [c.strip() for c in a.conf_cols.split(',') if c.strip()]
    spots, conf = [], []
    for ln in lines[1:]:
        if not ln: continue
        p = ln.split('\t'); g = p[col['gloss']]
        if g in wi and (allow is None or g in allow):
            spots.append((p[col['clip_id']], int(p[col['fr_a']]), int(p[col['fr_b']]), wi[g]))
            conf.append([(-1.0 if c.startswith('-') else 1.0) * float(p[col[c.lstrip('-')]]) for c in cols])
    need = {}
    for i, (c, fa, fb, _) in enumerate(spots): need.setdefault((c, fa, fb), []).append(i)
    need_clips = {s[0] for s in spots}; E = np.zeros((len(spots), 512), np.float32); found = np.zeros(len(spots), bool); rand = []
    for fp in sorted(glob.glob(os.path.join(a.bank_dir, 'bank_*.npz'))):
        b = np.load(fp, allow_pickle=True); cids = b['clip_ids']; meta = b['meta']; emb = b['emb']
        cmask = np.array([str(c) in need_clips for c in cids])
        for r in np.where(cmask[meta[:, 0]])[0]:
            for i in need.get((str(cids[meta[r, 0]]), int(meta[r, 3]), int(meta[r, 4])), ()): E[i] = emb[r]; found[i] = True
        rand.append(emb[rng.choice(len(emb), a.n_random // 8 + 1, replace=False)].astype(np.float32))
    keep = np.where(found)[0]; y = torch.tensor([spots[i][3] for i in keep], device=dev); conf = np.array(conf, np.float64)[keep]
    Z = F.normalize(torch.from_numpy(E[keep]).to(dev), dim=-1); S = word_scores(Z, ref, ref_y, nW)
    rank = (S > S.gather(1, y[:, None])).sum(1).cpu().numpy(); yn = y.cpu().numpy()
    res = {'judge_ckpt': a.align_ckpt, 'spots': a.spots, 'words_file': a.words_file, 'n_words_eval': int(len(set(yn.tolist()))), 'n_instances': int(len(yn)),
           'n_not_in_bank': int((~found).sum()), 'chance_R1': 1 / nW, 'ceiling_isolated': prec(word_scores(ref, ref, ref_y, nW, exclude_self=True), ref_y),
           'all': {'n': int(len(yn)), 'R1': float((rank == 0).mean()), 'R5': float((rank < 5).mean())}, 'by_conf': {}}
    for k, c in enumerate(cols):
        o = np.argsort(-conf[:, k]); out = {'top_share': {}, 'dict_topN': {}}
        for share in [0.1, 0.25, 0.5, 1.0]:
            ix = o[:max(1, int(len(o) * share))]; out['top_share'][str(share)] = {'n': int(len(ix)), 'R1': float((rank[ix] == 0).mean()), 'R5': float((rank[ix] < 5).mean())}
        for Ntop in [5, 20, 100]:
            p1 = []
            for w in sorted(set(yn.tolist())):
                ix = np.where(yn == w)[0]; ix = ix[np.argsort(-conf[ix, k])][:Ntop]; p1.append(float((rank[ix] == 0).mean()))
            out['dict_topN'][str(Ntop)] = {'P1_mean_over_words': float(np.mean(p1)), 'words_with_P1_ge_0.5': int(np.sum(np.array(p1) >= 0.5)), 'n_words': len(p1)}
        res['by_conf'][c] = out
    per = {}
    for w in sorted(set(yn.tolist())):
        ix = np.where(yn == w)[0]; per[words[w]] = {'n': int(len(ix)), 'P1_all': float((rank[ix] == 0).mean())}
    res['per_word'] = per
    R = F.normalize(torch.from_numpy(np.concatenate(rand)[:a.n_random]).to(dev), dim=-1); yr = y[torch.from_numpy(rng.integers(0, len(y), len(R))).to(dev)]
    res['random_segments'] = prec(word_scores(R, ref, ref_y, nW), yr)
    print(json.dumps({k: v for k, v in res.items() if k != 'per_word'}, indent=1)); os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1); print('saved', a.out)
    # per-instance dump (judge rank of the true word, confidences, word id) for offline analysis of confidence combinations
    np.savez(a.out.replace('.json', '_inst.npz'), rank=rank, word=yn, conf=conf, conf_cols=np.array(cols), words=np.array(words),
             clip=np.array([spots[i][0] for i in keep]), fr=np.array([[spots[i][1], spots[i][2]] for i in keep]))


if __name__ == '__main__':
    main()
