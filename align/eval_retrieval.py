"""Roadmap 4.5 step 2 (= 5.2 evaluation): retrieval quality vs corpus scale.

For each held-out clip: chunk its sentence (chunk cache, whole-sentence fallback), embed chunks with
the 4.2 aligner, retrieve the top-1 segment per chunk from the bank (same-VIDEO segments excluded),
stitch retrieved segments in chunk order, FK both stitched and GT motion to root-zeroed upper-body
joints, and score DTW mean path distance. Baseline = random segments (same count) stitched the same
way. Curve = bank subsampled at --scales (coverage argument on the retrieval side).
ratio = dtw(retrieved, GT) / dtw(random, GT); < 1 means retrieval brings you closer to the true signing.

  python align/eval_retrieval.py --align_ckpt <pt> --bank_dir <dir> --val_list <txt> --out <json>
"""
import argparse, csv, glob, hashlib, json, os, random, sys
import numpy as np, torch, torch.nn.functional as F

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, 'infer_eval'))
from align.align_dataset import INDEX, load_chunks, merge_stopword_chunks
from align.train_align import AlignModel
from infer_eval.generate_smplx_param import load_smplx_model
from utils.motion_ae_fid import smplx_aa_to_upper3d
from config import How2Sign_SMPLX_Config

CHUNKS = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/alignment/chunks'


def dtw_mean(A, B, cap=150):
    """mean frame L2 along the DTW path between (Ta,D) and (Tb,D); both capped to <=cap frames."""
    if len(A) > cap: A = A[np.round(np.linspace(0, len(A) - 1, cap)).astype(int)]
    if len(B) > cap: B = B[np.round(np.linspace(0, len(B) - 1, cap)).astype(int)]
    d = np.linalg.norm(A[:, None] - B[None], axis=-1)  # (Ta, Tb)
    Ta, Tb = d.shape
    acc = np.full((Ta + 1, Tb + 1), np.inf); acc[0, 0] = 0.0
    for i in range(1, Ta + 1):
        row = acc[i]; prev = acc[i - 1]
        for j in range(1, Tb + 1):
            row[j] = d[i - 1, j - 1] + min(prev[j], prev[j - 1], row[j - 1])
    # path length lower-bounded by max(Ta,Tb); normalize by path cost per step via backtrack-free bound
    return acc[Ta, Tb] / max(Ta, Tb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--align_ckpt', required=True); ap.add_argument('--bank_dir', required=True)
    ap.add_argument('--val_list', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--index', default=INDEX); ap.add_argument('--n_clips', type=int, default=400)
    ap.add_argument('--scales', type=float, nargs='+', default=[0.01, 0.1, 1.0])
    ap.add_argument('--seq_len', type=int, default=100); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    rows = {}
    with open(a.index) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            rows[r['clip_id']] = (r['video'], r['npz'], r['text'].strip())
    wanted = [l.strip() for l in open(a.val_list) if l.strip()]
    wanted = [c for c in wanted if c in rows and rows[c][2]]
    random.shuffle(wanted); wanted = wanted[:a.n_clips]

    # bank
    emb, meta, cids = [], [], []
    off = 0
    for fp in sorted(glob.glob(os.path.join(a.bank_dir, 'bank_*.npz'))):
        d = np.load(fp, allow_pickle=False)
        emb.append(d['emb']); m = d['meta'].copy(); m[:, 0] += off
        meta.append(m); cids.extend([str(x) for x in d['clip_ids']]); off = len(cids)
    emb = np.concatenate(emb); meta = np.concatenate(meta)
    seg_video = np.array([rows[cids[i]][0] if cids[i] in rows else '?' for i in meta[:, 0]])
    print(f'[bank] {len(emb)} segments, {len(cids)} clips', flush=True)
    E = torch.from_numpy(emb.astype(np.float16)).to(dev)
    for i in range(0, len(E), 1_000_000):  # chunked in-place normalize, stays half
        E[i:i + 1_000_000] = F.normalize(E[i:i + 1_000_000].float(), dim=-1).half()

    ck = torch.load(a.align_ckpt, map_location='cpu'); c = ck['config']
    model = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']).to(dev)
    model.load_state_dict(ck['model']); model.eval()
    smpl_x = load_smplx_model(How2Sign_SMPLX_Config().HUMAN_MODELS_PATH)
    h2c = load_chunks(CHUNKS)

    def fk(aa159):  # (T,159) root-zeroed -> (T,132) np
        j = smplx_aa_to_upper3d(aa159, smpl_x, device=dev)
        return (j.detach().cpu().numpy() if hasattr(j, 'detach') else np.asarray(j)).astype(np.float32)

    def seg_motion(si):
        ci, _, _, fa, fb = meta[si]
        aa = np.load(rows[cids[ci]][1])['axis_angle'][fa:fb].copy()
        aa[:, 0, :] = 0.0
        return aa.reshape(len(aa), -1).astype(np.float32)

    rng = np.random.default_rng(a.seed)
    res = {s: [] for s in a.scales}; sims = {s: [] for s in a.scales}
    keep_masks = {}
    for s in a.scales:
        keep_masks[s] = np.ones(len(emb), bool) if s >= 1.0 else (rng.random(len(emb)) < s)

    for n, cid in enumerate(wanted):
        video, npz, text = rows[cid]
        ch = h2c.get(hashlib.md5(text.encode()).hexdigest())
        ch = merge_stopword_chunks(ch) if ch else [text]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            zt = model.embed_chunks(ch, dev).half()  # (C, e), normalized
        g = np.load(npz)['axis_angle'].copy(); g[:, 0, :] = 0.0
        G = fk(g.reshape(len(g), -1).astype(np.float32))
        not_same_video = torch.from_numpy(seg_video != video).to(dev)
        for s in a.scales:
            ok = torch.from_numpy(keep_masks[s]).to(dev) & not_same_video
            sim = (zt @ E.T).float(); sim[:, ~ok] = -1e4
            top = sim.argmax(-1).cpu().numpy(); sims[s].append(float(sim.max(-1).values.mean()))
            parts = [seg_motion(si) for si in top]
            R = fk(np.concatenate(parts, 0))
            rnd_pool = np.where(keep_masks[s])[0]
            rparts = [seg_motion(si) for si in rng.choice(rnd_pool, size=len(top))]
            Rr = fk(np.concatenate(rparts, 0))
            res[s].append(dtw_mean(R, G) / max(dtw_mean(Rr, G), 1e-9))
        if (n + 1) % 25 == 0: print(f'  {n+1}/{len(wanted)}', flush=True)

    out = {'bank_dir': a.bank_dir, 'n_clips': len(wanted), 'n_segments_bank': int(len(emb)), 'scales': {}}
    for s in a.scales:
        r = np.array(res[s])
        out['scales'][str(s)] = {'dtw_ratio_mean': float(r.mean()), 'median': float(np.median(r)),
                                 'p25': float(np.percentile(r, 25)), 'p75': float(np.percentile(r, 75)),
                                 'frac_below_1': float((r < 1).mean()), 'mean_top1_cos': float(np.mean(sims[s]))}
        print(s, json.dumps(out['scales'][str(s)]), flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out, flush=True)


if __name__ == '__main__': main()
