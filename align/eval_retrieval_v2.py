"""Retrieval eval v2: is the DTW-ratio protocol able to see aligner quality at all?

Adds to eval_retrieval.py (same 400 pool_val clips, same bank, same stitching):
  conditions: text  = chunk -> top-1 bank segment (the real system)
              oracle = GT clip's own segment embeddings (motion side of the aligner) -> top-1 bank segment
                       (same-video excluded) = ceiling of motion-similarity retrieval + stitching
              random = random bank segments (denominator)
  spaces:     raw   = root-zeroed upper-body joints (as v1)
              vel   = frame differences of raw (removes static pose / proportion offsets)
              style = per-sequence mean pose removed + shoulder-width scaled
              ae    = pooled MoAE bottleneck L2 (sequence resampled to 100 frames)
ratio = d(cond, GT) / d(random, GT) per clip; reported per (condition, space, scale).

  python align/eval_retrieval_v2.py --align_ckpt <pt> --bank_dir <dir> --val_list <txt> --out <json>
"""
import argparse, csv, glob, hashlib, json, os, random, sys
import numpy as np, torch, torch.nn.functional as F

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, 'infer_eval'))
from align.align_dataset import INDEX, SEGMENTS, PARTS, load_chunks, merge_stopword_chunks
from align.train_align import AlignModel
from align.eval_retrieval import dtw_mean
from infer_eval.generate_smplx_param import load_smplx_model
from utils.motion_ae_fid import smplx_aa_to_upper3d, load_motion_ae, encode_motion
from config import How2Sign_SMPLX_Config

CHUNKS = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/alignment/chunks'
TOK = '/projects/kosecka/hongrui/dataset/smplx_fits/pooled_tokens/vq_K512_w64'
AE = '/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/MotionAE_PooledSMPLX/20260911_130058_job9884582/best_model.pt'
SH_L, SH_R = 8, 9  # shoulders in the 44-joint upper-body subset


def resample(X, n):
    return X[np.round(np.linspace(0, len(X) - 1, n)).astype(int)]


def to_space(J, space):
    """J: (T,132) raw joints."""
    if space == 'raw': return J
    if space == 'vel': return np.diff(J, axis=0) if len(J) > 1 else np.zeros_like(J)
    if space == 'style':
        P = J.reshape(len(J), 44, 3); sw = np.linalg.norm(P[:, SH_L] - P[:, SH_R], axis=-1).mean() + 1e-6
        return ((P - P.mean(0, keepdims=True)) / sw).reshape(len(J), -1)
    raise ValueError(space)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--align_ckpt', required=True); ap.add_argument('--bank_dir', required=True)
    ap.add_argument('--val_list', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--index', default=INDEX); ap.add_argument('--n_clips', type=int, default=400)
    ap.add_argument('--scales', type=float, nargs='+', default=[0.1, 1.0]); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--ae', default=AE); ap.add_argument('--max_tok', type=int, default=256)
    a = ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows = {}
    with open(a.index) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE): rows[r['clip_id']] = (r['video'], r['npz'], r['text'].strip(), r['corpus'])
    wanted = [l.strip() for l in open(a.val_list) if l.strip()]
    wanted = [c for c in wanted if c in rows and rows[c][2]]
    random.shuffle(wanted); wanted = wanted[:a.n_clips]  # same seed/order as v1 eval
    wset = set(wanted); segs = {}
    with open(SEGMENTS) as f:
        next(f)
        for l in f:
            cid, T, fps, bounds, rest = l.rstrip('\n').split('\t')
            if cid in wset:
                b = list(map(int, bounds.split(','))); rf = list(map(int, rest.split(',')))
                segs[cid] = [(x, y) for x, y, r in zip(b[:-1], b[1:], rf) if not r]

    emb, meta, cids, off = [], [], [], 0
    for fp in sorted(glob.glob(os.path.join(a.bank_dir, 'bank_*.npz'))):
        d = np.load(fp); emb.append(d['emb']); m = d['meta'].copy(); m[:, 0] += off; meta.append(m)
        cids.extend([str(x) for x in d['clip_ids']]); off = len(cids)
    emb = np.concatenate(emb); meta = np.concatenate(meta)
    seg_video = np.array([rows[cids[i]][0] if cids[i] in rows else '?' for i in meta[:, 0]])
    print(f'[bank] {len(emb)} segments, {len(cids)} clips', flush=True)
    E = torch.from_numpy(emb.astype(np.float16)).to(dev)
    for i in range(0, len(E), 1_000_000): E[i:i + 1_000_000] = F.normalize(E[i:i + 1_000_000].float(), dim=-1).half()

    ck = torch.load(a.align_ckpt, map_location='cpu'); c = ck['config']
    model = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']).to(dev); model.load_state_dict(ck['model']); model.eval()
    smpl_x = load_smplx_model(How2Sign_SMPLX_Config().HUMAN_MODELS_PATH); h2c = load_chunks(CHUNKS)
    ae, _ = load_motion_ae(a.ae, dev)

    def fk(aa159):
        j = smplx_aa_to_upper3d(aa159, smpl_x, device=dev)
        return (j.detach().cpu().numpy() if hasattr(j, 'detach') else np.asarray(j)).astype(np.float32)
    def seg_motion(si):
        ci, _, _, fa, fb = meta[si]; aa = np.load(rows[cids[ci]][1])['axis_angle'][fa:fb].copy(); aa[:, 0, :] = 0.0
        return aa.reshape(len(aa), -1).astype(np.float32)
    def gt_seg_emb(cid):
        d = np.load(os.path.join(TOK, cid.replace(':', '__') + '.npz')); x = np.stack([d[p].astype(np.int64) for p in PARTS], 1)
        sp = [(fa // 4, max(fa // 4 + 1, -(-fb // 4))) for fa, fb in segs.get(cid, [])]
        if len(x) > a.max_tok:
            s = (len(x) - a.max_tok) // 2; x = x[s:s + a.max_tok]; sp = [(p - s, min(q - s, a.max_tok)) for p, q in sp if p >= s and p < s + a.max_tok]
        if not sp: sp = [(0, len(x))]
        sp = [(p, min(q, len(x))) for p, q in sp]
        xt = torch.from_numpy(x)[None].to(dev); m = torch.ones(1, len(x), dtype=torch.bool, device=dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            return model.embed_segments(xt, m, [(0, p, q) for p, q in sp]).half()
    def dists(R, G):
        out = {}
        for sp in ('raw', 'vel', 'style'): out[sp] = dtw_mean(to_space(R, sp), to_space(G, sp))
        out['ae'] = float(np.linalg.norm(encode_motion(ae, resample(R, 100), dev) - encode_motion(ae, resample(G, 100), dev)))
        return out

    rng = np.random.default_rng(a.seed); keep = {s: (np.ones(len(emb), bool) if s >= 1.0 else rng.random(len(emb)) < s) for s in a.scales}
    conds = ('text', 'oracle', 'oracle_c'); spaces = ('raw', 'vel', 'style', 'ae')
    res = {s: {cd: {sp: [] for sp in spaces} for cd in conds} for s in a.scales}; corp = []
    for n, cid in enumerate(wanted):
        video, npz, text, co = rows[cid]; corp.append(co)
        ch = h2c.get(hashlib.md5(text.encode()).hexdigest()); ch = merge_stopword_chunks(ch) if ch else [text]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')): zt = model.embed_chunks(ch, dev).half()
        zo = gt_seg_emb(cid)
        g = np.load(npz)['axis_angle'].copy(); g[:, 0, :] = 0.0; G = fk(g.reshape(len(g), -1).astype(np.float32))
        nsv = torch.from_numpy(seg_video != video).to(dev)
        for s in a.scales:
            ok = torch.from_numpy(keep[s]).to(dev) & nsv; pool = np.where(keep[s])[0]
            C = len(ch); sel = torch.linspace(0, len(zo) - 1, steps=min(C, len(zo))).round().long().to(zo.device)
            q = {'text': zt, 'oracle': zo, 'oracle_c': zo[sel]}  # oracle_c = motion query but only as many segments as the text has chunks
            for cd in conds:
                sim = (q[cd] @ E.T).float(); sim[:, ~ok] = -1e4; top = sim.argmax(-1).cpu().numpy()
                R = fk(np.concatenate([seg_motion(si) for si in top], 0))
                Rr = fk(np.concatenate([seg_motion(si) for si in rng.choice(pool, size=len(top))], 0))
                dR, dRr = dists(R, G), dists(Rr, G)
                for sp in spaces: res[s][cd][sp].append(dR[sp] / max(dRr[sp], 1e-9))
        if (n + 1) % 25 == 0: print(f'  {n+1}/{len(wanted)}', flush=True)
    corp = np.array(corp)
    out = {'align_ckpt': a.align_ckpt, 'bank_dir': a.bank_dir, 'n_clips': len(wanted), 'scales': {}}
    for s in a.scales:
        out['scales'][str(s)] = {}
        for cd in conds:
            for sp in spaces:
                r = np.array(res[s][cd][sp]); e = {'mean': float(r.mean()), 'median': float(np.median(r)), 'frac_below_1': float((r < 1).mean()),
                                                  'per_corpus_median': {c_: float(np.median(r[corp == c_])) for c_ in np.unique(corp)}}
                out['scales'][str(s)][f'{cd}/{sp}'] = e; print(s, cd, sp, json.dumps({k: v for k, v in e.items() if k != 'per_corpus_median'}), flush=True)
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); json.dump(out, open(a.out, 'w'), indent=1); print('wrote', a.out, flush=True)


if __name__ == '__main__': main()
