"""Roadmap 4.5-3: retrieval-based generation minimal closed loop + collapse diagnostic.

Per test sentence: phrase chunks -> top-k segments from the bank (same-video excluded) -> greedy
context-aware selection (among top-k, pick the candidate whose start pose is closest to the previous
segment's end pose; first chunk takes top-1) -> stitch with short linear transitions in axis-angle
space (gap length scales with pose discontinuity, 2-6 frames) -> (T,159) root-zeroed motion.
Diagnostic identical to verify_token_lm_pooled: raw-joint descriptors, rho_div / rho_faith per corpus
vs the noise floors. Retrieved segments are real human motion by construction; rho_faith measures
whether retrieval tracks the text. Also saves --n_save stitched npz for later rendering.

  python align/retrieve_stitch.py --align_ckpt <pt> --bank_dir <dir> --test_list <txt> --output <json>
"""
import argparse, csv, glob, hashlib, json, os, random, sys
import numpy as np, torch, torch.nn.functional as F

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, 'infer_eval'))
from align.align_dataset import INDEX, load_chunks, merge_stopword_chunks
from align.train_align import AlignModel
from tools.verify_token_lm_pooled import cross_pairs_l2, pairwise_l2, resample
from infer_eval.generate_smplx_param import load_smplx_model
from utils.motion_ae_fid import smplx_aa_to_upper3d
from config import How2Sign_SMPLX_Config

CHUNKS = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/alignment/chunks'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--align_ckpt', required=True); ap.add_argument('--bank_dir', required=True)
    ap.add_argument('--test_list', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--index', default=INDEX); ap.add_argument('--n_sentences', type=int, default=0)
    ap.add_argument('--topk', type=int, default=10); ap.add_argument('--target_seq_len', type=int, default=100)
    ap.add_argument('--n_save', type=int, default=8)
    ap.add_argument('--noise_floor', default='/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/results/zlog/curation/noise_floor_pooled.json')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    rows = {}
    with open(a.index) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            rows[r['clip_id']] = (r['corpus'], r['video'], r['npz'], r['text'].strip())
    wanted = [l.strip() for l in open(a.test_list) if l.strip()]
    wanted = [c for c in wanted if c in rows and rows[c][3]]
    if a.n_sentences and a.n_sentences < len(wanted): wanted = random.sample(wanted, a.n_sentences)
    print(f'[data] {len(wanted)} test sentences', flush=True)

    emb, meta, cids = [], [], []; off = 0
    for fp in sorted(glob.glob(os.path.join(a.bank_dir, 'bank_*.npz'))):
        d = np.load(fp); emb.append(d['emb']); m = d['meta'].copy(); m[:, 0] += off
        meta.append(m); cids.extend([str(x) for x in d['clip_ids']]); off = len(cids)
    emb = np.concatenate(emb); meta = np.concatenate(meta)
    seg_video = np.array([rows[cids[i]][1] if cids[i] in rows else '?' for i in meta[:, 0]])
    E = torch.from_numpy(emb.astype(np.float16)).to(dev)
    for i in range(0, len(E), 1_000_000):
        E[i:i + 1_000_000] = F.normalize(E[i:i + 1_000_000].float(), dim=-1).half()
    print(f'[bank] {len(emb)} segments', flush=True)

    ck = torch.load(a.align_ckpt, map_location='cpu'); c = ck['config']
    model = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']).to(dev)
    model.load_state_dict(ck['model']); model.eval()
    smpl_x = load_smplx_model(How2Sign_SMPLX_Config().HUMAN_MODELS_PATH)
    h2c = load_chunks(CHUNKS)

    aa_cache = {}
    def seg_aa(si):  # (T,159) root-zeroed axis-angle of bank segment si
        ci, _, _, fa, fb = meta[si]
        npz = rows[cids[ci]][2]
        if npz not in aa_cache:
            x = np.load(npz)['axis_angle'].copy(); x[:, 0, :] = 0.0
            aa_cache[npz] = x.reshape(len(x), -1).astype(np.float32)
            if len(aa_cache) > 4000: aa_cache.pop(next(iter(aa_cache)))
        return aa_cache[npz][fa:fb]

    def fk_desc(aa159):
        j = smplx_aa_to_upper3d(resample(aa159, a.target_seq_len), smpl_x, device=dev)
        return (j.detach().cpu().numpy() if hasattr(j, 'detach') else np.asarray(j)).reshape(-1).astype(np.float32)

    def stitch(parts):
        out = [parts[0]]
        for p in parts[1:]:
            prev = out[-1][-1]; gap = float(np.linalg.norm(prev - p[0]))
            n = int(np.clip(round(gap * 2.0), 2, 6))
            w = np.linspace(0, 1, n + 2)[1:-1, None].astype(np.float32)
            out.append(prev[None] * (1 - w) + p[0][None] * w); out.append(p)
        return np.concatenate(out, 0)

    os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
    save_dir = os.path.splitext(a.output)[0] + '_samples'; os.makedirs(save_dir, exist_ok=True)
    Z, Z_gt, corp, seg_counts = [], [], [], []
    for n, cid in enumerate(wanted):
        corpus, video, npz, text = rows[cid]
        ch = h2c.get(hashlib.md5(text.encode()).hexdigest())
        ch = merge_stopword_chunks(ch) if ch else [text]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            zt = model.embed_chunks(ch, dev).half()
        sim = (zt @ E.T).float()
        sim[:, torch.from_numpy(seg_video == video).to(dev)] = -1e4
        top = sim.topk(a.topk, dim=-1).indices.cpu().numpy()  # (C, k)
        parts, prev_end = [], None
        for ci_ in range(len(ch)):
            cands = [seg_aa(si) for si in top[ci_]]
            cands = [x for x in cands if len(x) >= 2] or [seg_aa(top[ci_][0])]
            if prev_end is None: pick = cands[0]
            else: pick = min(cands, key=lambda x: float(np.linalg.norm(x[0] - prev_end)))
            parts.append(pick); prev_end = pick[-1]
        R = stitch(parts); seg_counts.append(len(parts))
        Z.append(fk_desc(R))
        g = np.load(npz)['axis_angle'].copy(); g[:, 0, :] = 0.0
        Z_gt.append(fk_desc(g.reshape(len(g), -1).astype(np.float32)))
        corp.append(corpus)
        if n < a.n_save:
            np.savez(os.path.join(save_dir, f'{n:02d}_{cid.replace(":", "__")}.npz'),
                     axis_angle=R.reshape(len(R), 53, 3), fps=25.0, text=text)
        if (n + 1) % 50 == 0: print(f'  {n+1}/{len(wanted)}', flush=True)
    Z = np.stack(Z); Z_gt = np.stack(Z_gt); corp = np.array(corp); nn_ = len(Z)

    floors = {}
    if os.path.exists(a.noise_floor):
        floors = {k: v['rho_ref'] for k, v in json.load(open(a.noise_floor))['per_corpus'].items()}
    perm = list(range(nn_)); random.shuffle(perm)
    out = {'align_ckpt': a.align_ckpt, 'bank_dir': a.bank_dir, 'n_sentences': nn_, 'topk': a.topk,
           'mean_segments_per_sentence': float(np.mean(seg_counts)),
           'd_inter': cross_pairs_l2(Z), 'd_inter_gt': cross_pairs_l2(Z_gt),
           'd_pair_to_gt': pairwise_l2(Z, Z_gt), 'd_pair_to_gt_random': pairwise_l2(Z, Z_gt[perm])}
    out['rho_div'] = out['d_inter'] / out['d_inter_gt']; out['rho_faith'] = out['d_pair_to_gt'] / out['d_pair_to_gt_random']
    per_corpus = {}
    for cc in sorted(set(corp.tolist())):
        m = corp == cc; nc = int(m.sum())
        if nc < 3: continue
        Zc, Zgc = Z[m], Z_gt[m]; pc = list(range(nc)); random.shuffle(pc)
        per_corpus[cc] = {'n': nc, 'rho_div': cross_pairs_l2(Zc) / cross_pairs_l2(Zgc),
                          'rho_faith': pairwise_l2(Zc, Zgc) / pairwise_l2(Zc, Zgc[pc]),
                          'noise_floor_rho_ref': floors.get(cc)}
        print(f"  [{cc}] n={nc} rho_div={per_corpus[cc]['rho_div']:.3f} rho_faith={per_corpus[cc]['rho_faith']:.3f} floor={floors.get(cc)}", flush=True)
    out['per_corpus'] = per_corpus
    json.dump(out, open(a.output, 'w'), indent=1)
    print(f"\nVERDICT: rho_div {out['rho_div']:.3f} rho_faith {out['rho_faith']:.3f} (samples in {save_dir})")
    print('wrote', a.output, flush=True)


if __name__ == '__main__': main()
