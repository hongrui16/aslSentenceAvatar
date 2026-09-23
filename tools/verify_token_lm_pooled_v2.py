"""v2 (2026-09-22): same diagnostic for the 4.4 v2 TokenLM (train_token_lm_v2.py, aligner-conditioned). v1 file untouched.
Conditional-collapse diagnostic for the 4.4 TokenLM (text -> VQ tokens) on the pooled fixed test set.

Same protocol/space/ratios as verify_clean_regression_pooled.py --raw_space: descriptor = flattened
(seq_len, 132) root-zeroed upper-body FK joints; rho_div = inter/inter_GT, rho_faith = pair/random,
per-corpus breakdown. Generation: greedy (deterministic, d_intra = 0) unless --temperature > 0.
Compare rho_faith against results/zlog/curation/noise_floor_pooled.json (rho_ref per corpus).

  python tools/verify_token_lm_pooled.py --checkpoint <best_model.pt> --vq_ckpt <vq best_model.pt> \
      --pool_test <test_fixed.txt> --output <json>
"""
import argparse, csv, json, os, random, sys
import numpy as np
import torch

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo); sys.path.insert(0, os.path.join(_repo, 'infer_eval'))

from infer_eval.generate_smplx_param import load_smplx_model
from utils.motion_ae_fid import smplx_aa_to_upper3d
from config import How2Sign_SMPLX_Config
from pretrain.part_vqvae import PartVQVAE
from pretrain.train_part_vqvae import assemble_aa159
from pretrain.train_token_lm import PARTS
from pretrain.train_token_lm_v2 import TokenLMv2

INDEX = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/curation/clip_index.tsv'


def resample(x, n):
    idx = np.round(np.linspace(0, len(x) - 1, n)).astype(int)
    return x[idx]


def cross_pairs_l2(M, chunk=256):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.as_tensor(np.asarray(M, np.float32), device=dev); n = X.shape[0]; sq = (X * X).sum(1); tot = 0.0
    for i in range(0, n, chunk):
        A = X[i:i + chunk]; d2 = (sq[i:i + chunk, None] + sq[None, :] - 2.0 * A @ X.T).clamp_min(0.0); d = d2.sqrt()
        idx = torch.arange(i, min(i + chunk, n), device=dev); d[idx - i, idx] = 0.0; tot += float(d.sum())
    return tot / (n * (n - 1))


def pairwise_l2(a, b):
    return float(np.linalg.norm(a - b, axis=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--vq_ckpt', required=True)
    ap.add_argument('--pool_test', required=True); ap.add_argument('--pool_index', default=INDEX)
    ap.add_argument('--output', required=True); ap.add_argument('--n_sentences', type=int, default=0)
    ap.add_argument('--target_seq_len', type=int, default=100); ap.add_argument('--batch', type=int, default=48)
    ap.add_argument('--temperature', type=float, default=0.0); ap.add_argument('--max_new', type=int, default=192)
    ap.add_argument('--noise_floor', default='/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/results/zlog/curation/noise_floor_pooled.json')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    wanted = [l.strip() for l in open(a.pool_test) if l.strip()]; wset = set(wanted)
    rows = {}
    with open(a.pool_index) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            if r['clip_id'] in wset: rows[r['clip_id']] = (r['corpus'], r['npz'], r['text'].strip())
    samples = [(c,) + rows[c] for c in wanted if c in rows and rows[c][2]]
    if a.n_sentences and a.n_sentences < len(samples): samples = random.sample(samples, a.n_sentences)
    print(f'[data] {len(samples)} test sentences', flush=True)

    ck = torch.load(a.vq_ckpt, map_location='cpu'); cfg = ck['config']
    vq = PartVQVAE(ck['part_dims'], cfg['K'], cfg['width'], cfg['down']).to(dev); vq.load_state_dict(ck['model']); vq.eval()
    lck = torch.load(a.checkpoint, map_location='cpu'); lc = lck['config']
    lm = TokenLMv2(lc['K'], lc['d'], lc['layers'], lc['heads'], lc['max_len'] + 8, text_encoder=lc['text_encoder'], text_mode=lc['text_mode'],
                   align_ckpt=lc['align_ckpt'] or None, align_module=lc['align_module'], gloss_dir=lc['gloss_dir']).to(dev)
    lm.load_state_dict(lck['model']); lm.eval()
    smpl_x = load_smplx_model(How2Sign_SMPLX_Config().HUMAN_MODELS_PATH)

    def fk_desc(aa159_np):  # (T,159) -> flattened (seq_len,132)
        j = smplx_aa_to_upper3d(resample(aa159_np, a.target_seq_len), smpl_x, device=dev)
        j = j.detach().cpu().numpy() if hasattr(j, 'detach') else np.asarray(j)
        return j.reshape(-1).astype(np.float32)

    Z, Z_gt, corp, gen_lens = [], [], [], []
    for i in range(0, len(samples), a.batch):
        chunk = samples[i:i + a.batch]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            toks = lm.generate([s[3] for s in chunk], dev, max_new=a.max_new, temperature=a.temperature)
        for (cid, corpus, npz, text), t in zip(chunk, toks):
            gen_lens.append(len(t))
            if len(t) < 2: t = torch.zeros(2, 4, dtype=torch.long)
            idx = {p: t[:, k][None].to(dev) for k, p in enumerate(PARTS)}
            with torch.no_grad():
                aa = assemble_aa159(vq.decode(idx))[0].float().cpu().numpy()
            Z.append(fk_desc(aa))
            g = np.load(npz)['axis_angle'].copy(); g[:, 0, :] = 0.0
            Z_gt.append(fk_desc(g.reshape(len(g), -1).astype(np.float32)))
            corp.append(corpus)
        if (i // a.batch) % 10 == 0: print(f'  {i + len(chunk)}/{len(samples)}', flush=True)
    Z = np.stack(Z); Z_gt = np.stack(Z_gt); corp = np.array(corp); n = len(Z)

    floors = {}
    if os.path.exists(a.noise_floor):
        floors = {c: v['rho_ref'] for c, v in json.load(open(a.noise_floor))['per_corpus'].items()}

    d_inter, d_inter_gt = cross_pairs_l2(Z), cross_pairs_l2(Z_gt)
    perm = list(range(n)); random.shuffle(perm)
    d_pair, d_rand = pairwise_l2(Z, Z_gt), pairwise_l2(Z, Z_gt[perm])
    per_corpus = {}
    for c in sorted(set(corp.tolist())):
        m = corp == c; nc = int(m.sum())
        if nc < 3: continue
        Zc, Zgc = Z[m], Z_gt[m]; pc = list(range(nc)); random.shuffle(pc)
        dp, dpr = pairwise_l2(Zc, Zgc), pairwise_l2(Zc, Zgc[pc])
        per_corpus[c] = {'n': nc, 'rho_div': cross_pairs_l2(Zc) / cross_pairs_l2(Zgc),
                         'rho_faith': dp / dpr, 'noise_floor_rho_ref': floors.get(c)}
        print(f"  [{c}] n={nc} rho_div={per_corpus[c]['rho_div']:.3f} rho_faith={per_corpus[c]['rho_faith']:.3f} floor={floors.get(c)}", flush=True)

    out = {'checkpoint': a.checkpoint, 'vq_ckpt': a.vq_ckpt, 'pool_test': a.pool_test,
           'space': 'raw_joints', 'arch': 'token_lm_v2', 'text_mode': lc['text_mode'], 'align_ckpt': lc.get('align_ckpt'), 'temperature': a.temperature, 'n_sentences': n,
           'gen_len_tokens': {'mean': float(np.mean(gen_lens)), 'p50': float(np.percentile(gen_lens, 50)),
                              'p90': float(np.percentile(gen_lens, 90)), 'frac_maxed': float(np.mean(np.array(gen_lens) >= a.max_new))},
           'd_inter': d_inter, 'd_inter_gt': d_inter_gt, 'd_pair_to_gt': d_pair, 'd_pair_to_gt_random': d_rand,
           'rho_div_inter_over_gt_inter': d_inter / d_inter_gt, 'rho_faith_pair_over_random': d_pair / d_rand,
           'per_corpus': per_corpus, 'noise_floor_file': a.noise_floor}
    os.makedirs(os.path.dirname(a.output) or '.', exist_ok=True)
    json.dump(out, open(a.output, 'w'), indent=1)
    print(f"\nVERDICT: rho_div {out['rho_div_inter_over_gt_inter']:.3f} rho_faith {out['rho_faith_pair_over_random']:.3f}")
    print('wrote', a.output, flush=True)


if __name__ == '__main__': main()
