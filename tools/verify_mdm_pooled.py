"""
Conditional-collapse diagnostic for the MDM diffusion model trained on the pooled
curation corpora (trainMotionDiffusion_pooled.py), fixed test set. Diffusion analogue
of tools/verify_clean_regression_pooled.py: same test list, same GT loading, same
FK / descriptor space, same distances, so the rows are directly comparable with the
regression rows and with the raw-joint noise floors (tools/noise_floor_pooled.py).

Differences from the regression script:
  - generation = DDIM sampling (model.generate, batched), sentence conditioning only
  - sampling is stochastic, so d_intra (same sentence, second seed) is measured on a
    subset (--intra_n) instead of being 0 by design
The MDM emits all non-root joints (156-D); lower-body channels carry zero loss weight
in training, so they are zeroed here exactly like the regression path (upper body only).
"""
import os, sys, json, random, argparse
import numpy as np
import torch

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo)
sys.path.insert(0, os.path.join(_repo, 'infer_eval'))
sys.path.insert(0, os.path.join(_repo, 'tools'))
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

from network.MotionDiffusionModelV1 import MotionDiffusionModelV1
from infer_eval.generate_smplx_param import load_model_weight, load_smplx_model
from config import Pooled_SMPLX_Config
from dataloader.PooledSMPLXDataset import PooledSMPLXDataset
from utils.rotation_conversion import get_joint_slices
from utils.motion_ae_fid import load_motion_ae, encode_motion, smplx_aa_to_upper3d
from verify_clean_regression_pooled import load_gt159, cross_pairs_l2, pairwise_l2, frechet_fid


@torch.no_grad()
def _gen_batch(model, sentences, seed, seq_len, device, num_steps, zero_slices):
    torch.manual_seed(seed); np.random.seed(seed)
    m = model.generate(list(sentences), seq_len=seq_len, device=device, num_steps=num_steps)  # (B, T, 159), root zeros
    m = m.float().cpu().numpy()
    m[:, :, zero_slices] = 0.0
    return m


def main(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = Pooled_SMPLX_Config()
    cfg.POOL_LIST_TEST = args.pool_test
    if args.pool_index: cfg.POOL_INDEX = args.pool_index
    cfg.USE_ROT6D = False; cfg.USE_UPPER_BODY = False; cfg.ROOT_NORMALIZE = True
    cfg.N_FEATS = 3; cfg.TARGET_SEQ_LEN = args.target_seq_len
    cfg.MODEL_VERSION = 'v1'

    meta = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cd = meta.get('config', {})
    for k in ('MODEL_DIM', 'N_HEADS', 'N_LAYERS', 'DROPOUT', 'MAX_SEQ_LEN', 'NUM_DIFFUSION_STEPS',
              'TEXT_ENCODER_TYPE', 'CLIP_MODEL_NAME', 'CLIP_DIM', 'T5_MODEL_NAME',
              'USE_LABEL_INDEX_COND', 'USE_UPPER_BODY', 'ROOT_NORMALIZE', 'USE_ROT6D', 'N_FEATS'):
        if k in cd:
            setattr(cfg, k, cd[k])
    assert not cfg.USE_ROT6D, "axis-angle checkpoints only"
    print(f"[ckpt] MDM v1 Pooled epoch={meta.get('epoch')} step={meta.get('global_step')} best_loss={meta.get('best_loss')}")
    ck_epoch, ck_step = meta.get('epoch'), meta.get('global_step')
    del meta

    test_ds = PooledSMPLXDataset(mode='test', cfg=cfg)
    cfg.INPUT_DIM = test_ds.input_dim
    print(f"[data] Pooled fixed TEST: {len(test_ds)} sentences")

    model = MotionDiffusionModelV1(cfg)
    model = load_model_weight(model, args.checkpoint, device)
    model.eval()

    g = get_joint_slices(n_feats=3)
    zero_slices = sorted(set(g['ROOT'] + g['LOWER_BODY']))

    if args.n_sentences <= 0 or args.n_sentences >= len(test_ds):
        idxs = list(range(len(test_ds)))
    else:
        idxs = random.sample(range(len(test_ds)), args.n_sentences)
    samples = [{'sentence': test_ds.data_list[i][0], 'src': test_ds.data_list[i][1], 'corpus': test_ds.corpus_list[i]} for i in idxs]
    n = len(samples)

    smpl_x = load_smplx_model(cfg.HUMAN_MODELS_PATH)
    ae_model = None if args.raw_space else load_motion_ae(args.motion_ae_ckpt, device=device)[0]

    def encode(m159):
        j3d = smplx_aa_to_upper3d(m159, smpl_x, device=device)  # (T, 132) root-zeroed upper-body joints
        if ae_model is None:
            j = j3d.detach().cpu().numpy() if hasattr(j3d, 'detach') else np.asarray(j3d)
            return j.reshape(-1).astype(np.float32)
        return encode_motion(ae_model, j3d, device=device)

    seed_a, seed_b = 42, 99
    n_intra = n if args.intra_n <= 0 else min(args.intra_n, n)
    intra_set = set(random.sample(range(n), n_intra))
    print(f"[gen] {n} sentences, DDIM {args.num_steps} steps, batch {args.batch_size}; d_intra on {n_intra}")

    Z, Z_gt, f_gen, f_gt = [], [], [], []
    Z_b = {}
    for s in range(0, n, args.batch_size):
        chunk = samples[s:s + args.batch_size]
        sents = [sm['sentence'] for sm in chunk]
        M = _gen_batch(model, sents, seed_a + s, args.target_seq_len, device, args.num_steps, zero_slices)
        for j, sm in enumerate(chunk):
            Z.append(encode(M[j])); f_gen.append(M[j].mean(axis=0))
            gt = load_gt159(sm['src'], args.target_seq_len, force_len=args.raw_space)
            Z_gt.append(encode(gt)); f_gt.append(gt.mean(axis=0))
        sel = [j for j in range(len(chunk)) if (s + j) in intra_set]
        if sel:
            Mb = _gen_batch(model, [sents[j] for j in sel], seed_b + s, args.target_seq_len, device, args.num_steps, zero_slices)
            for k_, j in enumerate(sel):
                Z_b[s + j] = encode(Mb[k_])
        print(f"  {min(s + args.batch_size, n)}/{n}", flush=True)
    Z = np.stack(Z); Z_gt = np.stack(Z_gt)
    ib = sorted(Z_b); Zb = np.stack([Z_b[i] for i in ib])

    d_intra = pairwise_l2(Z[ib], Zb)
    d_inter = cross_pairs_l2(Z)
    d_inter_gt = cross_pairs_l2(Z_gt)
    d_pair = pairwise_l2(Z, Z_gt)
    perm = list(range(n)); random.shuffle(perm)
    d_pair_rand = pairwise_l2(Z, Z_gt[perm])
    fid_pose = frechet_fid(np.stack(f_gt), np.stack(f_gen)) if n > 160 else None
    fid_ae = frechet_fid(Z_gt, Z) if (not args.raw_space and n > 64) else None

    # per-corpus breakdown (pairs restricted within corpus) -> compare rho_faith with the per-corpus noise floors
    floors = {}
    if args.noise_floor and os.path.isfile(args.noise_floor):
        floors = {c: v.get('rho_ref') for c, v in json.load(open(args.noise_floor)).get('per_corpus', {}).items()}
    per_corpus = {}
    corp = np.array([sm['corpus'] for sm in samples]); ib_arr = np.array(ib)
    for c in sorted(set(corp.tolist())):
        m_ = corp == c; nc = int(m_.sum())
        if nc < 3: continue
        Zc, Zgc = Z[m_], Z_gt[m_]; pc = list(range(nc)); random.shuffle(pc)
        di, dig, dp, dpr = cross_pairs_l2(Zc), cross_pairs_l2(Zgc), pairwise_l2(Zc, Zgc), pairwise_l2(Zc, Zgc[pc])
        mb = m_[ib_arr]
        dia = pairwise_l2(Z[ib_arr[mb]], Zb[mb]) if mb.sum() > 0 else None
        per_corpus[c] = {'n': nc, 'd_intra': dia, 'd_inter': di, 'd_inter_gt': dig, 'd_pair_to_gt': dp, 'd_pair_to_gt_random': dpr,
                         'rho_cond_inter_over_intra': di / dia if dia else None,
                         'rho_div_inter_over_gt_inter': di / dig if dig else None,
                         'rho_faith_pair_over_random': dp / dpr if dpr else None,
                         'rho_ref_noise_floor': floors.get(c)}
        fl = f" floor={floors[c]:.3f}" if floors.get(c) is not None else ""
        print(f"  [{c}] n={nc} rho_div={di / dig:.3f} rho_faith={dp / dpr:.3f}{fl}")

    out = {'checkpoint': args.checkpoint, 'ckpt_epoch': ck_epoch, 'ckpt_step': ck_step,
           'dataset': 'PooledSMPLX', 'split': 'test_fixed', 'pool_test': args.pool_test,
           'space': 'raw_joints' if args.raw_space else 'moae', 'arch': 'mdm_v1', 'cond_mode': 'sentence',
           'ddim_steps': args.num_steps, 'n_sentences': n, 'n_intra': n_intra, 'target_seq_len': args.target_seq_len,
           'per_corpus': per_corpus,
           'd_intra': d_intra, 'd_inter': d_inter, 'd_inter_gt': d_inter_gt,
           'd_pair_to_gt': d_pair, 'd_pair_to_gt_random': d_pair_rand,
           'fid_pose': fid_pose, 'fid_ae': fid_ae,
           'rho_cond_inter_over_intra': d_inter / d_intra if d_intra else None,
           'rho_div_inter_over_gt_inter': d_inter / d_inter_gt if d_inter_gt else None,
           'rho_faith_pair_over_random': d_pair / d_pair_rand if d_pair_rand else None}

    print("\n" + "=" * 70)
    print("VERDICT (Pooled MDM, fixed TEST):")
    print(f"  d_intra             = {d_intra:.4f}   (same sentence, second seed)")
    print(f"  d_inter             = {d_inter:.4f}")
    print(f"  d_inter_GT          = {d_inter_gt:.4f}")
    print(f"  d_pair_to_gt        = {d_pair:.4f}")
    print(f"  d_pair_to_gt_random = {d_pair_rand:.4f}")
    if fid_pose is not None: print(f"  FID (pose mean-pool) = {fid_pose:.4f}")
    if fid_ae is not None:   print(f"  FID (MoAE latent)    = {fid_ae:.4f}")
    print(f"  rho_cond (inter/intra)    = {out['rho_cond_inter_over_intra']:.3f}  (~1 = text ignored)")
    print(f"  rho_div  (inter/inter_GT) = {out['rho_div_inter_over_gt_inter']:.3f}")
    print(f"  rho_faith(pair/random)    = {out['rho_faith_pair_over_random']:.3f}  (compare per-corpus values with the noise floors)")
    print("=" * 70)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"saved -> {args.output}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--motion_ae_ckpt', default=None)
    p.add_argument('--raw_space', action='store_true', help='no MoAE: distances in the root-zeroed 3D joint trajectory space (T*132)')
    p.add_argument('--n_sentences', type=int, default=-1)
    p.add_argument('--intra_n', type=int, default=1000, help='sentences that get a second-seed sample for d_intra (<=0 = all)')
    p.add_argument('--num_steps', type=int, default=50, help='DDIM steps')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--target_seq_len', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', type=str, default=None)
    p.add_argument('--pool_test', type=str, required=True)
    p.add_argument('--pool_index', type=str, default=None)
    p.add_argument('--noise_floor', type=str, default=None, help='noise_floor_pooled.json, only used to print/store rho_ref next to rho_faith')
    main(p.parse_args())
