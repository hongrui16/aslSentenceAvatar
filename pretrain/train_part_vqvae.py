"""Train the part-wise VQ-VAE motion tokenizer on the pooled 25-fps SMPL-X corpus (roadmap 4.1, stage 1).

Loss = masked MSE on rot6d/expression per part + lambda_vel * velocity MSE + beta * commitment + lambda_fk * FK joint MSE
(44 upper-body joints from the decoded full pose through the fast SMPL-X kinematic tree). Val every --val_every steps on the
centre windows of the val list; best = lowest val recon+fk. Checkpoint dict: {'model', 'config', 'step', 'val'}.
  python pretrain/train_part_vqvae.py --train_list .../pretrain_all.txt --val_list .../pool_val.txt --steps 200000
"""
import argparse, os, sys, json, time, math, random
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from pretrain.motion_window_dataset import MotionWindowDataset, collate, PART_DIMS
from pretrain.part_vqvae import PartVQVAE
from utils.rotation_conversion import rot6d_to_matrix
from utils.smplx_fk_diff_fast import SMPLXForwardKinematicsFast

INDEX = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/curation/clip_index.tsv'


def safe_rot6d_to_aa(r6, eps=1e-6):
    R = rot6d_to_matrix(r6); tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)  # 2 sin(theta) * axis
    n = torch.sqrt((v * v).sum(-1, keepdim=True) + eps); theta = torch.atan2(n.squeeze(-1) / 2, (tr - 1) / 2)[..., None]
    return v / n * theta


def assemble_aa159(out):  # decoded parts -> (B, T, 159) axis-angle with zero root, for the FK loss
    B, T, _ = out['body'].shape; z = out['body'].new_zeros(B, T, 1, 3)
    body = safe_rot6d_to_aa(out['body'].reshape(B, T, 21, 6)); lh = safe_rot6d_to_aa(out['lhand'].reshape(B, T, 15, 6)); rh = safe_rot6d_to_aa(out['rhand'].reshape(B, T, 15, 6)); jaw = safe_rot6d_to_aa(out['face'][..., :6].reshape(B, T, 1, 6))
    return torch.cat([z, body, lh, rh, jaw], 2).reshape(B, T, 159)


def losses(model, batch, fk, a, device):
    feats = {p: batch[p].to(device, non_blocking=True) for p in PART_DIMS}; mask = batch['mask'].to(device)[..., None]
    out, idx, commit, ppl = model(feats); rec = {}; vel = {}
    for p in PART_DIMS:
        rec[p] = ((out[p] - feats[p]) ** 2 * mask).sum() / (mask.sum() * feats[p].shape[-1])
        vel[p] = (((out[p][:, 1:] - out[p][:, :-1]) - (feats[p][:, 1:] - feats[p][:, :-1])) ** 2 * mask[:, 1:]).sum() / (mask[:, 1:].sum() * feats[p].shape[-1])
    L_rec = sum(rec.values()); L_vel = sum(vel.values()); L_commit = sum(commit.values())
    with torch.autocast('cuda', enabled=False):
        J_gt = fk(batch['aa159'].to(device).float()); J = fk(assemble_aa159({p: o.float() for p, o in out.items()}))
        L_fk = (((J - J_gt) ** 2).sum(-1) * mask.float()).sum() / (mask.sum() * 44)
    total = L_rec + a.lambda_vel * L_vel + a.beta * L_commit + a.lambda_fk * L_fk
    return total, dict(rec=L_rec.item(), vel=L_vel.item(), commit=L_commit.item(), fk=L_fk.item(), mpjpe_cm=100 * (((J - J_gt) ** 2).sum(-1).sqrt() * mask.float()).sum().item() / (mask.sum().item() * 44), **{f'ppl_{p}': ppl[p].item() for p in PART_DIMS}, **{f'rec_{p}': rec[p].item() for p in PART_DIMS})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--index', default=INDEX); ap.add_argument('--train_list', required=True); ap.add_argument('--val_list', required=True)
    ap.add_argument('--window', type=int, default=64); ap.add_argument('--batch_size', type=int, default=256); ap.add_argument('--steps', type=int, default=200000); ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--warmup', type=int, default=2000)
    ap.add_argument('--K', type=int, default=512); ap.add_argument('--width', type=int, default=256); ap.add_argument('--down', type=int, default=2); ap.add_argument('--beta', type=float, default=0.25); ap.add_argument('--lambda_vel', type=float, default=0.5); ap.add_argument('--lambda_fk', type=float, default=1.0)
    ap.add_argument('--val_every', type=int, default=2000); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--max_items', type=int, default=None); ap.add_argument('--out_dir', default='/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/MotionVQ_Pooled'); ap.add_argument('--tag', default='vq'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--resume', default=None)
    a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = f"{time.strftime('%Y%m%d_%H%M%S')}_job{os.environ.get('SLURM_JOB_ID', 'local')}_{a.tag}"; ckdir = os.path.join(a.out_dir, run); logdir = os.path.join(_repo, 'zlog', 'MotionVQ_Pooled', run); os.makedirs(ckdir, exist_ok=True); os.makedirs(logdir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(logdir, 'config.json'), 'w'), indent=1); log = open(os.path.join(logdir, 'train.log'), 'a')
    def P(*s):
        m = ' '.join(str(x) for x in s); print(m, flush=True); log.write(m + '\n'); log.flush()
    tr = MotionWindowDataset(a.index, a.train_list, a.window, True, a.max_items); va = MotionWindowDataset(a.index, a.val_list, a.window, False, a.max_items)
    dl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=a.workers > 0); dv = DataLoader(va, a.batch_size, shuffle=False, num_workers=min(4, a.workers), collate_fn=collate)
    model = PartVQVAE(PART_DIMS, a.K, a.width, a.down).to(device); fk = SMPLXForwardKinematicsFast().to(device).eval()
    for p_ in fk.parameters(): p_.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), a.lr, betas=(0.9, 0.99), weight_decay=1e-4); sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    step, best = 0, float('inf')
    if a.resume: ck = torch.load(a.resume, map_location='cpu'); model.load_state_dict(ck['model']); step = ck['step']; P('resumed', a.resume, 'step', step)
    P(f'params {sum(p.numel() for p in model.parameters())/1e6:.1f}M | train clips {len(tr)} val clips {len(va)} | device {device}')
    model.train(); t0 = time.time(); agg = {}
    while step < a.steps:
        for batch in dl:
            for g in opt.param_groups: g['lr'] = a.lr * sched(step)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')): total, st = losses(model, batch, fk, a, device)
            opt.zero_grad(set_to_none=True); total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            for k, v in st.items(): agg[k] = agg.get(k, 0) + v
            if step % 100 == 0:
                P(f"step {step} lr {opt.param_groups[0]['lr']:.2e} total {total.item():.4f} " + ' '.join(f'{k} {v/100:.4f}' for k, v in agg.items()) + f' | {(time.time()-t0)/step:.2f}s/it'); agg = {}
            if step % a.val_every == 0 or step == a.steps:
                model.eval(); vs = {}; n = 0
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(device == 'cuda')):
                    for vb in dv:
                        _, s_ = losses(model, vb, fk, a, device); n += 1
                        for k, v in s_.items(): vs[k] = vs.get(k, 0) + v
                vs = {k: v / n for k, v in vs.items()}; score = vs['rec'] + a.lambda_fk * vs['fk']; usage = {p: int((model.vq[p].ema_count > 1.0).sum()) for p in PART_DIMS}
                P(f'VAL step {step} score {score:.5f} ' + ' '.join(f'{k} {v:.4f}' for k, v in vs.items()) + f' | codes used {usage}')
                ck = {'model': model.state_dict(), 'config': vars(a), 'step': step, 'val': vs, 'part_dims': PART_DIMS}; torch.save(ck, os.path.join(ckdir, 'newest_model.pt'))
                if score < best: best = score; torch.save(ck, os.path.join(ckdir, 'best_model.pt')); P(f'  -> new best {best:.5f}')
                model.train()
            if step >= a.steps: break
    P('DONE', ckdir)


if __name__ == '__main__':
    main()
