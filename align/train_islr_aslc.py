"""Isolated-sign classifier on ASL Citizen (2026-09-21): MMM motion encoder (4.1, motion-only pretraining) + mean pool + linear head over
the 2,731 ASL Citizen glosses. Official signer-disjoint split (train 40,154 / val 10,304 / test 32,928 clips; tokens from
align/tokenize_asl_citizen.py, trimmed to the hand-visible span).
Purpose: (1) an INDEPENDENT judge with real labels for mined sign instances (no lineage with any aligner / spotter: it never sees text
or our captions); (2) a second filter for dictionary instances. LICENCE: in-house only, weights and data are not released.
Augmentation: random token replacement (--tok_drop) and a random temporal crop that keeps >= 70 % of the clip. Left / right handedness
is NOT normalised (34.6 % of the clips are left-dominant); the classifier sees both renditions under the same label.
--init none trains the same architecture from scratch = ablation of the 4.1 pretraining on a labelled task.
  python align/train_islr_aslc.py --tokens <npz> --mmm_ckpt <pt> --tag islr_aslc_mmm
"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from pretrain.train_mmm import MMM


class ISLR(nn.Module):
    def __init__(self, mmm_ckpt, n_cls, init='mmm', drop=0.2):
        super().__init__()
        ck = torch.load(mmm_ckpt, map_location='cpu'); c = ck['config']; self.mmm_cfg = c
        self.mmm = MMM(c['K'], c['d'], c['layers'], c['heads'], c['max_len'] + 8)
        if init == 'mmm': self.mmm.load_state_dict(ck['model'])
        self.mmm.heads = None
        self.head = nn.Sequential(nn.LayerNorm(c['d']), nn.Dropout(drop), nn.Linear(c['d'], n_cls))
    def features(self, x, m):
        h = self.mmm.encode(x, m); w = m.float().unsqueeze(-1); return (h * w).sum(1) / w.sum(1).clamp_min(1)
    def forward(self, x, m): return self.head(self.features(x, m))


def batchify(tok, off, idx, max_len, K=512, train=False, tok_drop=0.0, rng=None):
    ts = []
    for i in idx:
        t = torch.from_numpy(tok[off[i]:off[i + 1]].astype(np.int64))
        if train and len(t) > 6:
            keep = int(np.ceil(len(t) * rng.uniform(0.7, 1.0))); s = rng.integers(0, len(t) - keep + 1); t = t[s:s + keep]
        ts.append(t[:max_len])
    L = max(len(t) for t in ts); x = torch.zeros(len(ts), L, 4, dtype=torch.long); m = torch.zeros(len(ts), L, dtype=torch.bool)
    for j, t in enumerate(ts): x[j, :len(t)] = t; m[j, :len(t)] = True
    if train and tok_drop > 0:
        r = torch.rand(x.shape) < tok_drop; x = torch.where(r, torch.randint(0, K, x.shape), x)
    return x, m


@torch.no_grad()
def evaluate(model, tok, off, idx, y, max_len, dev, bs=256):
    model.eval(); c1 = c5 = 0
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]; x, m = batchify(tok, off, b, max_len)
        with torch.autocast('cuda', dtype=torch.bfloat16): lg = model(x.to(dev), m.to(dev)).float()
        t5 = lg.topk(5, dim=1).indices.cpu(); yy = torch.from_numpy(y[b])
        c1 += int((t5[:, 0] == yy).sum()); c5 += int((t5 == yy[:, None]).any(1).sum())
    return c1 / len(idx), c5 / len(idx)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tokens', default='/scratch/rhong5/dataset/pooled_tokens/asl_citizen_tokens_v1.npz')
    ap.add_argument('--mmm_ckpt', default='/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/MotionMMM_Pooled/20260913_122233_job86413_mmm_d512_L8/best_model.pt')
    ap.add_argument('--init', default='mmm', choices=['mmm', 'none']); ap.add_argument('--epochs', type=int, default=40); ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--lr_enc', type=float, default=1e-4); ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--label_smooth', type=float, default=0.1); ap.add_argument('--tok_drop', type=float, default=0.1); ap.add_argument('--max_items', type=int, default=0)
    ap.add_argument('--out_dir', default='/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/ISLR_ASLC'); ap.add_argument('--tag', default='islr_aslc_mmm'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); dev = 'cuda'; torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    d = np.load(a.tokens, allow_pickle=True); tok, off = d['tokens'], d['offsets']; gl = d['gloss'].astype(str); sp = d['split'].astype(str)
    classes = sorted(set(gl[sp == 'train'].tolist())); ci = {g: k for k, g in enumerate(classes)}; y = np.array([ci.get(g, -1) for g in gl])
    ok = (y >= 0) & (np.diff(off) >= 2); idx = {s: np.where((sp == s) & ok)[0] for s in ['train', 'val', 'test']}
    if a.max_items: idx = {s: v[:a.max_items] for s, v in idx.items()}
    print({s: len(v) for s, v in idx.items()}, 'classes', len(classes), flush=True)
    model = ISLR(a.mmm_ckpt, len(classes), a.init).to(dev); max_len = model.mmm.pos.num_embeddings
    opt = torch.optim.AdamW([{'params': model.mmm.parameters(), 'lr': a.lr_enc if a.init == 'mmm' else a.lr}, {'params': model.head.parameters(), 'lr': a.lr}], weight_decay=a.wd)
    steps = a.epochs * (len(idx['train']) // a.batch_size); warm = max(1, int(0.05 * steps)); base = [g['lr'] for g in opt.param_groups]
    run = os.path.join(a.out_dir, time.strftime('%Y%m%d_%H%M%S') + f"_job{os.environ.get('SLURM_JOB_ID', '0')}_{a.tag}"); os.makedirs(run, exist_ok=True)
    best, step, log = -1, 0, []
    for ep in range(a.epochs):
        model.train(); perm = rng.permutation(idx['train']); tl = tn = 0
        for i in range(0, len(perm) - a.batch_size + 1, a.batch_size):
            f = step / warm if step < warm else 0.5 * (1 + np.cos(np.pi * (step - warm) / max(1, steps - warm)))
            for g, b0 in zip(opt.param_groups, base): g['lr'] = b0 * f
            b = perm[i:i + a.batch_size]; x, m = batchify(tok, off, b, max_len, model.mmm_cfg['K'], True, a.tok_drop, rng)
            with torch.autocast('cuda', dtype=torch.bfloat16): lg = model(x.to(dev), m.to(dev))
            loss = F.cross_entropy(lg.float(), torch.from_numpy(y[b]).to(dev), label_smoothing=a.label_smooth)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1; tl += float(loss); tn += 1
        v1, v5 = evaluate(model, tok, off, idx['val'], y, max_len, dev); log.append({'epoch': ep + 1, 'train_loss': tl / max(tn, 1), 'val_top1': v1, 'val_top5': v5})
        print(f'ep {ep + 1}/{a.epochs} loss {tl / max(tn, 1):.4f} VAL top1 {v1:.4f} top5 {v5:.4f}', flush=True)
        if v1 > best:
            best = v1; torch.save({'model': model.state_dict(), 'classes': classes, 'config': vars(a), 'mmm_cfg': model.mmm_cfg, 'epoch': ep + 1, 'val_top1': v1, 'val_top5': v5}, os.path.join(run, 'best_model.pt'))
    ck = torch.load(os.path.join(run, 'best_model.pt'), map_location='cpu'); model.load_state_dict(ck['model'])
    t1, t5 = evaluate(model, tok, off, idx['test'], y, max_len, dev)
    res = {'run': run, 'init': a.init, 'best_epoch': ck['epoch'], 'val_top1': ck['val_top1'], 'val_top5': ck['val_top5'], 'test_top1': t1, 'test_top5': t5, 'n': {s: int(len(v)) for s, v in idx.items()}, 'n_classes': len(classes), 'log': log}
    json.dump(res, open(os.path.join(run, 'result.json'), 'w'), indent=1); print('TEST', json.dumps({k: v for k, v in res.items() if k != 'log'}), flush=True)


if __name__ == '__main__':
    main()
