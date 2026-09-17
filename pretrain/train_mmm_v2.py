"""MMM v2 (2026-09-17): same model as train_mmm.py, MIXED masking instead of whole-time-step masking.
Per (time step, part) cell: with prob --mask_time a whole time step is masked (all 4 parts, as v1); independently with
prob --mask_part single cells are masked, so the model must also learn cross-part conditional structure
("given body and right hand at t, what is the left hand"). Selected cells: 80% [MASK] / 10% random / 10% kept.
Defaults mask_time 0.15 + mask_part 0.15 ~ 28% of cells. Everything else identical to v1 (see pretrain/VERSIONS.md).
"""
import argparse, os, sys, glob, json, math, random, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
PARTS = ['body', 'lhand', 'rhand', 'face']

class TokenDataset(Dataset):
    def __init__(self, tok_dir, list_txt, max_len=256, train=True):
        ids = [l.strip() for l in open(list_txt) if l.strip()]; self.files = [os.path.join(tok_dir, c.replace(':', '__') + '.npz') for c in ids]; self.files = [f for f in self.files if os.path.exists(f)]
        self.max_len, self.train = max_len, train; print(f'[TokenDataset] {len(self.files)} clips', flush=True)
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i]); toks = np.stack([d[p].astype(np.int64) for p in PARTS], 1)  # (n, 4)
        n = len(toks)
        if n > self.max_len: s = random.randint(0, n - self.max_len) if self.train else (n - self.max_len) // 2; toks = toks[s:s + self.max_len]
        return torch.from_numpy(toks)

def collate(batch):
    L = max(len(b) for b in batch); x = torch.zeros(len(batch), L, 4, dtype=torch.long); m = torch.zeros(len(batch), L, dtype=torch.bool)
    for i, b in enumerate(batch): x[i, :len(b)] = b; m[i, :len(b)] = True
    return x, m

class MMM(nn.Module):
    def __init__(self, K=512, d=512, layers=8, heads=8, max_len=512, drop=0.1):
        super().__init__(); self.K = K; self.mask_id = K  # extra [MASK] id per part
        self.emb = nn.ModuleList([nn.Embedding(K + 1, d) for _ in PARTS]); self.part_proj = nn.Linear(4 * d, d); self.pos = nn.Embedding(max_len, d)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True), layers); self.norm = nn.LayerNorm(d); self.heads = nn.ModuleList([nn.Linear(d, K) for _ in PARTS])
    def encode(self, x, pad_mask):  # x (B, L, 4) -> (B, L, d)
        h = self.part_proj(torch.cat([e(x[..., i]) for i, e in enumerate(self.emb)], -1)) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        return self.norm(self.enc(h, src_key_padding_mask=~pad_mask))
    def forward(self, x, pad_mask): h = self.encode(x, pad_mask); return torch.stack([hd(h) for hd in self.heads], 2)  # (B, L, 4, K)

def mask_tokens(x, valid, K, p_time=0.15, p_part=0.15):
    """x (B,L,4). Returns masked input and a (B,L,4) bool of cells to predict."""
    B, L, P = x.shape
    sel_t = (torch.rand(B, L, device=x.device) < p_time)[..., None].expand(B, L, P)          # whole time steps
    sel_p = torch.rand(B, L, P, device=x.device) < p_part                                       # single cells
    sel = (sel_t | sel_p) & valid[..., None]
    r = torch.rand(B, L, P, device=x.device); inp = x.clone()
    inp[sel & (r < 0.8)] = K
    rnd = sel & (r >= 0.8) & (r < 0.9); inp[rnd] = torch.randint(0, K, (int(rnd.sum()),), device=x.device)
    return inp, sel

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tok_dir', required=True); ap.add_argument('--train_list', required=True); ap.add_argument('--val_list', required=True); ap.add_argument('--K', type=int, default=512)
    ap.add_argument('--d', type=int, default=512); ap.add_argument('--layers', type=int, default=8); ap.add_argument('--heads', type=int, default=8); ap.add_argument('--max_len', type=int, default=256); ap.add_argument('--mask_time', type=float, default=0.15); ap.add_argument('--mask_part', type=float, default=0.15)
    ap.add_argument('--batch_size', type=int, default=128); ap.add_argument('--steps', type=int, default=100000); ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--warmup', type=int, default=2000); ap.add_argument('--val_every', type=int, default=2000); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out_dir', default='/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/MotionMMM_Pooled'); ap.add_argument('--tag', default='mmm'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = f"{time.strftime('%Y%m%d_%H%M%S')}_job{os.environ.get('SLURM_JOB_ID', 'local')}_{a.tag}"; ckdir = os.path.join(a.out_dir, run); logdir = os.path.join(_repo, 'zlog', 'MotionMMM_Pooled', run); os.makedirs(ckdir, exist_ok=True); os.makedirs(logdir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(logdir, 'config.json'), 'w'), indent=1); log = open(os.path.join(logdir, 'train.log'), 'a')
    def P(*s):
        m = ' '.join(str(x) for x in s); print(m, flush=True); log.write(m + '\n'); log.flush()
    tr = TokenDataset(a.tok_dir, a.train_list, a.max_len, True); va = TokenDataset(a.tok_dir, a.val_list, a.max_len, False)
    dl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=a.workers > 0); dv = DataLoader(va, a.batch_size, shuffle=False, num_workers=min(4, a.workers), collate_fn=collate)
    model = MMM(a.K, a.d, a.layers, a.heads, a.max_len + 8).to(dev); opt = torch.optim.AdamW(model.parameters(), a.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)); P(f'params {sum(p.numel() for p in model.parameters())/1e6:.1f}M train {len(tr)} val {len(va)}')
    step, best, t0, agg = 0, -1.0, time.time(), {}
    def run_batch(x, valid):
        x, valid = x.to(dev), valid.to(dev); inp, sel = mask_tokens(x, valid, a.K, a.mask_time, a.mask_part); logits = model(inp, valid)
        loss = F.cross_entropy(logits[sel], x[sel])  # sel is (B,L,4): logits[sel] -> (n_cells, K), x[sel] -> (n_cells,)
        acc = {}
        for i, p in enumerate(PARTS):
            m = sel[..., i]; acc[p] = (logits[..., i, :][m].argmax(-1) == x[..., i][m]).float().mean().item() if m.any() else 0.0
        return loss, acc
    model.train()
    while step < a.steps:
        for x, valid in dl:
            for g in opt.param_groups: g['lr'] = a.lr * sched(step)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')): loss, acc = run_batch(x, valid)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            agg['loss'] = agg.get('loss', 0) + loss.item()
            for p in PARTS: agg[p] = agg.get(p, 0) + acc[p]
            if step % 100 == 0: P(f"step {step} lr {opt.param_groups[0]['lr']:.2e} " + ' '.join(f'{k} {v/100:.4f}' for k, v in agg.items()) + f' | {(time.time()-t0)/step:.2f}s/it'); agg = {}
            if step % a.val_every == 0 or step == a.steps:
                model.eval(); vs = {}; n = 0
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                    for x, valid in dv:
                        loss, acc = run_batch(x, valid); n += 1; vs['loss'] = vs.get('loss', 0) + loss.item()
                        for p in PARTS: vs[p] = vs.get(p, 0) + acc[p]
                vs = {k: v / n for k, v in vs.items()}; score = sum(vs[p] for p in PARTS) / 4; P(f'VAL step {step} masked-acc {score:.4f} ' + ' '.join(f'{k} {v:.4f}' for k, v in vs.items()))
                ck = {'model': model.state_dict(), 'config': vars(a), 'step': step, 'val': vs}; torch.save(ck, os.path.join(ckdir, 'newest_model.pt'))
                if score > best: best = score; torch.save(ck, os.path.join(ckdir, 'best_model.pt')); P(f'  -> new best {best:.4f}')
                model.train()
            if step >= a.steps: break
    P('DONE', ckdir)

if __name__ == '__main__': main()
