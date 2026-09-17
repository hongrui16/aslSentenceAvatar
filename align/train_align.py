"""Roadmap 4.2: segment-level text-motion contrastive alignment (MIL-NCE, BOBSL-spotting style).

Motion side: the 4.1 MMM encoder (init from --mmm_ckpt) over the clip's VQ token streams; a segment embedding =
mean of encoder states over the segment's token span -> projection -> L2 norm. Text side: pluggable encoder
(--text_encoder t5-base now; a Qwen embedding path can be added later) over phrase-chunk strings -> mean-pooled
states -> projection -> L2 norm. Loss: symmetric MIL-NCE with within-clip positives and cross-clip negatives;
correspondence inside a clip is unknown by design (weak supervision). Val metric: clip-level retrieval R@1/R@10
(mean chunk embedding vs mean segment embedding over the val set) — the direct preview of the 4.2 deliverable.

  python align/train_align.py --tok_dir <dir> --chunk_dir <dir> --mmm_ckpt <best_model.pt> --steps 60000
"""
import argparse, json, math, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from align.align_dataset import ClipAlignDataset, collate
from pretrain.train_mmm import MMM

SUB = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/curation/subsets'


class TextEncoderT5(nn.Module):
    def __init__(self, name='t5-base'):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel
        self.tk = AutoTokenizer.from_pretrained(name); self.enc = T5EncoderModel.from_pretrained(name); self.dim = self.enc.config.d_model
    def forward(self, texts, dev):
        b = self.tk(texts, return_tensors='pt', padding=True, truncation=True, max_length=32).to(dev)
        h = self.enc(**b).last_hidden_state; m = b['attention_mask'][..., None].float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1)


class TextEncoderQwen(nn.Module):
    """Frozen causal-LM backbone (e.g. qwen:Qwen/Qwen2.5-1.5B-Instruct), masked mean-pool of the last
    hidden state; only the outer projection trains. bf16 weights to fit MIG slices."""
    def __init__(self, name):
        super().__init__()
        from transformers import AutoTokenizer, AutoModel
        self.tk = AutoTokenizer.from_pretrained(name); self.enc = AutoModel.from_pretrained(name, dtype=torch.bfloat16)
        self.dim = self.enc.config.hidden_size
        for p in self.enc.parameters(): p.requires_grad = False
        self.enc.eval()
    def train(self, mode=True):
        super().train(mode); self.enc.eval(); return self
    def forward(self, texts, dev):
        b = self.tk(texts, return_tensors='pt', padding=True, truncation=True, max_length=32).to(dev)
        with torch.no_grad():
            h = self.enc(**b).last_hidden_state
        m = b['attention_mask'][..., None].float()
        return ((h.float() * m).sum(1) / m.sum(1).clamp(min=1))


class AlignModel(nn.Module):
    def __init__(self, mmm_ckpt, text_encoder='t5-base', e=512):
        super().__init__()
        ck = torch.load(mmm_ckpt, map_location='cpu'); c = ck['config']
        self.mmm = MMM(c['K'], c['d'], c['layers'], c['heads'], c['max_len'] + 8); self.mmm.load_state_dict(ck['model'])
        self.mmm.heads = None  # MLM heads not needed
        self.text = TextEncoderQwen(text_encoder[5:]) if text_encoder.startswith('qwen:') else TextEncoderT5(text_encoder)
        self.proj_m = nn.Linear(c['d'], e); self.proj_t = nn.Linear(self.text.dim, e)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
    def embed_segments(self, x, pad_mask, spans):
        h = self.mmm.encode(x, pad_mask)  # (B, L, d)
        seg = torch.stack([h[i, a:b].mean(0) for i, a, b in spans])
        return F.normalize(self.proj_m(seg), dim=-1)
    def embed_chunks(self, texts, dev):
        return F.normalize(self.proj_t(self.text(texts, dev)), dim=-1)


def milnce(zt, zm, chk_own, seg_own, scale):
    """Symmetric MIL-NCE. zt (Nt,e) chunks, zm (Nm,e) segments; own = clip index per row."""
    sim = zt @ zm.T * scale                      # (Nt, Nm)
    pos = (chk_own[:, None] == seg_own[None, :])  # within-clip candidate pairs
    clips = chk_own.unique(); losses = []
    lse_all_t = torch.logsumexp(sim, dim=1); lse_all_m = torch.logsumexp(sim, dim=0)
    neg_inf = torch.finfo(sim.dtype).min
    for c in clips:
        ti = chk_own == c; mi = seg_own == c
        p = sim[ti][:, mi].reshape(-1)
        num = torch.logsumexp(p, 0)
        den = torch.logsumexp(torch.cat([sim[ti].reshape(-1), sim[:, mi].reshape(-1), p]), 0)  # pairs touching clip c (p double-counted in both directions consistently with MIL-NCE denom union)
        losses.append(den - num)
    return torch.stack(losses).mean(), pos, sim


@torch.no_grad()
def val_retrieval(model, dv, dev):
    zt, zm, loss_sum, n = [], [], 0.0, 0
    for x, m, spans, seg_own, chunks, chk_own in dv:
        x, m, seg_own, chk_own = x.to(dev), m.to(dev), seg_own.to(dev), chk_own.to(dev)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            zs = model.embed_segments(x, m, spans); zc = model.embed_chunks(chunks, dev)
            l, _, _ = milnce(zc, zs, chk_own, seg_own, model.logit_scale.exp().clamp(max=100))
        loss_sum += l.item(); n += 1
        B = int(m.shape[0])
        zt.append(torch.stack([zc[chk_own == i].mean(0) for i in range(B)]).float())
        zm.append(torch.stack([zs[seg_own == i].mean(0) for i in range(B)]).float())
    zt = F.normalize(torch.cat(zt), dim=-1); zm = F.normalize(torch.cat(zm), dim=-1)
    sim = zt @ zm.T; rank = (sim > sim.diag()[:, None]).sum(1)
    return {'loss': loss_sum / max(n, 1), 'R1': (rank == 0).float().mean().item(), 'R10': (rank < 10).float().mean().item(), 'medR': rank.median().item() + 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tok_dir', default='/projects/kosecka/hongrui/dataset/smplx_fits/pooled_tokens/vq_K512_w64')
    ap.add_argument('--chunk_dir', default='/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/alignment/chunks')
    ap.add_argument('--mmm_ckpt', required=True); ap.add_argument('--text_encoder', default='t5-base')
    ap.add_argument('--train_list', default=f'{SUB}/pretrain_all.txt'); ap.add_argument('--val_list', default=f'{SUB}/pool_val.txt')
    ap.add_argument('--e', type=int, default=512); ap.add_argument('--batch_size', type=int, default=48); ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--lr_text', type=float, default=2e-5); ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--val_every', type=int, default=2000); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--max_items', type=int, default=0)
    ap.add_argument('--out_dir', default='/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/Align_Pooled'); ap.add_argument('--tag', default='align'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = f"{time.strftime('%Y%m%d_%H%M%S')}_job{os.environ.get('SLURM_JOB_ID', 'local')}_{a.tag}"; ckdir = os.path.join(a.out_dir, run); logdir = os.path.join(_repo, 'zlog', 'Align_Pooled', run)
    os.makedirs(ckdir, exist_ok=True); os.makedirs(logdir, exist_ok=True); json.dump(vars(a), open(os.path.join(logdir, 'config.json'), 'w'), indent=1)
    log = open(os.path.join(logdir, 'train.log'), 'a')
    def P(*s):
        m = ' '.join(str(x) for x in s); print(m, flush=True); log.write(m + '\n'); log.flush()
    mi = a.max_items or None
    tr = ClipAlignDataset(a.tok_dir, a.train_list, a.chunk_dir, train=True, max_items=mi)
    va = ClipAlignDataset(a.tok_dir, a.val_list, a.chunk_dir, train=False, max_items=min(mi or 2000, 2000))
    dl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=a.workers > 0)
    dv = DataLoader(va, a.batch_size, shuffle=False, num_workers=min(4, a.workers), collate_fn=collate)
    model = AlignModel(a.mmm_ckpt, a.text_encoder, a.e).to(dev)
    groups = [{'params': [p for n, p in model.named_parameters() if not n.startswith('text.')], 'lr': a.lr},
              {'params': model.text.parameters(), 'lr': a.lr_text}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=0.01)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    P(f'params {sum(p.numel() for p in model.parameters())/1e6:.1f}M train {len(tr)} val {len(va)}')
    step, best, t0, agg = 0, -1.0, time.time(), {}
    model.train()
    while step < a.steps:
        for x, m, spans, seg_own, chunks, chk_own in dl:
            for g, base in zip(opt.param_groups, [a.lr, a.lr_text]): g['lr'] = base * sched(step)
            x, m, seg_own, chk_own = x.to(dev), m.to(dev), seg_own.to(dev), chk_own.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                zs = model.embed_segments(x, m, spans); zc = model.embed_chunks(chunks, dev)
                loss, _, _ = milnce(zc, zs, chk_own, seg_own, model.logit_scale.exp().clamp(max=100))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            agg['loss'] = agg.get('loss', 0) + loss.item()
            if step % 100 == 0: P(f"step {step} lr {opt.param_groups[0]['lr']:.2e} loss {agg['loss']/100:.4f} scale {model.logit_scale.exp().item():.1f} | {(time.time()-t0)/step:.2f}s/it"); agg = {}
            if step % a.val_every == 0 or step == a.steps:
                model.eval(); vs = val_retrieval(model, dv, dev); model.train()
                P(f"VAL step {step} loss {vs['loss']:.4f} R1 {vs['R1']:.4f} R10 {vs['R10']:.4f} medR {vs['medR']:.0f}")
                ck = {'model': model.state_dict(), 'config': vars(a), 'step': step, 'val': vs}; torch.save(ck, os.path.join(ckdir, 'newest_model.pt'))
                if vs['R1'] > best: best = vs['R1']; torch.save(ck, os.path.join(ckdir, 'best_model.pt')); P(f'  -> new best R1 {best:.4f}')
            if step >= a.steps: break
    P('DONE', ckdir)


if __name__ == '__main__': main()
