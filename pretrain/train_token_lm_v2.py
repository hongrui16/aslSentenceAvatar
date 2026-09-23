"""4.4 v2 (2026-09-22): text -> motion-token LM conditioned on the 4.2 aligner's TEXT side (v8: gloss units, anchors = all SignBank
+ ASL Citizen train). v1 (train_token_lm.py, untouched) conditions on a t5-base that learns from captions only and collapsed on
every data group. Changes here, all behind --text_mode align:
  * text units = pseudo-gloss sequence of the caption (data_records/alignment/gloss, md5 lookup as in align_dataset_v2);
  * memory = one 512-d vector per gloss from the aligner text side (AlignModel.embed_chunks: t5-base -> proj -> L2 norm),
    FROZEN (--text_lr 0) or slowly tuned; the decoder cross-attends to these gloss vectors instead of t5 subword states.
  * --max_items for smoke tests.
Decoder / loss / generation are identical to v1 (8-layer causal transformer over the 4 VQ streams, EOS per part, teacher forcing).
  python pretrain/train_token_lm_v2.py --train_list <subset.txt> --text_mode align --align_ckpt <v8 newest_model.pt> --tag lm2_<group>
"""
import argparse, hashlib, importlib, json, math, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from pretrain.train_token_lm import TextTokenDataset, collate, TokenLM, PARTS, SUB
from align.align_dataset_v2 import load_chunks
GLOSS_DIR = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/alignment/gloss'


class TokenLMv2(TokenLM):
    def __init__(self, K=512, d=512, layers=8, heads=8, max_len=256, drop=0.1, text_encoder='t5-base', text_mode='align',
                 align_ckpt=None, align_module='align.train_align_v6', gloss_dir=GLOSS_DIR, max_gloss=30):
        super().__init__(K, d, layers, heads, max_len, drop, text_encoder)
        self.text_mode, self.max_gloss = text_mode, max_gloss
        if text_mode == 'align':
            del self.txt, self.txt_proj  # the t5 of v1 is not used
            ck = torch.load(align_ckpt, map_location='cpu'); c = ck['config']
            AlignModel = importlib.import_module(align_module).AlignModel
            am = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']); am.load_state_dict(ck['model'])
            self.txt, self.txt_proj = am.text, am.proj_t          # aligner text side (t5-base + projection), 512-d units
            del am
            self.mem_proj = nn.Linear(c['e'], d) if c['e'] != d else nn.Identity()
            self.h2g = load_chunks(gloss_dir); self.h2g.pop('__gloss__', None)
    def glosses(self, text):
        g = self.h2g.get(hashlib.md5(text.strip().encode()).hexdigest())
        return (g or [text.strip()])[:self.max_gloss]
    def encode_text(self, texts, dev):
        if self.text_mode != 'align': return super().encode_text(texts, dev)
        gl = [self.glosses(t) for t in texts]; flat = [u for g in gl for u in g]
        z = F.normalize(self.txt_proj(self.txt(flat, dev)), dim=-1)     # (sum n_i, e)
        n = max(len(g) for g in gl); mem = z.new_zeros(len(texts), n, z.shape[-1]); pad = torch.ones(len(texts), n, dtype=torch.bool, device=dev); k = 0
        for i, g in enumerate(gl): mem[i, :len(g)] = z[k:k + len(g)]; pad[i, :len(g)] = False; k += len(g)
        return self.mem_proj(mem), pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tok_dir', default='/projects/kosecka/hongrui/dataset/smplx_fits/pooled_tokens/vq_K512_w64')
    ap.add_argument('--train_list', required=True); ap.add_argument('--val_list', default=f'{SUB}/pool_val.txt')
    ap.add_argument('--K', type=int, default=512); ap.add_argument('--d', type=int, default=512)
    ap.add_argument('--layers', type=int, default=8); ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--max_len', type=int, default=192); ap.add_argument('--text_encoder', default='t5-base')
    ap.add_argument('--text_mode', default='align', choices=['align', 't5']); ap.add_argument('--align_ckpt', default='')
    ap.add_argument('--align_module', default='align.train_align_v6'); ap.add_argument('--gloss_dir', default=GLOSS_DIR)
    ap.add_argument('--batch_size', type=int, default=64); ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--lr_text', type=float, default=0.0)
    ap.add_argument('--warmup', type=int, default=2000); ap.add_argument('--val_every', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=8); ap.add_argument('--max_items', type=int, default=0)
    ap.add_argument('--out_dir', default='/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/TokenLM_Pooled')
    ap.add_argument('--tag', default='lm2'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = f"{time.strftime('%Y%m%d_%H%M%S')}_job{os.environ.get('SLURM_JOB_ID', 'local')}_{a.tag}"
    ckdir = os.path.join(a.out_dir, run); logdir = os.path.join(_repo, 'zlog', 'TokenLM_Pooled', run)
    os.makedirs(ckdir, exist_ok=True); os.makedirs(logdir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(logdir, 'config.json'), 'w'), indent=1); log = open(os.path.join(logdir, 'train.log'), 'a')
    def P(*s):
        m = ' '.join(str(x) for x in s); print(m, flush=True); log.write(m + '\n'); log.flush()
    tr = TextTokenDataset(a.tok_dir, a.train_list, max_len=a.max_len, train=True)
    va = TextTokenDataset(a.tok_dir, a.val_list, max_len=a.max_len, train=False)
    if a.max_items: tr.items = tr.items[:a.max_items]; va.items = va.items[:max(64, a.max_items // 10)]
    dl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=a.workers > 0)
    dv = DataLoader(va, a.batch_size, shuffle=False, num_workers=min(4, a.workers), collate_fn=collate)
    model = TokenLMv2(a.K, a.d, a.layers, a.heads, a.max_len + 8, text_encoder=a.text_encoder, text_mode=a.text_mode,
                      align_ckpt=a.align_ckpt or None, align_module=a.align_module, gloss_dir=a.gloss_dir).to(dev)
    text_params = list(model.txt.parameters()) + (list(model.txt_proj.parameters()) if a.text_mode == 'align' else [])
    if a.lr_text <= 0:
        for p in text_params: p.requires_grad_(False)
    tid = {id(p) for p in text_params}
    groups = [{'params': [p for p in model.parameters() if id(p) not in tid], 'lr': a.lr}]
    if a.lr_text > 0: groups.append({'params': text_params, 'lr': a.lr_text})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=0.01); bases = [a.lr] + ([a.lr_text] if a.lr_text > 0 else [])
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    n_gl = sum(1 for _, t in tr.items[:2000] if model.text_mode == 'align' and model.h2g.get(hashlib.md5(t.strip().encode()).hexdigest()))
    P(f'params {sum(p.numel() for p in model.parameters())/1e6:.1f}M (trainable {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M) '
      f'train {len(tr)} val {len(va)} text_mode {a.text_mode} lr_text {a.lr_text} gloss-hit(first 2000) {n_gl}')
    step, best, t0, agg = 0, math.inf, time.time(), {}
    model.train()
    if a.lr_text <= 0: model.txt.eval()
    while step < a.steps:
        for tgt, val, texts in dl:
            for g, base in zip(opt.param_groups, bases): g['lr'] = base * sched(step)
            tgt, val = tgt.to(dev), val.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                loss, acc = model(tgt, val, texts, dev)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); step += 1
            agg['loss'] = agg.get('loss', 0) + loss.item(); agg['acc'] = agg.get('acc', 0) + acc
            if step % 100 == 0:
                P(f"step {step} lr {opt.param_groups[0]['lr']:.2e} loss {agg['loss']/100:.4f} acc {agg['acc']/100:.4f} | {(time.time()-t0)/step:.2f}s/it"); agg = {}
            if step % a.val_every == 0 or step == a.steps:
                model.eval(); vl, vc, n = 0.0, 0.0, 0
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                    for tgt, val, texts in dv:
                        l, c = model(tgt.to(dev), val.to(dev), texts, dev); vl += l.item(); vc += c; n += 1
                vl, vc = vl / n, vc / n; P(f'VAL step {step} loss {vl:.4f} acc {vc:.4f}')
                ck = {'model': model.state_dict(), 'config': vars(a), 'step': step, 'val': {'loss': vl, 'acc': vc}}
                torch.save(ck, os.path.join(ckdir, 'newest_model.pt'))
                if vl < best: best = vl; torch.save(ck, os.path.join(ckdir, 'best_model.pt')); P(f'  -> new best loss {best:.4f}')
                model.train()
                if a.lr_text <= 0: model.txt.eval()
            if step >= a.steps: break
    P('DONE', ckdir)


if __name__ == '__main__': main()
