"""Roadmap 4.4 (line 2 step 3): end-to-end text -> motion-token LM (SOKE-style), the upward-scaling control.

Text side: T5-base encoder (trainable, low lr, same choice as 4.2). Motion side: causal transformer
decoder over the 4 part-token streams from the 4.1 VQ tokenizer (6.25 Hz), cross-attending to the text
memory; one step predicts the 4 part tokens of the next time position. Vocab per part = K real tokens
+ EOS (id K); BOS/pad = id K+1 (embedding only, never predicted). Trained per data GROUP
(curated_100h / random_matched_100h / pool_full ...) with EQUAL optimizer steps across groups; same
diagnostics afterwards (tools/verify_token_lm_pooled.py, 4.4-3).

  python pretrain/train_token_lm.py --train_list <subset.txt> --steps 60000 --tag lm_<group>
"""
import argparse, csv, json, math, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
PARTS = ['body', 'lhand', 'rhand', 'face']
INDEX = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/curation/clip_index.tsv'
SUB = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/curation/subsets'


class TextTokenDataset(Dataset):
    def __init__(self, tok_dir, list_txt, index_tsv=INDEX, max_len=192, train=True):
        ids = [l.strip() for l in open(list_txt) if l.strip()]; wset = set(ids)
        text = {}
        with open(index_tsv) as f:
            for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
                if r['clip_id'] in wset: text[r['clip_id']] = r['text'].strip()
        self.items = []
        for c in ids:
            fp = os.path.join(tok_dir, c.replace(':', '__') + '.npz')
            if c in text and text[c] and os.path.exists(fp): self.items.append((fp, text[c]))
        self.max_len, self.train = max_len, train
        print(f'[TextTokenDataset] {len(self.items)} clips from {list_txt}', flush=True)
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        fp, txt = self.items[i]
        d = np.load(fp); toks = np.stack([d[p].astype(np.int64) for p in PARTS], 1)  # (n, 4)
        if len(toks) > self.max_len - 1: toks = toks[:self.max_len - 1]  # -1 leaves room for EOS
        return torch.from_numpy(toks), txt


def collate(batch):
    L = max(len(b[0]) for b in batch) + 1  # + EOS
    tgt = torch.full((len(batch), L, 4), -100, dtype=torch.long)
    val = torch.zeros(len(batch), L, dtype=torch.bool)
    for i, (t, _) in enumerate(batch):
        tgt[i, :len(t)] = t; tgt[i, len(t)] = -1  # -1 placeholder for EOS, filled with K in model
        val[i, :len(t) + 1] = True
    return tgt, val, [b[1] for b in batch]


class TokenLM(nn.Module):
    def __init__(self, K=512, d=512, layers=8, heads=8, max_len=256, drop=0.1, text_encoder='t5-base'):
        super().__init__(); self.K = K; self.eos, self.bos = K, K + 1
        from transformers import AutoTokenizer, T5EncoderModel
        self.tk = AutoTokenizer.from_pretrained(text_encoder); self.txt = T5EncoderModel.from_pretrained(text_encoder)
        self.txt_proj = nn.Linear(self.txt.config.d_model, d)
        self.emb = nn.ModuleList([nn.Embedding(K + 2, d) for _ in PARTS])
        self.part_proj = nn.Linear(4 * d, d); self.pos = nn.Embedding(max_len, d)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True), layers)
        self.norm = nn.LayerNorm(d); self.heads = nn.ModuleList([nn.Linear(d, K + 1) for _ in PARTS])  # K real + EOS
    def encode_text(self, texts, dev):
        b = self.tk(texts, return_tensors='pt', padding=True, truncation=True, max_length=64).to(dev)
        return self.txt_proj(self.txt(**b).last_hidden_state), ~b['attention_mask'].bool()
    def decode(self, inp, mem, mem_pad):  # inp (B, L, 4) token ids (bos/pad = K+1)
        L = inp.shape[1]
        h = self.part_proj(torch.cat([e(inp[..., i]) for i, e in enumerate(self.emb)], -1)) + self.pos(torch.arange(L, device=inp.device))[None]
        cm = nn.Transformer.generate_square_subsequent_mask(L, device=inp.device)
        h = self.norm(self.dec(h, mem, tgt_mask=cm, memory_key_padding_mask=mem_pad))
        return torch.stack([hd(h) for hd in self.heads], 2)  # (B, L, 4, K+1)
    def forward(self, tgt, val, texts, dev):
        tgt = tgt.clone(); tgt[tgt == -1] = self.eos
        inp = torch.cat([torch.full((tgt.shape[0], 1, 4), self.bos, dtype=torch.long, device=tgt.device), tgt[:, :-1]], 1)
        inp[inp < 0] = self.bos  # pad positions
        mem, mem_pad = self.encode_text(texts, dev)
        logits = self.decode(inp, mem, mem_pad)
        loss = F.cross_entropy(logits[val].reshape(-1, self.K + 1), tgt[val].reshape(-1))
        acc = (logits[val].argmax(-1) == tgt[val]).float().mean().item()
        return loss, acc
    @torch.no_grad()
    def generate(self, texts, dev, max_new=192, temperature=0.0):
        mem, mem_pad = self.encode_text(texts, dev); B = len(texts)
        seq = torch.full((B, 1, 4), self.bos, dtype=torch.long, device=dev)
        done = torch.zeros(B, dtype=torch.bool, device=dev); outs = []
        for _ in range(max_new):
            logits = self.decode(seq, mem, mem_pad)[:, -1]  # (B, 4, K+1)
            if temperature > 0:
                nxt = torch.multinomial(F.softmax(logits.reshape(B * 4, -1) / temperature, -1), 1).reshape(B, 4)
            else:
                nxt = logits.argmax(-1)
            done |= (nxt == self.eos).any(-1)
            outs.append(nxt.clone()); nxt = nxt.clamp(max=self.K - 1)
            seq = torch.cat([seq, nxt[:, None]], 1)
            if done.all(): break
        out = torch.stack(outs, 1)  # (B, T, 4) with possible EOS ids
        toks = []
        for b in range(B):
            e = (out[b] == self.eos).any(-1).nonzero()
            t = out[b][:int(e[0])] if len(e) else out[b]
            toks.append(t.clamp(max=self.K - 1).cpu())
        return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tok_dir', default='/projects/kosecka/hongrui/dataset/smplx_fits/pooled_tokens/vq_K512_w64')
    ap.add_argument('--train_list', required=True); ap.add_argument('--val_list', default=f'{SUB}/pool_val.txt')
    ap.add_argument('--K', type=int, default=512); ap.add_argument('--d', type=int, default=512)
    ap.add_argument('--layers', type=int, default=8); ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--max_len', type=int, default=192); ap.add_argument('--text_encoder', default='t5-base')
    ap.add_argument('--batch_size', type=int, default=64); ap.add_argument('--steps', type=int, default=60000)
    ap.add_argument('--lr', type=float, default=3e-4); ap.add_argument('--lr_text', type=float, default=2e-5)
    ap.add_argument('--warmup', type=int, default=2000); ap.add_argument('--val_every', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out_dir', default='/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/TokenLM_Pooled')
    ap.add_argument('--tag', default='lm'); ap.add_argument('--seed', type=int, default=0)
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
    dl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate, drop_last=True, persistent_workers=a.workers > 0)
    dv = DataLoader(va, a.batch_size, shuffle=False, num_workers=min(4, a.workers), collate_fn=collate)
    model = TokenLM(a.K, a.d, a.layers, a.heads, a.max_len + 8, text_encoder=a.text_encoder).to(dev)
    groups = [{'params': [p for n, p in model.named_parameters() if not n.startswith('txt.')], 'lr': a.lr},
              {'params': model.txt.parameters(), 'lr': a.lr_text}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=0.01)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    P(f'params {sum(p.numel() for p in model.parameters())/1e6:.1f}M train {len(tr)} val {len(va)}')
    step, best, t0, agg = 0, math.inf, time.time(), {}
    model.train()
    while step < a.steps:
        for tgt, val, texts in dl:
            for g, base in zip(opt.param_groups, [a.lr, a.lr_text]): g['lr'] = base * sched(step)
            tgt, val = tgt.to(dev), val.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                loss, acc = model(tgt, val, texts, dev)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
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
            if step >= a.steps: break
    P('DONE', ckdir)


if __name__ == '__main__': main()
