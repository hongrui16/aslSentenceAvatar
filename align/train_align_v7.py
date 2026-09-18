"""Aligner v4 = v1 MIL-NCE + phrase-level supervision levers (track A of the 2026-09-16 decision).

Fine-tunes from an existing AlignModel checkpoint (--init_ckpt) with the same data/collate as train_align.py and adds:
  --anchor_w  SignBank gloss anchor: each step samples --anchor_bs citation signs (train split of
              signbank_split.json, unique gloss texts), embeds the whole sign (motion side) and its gloss text
              (text side) and applies a symmetric InfoNCE. Gives the text side a direct gloss-level target.
  --mono_w    within-clip monotone soft alignment: forward-backward posteriors over the chunk x segment
              similarity (monotone many-to-many path, detached) become soft targets for a cross-entropy whose
              denominator is every segment (resp. chunk) in the batch. Forces chunk<->segment correspondence
              instead of clip-level topical similarity.
VAL adds sbR1 = held-out SignBank gloss probe R@1 (motion->text), the phrase-level signal we care about.

  python align/train_align_v4.py --init_ckpt <v1 best> --anchor_w 0.5 --mono_w 1.0 --steps 30000
"""
import argparse, json, math, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from align.align_dataset_v2 import ClipAlignDataset, collate, PARTS as PARTS_
from align.train_align_v6 import AlignModel, milnce, val_retrieval, SUB

DR = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/alignment'
NEG = -1e9


class SignBank:
    """Tokenized citation signs + gloss texts (align/tokenize_signbank.py) with the train/held-out split."""
    def __init__(self, npz, split_json, max_len):
        d = np.load(npz); self.stems = d['stems'].tolist(); self.texts = d['texts'].tolist(); self.tok = d['tokens']; self.off = d['offsets']
        sp = json.load(open(split_json)); tr, ho = set(sp['train_stems']), set(sp['heldout_stems'])
        self.train = [i for i, s in enumerate(self.stems) if s in tr]; self.held = [i for i, s in enumerate(self.stems) if s in ho]
        self.max_len = max_len
        by_text = {}
        for i in self.train: by_text.setdefault(self.texts[i], []).append(i)
        self.train_groups = list(by_text.values())
    def tokens(self, i): return torch.from_numpy(self.tok[self.off[i]:self.off[i + 1]].astype(np.int64))[:self.max_len]
    def batch(self, idx):
        ts = [self.tokens(i) for i in idx]; L = max(len(t) for t in ts)
        x = torch.zeros(len(ts), L, 4, dtype=torch.long); m = torch.zeros(len(ts), L, dtype=torch.bool); spans = []
        for j, t in enumerate(ts): x[j, :len(t)] = t; m[j, :len(t)] = True; spans.append((j, 0, len(t)))
        return x, m, spans, [self.texts[i] for i in idx]
    def sample(self, n, rng):
        groups = rng.sample(self.train_groups, min(n, len(self.train_groups)))
        return [rng.choice(g) for g in groups]  # one variant per gloss text -> unique texts in the batch


def anchor_loss(model, sb, n, rng, dev, scale):
    idx = sb.sample(n, rng); x, m, spans, texts = sb.batch(idx)
    zm = model.embed_segments(x.to(dev), m.to(dev), spans); zt = model.embed_chunks(texts, dev)
    logits = (zt @ zm.T).float() * scale; tgt = torch.arange(len(idx), device=dev)
    return 0.5 * (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.T, tgt))


@torch.no_grad()
def signbank_probe(model, sb, dev, bs=128):
    idx = sb.held; zm, zt = [], []
    for i in range(0, len(idx), bs):
        x, m, spans, texts = sb.batch(idx[i:i + bs])
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            zm.append(model.embed_segments(x.to(dev), m.to(dev), spans).float()); zt.append(model.embed_chunks(texts, dev).float())
    zm = F.normalize(torch.cat(zm), dim=-1); zt = F.normalize(torch.cat(zt), dim=-1)
    # same gloss text can appear with several variants: rank against unique texts, positive = own text
    texts = [sb.texts[i] for i in idx]; uniq = sorted(set(texts)); ti = {t: k for k, t in enumerate(uniq)}
    zt_u = torch.stack([zt[texts.index(t)] for t in uniq]); own = torch.tensor([ti[t] for t in texts], device=dev)
    sim = zm @ zt_u.T; rank = (sim > sim.gather(1, own[:, None])).sum(1)
    return {'sbR1': (rank == 0).float().mean().item(), 'sbR10': (rank < 10).float().mean().item(), 'sbMedR': rank.median().item() + 1, 'sbN': len(uniq)}


def fb_posterior(s):
    """s: (C,S) numpy float32 scaled similarities. Returns P(i,j) = prob. that a monotone path (steps
    (1,1),(1,0),(0,1) from (0,0) to (C-1,S-1)) passes through cell (i,j), computed by forward-backward
    over anti-diagonals in log space."""
    C, S = s.shape
    A = np.full((C + 1, S + 1), NEG, np.float64); A[1, 1] = s[0, 0]
    for d in range(1, C + S - 1):
        i = np.arange(max(0, d - S + 1), min(C - 1, d) + 1); j = d - i
        prev = np.stack([A[i, j], A[i, j + 1], A[i + 1, j]])  # alpha[i-1,j-1], alpha[i-1,j], alpha[i,j-1]
        A[i + 1, j + 1] = s[i, j] + np.logaddexp.reduce(prev, axis=0)
    sp = np.full((C + 1, S + 1), NEG, np.float64); sp[:C, :S] = s
    B = np.full((C + 1, S + 1), NEG, np.float64); B[C - 1, S - 1] = 0.0
    for d in range(C + S - 3, -1, -1):
        i = np.arange(max(0, d - S + 1), min(C - 1, d) + 1); j = d - i
        nxt = np.stack([sp[i + 1, j + 1] + B[i + 1, j + 1], sp[i + 1, j] + B[i + 1, j], sp[i, j + 1] + B[i, j + 1]])
        B[i, j] = np.logaddexp.reduce(nxt, axis=0)
    logZ = A[C, S]
    post = np.exp(np.clip(A[1:, 1:] + B[:C, :S] - logZ, -50, 0))
    return post.astype(np.float32)


def mono_loss(sim, chk_own, seg_own, scale):
    """sim (Nt,Nm) unscaled cosines. Soft monotone targets per clip; CE against all segments / chunks in the batch."""
    logp_t = F.log_softmax(sim * scale, dim=1); logp_m = F.log_softmax(sim * scale, dim=0)
    s_np = (sim.detach().float() * float(scale)).cpu().numpy(); ct, cm, n = 0.0, 0.0, 0
    co = chk_own.cpu().numpy(); so = seg_own.cpu().numpy()
    for c in np.unique(co):
        ti = np.where(co == c)[0]; mi = np.where(so == c)[0]
        if len(ti) < 2 or len(mi) < 2: continue
        P = fb_posterior(s_np[np.ix_(ti, mi)])
        Pt = torch.from_numpy(P / np.maximum(P.sum(1, keepdims=True), 1e-6)).to(sim.device)
        Pm = torch.from_numpy(P / np.maximum(P.sum(0, keepdims=True), 1e-6)).to(sim.device)
        ti_t = torch.from_numpy(ti).to(sim.device); mi_t = torch.from_numpy(mi).to(sim.device)
        ct = ct - (Pt * logp_t[ti_t][:, mi_t]).sum() / len(ti)
        cm = cm - (Pm * logp_m[ti_t][:, mi_t]).sum() / len(mi)
        n += 1
    if n == 0: return sim.new_zeros(())
    return 0.5 * (ct + cm) / n


# ---------------- round 2: citation-NN sign spots as hard chunk<->segment pairs ----------------
import re as _re
_norm = lambda s: _re.sub(r"[^a-z0-9' ]+", ' ', s.lower()).split()
def _lemmas(words):
    out = set(words)
    for x in words:
        for suf in ('s', 'es', 'ed', 'ing', 'd'):
            if x.endswith(suf) and len(x) - len(suf) >= 3: out.add(x[:-len(suf)])
        if x.endswith('ies'): out.add(x[:-3] + 'y')
    return out


class SpotAlignDataset(ClipAlignDataset):
    """ClipAlignDataset + per-clip list of (chunk_idx, seg_idx) hard pairs from align/spot_signs.py (tsv: clip_id fr_a fr_b gloss cos margin)."""
    def __init__(self, *a, spots_tsv=None, min_cos=0.4, **k):
        super().__init__(*a, **k); self.pairs = [[] for _ in self.items]; n_pairs = 0
        if spots_tsv:
            by_clip = {}
            with open(spots_tsv) as f:
                next(f)
                for l in f:
                    cid, fa, fb, g, cos, mar = l.rstrip('\n').split('\t')
                    if float(cos) >= min_cos: by_clip.setdefault(cid, []).append((int(fa), int(fb), g))
            for i, (tok, ch, segs) in enumerate(self.items):
                cid = os.path.basename(tok)[:-4]; sp = by_clip.get(cid.replace('__', ':')) or by_clip.get(cid)
                if not sp: continue
                seg_idx = {tuple(sg): j for j, sg in enumerate(segs)}; ch_words = [_lemmas(set(_norm(c))) for c in ch]
                for fa, fb, g in sp:
                    j = seg_idx.get((fa, fb));
                    if j is None: continue
                    gw = [w for w in _norm(g)]
                    for ci, cw in enumerate(ch_words):
                        if all(w in cw for w in gw): self.pairs[i].append((ci, j)); n_pairs += 1
        print(f'[SpotAlignDataset] {n_pairs} pairs over {sum(1 for p in self.pairs if p)} clips', flush=True)
    def __getitem__(self, i):
        tok, chunks, segs = self.items[i]
        d = np.load(tok); x = np.stack([d[p].astype(np.int64) for p in PARTS_], 1)
        spans = [(a // 4, max(a // 4 + 1, -(-b // 4))) for a, b in segs]; n = len(x); keep = list(range(len(spans)))
        if n > self.max_tok:
            s0 = random.randint(0, n - self.max_tok) if self.train else (n - self.max_tok) // 2
            x = x[s0:s0 + self.max_tok]
            keep = [j for j, (a, b) in enumerate(spans) if a >= s0 and a < s0 + self.max_tok]
            spans = [(spans[j][0] - s0, min(spans[j][1] - s0, self.max_tok)) for j in keep]
            if not spans: spans = [(0, len(x))]; keep = []
        spans = [(a, min(b, len(x))) for a, b in spans]; remap = {j: k for k, j in enumerate(keep)}
        pairs = [(ci, remap[j]) for ci, j in self.pairs[i] if j in remap]
        return torch.from_numpy(x), spans, chunks, pairs


def collate_spot(batch):
    x, m, spans, seg_own, chunks, chk_own = collate([b[:3] for b in batch])
    pairs, coff, soff = [], 0, 0
    for _, sp, ch, pr in batch:
        pairs += [(coff + ci, soff + sj) for ci, sj in pr]; coff += len(ch); soff += len(sp)
    return x, m, spans, seg_own, chunks, chk_own, pairs


def spot_loss(sim, pairs, scale):
    if not pairs: return sim.new_zeros(())
    t = torch.tensor([p[0] for p in pairs], device=sim.device); m = torch.tensor([p[1] for p in pairs], device=sim.device)
    lg = sim * scale
    return 0.5 * (F.cross_entropy(lg[t], m) + F.cross_entropy(lg.T[m], t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tok_dir', default='/projects/kosecka/hongrui/dataset/smplx_fits/pooled_tokens/vq_K512_w64')
    ap.add_argument('--chunk_dir', default=f'{DR}/chunks'); ap.add_argument('--init_ckpt', required=True)
    ap.add_argument('--train_list', default=f'{SUB}/pretrain_all.txt'); ap.add_argument('--val_list', default=f'{SUB}/pool_val.txt')
    ap.add_argument('--sb_npz', default=f'{DR}/signbank_tokens.npz'); ap.add_argument('--sb_split', default=f'{DR}/signbank_split.json')
    ap.add_argument('--anchor_w', type=float, default=0.0); ap.add_argument('--anchor_bs', type=int, default=64)
    ap.add_argument('--mono_w', type=float, default=0.0); ap.add_argument('--mil_w', type=float, default=1.0)
    ap.add_argument('--spot_w', type=float, default=0.0); ap.add_argument('--spots', default=f'{DR}/spots_v4_anchor.tsv'); ap.add_argument('--spot_min_cos', type=float, default=0.4)
    ap.add_argument('--batch_size', type=int, default=48); ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--lr', type=float, default=5e-5); ap.add_argument('--lr_text', type=float, default=1e-5); ap.add_argument('--warmup', type=int, default=500)
    ap.add_argument('--val_every', type=int, default=2000); ap.add_argument('--workers', type=int, default=8); ap.add_argument('--max_items', type=int, default=0)
    ap.add_argument('--out_dir', default='/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/Align_Pooled'); ap.add_argument('--tag', default='align_v4'); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed); rng = random.Random(a.seed); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = f"{time.strftime('%Y%m%d_%H%M%S')}_job{os.environ.get('SLURM_JOB_ID', 'local')}_{a.tag}"; ckdir = os.path.join(a.out_dir, run); logdir = os.path.join(_repo, 'zlog', 'Align_Pooled', run)
    os.makedirs(ckdir, exist_ok=True); os.makedirs(logdir, exist_ok=True); json.dump(vars(a), open(os.path.join(logdir, 'config.json'), 'w'), indent=1)
    log = open(os.path.join(logdir, 'train.log'), 'a')
    def P(*s):
        m = ' '.join(str(x) for x in s); print(m, flush=True); log.write(m + '\n'); log.flush()
    ck = torch.load(a.init_ckpt, map_location='cpu'); c = ck['config']
    model = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']).to(dev); model.load_state_dict(ck['model'])
    cfg = dict(vars(a)); cfg.update({'mmm_ckpt': c['mmm_ckpt'], 'text_encoder': c['text_encoder'], 'e': c['e']})  # so downstream loaders work unchanged
    mi = a.max_items or None
    tr = SpotAlignDataset(a.tok_dir, a.train_list, a.chunk_dir, train=True, max_items=mi, spots_tsv=a.spots if a.spot_w > 0 else None, min_cos=a.spot_min_cos)
    va = ClipAlignDataset(a.tok_dir, a.val_list, a.chunk_dir, train=False, max_items=min(mi or 2000, 2000))
    dl = DataLoader(tr, a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate_spot, drop_last=True, persistent_workers=a.workers > 0)
    dv = DataLoader(va, a.batch_size, shuffle=False, num_workers=min(4, a.workers), collate_fn=collate)
    sb = SignBank(a.sb_npz, a.sb_split, model.mmm.pos.num_embeddings)
    P(f'init {a.init_ckpt} | train {len(tr)} val {len(va)} | signbank train {len(sb.train)} ({len(sb.train_groups)} texts) heldout {len(sb.held)} | anchor_w {a.anchor_w} mono_w {a.mono_w} mil_w {a.mil_w} spot_w {a.spot_w}')
    groups = [{'params': [p for n, p in model.named_parameters() if not n.startswith('text.')], 'lr': a.lr}, {'params': model.text.parameters(), 'lr': a.lr_text}]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=0.01)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    model.eval(); vs = val_retrieval(model, dv, dev); vs.update(signbank_probe(model, sb, dev)); model.train()
    P(f"VAL step 0 loss {vs['loss']:.4f} R1 {vs['R1']:.4f} R10 {vs['R10']:.4f} medR {vs['medR']:.0f} sbR1 {vs['sbR1']:.4f} sbR10 {vs['sbR10']:.4f} sbMedR {vs['sbMedR']:.0f}")
    step, best, t0, agg = 0, -1.0, time.time(), {}
    while step < a.steps:
        for x, m, spans, seg_own, chunks, chk_own, pairs in dl:
            for g, base in zip(opt.param_groups, [a.lr, a.lr_text]): g['lr'] = base * sched(step)
            x, m, seg_own, chk_own = x.to(dev), m.to(dev), seg_own.to(dev), chk_own.to(dev)
            scale = model.logit_scale.exp().clamp(max=100)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                zs = model.embed_segments(x, m, spans); zc = model.embed_chunks(chunks, dev)
                l_mil, _, sim = milnce(zc, zs, chk_own, seg_own, scale)
                loss = a.mil_w * l_mil; parts = {'mil': l_mil.item()}
                if a.mono_w > 0:
                    l_mono = mono_loss((zc @ zs.T).float(), chk_own, seg_own, scale); loss = loss + a.mono_w * l_mono; parts['mono'] = l_mono.item()
                if a.anchor_w > 0:
                    l_anc = anchor_loss(model, sb, a.anchor_bs, rng, dev, scale); loss = loss + a.anchor_w * l_anc; parts['anchor'] = l_anc.item()
                if a.spot_w > 0:
                    l_spot = spot_loss((zc @ zs.T).float(), pairs, scale); loss = loss + a.spot_w * l_spot; parts['spot'] = l_spot.item(); parts['npair'] = float(len(pairs))
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); step += 1
            for k, v in parts.items(): agg[k] = agg.get(k, 0) + v
            agg['loss'] = agg.get('loss', 0) + loss.item()
            if step % 100 == 0: P(f"step {step} lr {opt.param_groups[0]['lr']:.2e} " + ' '.join(f'{k} {v/100:.4f}' for k, v in agg.items()) + f" scale {scale.item():.1f} | {(time.time()-t0)/step:.2f}s/it"); agg = {}
            if step % a.val_every == 0 or step == a.steps:
                model.eval(); vs = val_retrieval(model, dv, dev); vs.update(signbank_probe(model, sb, dev)); model.train()
                P(f"VAL step {step} loss {vs['loss']:.4f} R1 {vs['R1']:.4f} R10 {vs['R10']:.4f} medR {vs['medR']:.0f} sbR1 {vs['sbR1']:.4f} sbR10 {vs['sbR10']:.4f} sbMedR {vs['sbMedR']:.0f}")
                ckd = {'model': model.state_dict(), 'config': cfg, 'step': step, 'val': vs}; torch.save(ckd, os.path.join(ckdir, 'newest_model.pt'))
                if vs['sbR1'] > best: best = vs['sbR1']; torch.save(ckd, os.path.join(ckdir, 'best_model.pt')); P(f'  -> new best sbR1 {best:.4f}')
            if step >= a.steps: break
    P('DONE', ckdir)


if __name__ == '__main__': main()
