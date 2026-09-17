"""Roadmap 4.5 step 1: corpus-wide segment embedding bank from the 4.2 aligner.

Embeds every non-rest segment of every clip in --list with AlignModel.embed_segments (same token-span
conversion as align_dataset: a//4 .. ceil(b/4)); long clips are windowed (window 256, stride 224, each
segment assigned to the first window fully containing it). Output per shard: one npz with
emb (N,512) f16 + clip_ids (unique strings) + per-segment clip_idx / token span / FRAME span
(original tsv frames, for motion lookup at retrieval time).

  python align/embed_segments.py --align_ckpt <best_model.pt> --list <clips.txt> --out_dir <dir> --shard 0 --nshards 8
"""
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from align.align_dataset_v2 import PARTS, SEGMENTS
from align.train_align_v6 import AlignModel

TOK = '/projects/kosecka/hongrui/dataset/smplx_fits/pooled_tokens/vq_K512_w64'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--align_ckpt', required=True); ap.add_argument('--list', required=True)
    ap.add_argument('--tok_dir', default=TOK); ap.add_argument('--segments', default=SEGMENTS)
    ap.add_argument('--out_dir', required=True); ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1); ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--max_tok', type=int, default=256)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    ck = torch.load(a.align_ckpt, map_location='cpu'); c = ck['config']
    model = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']).to(dev)
    model.load_state_dict(ck['model']); model.eval(); model.text = None  # motion side only

    wanted = [l.strip() for l in open(a.list) if l.strip()][a.shard::a.nshards]; wset = set(wanted)
    segs = {}
    with open(a.segments) as f:
        next(f)
        for l in f:
            cid, T, fps, bounds, rest = l.rstrip('\n').split('\t')
            if cid in wset:
                b = list(map(int, bounds.split(','))); rf = list(map(int, rest.split(',')))
                segs[cid] = [(x, y) for x, y, r in zip(b[:-1], b[1:], rf) if not r]

    clip_ids, cid_idx, emb, meta = [], {}, [], []  # meta rows: (clip_idx, tok_a, tok_b, fr_a, fr_b)
    buf = []  # (x windowed tokens, spans rel to window, frame spans)

    @torch.no_grad()
    def flush():
        if not buf: return
        L = max(len(b[0]) for b in buf)
        x = torch.zeros(len(buf), L, 4, dtype=torch.long); m = torch.zeros(len(buf), L, dtype=torch.bool)
        spans, rows = [], []
        for i, (xi, sp, fr, ci) in enumerate(buf):
            x[i, :len(xi)] = torch.from_numpy(xi); m[i, :len(xi)] = True
            for (ta, tb), (fa, fb) in zip(sp, fr):
                spans.append((i, ta, tb)); rows.append((ci, ta, tb, fa, fb))
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
            z = model.embed_segments(x.to(dev), m.to(dev), spans)
        emb.append(z.float().cpu().numpy().astype(np.float16)); meta.extend(rows); buf.clear()

    n_clip, n_seg = 0, 0
    for cid in wanted:
        fp = os.path.join(a.tok_dir, cid.replace(':', '__') + '.npz')
        ss = segs.get(cid)
        if not ss or not os.path.exists(fp): continue
        d = np.load(fp); x = np.stack([d[p].astype(np.int64) for p in PARTS], 1)
        tok_spans = [(fa // 4, max(fa // 4 + 1, -(-fb // 4))) for fa, fb in ss]
        ci = cid_idx.setdefault(cid, len(clip_ids))
        if ci == len(clip_ids): clip_ids.append(cid)
        starts = [0] if len(x) <= a.max_tok else list(range(0, len(x), a.max_tok - 32))
        wins = [(w0, min(w0 + a.max_tok, len(x))) for w0 in starts]
        assign = {}  # window idx -> list of (rel span, frame span); each segment goes to its FIRST fitting window
        for (ta, tb), (fa, fb) in zip(tok_spans, ss):
            for wi, (w0, w1) in enumerate(wins):
                if ta >= w0 and min(tb, len(x)) <= w1:
                    assign.setdefault(wi, []).append(((ta - w0, min(tb, len(x)) - w0), (fa, fb))); break
        for wi, lst in assign.items():
            w0, w1 = wins[wi]
            buf.append((x[w0:w1], [s for s, _ in lst], [f for _, f in lst], ci)); n_seg += len(lst)
            if len(buf) >= a.batch: flush()
        n_clip += 1
        if n_clip % 5000 == 0: print(a.shard, n_clip, n_seg, flush=True)
    flush()
    E = np.concatenate(emb) if emb else np.zeros((0, 512), np.float16)
    M = np.array(meta, np.int64) if meta else np.zeros((0, 5), np.int64)
    out = os.path.join(a.out_dir, f'bank_{a.shard:02d}.npz')
    np.savez(out, emb=E, meta=M, clip_ids=np.array(clip_ids))
    print(a.shard, 'DONE', n_clip, 'clips', len(E), 'segments ->', out, flush=True)


if __name__ == '__main__': main()
