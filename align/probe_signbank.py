"""4.2 acceptance probe: gloss-level retrieval on SignBank citation videos (decided with user 09-14).

Each fused SignBank npz = one sign; label = gloss from the filename. Motion path mirrors training:
clip_feats -> PartVQVAE.encode -> MMM encoder -> span mean (whole clip) -> proj -> L2. Text path:
normalized gloss string -> T5 -> proj -> L2. Metric: retrieval over all glosses in BOTH directions
(motion->text and text->motion), R@1/R@5/R@10/medR; chance R@1 = 1/N.
Caveat: citation form differs from running form (this probes the embedding space, not final 4.3 quality).

  python align/probe_signbank.py --fused <dir> --vq_ckpt <pt> --align_ckpt <pt> --out <json>
"""
import argparse, glob, json, os, re, sys
import numpy as np, torch, torch.nn.functional as F

_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from pretrain.part_vqvae import PartVQVAE
from pretrain.tokenize_corpus import clip_feats
from align.train_align import AlignModel

PARTS = ['body', 'lhand', 'rhand', 'face']


def gloss_text(stem):
    """SignBank filename -> query text: strip per-token trailing lowercase variant markers
    (ABOUTb -> ABOUT, EIGHT-HOURSrot -> EIGHT-HOURS), hyphens -> spaces, lowercase."""
    toks = []
    for t in stem.split('-'):
        m = re.match(r"^([A-Z0-9'#+]+)[a-z]*$", t)
        toks.append(m.group(1) if m else t)
    return ' '.join(toks).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fused', required=True); ap.add_argument('--vq_ckpt', required=True)
    ap.add_argument('--align_ckpt', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--batch', type=int, default=64)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    ck = torch.load(a.vq_ckpt, map_location='cpu'); cfg = ck['config']
    vq = PartVQVAE(ck['part_dims'], cfg['K'], cfg['width'], cfg['down']).to(dev); vq.load_state_dict(ck['model']); vq.eval()
    ack = torch.load(a.align_ckpt, map_location='cpu'); ac = ack['config']
    model = AlignModel(ac['mmm_ckpt'], ac['text_encoder'], ac['e']).to(dev); model.load_state_dict(ack['model']); model.eval()
    max_len = model.mmm.pos.num_embeddings

    files = sorted(glob.glob(os.path.join(a.fused, '*.npz')))
    glosses, toks_all, n_bad = [], [], 0
    with torch.no_grad():
        for f in files:
            try:
                feats, T = clip_feats(f)
                idx = vq.encode({p: torch.from_numpy(v)[None].to(dev) for p, v in feats.items()})
            except Exception as e:
                n_bad += 1; continue
            t = torch.stack([idx[p][0] for p in PARTS], -1)  # (n, 4)
            glosses.append(os.path.splitext(os.path.basename(f))[0]); toks_all.append(t[:max_len].cpu())
    print(f'encoded {len(glosses)} signs ({n_bad} bad)', flush=True)

    zm = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
        for i in range(0, len(toks_all), a.batch):
            batch = toks_all[i:i + a.batch]; L = max(len(t) for t in batch)
            x = torch.zeros(len(batch), L, 4, dtype=torch.long)     # pad id 0, mask True=valid (as in align_dataset.collate)
            m = torch.zeros(len(batch), L, dtype=torch.bool)
            spans = []
            for j, t in enumerate(batch):
                x[j, :len(t)] = t; m[j, :len(t)] = True; spans.append((j, 0, len(t)))
            zm.append(model.embed_segments(x.to(dev), m.to(dev), spans).float().cpu())
        zm = F.normalize(torch.cat(zm), dim=-1)
        texts = [gloss_text(g) for g in glosses]
        zt = []
        for i in range(0, len(texts), a.batch):
            zt.append(model.embed_chunks(texts[i:i + a.batch], dev).float().cpu())
        zt = F.normalize(torch.cat(zt), dim=-1)

    sim = zm @ zt.T  # (N motion, N text)
    res = {'n': len(glosses), 'chance_R1': round(1.0 / len(glosses), 5), 'n_bad': n_bad}
    for name, s in [('motion2text', sim), ('text2motion', sim.T)]:
        rank = (s > s.diag()[:, None]).sum(1)
        res[name] = {'R1': round((rank == 0).float().mean().item(), 4),
                     'R5': round((rank < 5).float().mean().item(), 4),
                     'R10': round((rank < 10).float().mean().item(), 4),
                     'medR': int(rank.median().item()) + 1}
        print(name, json.dumps(res[name]), flush=True)
    # qualitative: 10 best / 10 worst motion->text queries
    rank_m = (sim > sim.diag()[:, None]).sum(1)
    order = rank_m.argsort()
    res['best10'] = [(glosses[i], int(rank_m[i]) + 1) for i in order[:10].tolist()]
    res['worst10'] = [(glosses[i], int(rank_m[i]) + 1) for i in order[-10:].tolist()]
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print('wrote', a.out, flush=True)


if __name__ == '__main__': main()
