"""Independent, ground-truth word-retrieval test for the aligners (2026-09-18): ASL3DWord (103 words, 1,539 clips by real signers,
GT word labels, NOT citation form and NOT touched by any aligner / spotter). Motion side: VQ v1 tokens -> aligner motion embedding of
the whole clip. Text side: the word. Metrics: motion->text R@1/R@5 over 103 words (chance 0.0097) and text->motion mAP / R@1 over
all clips (per word, positives = clips of that word). Same protocol for every model, no learned judge involved.
  python align/probe_asl3dword.py --vq_ckpt <pt> --align_ckpt <pt> --out <json> [--model_module align.train_align_v6]
"""
import argparse, importlib, json, os, pickle, sys
import numpy as np, torch, torch.nn.functional as F
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from pretrain.part_vqvae import PartVQVAE
from pretrain.motion_window_dataset import aa_to_rot6d_np
D = '/home/rhong5/research_pro/hand_modeling_pro/aslAvatar/data/ASL3DWord'
PARTS = ['body', 'lhand', 'rhand', 'face']


def feats_from_aa(aa):
    aa = aa.astype(np.float32).copy(); aa[:, 0] = 0.0; T = len(aa); pad = (-T) % 4
    if pad: aa = np.concatenate([aa, np.repeat(aa[-1:], pad, 0)])
    r6 = aa_to_rot6d_np(aa); W = len(aa); ex = np.zeros((W, 10), np.float32)
    return {'body': r6[:, 1:22].reshape(W, -1), 'lhand': r6[:, 22:37].reshape(W, -1), 'rhand': r6[:, 37:52].reshape(W, -1), 'face': np.concatenate([r6[:, 52], ex], -1)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--vq_ckpt', required=True); ap.add_argument('--align_ckpt', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--model_module', default='align.train_align'); ap.add_argument('--splits', nargs='+', default=['train', 'test']); ap.add_argument('--min_frames', type=int, default=5)
    a = ap.parse_args(); dev = 'cuda'
    AlignModel = importlib.import_module(a.model_module).AlignModel
    ck = torch.load(a.vq_ckpt, map_location='cpu'); cfg = ck['config']
    vq = PartVQVAE(ck['part_dims'], cfg['K'], cfg['width'], cfg['down']).to(dev); vq.load_state_dict(ck['model']); vq.eval()
    ack = torch.load(a.align_ckpt, map_location='cpu'); ac = ack['config']
    model = AlignModel(ac['mmm_ckpt'], ac['text_encoder'], ac['e']).to(dev); model.load_state_dict(ack['model']); model.eval(); max_len = model.mmm.pos.num_embeddings
    poses, labels = [], []
    for sp in a.splits:
        P = pickle.load(open(f'{D}/{sp}/samples_pose.pkl', 'rb')); L = pickle.load(open(f'{D}/{sp}/samples_label.pkl', 'rb'))
        for p, l in zip(P, L):
            if len(p) >= a.min_frames: poses.append(np.asarray(p)); labels.append(str(l).strip().lower().replace('_', ' '))
    words = sorted(set(labels)); wi = {w: i for i, w in enumerate(words)}; y = torch.tensor([wi[l] for l in labels], device=dev)
    toks = []
    with torch.no_grad():
        for p in poses:
            f = feats_from_aa(p); idx = vq.encode({k: torch.from_numpy(v)[None].to(dev) for k, v in f.items()})
            toks.append(torch.stack([idx[k][0] for k in PARTS], -1)[:max_len].cpu())
    zm = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(toks), 64):
            b = toks[i:i + 64]; L = max(len(t) for t in b); x = torch.zeros(len(b), L, 4, dtype=torch.long); m = torch.zeros(len(b), L, dtype=torch.bool); spans = []
            for j, t in enumerate(b): x[j, :len(t)] = t; m[j, :len(t)] = True; spans.append((j, 0, len(t)))
            zm.append(model.embed_segments(x.to(dev), m.to(dev), spans).float())
        zm = F.normalize(torch.cat(zm), dim=-1); zt = F.normalize(model.embed_chunks(words, dev).float(), dim=-1)
    sim = zm @ zt.T  # (N clips, W words)
    rank = (sim > sim.gather(1, y[:, None])).sum(1)
    res = {'n_clips': len(poses), 'n_words': len(words), 'chance_R1': 1 / len(words),
           'motion2text': {'R1': float((rank == 0).float().mean()), 'R5': float((rank < 5).float().mean()), 'R10': float((rank < 10).float().mean()), 'medR': int(rank.median()) + 1}}
    # text->motion: per word, average precision over all clips
    aps, r1 = [], []
    for w in range(len(words)):
        s = sim[:, w]; pos = (y == w); order = s.argsort(descending=True); hits = pos[order].float()
        prec = hits.cumsum(0) / torch.arange(1, len(hits) + 1, device=dev); aps.append(float((prec * hits).sum() / hits.sum())); r1.append(float(hits[0]))
    res['text2motion'] = {'mAP': float(np.mean(aps)), 'R1': float(np.mean(r1)), 'chance_mAP_approx': float(np.mean([(y == w).float().mean().item() for w in range(len(words))]))}
    print(json.dumps(res, indent=1)); os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__': main()
