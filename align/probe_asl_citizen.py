"""Ground-truth word-retrieval test of an aligner on the ASL Citizen TEST split (2026-09-21). Same protocol as probe_asl3dword.py
but 2,731 words / 32,928 clips / 11 signers never seen by any of our models (ASL3DWord: 103 words / 1,539 clips).
Motion side: pre-tokenized ASL Citizen clips (tokenize_asl_citizen.py, trimmed to the hand-visible span) -> aligner motion embedding of
the whole clip. Text side: the query word from the file name (e.g. "not mind"). Metrics: motion->text R@1/R@5/R@10/medR over the 2,305
distinct query texts (chance R@1 0.0004) and text->motion mAP / R@1 over all test clips. LICENCE: in-house evaluation only.
  python align/probe_asl_citizen.py --align_ckpt <pt> --model_module align.train_align_v6 --out <json>
"""
import argparse, importlib, json, os, sys
import numpy as np, torch, torch.nn.functional as F
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
ASLC = '/scratch/rhong5/dataset/pooled_tokens/asl_citizen_tokens_v1.npz'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--align_ckpt', required=True); ap.add_argument('--model_module', default='align.train_align')
    ap.add_argument('--tokens', default=ASLC); ap.add_argument('--split', default='test'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); dev = 'cuda'
    AlignModel = importlib.import_module(a.model_module).AlignModel
    ck = torch.load(a.align_ckpt, map_location='cpu'); c = ck['config']
    model = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']).to(dev); model.load_state_dict(ck['model']); model.eval(); max_len = model.mmm.pos.num_embeddings
    d = np.load(a.tokens, allow_pickle=True); keep = np.where((d['split'].astype(str) == a.split) & (d['has_hands'] > 0))[0]; off = d['offsets']; tok = d['tokens']
    texts = [str(d['text'][i]) for i in keep]; words = sorted(set(texts)); wi = {w: k for k, w in enumerate(words)}; y = torch.tensor([wi[t] for t in texts], device=dev)
    zm = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for i in range(0, len(keep), 128):
            ts = [torch.from_numpy(tok[off[j]:off[j + 1]].astype(np.int64))[:max_len] for j in keep[i:i + 128]]; L = max(len(t) for t in ts)
            x = torch.zeros(len(ts), L, 4, dtype=torch.long); m = torch.zeros(len(ts), L, dtype=torch.bool); spans = []
            for j, t in enumerate(ts): x[j, :len(t)] = t; m[j, :len(t)] = True; spans.append((j, 0, len(t)))
            zm.append(model.embed_segments(x.to(dev), m.to(dev), spans).float())
        zm = F.normalize(torch.cat(zm), dim=-1); zt = []
        for i in range(0, len(words), 256): zt.append(model.embed_chunks(words[i:i + 256], dev).float())
        zt = F.normalize(torch.cat(zt), dim=-1)
    sim = zm @ zt.T; rank = (sim > sim.gather(1, y[:, None])).sum(1)
    res = {'ckpt': a.align_ckpt, 'split': a.split, 'n_clips': int(len(keep)), 'n_words': len(words), 'chance_R1': 1 / len(words),
           'motion2text': {'R1': float((rank == 0).float().mean()), 'R5': float((rank < 5).float().mean()), 'R10': float((rank < 10).float().mean()), 'medR': int(rank.median()) + 1}}
    aps, r1 = [], []
    for w in range(len(words)):
        s_ = sim[:, w]; pos = (y == w); order = s_.argsort(descending=True); hits = pos[order].float()
        prec = hits.cumsum(0) / torch.arange(1, len(hits) + 1, device=dev); aps.append(float((prec * hits).sum() / hits.sum())); r1.append(float(hits[0]))
    res['text2motion'] = {'mAP': float(np.mean(aps)), 'R1': float(np.mean(r1))}
    print(json.dumps(res, indent=1)); os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True); json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
