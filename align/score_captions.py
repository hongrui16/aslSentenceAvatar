"""Caption-motion agreement scores from the 4.2 aligner (caption screening).

Per clip: text embedding = mean of chunk embeddings, motion embedding = mean of non-rest segment embeddings
(exactly the val_retrieval recipe of train_align.py), both L2-normalised, stored f16 per shard. Scoring
against random captions (percentile ranks) is done afterwards by tools/align/analyze_caption_scores.py so
the reference pool can be chosen freely.

  python align/score_captions.py --align_ckpt <best_model.pt> --list <clips.txt> --out_dir <dir> --shard 0 --nshards 4
"""
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from align.align_dataset import ClipAlignDataset, collate
from align.train_align import AlignModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--align_ckpt', required=True); ap.add_argument('--list', required=True); ap.add_argument('--out_dir', required=True)
    ap.add_argument('--tok_dir', default='/projects/kosecka/hongrui/dataset/smplx_fits/pooled_tokens/vq_K512_w64')
    ap.add_argument('--chunk_dir', default='/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/alignment/chunks')
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--batch', type=int, default=64); ap.add_argument('--workers', type=int, default=4); ap.add_argument('--max_items', type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.align_ckpt, map_location='cpu'); c = ck['config']
    model = AlignModel(c['mmm_ckpt'], c['text_encoder'], c['e']).to(dev); model.load_state_dict(ck['model']); model.eval()
    lst = [l.strip() for l in open(a.list) if l.strip()][a.shard::a.nshards]
    tmp = os.path.join(a.out_dir, f'list_{a.shard:02d}.txt'); open(tmp, 'w').write('\n'.join(lst) + '\n')
    ds = ClipAlignDataset(a.tok_dir, tmp, a.chunk_dir, train=False, max_items=a.max_items or None)
    cids = [os.path.basename(t)[:-4].replace('__', ':') for t, _, _ in ds.items]
    dl = DataLoader(ds, a.batch, shuffle=False, num_workers=a.workers, collate_fn=collate)
    ZT, ZM, NS, NC = [], [], [], []
    with torch.no_grad():
        for i, (x, m, spans, seg_own, chunks, chk_own) in enumerate(dl):
            x, m, seg_own, chk_own = x.to(dev), m.to(dev), seg_own.to(dev), chk_own.to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=(dev == 'cuda')):
                zs = model.embed_segments(x, m, spans); zc = model.embed_chunks(chunks, dev)
            B = int(m.shape[0])
            zt = F.normalize(torch.stack([zc[chk_own == j].mean(0) for j in range(B)]).float(), dim=-1)
            zm = F.normalize(torch.stack([zs[seg_own == j].mean(0) for j in range(B)]).float(), dim=-1)
            ZT.append(zt.cpu().numpy().astype(np.float16)); ZM.append(zm.cpu().numpy().astype(np.float16))
            NS.extend(int((seg_own == j).sum()) for j in range(B)); NC.extend(int((chk_own == j).sum()) for j in range(B))
            if i % 200 == 0: print(a.shard, i * a.batch, '/', len(ds), flush=True)
    out = os.path.join(a.out_dir, f'scores_{a.shard:02d}.npz')
    np.savez(out, zt=np.concatenate(ZT), zm=np.concatenate(ZM), clip_ids=np.array(cids), n_seg=np.array(NS), n_chunk=np.array(NC))
    print(a.shard, 'DONE', len(cids), '->', out, flush=True)


if __name__ == '__main__': main()
