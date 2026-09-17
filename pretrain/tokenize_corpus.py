"""Tokenize whole clips with a trained PartVQVAE: per clip -> npz with int16 token ids per part at 25/4 = 6.25 Hz.
  python pretrain/tokenize_corpus.py --ckpt <best_model.pt> --list <clip ids> --out_dir <dir> --shard 0 --nshards 4"""
import argparse, os, sys, csv, numpy as np, torch
_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, _repo)
from pretrain.motion_window_dataset import aa_to_rot6d_np, PART_DIMS
from pretrain.part_vqvae import PartVQVAE
INDEX = '/home/rhong5/research_pro/hand_modeling_pro/asl_data_curation/data_records/curation/clip_index.tsv'

def clip_feats(npz):
    d = np.load(npz); aa = d['axis_angle'].astype(np.float32); ex = d['expression'].astype(np.float32); aa[:, 0] = 0.0; T = len(aa); pad = (-T) % 4
    if pad: aa = np.concatenate([aa, np.repeat(aa[-1:], pad, 0)]); ex = np.concatenate([ex, np.repeat(ex[-1:], pad, 0)])
    r6 = aa_to_rot6d_np(aa); W = len(aa)
    return {'body': r6[:, 1:22].reshape(W, -1), 'lhand': r6[:, 22:37].reshape(W, -1), 'rhand': r6[:, 37:52].reshape(W, -1), 'face': np.concatenate([r6[:, 52], ex], -1)}, T

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--list', required=True); ap.add_argument('--index', default=INDEX); ap.add_argument('--out_dir', required=True); ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(a.ckpt, map_location='cpu'); cfg = ck['config']; model = PartVQVAE(ck['part_dims'], cfg['K'], cfg['width'], cfg['down']).to(dev); model.load_state_dict(ck['model']); model.eval()
    wanted = [l.strip() for l in open(a.list) if l.strip()][a.shard::a.nshards]; wset = set(wanted); rows = {}
    with open(a.index) as f:
        for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
            if r['clip_id'] in wset: rows[r['clip_id']] = r['npz']
    n = 0
    for cid in wanted:
        out = os.path.join(a.out_dir, cid.replace(':', '__') + '.npz')
        if os.path.exists(out) or cid not in rows: continue
        try: feats, T = clip_feats(rows[cid])
        except Exception as e: print('skip', cid, e, flush=True); continue
        with torch.no_grad(): idx = model.encode({p: torch.from_numpy(v)[None].to(dev) for p, v in feats.items()})
        np.savez(out, **{p: idx[p][0].cpu().numpy().astype(np.int16) for p in PART_DIMS}, T=np.int32(T), n_tok=np.int32(len(idx['body'][0]))); n += 1
        if n % 5000 == 0: print(a.shard, n, flush=True)
    print(a.shard, 'DONE', n, flush=True)

if __name__ == '__main__': main()
