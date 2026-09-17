"""Fixed-fps motion windows from the pooled 25-fps SMPL-X clips (roadmap 4.1 self-supervised pretraining).

Item = a random W-frame window of one clip: rot6d of the 53 SMPL-X joints (root zeroed) + 10 expression coeffs = 328-D per
frame, plus the raw axis-angle (159-D) for the FK loss and a validity mask (clips shorter than W are padded by repeating
the last frame). Eval mode takes the centre window. No temporal resampling: 1 frame = 40 ms everywhere.
"""
import csv, os, random
import numpy as np, torch
from torch.utils.data import Dataset

PARTS = {'body': (1, 22), 'lhand': (22, 37), 'rhand': (37, 52), 'face': (52, 53)}  # joint index ranges into the 53-joint layout
PART_DIMS = {'body': 21 * 6, 'lhand': 15 * 6, 'rhand': 15 * 6, 'face': 6 + 10}    # face = jaw rot6d + expression


def aa_to_rot6d_np(aa):  # (..., 3) -> (..., 6) via Rodrigues, first two columns of R
    th = np.linalg.norm(aa, axis=-1, keepdims=True); k = aa / np.maximum(th, 1e-8)
    c, s = np.cos(th), np.sin(th); kx, ky, kz = k[..., 0:1], k[..., 1:2], k[..., 2:3]; one = 1 - c
    col1 = np.concatenate([c + kx * kx * one, kz * s + kx * ky * one, -ky * s + kx * kz * one], -1)   # R[:,0]
    col2 = np.concatenate([-kz * s + kx * ky * one, c + ky * ky * one, kx * s + ky * kz * one], -1)   # R[:,1]
    return np.concatenate([col1, col2], -1).astype(np.float32)


class MotionWindowDataset(Dataset):
    def __init__(self, index_tsv, list_txt, window=64, train=True, max_items=None):
        wanted = [l.strip() for l in open(list_txt) if l.strip()]; wset = set(wanted); rows = {}
        with open(index_tsv) as f:
            for r in csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE):
                if r['clip_id'] in wset: rows[r['clip_id']] = (r['npz'], r['corpus'])
        self.items = [rows[c] for c in wanted if c in rows]
        if max_items: self.items = self.items[:max_items]
        self.window, self.train = window, train
        print(f'[MotionWindowDataset] {len(self.items)} clips, window={window}, train={train}', flush=True)

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        npz, corpus = self.items[i]
        try:
            d = np.load(npz); aa = d['axis_angle'].astype(np.float32); ex = d['expression'].astype(np.float32)
        except Exception:
            return self[random.randrange(len(self))]
        T, W = len(aa), self.window
        if T >= W:
            s = random.randint(0, T - W) if self.train else (T - W) // 2; aa, ex = aa[s:s + W], ex[s:s + W]; mask = np.ones(W, np.float32)
        else:
            pad = W - T; aa = np.concatenate([aa, np.repeat(aa[-1:], pad, 0)]); ex = np.concatenate([ex, np.repeat(ex[-1:], pad, 0)]); mask = np.concatenate([np.ones(T, np.float32), np.zeros(pad, np.float32)])
        aa[:, 0] = 0.0  # root-zeroed, as everywhere in this project
        if not np.isfinite(aa).all() or not np.isfinite(ex).all(): return self[random.randrange(len(self))]
        r6 = aa_to_rot6d_np(aa)  # (W, 53, 6)
        feats = {'body': r6[:, 1:22].reshape(W, -1), 'lhand': r6[:, 22:37].reshape(W, -1), 'rhand': r6[:, 37:52].reshape(W, -1), 'face': np.concatenate([r6[:, 52], ex], -1)}
        return {k: torch.from_numpy(v) for k, v in feats.items()} | {'aa159': torch.from_numpy(aa.reshape(W, -1)), 'mask': torch.from_numpy(mask), 'corpus': corpus}


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in batch[0] if k != 'corpus'}; out['corpus'] = [b['corpus'] for b in batch]; return out
