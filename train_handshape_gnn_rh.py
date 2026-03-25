"""
LSTM Handshape Classifier  —  RIGHT HAND ONLY, wrist-centred
- Right-hand MediaPipe 3D keypoints only (21 landmarks)
- Root normalization: right-hand wrist (local id 0) subtracted from all joints
- Matches glosses with ASL-LEX 2.0 for Handshape labels
- Logs train/test video IDs, saves train.log, plots loss/acc curves
- Saves best checkpoint to log_dir

Usage:
    python train_handshape_gnn_rh.py \
        --keypoints_dir ./keypoints \
        --asl_lex_csv  ./ASL-LEX_View_Data.csv \
        --log_dir ./runs/handshape_gnn_rh \
        --epochs 50 --batch_size 32 --lr 1e-3
"""

import os
import json
import csv
import argparse
import random
import logging
import numpy as np
from collections import Counter, defaultdict
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found, skipping plots.")


# ============================================================
# Right-hand only — 21 landmarks (MediaPipe Hands)
#   0        : wrist  (used as root for normalization)
#   1 – 4    : thumb
#   5 – 8    : index
#   9 – 12   : middle
#   13 – 16  : ring
#   17 – 20  : pinky
NUM_HAND_LANDMARKS = 21
TOTAL_LANDMARKS    = NUM_HAND_LANDMARKS   # 21
FEAT_DIM           = TOTAL_LANDMARKS * 3  # 63


# ============================================================
# Graph topology  (right hand, 21 nodes, local ids 0-20)
# ============================================================
_HAND_EDGES = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross-links
    (5, 9), (9, 13), (13, 17),
]


def build_adjacency() -> torch.Tensor:
    """
    Build normalised adjacency matrix A_hat = D^{-1/2} A D^{-1/2}
    for the 21-node right-hand graph (with self-loops).
    """
    N = TOTAL_LANDMARKS   # 21
    A = np.zeros((N, N), dtype=np.float32)
    np.fill_diagonal(A, 1.0)   # self-loops

    for i, j in _HAND_EDGES:
        A[i, j] = A[j, i] = 1.0

    D = A.sum(axis=1)
    D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    A_hat = D_inv_sqrt[:, None] * A * D_inv_sqrt[None, :]

    return torch.tensor(A_hat, dtype=torch.float32)


# ============================================================
# 1. Dataset
# ============================================================
class HandshapeKeypointDataset(Dataset):
    """
    Returns per video:
        keypoints : (seq_len, 63)  — right-hand 21 landmarks × 3, wrist-centred
        label     : int
        mask      : (seq_len,) bool

    Right-hand landmarks are zero-filled when not detected in a frame.
    """

    def __init__(self, samples, handshape_to_idx, seq_len=40):
        self.samples          = samples            # [(video_dir, handshape), ...]
        self.handshape_to_idx = handshape_to_idx
        self.seq_len          = seq_len
        self.feat_dim         = FEAT_DIM           # 63

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _extract_right_hand(data):
        """
        Extract right-hand landmark coords (21 × 3).
        Returns a list of 63 floats; all zeros if not detected.
        """
        landmarks = data.get('right_hand_landmarks_3d', [])
        if not landmarks or not data.get('detected_right_hand', False):
            return [0.0] * (NUM_HAND_LANDMARKS * 3)
        lm_by_id = {lm['id']: lm for lm in landmarks}
        vec = []
        for i in range(NUM_HAND_LANDMARKS):
            lm = lm_by_id.get(i, {'x': 0.0, 'y': 0.0, 'z': 0.0})
            vec.extend([lm['x'], lm['y'], lm['z']])
        return vec

    def __getitem__(self, idx):
        video_dir, handshape = self.samples[idx]
        label = self.handshape_to_idx[handshape]

        json_files = sorted(f for f in os.listdir(video_dir) if f.endswith('.json'))

        frames   = []
        img_size = None   # read once from the first valid frame

        for jf in json_files:
            with open(os.path.join(video_dir, jf), 'r') as f:
                data = json.load(f)
            if not data.get('detected', False):
                continue

            # ---- Read image resolution once ----
            if img_size is None:
                sz = data.get('image_size', {})
                w  = float(sz.get('width',  320))
                h  = float(sz.get('height', 240))
                img_size = max(w, h)

            # ---- Right hand only (21 landmarks) ----
            rh_vec = self._extract_right_hand(data)
            frames.append(rh_vec)

        if img_size is None:
            img_size = 320.0   # fallback

        if len(frames) == 0:
            frames = [[0.0] * self.feat_dim]

        frames = np.array(frames, dtype=np.float32)   # (T, 63)
        T = frames.shape[0]

        # ---- Wrist-centred normalisation ----
        # Reshape to (T, 21, 3) so we can index by landmark
        frames_3d = frames.reshape(T, TOTAL_LANDMARKS, 3)

        # Wrist = local landmark id 0  →  node index 0
        wrist = frames_3d[:, 0, :]          # (T, 3)

        # Subtract wrist from every landmark
        frames_3d = frames_3d - wrist[:, np.newaxis, :]   # (T, 21, 3)

        # Divide by image resolution to remove absolute scale / camera-distance drift
        frames_3d /= img_size

        frames = frames_3d.reshape(T, self.feat_dim)      # (T, 63)

        # ---- Pad / sample to fixed seq_len ----
        mask = np.zeros(self.seq_len, dtype=bool)
        if T >= self.seq_len:
            indices = np.linspace(0, T - 1, self.seq_len, dtype=int)
            frames  = frames[indices]
            mask[:] = True
        else:
            pad    = np.zeros((self.seq_len - T, self.feat_dim), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)
            mask[:T] = True

        return (torch.tensor(frames, dtype=torch.float32),
                label,
                torch.tensor(mask, dtype=torch.bool))


# ============================================================
# 2. Model
# ============================================================
class GraphConvLayer(nn.Module):
    """
    Single GCN layer: H' = ReLU(A_hat @ H @ W + b)
    Works on batched sequences: input (B, T, N, C_in) → (B, T, N, C_out)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj: (N, N) — pre-normalised, fixed
        # x  : (B, T, N, C)
        agg = torch.matmul(adj, x)
        return torch.relu(self.linear(agg))


class HandshapeGNN(nn.Module):
    """
    Spatial GCN (per frame) + temporal Bi-LSTM + classifier.

    Pipeline:
      (B, T, 63)
        → reshape → (B, T, 21, 3)
        → GCN ×2  → (B, T, 21, gcn_out)
        → mean-pool over nodes → (B, T, gcn_out)
        → Bi-LSTM  → masked mean-pool → (B, lstm_hidden*2)
        → MLP classifier
    """

    def __init__(self, adj: torch.Tensor,
                 gcn_hidden:  int   = 64,
                 gcn_out:     int   = 128,
                 lstm_hidden: int   = 256,
                 num_layers:  int   = 2,
                 num_classes: int   = 10,
                 dropout:     float = 0.3):
        super().__init__()
        self.register_buffer('adj', adj)          # (21, 21), non-trainable
        self.num_nodes = adj.shape[0]             # 21

        # Spatial GCN stack  (input: 3 xyz coords per node)
        self.gcn1     = GraphConvLayer(3, gcn_hidden)
        self.gcn2     = GraphConvLayer(gcn_hidden, gcn_out)
        self.gcn_norm = nn.LayerNorm(gcn_out)

        # Temporal Bi-LSTM over node-pooled frame features
        self.lstm = nn.LSTM(
            input_size=gcn_out,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, T, _ = x.shape

        # Reshape flat features → per-node xyz
        x = x.view(B, T, self.num_nodes, 3)      # (B, T, 21, 3)

        # Spatial GCN
        x = self.gcn1(x, self.adj)               # (B, T, 21, gcn_hidden)
        x = self.gcn2(x, self.adj)               # (B, T, 21, gcn_out)
        x = self.gcn_norm(x)

        # Global mean pool over nodes → (B, T, gcn_out)
        x = x.mean(dim=2)

        # Temporal Bi-LSTM
        out, _ = self.lstm(x)                    # (B, T, lstm_hidden*2)
        if mask is not None:
            mask_f = mask.unsqueeze(-1).float()
            pooled = (out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        else:
            pooled = out.mean(1)

        return self.classifier(pooled)


# ============================================================
# 3. Data helpers
# ============================================================
def load_asl_lex_handshapes(csv_path):
    """Returns {gloss_lower: handshape_str}"""
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            gloss = row['Entry ID'].strip().lower()
            hs    = row['Handshape'].strip()
            if hs and hs != 'N/A':
                mapping[gloss] = hs
    return mapping


def collect_samples(keypoints_dir, gloss_to_hs):
    """Returns [(video_dir, handshape, gloss, video_id), ...]"""
    samples = []
    for gloss_folder in sorted(os.listdir(keypoints_dir)):
        gpath = os.path.join(keypoints_dir, gloss_folder)
        if not os.path.isdir(gpath):
            continue
        gl = gloss_folder.strip().lower()
        if gl not in gloss_to_hs:
            continue
        hs = gloss_to_hs[gl]
        for vid in sorted(os.listdir(gpath)):
            vdir = os.path.join(gpath, vid)
            if not os.path.isdir(vdir):
                continue
            if not any(f.endswith('.json') for f in os.listdir(vdir)):
                continue
            samples.append((vdir, hs, gl, vid))
    return samples


def split_train_test(samples, test_ratio=0.2, seed=42):
    """Stratified split by handshape."""
    by_class = defaultdict(list)
    for s in samples:
        by_class[s[1]].append(s)
    rng = random.Random(seed)
    train, test = [], []
    for hs, items in by_class.items():
        rng.shuffle(items)
        n_test = max(1, int(len(items) * test_ratio))
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


# ============================================================
# 4. Train / Eval
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger, debug=False):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0
    step = 0
    for kpts, labels, mask in loader:
        kpts, mask = kpts.to(device), mask.to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        logits = model(kpts, mask)
        loss   = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot_loss += loss.item() * kpts.size(0)
        correct  += (logits.argmax(-1) == labels).sum().item()
        total    += kpts.size(0)

        if step == 0:
            mem = torch.cuda.max_memory_allocated(device) / 2**30
            logger.info(f"[epoch {epoch}] Peak GPU: {mem:.2f} GB")
        step += 1

        if debug and step > 10:
            break

    return tot_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    tot_loss, correct, total = 0.0, 0, 0
    for kpts, labels, mask in loader:
        kpts, mask = kpts.to(device), mask.to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        logits = model(kpts, mask)
        loss   = criterion(logits, labels)
        tot_loss += loss.item() * kpts.size(0)
        correct  += (logits.argmax(-1) == labels).sum().item()
        total    += kpts.size(0)
    return tot_loss / total, correct / total


# ============================================================
# 5. Plotting
# ============================================================
def plot_curves(history, save_dir):
    if not HAS_MPL:
        return
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], 'b-o', ms=3, label='Train')
    ax1.plot(epochs, history['test_loss'],  'r-o', ms=3, label='Test')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.set_title('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a*100 for a in history['train_acc']], 'b-o', ms=3, label='Train')
    ax2.plot(epochs, [a*100 for a in history['test_acc']],  'r-o', ms=3, label='Test')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Accuracy')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(path, dpi=150); plt.close()
    print(f"Curves saved: {path}")


# ============================================================
# 6. Logger
# ============================================================
def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger('handshape')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'), mode='w')
    fh.setFormatter(fmt); logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt); logger.addHandler(ch)
    return logger


# ============================================================
# 7. Main
# ============================================================
def main(args):
    project_name = 'handshape_gnn_rh'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    slurm_id  = os.getenv('SLURM_JOB_ID')
    if slurm_id:
        timestamp += f"_job{slurm_id}"
    if args.debug:
        timestamp = "debug_" + timestamp

    logging_dir = os.path.join(args.log_dir, project_name, timestamp)
    os.makedirs(logging_dir, exist_ok=True)
    logger = setup_logger(logging_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info("=" * 50)
    logger.info("ARGS:")
    for k, v in sorted(vars(args).items()):
        logger.info(f"  {k:<20s} = {v}")
    logger.info("=" * 50)
    logger.info("Mode: RIGHT HAND ONLY, wrist-centred normalisation")

    # ---- ASL-LEX ----
    gloss_to_hs = load_asl_lex_handshapes(args.asl_lex_csv)
    logger.info(f"ASL-LEX glosses with handshape: {len(gloss_to_hs)}")

    # ---- Collect & match ----
    samples = collect_samples(args.keypoints_dir, gloss_to_hs)
    logger.info(f"Matched video samples: {len(samples)}")
    if not samples:
        logger.error("No matched samples. Check gloss folder names vs ASL-LEX Entry IDs.")
        return

    # ---- Handshape vocab ----
    hs_counter = Counter(hs for _, hs, _, _ in samples)
    hs_names   = [hs for hs, _ in hs_counter.most_common()]
    hs_to_idx  = {hs: i for i, hs in enumerate(hs_names)}
    num_classes = len(hs_to_idx)
    logger.info(f"Handshape classes: {num_classes}")
    for hs, cnt in hs_counter.most_common():
        logger.info(f"  [{hs_to_idx[hs]:3d}] {hs:20s}: {cnt}")

    # ---- Train / Test split ----
    train_samples, test_samples = split_train_test(samples, 0.2, args.seed)
    logger.info(f"Train: {len(train_samples)}  |  Test: {len(test_samples)}")

    # ---- Log video IDs ----
    logger.info("=" * 70)
    logger.info("TRAIN video IDs:")
    for _, hs, gl, vid in sorted(train_samples, key=lambda x: (x[2], x[3])):
        logger.info(f"  gloss={gl:20s}  video_id={vid}  handshape={hs}")
    logger.info("=" * 70)
    logger.info("TEST video IDs:")
    for _, hs, gl, vid in sorted(test_samples, key=lambda x: (x[2], x[3])):
        logger.info(f"  gloss={gl:20s}  video_id={vid}  handshape={hs}")
    logger.info("=" * 70)

    # Save split lists as separate txt files
    for name, data in [('train', train_samples), ('test', test_samples)]:
        p = os.path.join(logging_dir, f'{name}_videos.txt')
        with open(p, 'w') as f:
            for _, hs, gl, vid in sorted(data, key=lambda x: (x[2], x[3])):
                f.write(f"{gl}\t{vid}\t{hs}\n")
        logger.info(f"Saved {name} list → {p}")

    # ---- Datasets ----
    train_ds = HandshapeKeypointDataset(
        [(vd, hs) for vd, hs, _, _ in train_samples], hs_to_idx, args.seq_len)
    test_ds  = HandshapeKeypointDataset(
        [(vd, hs) for vd, hs, _, _ in test_samples],  hs_to_idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ---- Model ----
    adj   = build_adjacency().to(device)
    model = HandshapeGNN(
        adj         = adj,
        gcn_hidden  = args.gcn_hidden,
        gcn_out     = args.gcn_out,
        lstm_hidden = args.hidden_dim,
        num_layers  = args.num_layers,
        num_classes = num_classes,
        dropout     = args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: GCN({args.gcn_hidden}→{args.gcn_out}) + BiLSTM(hidden={args.hidden_dim}×2), "
                f"nodes=21, classes={num_classes}, params={n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop ----
    history  = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    ckpt_path = os.path.join(logging_dir, 'best_model.pth')

    logger.info(f"{'Ep':>4} | {'TrLoss':>8} {'TrAcc':>8} | {'TeLoss':>8} {'TeAcc':>8} | LR")
    logger.info("-" * 65)

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer,
                                          device, ep, logger, debug=args.debug)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['test_loss'].append(te_loss)
        history['test_acc'].append(te_acc)

        logger.info(f"{ep:4d} | {tr_loss:8.4f} {tr_acc:7.2%} | {te_loss:8.4f} {te_acc:7.2%} | {lr:.2e}")

        if te_acc > best_acc:
            best_acc = te_acc
            torch.save({
                'epoch':              ep,
                'model_state_dict':   model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'handshape_to_idx':   hs_to_idx,
                'handshape_names':    hs_names,
                'test_acc':           te_acc,
                'args':               vars(args),
            }, ckpt_path)
            logger.info(f"  >> Best! Saved → {ckpt_path}")

        if args.debug and ep > 3:
            break

    logger.info("=" * 65)
    logger.info(f"Best test acc: {best_acc:.2%}")
    logger.info(f"Checkpoint: {ckpt_path}")

    # ---- Plot ----
    plot_curves(history, logging_dir)
    logger.info(f"All outputs in: {logging_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypoints_dir', type=str, default='/scratch/rhong5/dataset/wlasl/video_frame_fitting/keypoints_v')
    parser.add_argument('--asl_lex_csv',   type=str, default='./data/ASL_LEX2.0/ASL-LEX_View_Data.csv')
    parser.add_argument('--log_dir',       type=str, default='./zlog')
    parser.add_argument('--seq_len',       type=int, default=40)
    parser.add_argument('--batch_size',    type=int, default=64)
    parser.add_argument('--epochs',        type=int, default=80)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--hidden_dim',    type=int, default=256)
    parser.add_argument('--gcn_hidden',    type=int, default=64)
    parser.add_argument('--gcn_out',       type=int, default=128)
    parser.add_argument('--num_layers',    type=int, default=2)
    parser.add_argument('--dropout',       type=float, default=0.3)
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument("--debug",         action="store_true", default=False)
    args = parser.parse_args()

    main(args)
