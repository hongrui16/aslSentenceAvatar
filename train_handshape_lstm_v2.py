"""
LSTM Handshape Classifier
- Upper-body MediaPipe 3D keypoints only (id 0-24, 25 landmarks)
- Matches glosses with ASL-LEX 2.0 for Handshape labels
- Logs train/test video IDs, saves train.log, plots loss/acc curves
- Saves best checkpoint to log_dir

Usage:
    python train_handshape_lstm.py \
        --keypoints_dir ./keypoints \
        --asl_lex_csv  ./ASL-LEX_View_Data.csv \
        --log_dir ./runs/handshape_lstm \
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
# Upper body landmark IDs  (MediaPipe Pose 33-landmark model)
#   0-10  : face (nose, eyes, ears, mouth)
#   11-12 : shoulders
#   13-14 : elbows
#   15-16 : wrists
#   17-22 : hands (pinky, index, thumb × L/R)
#   23-24 : hips
# Excluded: 25-32 (knees, ankles, heels, foot indices)
UPPER_BODY_IDS = list(range(0, 25))         # 25 landmarks
NUM_UPPER_LANDMARKS = len(UPPER_BODY_IDS)   # 25

# Hand landmarks (MediaPipe Holistic, 21 per hand)
NUM_HAND_LANDMARKS = 21   # wrist + 4 fingers × 4 joints + thumb × 4 joints

# Total feature dimension per frame:
#   upper_body (25×3) + left_hand (21×3) + right_hand (21×3) = 67×3 = 201
TOTAL_LANDMARKS = NUM_UPPER_LANDMARKS + NUM_HAND_LANDMARKS * 2  # 67
FEAT_DIM = TOTAL_LANDMARKS * 3  # 201

# ---- Mode macro ----
# When True, body-pose landmarks (nodes 0-24) are zeroed out in the dataloader,
# so the model trains on hand keypoints only (left hand: 25-45, right hand: 46-66).
HANDS_ONLY: bool = False


# ============================================================
# 1. Dataset
# ============================================================
class HandshapeKeypointDataset(Dataset):
    """
    Returns per video:
        keypoints : (seq_len, 201)  = upper_body(25×3) + left_hand(21×3) + right_hand(21×3)
        label     : int
        mask      : (seq_len,) bool

    Hand landmarks are zero-filled when not detected in a frame.
    """

    def __init__(self, samples, handshape_to_idx, seq_len=40, hands_only: bool = False):
        self.samples = samples            # [(video_dir, handshape), ...]
        self.handshape_to_idx = handshape_to_idx
        self.seq_len = seq_len
        self.feat_dim = FEAT_DIM          # 201
        self.hands_only = hands_only      # if True, zero out body-pose nodes 0-24

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _extract_hand(data, key, num=NUM_HAND_LANDMARKS):
        """Extract hand landmark coords; return zeros if not detected."""
        landmarks = data.get(key, [])
        if not landmarks or not data.get(
            'detected_left_hand' if 'left' in key else 'detected_right_hand', False
        ):
            return [0.0] * (num * 3)
        lm_by_id = {lm['id']: lm for lm in landmarks}
        vec = []
        for i in range(num):
            lm = lm_by_id.get(i, {'x': 0.0, 'y': 0.0, 'z': 0.0})
            vec.extend([lm['x'], lm['y'], lm['z']])
        return vec

    def __getitem__(self, idx):
        video_dir, handshape = self.samples[idx]
        label = self.handshape_to_idx[handshape]

        json_files = sorted(f for f in os.listdir(video_dir) if f.endswith('.json'))

        frames = []
        img_size = None   # read from first valid frame's JSON

        for jf in json_files:
            with open(os.path.join(video_dir, jf), 'r') as f:
                data = json.load(f)
            if not data.get('detected', False):
                continue

            # ---- Read image resolution once (same for all frames in a video) ----
            if img_size is None:
                sz = data.get('image_size', {})
                w = float(sz.get('width',  320))
                h = float(sz.get('height', 240))
                img_size = max(w, h)   # use larger dim so both axes stay in [-1, 1]

            # ---- Upper body pose (25 landmarks) ----
            lm_by_id = {lm['id']: lm for lm in data['pose_world_landmarks_3d']}
            pose_vec = []
            for lid in UPPER_BODY_IDS:
                lm = lm_by_id.get(lid, {'x': 0.0, 'y': 0.0, 'z': 0.0})
                pose_vec.extend([lm['x'], lm['y'], lm['z']])

            # ---- Hands (21 landmarks each, zero if missing) ----
            lh_vec = self._extract_hand(data, 'left_hand_landmarks_3d')
            rh_vec = self._extract_hand(data, 'right_hand_landmarks_3d')

            frames.append(pose_vec + lh_vec + rh_vec)

        if img_size is None:
            img_size = 320.0   # fallback if every frame was undetected

        if len(frames) == 0:
            frames = [[0.0] * self.feat_dim]

        frames = np.array(frames, dtype=np.float32)   # (T, 201)
        T = frames.shape[0]

        # ---- Root normalization: subtract mid-shoulder, then divide by img_size ----
        # Reshape to (T, 67, 3) for landmark-wise operations
        frames_3d = frames.reshape(T, TOTAL_LANDMARKS, 3)

        # Root = midpoint of left shoulder (id=11) and right shoulder (id=12)
        # UPPER_BODY_IDS = range(0,25), so id 11 → index 11, id 12 → index 12
        idx_ls = UPPER_BODY_IDS.index(11)   # left  shoulder
        idx_rs = UPPER_BODY_IDS.index(12)   # right shoulder
        root = (frames_3d[:, idx_ls, :] + frames_3d[:, idx_rs, :]) / 2.0  # (T, 3)

        # Subtract root from every landmark; broadcast over the 67-landmark axis
        frames_3d = frames_3d - root[:, np.newaxis, :]   # (T, 67, 3)

        # Divide by image resolution to remove absolute scale / camera-distance drift
        frames_3d /= img_size                             # coords now unitless

        # ---- HANDS_ONLY: zero out body-pose nodes (0-24), keep hands (25-66) ----
        if self.hands_only:
            frames_3d[:, :NUM_UPPER_LANDMARKS, :] = 0.0

        frames = frames_3d.reshape(T, self.feat_dim)      # (T, 201)

        mask = np.zeros(self.seq_len, dtype=bool)
        if T >= self.seq_len:
            indices = np.linspace(0, T - 1, self.seq_len, dtype=int)
            frames = frames[indices]
            mask[:] = True
        else:
            pad = np.zeros((self.seq_len - T, self.feat_dim), dtype=np.float32)
            frames = np.concatenate([frames, pad], axis=0)
            mask[:T] = True

        return (torch.tensor(frames, dtype=torch.float32),
                label,
                torch.tensor(mask, dtype=torch.bool))


# ============================================================
# 2. Model
# ============================================================
class HandshapeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2,
                 num_classes=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask=None):
        out, _ = self.lstm(x)
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
            hs = row['Handshape'].strip()
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
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger, debug = False):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0
    step = 0
    for kpts, labels, mask in loader:
        kpts, mask = kpts.to(device), mask.to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)
        logits = model(kpts, mask)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tot_loss += loss.item() * kpts.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += kpts.size(0)
        
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
        loss = criterion(logits, labels)
        tot_loss += loss.item() * kpts.size(0)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += kpts.size(0)
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
    project_name = 'handshape_lstm'
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    slurm_id = os.getenv('SLURM_JOB_ID')
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
    logger.info(f"Mode: {'HANDS_ONLY (body-pose zeroed)' if args.hands_only else 'FULL (body + hands)'}")

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
    hs_names = [hs for hs, _ in hs_counter.most_common()]
    hs_to_idx = {hs: i for i, hs in enumerate(hs_names)}
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
        [(vd, hs) for vd, hs, _, _ in train_samples], hs_to_idx, args.seq_len,
        hands_only=args.hands_only)
    test_ds  = HandshapeKeypointDataset(
        [(vd, hs) for vd, hs, _, _ in test_samples],  hs_to_idx, args.seq_len,
        hands_only=args.hands_only)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ---- Model ----
    input_dim = FEAT_DIM  # 201
    model = HandshapeLSTM(input_dim, args.hidden_dim, args.num_layers,
                          num_classes, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: in={input_dim}, hidden={args.hidden_dim}, "
                f"layers={args.num_layers}, classes={num_classes}, params={n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop ----
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    best_acc = 0.0
    ckpt_path = os.path.join(logging_dir, 'best_model.pth')

    logger.info(f"{'Ep':>4} | {'TrLoss':>8} {'TrAcc':>8} | {'TeLoss':>8} {'TeAcc':>8} | LR")
    logger.info("-" * 65)

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, ep, logger, debug = args.debug)
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
                'epoch': ep,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'handshape_to_idx': hs_to_idx,
                'handshape_names': hs_names,
                'test_acc': te_acc,
                'args': vars(args),
            }, ckpt_path)
            logger.info(f"  >> Best! Saved → {ckpt_path}")
            
        if args.debug and ep >3:
            break

    logger.info("=" * 65)
    logger.info(f"Best test acc: {best_acc:.2%}")
    logger.info(f"Checkpoint: {ckpt_path}")

    # ---- Plot ----
    plot_curves(history, logging_dir)
    logger.info(f"All outputs in: {logging_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--keypoints_dir', type=str, default='/scratch/rhong5/dataset/wlasl/video_frame_fitting/keypoints')
    parser.add_argument('--asl_lex_csv',   type=str, default='./data/ASL_LEX2.0/ASL-LEX_View_Data.csv')
    parser.add_argument('--log_dir',       type=str, default='./zlog')
    parser.add_argument('--seq_len',       type=int, default=40)
    parser.add_argument('--batch_size',    type=int, default=64)
    parser.add_argument('--epochs',        type=int, default=80)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--hidden_dim',    type=int, default=256)
    parser.add_argument('--num_layers',    type=int, default=2)
    parser.add_argument('--dropout',       type=float, default=0.3)
    parser.add_argument('--seed',          type=int, default=42)
    parser.add_argument("--debug",       action="store_true", default=False)
    parser.add_argument("--hands_only",  action="store_true", default=False,
                        help="Zero out body-pose landmarks (nodes 0-24); train on hands only")
    args = parser.parse_args()

    main(args)