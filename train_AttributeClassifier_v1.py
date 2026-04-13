"""
Train Handshape Classifier (ResNet34 + LSTM)
=============================================
Usage:
    python train_HandshapeClassifier.py --task finger    # Selected Fingers (10 classes)
    python train_HandshapeClassifier.py --task handshape # Flexion (9 classes)
"""

import argparse
import os
import time
import logging
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from dataloader.SignBankHandshapeDataset import (
    SignBankHandshapeDataset, get_task_info
)
from network.HandshapeClassifier import HandshapeClassifier




def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)


def plot_curves(log_dir, train_losses, test_losses, train_accs, test_accs):
    """Save training curves to log_dir."""
    epochs = list(range(1, len(train_losses) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, train_losses, 'b-o', markersize=3, label='Train')
    axes[0].plot(epochs, test_losses, 'r-o', markersize=3, label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, train_accs, 'b-o', markersize=3, label='Train')
    axes[1].plot(epochs, test_accs, 'r-o', markersize=3, label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=150)
    plt.close()


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for step, (frames, labels) in enumerate(loader):
        frames = frames.to(device)          # (B, T, C, H, W)
        labels = labels.to(device)          # (B,)

        logits = model(frames)              # (B, num_classes)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 0 and device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            logger.info(
                f'Epoch {epoch} step=0 | batch_size={labels.size(0)} | '
                f'GPU mem allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB'
            )

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        logits = model(frames)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def detailed_evaluate(model, loader, device, class_names, logger):
    """Per-class accuracy, top-1 and top-5 accuracy."""
    model.eval()
    num_classes = len(class_names)
    all_labels = []
    all_logits = []

    for frames, labels in loader:
        frames = frames.to(device)
        logits = model(frames)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)   # (N, C)
    all_labels = torch.cat(all_labels, dim=0)    # (N,)
    N = all_labels.size(0)

    # Top-1
    top1_preds = all_logits.argmax(dim=1)
    top1_correct = (top1_preds == all_labels).sum().item()
    top1_acc = 100.0 * top1_correct / N

    # Top-5
    k = min(5, num_classes)
    top5_preds = all_logits.topk(k, dim=1).indices  # (N, k)
    top5_correct = sum(
        all_labels[i].item() in top5_preds[i].tolist() for i in range(N)
    )
    top5_acc = 100.0 * top5_correct / N

    logger.info('=' * 60)
    logger.info(f'Top-1 Accuracy: {top1_acc:.1f}% ({top1_correct}/{N})')
    logger.info(f'Top-{k} Accuracy: {top5_acc:.1f}% ({top5_correct}/{N})')
    logger.info('=' * 60)

    # Per-class accuracy
    logger.info(f'{"Class":>25s} | {"Correct":>7s} | {"Total":>5s} | {"Acc":>6s}')
    logger.info('-' * 55)
    for c in range(num_classes):
        mask = (all_labels == c)
        total_c = mask.sum().item()
        if total_c == 0:
            logger.info(f'{class_names[c]:>25s} | {"--":>7s} | {0:>5d} | {"N/A":>6s}')
            continue
        correct_c = (top1_preds[mask] == c).sum().item()
        acc_c = 100.0 * correct_c / total_c
        logger.info(f'{class_names[c]:>25s} | {correct_c:>7d} | {total_c:>5d} | {acc_c:>5.1f}%')
    logger.info('=' * 60)

    return top1_acc, top5_acc


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Resolve task ──
    task = args.task
    csv_col, class_names, class_to_idx, num_classes = get_task_info(task)

    # Timestamp + job_id for this run
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    job_id = os.environ.get('SLURM_JOB_ID', '')
    run_name = f'{timestamp}_job{job_id}' if job_id else timestamp
    log_dir = os.path.join(args.log_dir, task, run_name)
    ckpt_dir = os.path.join(args.ckpt_dir, task, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = setup_logging(log_dir)
    logger.info(f'Log dir:  {log_dir}')
    logger.info(f'Ckpt dir: {ckpt_dir}')
    logger.info(f'Task: {task} ({csv_col}, {num_classes} classes)')
    logger.info(f'Args: {vars(args)}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    # ── Load all valid samples to get indices for splitting ──
    frames_dir = args.frames_dir if args.frames_dir else None
    all_samples = SignBankHandshapeDataset._load_annotations(
        args.csv_path, args.video_dir, task=task, frames_dir=frames_dir
    )
    all_labels = [label for _, label in all_samples]
    n_total = len(all_samples)
    logger.info(f'Total valid samples: {n_total}')

    # Stratified train/test split
    indices = list(range(n_total))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_ratio, random_state=args.seed,
        stratify=all_labels
    )
    logger.info(f'Train: {len(train_idx)}, Test: {len(test_idx)}')

    # ── Datasets ──
    train_ds = SignBankHandshapeDataset(
        args.csv_path, args.video_dir, task=task,
        n_frames=args.n_frames, img_size=args.img_size,
        split_indices=train_idx, augment=True, frames_dir=frames_dir,
    )
    test_ds = SignBankHandshapeDataset(
        args.csv_path, args.video_dir, task=task,
        n_frames=args.n_frames, img_size=args.img_size,
        split_indices=test_idx, augment=False, frames_dir=frames_dir,
    )

    # ── Log class distribution ──
    logger.info('='*60)
    logger.info('Class distribution (Train):')
    train_counts = train_ds.get_class_counts()
    for cls in class_names:
        logger.info(f'  {cls:>20s}: {train_counts.get(cls, 0)}')
    logger.info(f'  {"TOTAL":>20s}: {sum(train_counts.values())}')

    logger.info('Class distribution (Test):')
    test_counts = test_ds.get_class_counts()
    for cls in class_names:
        logger.info(f'  {cls:>20s}: {test_counts.get(cls, 0)}')
    logger.info(f'  {"TOTAL":>20s}: {sum(test_counts.values())}')
    logger.info('='*60)

    # ── Compute class weights for imbalanced data ──
    train_label_counts = torch.zeros(num_classes)
    for _, label in train_ds.samples:
        train_label_counts[label] += 1
    class_weights = 1.0 / train_label_counts.clamp(min=1)
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = class_weights.to(device)
    logger.info(f'Class weights: {class_weights.tolist()}')

    # ── DataLoaders ──
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Model ──
    model = HandshapeClassifier(
        num_classes=num_classes,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model params: {n_params:,} total, {n_trainable:,} trainable')

    # ── Eval-only mode ──
    if args.eval_only:
        assert args.checkpoint, '--checkpoint is required for --eval_only'
        logger.info(f'Loading checkpoint: {args.checkpoint}')
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f'Loaded model from epoch {ckpt.get("epoch", "?")}')
        detailed_evaluate(model, test_loader, device, class_names, logger)
        return

    # ── Optimizer & Scheduler ──
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── Training Loop ──
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        elapsed = time.time() - t0
        logger.info(
            f'Epoch {epoch:3d}/{args.epochs} | '
            f'Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | '
            f'Test Loss: {test_loss:.4f} Acc: {test_acc:.1f}% | '
            f'LR: {scheduler.get_last_lr()[0]:.2e} | '
            f'Time: {elapsed:.0f}s'
        )

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
                'args': vars(args),
            }, os.path.join(ckpt_dir, 'best_model.pt'))
            logger.info(f'  ★ New best test acc: {best_test_acc:.1f}%')

        # Save curves every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs:
            plot_curves(log_dir, train_losses, test_losses, train_accs, test_accs)

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_accs[-1],
        'test_loss': test_losses[-1],
        'args': vars(args),
    }, os.path.join(ckpt_dir, 'final_model.pt'))

    logger.info(f'Training complete. Best test acc: {best_test_acc:.1f}%')
    logger.info(f'Checkpoints: {ckpt_dir}')
    logger.info(f'Curves: {os.path.join(log_dir, "training_curves.png")}')

    # ── Detailed evaluation with best model ──
    logger.info('')
    logger.info('Loading best model for detailed evaluation...')
    best_ckpt = torch.load(os.path.join(ckpt_dir, 'best_model.pt'),
                           map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt['model_state_dict'])
    logger.info(f'Loaded best model from epoch {best_ckpt["epoch"]}')
    detailed_evaluate(model, test_loader, device, class_names, logger)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train Handshape Classifier')
    p.add_argument('--task', type=str, default='handshape',
                   choices=['finger', 'handshape'],
                   help='Classification task: "finger" or "handshape"')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--n_frames', type=int, default=4)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--csv_path', type=str,
                   default='./data/ASL_signbank/asl_signbank_dictionary-export.csv')
    p.add_argument('--video_dir', type=str,
                   default='/scratch/rhong5/dataset/asl_signbank/videos/')
    p.add_argument('--frames_dir', type=str, default='',
                   help='Path to preextracted .pt frames (if empty, decode from video)')
    p.add_argument('--log_dir', type=str, default='./zlog/AttributeClassifier')
    p.add_argument('--ckpt_dir', type=str,
                   default='/scratch/rhong5/weights/temp_training_weights/aslSentenceAvatar/AttributeClassifier')
    p.add_argument('--test_ratio', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval_only', action='store_true',
                   help='Skip training, only run detailed evaluation')
    p.add_argument('--checkpoint', type=str, default='',
                   help='Path to checkpoint .pt file (for --eval_only)')
    args = p.parse_args()
    main(args)
