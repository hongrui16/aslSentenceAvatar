import os
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np
# Plot training curves
from shapely import length
from torch.nn.utils.rnn import pad_sequence
import torch


def plot_training_curves(fig_path, start_epoch, train_hist, eval_hist = None):    
    epochs = list(range(start_epoch, start_epoch + len(train_hist['total'])))
    
    ncols = 0
    for k in train_hist.keys():
        v = train_hist[k]
        if len(v) > 0:
            ncols += 1

    if eval_hist is not None:
        nrows = 2
    else:
        nrows = 1

    
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4))
    # make axes always 1D
    axes = np.array(axes).reshape(-1)

    col_ind = 0
    for k in train_hist.keys():
        # print("train_hist['total']:", train_hist['total'])
        # Total loss
        axes[col_ind].plot(epochs, train_hist[k], 'b-o', linewidth=2, markersize=4)
        axes[col_ind].set_xlabel('Epoch')
        axes[col_ind].set_ylabel('Loss')
        axes[col_ind].set_title(f'Train {k} Loss')
        axes[col_ind].grid(True, alpha=0.3)
        col_ind += 1
    

    
        
    if eval_hist is not None:
        col_ind = ncols
        for k in eval_hist.keys():
            # print("train_hist['total']:", train_hist['total'])
            # Total loss
            axes[col_ind].plot(epochs, eval_hist[k], 'r-o', linewidth=2, markersize=4)
            axes[col_ind].set_xlabel('Epoch')
            axes[col_ind].set_ylabel('Loss')
            axes[col_ind].set_title(f'Val {k} Loss')
            axes[col_ind].grid(True, alpha=0.3)
            col_ind += 1

    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(fig_path, dpi=150)
    plt.close()

        
        


def backup_code(
    project_root,
    backup_dir,
    logger,
    exclude_dirs=('zlog', 'log', 'temp', 'output')
):
    """
    Backup all .py files under project_root to backup_dir,
    skipping specified directories.
    """
    project_root = Path(project_root).resolve()
    dst_root = Path(backup_dir).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    exclude_dirs = set(exclude_dirs)

    for file_path in project_root.rglob('*.py'):
        # 跳过指定目录
        if any(part in exclude_dirs for part in file_path.parts):
            continue

        relative_path = file_path.relative_to(project_root)
        dst_path = dst_root / relative_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open('r', encoding='utf-8', errors='ignore') as src_file:
            with dst_path.open('w', encoding='utf-8') as dst_file:
                dst_file.write(src_file.read())

    logger.info(f'Backed up code to: {dst_root}')




def collate_fn(batch):
    """
    Custom collate function for variable-length sequences.
    
    Args:
        batch: List of (pose_tensor, label, length) tuples
        
    Returns:
        poses_padded: (B, max_T, D) - padded pose sequences
        labels: List[str] - gloss labels
        gloss_with_attributes: List[str] - gloss labels with attributes
    """
    poses, labels, gloss_with_attributes = zip(*batch)
    
    # Pad sequences to max length in batch
    poses_padded = pad_sequence(poses, batch_first=True, padding_value=0.0)
    lengths = len(poses)
    # Convert lengths to tensor
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return poses_padded, list(labels), lengths_tensor


def create_padding_mask(lengths, max_len, device):
    """
    Create padding mask from lengths.
    
    Args:
        lengths: (B,) - actual sequence lengths
        max_len: int - padded sequence length
        device: torch device
        
    Returns:
        mask: (B, max_len) - True where padded
    """
    lengths = lengths.to(device)
    B = lengths.shape[0]
    indices = torch.arange(max_len, device=device).expand(B, -1)
    mask = indices >= lengths.unsqueeze(1)
    return mask


