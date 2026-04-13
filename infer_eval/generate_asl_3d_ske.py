"""
ASL Avatar Inference Script (V1)

Three main functionalities:
1. Reconstruction: Load test skeleton, mask part, reconstruct with gloss condition
2. Generation: Generate motion directly from gloss condition
3. Visualization: 2D skeleton visualization with GT, masked, reconstruction, generation

Usage:
    python generate_asl_3d_ske.py --checkpoint path/to/checkpoint.pt --output_dir ./results
    python generate_asl_3d_ske.py --checkpoint path/to/checkpoint.pt --num_glosses 50 --mask_ratio 0.5
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from tqdm import tqdm

# Import from your project
from aslAvatarModel import ASLAvatarModel
from dataloader.ASLLVDDataset import ASLLVDSkeletonDataset


class Config:
    """Must match training config"""
    def __init__(self):
        # Dataset
        self.DATASET_NAME = "ASLLVD_Skeleton3D"
        self.SKELETON_DIR = '/scratch/rhong5/dataset/ASLLVD/asl-skeleton3d/normalized/3d'
        self.PHONO_DIR = '/scratch/rhong5/dataset/ASLLVD/asl-phono/phonology/3d'
        self.TRAIN_SPLIT_FILE = "/scratch/rhong5/dataset/ASLLVD/train_split.txt"
        self.TEST_SPLIT_FILE = "/scratch/rhong5/dataset/ASLLVD/test_split.txt"
        
        # Data dimensions
        self.INPUT_DIM = 216
        self.MAX_SEQ_LEN = 50
        self.MIN_SEQ_LEN = 8
        self.INTERPOLATE_SHORT_SEQ = True
        
        # CLIP
        self.CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
        self.CLIP_DIM = 512
        
        # Model
        self.LATENT_DIM = 256
        self.MODEL_DIM = 512
        self.N_HEADS = 8
        self.N_LAYERS = 4
        self.DROPOUT = 0.1


# =============================================================================
# Joint Configuration for Visualization
# =============================================================================

# Upper body joints (14)
UPPER_BODY_JOINTS = [
    'nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
    'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'left_hip',
    'right_eye', 'left_eye', 'right_ear', 'left_ear'
]

# Body skeleton connections (for visualization)
BODY_CONNECTIONS = [
    (0, 1),   # nose -> neck
    (1, 2),   # neck -> right_shoulder
    (2, 3),   # right_shoulder -> right_elbow
    (3, 4),   # right_elbow -> right_wrist
    (1, 5),   # neck -> left_shoulder
    (5, 6),   # left_shoulder -> left_elbow
    (6, 7),   # left_elbow -> left_wrist
    (1, 8),   # neck -> right_hip
    (1, 9),   # neck -> left_hip
    (0, 10),  # nose -> right_eye
    (0, 11),  # nose -> left_eye
    (10, 12), # right_eye -> right_ear
    (11, 13), # left_eye -> left_ear
]

# Hand connections (21 joints per hand)
HAND_CONNECTIONS = [
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
    # Palm
    (5, 9), (9, 13), (13, 17)
]


# =============================================================================
# Model Loading
# =============================================================================

def load_model(checkpoint_path: str, device: str = 'cuda') -> Tuple[ASLAvatarModel, Config]:
    """Load trained model from checkpoint"""
    cfg = Config()
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Override config if saved in checkpoint
    if 'config' in checkpoint:
        saved_cfg = checkpoint['config']
        for key, value in saved_cfg.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    
    # Build model
    model = ASLAvatarModel(cfg)
    
    # Load weights (partial, excluding CLIP)
    model_state = checkpoint.get('model_state_dict', checkpoint)
    current_state = model.state_dict()
    
    loaded_keys = []
    for key in model_state:
        if key in current_state and current_state[key].shape == model_state[key].shape:
            current_state[key] = model_state[key]
            loaded_keys.append(key)
    
    model.load_state_dict(current_state, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Loaded {len(loaded_keys)} weight tensors")
    
    return model, cfg


# =============================================================================
# Masking Functions
# =============================================================================
def apply_frame_mask(motion: torch.Tensor, mask_ratio: Optional[float] = None, 
                     padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply frame masking.
    
    Args:
        motion: (B, T, D) or (T, D) - pose sequences
        mask_ratio: fraction of frames to mask. If None, random mask; if float, fixed uniform mask
        padding_mask: (B, T) or (T,) - True where padded
    
    Returns:
        masked_motion: same shape as input
        frame_mask: (B, T) or (T,) - True where masked
    """
    if motion.dim() == 2:
        motion = motion.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, T, D = motion.shape
    device = motion.device
    
    if mask_ratio is None:
        # Random mask: random ratio between 0.3 and 0.7
        ratio = 0.3 + 0.4 * torch.rand(1).item()
        frame_mask = torch.rand(B, T, device=device) < ratio
    else:
        # Fixed uniform mask
        num_to_mask = int(T * mask_ratio)
        if num_to_mask > 0:
            mask_indices = torch.linspace(0, T - 1, num_to_mask + 2, device=device)[1:-1].long()
            frame_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
            frame_mask[:, mask_indices] = True
        else:
            frame_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    
    # Don't mask padded positions
    if padding_mask is not None:
        if padding_mask.dim() == 1:
            padding_mask = padding_mask.unsqueeze(0)
        frame_mask = frame_mask & ~padding_mask
    
    # Apply mask (zero out masked frames)
    masked_motion = motion.clone()
    masked_motion[frame_mask] = 0
    
    if squeeze_output:
        return masked_motion.squeeze(0), frame_mask.squeeze(0)
    return masked_motion, frame_mask


# =============================================================================
# Reconstruction Function
# =============================================================================

@torch.no_grad()
def reconstruct_with_mask(model: ASLAvatarModel, 
                          motion: torch.Tensor,
                          gloss: str,
                          mask_ratio: float,
                          padding_mask: Optional[torch.Tensor] = None,
                          device: str = 'cuda') -> Dict:
    """
    Reconstruct motion from masked input with gloss condition.
    
    Args:
        model: trained model
        motion: (T, D) ground truth motion
        gloss: gloss label
        mask_ratio: fraction of frames to mask
        padding_mask: (T,) - True where padded
        device: torch device
    
    Returns:
        dict with gt, masked, reconstruction, and metadata
    """
    model.eval()
    
    # Prepare input
    motion = motion.unsqueeze(0).to(device)  # (1, T, D)
    if padding_mask is not None:
        padding_mask = padding_mask.unsqueeze(0).to(device)  # (1, T)
    
    # Apply masking
    masked_motion, frame_mask = apply_frame_mask(motion, mask_ratio, padding_mask)
    
    # Forward pass (reconstruction mode)
    recon_motion, mu, logvar = model(masked_motion, [gloss], padding_mask)
    
    return {
        'gt': motion.squeeze(0).cpu().numpy(),           # (T, D)
        'masked': masked_motion.squeeze(0).cpu().numpy(), # (T, D)
        'recon': recon_motion.squeeze(0).cpu().numpy(),   # (T, D)
        'frame_mask': frame_mask.squeeze(0).cpu().numpy(), # (T,)
        'gloss': gloss
    }


# =============================================================================
# Generation Function
# =============================================================================

@torch.no_grad()
def generate_from_gloss(model: ASLAvatarModel,
                        gloss: str,
                        seq_len: int = 50,
                        input_dim: int = 216,
                        device: str = 'cuda') -> np.ndarray:
    """
    Generate motion directly from gloss condition (unconditional on motion).
    
    Args:
        model: trained model
        gloss: gloss label
        seq_len: output sequence length
        input_dim: motion feature dimension
        device: torch device
    
    Returns:
        generated motion: (T, D)
    """
    model.eval()
    generated = model.generate([gloss], seq_len=seq_len, device=device)
    
    return generated.squeeze(0).cpu().numpy()


# =============================================================================
# Visualization Functions
# =============================================================================

def reshape_motion(motion: np.ndarray) -> np.ndarray:
    """
    Reshape motion from (T, 216) to (T, 72, 3).
    
    Joint order: upper_body(14) + face(16) + hand_left(21) + hand_right(21) = 72
    """
    T = motion.shape[0]
    return motion.reshape(T, -1, 3)


def plot_skeleton_2d(ax, joints_3d: np.ndarray, title: str = "", color: str = 'blue'):
    """
    Plot a single frame skeleton in 2D (X-Y projection).
    
    Args:
        ax: matplotlib axis
        joints_3d: (72, 3) or (N, 3) joints
        title: frame title
        color: joint/bone color
    """
    # Use X and Y for 2D projection (frontal view)
    x = joints_3d[:, 0]
    y = joints_3d[:, 1]
    
    # Flip Y for proper orientation (positive Y should be up)
    y = -y
    
    num_joints = joints_3d.shape[0]
    
    # Upper body (first 14 joints)
    body_joints = min(14, num_joints)
    
    # Plot body connections
    for (i, j) in BODY_CONNECTIONS:
        if i < body_joints and j < body_joints:
            ax.plot([x[i], x[j]], [y[i], y[j]], color=color, linewidth=1.5, alpha=0.8)
    
    # Plot body joints
    ax.scatter(x[:body_joints], y[:body_joints], c=color, s=15, zorder=5)
    
    # Skip face joints (14:30) for cleaner visualization
    
    # Left hand (joints 30:51)
    if num_joints >= 51:
        hand_l_start = 30
        for (i, j) in HAND_CONNECTIONS:
            ax.plot([x[hand_l_start + i], x[hand_l_start + j]], 
                   [y[hand_l_start + i], y[hand_l_start + j]], 
                   color='green', linewidth=1, alpha=0.7)
        ax.scatter(x[hand_l_start:hand_l_start+21], y[hand_l_start:hand_l_start+21], 
                  c='green', s=8, zorder=5)
    
    # Right hand (joints 51:72)
    if num_joints >= 72:
        hand_r_start = 51
        for (i, j) in HAND_CONNECTIONS:
            ax.plot([x[hand_r_start + i], x[hand_r_start + j]], 
                   [y[hand_r_start + i], y[hand_r_start + j]], 
                   color='red', linewidth=1, alpha=0.7)
        ax.scatter(x[hand_r_start:hand_r_start+21], y[hand_r_start:hand_r_start+21], 
                  c='red', s=8, zorder=5)
    
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=8)


def select_frames(motion: np.ndarray, num_frames: int = 8) -> Tuple[np.ndarray, List[int]]:
    """
    Select evenly spaced frames from motion sequence.
    
    Args:
        motion: (T, D) motion sequence
        num_frames: number of frames to select
    
    Returns:
        selected_motion: (num_frames, D)
        frame_indices: list of selected frame indices
    """
    T = motion.shape[0]
    if T <= num_frames:
        # If sequence is shorter, use all frames and pad with last frame
        indices = list(range(T)) + [T-1] * (num_frames - T)
    else:
        indices = np.linspace(0, T-1, num_frames, dtype=int).tolist()
    
    return motion[indices], indices


def visualize_gloss_comparison(gt_motion: np.ndarray,
                                masked_motion: np.ndarray,
                                recon_motion: np.ndarray,
                                gen_motion: np.ndarray,
                                gloss: str,
                                frame_mask: Optional[np.ndarray] = None,
                                num_frames: int = 8,
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create 4-row visualization for a single gloss.
    
    Row 1: Ground Truth (8 frames)
    Row 2: Masked GT
    Row 3: Reconstruction
    Row 4: Generation
    
    Args:
        gt_motion: (T, D) ground truth
        masked_motion: (T, D) masked ground truth
        recon_motion: (T, D) reconstruction
        gen_motion: (T, D) generation
        gloss: gloss label
        frame_mask: (T,) boolean mask showing which frames were masked
        num_frames: number of frames to display
        save_path: optional path to save figure
    
    Returns:
        matplotlib figure
    """
    # Select frames
    gt_frames, indices = select_frames(gt_motion, num_frames)
    masked_frames, _ = select_frames(masked_motion, num_frames)
    recon_frames, _ = select_frames(recon_motion, num_frames)
    gen_frames, _ = select_frames(gen_motion, num_frames)
    
    # Reshape to (num_frames, 72, 3)
    gt_frames = reshape_motion(gt_frames)
    masked_frames = reshape_motion(masked_frames)
    recon_frames = reshape_motion(recon_frames)
    gen_frames = reshape_motion(gen_frames)
    
    # Create figure
    fig, axes = plt.subplots(4, num_frames, figsize=(num_frames * 2, 8))
    
    # Title with gloss name
    fig.suptitle(f'Gloss: {gloss}', fontsize=14, fontweight='bold', y=0.98)
    
    row_labels = ['GT', 'Masked', 'Recon', 'Gen']
    row_colors = ['blue', 'gray', 'orange', 'purple']
    
    for row, (frames, label, color) in enumerate(zip(
        [gt_frames, masked_frames, recon_frames, gen_frames],
        row_labels,
        row_colors
    )):
        for col in range(num_frames):
            ax = axes[row, col]
            
            # For masked row, indicate masked frames
            frame_title = ""
            if row == 1 and frame_mask is not None:
                if indices[col] < len(frame_mask) and frame_mask[indices[col]]:
                    frame_title = "✗"
                    color = 'lightgray'
                else:
                    color = 'gray'
            
            plot_skeleton_2d(ax, frames[col], title=frame_title, color=color)
            
            # Add frame index for first row
            if row == 0:
                ax.set_title(f'F{indices[col]}', fontsize=8)
        
        # Add row label on the left
        axes[row, 0].annotate(
            label, xy=(-0.3, 0.5), xycoords='axes fraction',
            fontsize=12, fontweight='bold', ha='right', va='center',
            color=row_colors[row]
        )
    
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig


def create_visualization_grid(results: List[Dict],
                               num_frames: int = 8,
                               output_dir: str = './visualizations'):
    """
    Create visualization for multiple glosses.
    
    Args:
        results: list of result dicts from process_gloss()
        num_frames: frames per row
        output_dir: output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, result in enumerate(tqdm(results, desc="Creating visualizations")):
        save_path = os.path.join(output_dir, f"{i:03d}_{result['gloss']}.png")
        
        visualize_gloss_comparison(
            gt_motion=result['gt'],
            masked_motion=result['masked'],
            recon_motion=result['recon'],
            gen_motion=result['gen'],
            gloss=result['gloss'],
            frame_mask=result.get('frame_mask'),
            num_frames=num_frames,
            save_path=save_path
        )
    
    print(f"Saved {len(results)} visualizations to {output_dir}")


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_gloss(model: ASLAvatarModel,
                  motion: torch.Tensor,
                  gloss: str,
                  length: int,
                  mask_ratio: float,
                  input_dim: int = 216,
                  device: str = 'cuda') -> Dict:
    """
    Process a single gloss: reconstruction and generation.
    
    Args:
        model: trained model
        motion: (T, D) ground truth motion
        gloss: gloss label
        length: actual sequence length (before padding)
        mask_ratio: fraction of frames to mask
        input_dim: motion feature dimension
        device: torch device
    
    Returns:
        dict with gt, masked, recon, gen, and metadata
    """
    T = motion.shape[0]
    
    # Create padding mask
    padding_mask = torch.zeros(T, dtype=torch.bool)
    if length < T:
        padding_mask[length:] = True
    
    # 1. Reconstruction with masking
    recon_result = reconstruct_with_mask(
        model, motion, gloss, mask_ratio, padding_mask, device
    )
    
    # 2. Generation from gloss only
    gen_motion = generate_from_gloss(model, gloss, seq_len=T, input_dim=input_dim, device=device)
    
    return {
        'gt': recon_result['gt'],
        'masked': recon_result['masked'],
        'recon': recon_result['recon'],
        'gen': gen_motion,
        'frame_mask': recon_result['frame_mask'],
        'gloss': gloss,
        'length': length
    }


def main(args):

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model, cfg = load_model(args.checkpoint, args.device)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = ASLLVDSkeletonDataset(mode='test', cfg=cfg)
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Get unique glosses from dataset
    unique_glosses = list(set([test_dataset[i][1] for i in range(len(test_dataset))]))
    print(f"Unique glosses in test set: {len(unique_glosses)}")
    
    # Select glosses (random sample if more than requested)
    if len(unique_glosses) > args.num_glosses:
        selected_glosses = random.sample(unique_glosses, args.num_glosses)
    else:
        selected_glosses = unique_glosses
    print(f"Processing {len(selected_glosses)} glosses")
    
    # Find samples for each selected gloss
    gloss_to_idx = {}
    for i in range(len(test_dataset)):
        _, gloss, _ = test_dataset[i]
        if gloss in selected_glosses and gloss not in gloss_to_idx:
            gloss_to_idx[gloss] = i
        if len(gloss_to_idx) == len(selected_glosses):
            break
    
    # Process each gloss
    results = []
    for gloss in tqdm(selected_glosses, desc="Processing glosses"):
        if gloss not in gloss_to_idx:
            continue
        
        idx = gloss_to_idx[gloss]
        motion, _, length = test_dataset[idx]
        
        result = process_gloss(
            model=model,
            motion=motion,
            gloss=gloss,
            length=length,
            mask_ratio=args.mask_ratio,
            input_dim=cfg.INPUT_DIM,
            device=args.device
        )
        results.append(result)
    
    print(f"Processed {len(results)} glosses")
    
    # Create visualizations
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    create_visualization_grid(results, args.num_frames, vis_dir)
    
    # Save numerical results
    results_file = os.path.join(args.output_dir, 'results.json')
    results_summary = []
    for r in results:
        results_summary.append({
            'gloss': r['gloss'],
            'length': r['length'],
            'num_masked_frames': int(r['frame_mask'].sum()),
            'gt_shape': list(r['gt'].shape),
        })
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'checkpoint': args.checkpoint,
                'mask_ratio': args.mask_ratio,
                'num_glosses': len(results),
                'num_frames': args.num_frames,
            },
            'results': results_summary
        }, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}")
    print(f"  Visualizations: {vis_dir}")
    print(f"  Summary: {results_file}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="ASL Avatar Inference & Visualization")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="./inference_results2",
                        help="Output directory")
    parser.add_argument("--num_glosses", type=int, default=50,
                        help="Number of glosses to process")
    parser.add_argument("--mask_ratio", type=float, default=0.6,
                        help="Fraction of frames to mask for reconstruction")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of frames to display per row")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    main(args)