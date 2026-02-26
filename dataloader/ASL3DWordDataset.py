"""
ASL3DWordDataset — 6D Rotation + Upper Body


This matches the official SignAvatar training pipeline:
  - dataset.py: axis_angle → rotation_matrix → 6D via geometry utils
  - encoder: batch["upper"] with njoints=44, nfeats=6

SMPL-X 53-joint layout (from root_pose + body_pose + lhand + rhand + jaw):
  [0]  root/pelvis         ← KEEP (upper)
  [1]  left_hip             ← REMOVE (lower)
  [2]  right_hip            ← REMOVE (lower)
  [3]  spine1               ← KEEP (upper)
  [4]  left_knee            ← REMOVE (lower)
  [5]  right_knee           ← REMOVE (lower)
  [6]  spine2               ← KEEP (upper)
  [7]  left_ankle           ← REMOVE (lower)
  [8]  right_ankle          ← REMOVE (lower)
  [9]  spine3               ← KEEP (upper)
  [10] left_foot            ← REMOVE (lower)
  [11] right_foot           ← REMOVE (lower)
  [12] neck                 ← KEEP (upper)
  [13] left_collar          ← KEEP (upper)
  [14] right_collar         ← KEEP (upper)
  [15] head                 ← KEEP (upper)
  [16] left_shoulder        ← KEEP (upper)
  [17] right_shoulder       ← KEEP (upper)
  [18] left_elbow           ← KEEP (upper)
  [19] right_elbow          ← KEEP (upper)
  [20] left_wrist           ← KEEP (upper)
  [21] right_wrist          ← KEEP (upper)
  [22-36] left hand (15)    ← KEEP (upper)
  [37-51] right hand (15)   ← KEEP (upper)
  [52] jaw                  ← REMOVE

Upper body = root(1) + upper_body(13) + lhand(15) + rhand(15) = 44 joints

Usage:
    from dataloader.WLASLSMPLXDatasetV2 import WLASLSMPLXDatasetV2

    dataset = WLASLSMPLXDatasetV2(mode='train', cfg=cfg)
    seq, gloss, length = dataset[0]
    # seq.shape = (target_seq_len, 264)   if use_rot6d=True, use_upper_body=True
    # seq.shape = (target_seq_len, 318)   if use_rot6d=True, use_upper_body=False
    # seq.shape = (target_seq_len, 159)   if use_rot6d=False, use_upper_body=False (v1 compat)
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import sys
import pickle

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
from utils.rotation_conversion import axis_angle_to_rot6d, rot6d_to_axis_angle
from utils.rotation_conversion import ALL_53_JOINTS, UPPER_BODY_INDICES, LOWER_BODY_INDICES, REMOVE_INDICES, FULL_JOINT_NAMES, ALL_INDICES

# ============================================================================
# Dataset
# ============================================================================

class ASL3DWordDataset(Dataset):
    """
    Dataset for WLASL SMPL-X parameters.

    File structure:
        smplx_params/
            train/
                samples_label.pkl
                samples_pose.pkl
            test/
                samples_label.pkl
                samples_pose.pkl

    One sample = one video (gloss/video_id/*.npz sorted by frame index).
    WLASL SMPL-X Dataset with 6D rotation and upper body selection.

    Config attributes used:
        ROOT_DIR:           path containing train/test subdirectories
        TARGET_SEQ_LEN:     max sequence length (default 40)
        USE_ROT6D:          use 6D rotation (default True)
        USE_UPPER_BODY:     use 44 upper body joints (default True)
        DROPOUT:            PE dropout (referenced by model, not used here)
    """

    def __init__(self, mode='train', cfg=None, logger = None, gloss_names = None):
        assert mode in ['train', 'test'], f"mode must be 'train' or 'test', got {mode}"
        self.mode = mode
        self.cfg = cfg
        self.logger = logger
        self.gloss_names = gloss_names

        
        # Configuration
        self.use_rot6d = getattr(cfg, 'USE_ROT6D', False) if cfg else False
        self.use_upper_body = getattr(cfg, 'USE_UPPER_BODY', False) if cfg else False
        self.target_seq_len = getattr(cfg, 'TARGET_SEQ_LEN', 40) if cfg else 40
        self.use_mini_dataset = getattr(cfg, 'USE_MINI_DATASET', False) if cfg else False
        self.mini_top_k = getattr(cfg, 'MINI_TOP_K', 5) if cfg else 5
        self.min_frames = 5
        self.root_dir = getattr(cfg, 'ROOT_DIR', './ASL3DWord') if cfg else './ASL3DWord'

        # Compute dimensions
        self.n_joints = len(ALL_INDICES)
        self.n_feats = 6 if self.use_rot6d else 3
        self.input_dim = self.n_joints * self.n_feats


        # Joint indices to use
        # self.joint_indices = UPPER_BODY_INDICES if self.use_upper_body else ALL_53_JOINTS
        self.joint_indices = ALL_53_JOINTS ## 上半身和下半身不在dataloader里面处理, 保持数据一致, 直接在网络端处理.
        
        
        # Data storage

        self.data_list = []      # [(gloss, video_dir_path), ...]
        self.gloss_to_idx = {}   # {gloss_name: int}
        self.gloss_name_list = []
        
        self._load_all_samples()
        if self.logger is not None:
            self.logger.info(f"[{mode}] Config: rot6d={self.use_rot6d}, upper_body={self.use_upper_body}, "
                f"joints={self.n_joints}, feats={self.n_feats}, input_dim={self.input_dim}")

    # ==================== Initialization ====================

            
    def _load_all_samples(self):
        split_dir = os.path.join(self.root_dir, self.mode)
        label_path = os.path.join(split_dir, 'samples_label.pkl')
        pose_path = os.path.join(split_dir, 'samples_pose.pkl')

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Pose file not found: {pose_path}")

        with open(label_path, 'rb') as f:
            labels = pickle.load(f)
        with open(pose_path, 'rb') as f:
            poses = pickle.load(f)

        assert len(labels) == len(poses), \
            f"Label/pose count mismatch: {len(labels)} labels vs {len(poses)} poses"

        skipped = 0
        for gloss, pose in zip(labels, poses):
            gloss = gloss.lower()
            # pose: [T, 53, 3]
            if isinstance(pose, torch.Tensor):
                pose = pose.numpy()
            pose = np.array(pose, dtype=np.float32)

            if pose.shape[0] < self.min_frames:
                skipped += 1
                continue

            # Filter by gloss_names if provided (for test set alignment)
            if self.gloss_names is not None and gloss not in self.gloss_names:
                continue

            if gloss not in self.gloss_to_idx:
                self.gloss_to_idx[gloss] = len(self.gloss_to_idx)
            self.data_list.append((gloss, pose))

        # Mini dataset: keep only top-K glosses by sample count
        if self.use_mini_dataset:
            from collections import Counter
            gloss_counts = Counter(g for g, _ in self.data_list)
            top_glosses = {g for g, _ in gloss_counts.most_common(self.mini_top_k)}

            self.data_list = [(g, p) for g, p in self.data_list if g in top_glosses]
            self.gloss_to_idx = {g: i for i, g in enumerate(sorted(top_glosses))}

            if self.logger is not None:
                self.logger.info(f"[{self.mode}] Mini dataset: kept top {self.mini_top_k} glosses "
                    f"-> {len(self.data_list)} samples, glosses: {sorted(top_glosses)}")

        self.gloss_name_list = [g for g, _ in sorted(self.gloss_to_idx.items(), key=lambda x: x[1])]

        if self.logger is not None:
            self.logger.info(f"[{self.mode}] {len(self.data_list)} samples, "
                f"{len(self.gloss_to_idx)} glosses "
                f"(skipped {skipped} < {self.min_frames} frames)")
            self.logger.info(f"[{self.mode}] Glosses: {sorted(self.gloss_to_idx.keys())}")



    # ==================== Sequence Processing ====================

    def _process_sequence(self, joint_seq):
        """
        Process a raw (T, 53, 3) axis-angle sequence into the final feature tensor.

        Pipeline:
            1. Select joints (53 → 44 if upper body)
            2. Convert to 6D rotation (if enabled)
            3. Root-relative normalization
            4. Flatten to (T, input_dim)

        Args:
            joint_seq: np.ndarray (T, 53, 3) axis-angle

        Returns:
            torch.Tensor (T, input_dim)
        """
        seq = torch.from_numpy(joint_seq)  # (T, 53, 3)

        # Step 1: Select joints
        seq = seq[:, self.joint_indices, :]  # (T, N_joints, 3)

        # Step 2: Root-relative normalization (on axis-angle, before conversion)
        # Subtract first frame's root orientation
        # first_root = seq[0:1, 0:1, :].clone()  # (1, 1, 3)
        if self.cfg.ROOT_NORMALIZE:
            # seq[:, 0:1, :] = seq[:, 0:1, :] - first_root
            seq[:, 0:1, :] = 0.0

        # Step 3: Convert to 6D if enabled
        if self.use_rot6d:
            seq = axis_angle_to_rot6d(seq)  # (T, N_joints, 6)

        # Step 4: Flatten
        T = seq.shape[0]
        seq = seq.reshape(T, -1)  # (T, input_dim)

        return seq

    # ==================== Main Interface ====================

    def __len__(self):
        return len(self.data_list)
    
    
    def __getitem__(self, idx):
        try:
            gloss, pose = self.data_list[idx]

            # pose: (T, 53, 3) numpy array
            seq = self._process_sequence(pose)  # (T, input_dim)
            actual_len = seq.shape[0]

            # Temporal resampling / padding
            if actual_len > self.target_seq_len:
                idx_sub = torch.linspace(0, actual_len - 1, self.target_seq_len).long()
                seq = seq[idx_sub]
                actual_len = self.target_seq_len
            elif actual_len < self.target_seq_len:
                # pad = torch.zeros(self.target_seq_len - actual_len, self.input_dim)
                # seq = torch.cat([seq, pad], dim=0)
                last_frame = seq[-1:].expand(self.target_seq_len - actual_len, -1)
                seq = torch.cat([seq, last_frame], dim=0)


            return seq, gloss, actual_len
        except Exception as e:
            if self.logger is not None:
                self.logger.info(f"[ERROR] __getitem__ idx={idx}, error={e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(self.target_seq_len, self.input_dim), "unknown", 1


    # ==================== Inverse Conversion (for generation / visualization) ====================

    def output_to_smplx_params(self, feature_seq):
        """
        Convert model output back to SMPL-X axis-angle parameters.
        Used in the generation pipeline for mesh rendering.

        Args:
            feature_seq: (T, input_dim) tensor — model output

        Returns:
            dict with keys matching SMPL-X parameter names:
                smplx_root_pose:  (T, 3)
                smplx_body_pose:  (T, 63)
                smplx_lhand_pose: (T, 45)
                smplx_rhand_pose: (T, 45)
                smplx_jaw_pose:   (T, 3)
        """
        if feature_seq.dim() == 1:
            feature_seq = feature_seq.unsqueeze(0)

        T = feature_seq.shape[0]

        # Step 1: Reshape to (T, N_joints, N_feats)
        seq = feature_seq.reshape(T, self.n_joints, self.n_feats)

        # Step 2: Convert back to axis-angle if needed
        if self.use_rot6d:
            seq = rot6d_to_axis_angle(seq)  # (T, N_joints, 3)

        # Step 3: Map back to full 53-joint structure
        full_seq = torch.zeros(T, 53, 3, dtype=seq.dtype, device=seq.device)
        for new_idx, orig_idx in enumerate(self.joint_indices):
            full_seq[:, orig_idx, :] = seq[:, new_idx, :]

        # Step 4: Split into SMPL-X parameter groups
        result = {
            'smplx_root_pose':  full_seq[:, 0, :],                             # (T, 3)
            'smplx_body_pose':  full_seq[:, 1:22, :].reshape(T, 63),           # (T, 63)
            'smplx_lhand_pose': full_seq[:, 22:37, :].reshape(T, 45),          # (T, 45)
            'smplx_rhand_pose': full_seq[:, 37:52, :].reshape(T, 45),          # (T, 45)
            'smplx_jaw_pose':   full_seq[:, 52, :],                            # (T, 3)
        }

        return result

    def output_to_flat_axis_angle(self, feature_seq):
        """
        Convert model output to flat axis-angle vector (159-dim).
        Compatible with V1 generation pipeline / SMPL-X param files.

        Args:
            feature_seq: (T, input_dim) tensor

        Returns:
            (T, 159) tensor — [root(3) + body(63) + lhand(45) + rhand(45) + jaw(3)]
        """
        params = self.output_to_smplx_params(feature_seq)
        return torch.cat([
            params['smplx_root_pose'],
            params['smplx_body_pose'],
            params['smplx_lhand_pose'],
            params['smplx_rhand_pose'],
            params['smplx_jaw_pose'],
        ], dim=-1)  # (T, 159)

    # ==================== Utility ====================

    def get_num_classes(self):
        return len(self.gloss_to_idx)

    def get_gloss_name(self, idx):
        for g, i in self.gloss_to_idx.items():
            if i == idx:
                return g
        return None

    def get_gloss_samples(self, gloss_name):
        return [i for i, (g, _) in enumerate(self.data_list) if g == gloss_name]

    @staticmethod
    def get_upper_body_joint_names():
        """Return ordered list of upper body joint names."""
        return [FULL_JOINT_NAMES[i] for i in UPPER_BODY_INDICES]

    @staticmethod
    def get_removed_joint_names():
        """Return list of removed (lower body + jaw) joint names."""
        return [FULL_JOINT_NAMES[i] for i in sorted(REMOVE_INDICES)]
