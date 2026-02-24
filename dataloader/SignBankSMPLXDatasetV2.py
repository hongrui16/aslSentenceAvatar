"""
SignBankSMPLXDataset — with 6D Rotation support (matching WLASL V2 pipeline)

Key features:
  - 6D continuous rotation representation (optional, via USE_ROT6D)
  - Upper body selection (optional, via USE_UPPER_BODY)
  - Consistent with WLASLSMPLXDatasetV2 architecture

When use_rot6d=True:
  INPUT_DIM = 53 × 6 = 318   (full body)
  INPUT_DIM = 44 × 6 = 264   (upper body only)

When use_rot6d=False:
  INPUT_DIM = 53 × 3 = 159   (full body, V1 compatible)

Expected file structure:
    smplx_params/
        DEVIL/
            DEVIL_000000_p0.npz
            DEVIL_000012_p0.npz
            ...
        HELLO/
            HELLO_000000_p0.npz
            ...

Each .npz contains SMPL-X parameters for a single frame.
One "sample" = all keyframes of a single gloss, sorted by frame index.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.rotation_conversion import axis_angle_to_rot6d, rot6d_to_axis_angle
from utils.rotation_conversion import ALL_53_JOINTS, UPPER_BODY_INDICES, LOWER_BODY_INDICES, REMOVE_INDICES, FULL_JOINT_NAMES, ALL_INDICES


class SignBankSMPLXDatasetV2(Dataset):
    """
    Dataset for ASL SignBank SMPL-X parameters extracted from keyframe videos.

    Config attributes used:
        ROOT_DIR:           path to smplx_params root directory
        TARGET_SEQ_LEN:     target sequence length (default 85)
        USE_ROT6D:          use 6D rotation (default False)
        USE_UPPER_BODY:     use 44 upper body joints (default False)
        ROOT_NORMALIZE:     zero out root orientation (default True)
    """

    def __init__(self, mode='train', cfg=None, logger=None):
        assert mode in ['train', 'test'], f"mode must be 'train' or 'test', got {mode}"
        self.mode = mode
        self.cfg = cfg
        self.logger = logger

        # Configuration
        self.use_rot6d = getattr(cfg, 'USE_ROT6D', False) if cfg else False
        self.use_upper_body = getattr(cfg, 'USE_UPPER_BODY', False) if cfg else False
        self.root_normalize = getattr(cfg, 'ROOT_NORMALIZE', True) if cfg else True
        self.target_seq_len = getattr(cfg, 'TARGET_SEQ_LEN', 40) if cfg else 40
        self.smplx_params_dir = getattr(cfg, 'ROOT_DIR', './smplx_params') if cfg else './smplx_params'

        # Compute dimensions (same logic as WLASL V2)
        self.n_joints = len(ALL_INDICES)
        self.n_feats = 6 if self.use_rot6d else 3
        self.input_dim = self.n_joints * self.n_feats

        # Joint indices — 上半身和下半身不在dataloader里面处理, 保持数据一致, 直接在网络端处理.
        self.joint_indices = ALL_53_JOINTS

        # Data storage
        self.data_list = []        # list of gloss folder names
        self.gloss_to_idx = {}     # {gloss_name: int}
        self.gloss_name_list = []

        self._check_dirs()
        self._load_all_glosses()

        if self.logger is not None:
            self.logger.info(f"[{mode}] SignBank Config: rot6d={self.use_rot6d}, upper_body={self.use_upper_body}, "
                f"joints={self.n_joints}, feats={self.n_feats}, input_dim={self.input_dim}")

    # ==================== Initialization ====================

    def _check_dirs(self):
        """Verify data directory exists."""
        if not os.path.exists(self.smplx_params_dir):
            raise FileNotFoundError(f"SMPL-X directory not found: {self.smplx_params_dir}")

    def _load_all_glosses(self):
        """Load all gloss folders that contain at least one .npz file."""
        all_dirs = sorted([
            d for d in os.listdir(self.smplx_params_dir)
            if os.path.isdir(os.path.join(self.smplx_params_dir, d))
        ])

        for gloss in all_dirs:
            gloss_dir = os.path.join(self.smplx_params_dir, gloss)
            if any(f.endswith('.npz') for f in os.listdir(gloss_dir)):
                gloss_lower = gloss.lower()
                if gloss_lower not in self.gloss_to_idx:
                    self.gloss_to_idx[gloss_lower] = len(self.gloss_to_idx)
                self.data_list.append(gloss)

        self.gloss_name_list = [g for g, _ in sorted(self.gloss_to_idx.items(), key=lambda x: x[1])]

        msg = f"[{self.mode}] Loaded {len(self.data_list)} glosses from {self.smplx_params_dir}"
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    # ==================== Frame Loading ====================

    def _get_sorted_npz(self, gloss_name):
        """
        Get all .npz files for a gloss, sorted by frame index.
        
        Filename pattern: GLOSS_FRAMENUM_pPERSON.npz
        e.g. DEVIL_000012_p0.npz → frame 12, person 0
        
        Returns:
            List of (frame_idx, filepath) tuples sorted by frame_idx
        """
        gloss_dir = os.path.join(self.smplx_params_dir, gloss_name)
        npz_files = [f for f in os.listdir(gloss_dir) if f.endswith('.npz')]

        frames = []
        for fname in npz_files:
            try:
                parts = fname.replace('.npz', '').split('_')
                frame_idx = None
                for part in parts:
                    if part.isdigit() and len(part) >= 4:
                        frame_idx = int(part)
                        break
                if frame_idx is not None:
                    frames.append((frame_idx, os.path.join(gloss_dir, fname)))
            except Exception as e:
                print(f"Warning: Could not parse {fname}: {e}")
                continue

        frames.sort(key=lambda x: x[0])
        return frames

    def _load_frame_joints(self, npz_path):
        """
        Load one frame as (53, 3) axis-angle joint rotations.
        Same layout as WLASL V2 for consistency.

        Returns:
            joints: np.ndarray of shape (53, 3)
        """
        data = np.load(npz_path, allow_pickle=True)

        root_pose = data.get('smplx_root_pose', np.zeros(3)).reshape(1, 3)
        body_pose = data.get('smplx_body_pose', np.zeros((21, 3))).reshape(21, 3)
        lhand_pose = data.get('smplx_lhand_pose', np.zeros((15, 3))).reshape(15, 3)
        rhand_pose = data.get('smplx_rhand_pose', np.zeros((15, 3))).reshape(15, 3)
        jaw_pose = data.get('smplx_jaw_pose', np.zeros(3)).reshape(1, 3)

        # Stack: (53, 3) — same order as WLASL V2 / SignAvatar
        joints = np.vstack([root_pose, body_pose, lhand_pose, rhand_pose, jaw_pose])
        return joints.astype(np.float32)

    # ==================== Sequence Processing ====================

    def _process_sequence(self, joint_seq):
        """
        Process a raw (T, 53, 3) axis-angle sequence into the final feature tensor.

        Pipeline:
            1. Select joints (53 → 44 if upper body, or keep all 53)
            2. Root-relative normalization
            3. Convert to 6D rotation (if enabled)
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
        if self.root_normalize:
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
        """
        Returns:
            seq: (target_seq_len, input_dim) — processed pose sequence
            label: str — gloss name (lowercase)
            actual_len: int — actual sequence length before padding
        """
        try:
            gloss_name = self.data_list[idx]
            label = gloss_name.lower()

            # Load all frames as (T, 53, 3)
            frames = self._get_sorted_npz(gloss_name)
            if not frames:
                return torch.zeros(self.target_seq_len, self.input_dim), label, 1

            joints_list = []
            for _, path in frames:
                try:
                    joints_list.append(self._load_frame_joints(path))
                except Exception as e:
                    print(f"Warning: {path}: {e}")

            if not joints_list:
                return torch.zeros(self.target_seq_len, self.input_dim), label, 1

            joint_seq = np.stack(joints_list, axis=0)  # (T, 53, 3)

            # Process: select joints, normalize, convert to 6D, flatten
            seq = self._process_sequence(joint_seq)  # (T, input_dim)
            actual_len = seq.shape[0]

            # Temporal resampling / padding to target_seq_len
            if actual_len > self.target_seq_len:
                # Keep first and last frame, uniformly sample middle
                middle = seq[1:-1]
                num_sample = self.target_seq_len - 2
                indices = torch.linspace(0, middle.shape[0] - 1, num_sample).long()
                seq = torch.cat([seq[:1], middle[indices], seq[-1:]], dim=0)
                actual_len = self.target_seq_len
            elif actual_len < self.target_seq_len:
                pad = torch.zeros(self.target_seq_len - actual_len, self.input_dim)
                seq = torch.cat([seq, pad], dim=0)

            return seq, label, actual_len

        except Exception as e:
            gloss_name = self.data_list[idx] if idx < len(self.data_list) else "unknown"
            msg = f"[ERROR] __getitem__ idx={idx}, gloss={gloss_name}, error={e}"
            if self.logger is not None:
                self.logger.info(msg)
            else:
                print(msg)
            import traceback
            traceback.print_exc()
            return torch.zeros(self.target_seq_len, self.input_dim), gloss_name.lower(), 1

    # ==================== Inverse Conversion (for generation / visualization) ====================

    def output_to_smplx_params(self, feature_seq):
        """
        Convert model output back to SMPL-X axis-angle parameters.
        Used in the generation pipeline for mesh rendering.

        Args:
            feature_seq: (T, input_dim) tensor — model output

        Returns:
            dict with keys:
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
            'smplx_root_pose':  full_seq[:, 0, :],                           # (T, 3)
            'smplx_body_pose':  full_seq[:, 1:22, :].reshape(T, 63),         # (T, 63)
            'smplx_lhand_pose': full_seq[:, 22:37, :].reshape(T, 45),        # (T, 45)
            'smplx_rhand_pose': full_seq[:, 37:52, :].reshape(T, 45),        # (T, 45)
            'smplx_jaw_pose':   full_seq[:, 52, :],                           # (T, 3)
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
        return [i for i, g in enumerate(self.data_list) if g.lower() == gloss_name]

    @staticmethod
    def get_upper_body_joint_names():
        return [FULL_JOINT_NAMES[i] for i in UPPER_BODY_INDICES]

    @staticmethod
    def get_removed_joint_names():
        return [FULL_JOINT_NAMES[i] for i in sorted(REMOVE_INDICES)]
