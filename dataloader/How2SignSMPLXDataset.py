"""
How2SignDataset — 6D Rotation + Upper Body

Dataset for Neural Sign Actors poses (How2Sign SMPL-X annotations).
Condition signal: full English sentence from How2Sign xlsx annotation.

Data structure:
    poses_root/
        {SENTENCE_NAME}/
            {SENTENCE_NAME}_0_3D.pkl
            {SENTENCE_NAME}_1_3D.pkl
            ...
    how2sign_realigned_*.xlsx  (columns: SENTENCE_NAME, SENTENCE, ...)

Each per-frame pkl contains:
    smplx_root_pose:  (3,)   → joint 0
    smplx_body_pose:  (63,)  → joints 1-21
    smplx_lhand_pose: (45,)  → joints 22-36
    smplx_rhand_pose: (45,)  → joints 37-51
    smplx_jaw_pose:   (3,)   → joint 52
    (smplx_shape, smplx_expr, cam_trans — not used for motion)

Assembled into (T, 53, 3) axis-angle, same format as ASL3DWordDataset.

Sampling strategy (target_len=200):
    T <= target_len : pad last frame
    T >  target_len : segment-based uniform random sampling (train)
                      exact linspace (eval)

Usage:
    dataset = How2SignDataset(mode='train', cfg=cfg)
    seq, sentence, sentence = dataset[0]
    # seq.shape = (200, 318)  if use_rot6d=True
    # seq.shape = (200, 159)  if use_rot6d=False
"""

import os
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import sys
import pickle

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.rotation_conversion import axis_angle_to_rot6d, rot6d_to_axis_angle
from utils.rotation_conversion import ALL_53_JOINTS, UPPER_BODY_INDICES, LOWER_BODY_INDICES, REMOVE_INDICES, FULL_JOINT_NAMES, ALL_INDICES


class How2SignSMPLXDataset(Dataset):
    """
    Dataset for How2Sign sentences paired with Neural Sign Actors SMPL-X poses.

    Config attributes used:
        ROOT_DIR:           path to poses_root (contains SENTENCE_NAME subdirs)
        XLSX_PATH:          path to how2sign_realigned_*.xlsx
        TARGET_SEQ_LEN:     target sequence length (default 200)
        USE_ROT6D:          use 6D rotation (default True)
        USE_UPPER_BODY:     use 44 upper body joints (default True)
        CAMERA:             camera filter string (default 'rgb_front')
    """

    def __init__(self, mode='train', cfg=None, logger=None):
        assert mode in ['train', 'val', 'test'], f"mode must be 'train'/'val'/'test', got {mode}"
        self.mode     = mode
        self.cfg      = cfg
        self.logger   = logger
        self.is_train = (mode == 'train')

        # Configuration
        self.use_rot6d      = getattr(cfg, 'USE_ROT6D',      False)  if cfg else False
        self.use_upper_body = getattr(cfg, 'USE_UPPER_BODY', False)  if cfg else False
        self.target_seq_len = getattr(cfg, 'TARGET_SEQ_LEN', 200)   if cfg else 200
        self.root_dir     = getattr(cfg, 'ROOT_DIR',        '/scratch/rhong5/dataset/Neural-Sign-Actors')   if cfg else '/scratch/rhong5/dataset/Neural-Sign-Actors'
        self.smplx_dir = os.path.join(self.root_dir, f'{mode}_poses/poses')
        self.xlsx_path      = os.path.join(self.root_dir, f'how2sign_realigned_{mode}.xlsx')
        self.camera         = getattr(cfg, 'CAMERA',          'rgb_front') if cfg else 'rgb_front'
        self.min_frames     = 10

        # Compute dimensions
        self.n_joints  = len(ALL_INDICES)
        self.n_feats   = 6 if self.use_rot6d else 3
        self.input_dim = self.n_joints * self.n_feats

        self.joint_indices = ALL_53_JOINTS

        # Data storage
        self.data_list = []   # [(sentence_str, pkl_paths_list), ...]
        self.gloss_name_list = []

        self._load_all_samples()

        if self.logger is not None:
            self.logger.info(
                f"[{mode}] Config: rot6d={self.use_rot6d}, upper_body={self.use_upper_body}, "
                f"joints={self.n_joints}, feats={self.n_feats}, input_dim={self.input_dim}"
            )

    # ==================== Initialization ====================

    def _load_all_samples(self):
        """Load (sentence, pkl_paths) pairs from xlsx + poses_root."""
        df = pd.read_excel(self.xlsx_path)

        if self.camera is not None:
            df = df[df["SENTENCE_NAME"].str.contains(self.camera, na=False)]

        n_missing = n_short = 0
        for _, row in df.iterrows():
            sname    = str(row["SENTENCE_NAME"]).strip()
            stext    = str(row["SENTENCE"]).strip()
            pose_dir = os.path.join(self.smplx_dir, sname)

            if not os.path.isdir(pose_dir):
                n_missing += 1
                continue

            pkls = sorted([
                os.path.join(pose_dir, f)
                for f in os.listdir(pose_dir)
                if f.endswith(".pkl")
            ])
            if len(pkls) < self.min_frames:
                n_short += 1
                continue

            self.data_list.append((stext, pkls))

        lengths  = np.array([len(d[1]) for d in self.data_list])
        n_pad    = (lengths <= self.target_seq_len).sum()
        n_sample = (lengths >  self.target_seq_len).sum()

        if self.logger is not None:
            self.logger.info(f"[{self.mode}] {len(self.data_list)} samples "
                             f"(missing={n_missing}, short={n_short})")
            self.logger.info(f"[{self.mode}] pad={n_pad}, uniform_sample={n_sample}")
        else:
            print(f"[{self.mode}] {len(self.data_list)} samples "
                  f"(missing={n_missing}, short<{self.min_frames}={n_short})")
            print(f"[{self.mode}] pad(T<={self.target_seq_len})={n_pad}, "
                  f"uniform_sample(T>{self.target_seq_len})={n_sample}")

    # ==================== Frame Loading ====================

    @staticmethod
    def _load_pose_from_pkls(pkl_paths):
        """
        Load per-frame pkl files and assemble into (T, 53, 3) axis-angle array.

        Neural Sign Actors per-frame pkl layout:
            smplx_root_pose:  (3,)   → joint 0
            smplx_body_pose:  (63,)  → joints 1-21  (21 joints)
            smplx_lhand_pose: (45,)  → joints 22-36 (15 joints)
            smplx_rhand_pose: (45,)  → joints 37-51 (15 joints)
            smplx_jaw_pose:   (3,)   → joint 52
        """
        frames = []
        for p in pkl_paths:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            frame = np.concatenate([
                np.array(d['smplx_root_pose']).reshape(1,  3),
                np.array(d['smplx_body_pose']).reshape(21, 3),
                np.array(d['smplx_lhand_pose']).reshape(15, 3),
                np.array(d['smplx_rhand_pose']).reshape(15, 3),
                np.array(d['smplx_jaw_pose']).reshape(1,  3),
            ], axis=0)  # (53, 3)
            frames.append(frame)
        return np.stack(frames, axis=0).astype(np.float32)  # (T, 53, 3)

    # ==================== Sampling ====================

    def _sample_indices(self, total_len):
        """
        Select target_seq_len frame indices from total_len frames.

        total_len <= target_seq_len : return all indices (will pad later)
        total_len >  target_seq_len :
            train → segment-based random (uniform coverage + randomness)
            eval  → exact linspace
        """
        if total_len <= self.target_seq_len:
            return list(range(total_len))

        if self.is_train:
            segments = np.array_split(np.arange(total_len), self.target_seq_len)
            indices  = [random.choice(seg.tolist()) for seg in segments]
        else:
            indices = np.round(
                np.linspace(0, total_len - 1, self.target_seq_len)
            ).astype(int).tolist()

        return indices



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
            sentence, pkl_paths = self.data_list[idx]

            # Select frame indices
            indices       = self._sample_indices(len(pkl_paths))
            selected_pkls = [pkl_paths[i] for i in indices]

            # Load → (T, 53, 3) axis-angle
            pose = self._load_pose_from_pkls(selected_pkls)

            # Process: joint selection, rot6d, root-norm, flatten → (T, input_dim)
            seq = self._process_sequence(pose)
            actual_len = seq.shape[0]

            # Pad if shorter than target (only for clips with T < target_seq_len)
            if actual_len < self.target_seq_len:
                last_frame = seq[-1:].expand(self.target_seq_len - actual_len, -1)
                seq = torch.cat([seq, last_frame], dim=0)

            # Return same interface as ASL3DWordDataset: (seq, cond, cond)
            return seq, sentence, sentence

        except Exception as e:
            if self.logger is not None:
                self.logger.info(f"[ERROR] __getitem__ idx={idx}, error={e}")
            import traceback
            traceback.print_exc()
            return torch.zeros(self.target_seq_len, self.input_dim), "", ""


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

    @staticmethod
    def get_upper_body_joint_names():
        return [FULL_JOINT_NAMES[i] for i in UPPER_BODY_INDICES]

    @staticmethod
    def get_removed_joint_names():
        return [FULL_JOINT_NAMES[i] for i in sorted(REMOVE_INDICES)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx",       required=True, help="how2sign_realigned_*.xlsx")
    parser.add_argument("--poses_root", required=True, help="path to poses/ directory")
    parser.add_argument("--target_len", type=int, default=200)
    args = parser.parse_args()

    class _Cfg:
        ROOT_DIR        = args.poses_root
        XLSX_PATH       = args.xlsx
        TARGET_SEQ_LEN  = args.target_len
        USE_ROT6D       = True
        USE_UPPER_BODY  = True
        CAMERA          = 'rgb_front'

    ds = How2SignSMPLXDataset(mode='train', cfg=_Cfg())
    seq, sentence, _ = ds[0]
    print(f"\nSample 0:")
    print(f"  seq shape : {seq.shape}")        # (200, input_dim)
    print(f"  sentence  : {sentence}")