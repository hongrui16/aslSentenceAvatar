import os
import json
import torch
import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SignBankSMPLXDataset(Dataset):
    """
    Dataset for ASL SignBank SMPL-X parameters extracted from keyframe videos.
    
    Expected file structure:
        smplx_params/
            DEVIL/
                DEVIL_000000_p0.npz
                DEVIL_000012_p0.npz
                ...
            HELLO/
                HELLO_000000_p0.npz
                ...
    
    Each .npz contains SMPL-X parameters for a single frame:
        - smplx_root_pose:   (3,)     Global orientation (axis-angle)
        - smplx_body_pose:   (21, 3)  Body joint rotations (axis-angle)
        - smplx_lhand_pose:  (15, 3)  Left hand joint rotations
        - smplx_rhand_pose:  (15, 3)  Right hand joint rotations
        - smplx_jaw_pose:    (3,)     Jaw rotation
        - smplx_shape:       (10,)    Body shape (betas)
        - smplx_expr:        (10,)    Facial expression coefficients
        - cam_trans:         (3,)     Camera translation
        - focal:             (2,)     Focal length [fx, fy]
        - princpt:           (2,)     Principal point [cx, cy]
        - smplx_joint_cam,   (1, 137, 3)  3D joints in camera space
    
    One "sample" = all keyframes of a single gloss, sorted by frame index,
    yielding a temporal sequence of SMPL-X poses.
    
    Pose Feature Vector (per frame):
        root_pose (3) + body_pose (63) + lhand_pose (45) + rhand_pose (45) + jaw_pose (3) = 159
        Optional: + expression (10) + shape (10) + cam_trans (3) = 182
    """

    # ==================== Pose Dimensions ====================
    ROOT_DIM = 3           # 1 joint × 3 (axis-angle)
    BODY_DIM = 63          # 21 joints × 3
    LHAND_DIM = 45         # 15 joints × 3
    RHAND_DIM = 45         # 15 joints × 3
    JAW_DIM = 3            # 1 joint × 3
    POSE_DIM = ROOT_DIM + BODY_DIM + LHAND_DIM + RHAND_DIM + JAW_DIM  # 159

    EXPR_DIM = 10
    SHAPE_DIM = 10
    CAM_TRANS_DIM = 3

    # Joint counts (for reshaping)
    BODY_JOINTS = 21
    LHAND_JOINTS = 15
    RHAND_JOINTS = 15

    def __init__(self, mode='train', cfg=None, logger = None):
        """
        Args:
            mode: 'train' or 'test'
            cfg: Config object with at least:
                - SMPLX_DIR: path to smplx_params root directory
                - TRAIN_SPLIT_FILE / TEST_SPLIT_FILE: split file paths
                - MAX_SEQ_LEN: max sequence length
                - MIN_SEQ_LEN: min sequence length (for interpolation)
                - INTERPOLATE_SHORT_SEQ: bool
                - INCLUDE_EXPRESSION: bool (optional, default True)
                - INCLUDE_SHAPE: bool (optional, default False)
                - INCLUDE_CAM_TRANS: bool (optional, default False)
        """
        assert mode in ['train', 'test'], f"mode must be 'train' or 'test', got {mode}"
        # self.cfg = cfg
        self.mode = mode
        self.data_list = []  # list of gloss folder names

        # Feature selection
        self.include_expr = getattr(cfg, 'INCLUDE_EXPRESSION', False)
        self.include_shape = getattr(cfg, 'INCLUDE_SHAPE', False)
        self.include_cam_trans = getattr(cfg, 'INCLUDE_CAM_TRANS', False)

        # Compute actual input dimension based on config
        self.input_dim = self.POSE_DIM
        if self.include_expr:
            self.input_dim += self.EXPR_DIM
        if self.include_shape:
            self.input_dim += self.SHAPE_DIM
        if self.include_cam_trans:
            self.input_dim += self.CAM_TRANS_DIM
        
        self.n_joints = 53
        self.gloss_name_list = []
        
        self.max_seq_len = getattr(cfg, 'MAX_SEQ_LEN', 85) if cfg is not None else 85
        self.target_seq_len = getattr(cfg, 'TARGET_SEQ_LEN', 85) if cfg is not None else 85 
        
        self.smplx_params_dir = getattr(cfg, 'ROOT_DIR', './smplx_params') if cfg is not None else './smplx_params'

        self._check_dirs()
        self._load_all_glosses()

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
            self.gloss_name_list.append(gloss.lower())
            if any(f.endswith('.npz') for f in os.listdir(gloss_dir)):
                self.data_list.append(gloss)

        print(f"Loaded {len(self.data_list)} glosses from {self.smplx_params_dir}")


    # ==================== Data Loading ====================

    def _get_npz_files(self, gloss_name):
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
                # Parse frame index from filename: GLOSS_FRAMENUM_pX.npz
                parts = fname.replace('.npz', '').split('_')
                # Frame number is the part that's all digits (6-digit zero-padded)
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

        # Sort by frame index
        frames.sort(key=lambda x: x[0])
        return frames

    def _load_frame(self, npz_path):
        """
        Load SMPL-X parameters from a single .npz file.
        
        Returns:
            torch.Tensor: (input_dim,) — concatenated pose features for one frame
        """
        data = np.load(npz_path, allow_pickle=True)

        # Core pose parameters (always included)
        root_pose = data.get('smplx_root_pose', np.zeros(3))                # (3,)
        body_pose = data.get('smplx_body_pose', np.zeros((21, 3)))           # (21, 3)
        lhand_pose = data.get('smplx_lhand_pose', np.zeros((15, 3)))         # (15, 3)
        rhand_pose = data.get('smplx_rhand_pose', np.zeros((15, 3)))         # (15, 3)
        jaw_pose = data.get('smplx_jaw_pose', np.zeros(3))                   # (3,)

        # Flatten all to 1D
        features = [
            root_pose.flatten(),      # (3,)
            body_pose.flatten(),      # (63,)
            lhand_pose.flatten(),     # (45,)
            rhand_pose.flatten(),     # (45,)
            jaw_pose.flatten(),       # (3,)
        ]

        # Optional features
        if self.include_expr:
            expr = data.get('smplx_expr', np.zeros(10))
            features.append(expr.flatten()[:self.EXPR_DIM])  # (10,)

        if self.include_shape:
            shape = data.get('smplx_shape', np.zeros(10))
            features.append(shape.flatten()[:self.SHAPE_DIM])  # (10,)

        if self.include_cam_trans:
            cam_trans = data.get('cam_trans', np.zeros(3))
            features.append(cam_trans.flatten()[:self.CAM_TRANS_DIM])  # (3,)

        feature_vec = np.concatenate(features, axis=0).astype(np.float32)
        return torch.tensor(feature_vec, dtype=torch.float32)

    # ==================== Normalization ====================

    def _normalize_pose(self, pose_seq):
        """
        Normalize SMPL-X pose sequence.
        
        For axis-angle rotations, we normalize the root orientation
        relative to the first frame (root-relative representation),
        making the sequence translation/rotation invariant.
        
        Args:
            pose_seq: (T, D) — pose sequence
            
        Returns:
            normalized_seq: (T, D)
        """
        T, D = pose_seq.shape

        # Extract root orientation (first 3 dims = axis-angle)
        root_poses = pose_seq[:, :self.ROOT_DIM].clone()  # (T, 3)

        # Make root orientation relative to first frame
        # In axis-angle: subtract first frame's rotation (approximate for small rotations)
        # For more accurate version, convert to rotation matrices
        first_root = root_poses[0:1, :]  # (1, 3)
        pose_seq = pose_seq.clone()
        pose_seq[:, :self.ROOT_DIM] = root_poses - first_root

        # If cam_trans is included, also make it relative to first frame
        if self.include_cam_trans:
            cam_start = self.POSE_DIM
            if self.include_expr:
                cam_start += self.EXPR_DIM
            if self.include_shape:
                cam_start += self.SHAPE_DIM
            cam_end = cam_start + self.CAM_TRANS_DIM
            cam_trans = pose_seq[:, cam_start:cam_end].clone()
            first_cam = cam_trans[0:1, :]
            pose_seq[:, cam_start:cam_end] = cam_trans - first_cam

        return pose_seq

        # ==================== Main Interface ====================

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx, debug=False):
        """
        Returns:
            pose_tensor: (T, input_dim) — normalized SMPL-X pose sequence
            label: str — gloss name (lowercase)
            length: int — actual sequence length (before padding)
        """
        gloss_name = self.data_list[idx]
        label = gloss_name.lower()

        # Get sorted frame files
        frame_list = self._get_npz_files(gloss_name)

        if len(frame_list) == 0:
            print(f"Warning: No .npz frames found for {gloss_name}")
            return torch.zeros(1, self.input_dim), label, 1

        num_frames = len(frame_list)

        # Load all frames
        pose_list = []
        for frame_idx, npz_path in frame_list:
            try:
                frame_features = self._load_frame(npz_path)
                pose_list.append(frame_features)
            except Exception as e:
                print(f"Warning: Error loading {npz_path}: {e}")
                continue

        if len(pose_list) == 0:
            print(f"Warning: All frames failed to load for {gloss_name}")
            return torch.zeros(1, self.input_dim), label, 1

        pose_tensor = torch.stack(pose_list)  # (T, input_dim)

        # Normalize
        pose_tensor = self._normalize_pose(pose_tensor)

        if pose_tensor.shape[0] > self.target_seq_len:
            # Keep first and last frame, uniformly sample middle frames
            middle = pose_tensor[1:-1]  # (T-2, input_dim)
            num_sample = self.target_seq_len - 2
            indices = torch.linspace(0, middle.shape[0] - 1, num_sample).long()
            pose_tensor = torch.cat([
                pose_tensor[:1],       # first frame
                middle[indices],       # uniformly sampled middle
                pose_tensor[-1:]       # last frame
            ], dim=0)
        elif pose_tensor.shape[0] < self.target_seq_len:
            # Pad with zeros to reach target length
            pad_len = self.target_seq_len - pose_tensor.shape[0]
            pad_tensor = torch.zeros(pad_len, self.input_dim)
            pose_tensor = torch.cat([pose_tensor, pad_tensor], dim=0)

        length = pose_tensor.shape[0]

        if debug:
            return pose_tensor, label, num_frames

        return pose_tensor, label, length


    def get_feature_indices(self):
        """
        Return a dict mapping parameter names to their index ranges
        in the feature vector. Useful for extracting specific params later.
        """
        idx = 0
        indices = {}

        indices['root_pose'] = (idx, idx + self.ROOT_DIM)
        idx += self.ROOT_DIM

        indices['body_pose'] = (idx, idx + self.BODY_DIM)
        idx += self.BODY_DIM

        indices['lhand_pose'] = (idx, idx + self.LHAND_DIM)
        idx += self.LHAND_DIM

        indices['rhand_pose'] = (idx, idx + self.RHAND_DIM)
        idx += self.RHAND_DIM

        indices['jaw_pose'] = (idx, idx + self.JAW_DIM)
        idx += self.JAW_DIM

        if self.include_expr:
            indices['expression'] = (idx, idx + self.EXPR_DIM)
            idx += self.EXPR_DIM

        if self.include_shape:
            indices['shape'] = (idx, idx + self.SHAPE_DIM)
            idx += self.SHAPE_DIM

        if self.include_cam_trans:
            indices['cam_trans'] = (idx, idx + self.CAM_TRANS_DIM)
            idx += self.CAM_TRANS_DIM

        return indices

    def to_smplx_dict(self, feature_vec):
        """
        Convert a feature vector back to a dict of SMPL-X parameters.
        Useful for visualization / mesh reconstruction.
        
        Args:
            feature_vec: (D,) or (T, D) tensor
            
        Returns:
            dict of parameter tensors
        """
        if feature_vec.dim() == 1:
            feature_vec = feature_vec.unsqueeze(0)  # (1, D)

        indices = self.get_feature_indices()
        result = {}

        s, e = indices['root_pose']
        result['smplx_root_pose'] = feature_vec[:, s:e]  # (T, 3)

        s, e = indices['body_pose']
        result['smplx_body_pose'] = feature_vec[:, s:e].view(-1, self.BODY_JOINTS, 3)  # (T, 21, 3)

        s, e = indices['lhand_pose']
        result['smplx_lhand_pose'] = feature_vec[:, s:e].view(-1, self.LHAND_JOINTS, 3)  # (T, 15, 3)

        s, e = indices['rhand_pose']
        result['smplx_rhand_pose'] = feature_vec[:, s:e].view(-1, self.RHAND_JOINTS, 3)  # (T, 15, 3)

        s, e = indices['jaw_pose']
        result['smplx_jaw_pose'] = feature_vec[:, s:e]  # (T, 3)

        if 'expression' in indices:
            s, e = indices['expression']
            result['smplx_expr'] = feature_vec[:, s:e]  # (T, 10)

        if 'shape' in indices:
            s, e = indices['shape']
            result['smplx_shape'] = feature_vec[:, s:e]  # (T, 10)

        if 'cam_trans' in indices:
            s, e = indices['cam_trans']
            result['cam_trans'] = feature_vec[:, s:e]  # (T, 3)

        return result

