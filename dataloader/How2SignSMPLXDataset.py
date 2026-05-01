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
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.rotation_conversion import axis_angle_to_rot6d, rot6d_to_axis_angle
from utils.rotation_conversion import ALL_53_JOINTS, UPPER_BODY_INDICES, LOWER_BODY_INDICES, REMOVE_INDICES, FULL_JOINT_NAMES, ALL_INDICES


def _load_one_sample(args):
    """Module-level worker for parallel preload (must be picklable).

    Source can be either a list of per-frame pkl paths (legacy) or a single
    .npz path (aggregated). Returns (idx, pose, expr, err).
    """
    idx, source = args
    try:
        if isinstance(source, str) and source.endswith('.npz'):
            d = np.load(source)
            return idx, d['axis_angle'].astype(np.float32), \
                   d['expression'].astype(np.float32), None
        # legacy per-frame pkl list
        pose_frames, expr_frames = [], []
        for p in source:
            with open(p, 'rb') as f:
                d = pickle.load(f)
            frame = np.concatenate([
                np.array(d['smplx_root_pose']).reshape(1,  3),
                np.array(d['smplx_body_pose']).reshape(21, 3),
                np.array(d['smplx_lhand_pose']).reshape(15, 3),
                np.array(d['smplx_rhand_pose']).reshape(15, 3),
                np.array(d['smplx_jaw_pose']).reshape(1,  3),
            ], axis=0)
            pose_frames.append(frame)
            if 'smplx_expr' in d:
                expr_frames.append(np.array(d['smplx_expr']).reshape(10).astype(np.float32))
            else:
                expr_frames.append(np.zeros(10, dtype=np.float32))
        pose = np.stack(pose_frames, axis=0).astype(np.float32)
        expr = np.stack(expr_frames, axis=0).astype(np.float32)
        return idx, pose, expr, None
    except Exception as e:
        return idx, None, None, str(e)


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
        self.use_expression = getattr(cfg, 'USE_EXPRESSION', False)  if cfg else False
        self.n_expr         = getattr(cfg, 'N_EXPR',           10)   if cfg else 10
        self.target_seq_len = getattr(cfg, 'TARGET_SEQ_LEN', 200)   if cfg else 200
        self.root_dir       = getattr(cfg, 'ROOT_DIR',        '/scratch/rhong5/dataset/Neural-Sign-Actors')   if cfg else '/scratch/rhong5/dataset/Neural-Sign-Actors'
        self.smplx_dir      = os.path.join(self.root_dir, f'{mode}_poses/poses')
        self.aggregated_dir = os.path.join(self.root_dir, f'{mode}_poses_aggregated')
        self.fk_joints_dir  = os.path.join(self.root_dir, f'{mode}_fk_joints44')
        self.xlsx_path      = os.path.join(self.root_dir, f'how2sign_realigned_{mode}.xlsx')
        self.camera         = getattr(cfg, 'CAMERA',          'rgb_front') if cfg else 'rgb_front'
        self.min_frames     = 10
        self.preload        = getattr(cfg, 'PRELOAD_TO_MEMORY', False) if cfg else False
        # Auto-detect: prefer per-sentence .npz when the aggregated dir exists.
        # Set USE_AGGREGATED_NPZ=False explicitly to force the legacy pkl path.
        self.use_aggregated = (
            getattr(cfg, 'USE_AGGREGATED_NPZ', True) if cfg else True
        ) and os.path.isdir(self.aggregated_dir)
        # FK joints cache (T, 44, 3) precomputed via tools/precompute_fk_joints.py.
        # When enabled, __getitem__ returns a 5-tuple with the GT joints so
        # the FK loss can skip its GT FK call. Set tentatively here; the
        # final value is re-checked after _load_all_samples() so a partial
        # cache (dir exists but some files still being written) safely falls
        # back to on-the-fly FK instead of crashing on FileNotFoundError.
        self._fk_cache_requested = bool(
            getattr(cfg, 'USE_FK_JOINTS_CACHE', False) if cfg else False
        )
        self.use_fk_cache = (
            self._fk_cache_requested and os.path.isdir(self.fk_joints_dir)
        )
        # Sentence-length filter — drop very-short ("hi"/"yes") and very-long
        # (>CLIP context) sentences. Applied at dataset-build time so train and
        # eval see the same distribution. None disables.
        self.filter_words_min = getattr(cfg, 'FILTER_WORDS_MIN', None) if cfg else None
        self.filter_words_max = getattr(cfg, 'FILTER_WORDS_MAX', None) if cfg else None

        # Always emit the sparse 53-joint layout; models (V1/V2/NSA) handle
        # `use_upper_body` themselves via their own bypass/tosave_slices logic.
        # Expression (10 blendshape coeffs) is appended at the tail when enabled.
        self.joint_indices = ALL_53_JOINTS
        self.n_joints      = len(self.joint_indices)
        self.n_feats       = 6 if self.use_rot6d else 3
        self.input_dim     = self.n_joints * self.n_feats + (self.n_expr if self.use_expression else 0)

        # Data storage
        self.data_list = []   # [(sentence_str, pkl_paths_list), ...]
        # Optional in-memory cache populated by _preload_all_pkls()
        # When self.preload, __getitem__ uses these instead of opening pkls.
        self.pose_cache = None  # list[np.ndarray (T, 53, 3)]
        self.expr_cache = None  # list[np.ndarray (T, 10)]
        # Instance-level blacklist of indices known to fail (e.g. corrupt
        # source files / NaN). MUST be instance-level: train and val instances
        # have different data_lists, so a bad idx in one is unrelated to the
        # other.
        self._bad_indices = set()


        self._load_all_samples()
        if self.preload:
            self._preload_all_pkls()

        # Verify FK cache covers every sample. A partial cache (e.g. the
        # precompute job is still running) auto-disables the cache path so
        # training falls back to on-the-fly FK instead of crashing.
        if self.use_fk_cache:
            n_missing = 0
            for stext, source in self.data_list:
                if isinstance(source, str) and source.endswith('.npz'):
                    sname = os.path.splitext(os.path.basename(source))[0]
                else:
                    sname = os.path.basename(os.path.dirname(source[0]))
                if not os.path.exists(os.path.join(self.fk_joints_dir, f'{sname}.npz')):
                    n_missing += 1
            if n_missing > 0:
                msg = (f"[{mode}] FK cache incomplete: {n_missing}/{len(self.data_list)} "
                       f"samples missing in {self.fk_joints_dir} — "
                       f"falling back to on-the-fly SMPL-X FK")
                if self.logger is not None:
                    self.logger.info(msg)
                else:
                    print(msg)
                self.use_fk_cache = False
            else:
                msg = f"[{mode}] FK cache OK ({len(self.data_list)} files)"
                if self.logger is not None:
                    self.logger.info(msg)
                else:
                    print(msg)

        if self.logger is not None:
            self.logger.info(
                f"[{mode}] Config: rot6d={self.use_rot6d}, upper_body={self.use_upper_body}, "
                f"joints={self.n_joints}, feats={self.n_feats}, input_dim={self.input_dim}"
            )

    # ==================== Initialization ====================

    def _load_all_samples(self):
        """Build (sentence, source) list. Source is .npz path (aggregated path)
        or list of pkl paths (legacy)."""
        df = pd.read_excel(self.xlsx_path)

        if self.camera is not None:
            df = df[df["SENTENCE_NAME"].str.contains(self.camera, na=False)]

        n_missing = n_short = n_corrupt = 0

        n_filtered_words = 0
        if self.use_aggregated:
            for _, row in df.iterrows():
                sname = str(row["SENTENCE_NAME"]).strip()
                stext = str(row["SENTENCE"]).strip()
                if not stext:
                    n_missing += 1
                    continue
                # Sentence-length filter (word count)
                w = len(stext.split())
                if self.filter_words_min is not None and w < self.filter_words_min:
                    n_filtered_words += 1
                    continue
                if self.filter_words_max is not None and w > self.filter_words_max:
                    n_filtered_words += 1
                    continue
                npz_path = os.path.join(self.aggregated_dir, f"{sname}.npz")
                if not os.path.exists(npz_path):
                    n_missing += 1
                    continue
                self.data_list.append((stext, npz_path))
        else:
            for _, row in df.iterrows():
                sname    = str(row["SENTENCE_NAME"]).strip()
                stext    = str(row["SENTENCE"]).strip()
                pose_dir = os.path.join(self.smplx_dir, sname)

                if not os.path.isdir(pose_dir) or not stext:
                    n_missing += 1
                    continue
                # Sentence-length filter (word count)
                w = len(stext.split())
                if self.filter_words_min is not None and w < self.filter_words_min:
                    n_filtered_words += 1
                    continue
                if self.filter_words_max is not None and w > self.filter_words_max:
                    n_filtered_words += 1
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

        # Length stats only meaningful in legacy pkl path (where source is a
        # list of frame-pkls). In aggregated mode, T lives inside the .npz so
        # we skip this stat to avoid 30k file opens.
        msg_src = 'npz' if self.use_aggregated else 'pkls'
        filt_msg = ''
        if self.filter_words_min is not None or self.filter_words_max is not None:
            filt_msg = f", filt_words={n_filtered_words} (range={self.filter_words_min}-{self.filter_words_max})"
        if self.use_aggregated:
            stat_msg = f"[{self.mode}] {len(self.data_list)} samples (src={msg_src}, missing={n_missing}{filt_msg})"
        else:
            lengths  = np.array([len(d[1]) for d in self.data_list])
            n_pad    = (lengths <= self.target_seq_len).sum()
            n_sample = (lengths >  self.target_seq_len).sum()
            stat_msg = (f"[{self.mode}] {len(self.data_list)} samples (src={msg_src}, "
                        f"missing={n_missing}, short={n_short}, corrupt={n_corrupt}, "
                        f"pad={n_pad}, uniform_sample={n_sample})")

        if self.logger is not None:
            self.logger.info(stat_msg)
        else:
            print(stat_msg)

    # ==================== In-memory preload ====================

    def _preload_all_pkls(self):
        """
        Load every per-frame pkl into RAM up-front via a process pool so
        __getitem__ can skip disk IO. The cache lives in the main process;
        DataLoader workers inherit it cheaply via fork (copy-on-write).

        Memory footprint at ~150 frames/sample × 53 × 3 × 4 bytes ≈ 95KB/sample;
        22k samples ≈ 2GB total.
        """
        self.pose_cache = [None] * len(self.data_list)
        self.expr_cache = [None] * len(self.data_list)

        n_workers = max(1, getattr(self.cfg, 'PRELOAD_WORKERS', 8))
        tasks = [(i, pkls) for i, (_, pkls) in enumerate(self.data_list)]

        msg = f"[{self.mode}] preloading {len(tasks)} samples with {n_workers} workers..."
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

        n_ok = n_bad = 0
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_load_one_sample, t) for t in tasks]
            pbar = tqdm(as_completed(futures),
                        total=len(futures),
                        desc=f"[{self.mode}] preload pkls",
                        ncols=100)
            for fut in pbar:
                idx, pose, expr, err = fut.result()
                if err is None:
                    self.pose_cache[idx] = pose
                    self.expr_cache[idx] = expr
                    n_ok += 1
                else:
                    n_bad += 1
                    if self.logger is not None:
                        self.logger.info(f"[preload] idx={idx} failed: {err}")

        # Drop entries that failed to load so __getitem__ never returns garbage
        if n_bad > 0:
            keep = [i for i, p in enumerate(self.pose_cache) if p is not None]
            self.data_list  = [self.data_list[i]  for i in keep]
            self.pose_cache = [self.pose_cache[i] for i in keep]
            self.expr_cache = [self.expr_cache[i] for i in keep]

        n_bytes = sum(p.nbytes + e.nbytes
                      for p, e in zip(self.pose_cache, self.expr_cache))
        msg = (f"[{self.mode}] preload done: ok={n_ok}, bad={n_bad}, "
               f"~{n_bytes / 1e9:.2f} GB in RAM, "
               f"{len(self.data_list)} samples retained")
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    # ==================== Frame Loading ====================

    @staticmethod
    def _load_pose_from_pkls(pkl_paths):
        """
        Load per-frame pkl files and assemble into:
            pose: (T, 53, 3) axis-angle
            expr: (T, 10)    expression blendshape coefficients

        Neural Sign Actors per-frame pkl layout:
            smplx_root_pose:  (3,)   → joint 0
            smplx_body_pose:  (63,)  → joints 1-21  (21 joints)
            smplx_lhand_pose: (45,)  → joints 22-36 (15 joints)
            smplx_rhand_pose: (45,)  → joints 37-51 (15 joints)
            smplx_jaw_pose:   (3,)   → joint 52
            smplx_expr:       (10,)  → expression blendshape coefficients
        """
        pose_frames = []
        expr_frames = []
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
            pose_frames.append(frame)

            # Expression: use zeros if key missing
            if 'smplx_expr' in d:
                expr = np.array(d['smplx_expr']).reshape(10,).astype(np.float32)
            else:
                expr = np.zeros(10, dtype=np.float32)
            expr_frames.append(expr)

        pose = np.stack(pose_frames, axis=0).astype(np.float32)  # (T, 53, 3)
        expr = np.stack(expr_frames, axis=0).astype(np.float32)  # (T, 10)
        return pose, expr

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

    def _process_sequence(self, joint_seq, expr_seq=None):
        """
        Process a raw (T, 53, 3) axis-angle sequence into the final feature tensor.

        Pipeline:
            1. Select joints (53 → 44 if upper body)
            2. Root-relative normalization
            3. Convert to 6D rotation (if enabled)
            4. Flatten pose to (T, pose_dim)
            5. Concatenate expression (T, 10) if use_expression

        Args:
            joint_seq: np.ndarray (T, 53, 3)
            expr_seq:  np.ndarray (T, 10) or None

        Returns:
            torch.Tensor (T, input_dim)
        """
        seq = torch.from_numpy(joint_seq)  # (T, 53, 3)

        # Step 1: Select joints
        seq = seq[:, self.joint_indices, :]  # (T, N_joints, 3)

        # Step 2: Root-relative normalization
        if self.cfg.ROOT_NORMALIZE:
            seq[:, 0:1, :] = 0.0

        # Step 3: Convert to 6D if enabled
        if self.use_rot6d:
            seq = axis_angle_to_rot6d(seq)  # (T, N_joints, 6)

        # Step 4: Flatten pose
        T = seq.shape[0]
        seq = seq.reshape(T, -1)  # (T, pose_dim)

        # Step 5: Append expression if enabled
        if self.use_expression and expr_seq is not None:
            expr = torch.from_numpy(expr_seq)  # (T, 10)
            seq  = torch.cat([seq, expr], dim=-1)  # (T, pose_dim + 10)

        return seq

    # ==================== Main Interface ====================

    def __len__(self):
        return len(self.data_list)

    # `self._bad_indices` is initialised in __init__ (instance-level — train
    # and val dataset instances must NOT share this set).

    def _load_one(self, idx):
        """Load + process a single sample. Raises on failure.

        Returns (seq, sentence, actual_len, gt_joints44).
        gt_joints44 is a (target_seq_len, 44, 3) tensor when self.use_fk_cache,
        else None.
        """
        sentence, source = self.data_list[idx]

        cached_pose = (self.pose_cache[idx]
                       if self.pose_cache is not None else None)
        if cached_pose is not None:
            total_len = cached_pose.shape[0]
            indices   = self._sample_indices(total_len)
            pose      = cached_pose[indices]
            expr      = self.expr_cache[idx][indices]
        elif isinstance(source, str) and source.endswith('.npz'):
            d = np.load(source)
            full_pose = d['axis_angle']
            full_expr = d['expression']
            indices = self._sample_indices(full_pose.shape[0])
            pose = full_pose[indices].astype(np.float32)
            expr = full_expr[indices].astype(np.float32)
        else:
            indices       = self._sample_indices(len(source))
            selected_pkls = [source[i] for i in indices]
            pose, expr    = self._load_pose_from_pkls(selected_pkls)

        seq = self._process_sequence(pose, expr if self.use_expression else None)

        # Reject NaN/Inf upstream of the network so loss can never go non-finite.
        if not torch.isfinite(seq).all():
            raise ValueError(f"non-finite values after _process_sequence (idx={idx})")

        actual_len = seq.shape[0]
        if actual_len < self.target_seq_len:
            pad = torch.zeros(self.target_seq_len - actual_len, seq.shape[-1])
            seq = torch.cat([seq, pad], dim=0)

        gt_joints44 = None
        if self.use_fk_cache:
            if isinstance(source, str) and source.endswith('.npz'):
                sname = os.path.splitext(os.path.basename(source))[0]
            else:
                # legacy pkl mode: source is a list of per-frame paths under
                # {root}/{mode}_poses/poses/{sname}/...
                sname = os.path.basename(os.path.dirname(source[0]))
            fk_path = os.path.join(self.fk_joints_dir, f'{sname}.npz')
            d_fk = np.load(fk_path)
            full_joints = d_fk['joints44']  # (T_full, 44, 3) float32
            sliced = full_joints[indices].astype(np.float32)  # (actual_len, 44, 3)
            gt_joints44 = torch.from_numpy(sliced)
            if gt_joints44.shape[0] < self.target_seq_len:
                pad_j = torch.zeros(
                    self.target_seq_len - gt_joints44.shape[0], 44, 3,
                    dtype=gt_joints44.dtype,
                )
                gt_joints44 = torch.cat([gt_joints44, pad_j], dim=0)

        return seq, sentence, actual_len, gt_joints44

    def __getitem__(self, idx):
        # Fast path: try the requested index. If it fails (corrupt pkl / NaN /
        # whatever), blacklist it and fall through to a deterministic safe
        # neighbour rather than returning zeros — zeros + actual_len=0 cause
        # downstream NaN loss when an entire micro-batch happens to be invalid.
        original_idx = idx
        if idx in self._bad_indices:
            idx = self._safe_neighbour(idx)

        for attempt in range(8):
            try:
                seq, sentence, actual_len, gt_joints44 = self._load_one(idx)
                if self.use_fk_cache:
                    return seq, sentence, sentence, actual_len, gt_joints44
                return seq, sentence, sentence, actual_len
            except Exception as e:
                self._bad_indices.add(idx)
                if self.logger is not None and attempt == 0:
                    self.logger.info(
                        f"[__getitem__] idx={idx} failed: {e} -> using neighbour"
                    )
                idx = self._safe_neighbour(idx)

        # Should never reach here; if it does, surface the problem loudly.
        raise RuntimeError(
            f"__getitem__ failed 8× starting from idx={original_idx}. "
            f"Bad-index set has {len(self._bad_indices)} entries."
        )

    def _safe_neighbour(self, idx):
        """Pick a deterministic non-blacklisted index near idx."""
        N = len(self.data_list)
        for delta in range(1, N):
            for cand in (idx + delta, idx - delta):
                cand %= N
                if cand not in self._bad_indices:
                    return cand
        # Everything blacklisted — give up.
        return idx


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