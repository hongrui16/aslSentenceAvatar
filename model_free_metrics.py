"""
model_free_metrics.py
=====================
Model-free evaluation metrics for sign motion generation.
NO trained classifier needed — all metrics computed on raw joint rotations.

Metrics:
  1. Raw FID           — Fréchet distance on temporally-pooled active-joint features
  2. Velocity FID      — FID on frame-to-frame differences (temporal dynamics)
  3. k-NN Accuracy     — classify Gen using k-nearest-neighbor on GT gallery
  4. Diversity          — cross-class L2 variance in feature space
  5. Multimodality     — within-class L2 variance in feature space
  6. Variance Ratio    — per-joint-group Gen_var / GT_var (mode collapse detector)

Usage:
    from model_free_metrics import ModelFreeEvaluator

    evaluator = ModelFreeEvaluator(n_feats=3, num_classes=103)
    results = evaluator.evaluate(gt_raw, gt_labels, gen_raw, gen_labels)
    evaluator.print_results(results, title="My Model — test")
"""

import numpy as np
import torch
from scipy import linalg


# ============================================================================
# Joint definitions (SMPL-X 53-joint skeleton)
# ============================================================================
# Matches the project's upper body convention:
#   UPPER_BODY = [0, 3, 6, 9, 12-21, 22-36, 37-51]  (44 joints)
#   For evaluation we EXCLUDE root [0] and [3] (normalized to zero) → 42 joints.
#   This way both full-body and upper-body models are compared fairly
#   on the same joint set.
#
# Excluded joints:
#   [0]  root       — normalized to zero, would just add noise
#   [1]  left_hip   — lower body
#   [2]  right_hip  — lower body
#   [4]  left_knee  — lower body
#   [5]  right_knee — lower body
#   [7]  left_ankle — lower body
#   [8]  right_ankle— lower body
#   [10] left_foot  — lower body
#   [11] right_foot — lower body
#   [52] jaw        — excluded per SignAvatar convention
# ============================================================================

ROOT_INDICES       = [0]
LOWER_BODY_INDICES = [1, 2, 4, 5, 7, 8, 10, 11]
EXCLUDED_INDICES   = sorted(ROOT_INDICES + LOWER_BODY_INDICES + [52])  # root + lower + jaw

SPINE_INDICES      = [3, 6, 9]               # spine1, spine2, spine3
TORSO_INDICES      = [12, 13, 14, 15]        # neck, left_collar, right_collar, head
ARMS_INDICES       = [16, 17, 18, 19, 20, 21]
LHAND_INDICES      = list(range(22, 37))      # 15 joints
RHAND_INDICES      = list(range(37, 52))      # 15 joints

ACTIVE_INDICES = sorted(
    SPINE_INDICES + TORSO_INDICES + ARMS_INDICES + LHAND_INDICES + RHAND_INDICES
)  # 43 joints (upper body minus root)
N_ACTIVE = len(ACTIVE_INDICES)

# Joint groups for per-group variance ratio
JOINT_GROUPS = {
    "spine": SPINE_INDICES,
    "torso": TORSO_INDICES,
    "arms":  ARMS_INDICES,
    "lhand": LHAND_INDICES,
    "rhand": RHAND_INDICES,
}

# Map global joint index → position within ACTIVE_INDICES
_active_pos = {j: pos for pos, j in enumerate(ACTIVE_INDICES)}
JOINT_GROUPS_LOCAL = {
    name: [_active_pos[j] for j in joints]
    for name, joints in JOINT_GROUPS.items()
}


# ============================================================================
# Feature extraction helpers
# ============================================================================

def extract_active_joints(x, n_feats):
    """
    Extract only active joints from motion tensor.

    Args:
        x: (B, T, 53*n_feats) or (B, T, 53, n_feats)

    Returns:
        (B, T, N_ACTIVE, n_feats)
    """
    if x.dim() == 3:
        B, T, D = x.shape
        x = x.view(B, T, 53, n_feats)
    return x[:, :, ACTIVE_INDICES, :]


def temporal_mean_pool(x):
    """(B, T, ...) → (B, ...) via mean over time."""
    return x.mean(dim=1)


def compute_velocity(x):
    """
    Frame-to-frame velocity.
    (B, T, J, C) → (B, T-1, J, C)
    """
    return x[:, 1:] - x[:, :-1]


# ============================================================================
# Metric: FID
# ============================================================================

def compute_fid(feats1, feats2, eps=1e-6):
    """
    Fréchet Inception Distance between two (N, D) feature sets.

    Args:
        feats1: (N1, D) reference features (GT)
        feats2: (N2, D) generated features

    Returns:
        float: FID score (lower = better)
    """
    f1 = feats1.cpu().numpy() if isinstance(feats1, torch.Tensor) else feats1
    f2 = feats2.cpu().numpy() if isinstance(feats2, torch.Tensor) else feats2

    mu1, sig1 = np.mean(f1, 0), np.cov(f1, rowvar=False)
    mu2, sig2 = np.mean(f2, 0), np.cov(f2, rowvar=False)

    sig1 = np.atleast_2d(sig1)
    sig2 = np.atleast_2d(sig2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sig1 @ sig2, disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm(
            (sig1 + np.eye(len(sig1)) * eps) @ (sig2 + np.eye(len(sig2)) * eps)
        )
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sig1) + np.trace(sig2) - 2 * np.trace(covmean))


# ============================================================================
# Metric: Diversity & Multimodality
# ============================================================================

def compute_diversity_multimodality(feats, labels, num_classes,
                                    diversity_times=200,
                                    multimodality_times=20, seed=None):
    """
    Diversity:       avg L2 between random pairs across all classes.
    Multimodality:   avg L2 between random same-class pairs.

    Args:
        feats:  (N, D) feature vectors
        labels: (N,)   class labels
        num_classes: total number of classes

    Returns:
        (diversity, multimodality) tuple of floats
    """
    if seed is not None:
        np.random.seed(seed)

    labels = labels.long()
    n = len(labels)

    # --- Diversity: random pairs ---
    i1 = np.random.randint(0, n, diversity_times)
    i2 = np.random.randint(0, n, diversity_times)
    diversity = sum(torch.dist(feats[a], feats[b]) for a, b in zip(i1, i2))
    diversity /= diversity_times

    # --- Multimodality: same-class pairs ---
    multimodality = 0.0
    quotas = np.repeat(multimodality_times, num_classes)
    while np.any(quotas > 0):
        a = np.random.randint(0, n)
        la = labels[a].item()
        if quotas[la] <= 0:
            continue
        b = np.random.randint(0, n)
        while labels[b].item() != la:
            b = np.random.randint(0, n)
        quotas[la] -= 1
        multimodality += torch.dist(feats[a], feats[b])
    multimodality /= (multimodality_times * num_classes)

    return diversity.item(), multimodality.item()


# ============================================================================
# Metric: k-NN Accuracy
# ============================================================================

def knn_accuracy(gen_feats, gen_labels, gt_feats, gt_labels, k=5):
    """
    For each generated sample, find k nearest GT neighbors → majority vote.
    No trained model needed.

    Args:
        gen_feats:  (N_gen, D)
        gen_labels: (N_gen,)
        gt_feats:   (N_gt, D)
        gt_labels:  (N_gt,)
        k: number of neighbors

    Returns:
        accuracy: float in [0, 1]
    """
    correct = 0
    chunk_size = 256
    n_gen = len(gen_feats)

    for start in range(0, n_gen, chunk_size):
        end = min(start + chunk_size, n_gen)
        gf = gen_feats[start:end]
        gl = gen_labels[start:end]

        dists = torch.cdist(gf, gt_feats)
        _, topk_idx = dists.topk(k, dim=1, largest=False)
        topk_labels = gt_labels[topk_idx]

        for i in range(len(gf)):
            counts = torch.bincount(topk_labels[i], minlength=gt_labels.max() + 1)
            pred = counts.argmax().item()
            if pred == gl[i].item():
                correct += 1

    return correct / n_gen


# ============================================================================
# Metric: Per-joint-group Variance Ratio
# ============================================================================

def compute_variance_ratio(gt_active, gen_active, n_feats):
    """
    For each joint group:
        ratio = mean(Gen per-feature variance) / mean(GT per-feature variance)

    ratio < 0.5 → mode collapse
    ratio > 2.0 → over-diverse / noisy
    ratio ≈ 1.0 → well matched

    Args:
        gt_active:  (N, T, N_ACTIVE, n_feats)
        gen_active: (N, T, N_ACTIVE, n_feats)
        n_feats: 3 or 6

    Returns:
        dict: {group_name: {"gt_var", "gen_var", "ratio"}}
    """
    results = {}
    for group_name, local_indices in JOINT_GROUPS_LOCAL.items():
        gt_g = gt_active[:, :, local_indices, :]
        gen_g = gen_active[:, :, local_indices, :]

        gt_var = gt_g.reshape(-1, len(local_indices) * n_feats).var(dim=0).mean().item()
        gen_var = gen_g.reshape(-1, len(local_indices) * n_feats).var(dim=0).mean().item()

        ratio = gen_var / (gt_var + 1e-10)
        results[group_name] = {
            "gt_var":  round(gt_var, 6),
            "gen_var": round(gen_var, 6),
            "ratio":   round(ratio, 4),
        }
    return results


# ============================================================================
# ModelFreeEvaluator — wraps all metrics into a single class
# ============================================================================

class ModelFreeEvaluator:
    """
    Compute all model-free metrics given raw motion tensors.

    Usage:
        evaluator = ModelFreeEvaluator(n_feats=3, num_classes=103)

        # Prepare data: (N, T, 53*n_feats) or (N, T, 53, n_feats)
        results = evaluator.evaluate(gt_motions, gt_labels, gen_motions, gen_labels)
        evaluator.print_results(results, "My Model")
    """

    def __init__(self, n_feats=3, num_classes=103, knn_k=5, seed=42):
        self.n_feats = n_feats
        self.num_classes = num_classes
        self.knn_k = knn_k
        self.seed = seed

    def _prepare(self, motions):
        """
        motions: (N, T, 53*n_feats) or (N, T, 53, n_feats)

        Returns:
            raw_active: (N, T, N_ACTIVE, n_feats)
            pooled:     (N, N_ACTIVE * n_feats)
        """
        raw = extract_active_joints(motions, self.n_feats)  # (N, T, 43, nf)
        pooled = temporal_mean_pool(raw).reshape(raw.shape[0], -1)  # (N, 43*nf)
        return raw, pooled

    def evaluate(self, gt_motions, gt_labels, gen_motions, gen_labels,
                 verbose=True):
        """
        Run all 6 model-free metrics.

        Args:
            gt_motions:  (N_gt, T, ...) GT motion tensor
            gt_labels:   (N_gt,) class indices
            gen_motions: (N_gen, T, ...) generated motion tensor
            gen_labels:  (N_gen,) class indices

        Returns:
            dict with all metric results
        """
        gt_raw, gt_pooled = self._prepare(gt_motions)
        gen_raw, gen_pooled = self._prepare(gen_motions)
        results = {}

        # 1. Raw FID
        if verbose: print("  [1/6] Raw FID ...")
        results["raw_fid"] = compute_fid(gt_pooled, gen_pooled)

        # 2 & 3. Diversity + Multimodality
        if verbose: print("  [2/6] Diversity & Multimodality ...")
        d_gt, m_gt = compute_diversity_multimodality(
            gt_pooled, gt_labels, self.num_classes, seed=self.seed)
        d_gen, m_gen = compute_diversity_multimodality(
            gen_pooled, gen_labels, self.num_classes, seed=self.seed)
        results["diversity_gt"]      = d_gt
        results["diversity_gen"]     = d_gen
        results["multimodality_gt"]  = m_gt
        results["multimodality_gen"] = m_gen

        # 4. k-NN Accuracy
        if verbose: print("  [3/6] k-NN Accuracy ...")
        results["knn_accuracy_gt"]  = knn_accuracy(
            gt_pooled, gt_labels, gt_pooled, gt_labels, k=self.knn_k)
        results["knn_accuracy_gen"] = knn_accuracy(
            gen_pooled, gen_labels, gt_pooled, gt_labels, k=self.knn_k)

        # 5. Velocity FID
        if verbose: print("  [4/6] Velocity FID ...")
        gt_vel = compute_velocity(gt_raw)
        gen_vel = compute_velocity(gen_raw)
        gt_vel_p = temporal_mean_pool(gt_vel).reshape(gt_vel.shape[0], -1)
        gen_vel_p = temporal_mean_pool(gen_vel).reshape(gen_vel.shape[0], -1)
        results["velocity_fid"] = compute_fid(gt_vel_p, gen_vel_p)

        # 6. Variance ratio per joint group
        if verbose: print("  [5/6] Variance ratio ...")
        n_min = min(gt_raw.shape[0], gen_raw.shape[0])
        results["variance_ratio"] = compute_variance_ratio(
            gt_raw[:n_min], gen_raw[:n_min], self.n_feats)

        if verbose: print("  [6/6] Done.")
        return results

    @staticmethod
    def print_results(results, title=""):
        """Pretty-print results dict."""
        print(f"\n{'='*65}")
        print(f"  Model-Free Metrics — {title}")
        print(f"{'='*65}")

        print(f"\n  Distribution Metrics:")
        print(f"  {'Metric':<25} {'GT':>12} {'Gen':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12}")
        print(f"  {'Raw FID ↓':<25} {'—':>12} {results['raw_fid']:>12.2f}")
        print(f"  {'Velocity FID ↓':<25} {'—':>12} {results['velocity_fid']:>12.2f}")
        print(f"  {'k-NN Accuracy ↑':<25} {results['knn_accuracy_gt']:>12.4f} {results['knn_accuracy_gen']:>12.4f}")
        print(f"  {'Diversity →':<25} {results['diversity_gt']:>12.4f} {results['diversity_gen']:>12.4f}")
        print(f"  {'Multimodality →':<25} {results['multimodality_gt']:>12.4f} {results['multimodality_gen']:>12.4f}")

        print(f"\n  Variance Ratio (Gen/GT — ideal ≈ 1.0):")
        print(f"  {'Joint Group':<15} {'GT Var':>12} {'Gen Var':>12} {'Ratio':>8}")
        print(f"  {'-'*15} {'-'*12} {'-'*12} {'-'*8}")
        for group, vals in results["variance_ratio"].items():
            flag = ""
            if vals["ratio"] < 0.5:
                flag = " ⚠ collapse"
            elif vals["ratio"] > 2.0:
                flag = " ⚠ noisy"
            print(f"  {group:<15} {vals['gt_var']:>12.6f} {vals['gen_var']:>12.6f} "
                  f"{vals['ratio']:>8.3f}{flag}")

        print(f"{'='*65}\n")