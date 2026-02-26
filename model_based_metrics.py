"""
model_based_metrics.py
======================
Model-based evaluation metrics for sign motion generation.
Trains a lightweight MLP classifier on GT active-joint features,
then uses it for Accuracy, FID, Diversity, Multimodality.

Metrics (same 4 as SignAvatar / Action2Motion protocol):
  1. Accuracy       — MLP classifier recognition rate
  2. FID            — Fréchet distance on learned features
  3. Diversity      — cross-class L2 on learned features
  4. Multimodality  — within-class L2 on learned features

Usage:
    from model_based_metrics import ModelBasedEvaluator

    evaluator = ModelBasedEvaluator(n_feats=3, num_classes=103)
    evaluator.train_classifier(gt_train_motions, gt_train_labels)
    results = evaluator.evaluate(gt_test_motions, gt_test_labels,
                                 gen_test_motions, gen_test_labels)
    evaluator.print_results(results, title="My Model — test")
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import linalg

from model_free_metrics import (
    ACTIVE_INDICES, N_ACTIVE,
    extract_active_joints, temporal_mean_pool,
    compute_diversity_multimodality,
)


# ============================================================================
# MLP Classifier
# ============================================================================

class MotionMLPClassifier(nn.Module):
    """
    Lightweight MLP classifier on temporally-pooled active-joint features.
    Returns both classification logits and intermediate features for FID.

    Architecture:
        input(D) → Linear(hidden) → ReLU → LayerNorm → Dropout
                 → Linear(hidden) → ReLU → LayerNorm
                 → Linear(num_classes)
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.classifier_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, D) temporally-pooled features

        Returns:
            dict: {"features": (B, hidden_dim), "yhat": (B, num_classes)}
        """
        feats = self.feature_net(x)
        logits = self.classifier_head(feats)
        return {"features": feats, "yhat": logits}


# ============================================================================
# Classifier training
# ============================================================================

def train_classifier(train_feats, train_labels, num_classes,
                     hidden_dim=256, epochs=100, lr=1e-3, device='cuda'):
    """
    Train an MLP classifier on GT active-joint features.

    Args:
        train_feats:  (N, D) tensor — temporally-pooled features
        train_labels: (N,) long tensor — class labels
        num_classes:  number of gloss classes
        hidden_dim:   MLP hidden dimension
        epochs:       training epochs
        lr:           learning rate
        device:       'cuda' or 'cpu'

    Returns:
        trained MotionMLPClassifier (eval mode), best_train_accuracy
    """
    if isinstance(train_feats, np.ndarray):
        train_feats = torch.tensor(train_feats, dtype=torch.float32)
    if isinstance(train_labels, np.ndarray):
        train_labels = torch.tensor(train_labels, dtype=torch.long)

    input_dim = train_feats.shape[1]
    clf = MotionMLPClassifier(input_dim, hidden_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(train_feats.to(device), train_labels.to(device))
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    clf.train()
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for xb, yb in loader:
            out = clf(xb)
            loss = criterion(out["yhat"], yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (out["yhat"].argmax(1) == yb).sum().item()
            total += xb.size(0)
        scheduler.step()
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in clf.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"    Classifier epoch {epoch+1:3d}/{epochs}: "
                  f"loss={total_loss/total:.4f}, acc={acc:.4f}")

    clf.load_state_dict(best_state)
    clf.eval()
    print(f"    Classifier best train acc: {best_acc:.4f}")
    return clf, best_acc


# ============================================================================
# Metric helper functions
# ============================================================================

@torch.no_grad()
def classify_accuracy(feats, labels, classifier, device):
    """Run classifier on pre-computed features, return accuracy."""
    feats_dev = feats.to(device)
    labels_dev = labels.to(device)
    out = classifier(feats_dev)
    preds = out["yhat"].argmax(dim=1)
    return (preds == labels_dev).float().mean().item()


@torch.no_grad()
def extract_hidden_features(feats, classifier, device, batch_size=256):
    """Extract hidden features from classifier for FID/Diversity."""
    all_hidden = []
    for i in range(0, len(feats), batch_size):
        xb = feats[i:i+batch_size].to(device)
        out = classifier(xb)
        all_hidden.append(out["features"].cpu())
    return torch.cat(all_hidden, dim=0)


def compute_fid(feats1, feats2, eps=1e-6):
    """FID between two sets of feature vectors."""
    f1 = feats1.cpu().numpy() if isinstance(feats1, torch.Tensor) else feats1
    f2 = feats2.cpu().numpy() if isinstance(feats2, torch.Tensor) else feats2

    mu1, sigma1 = np.mean(f1, axis=0), np.cov(f1, rowvar=False)
    mu2, sigma2 = np.mean(f2, axis=0), np.cov(f2, rowvar=False)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2)
                 - 2 * np.trace(covmean))


# ============================================================================
# ModelBasedEvaluator — wraps training + all metrics
# ============================================================================

class ModelBasedEvaluator:
    """
    Train MLP on GT, then compute Accuracy/FID/Diversity/Multimodality.

    Usage:
        evaluator = ModelBasedEvaluator(n_feats=3, num_classes=103)

        # Step 1: train classifier on GT train set
        evaluator.train_classifier(gt_train_motions, gt_train_labels)

        # Step 2: evaluate on test set
        results = evaluator.evaluate(gt_test_motions, gt_test_labels,
                                     gen_test_motions, gen_test_labels)
    """

    def __init__(self, n_feats=3, num_classes=103,
                 hidden_dim=256, clf_epochs=100, lr=1e-3,
                 device='cuda', seed=42):
        self.n_feats = n_feats
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.clf_epochs = clf_epochs
        self.lr = lr
        self.device = device
        self.seed = seed
        self.classifier = None
        self.train_acc = None

    def _pool_motions(self, motions):
        """
        motions: (N, T, 53*n_feats) or (N, T, 53, n_feats)
        Returns: (N, N_ACTIVE * n_feats) pooled features
        """
        raw = extract_active_joints(motions, self.n_feats)
        return temporal_mean_pool(raw).reshape(raw.shape[0], -1)

    def train(self, gt_train_motions, gt_train_labels):
        """
        Train the MLP classifier on GT training data.

        Args:
            gt_train_motions: (N, T, ...) GT training motions
            gt_train_labels:  (N,) class labels
        """
        print("\n>>> Training MLP classifier on GT active joints ...")
        pooled = self._pool_motions(gt_train_motions)
        self.classifier, self.train_acc = train_classifier(
            pooled, gt_train_labels, self.num_classes,
            hidden_dim=self.hidden_dim, epochs=self.clf_epochs,
            lr=self.lr, device=self.device)
        return self.classifier

    def evaluate(self, gt_motions, gt_labels, gen_motions, gen_labels,
                 verbose=True):
        """
        Compute all 4 model-based metrics.

        Args:
            gt_motions:  (N_gt, T, ...)  GT test motions
            gt_labels:   (N_gt,)         GT labels
            gen_motions: (N_gen, T, ...) generated motions
            gen_labels:  (N_gen,)        gen labels

        Returns:
            dict with accuracy_gt/gen, fid_gt/gen, diversity_gt/gen,
                 multimodality_gt/gen, classifier_train_acc
        """
        assert self.classifier is not None, \
            "Call train() first to train the classifier!"

        gt_pooled = self._pool_motions(gt_motions)
        gen_pooled = self._pool_motions(gen_motions)
        results = {}

        # Classifier train accuracy (upper bound reference)
        results["classifier_train_acc"] = self.train_acc

        # --- Accuracy ---
        if verbose: print("  Computing accuracy ...")
        results["accuracy_gt"] = classify_accuracy(
            gt_pooled, gt_labels, self.classifier, self.device)
        results["accuracy_gen"] = classify_accuracy(
            gen_pooled, gen_labels, self.classifier, self.device)

        # --- Extract hidden features ---
        if verbose: print("  Extracting hidden features ...")
        h_gt = extract_hidden_features(
            gt_pooled, self.classifier, self.device)
        h_gen = extract_hidden_features(
            gen_pooled, self.classifier, self.device)

        # --- FID ---
        if verbose: print("  Computing FID ...")
        results["fid_gt"] = compute_fid(h_gt, h_gt)    # sanity: ~0
        results["fid_gen"] = compute_fid(h_gt, h_gen)

        # --- Diversity & Multimodality ---
        if verbose: print("  Computing diversity & multimodality ...")
        d_gt, m_gt = compute_diversity_multimodality(
            h_gt, gt_labels, self.num_classes, seed=self.seed)
        d_gen, m_gen = compute_diversity_multimodality(
            h_gen, gen_labels, self.num_classes, seed=self.seed)

        results["diversity_gt"]      = d_gt
        results["diversity_gen"]     = d_gen
        results["multimodality_gt"]  = m_gt
        results["multimodality_gen"] = m_gen

        if verbose: print("  Done.")
        return results

    @staticmethod
    def print_results(results, title=""):
        """Pretty-print results dict."""
        print(f"\n{'='*60}")
        print(f"  Model-Based Metrics — {title}")
        print(f"{'='*60}")
        if "classifier_train_acc" in results:
            print(f"  Classifier train accuracy (upper bound): "
                  f"{results['classifier_train_acc']:.4f}")
        print(f"\n  {'Metric':<25} {'GT':>12} {'Gen':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12}")
        print(f"  {'Accuracy ↑':<25} {results['accuracy_gt']:>12.4f} "
              f"{results['accuracy_gen']:>12.4f}")
        print(f"  {'FID ↓':<25} {results['fid_gt']:>12.4f} "
              f"{results['fid_gen']:>12.4f}")
        print(f"  {'Diversity →':<25} {results['diversity_gt']:>12.4f} "
              f"{results['diversity_gen']:>12.4f}")
        print(f"  {'Multimodality →':<25} {results['multimodality_gt']:>12.4f} "
              f"{results['multimodality_gen']:>12.4f}")
        print(f"{'='*60}\n")