"""
MDM on the pooled SMPL-X fits with DOWN-WEIGHTED fingers and face (2026-09-19).

Same protocol as trainMotionDiffusion_pooled.py (which stays untouched, like trainMotionDiffusion.py). The only change is
the per-group loss weights: fingers are fitted from low-resolution video and are the least reliable part of the data, the
face even more so (user rule 2026-09-17, same ratios as the VQ-VAE v2 run: fingers x0.5, face x0.25).

    group   v1 weight   this file (defaults)
    TORSO     0.5         0.5
    ARMS      5.0         5.0      (shoulders, elbows, wrists: unchanged)
    LHAND     5.0         2.5      (15 finger joints)
    RHAND     5.0         2.5
    JAW       0.1         0.025
The same weights are used for the reconstruction and the velocity term. The reported loss is therefore NOT comparable with
the v1 runs; compare models with the collapse diagnostic (tools/verify_mdm_pooled.py), not with the training loss.

Usage: identical to trainMotionDiffusion_pooled.py, plus optional --w_hand / --w_jaw / --w_arms / --w_torso.
"""
import sys
import argparse

import torch.nn.functional as F

import trainMotionDiffusion as base
from utils.rotation_conversion import get_joint_slices


def parse_extra():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--target_seq_len", type=int, default=100)
    p.add_argument("--filter_words_min", type=int, default=3)
    p.add_argument("--filter_words_max", type=int, default=30)
    p.add_argument("--run_tag", type=str, default="")
    p.add_argument("--w_torso", type=float, default=0.5)
    p.add_argument("--w_arms", type=float, default=5.0)
    p.add_argument("--w_hand", type=float, default=2.5)
    p.add_argument("--w_jaw", type=float, default=0.025)
    extra, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return extra


if __name__ == "__main__":
    extra = parse_extra()

    class PooledMDMConfig(base.Pooled_SMPLX_Config):
        def __init__(self):
            super().__init__()
            self.TARGET_SEQ_LEN = extra.target_seq_len
            self.MAX_SEQ_LEN = max(self.MAX_SEQ_LEN, extra.target_seq_len)
            self.FILTER_WORDS_MIN = extra.filter_words_min
            self.FILTER_WORDS_MAX = extra.filter_words_max
            self.LOSS_W_TORSO, self.LOSS_W_ARMS = extra.w_torso, extra.w_arms
            self.LOSS_W_HAND, self.LOSS_W_JAW = extra.w_hand, extra.w_jaw
            if extra.run_tag:
                self.PROJECT_NAME = f"{self.PROJECT_NAME}_{extra.run_tag}"

    class HWDiffusionTrainer(base.DiffusionTrainer):
        def compute_loss(self, x_0_pred, x_0, padding_mask):
            valid = ~padding_mask
            g = get_joint_slices(n_feats=6 if self.cfg.USE_ROT6D else 3)
            w = [(g['TORSO'], self.cfg.LOSS_W_TORSO), (g['ARMS'], self.cfg.LOSS_W_ARMS), (g['LHAND'], self.cfg.LOSS_W_HAND),
                 (g['RHAND'], self.cfg.LOSS_W_HAND), (g['JAW'], self.cfg.LOSS_W_JAW)]  # ROOT and LOWER_BODY stay at 0 as in v1

            def masked(pred, gt, m):
                mse = F.mse_loss(pred, gt, reduction='none')
                mk = m.unsqueeze(-1).expand_as(mse).float()
                return (mse * mk).sum() / (mk.sum() + 1e-8)

            mse_loss = sum(wt * masked(x_0_pred[..., sl], x_0[..., sl], valid) for sl, wt in w)
            vel_gt, vel_pred = x_0[:, 1:] - x_0[:, :-1], x_0_pred[:, 1:] - x_0_pred[:, :-1]
            vel_valid = valid[:, 1:] & valid[:, :-1]
            vel_loss = sum(wt * masked(vel_pred[..., sl], vel_gt[..., sl], vel_valid) for sl, wt in w)
            return mse_loss + self.cfg.VEL_WEIGHT * vel_loss, mse_loss, vel_loss

    base.Pooled_SMPLX_Config = PooledMDMConfig
    args = base.parse_args()
    trainer = HWDiffusionTrainer(args)
    trainer.logger.info(f"[hw] loss weights torso={extra.w_torso} arms={extra.w_arms} hand={extra.w_hand} jaw={extra.w_jaw}")
    trainer.train()
