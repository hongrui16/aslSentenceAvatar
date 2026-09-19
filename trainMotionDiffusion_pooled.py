"""
MDM on the pooled SMPL-X fits (generation baseline for the retrieval pipeline).

Thin wrapper around trainMotionDiffusion.py, which stays untouched. It only adds the
protocol knobs the pooled regression run already used (trainMotionRegression_clean.py):
    --target_seq_len 100 --filter_words_min 3 --filter_words_max 30 --run_tag <group>
All other flags are passed through to trainMotionDiffusion.parse_args().

Usage:
    python trainMotionDiffusion_pooled.py --dataset PooledSMPLX \
        --pool_train <subsets>/pool_full.txt --pool_test <subsets>/test_fixed.txt \
        --run_tag pool_full --target_seq_len 100 --filter_words_min 3 --filter_words_max 30 \
        --batch_size 64 --epochs 45
"""
import sys
import argparse

import trainMotionDiffusion as base


def parse_extra():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--target_seq_len", type=int, default=100)
    p.add_argument("--filter_words_min", type=int, default=3)
    p.add_argument("--filter_words_max", type=int, default=30)
    p.add_argument("--run_tag", type=str, default="")
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
            if extra.run_tag:
                self.PROJECT_NAME = f"{self.PROJECT_NAME}_{extra.run_tag}"

    base.Pooled_SMPLX_Config = PooledMDMConfig
    args = base.parse_args()
    trainer = base.DiffusionTrainer(args)
    trainer.train()
