"""
ASL Avatar Diffusion Training Script (V3)

Trains MotionDiffusionModel (MDM-style) instead of CVAE.

Key differences from train_v2.py (CVAE):
    - No KL loss, no reparameterization
    - No curriculum masking (diffusion handles this via noise levels)
    - Training: sample t, add noise to x_0, model predicts x_0, MSE + velocity loss
    - Much simpler loss computation

Usage:
    python train_v3.py --dataset WLASL_SMPLX --use_rot6d --use_upper_body
    python train_v3.py --dataset WLASL_SMPLX --use_rot6d --use_upper_body --debug
    python train_v3.py --dataset WLASL_SMPLX --use_rot6d --use_upper_body --use_mini_dataset
"""

import os
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger

from dataloader.SignBankSMPLXDataset import SignBankSMPLXDataset
from dataloader.WLASLSMPLXDataset import WLASLSMPLXDataset
from dataloader.WLASLSMPLXDatasetV2 import WLASLSMPLXDatasetV2
from dataloader.SignBankSMPLXDatasetV2 import SignBankSMPLXDatasetV2

from aslAvatarModel_v5 import MotionDiffusionModel

from utils.utils import plot_training_curves, backup_code, collate_fn, create_padding_mask
from config import SignBank_SMPLX_Config, WLASL_SMPLX_Config
from utils.rotation_conversion import get_joint_slices


class DiffusionTrainer:
    """Diffusion training with Accelerate support"""

    def __init__(self, args):
        self.args = args
        self.debug = args.debug

        # ---- Config ----
        if args.dataset == "SignBank_SMPLX":
            Config = SignBank_SMPLX_Config
        elif args.dataset == "WLASL_SMPLX":
            Config = WLASL_SMPLX_Config
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        self.cfg = Config()
        self.cfg.DATASET_NAME = args.dataset
        self.cfg.USE_UPPER_BODY = args.use_upper_body
        self.cfg.USE_ROT6D = args.use_rot6d
        self.cfg.USE_MINI_DATASET = args.use_mini_dataset
        self.cfg.USE_LABEL_INDEX_COND = args.use_label_index_cond
        self.cfg.PROJECT_NAME = "ASLAvatar_Diffusion"
        self.cfg.ROOT_NORMALIZE = not args.no_root_normalize
        self.cfg.MODEL_VERSION = 'v5'

        # Diffusion-specific config with defaults
        self.cfg.NUM_DIFFUSION_STEPS = getattr(self.cfg, 'NUM_DIFFUSION_STEPS', 1000)
        self.cfg.VEL_WEIGHT = getattr(self.cfg, 'VEL_WEIGHT', 1.0)
        self.cfg.N_FEATS = 6 if self.cfg.USE_ROT6D else 3

        if args.batch_size:
            self.cfg.TRAIN_BSZ = args.batch_size
        if args.epochs:
            self.cfg.MAX_EPOCHS = args.epochs
        if args.lr:
            self.cfg.LEARNING_RATE = args.lr

        # ---- Directories ----
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        slurm_id = os.getenv('SLURM_JOB_ID')
        if slurm_id:
            timestamp += f"_job{slurm_id}"
        if self.debug:
            timestamp = "debug_" + timestamp

        self.logging_dir = os.path.join(
            self.cfg.LOG_DIR, self.cfg.PROJECT_NAME, self.cfg.DATASET_NAME, timestamp
        )
        if self.debug:
            self.ckpt_dir = self.logging_dir
        else:
            self.ckpt_dir = os.path.join(
                self.cfg.CKPT_DIR, self.cfg.PROJECT_NAME, self.cfg.DATASET_NAME, timestamp
            )

        # ---- Accelerator ----
        acc_config = ProjectConfiguration(project_dir=self.logging_dir, logging_dir=self.logging_dir)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.GRAD_ACCUM,
            mixed_precision=self.cfg.MIXED_PRECISION,
            project_config=acc_config,
        )
        self.device = self.accelerator.device

        if self.accelerator.is_main_process:
            os.makedirs(self.logging_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir, exist_ok=True)

        self._setup_logging()
        self._build_components()

        # Backup code
        if self.accelerator.is_main_process:
            src_dir = os.path.dirname(os.path.abspath(__file__))
            dst_dir = os.path.join(self.logging_dir, 'code_backup')
            os.makedirs(dst_dir, exist_ok=True)
            backup_code(project_root=src_dir, backup_dir=dst_dir, logger=self.logger)

        self.global_step = 0
        self.best_loss = float('inf')
        self.start_epoch = 0

        if args.resume:
            self.load_checkpoint(args.resume, finetune=args.finetune)

        # Log config
        self.logger.info("Config:")
        for k, v in vars(self.cfg).items():
            self.logger.info(f"  {k}: {v}")
        self.logger.info("-------------------------------\n")

    # ================================================================== setup
    def _setup_logging(self):
        log_file = os.path.join(self.logging_dir, "train.log")
        handlers = []
        if self.accelerator.is_main_process:
            handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=handlers,
        )
        self.logger = get_logger(__name__)
        if self.accelerator.is_main_process:
            self.logger.info(f"Logging dir: {self.logging_dir}")
            self.logger.info(f"Checkpoint dir: {self.ckpt_dir}")

    def _build_components(self):
        self.logger.info("Building components...")

        # Dataset
        if self.cfg.DATASET_NAME == "SignBank_SMPLX":
            if self.cfg.DATASET_VERSION.lower() == 'v1':
                train_dataset = SignBankSMPLXDataset(mode='train', cfg=self.cfg, logger=self.logger)
                test_dataset = None
            elif self.cfg.DATASET_VERSION.lower() == 'v2':
                train_dataset = SignBankSMPLXDatasetV2(mode='train', cfg=self.cfg, logger=self.logger)
                test_dataset = None
        elif self.cfg.DATASET_NAME == "WLASL_SMPLX":
            if self.cfg.DATASET_VERSION.lower() == 'v1':
                train_dataset = WLASLSMPLXDataset(mode='train', cfg=self.cfg, logger=self.logger)
                test_dataset = WLASLSMPLXDataset(mode='test', cfg=self.cfg, logger=self.logger)
            elif self.cfg.DATASET_VERSION.lower() == 'v2':
                train_dataset = WLASLSMPLXDatasetV2(mode='train', cfg=self.cfg, logger=self.logger)
                test_dataset = WLASLSMPLXDatasetV2(mode='test', cfg=self.cfg, logger=self.logger,
                                                     gloss_names=train_dataset.gloss_name_list)
        else:
            raise ValueError(f"Unknown dataset: {self.cfg.DATASET_NAME}")

        if self.cfg.TRAIN_BSZ > len(train_dataset):
            self.cfg.TRAIN_BSZ = len(train_dataset) // 2

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.cfg.TRAIN_BSZ, shuffle=True,
            collate_fn=collate_fn, num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True, drop_last=True,
        )
        self.logger.info(f"Train: {len(train_dataset)} samples, {len(self.train_loader)} batches")

        if test_dataset is not None:
            self.test_loader = DataLoader(
                test_dataset, batch_size=self.cfg.EVAL_BSZ, shuffle=False,
                collate_fn=collate_fn, num_workers=self.cfg.NUM_WORKERS, pin_memory=True,
            )
            self.logger.info(f"Test: {len(test_dataset)} samples")
        else:
            self.test_loader = None

        # Store dataset info in config
        self.cfg.INPUT_DIM = train_dataset.input_dim
        self.cfg.GLOSS_NAME_LIST = train_dataset.gloss_name_list
        self.cfg.NUM_CLASSES = len(self.cfg.GLOSS_NAME_LIST)
        


        # Model
        self.logger.info("Building MotionDiffusionModel...")
        self.model = MotionDiffusionModel(self.cfg)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total params: {total_params:,}")
        self.logger.info(f"Trainable params: {trainable_params:,}")

        # Optimizer
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=self.cfg.LEARNING_RATE, weight_decay=self.cfg.WEIGHT_DECAY,
        )

        # LR scheduler
        num_training_steps = self.cfg.MAX_EPOCHS * len(self.train_loader)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_training_steps, eta_min=self.cfg.LEARNING_RATE * 0.01,
        )

        # Accelerate prepare
        (self.model, self.optimizer, self.train_loader, self.lr_scheduler) = \
            self.accelerator.prepare(self.model, self.optimizer, self.train_loader, self.lr_scheduler)

        if self.test_loader is not None:
            self.test_loader = self.accelerator.prepare(self.test_loader)

    # ================================================================== loss
    def compute_loss(self, x_0_pred, x_0, padding_mask):
        """
        Diffusion loss: MSE(x_0_pred, x_0) + velocity loss.
        No KL term — diffusion doesn't need it.
        """
        valid_mask = ~padding_mask  # (B, T)
        n_feats = 6 if self.cfg.USE_ROT6D else 3


        def masked_mse(pred, gt):
            mse = F.mse_loss(pred, gt, reduction='none')
            mask = valid_mask.unsqueeze(-1).expand_as(mse).float()
            return (mse * mask).sum() / (mask.sum() + 1e-8)

        joint_groups = get_joint_slices(n_feats=n_feats)

        ROOT = joint_groups['ROOT']
        LOWER_BODY = joint_groups['LOWER_BODY']
        TORSO      = joint_groups['TORSO']
        ARMS       = joint_groups['ARMS']
        LHAND      = joint_groups['LHAND']
        RHAND      = joint_groups['RHAND']
        JAW        = joint_groups['JAW']
        
        # Weighted reconstruction
        mse_loss = (0.5 * masked_mse(x_0_pred[..., TORSO],  x_0[..., TORSO])
                  + 0.0 * masked_mse(x_0_pred[..., ROOT],  x_0[..., ROOT])
                  + 0.0 * masked_mse(x_0_pred[..., LOWER_BODY],  x_0[..., LOWER_BODY])
                  + 5.0 * masked_mse(x_0_pred[..., ARMS], x_0[..., ARMS])
                  + 5.0 * masked_mse(x_0_pred[..., LHAND], x_0[..., LHAND])
                  + 5.0 * masked_mse(x_0_pred[..., RHAND], x_0[..., RHAND])
                  + 0.1 * masked_mse(x_0_pred[..., JAW],   x_0[..., JAW]))

        # Weighted velocity
        vel_gt   = x_0[:, 1:] - x_0[:, :-1]
        vel_pred = x_0_pred[:, 1:] - x_0_pred[:, :-1]
        vel_valid = valid_mask[:, 1:] & valid_mask[:, :-1]

        def masked_vel(pred, gt):
            mse = F.mse_loss(pred, gt, reduction='none')
            mask = vel_valid.unsqueeze(-1).expand_as(mse).float()
            return (mse * mask).sum() / (mask.sum() + 1e-8)

        vel_loss = (0.5 * masked_vel(vel_pred[..., TORSO],  vel_gt[..., TORSO])
                  + 0.0 * masked_vel(vel_pred[..., ROOT],  vel_gt[..., ROOT])
                  + 0.0 * masked_vel(vel_pred[..., LOWER_BODY],  vel_gt[..., LOWER_BODY])
                  + 5.0 * masked_vel(vel_pred[..., ARMS],  vel_gt[..., ARMS])
                  + 5.0 * masked_vel(vel_pred[..., LHAND], vel_gt[..., LHAND])
                  + 5.0 * masked_vel(vel_pred[..., RHAND], vel_gt[..., RHAND])
                  + 0.1 * masked_vel(vel_pred[..., JAW],   vel_gt[..., JAW]))

        total = mse_loss + self.cfg.VEL_WEIGHT * vel_loss
        return total, mse_loss, vel_loss

    # ================================================================== train epoch
    def train_epoch(self, epoch):
        self.model.train()
        unwrapped = self.accelerator.unwrap_model(self.model)

        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_vel = 0.0
        num_batches = 0

        num_prints = 50
        total_steps = len(self.train_loader)
        print_every = max(total_steps // num_prints, 1)
        progress_bar = tqdm(total=num_prints,
                            disable=not self.accelerator.is_local_main_process,
                            desc=f"Epoch {epoch}")

        for step, batch in enumerate(self.train_loader):
            if step % print_every == 0 and progress_bar.n < num_prints:
                progress_bar.update(1)

            with self.accelerator.accumulate(self.model):
                motion, gloss, lengths = batch
                B, T, _ = motion.shape

                # Condition
                if self.cfg.USE_LABEL_INDEX_COND:
                    cond = torch.tensor(
                        [self.cfg.GLOSS_NAME_LIST.index(gl) for gl in gloss],
                        dtype=torch.long,
                    ).to(motion.device)
                else:
                    cond = list(gloss)

                padding_mask = create_padding_mask(lengths, T, self.device)

                # ---- Diffusion training step ----
                # Sample random timestep per sample
                t = torch.randint(0, unwrapped.num_timesteps, (B,), device=motion.device)

                # Add noise
                noise = torch.randn_like(motion)
                x_t = unwrapped.q_sample(motion, t, noise)

                # Predict x_0
                x_0_pred = self.model(x_t, t, cond, padding_mask, motion)

                # Loss
                loss, mse, vel = self.compute_loss(x_0_pred, motion, padding_mask)

                # Backward
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                epoch_mse += mse.item()
                epoch_vel += vel.item()
                num_batches += 1
                self.global_step += 1

                if step == 0:
                    mem = torch.cuda.max_memory_allocated(self.device) / 2**30
                    self.logger.info(f"[epoch {epoch}] Peak GPU: {mem:.2f} GB")

                if self.debug and step >= 10:
                    break

        progress_bar.close()
        return {
            'loss': epoch_loss / max(num_batches, 1),
            'mse': epoch_mse / max(num_batches, 1),
            'vel': epoch_vel / max(num_batches, 1),
        }

    # ================================================================== eval
    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)

        total_loss = 0.0
        total_mse = 0.0
        total_vel = 0.0
        num_batches = 0

        for batch in tqdm(self.test_loader, desc="Evaluating",
                          disable=not self.accelerator.is_local_main_process):
            motion, gloss, lengths = batch
            B, T, _ = motion.shape

            if self.cfg.USE_LABEL_INDEX_COND:
                cond = torch.tensor(
                    [self.cfg.GLOSS_NAME_LIST.index(gl) for gl in gloss],
                    dtype=torch.long,
                ).to(motion.device)
            else:
                cond = list(gloss)

            padding_mask = create_padding_mask(lengths, T, self.device)

            # Evaluate at a fixed medium noise level (t=500)
            t = torch.full((B,), 500, dtype=torch.long, device=motion.device)
            noise = torch.randn_like(motion)
            x_t = unwrapped.q_sample(motion, t, noise)
            x_0_pred = self.model(x_t, t, cond, padding_mask, motion)

            loss, mse, vel = self.compute_loss(x_0_pred, motion, padding_mask)
            total_loss += loss.item()
            total_mse += mse.item()
            total_vel += vel.item()
            num_batches += 1

        metrics = {
            'loss': total_loss / max(num_batches, 1),
            'mse': total_mse / max(num_batches, 1),
            'vel': total_vel / max(num_batches, 1),
        }

        if self.accelerator.is_main_process:
            self.logger.info(
                f"Eval Epoch {epoch+1}: loss={metrics['loss']:.4f}, "
                f"mse={metrics['mse']:.4f}, vel={metrics['vel']:.4f}"
            )
        return metrics

    # ================================================================== main loop
    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("Starting Diffusion Training")
        self.logger.info(f"  Epochs: {self.cfg.MAX_EPOCHS}")
        self.logger.info(f"  Batch size: {self.cfg.TRAIN_BSZ}")
        self.logger.info(f"  Learning rate: {self.cfg.LEARNING_RATE}")
        self.logger.info(f"  Diffusion steps: {self.cfg.NUM_DIFFUSION_STEPS}")
        self.logger.info(f"  Vel weight: {self.cfg.VEL_WEIGHT}")
        self.logger.info("=" * 60)

        train_hist = {'total': [], 'mse': [], 'vel': []}
        eval_hist = {'total': [], 'mse': [], 'vel': []}

        for epoch in range(self.start_epoch, self.cfg.MAX_EPOCHS):
            train_metrics = self.train_epoch(epoch)

            train_hist['total'].append(train_metrics['loss'])
            train_hist['mse'].append(train_metrics['mse'])
            train_hist['vel'].append(train_metrics['vel'])

            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Train Epoch {epoch+1}/{self.cfg.MAX_EPOCHS}: "
                    f"loss={train_metrics['loss']:.4f}, "
                    f"mse={train_metrics['mse']:.4f}, vel={train_metrics['vel']:.4f}"
                )

            if self.test_loader is not None:
                eval_metrics = self.evaluate(epoch)
                eval_hist['total'].append(eval_metrics['loss'])
                eval_hist['mse'].append(eval_metrics['mse'])
                eval_hist['vel'].append(eval_metrics['vel'])
            else:
                eval_metrics = train_metrics
                eval_hist = None

            is_best = eval_metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = eval_metrics['loss']
            self.save_checkpoint(epoch, eval_metrics, is_best)

            # Plot curves
            if self.accelerator.is_main_process and train_hist['total']:
                fig = os.path.join(self.logging_dir, 'training_curves.png')
                if os.path.exists(fig):
                    os.remove(fig)
                # Adapt hist format for existing plot function

                try:
                    plot_training_curves(fig, self.start_epoch, train_hist, eval_hist)
                except Exception:
                    pass  # Non-critical

            if self.debug and epoch >= 5:
                break

        self.logger.info("Training complete!")
        self.accelerator.end_training()

    # ================================================================== checkpoint
    def save_checkpoint(self, epoch, metrics=None, is_best=False):
        if not self.accelerator.is_main_process:
            return

        unwrapped = self.accelerator.unwrap_model(self.model)
        state_dict = {
            k: v for k, v in unwrapped.state_dict().items()
            if "text_encoder" not in k and "tokenizer" not in k
        }

        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': vars(self.cfg),
            'metrics': metrics,
            'best_loss': self.best_loss,
        }

        ckpt_path = os.path.join(self.ckpt_dir, "newest_model.pt")
        torch.save(checkpoint, ckpt_path)
        self.logger.info(f"Saved checkpoint: {ckpt_path}")

        if is_best:
            best_path = os.path.join(self.ckpt_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path, finetune=False):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        unwrapped = self.accelerator.unwrap_model(self.model)

        model_state = ckpt['model_state_dict']
        current_state = unwrapped.state_dict()
        loaded = 0
        for key in model_state:
            if key in current_state and current_state[key].shape == model_state[key].shape:
                current_state[key] = model_state[key]
                loaded += 1
        unwrapped.load_state_dict(current_state, strict=False)
        self.logger.info(f"Loaded {loaded} keys")

        if finetune:
            self.logger.info("Finetune mode: resetting training state")
        else:
            if 'optimizer_state_dict' in ckpt:
                try:
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    self.logger.warning(f"Could not load optimizer: {e}")
            if 'scheduler_state_dict' in ckpt:
                try:
                    self.lr_scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                except Exception as e:
                    self.logger.warning(f"Could not load scheduler: {e}")
            self.start_epoch = ckpt.get('epoch', -1) + 1
            self.global_step = ckpt.get('global_step', 0)
            self.best_loss = ckpt.get('best_loss', float('inf'))
            self.logger.info(f"Resumed from epoch {self.start_epoch}")

        del ckpt
        torch.cuda.empty_cache()


# ================================================================== CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Train ASL Avatar Diffusion Model")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="WLASL_SMPLX",
                        choices=["SignBank_SMPLX", "WLASL_SMPLX"])
    parser.add_argument("--use_upper_body", action="store_true", default=False)
    parser.add_argument("--use_rot6d", action="store_true", default=False)
    parser.add_argument("--use_mini_dataset", action="store_true", default=False)
    parser.add_argument("--use_label_index_cond", action="store_true", default=False)
    parser.add_argument("--no_root_normalize", action="store_true", default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = DiffusionTrainer(args)
    trainer.train()
