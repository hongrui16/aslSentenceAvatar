"""
ASL Avatar Diffusion Training — Voting + Cross-Attention Fusion
================================================================

Trains the voting-gated cross-attention motion diffusion model.
LLM draft pseudo-gloss loaded from pre-computed cache.

Key difference from trainMotionDiffusion_voting.py:
    voting:       single condition vector per sample
    votingfusion: per-frame cross-attention fusion (motion attends to gloss)

Prerequisite:
    python tools/generate_llm_draft_gloss.py --modes train val test

Usage:
    accelerate launch trainMotionDiffusion_votingfusion.py \
        --use_upper_body --batch_size 100 --epochs 150
"""

import os
import logging
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger

from dataloader.How2SignSMPLXVotingDataset import How2SignSMPLXVotingDataset
from network.MotionDiffusionModelV1_votingfusion import MotionDiffusionModelV1_VotingFusion

from utils.utils import plot_training_curves, backup_code, collate_fn, create_padding_mask
from config import How2Sign_SMPLX_Config
from utils.rotation_conversion import get_joint_slices


class VotingFusionDiffusionTrainer:

    def __init__(self, args):
        self.args = args
        self.debug = args.debug

        self.cfg = How2Sign_SMPLX_Config()
        self.cfg.DATASET_NAME = args.dataset
        self.cfg.USE_UPPER_BODY = not args.no_upper_body
        self.cfg.USE_ROT6D = args.use_rot6d
        self.cfg.USE_MINI_DATASET = args.use_mini_dataset
        self.cfg.ROOT_NORMALIZE = not args.no_root_normalize
        self.cfg.TEXT_ENCODER_TYPE = args.text_encoder_type
        self.cfg.MODEL_VERSION = 'v1'
        self.cfg.N_FEATS = 6 if self.cfg.USE_ROT6D else 3

        self.cfg.NUM_DIFFUSION_STEPS = getattr(self.cfg, 'NUM_DIFFUSION_STEPS', 1000)
        self.cfg.VEL_WEIGHT = getattr(self.cfg, 'VEL_WEIGHT', 1.0)
        self.cfg.PREDICTION_TYPE = args.prediction_type
        self.cfg.UNCOND_PROB = args.uncond_prob
        self.cfg.GUIDANCE_SCALE = args.guidance_scale
        self.cfg.COND_MODE = 'votingfusion'

        # Voting module config
        self.cfg.VOTING_N_LAYERS = args.voting_n_layers
        self.cfg.VOTING_N_HEADS = args.voting_n_heads
        self.cfg.VOTING_FF_MULT = args.voting_ff_mult
        self.cfg.VOTING_MAX_WORDS = args.voting_max_words

        # Fusion module config
        self.cfg.FUSION_N_LAYERS = args.fusion_n_layers
        self.cfg.FUSION_N_HEADS = args.fusion_n_heads

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
            self.cfg.LOG_DIR, f"{self.cfg.PROJECT_NAME}_{self.cfg.MODEL_VERSION}_votingfusion",
            self.cfg.DATASET_NAME, timestamp
        )
        if self.debug:
            self.ckpt_dir = self.logging_dir
        else:
            self.ckpt_dir = os.path.join(
                self.cfg.CKPT_DIR, f"{self.cfg.PROJECT_NAME}_{self.cfg.MODEL_VERSION}_votingfusion",
                self.cfg.DATASET_NAME, timestamp
            )

        # ---- Accelerator ----
        project_config = ProjectConfiguration(
            project_dir=self.logging_dir,
            logging_dir=self.logging_dir,
        )
        self.accelerator = Accelerator(
            mixed_precision=self.cfg.MIXED_PRECISION.lower() if hasattr(self.cfg, 'MIXED_PRECISION') else 'no',
            project_config=project_config,
            gradient_accumulation_steps=getattr(self.cfg, 'GRAD_ACCUM', 1),
        )
        self.device = self.accelerator.device

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.logging_dir, 'train.log')),
                logging.StreamHandler(),
            ] if self.accelerator.is_main_process else [logging.NullHandler()],
        )
        self.logger = get_logger(__name__)
        self.logger.info(f"Logging dir: {self.logging_dir}")
        self.logger.info(f"Checkpoint dir: {self.ckpt_dir}")

        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self._build_components()

        self.global_step = 0
        self.start_epoch = 0
        self.best_loss = float('inf')

        if args.resume:
            self.load_checkpoint(args.resume, finetune=args.finetune)

    def _build_components(self):
        self.logger.info("Building components...")

        train_dataset = How2SignSMPLXVotingDataset(mode='train', cfg=self.cfg, logger=self.logger)
        test_dataset = How2SignSMPLXVotingDataset(mode='test', cfg=self.cfg, logger=self.logger)

        if self.cfg.TRAIN_BSZ > len(train_dataset):
            self.cfg.TRAIN_BSZ = len(train_dataset) // 2

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.cfg.TRAIN_BSZ, shuffle=True,
            collate_fn=collate_fn, num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True, drop_last=True,
        )
        self.logger.info(f"Train: {len(train_dataset)} samples, {len(self.train_loader)} batches")

        self.test_loader = DataLoader(
            test_dataset, batch_size=self.cfg.EVAL_BSZ, shuffle=False,
            collate_fn=collate_fn, num_workers=self.cfg.NUM_WORKERS, pin_memory=True,
        )

        self.logger.info("Building MotionDiffusionModelV1_VotingFusion...")
        self.model = MotionDiffusionModelV1_VotingFusion(self.cfg)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total params: {total_params:,}")
        self.logger.info(f"Trainable params: {trainable_params:,}")

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.LEARNING_RATE, weight_decay=self.cfg.WEIGHT_DECAY,
        )

        total_steps = len(self.train_loader) * self.cfg.MAX_EPOCHS
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=1e-6,
        )

        (
            self.model, self.optimizer, self.train_loader,
            self.test_loader, self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader,
            self.test_loader, self.lr_scheduler,
        )

        if self.accelerator.is_main_process:
            backup_code(self.logging_dir)
            self._log_config()

    def _log_config(self):
        for key in sorted(vars(self.cfg)):
            if not key.startswith('_'):
                self.logger.info(f"  {key}: {getattr(self.cfg, key)}")

    # ================================================================== loss
    def compute_loss(self, output, target, padding_mask):
        B, T, D = output.shape
        n_feats = self.cfg.N_FEATS
        valid_mask = ~padding_mask if padding_mask is not None else torch.ones(B, T, dtype=torch.bool, device=output.device)

        def masked_mse(pred, gt):
            mse = F.mse_loss(pred, gt, reduction='none')
            mask = valid_mask.unsqueeze(-1).expand_as(mse).float()
            return (mse * mask).sum() / (mask.sum() + 1e-8)

        joint_groups = get_joint_slices(n_feats=n_feats)
        ROOT       = joint_groups['ROOT']
        LOWER_BODY = joint_groups['LOWER_BODY']
        TORSO      = joint_groups['TORSO']
        ARMS       = joint_groups['ARMS']
        LHAND      = joint_groups['LHAND']
        RHAND      = joint_groups['RHAND']
        JAW        = joint_groups['JAW']

        mse_loss = (0.5 * masked_mse(output[..., TORSO],  target[..., TORSO])
                  + 0.0 * masked_mse(output[..., ROOT],  target[..., ROOT])
                  + 0.0 * masked_mse(output[..., LOWER_BODY],  target[..., LOWER_BODY])
                  + 5.0 * masked_mse(output[..., ARMS], target[..., ARMS])
                  + 5.0 * masked_mse(output[..., LHAND], target[..., LHAND])
                  + 5.0 * masked_mse(output[..., RHAND], target[..., RHAND])
                  + 0.1 * masked_mse(output[..., JAW],   target[..., JAW]))

        if self.cfg.PREDICTION_TYPE == 'epsilon':
            vel_loss = torch.zeros((), device=output.device, dtype=output.dtype)
        else:
            vel_gt   = target[:, 1:] - target[:, :-1]
            vel_pred = output[:, 1:] - output[:, :-1]
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
                motion, sentence, gloss_strings, actual_lengths = batch

                B, T, _ = motion.shape
                lengths      = actual_lengths.to(motion.device)
                padding_mask = create_padding_mask(lengths, T, self.device)

                t = torch.randint(0, unwrapped.num_timesteps, (B,), device=motion.device)
                noise = torch.randn_like(motion)
                x_t = unwrapped.q_sample(motion, t, noise)

                output = self.model(x_t, t, list(sentence), padding_mask, motion,
                                    gloss_input=list(gloss_strings))

                target = noise if self.cfg.PREDICTION_TYPE == 'epsilon' else motion
                loss, mse, vel = self.compute_loss(output, target, padding_mask)

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
                    self.logger.info(
                        f"[epoch {epoch}] Peak GPU: {mem:.2f} GB (batch_size={self.cfg.TRAIN_BSZ})"
                    )
                    if unwrapped._voting_gates is not None:
                        gates = unwrapped._voting_gates
                        mask = unwrapped._voting_word_mask
                        valid_gates = gates[~mask]
                        if valid_gates.numel() > 0:
                            self.logger.info(
                                f"  Gate stats: mean={valid_gates.mean():.3f}, "
                                f"std={valid_gates.std():.3f}, "
                                f"min={valid_gates.min():.3f}, max={valid_gates.max():.3f}"
                            )

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
            motion, sentence, gloss_strings, actual_lengths = batch

            B, T, _ = motion.shape
            lengths      = actual_lengths.to(motion.device)
            padding_mask = create_padding_mask(lengths, T, self.device)

            t = torch.full((B,), 500, dtype=torch.long, device=motion.device)
            noise = torch.randn_like(motion)
            x_t = unwrapped.q_sample(motion, t, noise)

            output = self.model(x_t, t, list(sentence), padding_mask, motion,
                                gloss_input=list(gloss_strings))

            target = noise if self.cfg.PREDICTION_TYPE == 'epsilon' else motion
            loss, mse, vel = self.compute_loss(output, target, padding_mask)

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
        self.logger.info("Starting VotingFusion Diffusion Training")
        self.logger.info(f"  Epochs: {self.cfg.MAX_EPOCHS}")
        self.logger.info(f"  Batch size: {self.cfg.TRAIN_BSZ}")
        self.logger.info(f"  Learning rate: {self.cfg.LEARNING_RATE}")
        self.logger.info(f"  Prediction type: {self.cfg.PREDICTION_TYPE}")
        self.logger.info(f"  Uncond prob (CFG): {self.cfg.UNCOND_PROB}")
        self.logger.info(f"  Voting: {self.cfg.VOTING_N_LAYERS}L, {self.cfg.VOTING_N_HEADS}H")
        self.logger.info(f"  Fusion: {self.cfg.FUSION_N_LAYERS}L, {self.cfg.FUSION_N_HEADS}H")
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

            eval_metrics = self.evaluate(epoch)
            eval_hist['total'].append(eval_metrics['loss'])
            eval_hist['mse'].append(eval_metrics['mse'])
            eval_hist['vel'].append(eval_metrics['vel'])

            is_best = eval_metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = eval_metrics['loss']
            self.save_checkpoint(epoch, eval_metrics, is_best)

            if self.accelerator.is_main_process and train_hist['total']:
                fig = os.path.join(self.logging_dir, 'training_curves.png')
                if os.path.exists(fig):
                    os.remove(fig)
                try:
                    plot_training_curves(fig, self.start_epoch, train_hist, eval_hist)
                except Exception:
                    pass

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train ASL VotingFusion Diffusion Model")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="How2SignSMPLX")
    parser.add_argument("--no_upper_body", action="store_true", default=False)
    parser.add_argument("--use_rot6d", action="store_true", default=False)
    parser.add_argument("--use_mini_dataset", action="store_true", default=False)
    parser.add_argument("--no_root_normalize", action="store_true", default=False)
    parser.add_argument("--text_encoder_type", type=str, default='clip', choices=["clip", "t5"])
    parser.add_argument("--prediction_type", type=str, default='epsilon', choices=["epsilon", "x0"])
    parser.add_argument("--uncond_prob", type=float, default=0.1)
    parser.add_argument("--guidance_scale", type=float, default=3.0)

    # Voting module
    parser.add_argument("--voting_n_layers", type=int, default=2)
    parser.add_argument("--voting_n_heads", type=int, default=4)
    parser.add_argument("--voting_ff_mult", type=int, default=2)
    parser.add_argument("--voting_max_words", type=int, default=64)

    # Fusion module
    parser.add_argument("--fusion_n_layers", type=int, default=2)
    parser.add_argument("--fusion_n_heads", type=int, default=8)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = VotingFusionDiffusionTrainer(args)
    trainer.train()
