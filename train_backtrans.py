"""
Back-Translation Training Script  (v2 — anti-posterior-collapse)
================================================================
Three key changes from v1:
    1. Two-stage training: T5 frozen for first FREEZE_EPOCHS, then unfrozen
    2. Differential learning rates: encoder LR >> decoder LR
    3. Decoder token dropout: scheduled drop rate

Usage:
    python train_backtrans.py --use_rot6d
    python train_backtrans.py --use_rot6d --freeze_epochs 25 --token_drop 0.3
    python train_backtrans.py --use_rot6d --resume path/to/checkpoint.pt
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

from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer as rouge_lib

from dataloader.How2SignSMPLXDataset import How2SignSMPLXDataset
from BackTranslationModel import BackTranslationModel
from utils.utils import plot_training_curves, backup_code, create_padding_mask
from config import How2Sign_SMPLX_Config


def collate_fn(batch):
    """How2SignSMPLXDataset returns 4-tuple: (seq, sentence, sentence, actual_len)"""
    seqs, sentences, _, lengths = zip(*batch)
    seqs    = torch.stack(seqs, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return seqs, list(sentences), lengths


class BackTransTrainer:

    def __init__(self, args):
        self.args  = args
        self.debug = args.debug

        # ── Config ────────────────────────────────────────────────────
        self.cfg = How2Sign_SMPLX_Config()
        self.cfg.DATASET_NAME       = "How2SignSMPLX"
        self.cfg.USE_UPPER_BODY     = args.use_upper_body
        self.cfg.USE_ROT6D          = args.use_rot6d
        self.cfg.ROOT_NORMALIZE     = not args.no_root_normalize
        self.cfg.N_FEATS            = 6 if self.cfg.USE_ROT6D else 3
        self.cfg.PROJECT_NAME       = "BackTranslation"

        # Model hyperparams
        self.cfg.MODEL_DIM          = getattr(self.cfg, "MODEL_DIM",     512)
        self.cfg.N_HEADS            = getattr(self.cfg, "N_HEADS",       8)
        self.cfg.N_LAYERS           = getattr(self.cfg, "N_LAYERS",      4)
        self.cfg.DROPOUT            = getattr(self.cfg, "DROPOUT",       0.1)
        self.cfg.T5_MODEL_NAME      = getattr(self.cfg, "T5_MODEL_NAME", "t5-base")

        # ── NEW: anti-collapse settings ───────────────────────────────
        self.cfg.N_POOL             = args.n_pool
        self.cfg.TOKEN_DROP_RATE    = args.token_drop      # applied during stage 1+2
        self.cfg.FREEZE_EPOCHS      = args.freeze_epochs   # stage-1 duration
        self.cfg.ENCODER_LR         = args.encoder_lr
        self.cfg.DECODER_LR         = args.decoder_lr

        # Always start with T5 frozen; trainer will unfreeze at the right epoch
        self.cfg.FREEZE_T5          = True

        if args.batch_size: self.cfg.TRAIN_BSZ     = args.batch_size
        if args.epochs:     self.cfg.MAX_EPOCHS    = args.epochs
        if args.lr:         self.cfg.LEARNING_RATE = args.lr

        # ── Directories ──────────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slurm_id  = os.getenv("SLURM_JOB_ID")
        if slurm_id:   timestamp += f"_job{slurm_id}"
        if self.debug: timestamp  = "debug_" + timestamp

        self.logging_dir = os.path.join(
            self.cfg.LOG_DIR, self.cfg.PROJECT_NAME, timestamp
        )
        self.ckpt_dir = (
            self.logging_dir if self.debug
            else os.path.join(self.cfg.CKPT_DIR, self.cfg.PROJECT_NAME, timestamp)
        )

        # ── Accelerator ──────────────────────────────────────────────
        acc_cfg = ProjectConfiguration(
            project_dir=self.logging_dir, logging_dir=self.logging_dir
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.GRAD_ACCUM,
            mixed_precision=self.cfg.MIXED_PRECISION,
            project_config=acc_cfg,
        )
        self.device = self.accelerator.device

        if self.accelerator.is_main_process:
            os.makedirs(self.logging_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir,    exist_ok=True)

        self._setup_logging()
        self._build_components()

        if self.accelerator.is_main_process:
            src_dir = os.path.dirname(os.path.abspath(__file__))
            dst_dir = os.path.join(self.logging_dir, "code_backup")
            os.makedirs(dst_dir, exist_ok=True)
            backup_code(
                project_root=src_dir, backup_dir=dst_dir, logger=self.logger
            )

        self.global_step = 0
        self.best_bleu4  = -1.0        # track best BLEU-4 instead of loss
        self.best_loss   = float("inf")
        self.start_epoch = 0
        self.t5_unfrozen = False        # track stage

        if args.resume:
            self.load_checkpoint(args.resume, finetune=args.finetune)

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
            self.logger.info(f"Logging dir : {self.logging_dir}")
            self.logger.info(f"Checkpoint  : {self.ckpt_dir}")

    def _build_components(self):
        self.logger.info("Building components...")

        # ── Dataset ───────────────────────────────────────────────────
        train_dataset = How2SignSMPLXDataset(
            mode="train", cfg=self.cfg, logger=self.logger
        )
        val_dataset = How2SignSMPLXDataset(
            mode="val", cfg=self.cfg, logger=self.logger
        )

        self.cfg.INPUT_DIM = train_dataset.input_dim

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.TRAIN_BSZ,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.EVAL_BSZ,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.cfg.NUM_WORKERS,
            pin_memory=True,
        )
        self.logger.info(
            f"Train: {len(train_dataset)} samples, "
            f"{len(self.train_loader)} batches"
        )
        self.logger.info(
            f"Val  : {len(val_dataset)}   samples, "
            f"{len(self.val_loader)}   batches"
        )

        # ── Model ─────────────────────────────────────────────────────
        self.logger.info("Building BackTranslationModel (v2)...")
        self.model     = BackTranslationModel(self.cfg)
        self.tokenizer = self.model.tokenizer

        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(f"Total params    : {total:,}")
        self.logger.info(f"Trainable params: {trainable:,}  (T5 frozen)")

        # ── Optimizer: stage 1 — encoder only ─────────────────────────
        self.encoder_params = (
            list(self.model.input_proj.parameters())
            + list(self.model.pe.parameters())
            + [self.model.pool_queries]
            + list(self.model.pool_cross_attn.parameters())
            + list(self.model.pool_ln.parameters())
            + list(self.model.pool_ffn.parameters())
            + list(self.model.pool_ln2.parameters())
            + list(self.model.pose_encoder.parameters())
            + list(self.model.pose_to_t5.parameters())
            + list(self.model.encoder_dropout.parameters())
        )

        self.optimizer = torch.optim.AdamW(
            [{"params": self.encoder_params, "lr": self.cfg.ENCODER_LR}],
            weight_decay=self.cfg.WEIGHT_DECAY,
        )

        num_training_steps = self.cfg.MAX_EPOCHS * len(self.train_loader)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
            eta_min=self.cfg.ENCODER_LR * 0.01,
        )

        # ── Accelerate ────────────────────────────────────────────────
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.lr_scheduler
        )
        self.val_loader = self.accelerator.prepare(self.val_loader)

    # ================================================================== stage switch
    def maybe_unfreeze_t5(self, epoch):
        """Called at the start of each epoch. Switches to stage 2 if ready."""
        if self.t5_unfrozen:
            return
        if epoch < self.cfg.FREEZE_EPOCHS:
            return

        self.logger.info("=" * 60)
        self.logger.info(
            f"STAGE 2: Unfreezing T5 at epoch {epoch} "
            f"(decoder_lr={self.cfg.DECODER_LR})"
        )
        self.logger.info("=" * 60)

        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.unfreeze_t5()

        # Add T5 parameters as a new param group with lower LR
        t5_params = [
            p for p in unwrapped.t5.parameters() if p.requires_grad
        ]
        self.optimizer.add_param_group(
            {"params": t5_params, "lr": self.cfg.DECODER_LR}
        )

        trainable = sum(
            p.numel() for p in unwrapped.parameters() if p.requires_grad
        )
        self.logger.info(f"Trainable params now: {trainable:,}")

        self.t5_unfrozen = True

    # ================================================================== tokenise
    def tokenise(self, sentences, device):
        enc = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        labels = enc["input_ids"].to(device)
        labels[labels == self.tokenizer.pad_token_id] = -100
        return labels

    # ================================================================== token drop schedule
    def get_token_drop_rate(self, epoch):
        """
        Stage 1 (frozen): full drop rate — force encoder learning.
        Stage 2 (unfrozen): linearly decay drop rate to half.
        """
        base = self.cfg.TOKEN_DROP_RATE
        if epoch < self.cfg.FREEZE_EPOCHS:
            return base
        # linear decay over remaining epochs
        progress = (epoch - self.cfg.FREEZE_EPOCHS) / max(
            self.cfg.MAX_EPOCHS - self.cfg.FREEZE_EPOCHS, 1
        )
        return base * (1.0 - 0.5 * progress)     # decays to base*0.5

    # ================================================================== train epoch
    def train_epoch(self, epoch):
        self.model.train()

        epoch_loss  = 0.0
        num_batches = 0
        drop_rate   = self.get_token_drop_rate(epoch)

        num_prints  = 50
        print_every = max(len(self.train_loader) // num_prints, 1)
        progress_bar = tqdm(
            total=num_prints,
            disable=not self.accelerator.is_local_main_process,
            desc=f"Epoch {epoch} (drop={drop_rate:.2f})",
        )

        for step, batch in enumerate(self.train_loader):
            if step % print_every == 0 and progress_bar.n < num_prints:
                progress_bar.update(1)

            with self.accelerator.accumulate(self.model):
                pose, sentences, actual_lengths = batch
                B, T, _ = pose.shape

                # Filter corrupt samples
                valid = [
                    i
                    for i, (s, l) in enumerate(zip(sentences, actual_lengths))
                    if s.strip() != "" and l > 0
                ]
                if len(valid) == 0:
                    continue
                if len(valid) < B:
                    pose           = pose[valid]
                    sentences      = [sentences[i] for i in valid]
                    actual_lengths = actual_lengths[valid]

                lengths      = actual_lengths.to(pose.device)
                padding_mask = create_padding_mask(lengths, T, self.device)
                labels       = self.tokenise(sentences, pose.device)

                loss = self.model(
                    pose, labels, padding_mask,
                    token_drop_rate=drop_rate,
                )

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(), max_norm=1.0
                    )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss  += loss.item()
                num_batches += 1
                self.global_step += 1

                if step == 0:
                    mem = torch.cuda.max_memory_allocated(self.device) / 2**30
                    self.logger.info(f"[epoch {epoch}] Peak GPU: {mem:.2f} GB")

                if self.debug and step >= 10:
                    break

        progress_bar.close()
        return epoch_loss / max(num_batches, 1)

    # ================================================================== eval
    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)

        total_loss  = 0.0
        num_batches = 0

        all_preds = []
        all_refs  = []
        show_samples = True

        for batch in tqdm(
            self.val_loader,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process,
        ):
            pose, sentences, actual_lengths = batch
            B, T, _ = pose.shape

            valid = [
                i
                for i, (s, l) in enumerate(zip(sentences, actual_lengths))
                if s.strip() != "" and l > 0
            ]
            if len(valid) == 0:
                continue
            if len(valid) < B:
                pose           = pose[valid]
                sentences      = [sentences[i] for i in valid]
                actual_lengths = actual_lengths[valid]

            lengths      = actual_lengths.to(pose.device)
            padding_mask = create_padding_mask(lengths, T, self.device)
            labels       = self.tokenise(sentences, pose.device)

            loss = self.model(pose, labels, padding_mask)
            total_loss  += loss.item()
            num_batches += 1

            preds = unwrapped.generate(pose, padding_mask)
            all_preds.extend(preds)
            all_refs.extend(sentences)

            if show_samples and self.accelerator.is_main_process:
                for i in range(min(3, len(preds))):
                    self.logger.info(f"  GT  : {sentences[i]}")
                    self.logger.info(f"  Pred: {preds[i]}")
                    self.logger.info("")
                show_samples = False

        val_loss = total_loss / max(num_batches, 1)

        # BLEU
        bleu_scorer = BLEU(max_ngram_order=4)
        bleu_result = bleu_scorer.corpus_score(all_preds, [all_refs])
        bleu1 = bleu_result.precisions[0]
        bleu2 = bleu_result.precisions[1]
        bleu3 = bleu_result.precisions[2]
        bleu4 = bleu_result.score

        # ROUGE-L
        scorer = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for ref, pred in zip(all_refs, all_preds)
        ]
        rouge = sum(rouge_scores) / len(rouge_scores) * 100

        if self.accelerator.is_main_process:
            stage = "STAGE-1 (T5 frozen)" if not self.t5_unfrozen else "STAGE-2"
            self.logger.info(
                f"Val Epoch {epoch + 1} [{stage}]: loss={val_loss:.4f} | "
                f"BLEU-1={bleu1:.2f} BLEU-2={bleu2:.2f} "
                f"BLEU-3={bleu3:.2f} BLEU-4={bleu4:.2f} "
                f"ROUGE={rouge:.2f}"
            )

        return {
            "loss":  val_loss,
            "bleu1": bleu1,
            "bleu2": bleu2,
            "bleu3": bleu3,
            "bleu4": bleu4,
            "rouge": rouge,
        }

    # ================================================================== main loop
    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("Back-Translation Training (v2 — anti-posterior-collapse)")
        self.logger.info(f"  Total epochs  : {self.cfg.MAX_EPOCHS}")
        self.logger.info(f"  Freeze epochs : {self.cfg.FREEZE_EPOCHS}")
        self.logger.info(f"  Encoder LR    : {self.cfg.ENCODER_LR}")
        self.logger.info(f"  Decoder LR    : {self.cfg.DECODER_LR}")
        self.logger.info(f"  Token drop    : {self.cfg.TOKEN_DROP_RATE}")
        self.logger.info(f"  N pool tokens : {self.cfg.N_POOL}")
        self.logger.info(f"  Batch size    : {self.cfg.TRAIN_BSZ}")
        self.logger.info(f"  T5 model      : {self.cfg.T5_MODEL_NAME}")
        self.logger.info("=" * 60)

        train_hist = {"total": []}
        val_hist   = {"total": [], "bleu4": [], "rouge": []}

        for epoch in range(self.start_epoch, self.cfg.MAX_EPOCHS):
            # ── Check stage transition ────────────────────────────────
            self.maybe_unfreeze_t5(epoch)

            train_loss = self.train_epoch(epoch)
            train_hist["total"].append(train_loss)

            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Train Epoch {epoch + 1}/{self.cfg.MAX_EPOCHS}: "
                    f"loss={train_loss:.4f}"
                )

            val_metrics = self.evaluate(epoch)
            val_hist["total"].append(val_metrics["loss"])
            val_hist["bleu4"].append(val_metrics["bleu4"])
            val_hist["rouge"].append(val_metrics["rouge"])

            # Best model by BLEU-4 (more meaningful than loss for generation)
            is_best = val_metrics["bleu4"] > self.best_bleu4
            if is_best:
                self.best_bleu4 = val_metrics["bleu4"]
            if val_metrics["loss"] < self.best_loss:
                self.best_loss = val_metrics["loss"]

            self.save_checkpoint(epoch, val_metrics, is_best)

            if self.accelerator.is_main_process:
                fig = os.path.join(self.logging_dir, "training_curves.png")
                try:
                    plot_training_curves(
                        fig, self.start_epoch, train_hist, val_hist
                    )
                except Exception:
                    pass

            if self.debug and epoch >= 5:
                break

        self.logger.info("Training complete!")
        self.accelerator.end_training()

    # ================================================================== checkpoint
    def save_checkpoint(self, epoch, val_metrics=None, is_best=False):
        if not self.accelerator.is_main_process:
            return

        unwrapped  = self.accelerator.unwrap_model(self.model)
        state_dict = {
            k: v
            for k, v in unwrapped.state_dict().items()
            if not k.startswith("tokenizer")
        }
        checkpoint = {
            "epoch":                epoch,
            "global_step":          self.global_step,
            "model_state_dict":     state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "config":               vars(self.cfg),
            "val_metrics":          val_metrics,
            "best_loss":            self.best_loss,
            "best_bleu4":           self.best_bleu4,
            "t5_unfrozen":          self.t5_unfrozen,
        }
        newest = os.path.join(self.ckpt_dir, "newest_model.pt")
        torch.save(checkpoint, newest)
        self.logger.info(f"Saved checkpoint: {newest}")

        if is_best:
            best = os.path.join(self.ckpt_dir, "best_model.pt")
            torch.save(checkpoint, best)
            self.logger.info(
                f"  ★ New best BLEU-4={val_metrics['bleu4']:.2f}  "
                f"ROUGE={val_metrics['rouge']:.2f}"
            )

    def load_checkpoint(self, checkpoint_path, finetune=False):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        ckpt      = torch.load(checkpoint_path, map_location="cpu")
        unwrapped = self.accelerator.unwrap_model(self.model)

        current = unwrapped.state_dict()
        loaded  = 0
        skipped = []
        for k, v in ckpt["model_state_dict"].items():
            if k in current and current[k].shape == v.shape:
                current[k] = v
                loaded += 1
            else:
                skipped.append(k)
        unwrapped.load_state_dict(current, strict=False)
        self.logger.info(f"Loaded {loaded} keys, skipped {len(skipped)}")
        if skipped:
            self.logger.info(f"  Skipped: {skipped[:10]}...")

        if not finetune:
            try:
                self.optimizer.load_state_dict(
                    ckpt["optimizer_state_dict"]
                )
            except Exception as e:
                self.logger.warning(f"Could not load optimizer: {e}")
            try:
                self.lr_scheduler.load_state_dict(
                    ckpt["scheduler_state_dict"]
                )
            except Exception as e:
                self.logger.warning(f"Could not load scheduler: {e}")
            self.start_epoch = ckpt.get("epoch", -1) + 1
            self.global_step = ckpt.get("global_step", 0)
            self.best_loss   = ckpt.get("best_loss", float("inf"))
            self.best_bleu4  = ckpt.get("best_bleu4", -1.0)
            self.t5_unfrozen = ckpt.get("t5_unfrozen", False)
            self.logger.info(f"Resumed from epoch {self.start_epoch}")
        else:
            self.logger.info("Finetune mode: resetting training state")

        del ckpt
        torch.cuda.empty_cache()


# ================================================================== CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Back-Translation Training (v2)"
    )
    # ── general ───────────────────────────────────────────────────────
    parser.add_argument("--batch_size",        type=int,   default=None)
    parser.add_argument("--epochs",            type=int,   default=None)
    parser.add_argument("--lr",                type=float, default=None,
                        help="(ignored in v2 — use --encoder_lr / --decoder_lr)")
    parser.add_argument("--resume",            type=str,   default=None)
    parser.add_argument("--finetune",          action="store_true", default=False)
    parser.add_argument("--debug",             action="store_true", default=False)
    parser.add_argument("--use_upper_body",    action="store_true", default=False)
    parser.add_argument("--use_rot6d",         action="store_true", default=False)
    parser.add_argument("--no_root_normalize", action="store_true", default=False)

    # ── v2: anti-posterior-collapse ───────────────────────────────────
    parser.add_argument("--freeze_epochs",     type=int,   default=25,
                        help="Number of epochs to keep T5 frozen (stage 1)")
    parser.add_argument("--encoder_lr",        type=float, default=1e-4,
                        help="Learning rate for pose encoder")
    parser.add_argument("--decoder_lr",        type=float, default=1e-5,
                        help="Learning rate for T5 decoder (after unfreeze)")
    parser.add_argument("--token_drop",        type=float, default=0.3,
                        help="Decoder token dropout rate")
    parser.add_argument("--n_pool",            type=int,   default=24,
                        help="Number of tokens after temporal pooling")

    # NOTE: --freeze_t5 removed; use --freeze_epochs to control staging
    return parser.parse_args()


if __name__ == "__main__":
    args    = parse_args()
    trainer = BackTransTrainer(args)
    trainer.train()