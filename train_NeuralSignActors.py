"""
train_nsa.py
============
Training script for NeuralSignActorsModel — faithful reproduction of
Neural Sign Actors (Baltatzis et al., arXiv 2312.02702).

Paper training details:
    - Adam optimiser, lr 1e-3 → 1e-6 linearly over 2000 epochs
    - ε prediction, linear noise schedule
    - CLIP-ViT-L-14 text encoder
    - How2Sign dataset

Usage:
    python train_nsa.py --use_rot6d
    python train_nsa.py --use_rot6d --use_expression
    python train_nsa.py --use_rot6d --debug
    python train_nsa.py --use_rot6d --resume path/to/best_model.pt
"""

import os
import logging
import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
from accelerate.logging import get_logger

from dataloader.How2SignSMPLXDataset       import How2SignSMPLXDataset
from dataloader.How2SignSMPLXPhonoDataset  import How2SignSMPLXPhonoDataset
from network.NeuralSignActorsModel         import NeuralSignActorsModel, nsa_loss
from network.PhonoSignActorsModel          import PhonoSignActorsModel
from utils.utils import plot_training_curves, backup_code, create_padding_mask
from config import How2Sign_SMPLX_Config


# ─────────────────────────────────────────────────────────────────────────────
# Collate function  —  How2SignSMPLXDataset returns 4-tuples
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    # Both datasets return 4-tuple: (seq, sentence, gloss_or_dup_sentence, length)
    seqs, sentences, gloss_strings, lengths = zip(*batch)
    return (
        torch.stack(seqs, dim=0),
        list(sentences),
        list(gloss_strings),
        torch.tensor(lengths, dtype=torch.long),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class NSATrainer:

    def __init__(self, args):
        self.args  = args
        self.debug = args.debug

        # ── config ────────────────────────────────────────────────────────────
        cfg = How2Sign_SMPLX_Config()
        cfg.DATASET_NAME         = "How2SignSMPLX"
        cfg.USE_ROT6D            = args.use_rot6d
        cfg.USE_UPPER_BODY       = args.use_upper_body
        cfg.ROOT_NORMALIZE       = args.root_normalize
        cfg.USE_EXPRESSION       = args.use_expression
        cfg.N_EXPR               = 10
        cfg.PROJECT_NAME         = "NeuralSignActors"
        cfg.MODEL_VERSION        = 'NeuralSignActors'
        cfg.N_FEATS              = 6 if cfg.USE_ROT6D else 3
        cfg.USE_MINI_DATASET     = args.use_mini_dataset
        cfg.USE_PHONO_ATTRIBUTE  = args.use_phono_attribute

        # Paper: CLIP-ViT-L-14
        cfg.CLIP_MODEL_NAME = 'openai/clip-vit-large-patch14'

        # Architecture (paper: 4 layers each)
        cfg.GNN_JOINT_DIM  = getattr(cfg, 'GNN_JOINT_DIM', 128)
        cfg.GNN_N_LAYERS   = 4
        cfg.LSTM_HIDDEN    = getattr(cfg, 'LSTM_HIDDEN',   512)
        cfg.LSTM_N_LAYERS  = 4

        cfg.NUM_DIFFUSION_STEPS = getattr(cfg, 'NUM_DIFFUSION_STEPS', 1000)

        # Phono conditioning flags (Paper 2)
        cfg.USE_GLOSS_CONDITION = args.use_gloss_condition
        cfg.USE_CROSS_ATTN      = args.use_cross_attn
        if cfg.USE_CROSS_ATTN and not cfg.USE_GLOSS_CONDITION:
            # cross-attn implies gloss condition
            cfg.USE_GLOSS_CONDITION = True

        # Paper: Adam, lr 1e-3 → 1e-6, 2000 epochs
        cfg.MAX_EPOCHS    = 2000
        cfg.LEARNING_RATE = 1e-3

        if args.batch_size: cfg.TRAIN_BSZ     = args.batch_size
        if args.epochs:     cfg.MAX_EPOCHS    = args.epochs
        if args.lr:         cfg.LEARNING_RATE = args.lr

        self.cfg = cfg

        # ── directories ───────────────────────────────────────────────────────
        timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
        slurm_id   = os.getenv('SLURM_JOB_ID')
        if slurm_id:   timestamp += f"_job{slurm_id}"
        if self.debug: timestamp  = "debug_" + timestamp

        self.logging_dir = os.path.join(cfg.LOG_DIR, cfg.PROJECT_NAME, timestamp)
        self.ckpt_dir    = self.logging_dir if self.debug else \
                           os.path.join(cfg.CKPT_DIR, cfg.PROJECT_NAME, timestamp)

        # ── accelerator ───────────────────────────────────────────────────────
        acc_cfg = ProjectConfiguration(
            project_dir=self.logging_dir, logging_dir=self.logging_dir
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.GRAD_ACCUM,
            mixed_precision=cfg.MIXED_PRECISION,
            project_config=acc_cfg,
        )
        self.device = self.accelerator.device

        if self.accelerator.is_main_process:
            os.makedirs(self.logging_dir, exist_ok=True)
            os.makedirs(self.ckpt_dir,    exist_ok=True)

        self._setup_logging()
        self._build_components()

        if self.accelerator.is_main_process:
            src = os.path.dirname(os.path.abspath(__file__))
            dst = os.path.join(self.logging_dir, 'code_backup')
            os.makedirs(dst, exist_ok=True)
            backup_code(src, dst, self.logger)

        self.global_step = 0
        self.best_loss   = float('inf')
        self.start_epoch = 0

        if args.resume:
            self.load_checkpoint(args.resume, finetune=args.finetune)

        self.logger.info("Config:")
        for k, v in vars(self.cfg).items():
            self.logger.info(f"  {k}: {v}")
        self.logger.info("-" * 60)

    # ── logging ───────────────────────────────────────────────────────────────
    def _setup_logging(self):
        handlers = []
        if self.accelerator.is_main_process:
            log_file = os.path.join(self.logging_dir, "train.log")
            handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
        logging.basicConfig(
            format="%(asctime)s  %(levelname)s  %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level=logging.INFO,
            handlers=handlers,
        )
        self.logger = get_logger(__name__)
        if self.accelerator.is_main_process:
            self.logger.info(f"Log dir  : {self.logging_dir}")
            self.logger.info(f"Ckpt dir : {self.ckpt_dir}")

    # ── build components ──────────────────────────────────────────────────────
    def _build_components(self):
        self.logger.info("Building dataset and model...")

        DatasetCls = How2SignSMPLXPhonoDataset if self.cfg.USE_GLOSS_CONDITION \
                     else How2SignSMPLXDataset
        train_ds = DatasetCls(mode='train', cfg=self.cfg, logger=self.logger)
        val_ds   = DatasetCls(mode='val',   cfg=self.cfg, logger=self.logger)

        # INPUT_DIM is set from dataset: n_joints*n_feats [+ n_expr]
        self.cfg.INPUT_DIM     = train_ds.input_dim
        self.cfg.NUM_CLASSES   = 0
        self.cfg.GLOSS_NAME_LIST = []

        if self.debug:
            self.cfg.TRAIN_BSZ = min(self.cfg.TRAIN_BSZ, 4)
            self.cfg.EVAL_BSZ  = min(self.cfg.EVAL_BSZ, 4)
            
        self.train_loader = DataLoader(
            train_ds,
            batch_size   = self.cfg.TRAIN_BSZ,
            shuffle      = True,
            collate_fn   = collate_fn,
            num_workers  = self.cfg.NUM_WORKERS,
            pin_memory   = True,
            drop_last    = True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size   = self.cfg.EVAL_BSZ,
            shuffle      = False,
            collate_fn   = collate_fn,
            num_workers  = self.cfg.NUM_WORKERS,
            pin_memory   = True,
        )
        self.logger.info(
            f"Train: {len(train_ds):,} samples  "
            f"Val: {len(val_ds):,} samples  "
            f"INPUT_DIM: {self.cfg.INPUT_DIM}"
        )

        # Model
        ModelCls = PhonoSignActorsModel if self.cfg.USE_GLOSS_CONDITION \
                   else NeuralSignActorsModel
        self.model = ModelCls(self.cfg)
        self.logger.info(
            f"Model: {ModelCls.__name__}  "
            f"(USE_GLOSS_CONDITION={self.cfg.USE_GLOSS_CONDITION}, "
            f"USE_CROSS_ATTN={self.cfg.USE_CROSS_ATTN})"
        )
        total     = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total params    : {total:,}")
        self.logger.info(f"Trainable params: {trainable:,}")

        # Paper: Adam optimiser, lr linearly decays 1e-3 → 1e-6 over 2000 epochs
        params         = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.cfg.LEARNING_RATE)
        n_steps        = self.cfg.MAX_EPOCHS * len(self.train_loader)
        end_factor     = 1e-6 / self.cfg.LEARNING_RATE
        self.lr_sched  = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor = 1.0,
            end_factor   = end_factor,
            total_iters  = n_steps,
        )

        (self.model, self.optimizer,
         self.train_loader, self.lr_sched) = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.lr_sched,
        )
        self.val_loader = self.accelerator.prepare(self.val_loader)

    # ── loss computation ──────────────────────────────────────────────────────
    def _compute_loss(self, motion: torch.Tensor,
                      sentences: list,
                      padding_mask: torch.Tensor,
                      gloss_strings: list = None,
                      return_diag: bool = False) -> torch.Tensor:
        """Sample random t, add noise, predict ε, compute NSA loss."""
        unwrapped = self.accelerator.unwrap_model(self.model)
        B = motion.shape[0]

        t     = torch.randint(0, unwrapped.T, (B,), device=motion.device)
        noise = torch.randn_like(motion)
        x_t   = unwrapped.q_sample(motion, t, noise)

        # Predict ε  (full space, bypassed = zeros).
        # PhonoSignActorsModel accepts gloss_strings; NSA ignores it.
        if self.cfg.USE_GLOSS_CONDITION:
            eps_pred = self.model(x_t, t, sentences, padding_mask,
                                  gloss_strings=gloss_strings)
        else:
            eps_pred = self.model(x_t, t, sentences, padding_mask)

        # Extract active part only for loss
        eps_active   = eps_pred[:, :, unwrapped.tosave_slices]
        noise_active = noise[:, :, unwrapped.tosave_slices]

        if unwrapped.use_expr:
            s = len(unwrapped.all_slices)
            eps_active   = torch.cat([eps_active,
                                      eps_pred[:, :, s: s + unwrapped.n_expr]], dim=-1)
            noise_active = torch.cat([noise_active,
                                      noise[:, :, s: s + unwrapped.n_expr]], dim=-1)

        loss = nsa_loss(
            eps_pred      = eps_active,
            eps_true      = noise_active,
            padding_mask  = padding_mask,
            n_feats       = unwrapped.n_feats,
            active_joints = unwrapped.active_joints,
            use_expr      = unwrapped.use_expr,
            n_expr        = unwrapped.n_expr,
        )
        if return_diag:
            return loss, eps_pred.detach()
        return loss

    # ── train epoch ───────────────────────────────────────────────────────────
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, n = 0.0, 0

        n_prints  = 50
        p_every   = max(len(self.train_loader) // n_prints, 1)
        pbar = tqdm(
            total   = n_prints,
            disable = not self.accelerator.is_local_main_process,
            desc    = f"Epoch {epoch}",
        )

        for step, (motion, sentences, gloss_strings, actual_len) in enumerate(self.train_loader):
            if step % p_every == 0 and pbar.n < n_prints:
                pbar.update(1)

            # ── register forward hooks on first diag step to capture activations ─
            diag_acts, hook_handles = {}, []
            if epoch <= 2 and (step % 50 == 0) and self.accelerator.is_main_process:
                unwrapped = self.accelerator.unwrap_model(self.model)
                def _make_hook(name):
                    def _hook(mod, inp, out):
                        t = out[0] if isinstance(out, tuple) else out
                        diag_acts[name] = (t.detach().abs().mean().item(),
                                           t.detach().float().std().item())
                    return _hook
                targets = {
                    'gnn_layer0': unwrapped.pose_encoder.gnn_layers[0],
                    'gnn_final':  unwrapped.pose_encoder,
                    'pose_proj':  unwrapped.pose_flat_proj,
                    't_proj':     unwrapped.t_proj,
                    'lstm':       unwrapped.lstm,
                    'rh_final':   unwrapped.regress_head,
                }
                for name, mod in targets.items():
                    hook_handles.append(mod.register_forward_hook(_make_hook(name)))

            with self.accelerator.accumulate(self.model):
                B, T, _ = motion.shape
                padding_mask = create_padding_mask(
                    actual_len.to(motion.device), T, self.device
                )
                loss, eps_pred_diag = self._compute_loss(
                    motion, sentences, padding_mask,
                    gloss_strings=gloss_strings, return_diag=True,
                )

                self.accelerator.backward(loss)

                # ── diagnostic: first batch of first 3 epochs ────────────────
                if epoch <= 2 and (step % 50 == 0) and self.accelerator.is_main_process:
                    for h in hook_handles:
                        h.remove()
                    unwrapped = self.accelerator.unwrap_model(self.model)
                    def _g(p):
                        return f"{p.grad.abs().mean().item():.4e}" if p.grad is not None else "None"
                    rh_w    = unwrapped.regress_head[-1].weight
                    lstm_hh = unwrapped.lstm.weight_hh_l0
                    gnn_w   = unwrapped.pose_encoder.gnn_layers[0].W
                    self.logger.info(
                        f"[DIAG e{epoch}] "
                        f"eps_pred |mean|={eps_pred_diag.abs().mean().item():.4e} "
                        f"std={eps_pred_diag.std().item():.4e} | "
                        f"grad rh={_g(rh_w)} lstm_hh0={_g(lstm_hh)} gnn_W0={_g(gnn_w)} | "
                        f"|w| rh={rh_w.abs().mean().item():.4e} "
                        f"lstm_hh0={lstm_hh.abs().mean().item():.4e}"
                    )
                    act_str = " ".join(
                        f"{k}(m={m:.3e},s={s:.3e})" for k, (m, s) in diag_acts.items()
                    )
                    self.logger.info(f"[DIAG e{epoch} acts] {act_str}")

                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_sched.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                n          += 1
                self.global_step += 1

                if step == 0:
                    mem = torch.cuda.max_memory_allocated(self.device) / 2 ** 30
                    self.logger.info(f"[Epoch {epoch}] Peak GPU: {mem:.2f} GB at batch size {motion.shape[0]}")

                if self.debug and step >= 10:
                    break

        pbar.close()
        return total_loss / max(n, 1)

    # ── validation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, epoch: int) -> float:
        self.model.eval()
        unwrapped = self.accelerator.unwrap_model(self.model)
        total_loss, n = 0.0, 0

        for motion, sentences, gloss_strings, actual_len in tqdm(
            self.val_loader, desc="Val",
            disable=not self.accelerator.is_local_main_process,
        ):
            B, T, _ = motion.shape
            padding_mask = create_padding_mask(
                actual_len.to(motion.device), T, self.device
            )

            # Evaluate at fixed mid noise level (t=500) — same as train_v1
            t_fixed = torch.full((B,), 500, dtype=torch.long, device=motion.device)
            noise   = torch.randn_like(motion)
            x_t     = unwrapped.q_sample(motion, t_fixed, noise)

            if self.cfg.USE_GLOSS_CONDITION:
                eps_pred = self.model(x_t, t_fixed, sentences, padding_mask,
                                      gloss_strings=gloss_strings)
            else:
                eps_pred = self.model(x_t, t_fixed, sentences, padding_mask)

            eps_active   = eps_pred[:, :, unwrapped.tosave_slices]
            noise_active = noise[:, :, unwrapped.tosave_slices]
            if unwrapped.use_expr:
                s = len(unwrapped.all_slices)
                eps_active   = torch.cat(
                    [eps_active,   eps_pred[:, :, s: s + unwrapped.n_expr]], dim=-1)
                noise_active = torch.cat(
                    [noise_active, noise[:, :,    s: s + unwrapped.n_expr]], dim=-1)

            loss = nsa_loss(
                eps_pred      = eps_active,
                eps_true      = noise_active,
                padding_mask  = padding_mask,
                n_feats       = unwrapped.n_feats,
                active_joints = unwrapped.active_joints,
                use_expr      = unwrapped.use_expr,
                n_expr        = unwrapped.n_expr,
            )
            total_loss += loss.item()
            n          += 1
        
            if self.debug and n >= 10:
                break
        
        val_loss = total_loss / max(n, 1)
        if self.accelerator.is_main_process:
            self.logger.info(f"Val Epoch {epoch+1}: loss={val_loss:.4f}")
        return val_loss

    # ── main training loop ────────────────────────────────────────────────────
    def train(self):
        self.logger.info("=" * 60)
        self.logger.info("Neural Sign Actors — Training")
        self.logger.info(f"  Epochs          : {self.cfg.MAX_EPOCHS}")
        self.logger.info(f"  Batch size      : {self.cfg.TRAIN_BSZ}")
        self.logger.info(f"  LR schedule     : {self.cfg.LEARNING_RATE:.0e} → 1e-6 (linear)")
        self.logger.info(f"  Diffusion steps : {self.cfg.NUM_DIFFUSION_STEPS}")
        self.logger.info(f"  GNN joint dim   : {self.cfg.GNN_JOINT_DIM}")
        self.logger.info(f"  LSTM hidden     : {self.cfg.LSTM_HIDDEN}")
        self.logger.info(f"  Use expression  : {self.cfg.USE_EXPRESSION}")
        self.logger.info("=" * 60)

        train_hist = {'total': []}
        val_hist   = {'total': []}

        for epoch in range(self.start_epoch, self.cfg.MAX_EPOCHS):
            tr_loss = self.train_epoch(epoch)
            train_hist['total'].append(tr_loss)

            if self.accelerator.is_main_process:
                self.logger.info(
                    f"Train Epoch {epoch+1}/{self.cfg.MAX_EPOCHS}: "
                    f"loss={tr_loss:.4f}  "
                    f"lr={self.lr_sched.get_last_lr()[0]:.2e}"
                )

            vl_loss = self.evaluate(epoch)
            val_hist['total'].append(vl_loss)

            is_best = vl_loss < self.best_loss
            if is_best:
                self.best_loss = vl_loss
            self.save_checkpoint(epoch, vl_loss, is_best)

            if self.accelerator.is_main_process:
                fig = os.path.join(self.logging_dir, 'training_curves.png')
                try:
                    plot_training_curves(fig, self.start_epoch, train_hist, val_hist)
                except Exception:
                    pass

            if self.debug and epoch >= 5:
                break

        self.logger.info("Training complete!")
        self.accelerator.end_training()

    # ── checkpoint ────────────────────────────────────────────────────────────
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        if not self.accelerator.is_main_process:
            return
        unwrapped  = self.accelerator.unwrap_model(self.model)
        state_dict = {
            k: v for k, v in unwrapped.state_dict().items()
            if 'text_encoder' not in k and 'tokenizer' not in k
        }
        ckpt = {
            'epoch':                epoch,
            'global_step':          self.global_step,
            'model_state_dict':     state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_sched.state_dict(),
            'config':               vars(self.cfg),
            'val_loss':             val_loss,
            'best_loss':            self.best_loss,
        }
        torch.save(ckpt, os.path.join(self.ckpt_dir, 'newest_model.pt'))
        self.logger.info(f"Saved newest_model.pt  (val={val_loss:.4f})")
        if is_best:
            torch.save(ckpt, os.path.join(self.ckpt_dir, 'best_model.pt'))
            self.logger.info(f"  → New best_model.pt  (val={val_loss:.4f})")

    def load_checkpoint(self, path: str, finetune: bool = False):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        ckpt      = torch.load(path, map_location='cpu')
        unwrapped = self.accelerator.unwrap_model(self.model)
        cur       = unwrapped.state_dict()
        n = 0
        for k, v in ckpt['model_state_dict'].items():
            if k in cur and cur[k].shape == v.shape:
                cur[k] = v
                n += 1
        unwrapped.load_state_dict(cur, strict=False)
        self.logger.info(f"Loaded {n} keys from {path}")

        if not finetune:
            try: self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            except Exception as e: self.logger.warning(f"optimizer: {e}")
            try: self.lr_sched.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception as e: self.logger.warning(f"scheduler: {e}")
            self.start_epoch = ckpt.get('epoch', -1) + 1
            self.global_step = ckpt.get('global_step', 0)
            self.best_loss   = ckpt.get('best_loss', float('inf'))
            self.logger.info(f"Resumed from epoch {self.start_epoch}")
        else:
            self.logger.info("Finetune mode: training state reset")

        del ckpt
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Neural Sign Actors Training")
    p.add_argument('--batch_size',          type=int,   default=None)
    p.add_argument('--epochs',              type=int,   default=None)
    p.add_argument('--lr',                  type=float, default=None)
    p.add_argument('--resume',              type=str,   default=None)
    p.add_argument('--finetune',            action='store_true')
    p.add_argument('--debug',               action='store_true')
    p.add_argument('--use_rot6d',           action='store_true', default=False)
    p.add_argument('--use_upper_body',      action='store_true', default=False)
    p.add_argument('--root_normalize',      action='store_true', default=False,
                help='Zero out root joint (not in paper)')
    p.add_argument('--use_expression',      action='store_true', default=False,
                help='Include expression params')
    p.add_argument('--use_mini_dataset',    action='store_true', default=False)
    p.add_argument('--use_phono_attribute', action='store_true', default=False)
    # Paper 2 flags
    p.add_argument('--use_gloss_condition', action='store_true', default=False,
                   help='Use pseudo-gloss string as a second CLIP condition (M1/M2)')
    p.add_argument('--use_cross_attn',      action='store_true', default=False,
                   help='Enable per-frame cross-attention over gloss tokens (M2). '
                        'Implies --use_gloss_condition.')
    return p.parse_args()


if __name__ == '__main__':
    args    = parse_args()
    trainer = NSATrainer(args)
    trainer.train()
