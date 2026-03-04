"""
ASL Avatar Motion Diffusion Model (V5)
========================================

MDM-style (Motion Diffusion Model) architecture for sign language generation.
Replaces CVAE with diffusion for better temporal dynamics.
Now supports switchable text encoders via cfg.TEXT_ENCODER_TYPE:
    - "clip"  (default) — CLIP ViT-B/32 pooler output
    - "t5"              — T5-base encoder, mean-pooled hidden states
Architecture:
    Denoiser = Transformer Encoder over [timestep_token, cond_token, motion_tokens...]
    
    Training:  x_0 → add noise → x_t → denoiser predicts x_0 → MSE loss
    Generate:  x_T ~ N(0,I) → DDIM denoise 50 steps → x_0

Key advantages over CVAE:
    - No information bottleneck (no single z vector for entire sequence)
    - Iterative refinement captures complex temporal dynamics
    - x_0 prediction enables direct velocity loss

Usage:
    model = MotionDiffusionModel(cfg)
    
    # Training: model returns predicted x_0
    x_0_pred = model(x_t, t, gloss_input, padding_mask)
    
    # Generation: DDIM sampling
    motion = model.generate(gloss_input, seq_len=85, device='cuda')
"""

import torch
import torch.nn as nn
import math
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import (
    CLIPTokenizer, CLIPTextModel,
    T5Tokenizer, T5EncoderModel,
)

from utils.rotation_conversion import LOWER_BODY_INDICES, ROOT_INDICES, TORSO_INDICES, ALL_INDICES, get_joint_slices
# =============================================================================
# Utilities
# =============================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine noise schedule (Improved DDPM)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


def sinusoidal_embedding(timesteps, dim):
    """Sinusoidal timestep embedding, same as DDPM."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device).float() / half)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# =============================================================================
# Model
# =============================================================================

class MotionDiffusionModel(nn.Module):
    """
    MDM-style Motion Diffusion Model with switchable text encoder.
    
    cfg.TEXT_ENCODER_TYPE:
        "clip"  — CLIPTextModel, pooler_output (512-d)
        "t5"    — T5EncoderModel, mean-pooled last_hidden_state (768-d for t5-base)
    Denoiser architecture (Transformer Encoder):
        Input sequence: [timestep_token, condition_token, motion_token_1, ..., motion_token_T]
        Self-attention over all tokens → extract motion tokens → output projection
    
    Diffusion:
        - Cosine noise schedule, 1000 steps
        - x_0 prediction (not epsilon)
        - DDIM sampling for fast inference (50 steps)
    
    Conditioning:
        - CLIP text encoding (default)
        - Label index embedding (if cfg.USE_LABEL_INDEX_COND)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.input_dim = cfg.INPUT_DIM
        
        
        joint_groups = get_joint_slices(n_feats=cfg.N_FEATS)
        root_slices = joint_groups['ROOT']
        lower_body_slices = joint_groups['LOWER_BODY']
        self.all_slices = joint_groups['ALL']
        
        
        self.bypass_slices = []
        if cfg.ROOT_NORMALIZE:
            self.input_dim -= len(ROOT_INDICES) * cfg.N_FEATS
            self.bypass_slices += root_slices
        if cfg.USE_UPPER_BODY:
            self.input_dim -= len(LOWER_BODY_INDICES) * cfg.N_FEATS
            self.bypass_slices += lower_body_slices
                                    
        self.tosave_slices = [i for i in self.all_slices if i not in set(self.bypass_slices)]  # all_slices - self.bypass_slices
        
        # Diffusion params
        self.num_timesteps = getattr(cfg, 'NUM_DIFFUSION_STEPS', 1000)
        self._register_schedule()

        # ==================== 1. Condition Module ====================
        self.use_label_index = getattr(cfg, 'USE_LABEL_INDEX_COND', False)
        self.text_encoder_type = getattr(cfg, 'TEXT_ENCODER_TYPE', 'clip').lower()

        if self.use_label_index:
            self.label_embedding = nn.Embedding(cfg.NUM_CLASSES, cfg.MODEL_DIM)
        elif self.text_encoder_type == 't5':
            # --- T5 encoder ---
            t5_name = getattr(cfg, 'T5_MODEL_NAME', 't5-base')
            print(f"Loading T5 model: {t5_name}...")
            self.tokenizer = T5Tokenizer.from_pretrained(t5_name)
            self.text_encoder = T5EncoderModel.from_pretrained(t5_name)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()

            t5_dim = self.text_encoder.config.d_model  # 768 for t5-base, 1024 for t5-large
            self.condition_proj = nn.Sequential(
                nn.Linear(t5_dim, cfg.MODEL_DIM),
                nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
                nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            )

        else:
            # --- CLIP encoder (default) ---
            clip_name = getattr(cfg, 'CLIP_MODEL_NAME', 'openai/clip-vit-base-patch32')
            print(f"Loading CLIP model: {clip_name}...")
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_name)
            self.text_encoder = CLIPTextModel.from_pretrained(clip_name)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()

            clip_dim = getattr(cfg, 'CLIP_DIM', 512)
            self.condition_proj = nn.Sequential(
                nn.Linear(clip_dim, cfg.MODEL_DIM),
                nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
                nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            )



        # ==================== 2. Timestep Embedding ====================
        self.timestep_proj = nn.Sequential(
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
        )

        # ==================== 3. Denoiser (Transformer Encoder) ====================
        self.pose_proj = nn.Linear(self.input_dim, cfg.MODEL_DIM)
        self.pe = PositionalEncoding(cfg.MODEL_DIM, getattr(cfg, 'MAX_SEQ_LEN', 200) + 10)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.MODEL_DIM,
            nhead=cfg.N_HEADS,
            dim_feedforward=cfg.MODEL_DIM * 4,
            dropout=cfg.DROPOUT,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.N_LAYERS)

        self.output_proj = nn.Sequential(
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL_DIM, self.input_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------ init
    def _register_schedule(self):
        """Pre-compute and register all diffusion schedule tensors."""
        betas = cosine_beta_schedule(self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def _init_weights(self):
        for module in [self.pose_proj, self.timestep_proj, self.output_proj]:
            for m in module.modules() if isinstance(module, nn.Sequential) else [module]:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        if self.use_label_index:
            nn.init.normal_(self.label_embedding.weight, std=0.02)
        else:
            for m in self.condition_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    
    # ------------------------------------------------------------------ condition
    def get_condition(self, cond_input, device):
        """
        Encode condition to (B, D_model).
        
        Routes to the appropriate encoder based on config:
            - label index  → embedding lookup
            - t5           → T5 encoder + mean pool + projection
            - clip         → CLIP encoder + pooler_output + projection
        """
        if self.use_label_index:
            if not isinstance(cond_input, torch.Tensor):
                cond_input = torch.tensor(cond_input, dtype=torch.long, device=device)
            return self.label_embedding(cond_input.to(device))

        elif self.text_encoder_type == 't5':
            inputs = self.tokenizer(
                cond_input, padding=True, truncation=True,
                max_length=77, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                encoder_out = self.text_encoder(**inputs)
                hidden = encoder_out.last_hidden_state          # (B, seq_len, t5_dim)
                # Mean pool over non-padding tokens
                attn_mask = inputs['attention_mask'].unsqueeze(-1).float()  # (B, seq_len, 1)
                pooled = (hidden * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1e-9)
            return self.condition_proj(pooled)

        else:
            # CLIP
            inputs = self.tokenizer(
                cond_input, padding=True, truncation=True,
                max_length=77, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                clip_emb = self.text_encoder(**inputs).pooler_output
            return self.condition_proj(clip_emb)

    
    # ------------------------------------------------------------------ diffusion ops
    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: x_0 → x_t."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]            # (B,)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]

        # Broadcast to (B, 1, 1) for (B, T, D)
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    
    # ------------------------------------------------------------------ denoiser
    def denoise(self, x_t, t, condition, padding_mask=None):
        """
        Core denoiser: predict x_0 from x_t.

        Input to Transformer Encoder:
            [timestep_token, cond_token, motion_token_1, ..., motion_token_T]

        Returns predicted x_0: (B, T, input_dim)
        """
        B, T, _ = x_t.shape
        device = x_t.device

        # Timestep token
        t_emb = sinusoidal_embedding(t, self.cfg.MODEL_DIM)    # (B, D)
        t_token = self.timestep_proj(t_emb).unsqueeze(1)        # (B, 1, D)

        # Condition token
        c_token = condition.unsqueeze(1)                         # (B, 1, D)

        # Motion tokens
        motion_tokens = self.pose_proj(x_t)               # (B, T, D)
        motion_tokens = self.pe(motion_tokens)                   # + positional encoding

        # Concat: [t, c, motion...]
        full_seq = torch.cat([t_token, c_token, motion_tokens], dim=1)  # (B, T+2, D)

        # Padding mask (extend for 2 prefix tokens)
        if padding_mask is not None:
            prefix = torch.zeros(B, 2, dtype=torch.bool, device=device)
            full_mask = torch.cat([prefix, padding_mask], dim=1)
        else:
            full_mask = None

        # Transformer
        out = self.transformer(full_seq, src_key_padding_mask=full_mask)

        # Extract motion tokens (skip first 2)
        motion_out = out[:, 2:, :]
        x_0_pred = self.output_proj(motion_out)  # (B, T, input_dim)



        return x_0_pred

    # ------------------------------------------------------------------ forward
    def forward(self, x_t, t, cond_input, padding_mask=None, motion = None):
        """
        Forward pass for training.

        Args:
            x_t:         (B, T, D) noisy motion
            t:           (B,) integer timesteps
            cond_input:  List[str] or LongTensor (B,)
            padding_mask: (B, T) True where padded
            motion:      (B, T, D) original motion 

        Returns:
            x_0_pred: (B, T, D) predicted clean motion
        """
        condition = self.get_condition(cond_input, x_t.device)        
        x_t = x_t[:, :, self.tosave_slices]
        
        x_0_pred = self.denoise(x_t, t, condition, padding_mask)
        if len(self.all_slices) == len(self.tosave_slices):
            return x_0_pred
        
        out = motion.clone()
        out = out.to(x_0_pred.dtype)
        out[:,:, self.tosave_slices] = x_0_pred
        return out
        
        

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, cond_input, seq_len=100, device='cuda',
                 num_steps=50, eta=0.0):
        """
        DDIM sampling: x_T → x_0.

        Args:
            cond_input:  List[str] or LongTensor for conditioning
            seq_len:     output sequence length
            device:      torch device
            num_steps:   DDIM steps (default 50, max = num_timesteps)
            eta:         DDIM stochasticity (0 = deterministic)

        Returns:
            motion: (B, T, D)
        """
        self.eval()
        B = len(cond_input) if isinstance(cond_input, (list, tuple)) else cond_input.shape[0]
        condition = self.get_condition(cond_input, device)

        # Build DDIM timestep schedule (evenly spaced)
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))
        timesteps = list(reversed(timesteps))  # [T-1, ..., 0]

        # Start from pure noise
        x_t = torch.randn(B, seq_len, self.input_dim,
                           device=device)

        for i, t_cur in enumerate(timesteps):
            t_batch = torch.full((B,), t_cur, dtype=torch.long, device=device)

            # Predict x_0
            x_0_pred = self.denoise(x_t, t_batch, condition, padding_mask=None)

            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
            else:
                # Last step → done
                x_t = x_0_pred
                break

            alpha_t = self.alphas_cumprod[t_cur]
            alpha_next = self.alphas_cumprod[t_next]

            # Predicted noise
            eps = (x_t - torch.sqrt(alpha_t) * x_0_pred) / torch.sqrt(1 - alpha_t)

            # DDIM step
            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
            x_t = (torch.sqrt(alpha_next) * x_0_pred
                    + torch.sqrt(1 - alpha_next - sigma**2) * eps
                    + sigma * torch.randn_like(x_t))

        B, T, D = x_t.shape
        if D < len(self.all_slices):
            out = torch.zeros(B, T, len(self.all_slices), dtype=x_t.dtype, device=device)
            out[:,:, self.tosave_slices] = x_t
        else:
            out = x_t
        
        return out

    def reconstruct(self, x_0, cond_input, padding_mask=None, noise_level=100):
        self.eval()
        device = x_0.device
        B = x_0.shape[0]
        condition = self.get_condition(cond_input, device)

        t = torch.full((B,), noise_level, dtype=torch.long, device=device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        x_0_pred = self.denoise(x_t[:, :, self.tosave_slices], t, condition, padding_mask)
        if len(self.all_slices) == len(self.tosave_slices):
            return x_0_pred

        out = x_0.clone()
        out[:, :, self.tosave_slices] = x_0_pred
        return out