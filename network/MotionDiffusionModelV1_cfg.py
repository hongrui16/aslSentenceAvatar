"""
ASL Avatar Motion Diffusion Model V1 + Epsilon Prediction + CFG
================================================================

Based on MotionDiffusionModelV1.py with two key additions:
  1. Epsilon prediction (predict noise instead of x_0)
  2. Classifier-Free Guidance (CFG) for sharper generation

Training:
    x_0 -> add noise -> x_t -> denoiser predicts epsilon -> MSE(eps_pred, eps)
    With probability uncond_prob, condition is replaced by learned null embedding.

Generate (DDIM + CFG):
    x_T ~ N(0,I) -> DDIM denoise with guided epsilon -> x_0
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

New cfg fields:
    PREDICTION_TYPE : 'epsilon' or 'x0'  (default 'epsilon')
    UNCOND_PROB     : float, probability of dropping condition (default 0.1)
    GUIDANCE_SCALE  : float, CFG scale at inference (default 3.0)
    COND_MODE       : 'sentence' | 'gloss' | 'sentence_gloss' (default 'sentence')

Usage:
    model = MotionDiffusionModelV1_CFG(cfg)
    # sentence-only (default):
    eps_pred = model(x_t, t, sentences, padding_mask, motion)
    # gloss-only:
    eps_pred = model(x_t, t, sentences, padding_mask, motion, gloss_input=glosses)
    # sentence+gloss:
    eps_pred = model(x_t, t, sentences, padding_mask, motion, gloss_input=glosses)
    # generation:
    motion = model.generate(sentences, seq_len=200, gloss_input=glosses)
"""

import torch
import torch.nn as nn
import math
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import (
    CLIPTokenizer, CLIPTextModel,
    T5Tokenizer, T5EncoderModel,
    AutoTokenizer, AutoModel,
)

from utils.rotation_conversion import LOWER_BODY_INDICES, ROOT_INDICES, TORSO_INDICES, ALL_INDICES, get_joint_slices
from network.PhonoAttributeEncoder import PhonoAttributeEncoder

# =============================================================================
# Utilities (same as V1)
# =============================================================================

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


def sinusoidal_embedding(timesteps, dim):
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

class MotionDiffusionModelV1_CFG(nn.Module):
    """
    MDM-style Motion Diffusion Model with epsilon prediction and CFG.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.input_dim = cfg.INPUT_DIM

        # Prediction type and CFG
        self.prediction_type = getattr(cfg, 'PREDICTION_TYPE', 'epsilon')
        self.uncond_prob = getattr(cfg, 'UNCOND_PROB', 0.1)

        # USE_3D_INPUT (and Phoenix-2D, which sets it via the trainer) means the
        # dataset emits raw flat features without SMPL-X joint-group structure,
        # so the model treats every dim as active and skips bypass.
        self.use_3d_input = bool(getattr(cfg, 'USE_3D_INPUT', False))
        if self.use_3d_input:
            self.all_slices    = list(range(self.input_dim))
            self.tosave_slices = list(range(self.input_dim))
            self.bypass_slices = []
        else:
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

            self.tosave_slices = [i for i in self.all_slices if i not in set(self.bypass_slices)]

        # Diffusion params
        self.num_timesteps = getattr(cfg, 'NUM_DIFFUSION_STEPS', 1000)
        self._register_schedule()

        # ==================== 1. Condition Module ====================
        self.use_label_index = getattr(cfg, 'USE_LABEL_INDEX_COND', False)
        self.text_encoder_type = getattr(cfg, 'TEXT_ENCODER_TYPE', 'clip').lower()

        if self.use_label_index:
            self.label_embedding = nn.Embedding(cfg.NUM_CLASSES, cfg.MODEL_DIM)
        elif self.text_encoder_type == 't5':
            t5_name = getattr(cfg, 'T5_MODEL_NAME', 't5-base')
            print(f"Loading T5 model: {t5_name}...")
            self.tokenizer = T5Tokenizer.from_pretrained(t5_name)
            self.text_encoder = T5EncoderModel.from_pretrained(t5_name)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()

            t5_dim = self.text_encoder.config.d_model
            self.condition_proj = nn.Sequential(
                nn.Linear(t5_dim, cfg.MODEL_DIM),
                nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
                nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            )
        elif self.text_encoder_type == 'mclip':
            # Multilingual encoder for non-English datasets (Phoenix-2014T DE).
            # Loads via AutoModel/AutoTokenizer; sentence repr = mask-aware mean
            # pool over last_hidden_state. Default: xlm-roberta-base (768-d).
            mclip_name = getattr(cfg, 'MCLIP_MODEL_NAME', 'xlm-roberta-base')
            print(f"Loading multilingual encoder: {mclip_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(mclip_name)
            self.text_encoder = AutoModel.from_pretrained(mclip_name)
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()

            mclip_dim = self.text_encoder.config.hidden_size
            self.condition_proj = nn.Sequential(
                nn.Linear(mclip_dim, cfg.MODEL_DIM),
                nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
                nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            )
        else:
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

        # Learned null embedding for classifier-free guidance
        self.null_cond_emb = nn.Parameter(torch.randn(cfg.MODEL_DIM) * 0.01)

        # ==================== 1b. Gloss Condition (for cond_mode='gloss'|'sentence_gloss') ==
        self.cond_mode = getattr(cfg, 'COND_MODE', 'sentence')
        self.gloss_encoding = getattr(cfg, 'GLOSS_ENCODING', 'per_word')
        if self.cond_mode in ('gloss', 'sentence_gloss') and not self.use_label_index:
            # Separate projection for pseudo-gloss text (shares the frozen text encoder)
            if self.text_encoder_type == 't5':
                raw_dim = self.text_encoder.config.d_model
            elif self.text_encoder_type == 'mclip':
                raw_dim = self.text_encoder.config.hidden_size
            else:
                raw_dim = getattr(cfg, 'CLIP_DIM', 512)
            self.gloss_proj = nn.Sequential(
                nn.Linear(raw_dim, cfg.MODEL_DIM),
                nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
                nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            )

            # Phonological attribute conditioning
            self.use_phono = getattr(cfg, 'USE_PHONO', False)
            phono_dim = getattr(cfg, 'PHONO_DIM', 64) if self.use_phono else 0
            if self.use_phono:
                from utils.signbank_phono import SignBankPhonoLookup
                signbank_csv = getattr(cfg, 'SIGNBANK_CSV',
                                       'data/ASL_signbank/asl_signbank_dictionary-export.csv')
                self.phono_lookup = SignBankPhonoLookup(signbank_csv)
                self.phono_encoder = PhonoAttributeEncoder(
                    self.phono_lookup.num_classes, phono_dim=phono_dim,
                )
                self.gloss_phono_proj = nn.Linear(raw_dim + phono_dim, raw_dim)

        # ==================== 2. Timestep Embedding ====================
        self.timestep_proj = nn.Sequential(
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
        )

        # ==================== 3. Denoiser (Transformer Encoder) ====================
        self.pose_proj = nn.Linear(self.input_dim, cfg.MODEL_DIM)
        self.pe = PositionalEncoding(cfg.MODEL_DIM, getattr(cfg, 'MAX_SEQ_LEN', 200) + 10)

        # Denoiser variant — `MODEL_ARCH` selects:
        #   'mdm' (default) | 'kin' | 'vel' | 'ms'
        self.model_arch = getattr(cfg, 'MODEL_ARCH', 'mdm').lower()
        if self.model_arch == 'mdm':
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.MODEL_DIM,
                nhead=cfg.N_HEADS,
                dim_feedforward=cfg.MODEL_DIM * 4,
                dropout=cfg.DROPOUT,
                activation='gelu',
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.N_LAYERS)
        else:
            from network.MotionDenoiserVariants import build_denoiser
            self.transformer = build_denoiser(
                arch=self.model_arch,
                d_model=cfg.MODEL_DIM, nhead=cfg.N_HEADS,
                dim_feedforward=cfg.MODEL_DIM * 4,
                dropout=cfg.DROPOUT, num_layers=cfg.N_LAYERS,
                max_T=getattr(cfg, 'MAX_SEQ_LEN', 200),
            )
            print(f"[MDM] using denoiser variant: {self.model_arch}")

        self.output_proj = nn.Sequential(
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            nn.GELU(),
            nn.Linear(cfg.MODEL_DIM, self.input_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------ init
    def _register_schedule(self):
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
    def _encode_text(self, text_input, device):
        """Encode text with frozen CLIP/T5/mclip, return raw embedding (before projection)."""
        if self.text_encoder_type == 't5':
            inputs = self.tokenizer(
                text_input, padding=True, truncation=True,
                max_length=77, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                hidden = self.text_encoder(**inputs).last_hidden_state
                attn_mask = inputs['attention_mask'].unsqueeze(-1).float()
                pooled = (hidden * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1e-9)
            return pooled
        elif self.text_encoder_type == 'mclip':
            # Multilingual encoder via AutoModel — mean-pool last_hidden_state.
            inputs = self.tokenizer(
                text_input, padding=True, truncation=True,
                max_length=77, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                hidden = self.text_encoder(**inputs).last_hidden_state
                attn_mask = inputs['attention_mask'].unsqueeze(-1).float()
                pooled = (hidden * attn_mask).sum(dim=1) / attn_mask.sum(dim=1).clamp(min=1e-9)
            return pooled
        else:
            inputs = self.tokenizer(
                text_input, padding=True, truncation=True,
                max_length=77, return_tensors="pt"
            ).to(device)
            with torch.no_grad():
                return self.text_encoder(**inputs).pooler_output

    def _embed_phono(self, gloss_strings, max_K, device):
        """Look up SignBank phono attributes and encode them."""
        if not getattr(self, 'use_phono', False):
            return None

        from utils.signbank_phono import PHONO_ATTRIBUTES
        B = len(gloss_strings)
        n_attrs = len(PHONO_ATTRIBUTES)
        attr_indices = torch.zeros(B, max_K, n_attrs, dtype=torch.long, device=device)
        found_mask = torch.zeros(B, max_K, dtype=torch.bool, device=device)

        for i, gloss_str in enumerate(gloss_strings):
            words = gloss_str.split()[:max_K] if gloss_str.strip() else ['']
            for j, w in enumerate(words):
                attrs = self.phono_lookup.get_attribute_indices(w)
                if attrs is not None:
                    found_mask[i, j] = True
                    for k, attr_name in enumerate(PHONO_ATTRIBUTES):
                        attr_indices[i, j, k] = attrs[attr_name]

        return self.phono_encoder(attr_indices, found_mask)

    def _encode_gloss_words_pooled(self, gloss_strings, device):
        """Encode each gloss word individually through CLIP, optionally fuse phono, then mean pool."""
        word_lists = [g.split() if g.strip() else [''] for g in gloss_strings]
        lengths = [len(wl) for wl in word_lists]
        max_K = max(lengths)
        all_words = [w for wl in word_lists for w in wl]

        with torch.no_grad():
            tokens = self.tokenizer(
                all_words, padding=True, truncation=True,
                max_length=77, return_tensors='pt',
            ).to(device)
            if self.text_encoder_type in ('t5', 'mclip'):
                hidden = self.text_encoder(**tokens).last_hidden_state
                attn = tokens['attention_mask'].unsqueeze(-1).float()
                word_embs = (hidden * attn).sum(1) / attn.sum(1).clamp(min=1e-9)
            else:
                word_embs = self.text_encoder(**tokens).pooler_output

        B = len(gloss_strings)
        enc_dim = word_embs.shape[-1]

        # Fuse phono attributes per-word before pooling
        if getattr(self, 'use_phono', False):
            per_word = torch.zeros(B, max_K, enc_dim, device=device, dtype=word_embs.dtype)
            idx = 0
            for i, length in enumerate(lengths):
                per_word[i, :length] = word_embs[idx:idx + length]
                idx += length

            phono_tokens = self._embed_phono(gloss_strings, max_K, device)
            fused = torch.cat([per_word, phono_tokens], dim=-1)
            per_word = self.gloss_phono_proj(fused)

            pooled = torch.zeros(B, enc_dim, device=device, dtype=per_word.dtype)
            for i, length in enumerate(lengths):
                pooled[i] = per_word[i, :length].mean(dim=0)
            return pooled

        pooled = torch.zeros(B, enc_dim, device=device, dtype=word_embs.dtype)
        idx = 0
        for i, length in enumerate(lengths):
            pooled[i] = word_embs[idx:idx + length].mean(dim=0)
            idx += length

        return pooled

    def get_condition(self, cond_input, device, gloss_input=None):
        """
        Build condition embedding based on self.cond_mode:
            'sentence'       : condition_proj(encode(sentence))
            'gloss'          : gloss_proj(per-word CLIP mean pool)
            'sentence_gloss' : condition_proj(encode(sentence)) + gloss_proj(per-word CLIP mean pool)
        """
        if self.use_label_index:
            if not isinstance(cond_input, torch.Tensor):
                cond_input = torch.tensor(cond_input, dtype=torch.long, device=device)
            return self.label_embedding(cond_input.to(device))

        if self.cond_mode == 'gloss':
            assert gloss_input is not None, "gloss_input required for cond_mode='gloss'"
            if self.gloss_encoding == 'whole_str':
                return self.gloss_proj(self._encode_text(gloss_input, device))
            return self.gloss_proj(self._encode_gloss_words_pooled(gloss_input, device))

        elif self.cond_mode == 'sentence_gloss':
            assert gloss_input is not None, "gloss_input required for cond_mode='sentence_gloss'"
            sent_emb  = self.condition_proj(self._encode_text(cond_input, device))
            if self.gloss_encoding == 'whole_str':
                gloss_emb = self.gloss_proj(self._encode_text(gloss_input, device))
            else:
                gloss_emb = self.gloss_proj(self._encode_gloss_words_pooled(gloss_input, device))
            return sent_emb + gloss_emb

        else:  # 'sentence' (default)
            return self.condition_proj(self._encode_text(cond_input, device))

    # ------------------------------------------------------------------ diffusion ops
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]

        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)

        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    # ------------------------------------------------------------------ denoiser
    def denoise(self, x_t, t, condition, padding_mask=None):
        """
        Core denoiser network.
        Returns (B, T, input_dim): interpreted as epsilon or x_0
        depending on self.prediction_type.
        """
        B, T, _ = x_t.shape
        device = x_t.device

        t_emb = sinusoidal_embedding(t, self.cfg.MODEL_DIM)
        t_token = self.timestep_proj(t_emb).unsqueeze(1)

        c_token = condition.unsqueeze(1)

        motion_tokens = self.pose_proj(x_t)
        motion_tokens = self.pe(motion_tokens)

        full_seq = torch.cat([t_token, c_token, motion_tokens], dim=1)

        if padding_mask is not None:
            prefix = torch.zeros(B, 2, dtype=torch.bool, device=device)
            full_mask = torch.cat([prefix, padding_mask], dim=1)
        else:
            full_mask = None

        out = self.transformer(full_seq, src_key_padding_mask=full_mask)

        motion_out = out[:, 2:, :]
        return self.output_proj(motion_out)

    # ------------------------------------------------------------------ forward
    def forward(self, x_t, t, cond_input, padding_mask=None, motion=None, gloss_input=None):
        """
        Forward pass for training.

        Returns:
            output: (B, T, D) - epsilon prediction or x_0 prediction
        """
        condition = self.get_condition(cond_input, x_t.device, gloss_input=gloss_input)

        # CFG: randomly replace condition with null embedding during training
        if self.training and self.uncond_prob > 0:
            B = condition.shape[0]
            drop_mask = torch.rand(B, device=condition.device) < self.uncond_prob
            null_emb = self.null_cond_emb.unsqueeze(0).expand(B, -1)
            condition = torch.where(drop_mask.unsqueeze(-1), null_emb, condition)

        x_t = x_t[:, :, self.tosave_slices]
        output = self.denoise(x_t, t, condition, padding_mask)

        if len(self.all_slices) == len(self.tosave_slices):
            return output

        # Pad bypass slices: zeros for eps prediction, GT for x_0 prediction.
        # Match `output`'s dtype (may be fp16/bf16 under autocast) to avoid
        # index-put dtype mismatch.
        if self.prediction_type == 'epsilon':
            out = torch.zeros_like(motion, dtype=output.dtype)
        else:
            out = motion.clone().to(output.dtype)
        out[:, :, self.tosave_slices] = output
        return out

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, cond_input, seq_len=100, device='cuda',
                 num_steps=50, eta=0.0, guidance_scale=None, gloss_input=None):
        """
        DDIM sampling with classifier-free guidance.
        """
        self.eval()
        if guidance_scale is None:
            guidance_scale = getattr(self.cfg, 'GUIDANCE_SCALE', 1.0)

        B = len(cond_input) if isinstance(cond_input, (list, tuple)) else cond_input.shape[0]
        condition = self.get_condition(cond_input, device, gloss_input=gloss_input)
        null_cond = self.null_cond_emb.unsqueeze(0).expand(B, -1)

        # Regression mode: single forward pass, no DDIM loop.
        if getattr(self.cfg, 'REGRESSION_MODE', False):
            x0 = torch.zeros(B, seq_len, self.input_dim, device=device)
            t0 = torch.zeros((B,), dtype=torch.long, device=device)
            output = self.denoise(x0, t0, condition)
            # Expand back to full all_slices (mirrors the DDIM tail at line 566+).
            if len(self.all_slices) == len(self.tosave_slices):
                return output
            B_, T_, D = output.shape
            out = torch.zeros(B_, T_, len(self.all_slices), dtype=output.dtype, device=device)
            out[:, :, self.tosave_slices] = output
            return out

        step_size = self.num_timesteps // num_steps
        timesteps = list(reversed(list(range(0, self.num_timesteps, step_size))))

        x_t = torch.randn(B, seq_len, self.input_dim, device=device)

        for i, t_cur in enumerate(timesteps):
            t_batch = torch.full((B,), t_cur, dtype=torch.long, device=device)
            alpha_t = self.alphas_cumprod[t_cur]

            if self.prediction_type == 'epsilon':
                # Model predicts noise
                eps_cond = self.denoise(x_t, t_batch, condition)
                if guidance_scale != 1.0:
                    eps_uncond = self.denoise(x_t, t_batch, null_cond)
                    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                else:
                    eps = eps_cond
                x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            else:
                # Model predicts x_0 (legacy)
                x_0_cond = self.denoise(x_t, t_batch, condition)
                if guidance_scale != 1.0:
                    x_0_uncond = self.denoise(x_t, t_batch, null_cond)
                    x_0_pred = x_0_uncond + guidance_scale * (x_0_cond - x_0_uncond)
                else:
                    x_0_pred = x_0_cond
                eps = (x_t - torch.sqrt(alpha_t) * x_0_pred) / torch.sqrt(1 - alpha_t)

            if i == len(timesteps) - 1:
                x_t = x_0_pred
                break

            t_next = timesteps[i + 1]
            alpha_next = self.alphas_cumprod[t_next]
            sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
            x_t = (torch.sqrt(alpha_next) * x_0_pred
                    + torch.sqrt(1 - alpha_next - sigma**2) * eps
                    + sigma * torch.randn_like(x_t))

        B, T, D = x_t.shape
        if D < len(self.all_slices):
            out = torch.zeros(B, T, len(self.all_slices), dtype=x_t.dtype, device=device)
            out[:, :, self.tosave_slices] = x_t
        else:
            out = x_t

        return out

    # ------------------------------------------------------------------ reconstruct
    def reconstruct(self, x_0, cond_input, padding_mask=None, noise_level=100):
        self.eval()
        device = x_0.device
        B = x_0.shape[0]
        condition = self.get_condition(cond_input, device)

        t = torch.full((B,), noise_level, dtype=torch.long, device=device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        raw = self.denoise(x_t[:, :, self.tosave_slices], t, condition, padding_mask)

        # Convert to x_0 space if epsilon prediction
        if self.prediction_type == 'epsilon':
            alpha_t = self.alphas_cumprod[noise_level]
            x_t_active = x_t[:, :, self.tosave_slices]
            raw = (x_t_active - torch.sqrt(1 - alpha_t) * raw) / torch.sqrt(alpha_t)

        if len(self.all_slices) == len(self.tosave_slices):
            return raw

        out = x_0.clone()
        out[:, :, self.tosave_slices] = raw
        return out
