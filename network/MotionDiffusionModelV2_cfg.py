"""
ASL Avatar Motion Diffusion Model V2 + Epsilon Prediction + CFG
================================================================

Based on MotionDiffusionModelV2.py (GNN pose encoder) with:
  1. Epsilon prediction (predict noise instead of x_0)
  2. Classifier-Free Guidance (CFG)

New cfg fields (same as V1_cfg):
    PREDICTION_TYPE : 'epsilon' or 'x0'  (default 'epsilon')
    UNCOND_PROB     : float (default 0.1)
    GUIDANCE_SCALE  : float (default 3.0)
"""

import torch
import torch.nn as nn
import math
from transformers import CLIPTokenizer, CLIPTextModel, T5Tokenizer, T5EncoderModel

from utils.rotation_conversion import (
    LOWER_BODY_INDICES, ROOT_INDICES, TORSO_INDICES,
    ALL_INDICES, get_joint_slices
)

# =============================================================================
# Kinematic tree + GNN (copied from V2 — no changes)
# =============================================================================

SMPLX_PARENTS = [
    -1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9,
    12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28,
    29, 20, 31, 32, 20, 34, 35, 21, 37, 38, 21, 40, 41, 21, 43,
    44, 21, 46, 47, 21, 49, 50, 12,
]


def build_edge_index(active_joints):
    joint_set = set(active_joints)
    orig_to_new = {orig: new for new, orig in enumerate(active_joints)}

    srcs, dsts, types = [], [], []
    for orig_i, parent_orig in enumerate(SMPLX_PARENTS):
        if orig_i not in joint_set:
            continue
        if parent_orig == -1 or parent_orig not in joint_set:
            continue
        i = orig_to_new[orig_i]
        p = orig_to_new[parent_orig]
        srcs.append(p); dsts.append(i); types.append(0)
        srcs.append(i); dsts.append(p); types.append(1)

    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)
    edge_type  = torch.tensor(types,        dtype=torch.long)
    return edge_index, edge_type


class KinematicGNNLayer(nn.Module):
    def __init__(self, feat_dim, n_joints, n_edge_types=2):
        super().__init__()
        self.msg_fns = nn.ModuleList([
            nn.Linear(feat_dim, feat_dim, bias=False)
            for _ in range(n_edge_types)
        ])
        self.pose_emb = nn.Embedding(n_joints, feat_dim)
        self.act  = nn.GELU()
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x, edge_index, edge_type):
        N_batch, N_joints, C = x.shape
        device = x.device
        agg = torch.zeros_like(x)
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        for et in range(len(self.msg_fns)):
            mask = (edge_type == et)
            if not mask.any():
                continue
            s = src_idx[mask]
            d = dst_idx[mask]
            f_src = x[:, s, :]
            f_dst = x[:, d, :]
            diff  = f_src - f_dst
            msg = self.msg_fns[et](diff)
            agg.index_add_(1, d.to(device), msg)

        joint_ids = torch.arange(N_joints, device=device)
        pose_emb  = self.pose_emb(joint_ids)
        agg = agg + pose_emb.unsqueeze(0)
        return self.norm(x + self.act(agg))


class KinematicGNNEncoder(nn.Module):
    def __init__(self, n_joints, n_feats, model_dim,
                 joint_dim=128, n_layers=4, active_joints=None):
        super().__init__()
        self.n_joints  = n_joints
        self.n_feats   = n_feats
        self.joint_dim = joint_dim

        self.input_proj = nn.Linear(n_feats, joint_dim)

        if active_joints is None:
            active_joints = list(range(n_joints))
        edge_index, edge_type = build_edge_index(active_joints)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_type',  edge_type)

        self.gnn_layers = nn.ModuleList([
            KinematicGNNLayer(joint_dim, n_joints)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(n_joints * joint_dim, model_dim)

    def forward(self, x):
        B, T, _ = x.shape
        h = x.reshape(B * T, self.n_joints, self.n_feats)
        h = self.input_proj(h)
        for layer in self.gnn_layers:
            h = layer(h, self.edge_index, self.edge_type)
        h = h.reshape(B * T, -1)
        h = self.output_proj(h)
        return h.reshape(B, T, -1)


# =============================================================================
# Utilities
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

class MotionDiffusionModelV2_CFG(nn.Module):
    """
    V2 (GNN pose encoder) + epsilon prediction + CFG.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.input_dim = cfg.INPUT_DIM

        # Prediction type and CFG
        self.prediction_type = getattr(cfg, 'PREDICTION_TYPE', 'epsilon')
        self.uncond_prob = getattr(cfg, 'UNCOND_PROB', 0.1)

        joint_groups      = get_joint_slices(n_feats=cfg.N_FEATS)
        root_slices       = joint_groups['ROOT']
        lower_body_slices = joint_groups['LOWER_BODY']
        self.all_slices   = joint_groups['ALL']

        self.bypass_slices = []
        if cfg.ROOT_NORMALIZE:
            self.input_dim -= len(ROOT_INDICES) * cfg.N_FEATS
            self.bypass_slices += root_slices
        if cfg.USE_UPPER_BODY:
            self.input_dim -= len(LOWER_BODY_INDICES) * cfg.N_FEATS
            self.bypass_slices += lower_body_slices

        self.tosave_slices = [i for i in self.all_slices
                              if i not in set(self.bypass_slices)]

        # Derive active joint indices
        n_feats = cfg.N_FEATS
        active_joints_set = sorted(set(s // n_feats for s in self.tosave_slices))
        all_joint_indices  = list(ALL_INDICES)
        active_orig_joints = [all_joint_indices[j] for j in active_joints_set]
        self.n_active_joints = len(active_joints_set)

        # Diffusion schedule
        self.num_timesteps = getattr(cfg, 'NUM_DIFFUSION_STEPS', 1000)
        self._register_schedule()

        # ==================== 1. Condition Module ====================
        self.use_label_index   = getattr(cfg, 'USE_LABEL_INDEX_COND', False)
        self.text_encoder_type = getattr(cfg, 'TEXT_ENCODER_TYPE', 'clip').lower()

        if self.use_label_index:
            self.label_embedding = nn.Embedding(cfg.NUM_CLASSES, cfg.MODEL_DIM)
        elif self.text_encoder_type == 't5':
            t5_name = getattr(cfg, 'T5_MODEL_NAME', 't5-base')
            print(f"Loading T5 model: {t5_name}...")
            self.tokenizer    = T5Tokenizer.from_pretrained(t5_name)
            self.text_encoder = T5EncoderModel.from_pretrained(t5_name)
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()
            t5_dim = self.text_encoder.config.d_model
            self.condition_proj = nn.Sequential(
                nn.Linear(t5_dim, cfg.MODEL_DIM), nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM), nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            )
        else:
            clip_name = getattr(cfg, 'CLIP_MODEL_NAME', 'openai/clip-vit-base-patch32')
            print(f"Loading CLIP model: {clip_name}...")
            self.tokenizer    = CLIPTokenizer.from_pretrained(clip_name)
            self.text_encoder = CLIPTextModel.from_pretrained(clip_name)
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            self.text_encoder.eval()
            clip_dim = getattr(cfg, 'CLIP_DIM', 512)
            self.condition_proj = nn.Sequential(
                nn.Linear(clip_dim, cfg.MODEL_DIM), nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM), nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            )

        # Learned null embedding for CFG
        self.null_cond_emb = nn.Parameter(torch.randn(cfg.MODEL_DIM) * 0.01)

        # ==================== 1b. Gloss Condition ====================
        self.cond_mode = getattr(cfg, 'COND_MODE', 'sentence')
        if self.cond_mode in ('gloss', 'sentence_gloss') and not self.use_label_index:
            if self.text_encoder_type == 't5':
                raw_dim = self.text_encoder.config.d_model
            else:
                raw_dim = getattr(cfg, 'CLIP_DIM', 512)
            self.gloss_proj = nn.Sequential(
                nn.Linear(raw_dim, cfg.MODEL_DIM), nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM), nn.GELU(),
                nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
            )

        # ==================== 2. Timestep Embedding ====================
        self.timestep_proj = nn.Sequential(
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM), nn.GELU(),
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
        )

        # ==================== 3. Pose Encoder: GNN ====================
        gnn_joint_dim = getattr(cfg, 'GNN_JOINT_DIM', 128)
        gnn_n_layers  = getattr(cfg, 'GNN_N_LAYERS',  4)

        self.pose_encoder = KinematicGNNEncoder(
            n_joints      = self.n_active_joints,
            n_feats       = n_feats,
            model_dim     = cfg.MODEL_DIM,
            joint_dim     = gnn_joint_dim,
            n_layers      = gnn_n_layers,
            active_joints = active_orig_joints,
        )

        self.pe = PositionalEncoding(cfg.MODEL_DIM, getattr(cfg, 'MAX_SEQ_LEN', 200) + 10)

        # ==================== 4. Transformer Denoiser ====================
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
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM), nn.GELU(),
            nn.Linear(cfg.MODEL_DIM, self.input_dim),
        )

        self._init_weights()

    # ------------------------------------------------------------------ init
    def _register_schedule(self):
        betas          = cosine_beta_schedule(self.num_timesteps)
        alphas         = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas',                         betas)
        self.register_buffer('alphas_cumprod',                alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod',           torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

    def _init_weights(self):
        for module in [self.timestep_proj, self.output_proj]:
            for m in (module.modules() if isinstance(module, nn.Sequential) else [module]):
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)
        if not self.use_label_index:
            for m in self.condition_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------ condition
    def _encode_text(self, text_input, device):
        if self.text_encoder_type == 't5':
            inputs = self.tokenizer(text_input, padding=True, truncation=True,
                                    max_length=77, return_tensors='pt').to(device)
            with torch.no_grad():
                hidden = self.text_encoder(**inputs).last_hidden_state
                mask   = inputs['attention_mask'].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return pooled
        else:
            inputs = self.tokenizer(text_input, padding=True, truncation=True,
                                    max_length=77, return_tensors='pt').to(device)
            with torch.no_grad():
                return self.text_encoder(**inputs).pooler_output

    def get_condition(self, cond_input, device, gloss_input=None):
        if self.use_label_index:
            if not isinstance(cond_input, torch.Tensor):
                cond_input = torch.tensor(cond_input, dtype=torch.long, device=device)
            return self.label_embedding(cond_input.to(device))

        if self.cond_mode == 'gloss':
            assert gloss_input is not None
            return self.gloss_proj(self._encode_text(gloss_input, device))
        elif self.cond_mode == 'sentence_gloss':
            assert gloss_input is not None
            sent_emb  = self.condition_proj(self._encode_text(cond_input, device))
            gloss_emb = self.gloss_proj(self._encode_text(gloss_input, device))
            return sent_emb + gloss_emb
        else:
            return self.condition_proj(self._encode_text(cond_input, device))

    # ------------------------------------------------------------------ diffusion ops
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha     = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha     = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    # ------------------------------------------------------------------ denoiser (GNN)
    def denoise(self, x_t, t, condition, padding_mask=None):
        B, T, _ = x_t.shape
        device  = x_t.device

        t_emb   = sinusoidal_embedding(t, self.cfg.MODEL_DIM)
        t_token = self.timestep_proj(t_emb).unsqueeze(1)

        c_token = condition.unsqueeze(1)

        motion_tokens = self.pose_encoder(x_t)
        motion_tokens = self.pe(motion_tokens)

        full_seq = torch.cat([t_token, c_token, motion_tokens], dim=1)

        if padding_mask is not None:
            prefix    = torch.zeros(B, 2, dtype=torch.bool, device=device)
            full_mask = torch.cat([prefix, padding_mask], dim=1)
        else:
            full_mask = None

        out        = self.transformer(full_seq, src_key_padding_mask=full_mask)
        motion_out = out[:, 2:, :]
        return self.output_proj(motion_out)

    # ------------------------------------------------------------------ forward
    def forward(self, x_t, t, cond_input, padding_mask=None, motion=None, gloss_input=None):
        condition = self.get_condition(cond_input, x_t.device, gloss_input=gloss_input)

        # CFG: randomly replace condition with null embedding during training
        if self.training and self.uncond_prob > 0:
            B = condition.shape[0]
            drop_mask = torch.rand(B, device=condition.device) < self.uncond_prob
            null_emb = self.null_cond_emb.unsqueeze(0).expand(B, -1)
            condition = torch.where(drop_mask.unsqueeze(-1), null_emb, condition)

        x_t    = x_t[:, :, self.tosave_slices]
        output = self.denoise(x_t, t, condition, padding_mask)

        if len(self.all_slices) == len(self.tosave_slices):
            return output

        # Match `output`'s dtype (may be fp16/bf16 under autocast).
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
        self.eval()
        if guidance_scale is None:
            guidance_scale = getattr(self.cfg, 'GUIDANCE_SCALE', 1.0)

        B         = len(cond_input) if isinstance(cond_input, (list, tuple)) else cond_input.shape[0]
        condition = self.get_condition(cond_input, device, gloss_input=gloss_input)
        null_cond = self.null_cond_emb.unsqueeze(0).expand(B, -1)

        step_size = self.num_timesteps // num_steps
        timesteps = list(reversed(list(range(0, self.num_timesteps, step_size))))

        x_t = torch.randn(B, seq_len, self.input_dim, device=device)

        for i, t_cur in enumerate(timesteps):
            t_batch = torch.full((B,), t_cur, dtype=torch.long, device=device)
            alpha_t = self.alphas_cumprod[t_cur]

            if self.prediction_type == 'epsilon':
                eps_cond = self.denoise(x_t, t_batch, condition)
                if guidance_scale != 1.0:
                    eps_uncond = self.denoise(x_t, t_batch, null_cond)
                    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
                else:
                    eps = eps_cond
                x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            else:
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

            t_next     = timesteps[i + 1]
            alpha_next = self.alphas_cumprod[t_next]
            sigma      = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
            x_t        = (torch.sqrt(alpha_next) * x_0_pred
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
        device    = x_0.device
        B         = x_0.shape[0]
        condition = self.get_condition(cond_input, device)
        t         = torch.full((B,), noise_level, dtype=torch.long, device=device)
        noise     = torch.randn_like(x_0)
        x_t       = self.q_sample(x_0, t, noise)
        raw       = self.denoise(x_t[:, :, self.tosave_slices], t, condition, padding_mask)

        if self.prediction_type == 'epsilon':
            alpha_t = self.alphas_cumprod[noise_level]
            x_t_active = x_t[:, :, self.tosave_slices]
            raw = (x_t_active - torch.sqrt(1 - alpha_t) * raw) / torch.sqrt(alpha_t)

        if len(self.all_slices) == len(self.tosave_slices):
            return raw
        out = x_0.clone()
        out[:, :, self.tosave_slices] = raw
        return out
