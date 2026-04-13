"""
ASL Avatar Motion Diffusion Model V2
=====================================
Same as V1 but replaces the flat Linear pose projection with a
Kinematic GNN Encoder that performs message passing along the
SMPL-X kinematic tree before feeding motion tokens to the Transformer.

Key difference from V1:
    V1 denoise:  x_t (B,T,D) → Linear → (B,T,model_dim) → Transformer
    V2 denoise:  x_t (B,T,D) → reshape (B*T, N_joints, n_feats)
                             → KinematicGNNEncoder
                             → (B,T,model_dim) → Transformer

Everything else (diffusion schedule, text encoder, DDIM sampling) is identical.
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
# SMPL-X Kinematic Tree  (53 joints, 0-indexed parent array)
# =============================================================================
SMPLX_PARENTS = [
    -1,  # 0  pelvis
     0,  # 1  left_hip
     0,  # 2  right_hip
     0,  # 3  spine1
     1,  # 4  left_knee
     2,  # 5  right_knee
     3,  # 6  spine2
     4,  # 7  left_ankle
     5,  # 8  right_ankle
     6,  # 9  spine3
     7,  # 10 left_foot
     8,  # 11 right_foot
     9,  # 12 neck
     9,  # 13 left_collar
     9,  # 14 right_collar
    12,  # 15 head
    13,  # 16 left_shoulder
    14,  # 17 right_shoulder
    16,  # 18 left_elbow
    17,  # 19 right_elbow
    18,  # 20 left_wrist
    19,  # 21 right_wrist
    20,  # 22 left_index1
    22,  # 23 left_index2
    23,  # 24 left_index3
    20,  # 25 left_middle1
    25,  # 26 left_middle2
    26,  # 27 left_middle3
    20,  # 28 left_pinky1
    28,  # 29 left_pinky2
    29,  # 30 left_pinky3
    20,  # 31 left_ring1
    31,  # 32 left_ring2
    32,  # 33 left_ring3
    20,  # 34 left_thumb1
    34,  # 35 left_thumb2
    35,  # 36 left_thumb3
    21,  # 37 right_index1
    37,  # 38 right_index2
    38,  # 39 right_index3
    21,  # 40 right_middle1
    40,  # 41 right_middle2
    41,  # 42 right_middle3
    21,  # 43 right_pinky1
    43,  # 44 right_pinky2
    44,  # 45 right_pinky3
    21,  # 46 right_ring1
    46,  # 47 right_ring2
    47,  # 48 right_ring3
    21,  # 49 right_thumb1
    49,  # 50 right_thumb2
    50,  # 51 right_thumb3
    12,  # 52 jaw
]


def build_edge_index(active_joints):
    """
    Build bidirectional edge index for active joints from SMPLX kinematic tree.

    Args:
        active_joints: List[int]  original SMPL-X joint indices that are active

    Returns:
        edge_index:   (2, E) LongTensor  [src, dst] in re-indexed space (0..N-1)
        edge_type:    (E,)   LongTensor  0=parent→child, 1=child→parent
    """
    joint_set = set(active_joints)
    # map original index → new compact index
    orig_to_new = {orig: new for new, orig in enumerate(active_joints)}

    srcs, dsts, types = [], [], []
    for orig_i, parent_orig in enumerate(SMPLX_PARENTS):
        if orig_i not in joint_set:
            continue
        if parent_orig == -1 or parent_orig not in joint_set:
            continue
        i = orig_to_new[orig_i]
        p = orig_to_new[parent_orig]
        # parent → child
        srcs.append(p); dsts.append(i); types.append(0)
        # child → parent
        srcs.append(i); dsts.append(p); types.append(1)

    edge_index = torch.tensor([srcs, dsts], dtype=torch.long)   # (2, E)
    edge_type  = torch.tensor(types,        dtype=torch.long)   # (E,)
    return edge_index, edge_type


# =============================================================================
# Kinematic GNN
# =============================================================================

class KinematicGNNLayer(nn.Module):
    """
    Single message-passing layer on the kinematic tree.

    Update rule (following Neural Sign Actors eq. 8):
        f'_i = LayerNorm( γ( Σ_{j∈N(i)} g_ij(f_j - f_i) + P_i ) + f_i )

    Anisotropy: g_ij depends on edge type (parent→child vs child→parent),
    implemented as two separate linear transforms.

    Args:
        feat_dim   : feature dimension per joint
        n_joints   : number of active joints
        n_edge_types: 2  (parent→child, child→parent)
    """

    def __init__(self, feat_dim, n_joints, n_edge_types=2):
        super().__init__()
        # Anisotropic message functions: one MLP per edge type
        self.msg_fns = nn.ModuleList([
            nn.Linear(feat_dim, feat_dim, bias=False)
            for _ in range(n_edge_types)
        ])
        # Pose embedding: unique learnable token per joint
        self.pose_emb = nn.Embedding(n_joints, feat_dim)
        # Non-linearity + norm
        self.act  = nn.GELU()
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x, edge_index, edge_type):
        """
        Args:
            x          : (N_batch, N_joints, feat_dim)   N_batch = B*T
            edge_index : (2, E)
            edge_type  : (E,)
        Returns:
            x'         : (N_batch, N_joints, feat_dim)
        """
        N_batch, N_joints, C = x.shape
        device = x.device

        # Accumulate messages into agg: (N_batch, N_joints, C)
        agg = torch.zeros_like(x)

        src_idx = edge_index[0]   # (E,)
        dst_idx = edge_index[1]   # (E,)

        for et in range(len(self.msg_fns)):
            mask = (edge_type == et)           # (E,)
            if not mask.any():
                continue
            s = src_idx[mask]                  # (E_t,)
            d = dst_idx[mask]                  # (E_t,)

            # diff = f_src - f_dst : (N_batch, E_t, C)
            f_src = x[:, s, :]                 # (N_batch, E_t, C)
            f_dst = x[:, d, :]
            diff  = f_src - f_dst              # (N_batch, E_t, C)

            msg = self.msg_fns[et](diff)       # (N_batch, E_t, C)

            # Scatter add to dst
            # agg[:, d, :] += msg  — use index_add for correctness
            agg.index_add_(1, d.to(device), msg)

        # Add pose embedding (same for all samples in batch)
        joint_ids = torch.arange(N_joints, device=device)
        pose_emb  = self.pose_emb(joint_ids)   # (N_joints, C)
        agg = agg + pose_emb.unsqueeze(0)       # broadcast over batch

        # Residual + norm
        return self.norm(x + self.act(agg))


class KinematicGNNEncoder(nn.Module):
    """
    Encodes (B, T, N_joints * n_feats) pose sequence using
    stacked KinematicGNNLayers, then projects to (B, T, model_dim).

    Args:
        n_joints      : number of active joints
        n_feats       : features per joint (3 for axis-angle, 6 for rot6d)
        model_dim     : output dimension (= Transformer model_dim)
        joint_dim     : internal GNN feature dimension (default 128)
        n_layers      : number of GNN layers (default 4)
        active_joints : List[int] original SMPL-X joint indices
    """

    def __init__(self, n_joints, n_feats, model_dim,
                 joint_dim=128, n_layers=4, active_joints=None):
        super().__init__()
        self.n_joints  = n_joints
        self.n_feats   = n_feats
        self.joint_dim = joint_dim

        # Project raw joint features to joint_dim
        self.input_proj = nn.Linear(n_feats, joint_dim)

        # Build edge index from kinematic tree
        if active_joints is None:
            active_joints = list(range(n_joints))
        edge_index, edge_type = build_edge_index(active_joints)
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_type',  edge_type)

        # Stacked GNN layers
        self.gnn_layers = nn.ModuleList([
            KinematicGNNLayer(joint_dim, n_joints)
            for _ in range(n_layers)
        ])

        # Final projection: flatten joints → model_dim
        self.output_proj = nn.Linear(n_joints * joint_dim, model_dim)

    def forward(self, x):
        """
        Args:
            x : (B, T, N_joints * n_feats)
        Returns:
            (B, T, model_dim)
        """
        B, T, _ = x.shape

        # Reshape to per-joint features
        h = x.reshape(B * T, self.n_joints, self.n_feats)   # (B*T, J, n_feats)
        h = self.input_proj(h)                               # (B*T, J, joint_dim)

        # GNN message passing
        for layer in self.gnn_layers:
            h = layer(h, self.edge_index, self.edge_type)    # (B*T, J, joint_dim)

        # Flatten joints and project
        h = h.reshape(B * T, -1)                             # (B*T, J*joint_dim)
        h = self.output_proj(h)                              # (B*T, model_dim)

        return h.reshape(B, T, -1)                           # (B, T, model_dim)


# =============================================================================
# Utilities  (identical to V1)
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
# Main Model
# =============================================================================

class MotionDiffusionModelV2(nn.Module):
    """
    Same as MotionDiffusionModelV1 but uses KinematicGNNEncoder
    instead of a flat Linear layer to encode pose tokens.

    New cfg fields:
        GNN_JOINT_DIM : internal GNN feature dim per joint (default 128)
        GNN_N_LAYERS  : number of GNN layers (default 4)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.input_dim = cfg.INPUT_DIM

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

        # Derive active joint indices from tosave_slices
        n_feats = cfg.N_FEATS
        active_joints_set = sorted(set(s // n_feats for s in self.tosave_slices))
        # Map to original SMPL-X indices via ALL_INDICES
        all_joint_indices  = list(ALL_INDICES)   # compact→original mapping
        active_orig_joints = [all_joint_indices[j] for j in active_joints_set]
        self.n_active_joints = len(active_joints_set)

        # Diffusion schedule
        self.num_timesteps = getattr(cfg, 'NUM_DIFFUSION_STEPS', 1000)
        self._register_schedule()

        # ==================== 1. Condition Module (identical to V1) ====================
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

        # ==================== 2. Timestep Embedding ====================
        self.timestep_proj = nn.Sequential(
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM), nn.GELU(),
            nn.Linear(cfg.MODEL_DIM, cfg.MODEL_DIM),
        )

        # ==================== 3. Pose Encoder: GNN (replaces V1 Linear) ====================
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

        # Temporal positional encoding (same as V1)
        self.pe = PositionalEncoding(cfg.MODEL_DIM, getattr(cfg, 'MAX_SEQ_LEN', 200) + 10)

        # ==================== 4. Transformer Denoiser (identical to V1) ====================
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

    # ------------------------------------------------------------------ schedule
    def _register_schedule(self):
        betas           = cosine_beta_schedule(self.num_timesteps)
        alphas          = 1.0 - betas
        alphas_cumprod  = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas',                        betas)
        self.register_buffer('alphas_cumprod',               alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod',          torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',torch.sqrt(1.0 - alphas_cumprod))

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

    # ------------------------------------------------------------------ condition (identical to V1)
    def get_condition(self, cond_input, device):
        if self.use_label_index:
            if not isinstance(cond_input, torch.Tensor):
                cond_input = torch.tensor(cond_input, dtype=torch.long, device=device)
            return self.label_embedding(cond_input.to(device))

        elif self.text_encoder_type == 't5':
            inputs = self.tokenizer(cond_input, padding=True, truncation=True,
                                    max_length=77, return_tensors='pt').to(device)
            with torch.no_grad():
                out    = self.text_encoder(**inputs)
                hidden = out.last_hidden_state
                mask   = inputs['attention_mask'].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return self.condition_proj(pooled)

        else:
            inputs = self.tokenizer(cond_input, padding=True, truncation=True,
                                    max_length=77, return_tensors='pt').to(device)
            with torch.no_grad():
                clip_emb = self.text_encoder(**inputs).pooler_output
            return self.condition_proj(clip_emb)

    # ------------------------------------------------------------------ diffusion ops (identical to V1)
    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha    = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha     = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus = sqrt_one_minus.unsqueeze(-1)
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    # ------------------------------------------------------------------ denoiser  ← KEY CHANGE
    def denoise(self, x_t, t, condition, padding_mask=None):
        """
        Same as V1 but uses KinematicGNNEncoder instead of Linear pose_proj.
        """
        B, T, _ = x_t.shape
        device  = x_t.device

        # Timestep token
        t_emb   = sinusoidal_embedding(t, self.cfg.MODEL_DIM)
        t_token = self.timestep_proj(t_emb).unsqueeze(1)    # (B, 1, D)

        # Condition token
        c_token = condition.unsqueeze(1)                     # (B, 1, D)

        # ── GNN pose encoding (replaces V1 linear projection) ──
        motion_tokens = self.pose_encoder(x_t)               # (B, T, D)
        motion_tokens = self.pe(motion_tokens)               # + temporal PE

        # Concat prefix tokens + motion tokens
        full_seq = torch.cat([t_token, c_token, motion_tokens], dim=1)  # (B, T+2, D)

        # Padding mask
        if padding_mask is not None:
            prefix    = torch.zeros(B, 2, dtype=torch.bool, device=device)
            full_mask = torch.cat([prefix, padding_mask], dim=1)
        else:
            full_mask = None

        out         = self.transformer(full_seq, src_key_padding_mask=full_mask)
        motion_out  = out[:, 2:, :]
        x_0_pred    = self.output_proj(motion_out)           # (B, T, input_dim)
        return x_0_pred

    # ------------------------------------------------------------------ forward (identical to V1)
    def forward(self, x_t, t, cond_input, padding_mask=None, motion=None):
        condition = self.get_condition(cond_input, x_t.device)
        x_t       = x_t[:, :, self.tosave_slices]
        x_0_pred  = self.denoise(x_t, t, condition, padding_mask)

        if len(self.all_slices) == len(self.tosave_slices):
            return x_0_pred

        out = motion.clone().to(x_0_pred.dtype)
        out[:, :, self.tosave_slices] = x_0_pred
        return out

    # ------------------------------------------------------------------ generation (identical to V1)
    @torch.no_grad()
    def generate(self, cond_input, seq_len=100, device='cuda', num_steps=50, eta=0.0):
        self.eval()
        B         = len(cond_input) if isinstance(cond_input, (list, tuple)) else cond_input.shape[0]
        condition = self.get_condition(cond_input, device)

        step_size = self.num_timesteps // num_steps
        timesteps = list(reversed(list(range(0, self.num_timesteps, step_size))))

        x_t = torch.randn(B, seq_len, self.input_dim, device=device)

        for i, t_cur in enumerate(timesteps):
            t_batch  = torch.full((B,), t_cur, dtype=torch.long, device=device)
            x_0_pred = self.denoise(x_t, t_batch, condition, padding_mask=None)

            if i == len(timesteps) - 1:
                x_t = x_0_pred
                break

            t_next     = timesteps[i + 1]
            alpha_t    = self.alphas_cumprod[t_cur]
            alpha_next = self.alphas_cumprod[t_next]
            eps        = (x_t - torch.sqrt(alpha_t) * x_0_pred) / torch.sqrt(1 - alpha_t)
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

    def reconstruct(self, x_0, cond_input, padding_mask=None, noise_level=100):
        self.eval()
        device    = x_0.device
        B         = x_0.shape[0]
        condition = self.get_condition(cond_input, device)
        t         = torch.full((B,), noise_level, dtype=torch.long, device=device)
        noise     = torch.randn_like(x_0)
        x_t       = self.q_sample(x_0, t, noise)
        x_0_pred  = self.denoise(x_t[:, :, self.tosave_slices], t, condition, padding_mask)

        if len(self.all_slices) == len(self.tosave_slices):
            return x_0_pred
        out = x_0.clone()
        out[:, :, self.tosave_slices] = x_0_pred
        return out
