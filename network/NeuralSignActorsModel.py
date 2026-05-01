"""
NeuralSignActorsModel.py
========================
Complete faithful reimplementation of Neural Sign Actors
(Baltatzis et al., arXiv 2312.02702).

Architecture — denoiser  ε_Θ(p^t_{1:F}, t, w):

    θ^t_{1:F}  (noisy pose)      ψ^t_{1:F}  (noisy expr, optional)
         │                              │
    GNN Pose Encoder              MLP Expr Encoder
    4× AnisotropicGNNLayer        4× MLP + LayerNorm
    on SMPL-X kinematic tree      per blendshape coeff
    FiLM(CLIP) after each layer   FiLM(CLIP) after each layer
         │                              │
         └──────────── || ──────────────┘
                       │  + timestep embedding
                ┌──────▼──────┐
                │  4-layer    │
                │  LSTM       │  ← auto-regressive temporal processing
                └──────┬──────┘
                       │
                ┌──────▼──────┐
                │  MLP Regr.  │
                │  Head       │
                └──────┬──────┘
                       │
                ε̂  (predicted noise)

Key design decisions vs MotionDiffusionModelV1:
    prediction  :  ε  (not x_0)
    schedule    :  linear DDPM  (not cosine)
    pose enc.   :  anisotropic GNN on kinematic tree  (not linear proj.)
    temporal    :  4-layer LSTM  (not Transformer)
    text        :  CLIP ViT-L-14, frozen  (not ViT-B/32)
    conditioning:  FiLM gating  (not prepend token)
    loss weight :  hand joints × 2  (paper Eq. 7)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel

from utils.rotation_conversion import (
    ROOT_INDICES, LOWER_BODY_INDICES, ALL_INDICES, get_joint_slices,
    LHAND_INDICES, RHAND_INDICES,
)

# ─────────────────────────────────────────────────────────────────────────────
# SMPL-X kinematic tree  (53 joints)
# ─────────────────────────────────────────────────────────────────────────────
SMPLX_PARENTS = [
    -1,
    0, 0, 0,          # l_hip, r_hip, spine1
    1, 2, 3,          # l_knee, r_knee, spine2
    4, 5, 6,          # l_ankle, r_ankle, spine3
    7, 8, 9, 9, 9,    # l_foot, r_foot, neck, l_collar, r_collar
    12, 13, 14,       # head, l_shoulder, r_shoulder
    16, 17, 18, 19,   # l_elbow, r_elbow, l_wrist, r_wrist
    # left hand (22-36)
    20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35,
    # right hand (37-51)
    21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50,
    # jaw
    15,
]
N_JOINTS = 53
assert len(SMPLX_PARENTS) == N_JOINTS, f"Expected 53, got {len(SMPLX_PARENTS)}"

# Directed edges: parent→child + child→parent
_EDGES_SRC = []
_EDGES_DST = []
for child, parent in enumerate(SMPLX_PARENTS):
    if parent >= 0:
        _EDGES_SRC += [parent, child]
        _EDGES_DST += [child,  parent]
N_EDGES = len(_EDGES_SRC)   # 52 undirected × 2 = 104


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def linear_beta_schedule(T: int) -> torch.Tensor:
    """Linear β schedule from DDPM (Ho et al., 2020)."""
    return torch.linspace(1e-4, 0.02, T, dtype=torch.float32)


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) *
                      torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([args.cos(), args.sin()], dim=-1)  # (B, dim)


# ─────────────────────────────────────────────────────────────────────────────
# FiLM gating
# Paper: "conditioning using a gating approach described in [26]"  (FFJORD)
#   h_out = sigmoid(W_γ·c) ⊙ h  +  tanh(W_β·c)
# ─────────────────────────────────────────────────────────────────────────────

class FiLM(nn.Module):
    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta  = nn.Linear(cond_dim, feat_dim)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """h: (..., feat_dim)  c: (B, cond_dim)"""
        g = torch.sigmoid(self.gamma(c))   # (B, feat_dim)
        b = torch.tanh   (self.beta (c))   # (B, feat_dim)
        for _ in range(h.dim() - 2):       # broadcast over F, J dims
            g = g.unsqueeze(1)
            b = b.unsqueeze(1)
        return h * g + b


# ─────────────────────────────────────────────────────────────────────────────
# Anisotropic GNN layer  —  Eq. 8
#   f'_i = γ( Σ_{j∈K_i}  g_ij(f_j − f_i)  +  P_i )
#   g_ij = edge-specific linear  (anisotropy)
#   P_i  = learnable joint positional embedding
# ─────────────────────────────────────────────────────────────────────────────

class AnisotropicGNNLayer(nn.Module):
    """
    One message-passing step on the SMPL-X kinematic tree.

    Vectorised via einsum — no Python loops.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        # Anisotropic edge weights  (N_EDGES, in_dim, out_dim)
        self.W = nn.Parameter(torch.empty(N_EDGES, in_dim, out_dim))
        nn.init.kaiming_uniform_(self.W.view(N_EDGES, -1), a=math.sqrt(5))

        # Learnable per-joint positional embedding  (N_JOINTS, out_dim)
        self.pose_emb = nn.Embedding(N_JOINTS, out_dim)
        nn.init.normal_(self.pose_emb.weight, std=0.02)

        # Layer norm + residual
        self.norm     = nn.LayerNorm(out_dim)
        self.res_proj = nn.Linear(in_dim, out_dim, bias=False) \
                        if in_dim != out_dim else nn.Identity()

        # Edge buffers  (registered → moves to device with .to())
        self.register_buffer('src', torch.tensor(_EDGES_SRC, dtype=torch.long))
        self.register_buffer('dst', torch.tensor(_EDGES_DST, dtype=torch.long))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        f:       (B, F, J, in_dim)
        returns: (B, F, J, out_dim)
        """
        B, Fr, J, _ = f.shape
        out_dim = self.W.shape[-1]
        src, dst = self.src, self.dst   # (E,)

        # Per-edge feature differences
        diff = f[:, :, src, :] - f[:, :, dst, :]   # (B, F, E, in_dim)

        # Anisotropic transform:  diff[e] @ W[e]
        #   (B, F, E, in_dim) × (E, in_dim, out_dim) → (B, F, E, out_dim)
        msgs = torch.einsum('bfei,eio->bfeo', diff, self.W)

        # Scatter-add messages to destination joints.
        # Match msgs.dtype (not f.dtype) to survive bf16/fp16 autocast.
        agg = torch.zeros(B, Fr, J, out_dim, device=f.device, dtype=msgs.dtype)
        idx = dst.view(1, 1, -1, 1).expand(B, Fr, -1, out_dim)
        agg.scatter_add_(2, idx, msgs)

        # Add per-joint positional embedding P_i
        agg = agg + self.pose_emb.weight   # broadcast over B, F

        # GELU + LayerNorm + residual
        return F.gelu(self.norm(agg)) + self.res_proj(f)


# ─────────────────────────────────────────────────────────────────────────────
# GNN Pose Encoder  —  4 stacked layers, increasing channels, FiLM conditioned
# ─────────────────────────────────────────────────────────────────────────────

class GNNPoseEncoder(nn.Module):
    """
    θ_{1:F} ∈ R^{B×F×J×n_feats}  →  h_pose ∈ R^{B×F×J×out_dim}

    Channel schedule (n_layers=4, base_dim=128):
        128 → 256 → 512 → 512   (capped at base_dim × 4)
    """
    def __init__(self, n_feats: int, base_dim: int, cond_dim: int, n_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(n_feats, base_dim)

        dims, d = [], base_dim
        for i in range(n_layers):
            out = min(base_dim * (2 ** i), base_dim * 4)
            dims.append((d, out))
            d = out
        self.out_dim = d

        self.gnn_layers  = nn.ModuleList([AnisotropicGNNLayer(di, do) for di, do in dims])
        self.film_layers = nn.ModuleList([FiLM(do, cond_dim) for _, do in dims])

    def forward(self, theta: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        theta: (B, F, J, n_feats)    — full 53-joint pose (zeros for bypassed)
        cond:  (B, cond_dim)          — projected CLIP embedding
        """
        h = self.input_proj(theta)
        for gnn, film in zip(self.gnn_layers, self.film_layers):
            h = gnn(h)
            h = film(h, cond)
        return h   # (B, F, J, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# MLP Expression Encoder  —  Eq. 9
#   g'_i = γ( MLP(g_i + E_i) )
# ─────────────────────────────────────────────────────────────────────────────

class MLPExpressionEncoder(nn.Module):
    """
    ψ_{1:F} ∈ R^{B×F×10}  →  h_expr ∈ R^{B×F×10×out_dim}

    Each of the 10 blendshape coefficients is encoded independently.
    Learnable expression embedding E_i added before first layer.
    """
    def __init__(self, n_expr: int, base_dim: int, cond_dim: int, n_layers: int = 4):
        super().__init__()
        self.n_expr     = n_expr
        self.input_proj = nn.Linear(1, base_dim)
        self.expr_emb   = nn.Embedding(n_expr, base_dim)
        nn.init.normal_(self.expr_emb.weight, std=0.02)

        dims, d = [], base_dim
        for i in range(n_layers):
            out = min(base_dim * (2 ** i), base_dim * 4)
            dims.append((d, out))
            d = out
        self.out_dim = d

        self.mlp_layers  = nn.ModuleList([
            nn.Sequential(nn.Linear(di, do), nn.LayerNorm(do))
            for di, do in dims
        ])
        self.film_layers = nn.ModuleList([FiLM(do, cond_dim) for _, do in dims])

    def forward(self, psi: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        psi:  (B, F, n_expr)
        cond: (B, cond_dim)
        """
        h = self.input_proj(psi.unsqueeze(-1))          # (B, F, 10, base_dim)
        h = h + self.expr_emb.weight                    # + E_i  (broadcast B, F)
        for mlp, film in zip(self.mlp_layers, self.film_layers):
            h = F.gelu(mlp(h))
            h = film(h, cond)
        return h   # (B, F, n_expr, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# NeuralSignActorsModel
# ─────────────────────────────────────────────────────────────────────────────

class NeuralSignActorsModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ── config ────────────────────────────────────────────────────────────
        n_feats      = cfg.N_FEATS
        base_dim     = getattr(cfg, 'GNN_JOINT_DIM',   128)
        n_gnn_layers = getattr(cfg, 'GNN_N_LAYERS',      4)
        lstm_hidden  = getattr(cfg, 'LSTM_HIDDEN',      512)
        lstm_layers  = getattr(cfg, 'LSTM_N_LAYERS',     4)
        use_expr     = getattr(cfg, 'USE_EXPRESSION', False)
        n_expr       = getattr(cfg, 'N_EXPR',            10)

        self.n_feats     = n_feats
        self.lstm_hidden = lstm_hidden
        self.use_expr    = use_expr
        self.n_expr      = n_expr

        # ── bypass / active joint logic ────────────────────────────────────────
        # USE_3D_INPUT: dataset has no SMPL-X kinematic tree (Phoenix MediaPipe
        # coords or How2Sign-3D body+hand 3D coords). Skip joint-group bypass and
        # GNN encoder; use a flat linear projection instead. This keeps NSA's
        # core (diffusion + LSTM + per-joint weighted L2 loss) but drops the
        # GNN, since the GNN's anisotropic edges are SMPL-X-specific.
        self.use_3d_input = bool(getattr(cfg, 'USE_3D_INPUT', False))

        if self.use_3d_input:
            input_dim = int(cfg.INPUT_DIM)
            assert input_dim % n_feats == 0, \
                f"INPUT_DIM={input_dim} not divisible by N_FEATS={n_feats}"
            n_active = input_dim // n_feats
            all_slices    = list(range(input_dim))
            tosave_slices = list(range(input_dim))
            active_joints = list(range(n_active))
            # Expression is SMPL-X specific — disable for 3D-input mode.
            use_expr = False
            self.use_expr = False
        else:
            # The dataset omits some joints from the flat tensor based on config
            # flags. We reconstruct the full 53-joint tensor with zeros for
            # bypassed joints, pass everything through the GNN, then extract
            # only active joints.
            joint_groups  = get_joint_slices(n_feats=n_feats)
            all_slices    = joint_groups['ALL']          # 53*n_feats feature indices
            bypass_slices = []
            if cfg.ROOT_NORMALIZE:
                bypass_slices += joint_groups['ROOT']
            if cfg.USE_UPPER_BODY:
                bypass_slices += joint_groups['LOWER_BODY']
            if getattr(cfg, 'EXCLUDE_JAW', False):
                bypass_slices += joint_groups['JAW']
            bypass_set    = set(bypass_slices)
            tosave_slices = [i for i in all_slices if i not in bypass_set]

            # Which of the 53 joints are active (have non-zero features in input)
            active_joints = sorted(set(s // n_feats for s in tosave_slices))
        n_active      = len(active_joints)

        self.all_slices    = all_slices
        self.tosave_slices = tosave_slices
        self.active_joints = active_joints
        self.n_active      = n_active

        # pose feature dim in the active flat tensor
        self.pose_input_dim = n_active * n_feats
        # total denoiser input dim
        self.input_dim = self.pose_input_dim + (n_expr if use_expr else 0)

        # ── diffusion schedule ─────────────────────────────────────────────────
        self.T = getattr(cfg, 'NUM_DIFFUSION_STEPS', 1000)
        betas  = linear_beta_schedule(self.T)
        alphas = 1.0 - betas
        acp    = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas',                betas)
        self.register_buffer('alphas_cumprod',       acp)
        self.register_buffer('sqrt_acp',             acp.sqrt())
        self.register_buffer('sqrt_one_minus_acp',   (1.0 - acp).sqrt())

        # ── 1. Text encoder (frozen) — CLIP (English) or mclip (multilingual)
        self.text_encoder_type = getattr(cfg, 'TEXT_ENCODER_TYPE', 'clip').lower()
        if self.text_encoder_type == 'mclip':
            from transformers import AutoTokenizer, AutoModel
            mclip_name = getattr(cfg, 'MCLIP_MODEL_NAME', 'xlm-roberta-base')
            print(f"[NeuralSignActors] Loading multilingual encoder {mclip_name} ...")
            self.tokenizer    = AutoTokenizer.from_pretrained(mclip_name)
            self.text_encoder = AutoModel.from_pretrained(mclip_name)
            text_dim = self.text_encoder.config.hidden_size
        else:
            clip_name = getattr(cfg, 'CLIP_MODEL_NAME', 'openai/clip-vit-large-patch14')
            print(f"[NeuralSignActors] Loading {clip_name} ...")
            self.tokenizer    = CLIPTokenizer.from_pretrained(clip_name)
            self.text_encoder = CLIPTextModel.from_pretrained(clip_name)
            text_dim = self.text_encoder.config.hidden_size  # 768 for ViT-L-14
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

        self.cond_proj = nn.Sequential(
            nn.Linear(text_dim, lstm_hidden),
            nn.GELU(),
            nn.Linear(lstm_hidden, lstm_hidden),
        )
        cond_dim = lstm_hidden

        # ── 2. GNN pose encoder (SMPL-X) or flat MLP (3D-input) ───────────────
        if self.use_3d_input:
            # No GNN — MediaPipe / How2Sign-3D topology is not SMPL-X.
            # Per-joint MLP keeps the "encode each joint independently then
            # flatten" structure of NSA without baking in the wrong tree.
            self.pose_encoder = None
            per_joint_hidden = base_dim * 2          # mirror GNN final width
            self.pose_per_joint = nn.Sequential(
                nn.Linear(n_feats, base_dim),
                nn.GELU(),
                nn.LayerNorm(base_dim),
                nn.Linear(base_dim, per_joint_hidden),
                nn.GELU(),
                nn.LayerNorm(per_joint_hidden),
            )
            self.pose_flat_proj = nn.Linear(n_active * per_joint_hidden, lstm_hidden)
            self.pose_flat_norm = nn.LayerNorm(lstm_hidden)
        else:
            self.pose_encoder = GNNPoseEncoder(
                n_feats   = n_feats,
                base_dim  = base_dim,
                cond_dim  = cond_dim,
                n_layers  = n_gnn_layers,
            )
            gnn_out_dim = self.pose_encoder.out_dim   # e.g. 512 for base_dim=128, 4 layers

            # Project flattened GNN output → lstm_hidden, with output LayerNorm
            # to prevent variance explosion in this large fan-in (12800→512) layer.
            # Without this, the output std is ~200 which causes downstream collapse.
            self.pose_flat_proj = nn.Linear(n_active * gnn_out_dim, lstm_hidden)
            self.pose_flat_norm = nn.LayerNorm(lstm_hidden)

        # ── 3. MLP expression encoder (optional) ──────────────────────────────
        if use_expr:
            expr_base = max(base_dim // 2, 32)
            self.expr_encoder   = MLPExpressionEncoder(
                n_expr   = n_expr,
                base_dim = expr_base,
                cond_dim = cond_dim,
                n_layers = n_gnn_layers,
            )
            self.expr_flat_proj = nn.Linear(
                n_expr * self.expr_encoder.out_dim, lstm_hidden // 4
            )
            lstm_in = lstm_hidden + lstm_hidden // 4 + lstm_hidden  # pose+expr+t
        else:
            self.expr_encoder   = None
            self.expr_flat_proj = None
            lstm_in = lstm_hidden + lstm_hidden   # pose + t

        # ── 4. Timestep embedding ─────────────────────────────────────────────
        self.t_proj = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.GELU(),
            nn.Linear(lstm_hidden, lstm_hidden),
        )

        # ── 5. 4-layer LSTM decoder (auto-regressive temporal processing) ──────
        # Pre-/post-LN stabilise signal scale (prevents pose_flat_proj blow-up
        # and LSTM output collapse under bf16).
        self.pre_lstm_norm  = nn.LayerNorm(lstm_in)
        self.lstm = nn.LSTM(
            input_size  = lstm_in,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            dropout     = 0.1 if lstm_layers > 1 else 0.0,
        )
        self.post_lstm_norm = nn.LayerNorm(lstm_hidden)

        # ── 6. MLP regression head → ε ───────────────────────────────────────
        noise_dim = n_active * n_feats + (n_expr if use_expr else 0)
        self.regress_head = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.GELU(),
            nn.Linear(lstm_hidden, noise_dim),
        )
        self.noise_dim = noise_dim

        self._init_weights()

    # ── weight initialisation ─────────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, p in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(p)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(p)
                    elif 'bias' in name:
                        nn.init.zeros_(p)
                        # forget-gate bias = 1 (gates order: i, f, g, o)
                        H = m.hidden_size
                        p.data[H:2*H] = 1.0

    # ── text conditioning ─────────────────────────────────────────────────────
    def get_condition(self, sentences, device) -> torch.Tensor:
        """Text → (B, lstm_hidden). CLIP uses pooler_output (EOT token);
        mclip uses mask-aware mean over last_hidden_state."""
        with torch.no_grad():
            inputs = self.tokenizer(
                sentences, padding=True, truncation=True,
                max_length=77, return_tensors='pt'
            ).to(device)
            if self.text_encoder_type == 'mclip':
                hidden = self.text_encoder(**inputs).last_hidden_state
                attn = inputs['attention_mask'].unsqueeze(-1).float()
                raw = (hidden * attn).sum(dim=1) / attn.sum(dim=1).clamp(min=1e-9)
            else:
                raw = self.text_encoder(**inputs).pooler_output  # (B, clip_dim)
        return self.cond_proj(raw)   # (B, lstm_hidden)

    # ── diffusion forward ─────────────────────────────────────────────────────
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor,
                 noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: x_0 → x_t"""
        if noise is None:
            noise = torch.randn_like(x0)
        sa  = self.sqrt_acp[t]
        som = self.sqrt_one_minus_acp[t]
        while sa.dim() < x0.dim():
            sa  = sa.unsqueeze(-1)
            som = som.unsqueeze(-1)
        return sa * x0 + som * noise

    # ── core denoiser ─────────────────────────────────────────────────────────
    def denoise(self, x_t_active: torch.Tensor,
                t: torch.Tensor,
                cond: torch.Tensor) -> torch.Tensor:
        """
        x_t_active: (B, F, pose_input_dim [+ n_expr])
        t:          (B,)
        cond:       (B, lstm_hidden)
        returns:    (B, F, noise_dim)  predicted ε
        """
        B, Frames, _ = x_t_active.shape
        device = x_t_active.device

        # ── split pose / expression ───────────────────────────────────────────
        pose_flat = x_t_active[..., :self.pose_input_dim]  # (B, F, n_active*n_feats)

        if self.use_3d_input:
            # Per-joint MLP path (Phoenix MediaPipe / How2Sign-3D coords).
            theta_jp = pose_flat.view(B, Frames, self.n_active, self.n_feats)
            h_pose   = self.pose_per_joint(theta_jp)          # (B, F, n_active, ph)
            h_pose   = self.pose_flat_proj(h_pose.flatten(2)) # (B, F, lstm_hidden)
            h_pose   = self.pose_flat_norm(h_pose)
        else:
            # Expand to full (B, F, 53, n_feats) with zeros for bypassed joints
            theta_full = torch.zeros(
                B, Frames, N_JOINTS, self.n_feats, device=device, dtype=x_t_active.dtype
            )
            for new_i, orig_i in enumerate(self.active_joints):
                s = slice(new_i * self.n_feats, (new_i + 1) * self.n_feats)
                theta_full[:, :, orig_i, :] = pose_flat[..., s]

            # ── GNN pose encoder ──────────────────────────────────────────────
            h_full = self.pose_encoder(theta_full, cond)          # (B, F, 53, gnn_out)
            h_pose = h_full[:, :, self.active_joints, :]          # (B, F, n_active, gnn_out)
            h_pose = self.pose_flat_proj(h_pose.flatten(2))       # (B, F, lstm_hidden)
            h_pose = self.pose_flat_norm(h_pose)                  # critical: stabilize scale

        # ── expression encoder ────────────────────────────────────────────────
        parts = [h_pose]
        if self.use_expr:
            psi    = x_t_active[..., self.pose_input_dim:]    # (B, F, n_expr)
            h_expr = self.expr_encoder(psi, cond)              # (B, F, 10, expr_out)
            h_expr = self.expr_flat_proj(h_expr.flatten(2))    # (B, F, lstm_h//4)
            parts.append(h_expr)

        # ── timestep embedding (broadcast over F) ─────────────────────────────
        t_emb   = sinusoidal_embedding(t, self.lstm_hidden)         # (B, D)
        t_token = self.t_proj(t_emb).unsqueeze(1).expand(-1, Frames, -1)
        parts.append(t_token)

        # ── 4-layer LSTM (auto-regressive temporal processing) ────────────────
        lstm_in  = torch.cat(parts, dim=-1)        # (B, F, lstm_in)
        lstm_in  = self.pre_lstm_norm(lstm_in)
        lstm_out, _ = self.lstm(lstm_in)           # (B, F, lstm_hidden)
        lstm_out = self.post_lstm_norm(lstm_out)

        # ── regression head ───────────────────────────────────────────────────
        return self.regress_head(lstm_out)         # (B, F, noise_dim)

    # ── training forward ──────────────────────────────────────────────────────
    def forward(self, x_t: torch.Tensor,
                t: torch.Tensor,
                sentences,
                padding_mask=None,
                motion=None) -> torch.Tensor:
        """
        Returns predicted ε in FULL dataset space (same shape as x_t).
        Bypassed joints have zero predicted noise.

        x_t:          (B, T, D_full)   noisy motion from dataset
        t:            (B,)
        sentences:    List[str]
        padding_mask: (B, T) True=padded  (reserved; not used inside LSTM)
        motion:       ignored (API compatibility with train_v1.py)
        """
        device = x_t.device
        cond   = self.get_condition(list(sentences), device)

        # Build active input: active pose features [+ expression]
        x_active = x_t[:, :, self.tosave_slices]              # (B, T, pose_active)
        if self.use_expr:
            expr_start = len(self.all_slices)                  # 53*n_feats
            x_expr     = x_t[:, :, expr_start: expr_start + self.n_expr]
            x_active   = torch.cat([x_active, x_expr], dim=-1)

        eps_active = self.denoise(x_active, t, cond)           # (B, T, noise_dim)

        # Expand back to full dataset space
        out = torch.zeros_like(x_t)
        out[:, :, self.tosave_slices] = eps_active[..., :self.pose_input_dim].to(out.dtype)
        if self.use_expr:
            expr_start = len(self.all_slices)
            out[:, :, expr_start: expr_start + self.n_expr] = \
                eps_active[..., self.pose_input_dim:].to(out.dtype)
        return out

    # ── generation (DDIM) ─────────────────────────────────────────────────────
    @torch.no_grad()
    def generate(self, sentences, seq_len: int = 200,
                 device: str = 'cuda', num_steps: int = 50,
                 eta: float = 0.0) -> torch.Tensor:
        """
        DDIM sampling: x_T ~ N(0,I)  →  x_0  →  full joint space output.

        Returns: (B, T, D_full)
        """
        self.eval()
        B    = len(sentences) if isinstance(sentences, (list, tuple)) else sentences.shape[0]
        cond = self.get_condition(list(sentences), device)

        step  = self.T // num_steps
        times = list(reversed(range(0, self.T, step)))   # T-1 → 0

        # Start from pure noise in active-feature space
        x_t = torch.randn(B, seq_len, self.input_dim, device=device)

        for i, t_cur in enumerate(times):
            t_b    = torch.full((B,), t_cur, dtype=torch.long, device=device)
            alpha  = self.alphas_cumprod[t_cur]

            eps    = self.denoise(x_t, t_b, cond)          # (B, F, noise_dim)

            # Recover x_0 estimate
            x0     = (x_t - (1 - alpha).sqrt() * eps) / alpha.sqrt()
            x0     = x0.clamp(-10.0, 10.0)

            if i == len(times) - 1:
                x_t = x0
                break

            t_next     = times[i + 1]
            alpha_next = self.alphas_cumprod[t_next]

            sigma  = eta * (
                (1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next)
            ).clamp(min=0.0).sqrt()
            dir_xt = (1 - alpha_next - sigma ** 2).clamp(min=0.0).sqrt() * eps
            x_t    = alpha_next.sqrt() * x0 + dir_xt + sigma * torch.randn_like(x_t)

        # Expand to full dataset space
        D_full = len(self.all_slices) + (self.n_expr if self.use_expr else 0)
        out    = torch.zeros(B, seq_len, D_full, device=device, dtype=x_t.dtype)
        out[:, :, self.tosave_slices] = x_t[..., :self.pose_input_dim].to(out.dtype)
        if self.use_expr:
            expr_start = len(self.all_slices)
            out[:, :, expr_start: expr_start + self.n_expr] = \
                x_t[..., self.pose_input_dim:].to(out.dtype)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# NSA Loss  —  Eq. 7  with hand joints × 2
# ─────────────────────────────────────────────────────────────────────────────

LHAND_SET = set(LHAND_INDICES)
RHAND_SET = set(RHAND_INDICES)


def nsa_loss(eps_pred: torch.Tensor,
             eps_true: torch.Tensor,
             padding_mask: torch.Tensor,
             n_feats: int,
             active_joints: list,
             use_expr: bool = False,
             n_expr: int = 10,
             loss_type: str = 'mse',
             hand_joints_override: set = None) -> torch.Tensor:
    """
    Paper Eq. 7:   L_t = || ε_t − ε_Θ ||_2

    loss_type='mse'  : ||·||_2^2 (squared) — standard diffusion loss, may collapse.
    loss_type='l2'   : ||·||_2 (unsquared, per-joint) — matches paper literally;
                       gradient magnitude is constant 1/||err||, prevents zero
                       collapse that MSE suffers from when prediction → 0.

    Hand joints (left + right) contribute with weight 2× (5× when Phoenix).

    hand_joints_override: optional set of joint indices to treat as "hand"
        instead of the default SMPL-X LHAND/RHAND sets. Used by Phoenix where
        joint topology is MediaPipe (body 25 + lhand 21 + rhand 21).

    eps_pred, eps_true: (B, T, noise_dim)
    padding_mask:       (B, T)  True = padded
    """
    valid = (~padding_mask).float().unsqueeze(-1)   # (B, T, 1)

    loss = torch.tensor(0.0, device=eps_pred.device, dtype=eps_pred.dtype)
    n_pose_feats = len(active_joints) * n_feats

    # Uniform per-joint weighting (no hand boost). The hand-boosted variant
    # over-fits hand marginals at the expense of arm trajectories — see the
    # mode-collapse diagnosis in tools/diagnose_mode_collapse.py.
    def _per_joint(err):
        if loss_type == 'l2':
            # L2 norm per (B,T,joint), masked, averaged
            norm = torch.sqrt((err ** 2).sum(dim=-1) + 1e-6)   # (B, T)
            return (norm * valid.squeeze(-1)).sum() / (valid.sum() + 1e-8)
        else:
            return (err ** 2 * valid).sum() / (valid.sum() * n_feats + 1e-8)

    for new_i, orig_i in enumerate(active_joints):
        sl  = slice(new_i * n_feats, (new_i + 1) * n_feats)
        err = eps_pred[..., sl] - eps_true[..., sl]            # (B, T, n_feats)
        loss = loss + _per_joint(err)

    if use_expr:
        sl  = slice(n_pose_feats, n_pose_feats + n_expr)
        err = eps_pred[..., sl] - eps_true[..., sl]
        if loss_type == 'l2':
            norm = torch.sqrt((err ** 2).sum(dim=-1) + 1e-6)
            loss = loss + (norm * valid.squeeze(-1)).sum() / (valid.sum() + 1e-8)
        else:
            loss = loss + (err ** 2 * valid).sum() / (valid.sum() * n_expr + 1e-8)

    return loss
