"""
PhonoSignActorsModel
====================
Extends NeuralSignActorsModel with a second conditioning stream from the
pseudo-gloss string (extracted from the English sentence).

Two operating modes (selected by cfg.USE_CROSS_ATTN):

  M1 — global gloss concat + FiLM
       CLIP(gloss_string) → pooled → project → add to cond_sentence
       Identical structure to NSA, just a richer global condition.

  M2 — global gloss concat + per-frame cross-attention (default)
       Same global condition as M1, PLUS:
         Motion frames (query) × gloss CLIP token sequence (key/value)
       Cross-attention output is concatenated into the LSTM input stream,
       so each frame can softly attend to the most relevant gloss tokens.

API:
    forward(x_t, t, sentences, padding_mask=None, gloss_strings=None, motion=None)

If `gloss_strings` is None, the model gracefully falls back to vanilla NSA
behaviour (zero gloss condition, cross-attention disabled).
"""
import torch
import torch.nn as nn

from network.NeuralSignActorsModel import NeuralSignActorsModel, N_JOINTS, sinusoidal_embedding


class PhonoSignActorsModel(NeuralSignActorsModel):

    def __init__(self, cfg):
        super().__init__(cfg)

        self.use_cross_attn = getattr(cfg, 'USE_CROSS_ATTN', True)
        self.gloss_n_heads  = getattr(cfg, 'GLOSS_N_HEADS', 8)
        lstm_hidden         = self.lstm_hidden
        clip_dim            = self.text_encoder.config.hidden_size

        # ── Gloss global projection (used in both M1 and M2) ──────────────
        self.gloss_cond_proj = nn.Sequential(
            nn.Linear(clip_dim, lstm_hidden),
            nn.GELU(),
            nn.Linear(lstm_hidden, lstm_hidden),
        )

        # ── Per-frame cross-attention (M2 only) ───────────────────────────
        if self.use_cross_attn:
            self.gloss_kv_proj = nn.Linear(clip_dim, lstm_hidden)
            self.gloss_q_proj  = nn.Linear(lstm_hidden, lstm_hidden)
            self.gloss_cross_attn = nn.MultiheadAttention(
                embed_dim=lstm_hidden,
                num_heads=self.gloss_n_heads,
                batch_first=True,
            )
            self.gloss_attn_norm = nn.LayerNorm(lstm_hidden)

            # Rebuild LSTM with +lstm_hidden input (cross-attn feature appended)
            new_lstm_in = self.lstm.input_size + lstm_hidden
            self.lstm = nn.LSTM(
                input_size  = new_lstm_in,
                hidden_size = lstm_hidden,
                num_layers  = cfg.LSTM_N_LAYERS,
                batch_first = True,
                dropout     = 0.1 if cfg.LSTM_N_LAYERS > 1 else 0.0,
            )
            # Rebuild pre-LN to match new LSTM input size
            self.pre_lstm_norm = nn.LayerNorm(new_lstm_in)

        self._init_gloss_weights()

    def _init_gloss_weights(self):
        for m in [self.gloss_cond_proj]:
            for p in m.modules():
                if isinstance(p, nn.Linear):
                    nn.init.xavier_uniform_(p.weight)
                    if p.bias is not None:
                        nn.init.zeros_(p.bias)
        if self.use_cross_attn:
            for mod in [self.gloss_kv_proj, self.gloss_q_proj]:
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            # Re-init the replaced LSTM weights
            for name, p in self.lstm.named_parameters():
                if 'weight_ih' in name: nn.init.xavier_uniform_(p)
                elif 'weight_hh' in name: nn.init.orthogonal_(p)
                elif 'bias' in name:
                    nn.init.zeros_(p)
                    H = self.lstm.hidden_size
                    p.data[H:2*H] = 1.0   # forget-gate bias = 1

    # ── encode gloss_string via CLIP ──────────────────────────────────────
    def get_gloss_condition(self, gloss_strings, device, return_tokens=False):
        """
        Returns:
            pooled_proj: (B, lstm_hidden)                          always
            tokens_raw : (B, N, clip_dim)    (only if return_tokens)
            attn_mask  : (B, N) bool, True=valid token (only if return_tokens)
        """
        # Replace empty strings with a placeholder so CLIP tokenizer doesn't choke
        cleaned = [g if g.strip() else 'none' for g in gloss_strings]
        with torch.no_grad():
            inputs = self.tokenizer(
                cleaned, padding=True, truncation=True,
                max_length=77, return_tensors='pt'
            ).to(device)
            out = self.text_encoder(**inputs)
            pooled = out.pooler_output        # (B, clip_dim)
            tokens = out.last_hidden_state    # (B, N, clip_dim)
            attn_mask = inputs.attention_mask.bool()  # (B, N)
        pooled_proj = self.gloss_cond_proj(pooled)
        if return_tokens:
            return pooled_proj, tokens, attn_mask
        return pooled_proj

    # ── override denoise() to inject per-frame cross-attention ────────────
    def denoise_with_gloss(self, x_t_active, t, cond, gloss_kv=None, gloss_mask=None):
        B, Frames, _ = x_t_active.shape
        device = x_t_active.device

        # ── rebuild full 53-joint tensor (same as parent) ─────────────────
        pose_flat = x_t_active[..., :self.pose_input_dim]
        theta_full = torch.zeros(B, Frames, N_JOINTS, self.n_feats,
                                 device=device, dtype=x_t_active.dtype)
        for new_i, orig_i in enumerate(self.active_joints):
            s = slice(new_i * self.n_feats, (new_i + 1) * self.n_feats)
            theta_full[:, :, orig_i, :] = pose_flat[..., s]

        # ── GNN pose encoder ──────────────────────────────────────────────
        h_full = self.pose_encoder(theta_full, cond)
        h_pose = h_full[:, :, self.active_joints, :]
        h_pose = self.pose_flat_proj(h_pose.flatten(2))   # (B, F, lstm_hidden)

        parts = [h_pose]

        # ── expression branch ─────────────────────────────────────────────
        if self.use_expr:
            psi    = x_t_active[..., self.pose_input_dim:]
            h_expr = self.expr_encoder(psi, cond)
            h_expr = self.expr_flat_proj(h_expr.flatten(2))
            parts.append(h_expr)

        # ── timestep ──────────────────────────────────────────────────────
        t_emb = sinusoidal_embedding(t, self.lstm_hidden)
        t_tok = self.t_proj(t_emb).unsqueeze(1).expand(-1, Frames, -1)
        parts.append(t_tok)

        # ── per-frame gloss cross-attention (M2 only) ─────────────────────
        if self.use_cross_attn and gloss_kv is not None:
            q = self.gloss_q_proj(h_pose)                # (B, F, lstm_hidden)
            key_padding_mask = ~gloss_mask if gloss_mask is not None else None
            attn_out, _ = self.gloss_cross_attn(
                query=q, key=gloss_kv, value=gloss_kv,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            attn_out = self.gloss_attn_norm(attn_out)
            parts.append(attn_out)
        elif self.use_cross_attn:
            # No gloss strings provided but model was built with cross-attn;
            # append zeros to keep LSTM input dim consistent.
            parts.append(torch.zeros(B, Frames, self.lstm_hidden,
                                     device=device, dtype=h_pose.dtype))

        lstm_in     = torch.cat(parts, dim=-1)
        lstm_in     = self.pre_lstm_norm(lstm_in)
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out    = self.post_lstm_norm(lstm_out)
        return self.regress_head(lstm_out)

    # ── training / inference forward ──────────────────────────────────────
    def forward(self, x_t, t, sentences, padding_mask=None,
                gloss_strings=None, motion=None):
        device = x_t.device

        cond_sentence = self.get_condition(list(sentences), device)

        gloss_kv   = None
        gloss_mask = None
        cond_gloss = torch.zeros_like(cond_sentence)

        if gloss_strings is not None:
            gl = list(gloss_strings)
            if self.use_cross_attn:
                cond_gloss, tokens_raw, gloss_mask = self.get_gloss_condition(
                    gl, device, return_tokens=True)
                gloss_kv = self.gloss_kv_proj(tokens_raw)
            else:
                cond_gloss = self.get_gloss_condition(gl, device)

        cond = cond_sentence + cond_gloss

        # Build active input (same as parent)
        x_active = x_t[:, :, self.tosave_slices]
        if self.use_expr:
            expr_start = len(self.all_slices)
            x_expr     = x_t[:, :, expr_start: expr_start + self.n_expr]
            x_active   = torch.cat([x_active, x_expr], dim=-1)

        eps_active = self.denoise_with_gloss(
            x_active, t, cond, gloss_kv=gloss_kv, gloss_mask=gloss_mask)

        out = torch.zeros_like(x_t)
        out[:, :, self.tosave_slices] = eps_active[..., :self.pose_input_dim].to(out.dtype)
        if self.use_expr:
            expr_start = len(self.all_slices)
            out[:, :, expr_start: expr_start + self.n_expr] = \
                eps_active[..., self.pose_input_dim:].to(out.dtype)
        return out

    # ── generation (DDIM) with gloss conditioning ─────────────────────────
    @torch.no_grad()
    def generate(self, sentences, gloss_strings=None, seq_len=200,
                 device='cuda', num_steps=50, eta=0.0):
        self.eval()
        B = len(sentences) if isinstance(sentences, (list, tuple)) else sentences.shape[0]

        cond_sentence = self.get_condition(list(sentences), device)
        gloss_kv   = None
        gloss_mask = None
        cond_gloss = torch.zeros_like(cond_sentence)
        if gloss_strings is not None:
            gl = list(gloss_strings)
            if self.use_cross_attn:
                cond_gloss, tokens_raw, gloss_mask = self.get_gloss_condition(
                    gl, device, return_tokens=True)
                gloss_kv = self.gloss_kv_proj(tokens_raw)
            else:
                cond_gloss = self.get_gloss_condition(gl, device)
        cond = cond_sentence + cond_gloss

        step  = self.T // num_steps
        times = list(reversed(range(0, self.T, step)))

        x_t = torch.randn(B, seq_len, self.input_dim, device=device)

        for i, t_cur in enumerate(times):
            t_b   = torch.full((B,), t_cur, dtype=torch.long, device=device)
            alpha = self.alphas_cumprod[t_cur]

            eps = self.denoise_with_gloss(x_t, t_b, cond,
                                          gloss_kv=gloss_kv, gloss_mask=gloss_mask)

            x0 = (x_t - (1 - alpha).sqrt() * eps) / alpha.sqrt()
            x0 = x0.clamp(-10.0, 10.0)

            if i == len(times) - 1:
                x_t = x0; break

            t_next     = times[i + 1]
            alpha_next = self.alphas_cumprod[t_next]
            sigma  = eta * ((1 - alpha_next) / (1 - alpha) *
                            (1 - alpha / alpha_next)).clamp(min=0.0).sqrt()
            dir_xt = (1 - alpha_next - sigma ** 2).clamp(min=0.0).sqrt() * eps
            x_t    = alpha_next.sqrt() * x0 + dir_xt + sigma * torch.randn_like(x_t)

        D_full = len(self.all_slices) + (self.n_expr if self.use_expr else 0)
        out    = torch.zeros(B, seq_len, D_full, device=device, dtype=x_t.dtype)
        out[:, :, self.tosave_slices] = x_t[..., :self.pose_input_dim].to(out.dtype)
        if self.use_expr:
            expr_start = len(self.all_slices)
            out[:, :, expr_start: expr_start + self.n_expr] = \
                x_t[..., self.pose_input_dim:].to(out.dtype)
        return out
