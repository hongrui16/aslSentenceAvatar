"""Part-wise VQ-VAE motion tokenizer (roadmap 4.1): four codebooks (body / left hand / right hand / face), 1D-conv
encoder-decoder with 4x temporal downsampling (25 fps -> 6.25 tokens/s per part), EMA codebook with dead-code reset
(T2M-GPT recipe). Reconstruction targets are rot6d (+ expression for the face); the trainer adds a FK joint loss."""
import torch, torch.nn as nn, torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, c, dil):
        super().__init__(); self.net = nn.Sequential(nn.ReLU(), nn.Conv1d(c, c, 3, padding=dil, dilation=dil), nn.ReLU(), nn.Conv1d(c, c, 1))
    def forward(self, x): return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, din, c=256, down=2):
        super().__init__(); L = [nn.Conv1d(din, c, 3, padding=1), nn.ReLU()]
        for _ in range(down): L += [nn.Conv1d(c, c, 4, stride=2, padding=1), ResBlock(c, 1), ResBlock(c, 3), ResBlock(c, 9)]
        self.net = nn.Sequential(*L)
    def forward(self, x): return self.net(x)  # (B, c, T/4)


class Decoder(nn.Module):
    def __init__(self, dout, c=256, up=2):
        super().__init__(); L = []
        for _ in range(up): L += [ResBlock(c, 9), ResBlock(c, 3), ResBlock(c, 1), nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv1d(c, c, 3, padding=1), nn.ReLU()]
        L += [nn.Conv1d(c, c, 3, padding=1), nn.ReLU(), nn.Conv1d(c, dout, 3, padding=1)]; self.net = nn.Sequential(*L)
    def forward(self, z): return self.net(z)


class EMAQuantizer(nn.Module):
    def __init__(self, K=512, d=256, decay=0.99, eps=1e-5, reset_every=200, reset_thresh=1.0):
        super().__init__(); self.K, self.d, self.decay, self.eps = K, d, decay, eps
        self.register_buffer('codebook', torch.randn(K, d)); self.register_buffer('ema_count', torch.zeros(K)); self.register_buffer('ema_sum', self.codebook.clone())
        self.register_buffer('usage', torch.zeros(K)); self.register_buffer('initialized', torch.zeros(1)); self.reset_every, self.reset_thresh = reset_every, reset_thresh; self.step = 0

    def forward(self, z):  # z: (B, d, T') -> zq (B, d, T'), idx (B, T'), commit loss, perplexity
        B, d, T = z.shape; flat = z.permute(0, 2, 1).reshape(-1, d)
        if self.training and self.initialized.item() == 0:  # data-dependent init: codebook = random encoder outputs of the first batch
            with torch.no_grad():
                r = flat[torch.randint(0, flat.shape[0], (self.K,), device=flat.device)].float(); self.codebook.copy_(r); self.ema_sum.copy_(r); self.ema_count.fill_(1.0); self.initialized.fill_(1)
        flat = flat.float(); dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.T + self.codebook.pow(2).sum(1)[None]
        idx = dist.argmin(1); onehot = F.one_hot(idx, self.K).float(); zq = (onehot @ self.codebook).view(B, T, d).permute(0, 2, 1).to(z.dtype)
        if self.training:
            with torch.no_grad():
                n = onehot.sum(0); self.ema_count.mul_(self.decay).add_(n, alpha=1 - self.decay); self.ema_sum.mul_(self.decay).add_(onehot.T @ flat, alpha=1 - self.decay)
                cnt = (self.ema_count + self.eps) / (self.ema_count.sum() + self.K * self.eps) * self.ema_count.sum(); self.codebook.copy_(self.ema_sum / cnt[:, None])
                self.usage.mul_(0.99).add_(n / max(n.sum(), 1), alpha=0.01); self.step += 1
                if self.step % self.reset_every == 0:  # dead-code reset: codes with tiny EMA count take random current encoder outputs
                    dead = self.ema_count < self.reset_thresh
                    if dead.any():
                        r = flat[torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)]; self.codebook[dead] = r; self.ema_sum[dead] = r; self.ema_count[dead] = 1.0
        commit = F.mse_loss(z.float(), zq.detach().float()); zq = z + (zq - z).detach()
        probs = onehot.mean(0); perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())
        return zq, idx.view(B, T), commit, perplexity


class PartVQVAE(nn.Module):
    def __init__(self, part_dims, K=512, c=256, down=2):
        super().__init__(); self.parts = list(part_dims); self.down = 2 ** down
        self.enc = nn.ModuleDict({p: Encoder(dim, c, down) for p, dim in part_dims.items()}); self.vq = nn.ModuleDict({p: EMAQuantizer(K, c) for p in part_dims}); self.dec = nn.ModuleDict({p: Decoder(dim, c, down) for p, dim in part_dims.items()})

    def forward(self, feats):  # feats[p]: (B, T, dim)
        out, idx, commit, ppl = {}, {}, {}, {}
        for p in self.parts:
            z = self.enc[p](feats[p].permute(0, 2, 1)); zq, idx[p], commit[p], ppl[p] = self.vq[p](z); out[p] = self.dec[p](zq).permute(0, 2, 1)
        return out, idx, commit, ppl

    @torch.no_grad()
    def encode(self, feats):
        return {p: self.vq[p](self.enc[p](feats[p].permute(0, 2, 1)))[1] for p in self.parts}

    @torch.no_grad()
    def decode(self, idx):
        return {p: self.dec[p](self.vq[p].codebook[idx[p]].permute(0, 2, 1)).permute(0, 2, 1) for p in self.parts}
