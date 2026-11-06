import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10_000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        t = t.float()
        half = self.dim // 2
        freq = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=t.dtype)
            / half
        )
        angles = t[:, None] * freq[None]
        emb = torch.cat((angles.cos(), angles.sin()), dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class FiLMResBlock(nn.Module):
    def __init__(self, channels, cond_dim, dilation=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.film = nn.Linear(cond_dim, 2 * channels)
        self.conv1 = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )
        self.conv2 = nn.Conv2d(
            channels, channels, 3, padding=dilation, dilation=dilation
        )

    def forward(self, x, cond):
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.norm1(x)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv1(F.silu(h))
        h = self.conv2(F.silu(self.norm2(h)))
        return (x + h) / math.sqrt(2.0)


def symmetric_zero_diagonal(x):
    x = 0.5 * (x + x.transpose(-1, -2))
    return x - torch.diag_embed(torch.diagonal(x, dim1=-2, dim2=-1))


class FiLMDDPM(nn.Module):
    """Symmetric noise predictor for 10x10 distance matrices.

    The condition is [B,10] epsilons in atom-ID order: chain 1 then chain 2.
    Classifier-free guidance drops the whole epsilon vector, never the timestep.
    Output is constrained to be symmetric with zero diagonal.
    """

    def __init__(self, channels=64, cond_dim=128, cond_drop_prob=0.1):
        super().__init__()
        self.cond_drop_prob = cond_drop_prob

        time_dim = 64
        eps_dim = 64

        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.eps_embed = nn.Sequential(
            nn.Linear(10, eps_dim),
            nn.SiLU(),
            nn.Linear(eps_dim, eps_dim),
        )
        self.null_eps = nn.Parameter(torch.zeros(eps_dim))
        self.cond_embed = nn.Sequential(
            nn.Linear(time_dim + eps_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.input = nn.Conv2d(1, channels, 3, padding=1)
        self.blocks = nn.ModuleList(
            FiLMResBlock(channels, cond_dim, dilation=d)
            for d in (1, 1, 2, 2, 1, 1)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 3, padding=1),
        )

    def _condition(self, t, epsilons, cond_drop_mask):
        t_emb = self.time_embed(t)
        eps_emb = self.eps_embed(epsilons)

        if cond_drop_mask is None:
            if self.training and self.cond_drop_prob > 0:
                cond_drop_mask = torch.rand(
                    epsilons.shape[0], device=epsilons.device
                ) < self.cond_drop_prob
            else:
                cond_drop_mask = torch.zeros(
                    epsilons.shape[0], dtype=torch.bool, device=epsilons.device
                )

        null = self.null_eps[None].expand_as(eps_emb)
        eps_emb = torch.where(cond_drop_mask[:, None], null, eps_emb)
        return self.cond_embed(torch.cat((t_emb, eps_emb), dim=-1))

    def forward(self, x_t, t, epsilons, cond_drop_mask=None):
        squeeze_channel = x_t.ndim == 3
        if squeeze_channel:
            x_t = x_t[:, None]
        if x_t.ndim != 4 or x_t.shape[1:] != (1, 10, 10):
            raise ValueError("x_t must have shape [B,10,10] or [B,1,10,10]")

        if epsilons.shape != (x_t.shape[0], 10):
            raise ValueError("epsilons must have shape [B,10] in atom-ID order")
        cond = self._condition(t, epsilons.to(x_t), cond_drop_mask)
        h = self.input(x_t)
        for block in self.blocks:
            h = block(h, cond)
        noise = symmetric_zero_diagonal(self.output(h))
        return noise[:, 0] if squeeze_channel else noise

    def predict_cfg(self, x_t, t, epsilons, guidance_scale=2.0):
        """Classifier-free-guided symmetric noise prediction for sampling."""
        batch = x_t.shape[0]
        device = x_t.device
        conditioned = torch.zeros(batch, dtype=torch.bool, device=device)
        unconditioned = torch.ones(batch, dtype=torch.bool, device=device)

        eps_cond = self(x_t, t, epsilons, conditioned)
        eps_uncond = self(x_t, t, epsilons, unconditioned)
        return eps_uncond + guidance_scale * (eps_cond - eps_uncond)

