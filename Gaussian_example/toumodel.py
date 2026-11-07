import torch
from typing import Optional, List, Tuple
import random
import math
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Diffusion
# ----------------------------
class Diffusion1D:
    def __init__(self, T: int, beta_start: float, beta_end: float, device: torch.device):
        self.T = int(T)
        betas = torch.linspace(beta_start, beta_end, self.T, device=device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """x_t = sqrt(ab_t)*x0 + sqrt(1-ab_t)*eps"""
        if eps is None:
            eps = torch.randn_like(x0)
        ab = self.alpha_bars[t].view(-1, 1).clamp(1e-12, 1.0)
        xt = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps
        return xt, eps



# ----------------------------
# Integer <-> bits (model never maps idx -> float)
# ----------------------------
def int_to_bits(idx: torch.Tensor, n_bits: int) -> torch.Tensor:
    """idx: (B,) long -> (B,n_bits) float bits, LSB-first"""
    idx = idx.long()
    shifts = torch.arange(n_bits, device=idx.device, dtype=torch.long)
    return ((idx[:, None] >> shifts[None, :]) & 1).float()


def bits_to_int(bits: torch.Tensor) -> torch.Tensor:
    """bits: (B,n_bits) float/bool LSB-first -> (B,) long"""
    n_bits = bits.shape[1]
    weights = (2 ** torch.arange(n_bits, device=bits.device, dtype=torch.long)).view(1, -1)
    return (bits.long() * weights).sum(dim=1)


def gumbel_noise_like(x: torch.Tensor) -> torch.Tensor:
    u = torch.rand_like(x).clamp(1e-6, 1.0 - 1e-6)
    return -torch.log(-torch.log(u))

# ----------------------------
# Time embedding
# ----------------------------
def sinusoidal_timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) long -> (B, dim) float"""
    half = dim // 2
    t = t.float()
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=t.device).float() / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

# ----------------------------
# Conditioning encoder (bits / logits -> cond vector)
# ----------------------------
class CondEncoderBits(nn.Module):
    def __init__(self, m_bits: int = 7, v_bits: int = 4, out_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.m_bits = m_bits
        self.v_bits = v_bits
        self.net = nn.Sequential(
            nn.Linear(m_bits + v_bits, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        *,
        m_idx: Optional[torch.Tensor] = None,
        v_idx: Optional[torch.Tensor] = None,
        m_bits: Optional[torch.Tensor] = None,
        v_bits: Optional[torch.Tensor] = None,
        m_bit_logits: Optional[torch.Tensor] = None,
        v_bit_logits: Optional[torch.Tensor] = None,
        tau: float = 1.0,
        g_m: Optional[torch.Tensor] = None,
        g_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if m_bits is not None and v_bits is not None:
            mb, vb = m_bits, v_bits
        elif m_bit_logits is not None and v_bit_logits is not None:
            if g_m is None:
                g_m = gumbel_noise_like(m_bit_logits).detach()
            if g_v is None:
                g_v = gumbel_noise_like(v_bit_logits).detach()
            mb = torch.sigmoid((m_bit_logits + g_m) / max(tau, 1e-6))
            vb = torch.sigmoid((v_bit_logits + g_v) / max(tau, 1e-6))
        else:
            if m_idx is None or v_idx is None:
                raise ValueError("Provide either (m_idx,v_idx) OR logits OR bits.")
            mb = int_to_bits(m_idx, self.m_bits)
            vb = int_to_bits(v_idx, self.v_bits)

        return self.net(torch.cat([mb, vb], dim=1))

# ----------------------------
# Epsilon model
# ----------------------------
class EpsModel(nn.Module):
    def __init__(self, cond_dim: int = 32, time_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.time_dim = time_dim
        self.cond = CondEncoderBits(out_dim=cond_dim)
        self.net = nn.Sequential(
            nn.Linear(1 + cond_dim + time_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        m_idx: Optional[torch.Tensor] = None,
        v_idx: Optional[torch.Tensor] = None,
        m_bits: Optional[torch.Tensor] = None,
        v_bits: Optional[torch.Tensor] = None,
        m_bit_logits: Optional[torch.Tensor] = None,
        v_bit_logits: Optional[torch.Tensor] = None,
        tau: float = 1.0,
        g_m: Optional[torch.Tensor] = None,
        g_v: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        te = sinusoidal_timestep_embedding(t, self.time_dim)
        ce = self.cond(
            m_idx=m_idx, v_idx=v_idx,
            m_bits=m_bits, v_bits=v_bits,
            m_bit_logits=m_bit_logits, v_bit_logits=v_bit_logits,
            tau=tau, g_m=g_m, g_v=g_v,
        )
        return self.net(torch.cat([x_t, ce, te], dim=1))
