"""Conditional diffusion model for three distances and two signed angles."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32)


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep features, shaped [batch, dim]."""
    half = dim // 2
    device = t.device
    t = t.float()
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(0, half, device=device).float() / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    """Gather 1D schedule tensor a[t] and reshape to broadcast over x."""
    out = a.gather(0, t)
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


def prepare_eps_cond(
    eps_cond: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Broadcast a raw epsilon pair to [batch, 2], or accept one pair per sample."""
    eps_cond = torch.as_tensor(eps_cond, device=device, dtype=dtype)

    if eps_cond.ndim == 1:
        if eps_cond.numel() != 2:
            raise ValueError(f"eps_cond must contain eps1 and eps2; got shape {tuple(eps_cond.shape)}")
        eps_cond = eps_cond.view(1, 2).expand(batch_size, 2)
    elif eps_cond.ndim == 2:
        if eps_cond.shape[1] != 2:
            raise ValueError(f"eps_cond must have shape [B, 2]; got {tuple(eps_cond.shape)}")
        if eps_cond.shape[0] == 1 and batch_size != 1:
            eps_cond = eps_cond.expand(batch_size, 2)
        elif eps_cond.shape[0] != batch_size:
            raise ValueError(f"eps_cond batch size {eps_cond.shape[0]} does not match x batch size {batch_size}")
    else:
        raise ValueError(f"eps_cond must have shape [2] or [B, 2]; got {tuple(eps_cond.shape)}")

    return eps_cond


def epsilon_features(eps_cond: torch.Tensor) -> torch.Tensor:
    """Raw epsilons and their pairwise polynomial features."""
    eps1 = eps_cond[:, 0:1]
    eps2 = eps_cond[:, 1:2]
    return torch.cat(
        [
            eps1,
            eps2,
            eps1 + eps2,
            eps1 - eps2,
            eps1 * eps2,
            eps1.square(),
            eps2.square(),
        ],
        dim=-1,
    )


class FiLMResidualBlock(nn.Module):
    def __init__(self, width: int, cond_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta
        h = self.fc1(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.norm2(h)
        h = self.fc2(h)
        return x + h


class FiLMResMLP(nn.Module):
    """Predict noise from the coordinates, diffusion time, and raw LJ epsilons."""

    def __init__(
        self,
        x_dim: int = 5,
        width: int = 128,
        depth: int = 6,
        time_dim: int = 64,
        cond_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.x_proj = nn.Linear(x_dim, width)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.eps_mlp = nn.Sequential(
            nn.Linear(7, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(2 * cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        self.blocks = nn.ModuleList(
            [FiLMResidualBlock(width=width, cond_dim=cond_dim, dropout=dropout) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, x_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, eps_cond: torch.Tensor) -> torch.Tensor:
        eps_cond = prepare_eps_cond(eps_cond, x_t.shape[0], x_t.device, x_t.dtype)

        t_emb = timestep_embedding(t, self.time_dim).to(dtype=x_t.dtype)
        t_h = self.time_mlp(t_emb)
        eps_h = self.eps_mlp(epsilon_features(eps_cond))
        cond = self.cond_mlp(torch.cat([t_h, eps_h], dim=-1))

        h = self.x_proj(x_t)
        for block in self.blocks:
            h = block(h, cond)
        h = self.out_norm(h)
        h = F.silu(h)
        return self.out(h)


class AngleDiffusion(nn.Module):
    """DDPM training and DDIM sampling in normalized geometry coordinates."""

    def __init__(self, model: nn.Module, num_steps: int = 1000):
        super().__init__()
        self.model = model
        self.num_steps = num_steps

        betas = linear_beta_schedule(num_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.ones(1), alpha_bars[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_prev", alpha_bars_prev)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_recip_alpha_bars", torch.sqrt(1.0 / alpha_bars))
        self.register_buffer(
            "sqrt_recipm1_alpha_bars",
            torch.sqrt(torch.clamp(1.0 / alpha_bars - 1.0, min=0.0)),
        )

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        mean = extract(self.sqrt_alpha_bars, t, x0.shape) * x0
        std = extract(self.sqrt_one_minus_alpha_bars, t, x0.shape)
        x_t = mean + std * noise
        return x_t, noise

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps_hat: torch.Tensor) -> torch.Tensor:
        return (
            extract(self.sqrt_recip_alpha_bars, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alpha_bars, t, x_t.shape) * eps_hat
        )

    def training_loss(self, x0: torch.Tensor, eps_cond: torch.Tensor) -> torch.Tensor:
        bsz = x0.shape[0]
        eps_cond = prepare_eps_cond(eps_cond, bsz, x0.device, x0.dtype)
        t = torch.randint(0, self.num_steps, (bsz,), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        pred = self.model(x_t, t, eps_cond)
        return F.mse_loss(pred, noise)

    def ddim_sample(
        self,
        eps_cond: torch.Tensor,
        shape: tuple[int, int],
        sample_steps: int = 100,
        eta: float = 0.0,
        clamp_x0: bool = True,
        return_all: bool = False,
    ) -> torch.Tensor:
        """Draw normalized coordinates; keep gradients for epsilon optimization.

        With eta=0, the initial Gaussian draw is the only sampling noise.
        return_all includes that draw and every subsequent sampling step.
        """
        device = self.alpha_bars.device
        x = torch.randn(shape, device=device)
        eps_cond = prepare_eps_cond(eps_cond, shape[0], device, x.dtype)

        times = torch.linspace(self.num_steps - 1, 0, sample_steps, device=device).long()
        traj = [x] if return_all else []

        for i, t_now in enumerate(times):
            t = torch.full((shape[0],), int(t_now.item()), device=device, dtype=torch.long)
            eps_hat = self.model(x, t, eps_cond)
            x0_hat = self.predict_x0_from_eps(x, t, eps_hat)
            if clamp_x0:
                x0_hat = x0_hat.clamp(-1.0, 1.0)

            if i == len(times) - 1:
                x = x0_hat
                if return_all:
                    traj.append(x)
                break

            t_prev_scalar = int(times[i + 1].item())
            t_prev = torch.full((shape[0],), t_prev_scalar, device=device, dtype=torch.long)
            alpha_bar_t = extract(self.alpha_bars, t, x.shape)
            alpha_bar_prev = extract(self.alpha_bars, t_prev, x.shape)

            sigma = eta * torch.sqrt(
                torch.clamp((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t), min=0.0)
                * torch.clamp(1.0 - alpha_bar_t / alpha_bar_prev, min=0.0)
            )
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)

            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma ** 2, min=0.0)) * eps_hat
            x = torch.sqrt(alpha_bar_prev) * x0_hat + dir_xt + sigma * noise
            if return_all:
                traj.append(x)

        if return_all:
            return torch.stack(traj, dim=1)
        return x

