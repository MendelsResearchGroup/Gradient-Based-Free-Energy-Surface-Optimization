import math
import random
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from toumodel import Diffusion1D, EpsModel, int_to_bits, gumbel_noise_like, bits_to_int


# ----------------------------
# Config (edit these)
# ----------------------------
SEED = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Diffusion schedule
T = 100
BETA_START = 1e-4
BETA_END = 0.1

# Training
TRAIN_STEPS = 6000
BATCH_SIZE = 1024
LR = 2e-4
LOG_EVERY = 500

# Disjoint training ranges
M_TRAIN_RANGES = ((0, 41), (60, 101))
V_TRAIN_RANGES = ((1, 4), (6, 11))

# Evaluation conditions
T1_COND = (90, 1)
T2_COND = (20, 8)

# Differentiable optimization
DDIM_STEPS = 40
OPT_ITERS = 140
OPT_LR = 0.12
TAU = 0.7
N_OPT = 2048

# KDE KL settings
KDE_H = 0.06
GRID_MIN = -1.0
GRID_MAX = 2.0
GRID_POINTS = 400

# Plot output
PLOT_PATH = "T1_T2_generated_trajectories.png"


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------
# Target distribution
# ----------------------------
def sample_target_given_ints(
    n: int,
    device: torch.device,
    m_idx: int,
    v_idx: int,
) -> torch.Tensor:
    m = torch.full((n, 1), float(m_idx) / 100.0, device=device)
    v = torch.full((n, 1), float(v_idx) / 100.0, device=device)
    return m + torch.sqrt(v) * torch.randn(n, 1, device=device)


def _allowed_from_ranges(
    ranges: Tuple[Tuple[int, int], ...],
    device: torch.device,
) -> torch.Tensor:
    allowed = []
    for lo, hi in ranges:
        allowed.extend(list(range(int(lo), int(hi))))

    if not allowed:
        raise ValueError("No allowed indices.")

    return torch.tensor(allowed, device=device, dtype=torch.long)


def sample_target_batch(
    n: int,
    device: torch.device,
    m_ranges: Tuple[Tuple[int, int], ...],
    v_ranges: Tuple[Tuple[int, int], ...],
):
    m_allowed = _allowed_from_ranges(m_ranges, device)
    v_allowed = _allowed_from_ranges(v_ranges, device)

    m_idx = m_allowed[torch.randint(0, m_allowed.numel(), (n,), device=device)]
    v_idx = v_allowed[torch.randint(0, v_allowed.numel(), (n,), device=device)]

    m = m_idx.float().view(-1, 1) / 100.0
    v = v_idx.float().view(-1, 1) / 100.0
    x0 = m + torch.sqrt(v) * torch.randn(n, 1, device=device)

    return x0, m_idx, v_idx


# ----------------------------
# Training
# ----------------------------
def train(model: EpsModel, diff: Diffusion1D) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for step in range(1, TRAIN_STEPS + 1):
        x0, m_idx, v_idx = sample_target_batch(
            BATCH_SIZE,
            DEVICE,
            M_TRAIN_RANGES,
            V_TRAIN_RANGES,
        )

        t = torch.randint(0, diff.T, (BATCH_SIZE,), device=DEVICE, dtype=torch.long)
        x_t, eps = diff.q_sample(x0, t)
        eps_hat = model(x_t, t, m_idx=m_idx, v_idx=v_idx)

        loss = F.mse_loss(eps_hat, eps)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % LOG_EVERY == 0 or step == 1:
            print(f"train step {step:5d}/{TRAIN_STEPS} | loss {loss.item():.4f}")


# ----------------------------
# Differentiable DDIM sampler
# ----------------------------
def ddim_sample_diff(
    model: EpsModel,
    diff: Diffusion1D,
    x_init: torch.Tensor,
    *,
    steps: int,
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
    model.eval()

    B = x_init.shape[0]
    x = x_init
    ts = torch.linspace(diff.T - 1, 0, steps, device=x.device).round().long()

    for i in range(len(ts)):
        t_int = int(ts[i].item())
        t = torch.full((B,), t_int, device=x.device, dtype=torch.long)

        eps_hat = model(
            x,
            t,
            m_idx=m_idx,
            v_idx=v_idx,
            m_bits=m_bits,
            v_bits=v_bits,
            m_bit_logits=m_bit_logits,
            v_bit_logits=v_bit_logits,
            tau=tau,
            g_m=g_m,
            g_v=g_v,
        )

        ab_t = diff.alpha_bars[t_int].clamp(1e-12, 1.0)
        t_prev = int(ts[i + 1].item()) if (i + 1) < len(ts) else 0
        ab_prev = diff.alpha_bars[t_prev].clamp(1e-12, 1.0)

        x0_hat = (x - torch.sqrt(1.0 - ab_t) * eps_hat) / torch.sqrt(ab_t)
        x = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1.0 - ab_prev) * eps_hat

    return x


# ----------------------------
# KDE and KL
# ----------------------------
def kde_density_on_grid(
    samples: torch.Tensor,
    grid: torch.Tensor,
    h: float,
) -> torch.Tensor:
    s = samples.view(-1, 1)
    g = grid.view(1, -1)

    z = (g - s) / h
    k = torch.exp(-0.5 * z * z)

    return k.mean(dim=0) / (math.sqrt(2.0 * math.pi) * h)


def kl_kde(
    p_samples: torch.Tensor,
    q_density: torch.Tensor,
    grid: torch.Tensor,
    h: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    p = kde_density_on_grid(p_samples, grid, h)
    dx = (grid[1] - grid[0]).clamp_min(1e-12)

    p = p / (p.sum() * dx + eps)
    q = q_density / (q_density.sum() * dx + eps)

    p = p.clamp_min(eps)
    q = q.clamp_min(eps)

    return (p * (p.log() - q.log())).sum() * dx


# ----------------------------
# Optimize condition logits through sampler
# ----------------------------
def optimize_condition_via_sampler(
    model: EpsModel,
    diff: Diffusion1D,
    *,
    start_cond: Tuple[int, int],
    target_cond: Tuple[int, int],
) -> Tuple[List[Tuple[int, float, torch.Tensor, torch.Tensor]], Tuple[int, int], List[float]]:
    start_m_idx, start_v_idx = start_cond
    target_m_idx, target_v_idx = target_cond

    x_init = torch.randn(N_OPT, 1, device=DEVICE)

    with torch.no_grad():
        T2 = sample_target_given_ints(N_OPT, DEVICE, target_m_idx, target_v_idx)
        grid = torch.linspace(GRID_MIN, GRID_MAX, GRID_POINTS, device=DEVICE)
        q = kde_density_on_grid(T2, grid, KDE_H).detach()

    m_bits0 = int_to_bits(torch.tensor([start_m_idx], device=DEVICE), 7)
    v_bits0 = int_to_bits(torch.tensor([start_v_idx], device=DEVICE), 4)

    init_scale = 6.0
    m_logits = nn.Parameter((m_bits0 * 2.0 - 1.0) * init_scale)
    v_logits = nn.Parameter((v_bits0 * 2.0 - 1.0) * init_scale)

    opt = torch.optim.Adam([m_logits, v_logits], lr=OPT_LR)

    g_m = gumbel_noise_like(m_logits).detach()
    g_v = gumbel_noise_like(v_logits).detach()

    ck_iters = sorted(set([1] + [int(round(k * OPT_ITERS / 5)) for k in range(1, 5)]))

    checkpoints: List[Tuple[int, float, torch.Tensor, torch.Tensor]] = []
    kl_hist: List[float] = []

    with torch.no_grad():
        mb = torch.sigmoid((m_logits + g_m) / TAU).repeat(N_OPT, 1)
        vb = torch.sigmoid((v_logits + g_v) / TAU).repeat(N_OPT, 1)

        G0 = ddim_sample_diff(
            model,
            diff,
            x_init,
            steps=DDIM_STEPS,
            m_bits=mb,
            v_bits=vb,
        )

        kl0 = float(kl_kde(G0, q, grid, KDE_H).cpu().item())
        print(f"initial KL(gen || T2) ≈ {kl0:.4f}")

    for it in range(1, OPT_ITERS + 1):
        mb = torch.sigmoid((m_logits + g_m) / TAU).repeat(N_OPT, 1)
        vb = torch.sigmoid((v_logits + g_v) / TAU).repeat(N_OPT, 1)

        G = ddim_sample_diff(
            model,
            diff,
            x_init,
            steps=DDIM_STEPS,
            m_bits=mb,
            v_bits=vb,
        )

        kl = kl_kde(G, q, grid, KDE_H)

        opt.zero_grad(set_to_none=True)
        kl.backward()
        opt.step()

        kl_val = float(kl.detach().cpu().item())
        kl_hist.append(kl_val)

        if it % 30 == 0 or it == 1:
            print(f"opt iter {it:4d}/{OPT_ITERS} | KL ≈ {kl_val:.4f}")

        if it in ck_iters:
            with torch.no_grad():
                mb1 = torch.sigmoid((m_logits + g_m) / TAU).clone()
                vb1 = torch.sigmoid((v_logits + g_v) / TAU).clone()

                G_ck = ddim_sample_diff(
                    model,
                    diff,
                    x_init,
                    steps=DDIM_STEPS,
                    m_bits=mb1.repeat(N_OPT, 1),
                    v_bits=vb1.repeat(N_OPT, 1),
                )

                kl_ck = float(kl_kde(G_ck, q, grid, KDE_H).cpu().item())
                checkpoints.append((it, kl_ck, mb1, vb1))

    with torch.no_grad():
        mb_hard = (torch.sigmoid((m_logits + g_m) / TAU) > 0.5).float()
        vb_hard = (torch.sigmoid((v_logits + g_v) / TAU) > 0.5).float()

        m_dec = int(bits_to_int(mb_hard).item())
        v_dec = int(bits_to_int(vb_hard).item())

    print(f"final decoded condition: m_idx={m_dec}, v_idx={v_dec}")
    print(f"(for target generator only: m={m_dec / 100:.2f}, v={v_dec / 100:.2f})")
    print(f"final KL(gen || T2) ≈ {kl_hist[-1]:.4f}")

    return checkpoints, (m_dec, v_dec), kl_hist


@torch.no_grad()
def plot_all(
    model: EpsModel,
    diff: Diffusion1D,
    *,
    T1: Tuple[int, int],
    T2: Tuple[int, int],
    checkpoints: List[Tuple[int, float, torch.Tensor, torch.Tensor]],
    save_path: str,
) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "axes.linewidth": 0.9,
        "savefig.dpi": 600,
    })

    n = 20000
    m1, v1 = T1
    m2, v2 = T2

    T1_s = sample_target_given_ints(n, DEVICE, m1, v1)
    T2_s = sample_target_given_ints(n, DEVICE, m2, v2)

    x_init = torch.randn(n, 1, device=DEVICE)

    grid = torch.linspace(GRID_MIN, GRID_MAX, 900, device=DEVICE)
    x_grid = grid.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    def plot_kde(
        samples: torch.Tensor,
        label: str,
        color,
        lw: float = 2.0,
        ls: str = "-",
        alpha: float = 1.0,
        zorder: int = 1,
    ) -> None:
        density = kde_density_on_grid(samples, grid, KDE_H)

        dx = grid[1] - grid[0]
        density = density / (density.sum() * dx)

        ax.plot(
            x_grid,
            density.detach().cpu().numpy(),
            label=label,
            color=color,
            linewidth=lw,
            linestyle=ls,
            alpha=alpha,
            zorder=zorder,
        )

    plot_kde(
        T1_s,
        rf"Initial: $m={m1 / 100:.2f}$, $v={v1 / 100:.2f}$",
        color="#0072B2",
        lw=2.4,
        zorder=5,
    )

    plot_kde(
        T2_s,
        rf"Target: $m={m2 / 100:.2f}$, $v={v2 / 100:.2f}$",
        color="#D55E00",
        lw=2.4,
        zorder=6,
    )

    plot_checkpoints = [ck for ck in checkpoints if ck[0] != 84]

    colors = plt.get_cmap("viridis")(
        torch.linspace(0.15, 0.85, len(plot_checkpoints)).cpu().numpy()
    )

    all_samples = [T1_s, T2_s]

    for color, (it, kl_i, mb1, vb1) in zip(colors, plot_checkpoints):
        mb = mb1.repeat(n, 1)
        vb = vb1.repeat(n, 1)

        Gi = ddim_sample_diff(
            model,
            diff,
            x_init,
            steps=DDIM_STEPS,
            m_bits=mb,
            v_bits=vb,
        )

        all_samples.append(Gi)

        plot_kde(
            Gi,
            rf"Iteration {it}: KL={kl_i:.3f}",
            color=color,
            lw=1.5,
            ls="--",
            alpha=0.9,
            zorder=2,
        )

    pooled = torch.cat([s.view(-1) for s in all_samples])
    lo, hi = torch.quantile(
        pooled,
        torch.tensor([0.005, 0.995], device=DEVICE),
    )
    pad = 0.05 * float(hi - lo)

    ax.set_xlim(float(lo) - pad, float(hi) + pad)
    ax.set_ylim(bottom=0)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel("Density")

    ax.grid(True, linewidth=0.4, alpha=0.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
    frameon=False,
    loc="center right",
    bbox_to_anchor=(-0.12, 0.5),
    handlelength=2.6,
)

    fig.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {save_path}")


# ----------------------------
# Main
# ----------------------------
def main():
    set_seed(SEED)

    diff = Diffusion1D(
        T=T,
        beta_start=BETA_START,
        beta_end=BETA_END,
        device=DEVICE,
    )

    model = EpsModel().to(DEVICE)

    print(
        f"device={DEVICE} | "
        f"T={T} | "
        f"alpha_bar[T-1]={float(diff.alpha_bars[-1].cpu()):.6f}"
    )

    train(model, diff)

    checkpoints, decoded_final, kl_hist = optimize_condition_via_sampler(
        model,
        diff,
        start_cond=T1_COND,
        target_cond=T2_COND,
    )

    plot_all(
        model,
        diff,
        T1=T1_COND,
        T2=T2_COND,
        checkpoints=checkpoints,
        save_path=PLOT_PATH,
    )

    print("Done.")


if __name__ == "__main__":
    main()
