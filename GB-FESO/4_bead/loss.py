"""KL-style KDE and squared MMD losses for angular or full geometry samples.

The last two columns are periodic angles. Other columns use raw distance units;
a scalar bandwidth therefore sets the same numerical scale for both units.
"""

import math

import torch


def _mixed_sqdist(
    a: torch.Tensor,
    b: torch.Tensor,
    angle_indices=None,
) -> torch.Tensor:
    """Squared distances with angular differences wrapped to [-pi, pi]."""
    d = a[:, None, :] - b[None, :, :]
    ndim = a.shape[-1]

    if angle_indices is None:
        angle_indices = tuple(range(ndim)) if ndim == 2 else (ndim - 2, ndim - 1)

    if len(angle_indices) > 0:
        idx = [i if i >= 0 else ndim + i for i in angle_indices]
        d_angle = d[..., idx]
        d[..., idx] = torch.atan2(torch.sin(d_angle), torch.cos(d_angle))

    return d.square().sum(dim=-1)


def _log_kde_eval(
    eval_x: torch.Tensor,
    centers: torch.Tensor,
    bandwidth: float | torch.Tensor,
    angle_indices=None,
) -> torch.Tensor:
    if torch.is_tensor(bandwidth):
        if torch.any(bandwidth <= 0):
            raise ValueError("bandwidth values must be positive")
        bw = bandwidth.to(device=eval_x.device, dtype=eval_x.dtype)
        d = eval_x[:, None, :] - centers[None, :, :]
        ndim = eval_x.shape[-1]
        if angle_indices is None:
            angle_indices = tuple(range(ndim)) if ndim == 2 else (ndim - 2, ndim - 1)
        idx = [i if i >= 0 else ndim + i for i in angle_indices]
        if idx:
            d_angle = d[..., idx]
            d[..., idx] = torch.atan2(torch.sin(d_angle), torch.cos(d_angle))
        sqdist = ((d / bw) ** 2).sum(dim=-1)
        log_norm = math.log(centers.shape[0]) + torch.log(bw).sum() + 0.5 * ndim * math.log(2.0 * math.pi)
        return torch.logsumexp(-0.5 * sqdist, dim=1) - log_norm

    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")
    d = eval_x.shape[-1]
    sqdist = _mixed_sqdist(eval_x, centers, angle_indices=angle_indices)
    log_k = -0.5 * sqdist / (bandwidth * bandwidth)
    log_norm = math.log(centers.shape[0]) + d * math.log(bandwidth) + 0.5 * d * math.log(2.0 * math.pi)
    return torch.logsumexp(log_k, dim=1) - log_norm


def kde_kl_target_to_generated(
    target_rad: torch.Tensor,
    generated_rad: torch.Tensor,
    bandwidth: float | torch.Tensor = 0.18,
    kl_lambda: float = 1.0,
    angle_indices=None,
) -> torch.Tensor:
    """Estimate KL(target || generated) + kl_lambda * KL(generated || target).

    Pass [N, 2] angles or [N, 5] geometry in raw units. Bandwidth is a scalar
    or a tensor with one value per coordinate. The KDE uses shortest angular
    differences and includes each sample's own kernel contribution.
    """
    log_p_target = _log_kde_eval(target_rad, target_rad.detach(), bandwidth, angle_indices).detach()
    log_q_target = _log_kde_eval(target_rad, generated_rad, bandwidth, angle_indices)
    forward = (log_p_target - log_q_target).mean()

    if float(kl_lambda) == 0.0:
        return forward

    log_q_generated = _log_kde_eval(generated_rad, generated_rad.detach(), bandwidth, angle_indices)
    log_p_generated = _log_kde_eval(generated_rad, target_rad.detach(), bandwidth, angle_indices)
    reverse = (log_q_generated - log_p_generated).mean()

    return forward + float(kl_lambda) * reverse


def mmd_target_to_generated(
    target_rad: torch.Tensor,
    generated_rad: torch.Tensor,
    bandwidth: float | torch.Tensor = 0.32,
    angle_indices=None,
) -> torch.Tensor:
    """Biased squared MMD with a periodic Gaussian kernel.

    mean K(target, target) + mean K(generated, generated) - 2 * mean K(target, generated).
    Diagonal terms are included. The target is fixed; both generated arguments
    contribute gradients. Sample counts may differ.

    Angular differences use the circle chord 2*sin(delta/2). Distances use
    ordinary subtraction. Bandwidth is a scalar or one value per coordinate.
    """
    if (target_rad.ndim != 2 or generated_rad.ndim != 2
            or target_rad.shape[1] != generated_rad.shape[1]
            or target_rad.shape[1] == 0
            or len(target_rad) == 0 or len(generated_rad) == 0):
        raise ValueError("MMD requires nonempty [N, D] and [M, D] samples with matching D")
    ndim = target_rad.shape[1]
    if angle_indices is None:
        angle_indices = tuple(range(max(0, ndim - 2), ndim))
    idx = list(angle_indices)
    bw = torch.as_tensor(bandwidth, device=generated_rad.device, dtype=generated_rad.dtype)
    if bw.ndim > 1 or bw.numel() not in (1, ndim):
        raise ValueError("bandwidth must be a scalar or one value per coordinate")
    if not bool(torch.isfinite(bw).all()) or bool((bw <= 0).any()):
        raise ValueError("bandwidth must be finite and positive")

    def kernel(a, b):
        delta = a[:, None, :] - b[None, :, :]
        if idx:
            delta[..., idx] = 2.0 * torch.sin(0.5 * delta[..., idx])
        return torch.exp(-0.5 * ((delta / bw) ** 2).sum(dim=-1))

    target = target_rad.detach()
    value = (kernel(target, target).mean()
             + kernel(generated_rad, generated_rad).mean()
             - 2.0 * kernel(target, generated_rad).mean())
    return value.clamp_min(0.0)  # Remove tiny negative roundoff at equality.
