import math
from typing import Optional, Sequence, Union

import torch


# -----------------------------
# Differentiable KDE KL loss
# -----------------------------

def _mixed_sqdist(
    a: torch.Tensor,
    b: torch.Tensor,
    angle_indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Pairwise squared distance for mixed Euclidean/periodic coordinates.

    For angle-only tensors [theta123, theta234], both dimensions are wrapped.
    For the geometry tensors [r12, r23, r34, theta123, theta234], only the
    last two angle dimensions are wrapped.
    """
    d = a[:, None, :] - b[None, :, :]
    ndim = a.shape[-1]

    if angle_indices is None:
        angle_indices = tuple(range(ndim)) if ndim == 2 else (ndim - 2, ndim - 1)

    if len(angle_indices) > 0:
        idx = [i if i >= 0 else ndim + i for i in angle_indices]
        d_angle = d[..., idx]
        d[..., idx] = torch.atan2(torch.sin(d_angle), torch.cos(d_angle))

    return d.square().sum(dim=-1)


# Old internal name kept for compatibility.
def _wrapped_sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _mixed_sqdist(a, b)


def _log_kde_eval(
    eval_x: torch.Tensor,
    centers: torch.Tensor,
    bandwidth: Union[float, torch.Tensor],
    angle_indices: Optional[Sequence[int]] = None,
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
    bandwidth: Union[float, torch.Tensor] = 0.18,
    kl_lambda: float = 1.0,
    angle_indices: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Differentiable KDE KL-style loss for angle-only or 5D geometry samples.

    Inputs can be [theta123, theta234] angle tensors or
    [r12, r23, r34, theta123, theta234] geometry tensors. Angle columns are
    compared periodically; distance columns are compared Euclideanly.
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