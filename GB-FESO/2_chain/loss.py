"""Differentiable losses between ensembles of collective-variable samples."""

import math

import torch


def mmd_loss(generated, target, bandwidth=1.0):
    """Return biased squared MMD with a Gaussian kernel.

    k(x,y) = exp(-||x-y||² / (2 sigma²)); sigma has the CVs' length units.
    Inputs are (N,C)/(M,C) CV samples with matching dtype/device.
    Includes diagonal terms; clips negative roundoff. Memory is quadratic.
    """
    if (generated.ndim != 2 or target.ndim != 2
            or generated.shape[1] != target.shape[1]):
        raise ValueError("Expected (N, C) and (M, C) tensors with the same C.")
    if generated.numel() == 0 or target.numel() == 0:
        raise ValueError("MMD needs nonempty sample sets and at least one CV.")
    if not generated.is_floating_point() or not target.is_floating_point():
        raise TypeError("MMD inputs must be floating-point tensors.")
    if generated.device != target.device or generated.dtype != target.dtype:
        raise ValueError("MMD inputs must have the same device and dtype.")
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive.")

    # Direct distances avoid cancellation in squared-norm matrix products.
    xx = torch.cdist(generated, generated, compute_mode="donot_use_mm_for_euclid_dist").square()
    yy = torch.cdist(target, target, compute_mode="donot_use_mm_for_euclid_dist").square()
    xy = torch.cdist(generated, target, compute_mode="donot_use_mm_for_euclid_dist").square()
    scale = 2.0 * bandwidth ** 2
    value = (
        torch.exp(-xx / scale).mean()
        + torch.exp(-yy / scale).mean()
        - 2.0 * torch.exp(-xy / scale).mean()
    )
    return value.clamp_min(0.0)
