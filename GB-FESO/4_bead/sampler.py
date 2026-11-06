"""Sample distances and angles at a chosen pair of LJ epsilons."""

import torch

from dataset import GeometryFrameDataset
from model import AngleDiffusion


def _eps_pair_tensor(eps_raw, device, dtype=torch.float32) -> torch.Tensor:
    eps = torch.as_tensor(eps_raw, device=device, dtype=dtype)
    if eps.numel() != 2:
        raise ValueError(f"Expected epsilon pair [eps1, eps2], got shape {tuple(eps.shape)}")
    return eps.reshape(2)


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def sample_geometry(
    model: AngleDiffusion,
    eps_raw,
    frame_ds: GeometryFrameDataset,
    num_samples: int,
    sample_steps: int = 100,
    device: str | None = None,
    clamp: bool = True,
) -> torch.Tensor:
    """Return [r12, r23, r34, theta123, theta234] in raw distance units/radians.

    This keeps gradients through DDIM. For plotting, call inside torch.no_grad().
    clamp limits normalized coordinates to [-1, 1] before converting units.
    """
    if device is None:
        device = next(model.parameters()).device

    eps_cond = _eps_pair_tensor(eps_raw, device=device).view(1, 2).expand(num_samples, 2)
    x = model.ddim_sample(
        eps_cond,
        shape=(num_samples, frame_ds.target_dim),
        sample_steps=sample_steps,
        eta=0.0,
        clamp_x0=clamp,
    )
    return frame_ds.denormalize_target(x, clamp=clamp)


def sample_angles(
    model: AngleDiffusion,
    eps_raw,
    frame_ds: GeometryFrameDataset,
    num_samples: int,
    sample_steps: int = 100,
    device: str | None = None,
) -> torch.Tensor:
    """Return just [theta123, theta234] in radians."""
    return sample_geometry(
        model=model,
        eps_raw=eps_raw,
        frame_ds=frame_ds,
        num_samples=num_samples,
        sample_steps=sample_steps,
        device=device,
    )[..., 3:5]
