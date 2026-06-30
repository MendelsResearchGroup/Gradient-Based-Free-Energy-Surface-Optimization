from typing import Optional, Sequence, Tuple, Union

import torch

from dataset import GeometryFrameDataset, AngleFrameDataset
from model import AngleDiffusion


EpsPairLike = Union[Sequence[float], torch.Tensor]


# -----------------------------
# Epsilon normalization helpers
# -----------------------------

def _eps_pair_tensor(eps_raw: EpsPairLike, device, dtype=torch.float32) -> torch.Tensor:
    eps = torch.as_tensor(eps_raw, device=device, dtype=dtype)
    if eps.numel() != 2:
        raise ValueError(f"Expected epsilon pair [eps1, eps2], got shape {tuple(eps.shape)}")
    return eps.reshape(2)


def eps_raw_to_norm(frame_ds: GeometryFrameDataset, eps_raw: EpsPairLike, device=None, dtype=torch.float32) -> torch.Tensor:
    # Backward-compatible name: epsilon conditions are now raw values.
    return _eps_pair_tensor(eps_raw, device=device, dtype=dtype)


def eps_norm_to_raw(frame_ds: GeometryFrameDataset, eps_norm: EpsPairLike, device=None, dtype=torch.float32) -> torch.Tensor:
    # Backward-compatible name: epsilon conditions are now raw values.
    return _eps_pair_tensor(eps_norm, device=device, dtype=dtype)


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


# -----------------------------
# Sampling
# -----------------------------

def sample_geometry(
    model: AngleDiffusion,
    eps_raw: EpsPairLike,
    frame_ds: GeometryFrameDataset,
    num_samples: int,
    sample_steps: int = 100,
    device: Optional[str] = None,
    clamp: bool = True,
) -> torch.Tensor:
    """
    Generate 5D internal coordinates in raw units.

    Returns [num_samples, 5] with columns:
      [r12, r23, r34, theta123, theta234]
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


def sample_geometry_for_trajectory(
    model: AngleDiffusion,
    frame_ds: GeometryFrameDataset,
    traj_id: int,
    num_samples: int,
    sample_steps: int = 100,
    device: Optional[str] = None,
    clamp: bool = True,
) -> torch.Tensor:
    return sample_geometry(
        model=model,
        eps_raw=frame_ds.get_traj_eps_raw(traj_id),
        frame_ds=frame_ds,
        num_samples=num_samples,
        sample_steps=sample_steps,
        device=device,
        clamp=clamp,
    )


def sample_angles(
    model: AngleDiffusion,
    eps_raw: EpsPairLike,
    frame_ds: GeometryFrameDataset,
    num_samples: int,
    sample_steps: int = 100,
    device: Optional[str] = None,
) -> torch.Tensor:
    """Backward-compatible angle-only sampler returning [theta123, theta234]."""
    return sample_geometry(
        model=model,
        eps_raw=eps_raw,
        frame_ds=frame_ds,
        num_samples=num_samples,
        sample_steps=sample_steps,
        device=device,
    )[..., 3:5]


def sample_angles_for_trajectory(
    model: AngleDiffusion,
    frame_ds: GeometryFrameDataset,
    traj_id: int,
    num_samples: int,
    sample_steps: int = 100,
    device: Optional[str] = None,
) -> torch.Tensor:
    return sample_angles(
        model=model,
        eps_raw=frame_ds.get_traj_eps_raw(traj_id),
        frame_ds=frame_ds,
        num_samples=num_samples,
        sample_steps=sample_steps,
        device=device,
    )