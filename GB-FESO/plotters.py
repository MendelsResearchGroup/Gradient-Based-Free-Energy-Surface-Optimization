import math
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import matplotlib.patheffects as pe
import torch

from dataset import AngleFrameDataset
from model import AngleDiffusion

EpsPairLike = Union[Sequence[float], torch.Tensor]


# -----------------------------
# Plotting helpers
# -----------------------------

def _eps_pair_tensor(eps_raw: EpsPairLike, device, dtype=torch.float32) -> torch.Tensor:
    eps = torch.as_tensor(eps_raw, device=device, dtype=dtype)
    if eps.numel() != 2:
        raise ValueError(f"Expected epsilon pair [eps1, eps2], got shape {tuple(eps.shape)}")
    return eps.reshape(2)


def _eps_raw_to_norm(frame_ds: AngleFrameDataset, eps_raw: EpsPairLike, device=None, dtype=torch.float32) -> torch.Tensor:
    # Backward-compatible name: epsilon conditions are now raw values.
    return _eps_pair_tensor(eps_raw, device=device, dtype=dtype)


def _sample_geometry(
    model: AngleDiffusion,
    eps_raw: EpsPairLike,
    frame_ds: AngleFrameDataset,
    num_samples: int,
    sample_steps: int,
    device,
) -> torch.Tensor:
    eps_cond = _eps_pair_tensor(eps_raw, device=device).view(1, 2).expand(num_samples, 2)
    target_dim = int(getattr(frame_ds, "target_dim", 2))
    x = model.ddim_sample(eps_cond, shape=(num_samples, target_dim), sample_steps=sample_steps, eta=0.0)
    x = x.clamp(-1.0, 1.0)
    if hasattr(frame_ds, "denormalize_target") and target_dim == 5:
        return frame_ds.denormalize_target(x)
    return model.normalized_to_radians(x)


def _sample_angles(
    model: AngleDiffusion,
    eps_raw: EpsPairLike,
    frame_ds: AngleFrameDataset,
    num_samples: int,
    sample_steps: int,
    device,
) -> torch.Tensor:
    samples = _sample_geometry(model, eps_raw, frame_ds, num_samples, sample_steps, device)
    return samples[..., -2:] if samples.shape[-1] > 2 else samples


def _sample_angles_in_batches(
    model: AngleDiffusion,
    eps_raw: EpsPairLike,
    frame_ds: AngleFrameDataset,
    total_samples: int,
    batch_size: int,
    sample_steps: int,
    device,
    seed: int,
) -> torch.Tensor:
    chunks = []
    remaining = int(total_samples)
    torch.manual_seed(seed)
    while remaining > 0:
        n = min(int(batch_size), remaining)
        chunk = _sample_angles(
            model=model,
            eps_raw=eps_raw,
            frame_ds=frame_ds,
            num_samples=n,
            sample_steps=sample_steps,
            device=device,
        )
        chunks.append(chunk.detach().cpu())
        remaining -= n
    return torch.cat(chunks, dim=0)


def _sample_geometry_in_batches(
    model: AngleDiffusion,
    eps_raw: EpsPairLike,
    frame_ds: AngleFrameDataset,
    total_samples: int,
    batch_size: int,
    sample_steps: int,
    device,
    seed: int,
) -> torch.Tensor:
    chunks = []
    remaining = int(total_samples)
    torch.manual_seed(seed)
    while remaining > 0:
        n = min(int(batch_size), remaining)
        chunk = _sample_geometry(
            model=model,
            eps_raw=eps_raw,
            frame_ds=frame_ds,
            num_samples=n,
            sample_steps=sample_steps,
            device=device,
        )
        chunks.append(chunk.detach().cpu())
        remaining -= n
    return torch.cat(chunks, dim=0)


def _target_samples(target_rad: torch.Tensor, total_samples: int) -> torch.Tensor:
    target_rad = target_rad.detach().cpu()
    n = int(target_rad.shape[0])
    if n <= 0:
        raise ValueError("target trajectory has no samples")
    if n == total_samples:
        return target_rad
    if n > total_samples:
        idx = torch.linspace(0, n - 1, total_samples).long()
        return target_rad.index_select(0, idx)
    repeats = math.ceil(total_samples / n)
    return target_rad.repeat((repeats, 1))[:total_samples]


def _heatmap_probability(angles_rad: torch.Tensor, bins: int):
    import numpy as np

    a = angles_rad.detach().cpu().numpy()
    counts, xedges, yedges = np.histogram2d(
        a[:, 0],
        a[:, 1],
        bins=bins,
        range=[[-math.pi, math.pi], [-math.pi, math.pi]],
    )
    prob = counts.astype(np.float64)
    total = prob.sum()
    if total > 0:
        prob /= total
    return prob.T, xedges, yedges


DEFAULT_BASIN_HALF_WIDTH = 0.45

ANGLE_BASIN_BOXES = (
    {
        "label": "A anti-diagonal",
        "short": "A",
        "boxes": (
            ((-1.02, 0.98), (0.48, 0.45)),
            ((1.02, -1.00), (0.47, 0.48)),
        ),
        "edgecolor": "tab:red",
        "linestyle": "-",
    },
    {
        "label": "B same-sign",
        "short": "B",
        "boxes": (
            ((0.70, 0.95), (0.50, 0.47)),
            ((-0.95, -0.95), (0.53, 0.50)),
        ),
        "edgecolor": "tab:blue",
        "linestyle": "--",
    },
    {
        "label": "C outer",
        "short": "C",
        "boxes": (
            ((1.10, 2.02), (0.35, 0.52)),
            ((1.50, 1.62), (0.25, 0.32)),
            ((2.05, 1.02), (0.50, 0.42)),
            ((-2.05, -1.02), (0.50, 0.42)),
            ((-1.50, -1.75), (0.25, 0.29)),
            ((-1.10, -2.02), (0.35, 0.52)),
        ),
        "edgecolor": "tab:orange",
        "linestyle": "-.",
    },
)


def _scaled_basin_boxes(basin, half_width: float):
    scale = float(half_width) / DEFAULT_BASIN_HALF_WIDTH
    for center, half_size in basin["boxes"]:
        yield center, (float(half_size[0]) * scale, float(half_size[1]) * scale)


def _add_angle_basin_squares(ax, half_width: float = DEFAULT_BASIN_HALF_WIDTH) -> None:
    """Overlay hard-coded symmetry-grouped angular basin boxes."""
    xlim = (-math.pi, math.pi)
    ylim = (-math.pi, math.pi)
    shifts = (-2.0 * math.pi, 0.0, 2.0 * math.pi)

    for basin in ANGLE_BASIN_BOXES:
        label_drawn = False
        for (center_x, center_y), (half_x, half_y) in _scaled_basin_boxes(basin, half_width):
            for shift_x in shifts:
                for shift_y in shifts:
                    cx = center_x + shift_x
                    cy = center_y + shift_y
                    left = cx - half_x
                    bottom = cy - half_y
                    right = cx + half_x
                    top = cy + half_y
                    if right < xlim[0] or left > xlim[1] or top < ylim[0] or bottom > ylim[1]:
                        continue

                    rect = Rectangle(
                        (left, bottom),
                        2.0 * half_x,
                        2.0 * half_y,
                        fill=False,
                        edgecolor=basin["edgecolor"],
                        linestyle=basin["linestyle"],
                        linewidth=2.0,
                        alpha=0.95,
                        zorder=8,
                    )
                    rect.set_path_effects([
                        pe.Stroke(linewidth=3.6, foreground="white"),
                        pe.Normal(),
                    ])
                    ax.add_patch(rect)

                    if not label_drawn and xlim[0] <= cx <= xlim[1] and ylim[0] <= cy <= ylim[1]:
                        ax.text(
                            cx,
                            cy,
                            basin["short"],
                            ha="center",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            color=basin["edgecolor"],
                            zorder=9,
                            bbox={
                                "boxstyle": "round,pad=0.15",
                                "facecolor": "white",
                                "edgecolor": "none",
                                "alpha": 0.85,
                            },
                        )
                        label_drawn = True


def _angle_basin_legend_handles():
    handles = []
    for basin in ANGLE_BASIN_BOXES:
        handles.append(
            Patch(
                facecolor="none",
                edgecolor=basin["edgecolor"],
                linestyle=basin["linestyle"],
                linewidth=2.0,
                label=basin["label"],
            )
        )
    return handles


# -----------------------------
# Public plotting functions
# -----------------------------

def plot_target_vs_generated_heatmaps(
    model: AngleDiffusion,
    frame_ds: AngleFrameDataset,
    traj_id: int,
    eps_raw=None,
    generated_rad: Optional[torch.Tensor] = None,
    num_samples: Optional[int] = None,
    sample_steps: int = 100,
    plot_batches: int = 15,
    plot_batch_size: int = 2048,
    bins: int = 80,
    seed: int = 0,
    save_path: Optional[str] = None,
    dpi: int = 180,
    show: bool = False,
    close: bool = False,
    step: Optional[int] = None,
    kl: Optional[float] = None,
    free_energy_floor: float = 1e-12,
    free_energy_cap: Optional[float] = 4,
):
    """
    Plot only free-energy heatmaps for one target trajectory, generated samples at
    the current epsilons, and generated samples at the actual target epsilons.

    Row 1 uses the two angle coordinates: theta_123, theta_234 and overlays
    hard-coded symmetry-grouped basin boxes.
    Row 2 uses the first two distance coordinates: r12, r23.

    eps_raw controls the generated/current distribution. If None, the target
    trajectory epsilons are used. The target trajectory is subsampled or repeated
    to the same total count used for generated samples.

    Returns (fig, axes), where axes has shape [2, 3].
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    if plot_batches <= 0:
        raise ValueError("plot_batches must be positive")
    if plot_batch_size <= 0:
        raise ValueError("plot_batch_size must be positive")

    device = next(model.parameters()).device
    target = frame_ds.traj_ds[traj_id]
    full_target = target["target"].to(dtype=torch.float32)
    has_distances = full_target.shape[-1] >= 5
    target_eps = tuple(float(v) for v in frame_ds.get_traj_eps_raw(traj_id))
    current_eps = target_eps if eps_raw is None else tuple(float(v) for v in _eps_pair_tensor(eps_raw, device="cpu"))

    total_samples = int(num_samples) if num_samples is not None else int(plot_batches) * int(plot_batch_size)
    target_geometry = _target_samples(full_target, total_samples)
    target_angles = target_geometry[..., -2:] if target_geometry.shape[-1] > 2 else target_geometry

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            generated_geometry = None
            if generated_rad is not None:
                generated_geometry = generated_rad.detach().to(dtype=torch.float32).cpu()

            if generated_geometry is None or generated_geometry.shape[0] < total_samples:
                generated_geometry = _sample_geometry_in_batches(
                    model=model,
                    eps_raw=current_eps,
                    frame_ds=frame_ds,
                    total_samples=total_samples,
                    batch_size=plot_batch_size,
                    sample_steps=sample_steps,
                    device=device,
                    seed=seed,
                )
            elif generated_geometry.shape[0] > total_samples:
                generated_geometry = generated_geometry[:total_samples]

            target_eps_generated_geometry = _sample_geometry_in_batches(
                model=model,
                eps_raw=target_eps,
                frame_ds=frame_ds,
                total_samples=total_samples,
                batch_size=plot_batch_size,
                sample_steps=sample_steps,
                device=device,
                seed=seed,
            )
    finally:
        model.train(was_training)

    generated_angles = generated_geometry[..., -2:] if generated_geometry.shape[-1] > 2 else generated_geometry
    target_eps_generated_angles = (
        target_eps_generated_geometry[..., -2:]
        if target_eps_generated_geometry.shape[-1] > 2
        else target_eps_generated_geometry
    )

    angle_ranges = [[-math.pi, math.pi], [-math.pi, math.pi]]
    target_angle_fe, angle_xedges, angle_yedges = _histogram_free_energy(
        target_angles,
        bins=bins,
        floor=free_energy_floor,
        cap=free_energy_cap,
        ranges=angle_ranges,
    )
    generated_angle_fe, _, _ = _histogram_free_energy(
        generated_angles,
        bins=bins,
        floor=free_energy_floor,
        cap=free_energy_cap,
        ranges=angle_ranges,
    )
    target_eps_generated_angle_fe, _, _ = _histogram_free_energy(
        target_eps_generated_angles,
        bins=bins,
        floor=free_energy_floor,
        cap=free_energy_cap,
        ranges=angle_ranges,
    )
    angle_fe_vmax = max(
        float(target_angle_fe.max()),
        float(generated_angle_fe.max()),
        float(target_eps_generated_angle_fe.max()),
    )

    dist_panels = None
    if has_distances:
        target_dist = target_geometry[..., :2]
        generated_dist = generated_geometry[..., :2]
        target_eps_generated_dist = target_eps_generated_geometry[..., :2]
        all_dist = torch.cat([target_dist, generated_dist, target_eps_generated_dist], dim=0)
        dist_min = all_dist.min(dim=0).values
        dist_max = all_dist.max(dim=0).values
        pad = 0.05 * torch.clamp(dist_max - dist_min, min=1e-6)
        dist_ranges = [
            [float(dist_min[0] - pad[0]), float(dist_max[0] + pad[0])],
            [float(dist_min[1] - pad[1]), float(dist_max[1] + pad[1])],
        ]
        target_dist_fe, dist_xedges, dist_yedges = _histogram_free_energy(
            target_dist,
            bins=bins,
            floor=free_energy_floor,
            cap=free_energy_cap,
            ranges=dist_ranges,
        )
        generated_dist_fe, _, _ = _histogram_free_energy(
            generated_dist,
            bins=bins,
            floor=free_energy_floor,
            cap=free_energy_cap,
            ranges=dist_ranges,
        )
        target_eps_generated_dist_fe, _, _ = _histogram_free_energy(
            target_eps_generated_dist,
            bins=bins,
            floor=free_energy_floor,
            cap=free_energy_cap,
            ranges=dist_ranges,
        )
        dist_fe_vmax = max(
            float(target_dist_fe.max()),
            float(generated_dist_fe.max()),
            float(target_eps_generated_dist_fe.max()),
        )
        dist_panels = [
            (target_dist_fe, "target r12/r23 FES"),
            (generated_dist_fe, "current generated r12/r23 FES"),
            (target_eps_generated_dist_fe, "target-eps generated r12/r23 FES"),
        ]
    else:
        dist_xedges = dist_yedges = None
        dist_fe_vmax = None

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    angle_panels = [
        (target_angle_fe, "target angle FES"),
        (generated_angle_fe, "current generated angle FES"),
        (target_eps_generated_angle_fe, "target-eps generated angle FES"),
    ]

    angle_mesh = None
    for ax, (free_energy, title) in zip(axes[0, :], angle_panels):
        angle_mesh = ax.pcolormesh(
            angle_xedges,
            angle_yedges,
            free_energy,
            shading="auto",
            vmin=0.0,
            vmax=angle_fe_vmax,
        )
        ax.set_title(title)
        ax.set_xlabel(r"$\theta_{123}$ [rad]")
        ax.set_ylabel(r"$\theta_{234}$ [rad]")
        ax.set_xlim(-math.pi, math.pi)
        ax.set_ylim(-math.pi, math.pi)
        ax.set_aspect("equal")

    dist_mesh = None
    if dist_panels is not None:
        for ax, (free_energy, title) in zip(axes[1, :], dist_panels):
            dist_mesh = ax.pcolormesh(
                dist_xedges,
                dist_yedges,
                free_energy,
                shading="auto",
                vmin=0.0,
                vmax=dist_fe_vmax,
            )
            ax.set_title(title)
            ax.set_xlabel(r"$r_{12}$")
            ax.set_ylabel(r"$r_{23}$")
            ax.set_xlim(dist_xedges[0], dist_xedges[-1])
            ax.set_ylim(dist_yedges[0], dist_yedges[-1])
            ax.set_aspect("auto")
    else:
        for ax in axes[1, :]:
            ax.axis("off")
            ax.text(0.5, 0.5, "r12/r23 FES requires 5D samples", ha="center", va="center")

    if angle_mesh is not None:
        fig.colorbar(angle_mesh, ax=axes[0, :], label=r"angle $F/k_BT = -\log P$ (shifted)")
    if dist_mesh is not None:
        fig.colorbar(dist_mesh, ax=axes[1, :], label=r"distance $F/k_BT = -\log P$ (shifted)")

    legend_labels = [
        f"target eps1={target_eps[0]:.6g}, eps2={target_eps[1]:.6g}",
        f"current eps1={current_eps[0]:.6g}, eps2={current_eps[1]:.6g}",
    ]
    if step is not None:
        legend_labels.append(f"optimization step={step}")
    if kl is not None:
        legend_labels.append(f"KL={kl:.6g}")

    fig.legend(
        handles=[Line2D([], [], linestyle="none", marker="s", label=label) for label in legend_labels],
        loc="upper center",
        ncol=min(len(legend_labels), 2),
        frameon=True,
    )
    fig.suptitle(f"trajectory {traj_id}: free-energy heatmaps", y=1.03)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig, axes

def _histogram_free_energy(
    samples_xy: torch.Tensor,
    bins: int,
    floor: float = 1e-12,
    cap: Optional[float] = None,
    ranges=None,
):
    import numpy as np

    if floor <= 0:
        raise ValueError("floor must be positive")

    a = samples_xy.detach().cpu().numpy()
    if ranges is None:
        ranges = [[-math.pi, math.pi], [-math.pi, math.pi]]
    counts, xedges, yedges = np.histogram2d(
        a[:, 0],
        a[:, 1],
        bins=bins,
        range=ranges,
    )
    p = counts.astype(np.float64)
    total = p.sum()
    if total > 0:
        p /= total

    free_energy = -np.log(np.maximum(p, floor))
    free_energy -= np.nanmin(free_energy)
    if cap is not None:
        free_energy = np.minimum(free_energy, cap)

    return free_energy.T, xedges, yedges



def _geometry_column_label(index: int) -> str:
    labels = {
        0: r"$r_{12}$",
        1: r"$r_{23}$",
        2: r"$r_{34}$",
        3: r"$\theta_{123}$ [rad]",
        4: r"$\theta_{234}$ [rad]",
    }
    return labels.get(int(index), f"column {index}")


def plot_target_vs_generated_distance_angle_fes(
    model: AngleDiffusion,
    frame_ds: AngleFrameDataset,
    traj_id: int,
    eps_raw=None,
    generated_rad: Optional[torch.Tensor] = None,
    num_samples: Optional[int] = None,
    sample_steps: int = 100,
    plot_batch_size: int = 2048,
    bins: int = 80,
    seed: int = 0,
    save_path: Optional[str] = None,
    dpi: int = 180,
    show: bool = False,
    close: bool = False,
    step: Optional[int] = None,
    kl: Optional[float] = None,
    distance_index: int = 0,
    angle_index: int = 3,
    free_energy_floor: float = 1e-12,
    free_energy_cap: Optional[float] = 4,
):
    """
    Plot target vs optimized/generated FES for one distance coordinate on x
    and one angle coordinate on y. Defaults to r12 vs theta123.
    """
    if bins <= 0:
        raise ValueError("bins must be positive")
    if plot_batch_size <= 0:
        raise ValueError("plot_batch_size must be positive")
    if distance_index not in (0, 1, 2):
        raise ValueError("distance_index must be 0, 1, or 2")
    if angle_index not in (3, 4):
        raise ValueError("angle_index must be 3 or 4")

    device = next(model.parameters()).device
    target = frame_ds.traj_ds[traj_id]
    full_target = target["target"].to(dtype=torch.float32)
    if full_target.shape[-1] < 5:
        raise ValueError("distance-angle FES requires 5D geometry samples")

    target_eps = tuple(float(v) for v in frame_ds.get_traj_eps_raw(traj_id))
    current_eps = target_eps if eps_raw is None else tuple(float(v) for v in _eps_pair_tensor(eps_raw, device="cpu"))

    if generated_rad is not None:
        generated_geometry = generated_rad.detach().to(dtype=torch.float32).cpu()
        if generated_geometry.shape[-1] < 5:
            raise ValueError("generated_rad must contain 5D geometry samples")
        total_samples = int(num_samples) if num_samples is not None else int(generated_geometry.shape[0])
        generated_geometry = generated_geometry[:total_samples]
    else:
        total_samples = int(num_samples) if num_samples is not None else int(full_target.shape[0])
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                generated_geometry = _sample_geometry_in_batches(
                    model=model,
                    eps_raw=current_eps,
                    frame_ds=frame_ds,
                    total_samples=total_samples,
                    batch_size=plot_batch_size,
                    sample_steps=sample_steps,
                    device=device,
                    seed=seed,
                )
        finally:
            model.train(was_training)

    target_geometry = _target_samples(full_target, total_samples)
    target_xy = target_geometry[:, [distance_index, angle_index]]
    generated_xy = generated_geometry[:, [distance_index, angle_index]]

    all_x = torch.cat([target_xy[:, 0], generated_xy[:, 0]], dim=0)
    x_min = float(all_x.min().item())
    x_max = float(all_x.max().item())
    pad = 0.05 * max(x_max - x_min, 1e-6)
    ranges = [[x_min - pad, x_max + pad], [-math.pi, math.pi]]

    target_fe, xedges, yedges = _histogram_free_energy(
        target_xy,
        bins=bins,
        floor=free_energy_floor,
        cap=free_energy_cap,
        ranges=ranges,
    )
    generated_fe, _, _ = _histogram_free_energy(
        generated_xy,
        bins=bins,
        floor=free_energy_floor,
        cap=free_energy_cap,
        ranges=ranges,
    )
    vmax = max(float(target_fe.max()), float(generated_fe.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    panels = [
        (axes[0], target_fe, "target distance-angle FES"),
        (axes[1], generated_fe, "optimized distance-angle FES"),
    ]

    mesh = None
    for ax, free_energy, title in panels:
        mesh = ax.pcolormesh(xedges, yedges, free_energy, shading="auto", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(_geometry_column_label(distance_index))
        ax.set_ylabel(_geometry_column_label(angle_index))
        ax.set_xlim(xedges[0], xedges[-1])
        ax.set_ylim(-math.pi, math.pi)
        ax.set_aspect("auto")

    if mesh is not None:
        fig.colorbar(mesh, ax=axes, label=r"$F/k_BT = -\log P$ (shifted)")

    legend_labels = [
        f"target eps1={target_eps[0]:.6g}, eps2={target_eps[1]:.6g}",
        f"optimized eps1={current_eps[0]:.6g}, eps2={current_eps[1]:.6g}",
    ]
    if step is not None:
        legend_labels.append(f"optimization step={step}")
    if kl is not None:
        legend_labels.append(f"KL={kl:.6g}")

    fig.legend(
        handles=[Line2D([], [], linestyle="none", marker="s", label=label) for label in legend_labels],
        loc="upper center",
        ncol=min(len(legend_labels), 2),
        frameon=True,
    )
    fig.suptitle(
        f"trajectory {traj_id}: {_geometry_column_label(distance_index)} vs {_geometry_column_label(angle_index)} FES",
        y=1.08,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig, axes


# -----------------------------
# Public plotting functions
# -----------------------------

def plot_target_vs_generated_free_energy(
    model: AngleDiffusion,
    frame_ds: AngleFrameDataset,
    traj_id: int,
    eps_raw=None,
    generated_rad: Optional[torch.Tensor] = None,
    num_samples: Optional[int] = None,
    sample_steps: int = 100,
    bins: int = 80,
    seed: int = 0,
    save_path: Optional[str] = None,
    dpi: int = 180,
    show: bool = False,
    close: bool = False,
    step: Optional[int] = None,
    metric: Optional[float] = None,
    free_energy_floor: float = 1e-12,
    free_energy_cap: Optional[float] = None,
):
    """
    Plot side-by-side free-energy surfaces for one target trajectory and generated samples.

    The plotted surface is F = -log(P), shifted so that min(F)=0. eps_raw controls
    the generated/current surface. If None, the target trajectory epsilons are used.
    During optimization, pass generated_rad to plot the already sampled current
    surface without resampling.

    Returns (fig, axes).
    """
    if bins <= 0:
        raise ValueError("bins must be positive")

    device = next(model.parameters()).device
    target = frame_ds.traj_ds[traj_id]
    target_raw = target["target"].to(device=device, dtype=torch.float32)
    target_rad = target_raw[..., -2:] if target_raw.shape[-1] > 2 else target_raw
    target_eps = tuple(float(v) for v in frame_ds.get_traj_eps_raw(traj_id))
    current_eps = target_eps if eps_raw is None else tuple(float(v) for v in _eps_pair_tensor(eps_raw, device="cpu"))

    if generated_rad is None:
        if num_samples is None:
            num_samples = int(target_rad.shape[0])

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                torch.manual_seed(seed)
                generated_rad = _sample_angles(
                    model=model,
                    eps_raw=current_eps,
                    frame_ds=frame_ds,
                    num_samples=num_samples,
                    sample_steps=sample_steps,
                    device=device,
                )
        finally:
            model.train(was_training)
    else:
        generated_rad = generated_rad.detach().to(device=device, dtype=torch.float32)
        generated_rad = generated_rad[..., -2:] if generated_rad.shape[-1] > 2 else generated_rad

    target_fe, xedges, yedges = _histogram_free_energy(
        target_rad,
        bins=bins,
        floor=free_energy_floor,
        cap=free_energy_cap,
    )
    generated_fe, _, _ = _histogram_free_energy(
        generated_rad,
        bins=bins,
        floor=free_energy_floor,
        cap=free_energy_cap,
    )
    vmax = max(float(target_fe.max()), float(generated_fe.max()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    panels = [
        (axes[0], target_fe, "target free-energy surface"),
        (axes[1], generated_fe, "current generated free-energy surface"),
    ]

    mesh = None
    for ax, free_energy, title in panels:
        mesh = ax.pcolormesh(xedges, yedges, free_energy, shading="auto", vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel(r"$\theta_{123}$ [rad]")
        ax.set_ylabel(r"$\theta_{234}$ [rad]")
        ax.set_xlim(-math.pi, math.pi)
        ax.set_ylim(-math.pi, math.pi)
        ax.set_aspect("equal")

    if mesh is not None:
        fig.colorbar(mesh, ax=axes, label=r"$F/k_BT = -\log P$ (shifted)")

    legend_labels = [
        f"target eps1={target_eps[0]:.6g}, eps2={target_eps[1]:.6g}",
        f"current eps1={current_eps[0]:.6g}, eps2={current_eps[1]:.6g}",
    ]
    if step is not None:
        legend_labels.append(f"optimization step={step}")
    if metric is not None:
        legend_labels.append(f"FE-MSE={metric:.6g}")

    fig.legend(
        handles=[Line2D([], [], linestyle="none", marker="s", label=label) for label in legend_labels],
        loc="upper center",
        ncol=min(len(legend_labels), 2),
        frameon=True,
    )
    fig.suptitle(f"trajectory {traj_id}: target vs generated free-energy surface", y=1.08)

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)

    return fig, axes


def plot_epsilon_loss_surface(
    model: AngleDiffusion,
    frame_ds: AngleFrameDataset,
    traj_id: int,
    loss_fn,
    eps1_range: Optional[Tuple[float, float]] = None,
    eps2_range: Optional[Tuple[float, float]] = None,
    grid_size: int = 25,
    num_samples: int = 512,
    sample_steps: int = 100,
    max_target_points: int = 1024,
    seed: int = 0,
    save_path: Optional[str] = None,
    dpi: int = 180,
    show: bool = False,
    close: bool = False,
    initial_eps: Optional[EpsPairLike] = None,
    optimized_eps: Optional[EpsPairLike] = None,
    title: Optional[str] = None,
):
    """
    Scan the actual inverse loss over a raw (eps1, eps2) grid and plot it.

    loss_fn must take (target_rad, generated_rad) and return a scalar tensor or float.
    The same DDIM initial noise is reused for every grid point, matching the optimizer
    setup and making the surface less noisy.

    Returns (fig, ax, surface), where surface contains eps grids, loss values,
    target eps, and the best grid point.
    """
    import numpy as np

    if grid_size <= 1:
        raise ValueError("grid_size must be greater than 1")

    device = next(model.parameters()).device
    eps_min = torch.as_tensor(frame_ds.eps_min, dtype=torch.float32)
    eps_max = torch.as_tensor(frame_ds.eps_max, dtype=torch.float32)

    if eps1_range is None:
        eps1_range = (float(eps_min[0]), float(eps_max[0]))
    if eps2_range is None:
        eps2_range = (float(eps_min[1]), float(eps_max[1]))

    eps1_values = np.linspace(float(eps1_range[0]), float(eps1_range[1]), grid_size)
    eps2_values = np.linspace(float(eps2_range[0]), float(eps2_range[1]), grid_size)

    target_rad = frame_ds.traj_ds[traj_id]["target"].to(device=device, dtype=torch.float32)
    if target_rad.shape[0] > max_target_points:
        idx = torch.linspace(0, target_rad.shape[0] - 1, max_target_points, device=device).long()
        target_rad = target_rad.index_select(0, idx)

    target_eps = tuple(float(v) for v in frame_ds.get_traj_eps_raw(traj_id))
    losses = np.empty((grid_size, grid_size), dtype=np.float64)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for j, eps2 in enumerate(eps2_values):
                for i, eps1 in enumerate(eps1_values):
                    torch.manual_seed(seed)
                    generated_rad = _sample_geometry(
                        model=model,
                        eps_raw=(float(eps1), float(eps2)),
                        frame_ds=frame_ds,
                        num_samples=num_samples,
                        sample_steps=sample_steps,
                        device=device,
                    )
                    if target_rad.shape[-1] == 2 and generated_rad.shape[-1] > 2:
                        generated_rad = generated_rad[..., -2:]
                    loss = loss_fn(target_rad, generated_rad)
                    losses[j, i] = float(loss.detach().item()) if torch.is_tensor(loss) else float(loss)
    finally:
        model.train(was_training)

    best_j, best_i = np.unravel_index(np.nanargmin(losses), losses.shape)
    best_eps = (float(eps1_values[best_i]), float(eps2_values[best_j]))
    best_loss = float(losses[best_j, best_i])

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 5.5), constrained_layout=True)
    mesh = ax.imshow(
        losses,
        origin="lower",
        aspect="auto",
        extent=(eps1_values[0], eps1_values[-1], eps2_values[0], eps2_values[-1]),
    )
    fig.colorbar(mesh, ax=ax, label="loss")

    ax.scatter([target_eps[0]], [target_eps[1]], marker="x", s=80, label="target eps")
    ax.scatter([best_eps[0]], [best_eps[1]], marker="o", s=60, facecolors="none", label="best grid eps")

    if initial_eps is not None:
        e = tuple(float(v) for v in _eps_pair_tensor(initial_eps, device="cpu"))
        ax.scatter([e[0]], [e[1]], marker="s", s=50, label="initial eps")

    if optimized_eps is not None:
        e = tuple(float(v) for v in _eps_pair_tensor(optimized_eps, device="cpu"))
        ax.scatter([e[0]], [e[1]], marker="^", s=60, label="optimized eps")

    ax.set_xlabel("eps1 raw")
    ax.set_ylabel("eps2 raw")
    ax.set_title(title or f"trajectory {traj_id}: epsilon loss surface")
    ax.legend(loc="best")

    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    if close:
        plt.close(fig)

    surface = {
        "eps1_values": eps1_values,
        "eps2_values": eps2_values,
        "loss": losses,
        "target_eps": target_eps,
        "best_eps": best_eps,
        "best_loss": best_loss,
    }
    return fig, ax, surface