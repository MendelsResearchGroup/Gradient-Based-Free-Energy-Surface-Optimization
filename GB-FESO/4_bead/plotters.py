"""FES comparisons and an optional epsilon loss-surface scan."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from sampler import sample_geometry
from utils import ANGLE_BASINS, _scaled_basin_boxes


def plot_fes_comparison(target, generated, save_path, bins=80, dpi=180, half_width=None, title=""):
    """Plot angular, distance-distance, and distance-angle marginals of saved samples."""
    samples = [torch.as_tensor(x).detach().cpu().numpy().copy() for x in (target, generated)]
    for x in samples:
        if x.ndim != 2 or x.shape[1] != 5 or not len(x) or not np.isfinite(x).all():
            raise ValueError("Expected nonempty, finite [N, 5] geometry")
        x[:, -2:] = (x[:, -2:] + np.pi) % (2 * np.pi) - np.pi
    labels = (r"$r_{12}$", r"$r_{23}$", r"$r_{34}$", r"$\theta_{123}$ [rad]", r"$\theta_{234}$ [rad]")
    fig, axes = plt.subplots(3, 2, figsize=(10, 12), layout="constrained")
    for row, columns in enumerate(((3, 4), (0, 1), (0, 3))):
        pooled = np.concatenate([x[:, columns] for x in samples])
        ranges = []
        for i, column in enumerate(columns):
            lo, hi = pooled[:, i].min(), pooled[:, i].max()
            pad = 0.05 * max(hi - lo, 1e-6)
            ranges.append([-np.pi, np.pi] if column >= 3 else [lo - pad, hi + pad])
        for ax, x, panel_title in zip(axes[row], samples, ("Target", "Generated at best eps")):
            counts, xedges, yedges = np.histogram2d(x[:, columns[0]], x[:, columns[1]], bins=bins, range=ranges)
            fes = -np.log(np.maximum(counts / counts.sum(), 1e-12))
            fes -= fes.min()
            mesh = ax.pcolormesh(xedges, yedges, fes.T, cmap="viridis", vmin=0, vmax=6, shading="auto")
            ax.set(title=panel_title, xlabel=labels[columns[0]], ylabel=labels[columns[1]])
            if row == 0:
                ax.set_aspect("equal")
                for basin, color in zip(ANGLE_BASINS, ("tab:red", "tab:blue", "tab:orange")):
                    for (cx, cy), (hx, hy) in _scaled_basin_boxes(basin, half_width):
                        ax.add_patch(Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                                               fill=False, edgecolor=color, linewidth=1.2))
        fig.colorbar(mesh, ax=axes[row], label=r"$F/k_BT$", shrink=0.85)
    fig.suptitle(title)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return save_path


@torch.no_grad()
def plot_epsilon_loss_surface(model, frame_ds, cfg, loss_fn, initial_eps, optimized_eps):
    """Scan the optimization objective, reusing one noise draw across grid points.

    The scan runs after optimization and preserves the caller's random state.
    Its minimum is a diagnostic and never replaces the selected optimization step.
    """
    if cfg.loss_surface_grid_size < 2:
        raise ValueError("loss_surface_grid_size must be at least 2")
    device = next(model.parameters()).device
    target = frame_ds.traj_ds[cfg.traj_id]["target"].to(device)
    if len(target) > cfg.loss_surface_max_target_points:
        indices = torch.linspace(0, len(target) - 1, cfg.loss_surface_max_target_points, device=device).long()
        target = target[indices]
    eps1_values = np.linspace(frame_ds.eps_min[0], frame_ds.eps_max[0], cfg.loss_surface_grid_size)
    eps2_values = np.linspace(frame_ds.eps_min[1], frame_ds.eps_max[1], cfg.loss_surface_grid_size)
    losses = np.empty((len(eps2_values), len(eps1_values)))
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        for j, eps2 in enumerate(eps2_values):
            for i, eps1 in enumerate(eps1_values):
                torch.manual_seed(cfg.seed)
                generated = sample_geometry(model, (eps1, eps2), frame_ds,
                                            cfg.loss_surface_num_samples, cfg.sample_steps, clamp=False)
                losses[j, i] = float(loss_fn(target, generated).item())
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    mesh = ax.imshow(losses, origin="lower", aspect="auto",
                     extent=(eps1_values[0], eps1_values[-1], eps2_values[0], eps2_values[-1]))
    label = f"{cfg.loss_dim}D " + ("KL" if cfg.loss_type.lower() == "kl" else "MMD²")
    fig.colorbar(mesh, ax=ax, label=label)
    for pair, marker, name in ((frame_ds.get_traj_eps_raw(cfg.traj_id), "x", "Target"),
                               (initial_eps, "s", "Initial"), (optimized_eps, "^", "Selected")):
        ax.scatter(pair[0], pair[1], marker=marker, s=70, label=name)
    ax.set(xlabel=r"$\epsilon_1$", ylabel=r"$\epsilon_2$", title=f"{label} optimization loss surface")
    ax.legend()
    path = Path(cfg.plot_dir or ".") / f"traj_{cfg.traj_id}_epsilon_loss_surface.png"
    fig.savefig(path, dpi=cfg.plot_dpi)
    plt.close(fig)
    return losses
