"""Plot saved optimization steps, target FES, and matching MD when available."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import parse_dump_geometry, read_epsilons_from_in
from sample_fes import histogram_fes


SUMMARY_DIR = Path("results")
MD_ROOT = Path("md_control")


def plot_optimization_progress(summary_path, md_root=None, dpi=180):
    """Plot step 0, available intermediates, best, target, and matching MD."""
    summary_path = Path(summary_path)
    with np.load(summary_path, allow_pickle=False) as data:
        steps = data["steps"]
        losses = data["losses"]
        eps = data["eps"]
        angles = data["angles"]
        target_angles = data["target_angles"]
        target_eps = data["target_eps"]
        loss_label = str(data["loss_label"])
        bins = int(data["bins"])
    md_root = Path(md_root) if md_root is not None else MD_ROOT
    best_eps = eps[-1]

    md_angles = []
    md_paths = []
    md_inputs = []
    for in_path in sorted(md_root.rglob("in.in")):
        md_eps = read_epsilons_from_in(str(in_path))
        # Older inputs use six decimal places; this is rounding tolerance only.
        if np.allclose(md_eps, best_eps, rtol=0.0, atol=1e-6):
            md_inputs.append(in_path)
            dump_path = in_path.parent / "dump.lammpstrj"
            if not dump_path.is_file() or dump_path.stat().st_size == 0:
                continue
            _, geometry = parse_dump_geometry(str(dump_path))
            if geometry.size == 0:
                raise ValueError(f"No MD frames in {dump_path}")
            md_angles.append(geometry[:, -2:])
            md_paths.append(str(dump_path))
    md_angles = np.concatenate(md_angles) if md_angles else None

    panels = []
    for i, step in enumerate(steps):
        title = f"step {step}"
        if i == len(steps) - 1:
            title = f"best: {title}"
        caption = f"{loss_label}={losses[i]:.3f}\neps=({eps[i, 0]:.3f}, {eps[i, 1]:.3f})"
        panels.append((angles[i], title, caption))
    panels.append((target_angles, "target",
                   f"eps=({target_eps[0]:.3f}, {target_eps[1]:.3f})"))
    panels.append((md_angles, "MD at best eps",
                   f"eps=({best_eps[0]:.3f}, {best_eps[1]:.3f})"))

    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.9),
                             layout="compressed", squeeze=False)
    fig.get_layout_engine().set(w_pad=0.02, wspace=0.01)
    axes = axes[0]
    mesh = None
    for ax, (samples, title, caption) in zip(axes, panels):
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(r"$\theta_{123}$ [rad]" + "\n" + caption, fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel(r"$\theta_{234}$ [rad]", fontsize=12)
            ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        else:
            ax.set_yticks([])
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_aspect("equal")
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        ax.tick_params(labelsize=11)
        if samples is None:
            ax.set_facecolor("0.95")
            ax.text(0.5, 0.5, "Matching MD evaluation\nnot available yet",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12)
            continue
        if samples.ndim != 2 or samples.shape[1] != 2 or not len(samples) or not np.isfinite(samples).all():
            plt.close(fig)
            raise ValueError(f"{title} requires nonempty, finite angular samples")
        # Wrap periodically instead of dropping samples outside [-pi, pi].
        samples = np.arctan2(np.sin(samples), np.cos(samples))
        free_energy, xedges, yedges = histogram_fes(
            torch.from_numpy(samples), bins=bins, cap=6.0,
        )
        mesh = ax.pcolormesh(xedges, yedges, free_energy, shading="auto",
                            cmap="viridis", vmin=0.0, vmax=6.0)
    colorbar = fig.colorbar(mesh, ax=axes, shrink=0.82, pad=0.01, fraction=0.02)
    colorbar.set_label(r"$F/k_BT$ (minimum shifted to 0)", fontsize=12)
    colorbar.ax.tick_params(labelsize=11)
    save_path = summary_path.with_suffix(".png")
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"saved optimization progress: {save_path}")
    if md_paths:
        print("MD evaluation: " + ", ".join(md_paths))
    elif md_inputs:
        print(f"MD inputs at eps=({best_eps[0]:.8g}, {best_eps[1]:.8g}) are prepared, "
              "but no nonempty dump.lammpstrj is available. Run MD and redraw: "
              + ", ".join(str(path.parent) for path in md_inputs))
    else:
        print(f"No matching MD input at eps=({best_eps[0]:.8g}, {best_eps[1]:.8g}) "
              f"in {md_root}; redraw after the matching MD run is available.")
    return save_path


if __name__ == "__main__":
    for summary_path in sorted(SUMMARY_DIR.rglob("*_optimization_progress.npz")):
        plot_optimization_progress(summary_path, md_root=MD_ROOT)
