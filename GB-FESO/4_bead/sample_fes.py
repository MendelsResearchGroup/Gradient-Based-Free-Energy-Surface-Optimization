"""Compare target and sampled angular free energies. Edit the paths below.

F/kBT = -log(P_bin) + C assumes equilibrium samples at the same temperature.
Each panel has its own zero and the same display cap. No smoothing is applied.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle
import numpy as np
import torch

from dataset import LammpsTrajectoryGeometryDataset
from train import TrainConfig, build_model
from utils import ANGLE_BASINS


DATA_ROOT = Path("data/validation")
CHECKPOINT_PATH = Path("ddpm_best_epoch_0080.pt")
OUTPUT_DIR = Path("sample_fes")
TRAJ_IDS = None  # None selects all trajectories; [0, 1] selects zero-based IDs.
NUM_SAMPLES = 20000
BATCH_SIZE = 2048
BINS = 80
FES_CAP = 6.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASINS = tuple((letter, color, basin["boxes"]) for letter, color, basin in
               zip("ABC", ("tab:red", "tab:blue", "tab:orange"), ANGLE_BASINS))


def histogram_fes(angles, bins=80, cap=6.0):
    """Return the angular histogram FES, transposed for pcolormesh."""
    angles = torch.as_tensor(angles).detach().cpu().numpy()
    if angles.ndim != 2 or angles.shape[1] != 2 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError("Expected nonempty, finite [N, 2] angles")
    if bins < 1 or not np.isfinite(cap) or cap <= 0:
        raise ValueError("Bins and FES cap must be positive")
    angles = (angles + np.pi) % (2 * np.pi) - np.pi
    counts, xedges, yedges = np.histogram2d(
        angles[:, 0], angles[:, 1], bins=bins,
        range=[[-np.pi, np.pi], [-np.pi, np.pi]],
    )
    probability = counts / counts.sum()
    free_energy = -np.log(np.maximum(probability, 1e-12))
    free_energy -= free_energy.min()
    return np.minimum(free_energy, cap).T, xedges, yedges


@torch.no_grad()
def plot_sample():
    if min(NUM_SAMPLES, BATCH_SIZE, BINS) <= 0 or not np.isfinite(FES_CAP) or FES_CAP <= 0:
        raise ValueError("Sample counts, bins, and FES cap must be positive")
    dataset = LammpsTrajectoryGeometryDataset(DATA_ROOT)
    traj_ids = range(len(dataset)) if TRAJ_IDS is None else list(TRAJ_IDS)
    if not traj_ids or any(i < 0 or i >= len(dataset) for i in traj_ids):
        raise ValueError("TRAJ_IDS must select existing trajectories")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    cfg = TrainConfig(**checkpoint["train_config"])
    if cfg.x_dim != 5:
        raise ValueError("Expected a checkpoint for five internal coordinates")
    model = build_model(cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for traj_id in traj_ids:
        target = dataset[traj_id]
        eps = target["eps"]
        target_angles = target["angles"]
        if not torch.isfinite(eps).all():
            raise ValueError(f"Trajectory {traj_id}: nonfinite epsilons")
        print(f"Trajectory {traj_id}: sampling at eps={eps.tolist()} on {DEVICE}", flush=True)
        torch.manual_seed(cfg.seed + traj_id)
        chunks = []
        for start in range(0, NUM_SAMPLES, BATCH_SIZE):
            count = min(BATCH_SIZE, NUM_SAMPLES - start)
            generated = model.ddim_sample(
                eps.to(DEVICE).expand(count, 2), shape=(count, 5),
                sample_steps=cfg.sample_steps, eta=0.0, clamp_x0=False,
            )
            # The angular marginal needs only the angle/pi normalization.
            chunks.append((np.pi * generated[:, -2:]).cpu())
        generated_angles = torch.cat(chunks)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5.4), layout="constrained")
        for ax, angles, title in zip(
            axes, (target_angles, generated_angles), ("Ground truth FES", "Generated FES"),
        ):
            fes, xedges, yedges = histogram_fes(angles, bins=BINS, cap=FES_CAP)
            mesh = ax.pcolormesh(xedges, yedges, fes, shading="auto", cmap="viridis",
                                 vmin=0, vmax=FES_CAP)
            ax.set(xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi), aspect="equal",
                   xticks=range(-3, 4), yticks=range(-3, 4))
            ax.set_title(title, fontsize=20, pad=10)
            ax.set_xlabel(r"$\theta_{123}$ [rad]", fontsize=18)
            ax.set_ylabel(r"$\theta_{234}$ [rad]", fontsize=18)
            ax.tick_params(labelsize=16)
            for letter, color, boxes in BASINS:
                for (cx, cy), (hx, hy) in boxes:
                    box = Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                                    fill=False, edgecolor=color, linewidth=1.4, zorder=3)
                    box.set_path_effects([pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()])
                    ax.add_patch(box)
                    ax.text(cx, cy, letter, ha="center", va="center", fontsize=9,
                            fontweight="bold", color=color, zorder=4,
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
        colorbar = fig.colorbar(mesh, ax=axes, shrink=0.8, pad=0.03)
        colorbar.set_label(r"$F/k_BT$", fontsize=18)
        colorbar.ax.tick_params(labelsize=16)
        fig.suptitle(rf"$\epsilon_1={eps[0].item():.6g};\quad \epsilon_2={eps[1].item():.6g}$",
                     fontsize=18)
        path = OUTPUT_DIR / f"traj_{traj_id:03d}_fes.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    plot_sample()
