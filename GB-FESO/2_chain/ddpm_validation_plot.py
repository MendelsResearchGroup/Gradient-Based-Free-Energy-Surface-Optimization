from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from sampler import DifferentiableDDIMSampler
from model import FiLMDDPM
from dataset import CONTACT_CUTOFF, LammpsDistanceDataset


DATA_ROOT = Path("data/val")
CHECKPOINT = Path("ddpm.pt")
OUTPUT_DIR = Path("validation_maps")

# None -> validate every trajectory found in DATA_ROOT.
VALIDATION_TRAJECTORY_IDS = None

DDIM_STEPS = 50
GUIDANCE_SCALE = 1.0
SAMPLE_BATCH_SIZE = 256
RANDOM_SEED = 0

# None -> use the same number of generated frames as GT frames in that trajectory.
GENERATED_FRAMES_PER_TRAJECTORY = None


def target_contact_map(dataset, trajectory_id, batch_size=SAMPLE_BATCH_SIZE):
    subset = dataset.trajectory(trajectory_id)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    total_contacts = torch.zeros(5, 5, dtype=torch.float64)
    total_frames = 0

    for batch in loader:
        distances = batch["distances"].double()
        contacts = (distances[:, :5, 5:] <= CONTACT_CUTOFF).double()
        total_contacts += contacts.sum(dim=0)
        total_frames += distances.shape[0]

    if total_frames == 0:
        raise ValueError(f"Trajectory {trajectory_id:04d} has no retained frames.")

    return total_contacts / total_frames, total_frames


@torch.no_grad()
def generated_contact_map(
    sampler,
    epsilons,
    frames,
    batch_size=SAMPLE_BATCH_SIZE,
    ddim_steps=DDIM_STEPS,
    guidance_scale=GUIDANCE_SCALE,
):
    if frames < 1 or batch_size < 1:
        raise ValueError("frames and batch_size must be positive.")
    total_contacts = torch.zeros(5, 5, dtype=torch.float64)
    generated = 0

    while generated < frames:
        current = min(batch_size, frames - generated)
        matrices = sampler.sample(
            epsilons=epsilons,
            batch_size=current,
            num_steps=ddim_steps,
            guidance_scale=guidance_scale,
        )
        if not torch.isfinite(matrices).all():
            raise FloatingPointError("Nonfinite generated distances; validation stopped.")
        contacts = (matrices[:, :5, 5:] <= CONTACT_CUTOFF).double().cpu()
        total_contacts += contacts.sum(dim=0)
        generated += current

    return total_contacts / frames


def save_contact_map_comparison(target_map, generated_map, output_file, title):
    difference = generated_map - target_map
    mse = difference.square().mean().item()
    mae = difference.abs().mean().item()
    difference_limit = max(float(difference.abs().max()), 1e-6)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
    target_image = axes[0].imshow(target_map.numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
    axes[1].imshow(generated_map.numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
    difference_image = axes[2].imshow(
        difference.numpy(),
        vmin=-difference_limit,
        vmax=difference_limit,
        cmap="RdBu_r",
    )

    axes[0].set_title("Reference")
    axes[1].set_title("Generated")
    axes[2].set_title(f"Generated - reference\nMSE={mse:.5f}   MAE={mae:.5f}")

    for axis, matrix in zip(axes, (target_map, generated_map, difference)):
        axis.set_xticks(range(5), labels=range(6, 11))
        axis.set_yticks(range(5), labels=range(1, 6))
        axis.set_xlabel("Chain 2 atom ID")
        axis.set_ylabel("Chain 1 atom ID")
        for i in range(5):
            for j in range(5):
                value = float(matrix[i, j])
                if matrix is difference:
                    text_color = "white" if abs(value) > 0.55 * difference_limit else "black"
                else:
                    text_color = "white" if value < 0.5 else "black"
                axis.text(
                    j,
                    i,
                    f"{value:+.3f}" if matrix is difference else f"{value:.3f}",
                    ha="center",
                    va="center",
                    color=text_color,
                )

    figure.colorbar(target_image, ax=axes[:2], label="contact probability", shrink=0.85)
    figure.colorbar(difference_image, ax=axes[2], label="probability difference", shrink=0.85)
    figure.suptitle(title)
    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return mse, mae


def main():
    if SAMPLE_BATCH_SIZE < 1:
        raise ValueError("SAMPLE_BATCH_SIZE must be positive.")
    if (GENERATED_FRAMES_PER_TRAJECTORY is not None
            and GENERATED_FRAMES_PER_TRAJECTORY < 1):
        raise ValueError("GENERATED_FRAMES_PER_TRAJECTORY must be positive or None.")
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Indexing validation trajectories under {DATA_ROOT}...", flush=True)
    dataset = LammpsDistanceDataset(DATA_ROOT)
    metadata_by_id = {traj["id"]: traj for traj in dataset.trajectories}
    validation_ids = (list(metadata_by_id) if VALIDATION_TRAJECTORY_IDS is None
                      else VALIDATION_TRAJECTORY_IDS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    if (checkpoint.get("condition_atom_ids") != list(range(1, 11))
            or checkpoint.get("spatial_dimensions") != 2):
        raise ValueError("Use a 2D checkpoint with ten epsilons in atom-ID order.")
    model = FiLMDDPM().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval().requires_grad_(False)
    sampler = DifferentiableDDIMSampler(
        model,
        diffusion_steps=checkpoint["diffusion_steps"],
        beta_start=checkpoint["beta_start"],
        beta_end=checkpoint["beta_end"],
    )

    print(f"Validating {len(validation_ids)} trajectories on {device}", flush=True)
    print(f"dataset ranges: {dataset.ranges}", flush=True)

    for trajectory_id in validation_ids:
        metadata = metadata_by_id[trajectory_id]
        epsilons = metadata["epsilons"]
        eps_text = ", ".join(f"{value:g}" for value in epsilons)

        print(f"Validating trajectory {trajectory_id:04d}...", flush=True)
        target_map, target_frames = target_contact_map(
            dataset, trajectory_id, batch_size=SAMPLE_BATCH_SIZE,
        )
        generated_frames = (
            target_frames
            if GENERATED_FRAMES_PER_TRAJECTORY is None
            else int(GENERATED_FRAMES_PER_TRAJECTORY)
        )
        generated_map = generated_contact_map(
            sampler,
            epsilons=epsilons,
            frames=generated_frames,
            batch_size=SAMPLE_BATCH_SIZE,
            ddim_steps=DDIM_STEPS,
            guidance_scale=GUIDANCE_SCALE,
        )
        output_file = OUTPUT_DIR / f"contact_map_{trajectory_id:04d}.png"
        title = (
            f"Contact probabilities | trajectory {trajectory_id:04d}\n"
            f"epsilons=[{eps_text}]\n"
            f"reference/generated frames={target_frames}/{generated_frames} | "
            f"DDIM steps={DDIM_STEPS} | CFG={GUIDANCE_SCALE:g} | seed={RANDOM_SEED}"
        )
        mse, mae = save_contact_map_comparison(
            target_map,
            generated_map,
            output_file,
            title,
        )

        print(
            f"{trajectory_id:04d}  epsilons=[{eps_text}]  "
            f"frames(gt/gen)={target_frames}/{generated_frames}  "
            f"MSE={mse:.6f} MAE={mae:.6f}  "
            f"saved={output_file}",
            flush=True,
        )


if __name__ == "__main__":
    main()
