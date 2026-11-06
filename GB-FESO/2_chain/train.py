from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import FiLMDDPM
from dataset import LammpsDistanceDataset


DATA_ROOT = Path("data/train")
VAL_ROOT = Path("data/val")
CHECKPOINT = Path("ddpm.pt")

EPOCHS = 80
BATCH_SIZE = 256
LEARNING_RATE = 2e-4
NUM_WORKERS = 4
VALIDATION_SEED = 0  # Reuse validation timesteps/noise for comparable epoch losses.

DIFFUSION_STEPS = 1000
BETA_START = 1e-4
BETA_END = 2e-2


def chain_symmetry_permutations(device):
    """Topology-preserving relabelings; epsilons must move with their beads."""
    identity = torch.arange(10, device=device)
    reverse_1 = torch.tensor([4, 3, 2, 1, 0, 5, 6, 7, 8, 9], device=device)
    reverse_2 = torch.tensor([0, 1, 2, 3, 4, 9, 8, 7, 6, 5], device=device)
    reverse_both = torch.tensor([4, 3, 2, 1, 0, 9, 8, 7, 6, 5], device=device)
    base = torch.stack((identity, reverse_1, reverse_2, reverse_both))
    swap = torch.tensor([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], device=device)
    return torch.cat((base, base[:, swap]), dim=0)


def random_chain_symmetry(x, epsilons, permutations):
    """Relabel both distance axes and the condition with the same permutation."""
    batch, n, _ = x.shape
    perm = permutations[torch.randint(len(permutations), (batch,), device=x.device)]
    x = x.gather(1, perm[:, :, None].expand(-1, -1, n))
    x = x.gather(2, perm[:, None, :].expand(-1, n, -1))
    return x, epsilons.gather(1, perm)


def symmetric_noise_like(x, generator=None):
    """N(0,1) on independent upper-triangle entries, mirrored below diagonal."""
    noise = torch.triu(
        torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator), diagonal=1,
    )
    return noise + noise.transpose(-1, -2)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = LammpsDistanceDataset(DATA_ROOT)
    val_dataset = LammpsDistanceDataset(VAL_ROOT)
    print(f"training frames: {len(dataset)}; validation frames: {len(val_dataset)}")
    print(f"ranges (reported only; not used for training): {dataset.ranges}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(VALIDATION_SEED),
    )

    model = FiLMDDPM(channels=64, cond_dim=128, cond_drop_prob=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    betas = torch.linspace(BETA_START, BETA_END, DIFFUSION_STEPS, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    sqrt_alpha_bar = alpha_bar.sqrt()
    sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

    permutations = chain_symmetry_permutations(device)
    upper_triangle = torch.triu(
        torch.ones(10, 10, dtype=torch.bool, device=device), diagonal=1
    )

    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch"):
            x0 = batch["distances"].to(device, non_blocking=True)
            epsilons = batch["epsilons"].to(device, non_blocking=True)

            x0, epsilons = random_chain_symmetry(x0, epsilons, permutations)

            t = torch.randint(DIFFUSION_STEPS, (x0.shape[0],), device=device)
            noise = symmetric_noise_like(x0)
            x_t = (
                sqrt_alpha_bar[t, None, None] * x0
                + sqrt_one_minus_alpha_bar[t, None, None] * noise
            )

            predicted_noise = model(x_t, t, epsilons)

            loss = F.mse_loss(
                predicted_noise[:, upper_triangle],
                noise[:, upper_triangle],
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = x0.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        epoch_loss = total_loss / total_samples
        model.eval()
        val_generator = torch.Generator(device=device).manual_seed(VALIDATION_SEED)
        val_total_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation {epoch}/{EPOCHS}", unit="batch"):
                x0 = batch["distances"].to(device, non_blocking=True)
                epsilons = batch["epsilons"].to(device, non_blocking=True)
                # Validate the conditional model in atom-ID order, without augmentation.
                t = torch.randint(
                    DIFFUSION_STEPS, (x0.shape[0],), device=device, generator=val_generator,
                )
                noise = symmetric_noise_like(x0, generator=val_generator)
                x_t = (
                    sqrt_alpha_bar[t, None, None] * x0
                    + sqrt_one_minus_alpha_bar[t, None, None] * noise
                )
                predicted_noise = model(x_t, t, epsilons)
                loss = F.mse_loss(predicted_noise[:, upper_triangle], noise[:, upper_triangle])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite validation loss; training stopped.")
                val_total_loss += loss.item() * x0.shape[0]
                val_samples += x0.shape[0]

        val_loss = val_total_loss / val_samples
        print(f"epoch {epoch:03d}  train_loss={epoch_loss:.6f}  val_loss={val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": epoch_loss,
                "val_loss": val_loss,
                "validation_seed": VALIDATION_SEED,
                "diffusion_steps": DIFFUSION_STEPS,
                "beta_start": BETA_START,
                "beta_end": BETA_END,
                "dataset_ranges": dataset.ranges,
                "condition_atom_ids": list(range(1, 11)),
                "spatial_dimensions": 2,
            }, CHECKPOINT)
            print(f"Saved best epoch {epoch} (val_loss={best_val_loss:.6f}) to {CHECKPOINT}")


if __name__ == "__main__":
    main()
