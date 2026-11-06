"""Train the model and save its best epoch."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import LammpsTrajectoryGeometryDataset, GeometryFrameDataset
from model import FiLMResMLP, AngleDiffusion


@dataclass
class TrainConfig:
    root_dir: str
    batch_size: int = 256
    lr: float = 1.5e-4
    weight_decay: float = 1e-4
    epochs: int = 80
    num_diffusion_steps: int = 1000
    sample_steps: int = 100
    x_dim: int = 5
    width: int = 256
    depth: int = 7
    time_dim: int = 64
    cond_dim: int = 128
    dropout: float = 0.05
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    best_epoch: int | None = None
    best_loss: float | None = None


def make_frame_dataset(root_dir: str) -> GeometryFrameDataset:
    traj_ds = LammpsTrajectoryGeometryDataset(root_dir)
    return GeometryFrameDataset(traj_ds)


def make_dataloader(root_dir: str, batch_size: int) -> tuple[DataLoader, GeometryFrameDataset]:
    frame_ds = make_frame_dataset(root_dir)
    loader = DataLoader(frame_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, frame_ds


def build_model(cfg: TrainConfig) -> AngleDiffusion:
    backbone = FiLMResMLP(
        x_dim=cfg.x_dim,
        width=cfg.width,
        depth=cfg.depth,
        time_dim=cfg.time_dim,
        cond_dim=cfg.cond_dim,
        dropout=cfg.dropout,
    )
    return AngleDiffusion(backbone, num_steps=cfg.num_diffusion_steps)


def train(cfg: TrainConfig) -> tuple[AngleDiffusion, GeometryFrameDataset]:
    torch.manual_seed(cfg.seed)

    loader, frame_ds = make_dataloader(cfg.root_dir, cfg.batch_size)
    if cfg.x_dim != frame_ds.target_dim:
        raise ValueError(f"cfg.x_dim={cfg.x_dim} but dataset target_dim={frame_ds.target_dim}")

    model = build_model(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_epoch = 0
    best_loss = float("inf")
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        for x0, eps_cond, _tid in loader:
            x0 = x0.to(cfg.device)
            eps_cond = eps_cond.to(cfg.device)

            loss = model.training_loss(x0, eps_cond)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item() * x0.shape[0]
            count += x0.shape[0]

        epoch_loss = running / max(count, 1)
        if epoch_loss < best_loss:
            best_epoch = epoch
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            marker = " *best"
        else:
            marker = ""

        print(f"epoch {epoch:4d} | loss {epoch_loss:.6f}{marker}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(cfg.device)
        model.eval()

    cfg.best_epoch = best_epoch
    cfg.best_loss = best_loss
    print(f"restored best epoch {best_epoch:4d} | loss {best_loss:.6f}")

    return model, frame_ds


def save_checkpoint(
    checkpoint_path: str,
    model: AngleDiffusion,
    frame_ds: GeometryFrameDataset,
    cfg: TrainConfig,
) -> None:
    torch.save(
        {
            "epoch": cfg.best_epoch if cfg.best_epoch is not None else cfg.epochs,
            "best_epoch": cfg.best_epoch,
            "best_loss": cfg.best_loss,
            "model_state_dict": model.state_dict(),
            "train_config": cfg.__dict__,
            "eps_min": frame_ds.eps_min.tolist(),
            "eps_max": frame_ds.eps_max.tolist(),
            "distance_min": frame_ds.distance_min.tolist(),
            "distance_max": frame_ds.distance_max.tolist(),
            "target_columns": list(frame_ds.target_columns),
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: str,
    root_dir: str | None = None,
    device: str | None = None,
) -> tuple[AngleDiffusion, GeometryFrameDataset, TrainConfig]:
    """Load weights, distance scaling, and epsilon bounds from the checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    cfg_dict = dict(checkpoint["train_config"])
    if root_dir is not None:
        cfg_dict["root_dir"] = root_dir
    cfg_dict["device"] = device
    cfg_dict.setdefault("x_dim", 5)

    cfg_fields = TrainConfig.__dataclass_fields__
    cfg = TrainConfig(**{k: v for k, v in cfg_dict.items() if k in cfg_fields})

    frame_ds = GeometryFrameDataset(
        LammpsTrajectoryGeometryDataset(cfg.root_dir),
        eps_range=(checkpoint["eps_min"], checkpoint["eps_max"]),
        distance_range=(checkpoint["distance_min"], checkpoint["distance_max"]),
    )
    model = build_model(cfg).to(cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, frame_ds, cfg


if __name__ == "__main__":
    cfg = TrainConfig(root_dir="data/train")
    model, frame_ds = train(cfg)
    checkpoint_path = Path(f"ddpm_best_epoch_{cfg.best_epoch:04d}.pt")
    save_checkpoint(checkpoint_path, model, frame_ds, cfg)
    print(f"Saved {checkpoint_path}")
