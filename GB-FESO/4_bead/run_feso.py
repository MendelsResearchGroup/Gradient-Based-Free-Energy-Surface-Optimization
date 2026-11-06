"""Recover epsilon conditions, then score the selected samples in five dimensions."""

import csv
import math
from pathlib import Path
import time
from dataclasses import dataclass, replace

import numpy as np
import torch
from scipy import ndimage, special

from ddpm_reparam import EpsilonOptConfig as BaseEpsilonOptConfig
from ddpm_reparam import optimize_epsilon_condition as optimize_base
from loss import kde_kl_target_to_generated, mmd_target_to_generated
from train import load_checkpoint
from utils import DEFAULT_BASIN_HALF_WIDTH, active_basin_indices, angle_view


@dataclass
class EpsilonOptConfig(BaseEpsilonOptConfig):
    fes_js_threshold: float = 0.10


def compare_fes(target_samples, generated_samples, bins=80, bandwidth=0.32, js_threshold=0.10):
    """Compare joint (theta123, theta234) probabilities on one periodic grid.

    The smoothing bandwidth is in radians, not histogram bins. Empty bins need
    no pseudocount: their contribution to Jensen-Shannon divergence is zero.
    """
    if not isinstance(bins, (int, np.integer)) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    if not 0 <= js_threshold <= 1:
        raise ValueError("js_threshold must be between 0 and 1")

    probabilities = []
    for samples in (target_samples, generated_samples):
        angles = angle_view(samples).detach().to(device="cpu", dtype=torch.float64).numpy()
        if len(angles) == 0 or not np.isfinite(angles).all():
            raise ValueError("FES comparison requires nonempty, finite angular samples")
        angles = (angles + math.pi) % (2 * math.pi) - math.pi
        counts, _, _ = np.histogram2d(
            angles[:, 0], angles[:, 1], bins=bins,
            range=[[-math.pi, math.pi], [-math.pi, math.pi]],
        )
        probability = ndimage.gaussian_filter(
            counts, sigma=bandwidth * bins / (2 * math.pi), mode="wrap",
        )
        probabilities.append(probability / probability.sum())

    p, q = probabilities
    midpoint = 0.5 * (p + q)
    divergence = 0.5 * (special.rel_entr(p, midpoint) + special.rel_entr(q, midpoint)).sum()
    # Clip only floating-point roundoff at the theoretical [0, 1] bounds.
    distance = float(np.sqrt(np.clip(divergence / math.log(2), 0.0, 1.0)))
    return {
        "fes_js_distance": distance,
        "fes_similar": distance <= js_threshold,
        "fes_js_threshold": js_threshold,
        "fes_bins": bins,
        "fes_bandwidth_rad": bandwidth,
    }


def optimize_epsilon_condition(model, frame_ds, cfg):
    """Finish optimization and best-step selection before computing 5D scores.

    Use the exact selected target/generated samples, with gradients disabled.
    FES and basin diagnostics describe the angular marginal in either mode.
    """
    if not cfg.eval_basin_metrics:
        raise ValueError("eval_basin_metrics must be True for recovery")
    if not 0 <= cfg.active_target_prob < 1 or not 0 <= cfg.fes_js_threshold <= 1:
        raise ValueError("Invalid basin probability or FES distance threshold")

    result = optimize_base(model, frame_ds, cfg)

    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    evaluation_start = time.perf_counter()
    with torch.no_grad():
        target = result["eval_target_rad"].to(device)
        generated = result["eval_generated_rad"].to(device)
        if target.shape[1] != 5 or generated.shape[1] != 5:
            raise ValueError("Final evaluation requires all five coordinates")
        kl_5d = float(kde_kl_target_to_generated(
            target, generated, bandwidth=cfg.kde_bandwidth, kl_lambda=cfg.kl_lambda,
        ).item())
        mmd_5d = float(mmd_target_to_generated(
            target, generated, bandwidth=cfg.mmd_bandwidth,
        ).item())
    if not math.isfinite(kl_5d) or not math.isfinite(mmd_5d):
        raise ValueError("Nonfinite final 5D loss")

    metrics = result["eval_metrics"]
    target_active = metrics["active_indices"]
    generated_active = active_basin_indices(
        metrics["generated_probs"], min_target_prob=cfg.active_target_prob,
        include_outside=False,
    )
    names = metrics["basin_names"]
    target_counts, generated_counts = metrics["target_counts"], metrics["generated_counts"]
    target_ab = all(i in target_active for i in (0, 1))
    generated_ab = all(i in generated_active for i in (0, 1))
    delta_target = math.log(target_counts[0] / target_counts[1]) if min(target_counts[:2]) > 0 else None
    delta_generated = math.log(generated_counts[0] / generated_counts[1]) if min(generated_counts[:2]) > 0 else None
    delta_error = abs(delta_generated - delta_target) if target_ab and generated_ab else None
    relative_error = (100 * delta_error / abs(delta_target)
                      if delta_error is not None and delta_target != 0 else None)

    recovery = {
        "trajID": cfg.traj_id,
        "start_eps1": result["initial_eps"][0],
        "start_eps2": result["initial_eps"][1],
        "clamp_eps": cfg.clamp_eps,
        "optimize_eps": cfg.optimize_eps,
        "loss_dim": cfg.loss_dim,
        "loss_type": result["loss_type"],
        "optimization_loss": result["eval_loss"],
        "optimization_kl": result["optimization_kl"],
        "eval_phase": result["eval_phase"],
        "eval_step": result["eval_step"],
        "evaluation_dim": 5,
        "evaluation_samples": "selected_step",
        "KL_5D": kl_5d,
        "MMD2_5D": mmd_5d,
        "eps1_target": result["target_eps"][0],
        "eps2_target": result["target_eps"][1],
        "eps1_optimized": result["eval_eps"][0],
        "eps2_optimized": result["eval_eps"][1],
        "kde_bandwidth": cfg.kde_bandwidth,
        "mmd_bandwidth": cfg.mmd_bandwidth,
        "kl_lambda": cfg.kl_lambda,
        "seed": cfg.seed,
        "num_samples": len(generated),
        "num_target_samples": len(target),
        "sample_steps": cfg.sample_steps,
        "steps": cfg.steps,
        "lr": cfg.lr,
        "lr_refine": cfg.lr_refine,
        "always_refine": cfg.always_refine,
        "refine_kl_threshold": cfg.refine_kl_threshold,
        "phase0_eps_noise_std": cfg.phase0_eps_noise_std,
        "phase0_high_high_eps_noise_std": cfg.phase0_high_high_eps_noise_std,
        "high_high_eps_threshold": cfg.high_high_eps_threshold,
        "basin_half_width": DEFAULT_BASIN_HALF_WIDTH if cfg.basin_half_width is None else cfg.basin_half_width,
        "delta_f_prob_floor": cfg.delta_f_prob_floor,
        "delta_f_error_cap": cfg.delta_f_error_cap,
        "df_median_abs_error": metrics["df_median_abs_error"],
        "same_basins_hit": bool(target_active) and set(target_active) == set(generated_active),
        "target_active_basins": ";".join(names[i] for i in target_active),
        "generated_active_basins": ";".join(names[i] for i in generated_active),
        "active_basin_threshold": cfg.active_target_prob,
        "target_A_count": target_counts[0],
        "target_B_count": target_counts[1],
        "generated_A_count": generated_counts[0],
        "generated_B_count": generated_counts[1],
        "both_basins_present": target_ab and generated_ab,
        "deltaG_target_kBT": delta_target,
        "deltaG_optimized_kBT": delta_generated,
        "abs_deltaG_error_kBT": delta_error,
        "relative_deltaG_error_percent": relative_error,
        **compare_fes(
            result["eval_target_rad"], result["eval_generated_rad"],
            bins=cfg.plot_bins, bandwidth=cfg.kde_bandwidth,
            js_threshold=cfg.fes_js_threshold,
        ),
        "optimization_seconds": result["optimization_seconds"],
    }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    recovery["evaluation_seconds"] = time.perf_counter() - evaluation_start
    result["recovery"] = recovery
    print(
        f"Selected {cfg.loss_dim}D {result['loss_type']} {result['eval_phase']} step {result['eval_step']}: "
        f"eps={result['eval_eps']} | final 5D KL={kl_5d:.6g}, MMD^2={mmd_5d:.6g}"
    )
    return result


def run_recovery(data_root, checkpoint_path, config, start_eps_pairs, traj_ids=None, output_root="results"):
    """Run one batch and write recovery.csv under <output_root>/<dimension>d_<loss>."""
    run_dir = Path(output_root) / f"{config.loss_dim}d_{config.loss_type.strip().lower()}"
    results_path = run_dir / "recovery.csv"
    if results_path.exists():
        raise FileExistsError(f"{results_path} already exists; choose a new output_root for another batch")
    model, frame_ds, _ = load_checkpoint(checkpoint_path, root_dir=str(data_root))
    traj_ids = list(range(len(frame_ds.traj_ds))) if traj_ids is None else list(traj_ids)
    start_eps_pairs = list(start_eps_pairs)
    if not traj_ids or not start_eps_pairs:
        raise ValueError("Select at least one trajectory and starting pair")
    if any(i < 0 or i >= len(frame_ds.traj_ds) for i in traj_ids):
        raise ValueError("traj_ids must select existing trajectories")
    run_dir.mkdir(parents=True, exist_ok=True)
    with results_path.open("x", newline="") as handle:
        writer = None
        for start_id, (eps1, eps2) in enumerate(start_eps_pairs):
            for traj_id in traj_ids:
                cfg = replace(
                    config, traj_id=traj_id, init_optimized_raw_1=eps1, init_optimized_raw_2=eps2,
                    plot_dir=str(run_dir / f"traj_{traj_id:03d}_start_{start_id:02d}"),
                )
                result = optimize_epsilon_condition(model, frame_ds, cfg)
                row = {"start_id": start_id, **result["recovery"]}
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                handle.flush()
                print(f"Saved recovery: {results_path}")
    return results_path


if __name__ == "__main__":
    DATA_ROOT = Path("data/validation")
    CHECKPOINT_PATH = Path("ddpm_best_epoch_0080.pt")
    OUTPUT_ROOT = Path("results")
    TARGET_TRAJ_IDS = None  # None runs all trajectories; [0, 1] selects specific IDs.
    START_EPS_PAIRS = [(0.25, 0.25), (0.25, 1.4), (1.4, 0.25), (1.4, 1.4)]

    config = EpsilonOptConfig(
        traj_id=0,  # Replaced by each selected trajectory ID.
        loss_dim=2,                     # 2 or 5; final scoring always uses 5.
        loss_type="kl",                 # "kl" or "mmd"
        optimize_eps="both",            # "eps1", "eps2", or "both"
        fixed_eps_raw=0.5,
        clamp_eps=True,
        kde_bandwidth=0.5,
        mmd_bandwidth=0.32,
        kl_lambda=1.0,
        steps=201,
        lr=5e-2,
        lr_refine=5e-2,
        always_refine=False,
        refine_kl_threshold=0.1,
        num_samples=3500,
        max_target_points=4496,
        sample_steps=100,
        seed=0,
        save_plots=True,
        plot_loss_surface=False,
        loss_surface_grid_size=25,
        loss_surface_num_samples=3500,
        md_eval_root="md_control",
    )
    run_recovery(DATA_ROOT, CHECKPOINT_PATH, config, START_EPS_PAIRS,
                 traj_ids=TARGET_TRAJ_IDS, output_root=OUTPUT_ROOT)


