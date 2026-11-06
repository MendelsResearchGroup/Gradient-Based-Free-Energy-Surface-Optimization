"""Optimize epsilon conditions using a 2D or 5D KL/MMD objective."""

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from dataset import GeometryFrameDataset
from model import AngleDiffusion
from loss import kde_kl_target_to_generated, mmd_target_to_generated
from sampler import _eps_pair_tensor
from utils import compare_basin_metrics, format_basin_metrics

EpsPairLike = Union[Sequence[float], torch.Tensor]


@dataclass
class EpsilonOptConfig:
    traj_id: int
    optimize_eps: str = "eps1"              # "eps1", "eps2", or "both"
    fixed_eps_raw: Optional[float] = None   # None => target trajectory value; ignored when optimize_eps="both"
    init_optimized_raw_1: Optional[Union[float, EpsPairLike]] = None  # None => midpoint; scalar or [eps1, eps2]
    init_optimized_raw_2: Optional[Union[float, EpsPairLike]] = None  # None => midpoint; scalar or [eps1, eps2]
    steps: int = 100
    lr: float = 5e-2
    lr_refine: float = 1e-3
    num_samples: int = 4496
    sample_steps: int = 100
    kde_bandwidth: float = 0.32
    kl_lambda: float = 1.0
    max_target_points: int = 4496
    seed: int = 0
    print_every: int = 10
    plot_dir: Optional[str] = None              # None => current working directory
    md_eval_root: Optional[str] = None         # None uses ./md_control
    plot_bins: int = 80
    plot_dpi: int = 180
    plot_loss_surface: bool = False            # scan raw eps1/eps2 grid using the selected loss
    loss_surface_grid_size: int = 25
    loss_surface_num_samples: int = 2048
    loss_surface_max_target_points: int = 2048
    phase0_eps_noise_std: float = 0.05
    phase0_high_high_eps_noise_std: float = 0.005
    high_high_eps_threshold: float = 0.9
    clamp_eps: bool = False                  # Clamp to the checkpoint's training epsilon range.
    eval_basin_metrics: bool = True
    basin_half_width: Optional[float] = None
    active_target_prob: float = 0.05
    min_generated_active_prob: float = 0.005
    delta_f_prob_floor: float = 1e-3
    delta_f_error_cap: Optional[float] = 6.0
    loss_type: str = "kl"                     # "kl" or "mmd" (squared MMD)
    loss_dim: int = 5                         # 2 angles or all 5 coordinates
    mmd_bandwidth: float = 0.32
    save_plots: bool = True
    always_refine: bool = False               # Use both phases for bandwidth comparisons.
    refine_kl_threshold: float = 0.1




def _parse_optimize_eps(name: str) -> Tuple[int, ...]:
    aliases = {
        "eps1": (0,), "epsilon1": (0,), "1": (0,),
        "eps2": (1,), "epsilon2": (1,), "2": (1,),
        "both": (0, 1), "all": (0, 1), "eps12": (0, 1), "eps1_eps2": (0, 1),
    }
    key = str(name).strip().lower()
    if key not in aliases:
        raise ValueError("optimize_eps must be 'eps1', 'eps2', or 'both'")
    return aliases[key]


def _initial_raw_for_optimized_eps(
    init_optimized_raw_1: Optional[Union[float, EpsPairLike]],
    init_optimized_raw_2: Optional[Union[float, EpsPairLike]],
    target_pair: torch.Tensor,
    eps_min: torch.Tensor,
    eps_max: torch.Tensor,
    opt_indices: Tuple[int, ...],
) -> torch.Tensor:
    init_pair = 0.5 * (eps_min + eps_max)
    if init_optimized_raw_1 is None and init_optimized_raw_2 is None:
        return init_pair[list(opt_indices)]
    if init_optimized_raw_1 is not None and torch.as_tensor(init_optimized_raw_1).numel() > 1:
        init_pair = _eps_pair_tensor(init_optimized_raw_1, device=target_pair.device, dtype=target_pair.dtype)
    else:
        if init_optimized_raw_1 is not None:
            init_pair[0] = float(torch.as_tensor(init_optimized_raw_1).item())
        if init_optimized_raw_2 is not None:
            init_pair[1] = float(torch.as_tensor(init_optimized_raw_2).item())
    return init_pair[list(opt_indices)]


def _format_distance_means(target_geometry: torch.Tensor, generated_geometry: torch.Tensor) -> str:
    if target_geometry.shape[-1] < 3 or generated_geometry.shape[-1] < 3:
        return "distance means unavailable"

    target_mean = target_geometry[:, 0:3].detach().mean(dim=0)
    generated_mean = generated_geometry[:, 0:3].detach().mean(dim=0)
    names = ("r12", "r23", "r34")
    return "dist mean gen/target: " + ", ".join(
        f"{name}={generated_mean[i].item():.6f}/{target_mean[i].item():.6f}"
        for i, name in enumerate(names)
    )


def optimize_epsilon_condition(
    model: AngleDiffusion,
    frame_ds: GeometryFrameDataset,
    cfg: EpsilonOptConfig,
) -> Dict[str, object]:
    """Optimize raw LJ epsilons through an affine two-layer parameterization.

    Both layers and their input are trained; the diffusion model stays frozen.
    loss_dim selects the optimization coordinates, including refinement KL and
    loss surfaces. Final 5D scoring is done by ddpm_recovery after this returns.
    """
    loss_type = cfg.loss_type.strip().lower()
    if loss_type not in ("kl", "mmd"):
        raise ValueError("loss_type must be 'kl' or 'mmd'")
    if not isinstance(cfg.loss_dim, int) or cfg.loss_dim not in (2, 5):
        raise ValueError("loss_dim must be 2 or 5")
    if any(not math.isfinite(bw) or bw <= 0 for bw in (cfg.kde_bandwidth, cfg.mmd_bandwidth)):
        raise ValueError("KDE and MMD bandwidths must be finite and positive")
    if not math.isfinite(cfg.kl_lambda) or cfg.kl_lambda < 0:
        raise ValueError("kl_lambda must be finite and nonnegative")
    if cfg.steps < 0 or min(cfg.num_samples, cfg.max_target_points, cfg.sample_steps) < 1:
        raise ValueError("steps must be nonnegative and sample counts must be positive")
    if not math.isfinite(cfg.refine_kl_threshold) or cfg.refine_kl_threshold < 0:
        raise ValueError("refine_kl_threshold must be finite and nonnegative")
    loss_label = f"{cfg.loss_dim}D " + ("KL" if loss_type == "kl" else "MMD^2")
    device = next(model.parameters()).device
    opt_indices = _parse_optimize_eps(cfg.optimize_eps)
    opt_indices_tensor = torch.as_tensor(opt_indices, device=device, dtype=torch.long)
    opt_names = [f"eps{i + 1}" for i in opt_indices]
    opt_name = "+".join(opt_names)
    fixed_indices = tuple(i for i in (0, 1) if i not in opt_indices)
    fixed_name = "+".join(f"eps{i + 1}" for i in fixed_indices) if fixed_indices else None

    target_pair = torch.as_tensor(frame_ds.get_traj_eps_raw(cfg.traj_id), device=device, dtype=torch.float32)
    eps_min = torch.as_tensor(frame_ds.eps_min, device=device, dtype=torch.float32)
    eps_max = torch.as_tensor(frame_ds.eps_max, device=device, dtype=torch.float32)
    lo = eps_min[opt_indices_tensor]
    hi = eps_max[opt_indices_tensor]

    init_values = _initial_raw_for_optimized_eps(
        init_optimized_raw_1=cfg.init_optimized_raw_1,
        init_optimized_raw_2=cfg.init_optimized_raw_2,
        target_pair=target_pair,
        eps_min=eps_min,
        eps_max=eps_max,
        opt_indices=opt_indices,
    )

    target_rad = frame_ds.traj_ds[cfg.traj_id]["target"].to(device=device, dtype=torch.float32)
    if target_rad.ndim != 2 or target_rad.shape[1] != 5 or not len(target_rad) or not torch.isfinite(target_rad).all():
        raise ValueError("Expected nonempty, finite [N, 5] target geometry")
    if target_rad.shape[0] > cfg.max_target_points:
        idx = torch.linspace(0, target_rad.shape[0] - 1, cfg.max_target_points, device=device).long()
        target_rad = target_rad.index_select(0, idx)
    target_eval_rad = target_rad.detach().cpu()

    def compute_loss(target_geometry, generated_geometry):
        target_geometry = target_geometry[..., -cfg.loss_dim:]
        generated_geometry = generated_geometry[..., -cfg.loss_dim:]
        if loss_type == "mmd":
            return mmd_target_to_generated(target_geometry, generated_geometry, bandwidth=cfg.mmd_bandwidth)
        return kde_kl_target_to_generated(
            target_geometry, generated_geometry, bandwidth=cfg.kde_bandwidth, kl_lambda=cfg.kl_lambda,
        )

    was_training = model.training
    requires_grad_state = [p.requires_grad for p in model.parameters()]

    plot_dir = cfg.plot_dir or "."
    if cfg.save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    def compute_basin_metrics(generated_geometry: torch.Tensor) -> Optional[Dict[str, object]]:
        if not cfg.eval_basin_metrics:
            return None
        return compare_basin_metrics(
            target_eval_rad,
            generated_geometry.detach().cpu(),
            half_width=cfg.basin_half_width,
            active_target_prob=cfg.active_target_prob,
            min_generated_active_prob=cfg.min_generated_active_prob,
            delta_f_prob_floor=cfg.delta_f_prob_floor,
            delta_f_error_cap=cfg.delta_f_error_cap,
        )

    def run_phase(start_pair: torch.Tensor, steps: int, lr: float, add_noise: bool, suffix: str = ""):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        phase_start = time.perf_counter()
        plot_snapshots = []
        eps_start = start_pair[opt_indices_tensor].clone().detach()
        if cfg.clamp_eps:
            eps_start = eps_start.clamp(lo, hi)
        eps_input = eps_start.clone().requires_grad_(True)
        eps_reparam = torch.nn.Sequential(
            torch.nn.Linear(len(opt_indices), 16),
            torch.nn.Linear(16, len(opt_indices)),
        ).to(device=device, dtype=eps_start.dtype)
        with torch.no_grad():
            # Preserve the requested starting condition with nonzero layer weights.
            eps_reparam[-1].bias.add_(eps_start - eps_reparam(eps_input))
        optimizer = torch.optim.Adam([eps_input, *eps_reparam.parameters()], lr=lr)
        history = {"step": [], "loss": [], "eps1": [], "eps2": [], "grad_eps1": [], "grad_eps2": []}
        best_loss = float("inf")
        best_step = None
        best_pair = None
        best_generated_rad = None
        best_metrics = None
        initial_pair_out = None
        initial_loss = None
        initial_generated_rad = None
        initial_metrics = None
        for step in range(steps + 1):
            optimizer.zero_grad(set_to_none=True)

            eps_opt = eps_reparam(eps_input)
            eps_opt.retain_grad()
            eps_pair_raw = target_pair.clone()
            eps_pair_raw[opt_indices_tensor] = eps_opt
            if fixed_indices:
                fixed_idx = fixed_indices[0]
                fixed_raw_value = float(target_pair[fixed_idx].item()) if cfg.fixed_eps_raw is None else float(cfg.fixed_eps_raw)
                eps_pair_raw[fixed_idx] = fixed_raw_value

            eps_cond = eps_pair_raw.view(1, 2).expand(cfg.num_samples, 2)
            generated = model.ddim_sample(
                eps_cond,
                shape=(cfg.num_samples, frame_ds.target_dim),
                sample_steps=cfg.sample_steps,
                eta=0.0,
                clamp_x0=False,
            )
            generated_rad = frame_ds.denormalize_target(generated, clamp=False)
            loss = compute_loss(target_rad, generated_rad)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite optimization loss at step {step}")

            if step < steps:
                loss.backward()
                grad_opt = eps_opt.grad.detach().clone() if eps_opt.grad is not None else None
            else:
                grad_opt = None

            grad_pair = [None, None]
            if grad_opt is not None:
                for j, i in enumerate(opt_indices):
                    grad_pair[i] = float(grad_opt[j].item())

            loss_value = float(loss.detach().item())
            eps1_value = float(eps_pair_raw[0].detach().item())
            eps2_value = float(eps_pair_raw[1].detach().item())
            need_print = cfg.print_every > 0 and (step == 0 or (step + 1) % cfg.print_every == 0 or step == steps)
            need_metrics = cfg.eval_basin_metrics and (step == 0 or loss_value < best_loss or need_print)
            current_metrics = compute_basin_metrics(generated_rad) if need_metrics else None

            history["step"].append(step)
            history["loss"].append(loss_value)
            history["eps1"].append(eps1_value)
            history["eps2"].append(eps2_value)
            history["grad_eps1"].append(grad_pair[0])
            history["grad_eps2"].append(grad_pair[1])

            if step == 0:
                initial_pair_out = (eps1_value, eps2_value)
                initial_loss = loss_value
                initial_generated_rad = generated_rad.detach().cpu().clone()
                initial_metrics = current_metrics

            if loss_value < best_loss:
                best_loss = loss_value
                best_step = step
                best_pair = (eps1_value, eps2_value)
                best_generated_rad = generated_rad.detach().cpu().clone()
                best_metrics = current_metrics if current_metrics is not None else compute_basin_metrics(generated_rad)

            if cfg.save_plots:
                plot_snapshots.append({
                    "step": step, "phase": int(bool(suffix)),
                    "eps": (eps1_value, eps2_value), "loss": loss_value,
                    "angles": generated_rad[:, -2:].detach().cpu().numpy().copy(),
                })

            if need_print:
                phase_msg = f"{suffix[1:]} " if suffix else ""
                metric_msg = "" if current_metrics is None else " | " + format_basin_metrics(current_metrics)
                print(
                    f"{phase_msg}step {step:4d} | {loss_label} {loss.item():.6f} | "
                    f"eps1={eps_pair_raw[0].item():.6f}, eps2={eps_pair_raw[1].item():.6f} | "
                    f"{_format_distance_means(target_rad, generated_rad)}"
                    f"{metric_msg}"
                )

            if step == steps:
                break

            optimizer.step()
            with torch.no_grad():
                eps_current = eps_reparam(eps_input)
                eps_next = eps_current.clone()
                if add_noise:
                    base_noise_std = cfg.phase0_eps_noise_std
                    if bool((start_pair >= cfg.high_high_eps_threshold).all()):
                        base_noise_std = cfg.phase0_high_high_eps_noise_std
                    noise_std = base_noise_std * (1.0 - step / max(steps, 1))
                    eps_next.add_(noise_std * torch.randn_like(eps_next))
                if cfg.clamp_eps:
                    eps_next.clamp_(lo, hi)
                # Shift the output in raw epsilon units, preserving both layer weights.
                eps_reparam[-1].bias.add_(eps_next - eps_current)

        best_kl = best_loss
        if loss_type == "mmd":
            with torch.no_grad():
                best_kl = float(kde_kl_target_to_generated(
                    target_rad[..., -cfg.loss_dim:], best_generated_rad.to(device)[..., -cfg.loss_dim:],
                    bandwidth=cfg.kde_bandwidth, kl_lambda=cfg.kl_lambda,
                ).item())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimization_seconds = time.perf_counter() - phase_start
        return {
            "optimization_seconds": optimization_seconds,
            "plot_snapshots": plot_snapshots,
            "history": history,
            "final_pair": (history["eps1"][-1], history["eps2"][-1]),
            "best_pair": best_pair,
            "best_step": best_step,
            "best_loss": best_loss,
            "best_kl": best_kl,
            "best_generated_rad": best_generated_rad,
            "best_metrics": best_metrics,
            "initial_pair": initial_pair_out,
            "initial_loss": initial_loss,
            "initial_generated_rad": initial_generated_rad,
            "initial_metrics": initial_metrics,
        }

    def save_phase_outputs(phase, suffix=""):
        from plotters import plot_fes_comparison

        path = Path(plot_dir) / f"best_traj_{cfg.traj_id}_phase{suffix or '0'}_fes.png"
        plot_fes_comparison(
            target_eval_rad, phase["best_generated_rad"], path,
            bins=cfg.plot_bins, dpi=cfg.plot_dpi, half_width=cfg.basin_half_width,
            title=f"{loss_label}: step {phase['best_step']}, loss={phase['best_loss']:.5g}, eps={phase['best_pair']}",
        )

    try:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            initial_pair = target_pair.clone()
            initial_pair[opt_indices_tensor] = init_values
            if fixed_indices:
                fixed_idx = fixed_indices[0]
                fixed_raw_value = float(target_pair[fixed_idx].item()) if cfg.fixed_eps_raw is None else float(cfg.fixed_eps_raw)
                initial_pair[fixed_idx] = fixed_raw_value
                fixed_msg = f", fixed {fixed_name}={fixed_raw_value:.6f}"
            else:
                fixed_msg = ""
            target_opt = ", ".join(f"{name}={target_pair[i].item():.6f}" for name, i in zip(opt_names, opt_indices))
            print(
                f"target: eps1={target_pair[0].item():.6f}, eps2={target_pair[1].item():.6f} | "
                f"optimizing raw {opt_name} toward target {target_opt}{fixed_msg} | "
                f"clamp_eps={cfg.clamp_eps}"
            )
            print(f"initial: eps1={initial_pair[0].item():.6f}, eps2={initial_pair[1].item():.6f}")

        torch.manual_seed(cfg.seed)
        phase0 = run_phase(initial_pair, cfg.steps, cfg.lr, True)
        final_pair = phase0["final_pair"]
        print(f"optimized: eps1={final_pair[0]:.6f}, eps2={final_pair[1]:.6f}")
        print(
            f"best {loss_label}: step={phase0['best_step']}, eps1={phase0['best_pair'][0]:.6f}, "
            f"eps2={phase0['best_pair'][1]:.6f}, {loss_label}={phase0['best_loss']:.6f}"
        )
        phase1 = None
        if cfg.always_refine or phase0["best_kl"] > cfg.refine_kl_threshold:
            print(f"refining from best phase-0 {loss_label}={phase0['best_loss']:.6f}, KL={phase0['best_kl']:.6f}")
            phase1 = run_phase(torch.as_tensor(phase0["best_pair"], device=device), max(1, cfg.steps), cfg.lr_refine, False, "_1")
            final_pair = phase1["final_pair"]
            print(f"refined: eps1={final_pair[0]:.6f}, eps2={final_pair[1]:.6f}")
            print(
                f"best {loss_label}_1: step={phase1['best_step']}, eps1={phase1['best_pair'][0]:.6f}, "
                f"eps2={phase1['best_pair'][1]:.6f}, {loss_label}={phase1['best_loss']:.6f}"
            )
        eval_phase = "phase0"
        eval_pair = phase0["best_pair"]
        eval_step = phase0["best_step"]
        eval_loss = phase0["best_loss"]
        eval_metrics = phase0["best_metrics"]
        if phase1 is not None and phase1["best_loss"] < eval_loss:
            eval_phase = "phase1"
            eval_pair = phase1["best_pair"]
            eval_step = phase1["best_step"]
            eval_loss = phase1["best_loss"]
            eval_metrics = phase1["best_metrics"]

        progress_path = None
        progress_plot = None
        if cfg.save_plots:
            save_phase_outputs(phase0)
            if phase1 is not None:
                save_phase_outputs(phase1, "1")
            if cfg.plot_loss_surface:
                from plotters import plot_epsilon_loss_surface

                plot_epsilon_loss_surface(
                    model, frame_ds, cfg, compute_loss,
                    initial_eps=phase0["initial_pair"], optimized_eps=eval_pair,
                )
            from plot_optimization_progress import plot_optimization_progress

            snapshots = phase0["plot_snapshots"]
            best_index = eval_step
            if eval_phase == "phase1":
                best_index += len(snapshots)
                snapshots = snapshots + phase1["plot_snapshots"]
            # Count evaluated conditions across phases, including refinement step 0.
            # Very early best steps have fewer than two distinct intermediates.
            indices = np.unique(np.rint(np.linspace(0, best_index, 4)).astype(int))
            selected = [snapshots[i] for i in indices]
            progress_path = os.path.join(plot_dir, f"traj_{cfg.traj_id}_optimization_progress.npz")
            md_eval_root = cfg.md_eval_root or "md_control"
            np.savez_compressed(
                progress_path, steps=indices,
                phase_steps=[s["step"] for s in selected],
                phases=[s["phase"] for s in selected],
                losses=[s["loss"] for s in selected],
                eps=[s["eps"] for s in selected],
                angles=np.stack([s["angles"] for s in selected]),
                target_angles=target_eval_rad[:, -2:].numpy(),
                target_eps=target_pair.detach().cpu().numpy(),
                loss_label=loss_label, loss_dim=cfg.loss_dim, loss_type=loss_type, bins=cfg.plot_bins,
                md_eval_root=os.path.abspath(md_eval_root),
            )
            progress_plot = str(plot_optimization_progress(
                progress_path, md_root=md_eval_root, dpi=cfg.plot_dpi,
            ))

        optimization_seconds = phase0["optimization_seconds"] + (
            0.0 if phase1 is None else phase1["optimization_seconds"]
        )
        print(f"optimization time: {optimization_seconds:.3f} s (trajectory {cfg.traj_id})")
        return {
            "optimization_seconds": optimization_seconds,
            "progress_path": progress_path,
            "progress_plot": progress_plot,
            "optimized": opt_name,
            "loss_type": loss_type,
            "loss_dim": cfg.loss_dim,
            "fixed": fixed_name,
            "clamp_eps": cfg.clamp_eps,
            "target_eps": (float(target_pair[0].item()), float(target_pair[1].item())),
            "initial_eps": (phase0["history"]["eps1"][0], phase0["history"]["eps2"][0]),
            "initial_loss": phase0["initial_loss"],
            "initial_metrics": phase0["initial_metrics"],
            "optimized_eps": final_pair,
            "best_eps": phase0["best_pair"],
            "best_step": phase0["best_step"],
            "best_loss": phase0["best_loss"],
            "best_metrics": phase0["best_metrics"],
            "eval_phase": eval_phase,
            "eval_eps": eval_pair,
            "eval_step": eval_step,
            "eval_loss": eval_loss,
            "optimization_kl": (phase0 if eval_phase == "phase0" else phase1)["best_kl"],
            "eval_metrics": eval_metrics,
            "eval_target_rad": target_eval_rad,
            "eval_generated_rad": (phase0 if eval_phase == "phase0" else phase1)["best_generated_rad"],
            "history": phase0["history"],
            "optimized_eps_1": None if phase1 is None else phase1["final_pair"],
            "best_eps_1": None if phase1 is None else phase1["best_pair"],
            "best_step_1": None if phase1 is None else phase1["best_step"],
            "best_loss_1": None if phase1 is None else phase1["best_loss"],
            "best_metrics_1": None if phase1 is None else phase1["best_metrics"],
            "history_1": None if phase1 is None else phase1["history"],
        }
    finally:
        for p, req in zip(model.parameters(), requires_grad_state):
            p.requires_grad_(req)
        model.train(was_training)
