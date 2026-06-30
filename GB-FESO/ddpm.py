import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch

from dataset import LammpsTrajectoryAngleDataset, AngleFrameDataset
from model import FiLMResMLP, AngleDiffusion
from loss import kde_kl_target_to_generated
from train import TrainConfig, make_dataloader, build_model,save_checkpoint, load_checkpoint,train
from sampler import wrap_to_pi, _eps_pair_tensor
from utils import compare_basin_metrics, format_basin_metrics

EpsPairLike = Union[Sequence[float], torch.Tensor]


# -----------------------------
# Training config and helpers
# -----------------------------
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
    plot_steps: Optional[Sequence[int]] = None  # optimization steps to save heatmaps for
    plot_dir: Optional[str] = None              # None => current working directory
    plot_bins: int = 80
    plot_dpi: int = 180
    plot_loss_surface: bool = False            # scan raw eps1/eps2 grid using the same KL loss
    loss_surface_grid_size: int = 25
    loss_surface_num_samples: int = 2048
    loss_surface_max_target_points: int = 2048
    phase0_eps_noise_std: float = 0.05
    phase0_high_high_eps_noise_std: float = 0.005
    high_high_eps_threshold: float = 0.9
    clamp_eps: bool = False                  # False => raw eps can leave training range; True => clamp to observed train range
    eval_basin_metrics: bool = True
    basin_half_width: Optional[float] = None
    active_target_prob: float = 0.05
    min_generated_active_prob: float = 0.005
    delta_f_prob_floor: float = 1e-3
    delta_f_error_cap: Optional[float] = 6.0



# -----------------------------
# Optimize eps1, eps2, or both
# -----------------------------

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
    frame_ds: AngleFrameDataset,
    cfg: EpsilonOptConfig,
) -> Dict[str, object]:
    """
    Optimize raw epsilon condition(s) by backpropagating through all DDIM sampler steps.

    cfg.optimize_eps chooses what is optimized:
      - "eps1": optimize pair_coeff 1 2, keep eps2 fixed
      - "eps2": optimize pair_coeff 2 4, keep eps1 fixed
      - "both": optimize eps1 and eps2 together

    For single-epsilon optimization, fixed_eps_raw defaults to the target trajectory value.
    For both-epsilon optimization, fixed_eps_raw is ignored.

    init_optimized_raw can be:
      - None: midpoint of each observed epsilon range
      - scalar: same initial value for each optimized epsilon
      - [eps1, eps2]: initial pair; optimized components are taken from it

    The optimized values are raw LJ epsilons. By default they are passed to the
    model directly without normalization or bounds. Set cfg.clamp_eps=True to
    clamp optimized epsilons to the observed training range after each update.
    """
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
    if target_rad.shape[0] > cfg.max_target_points:
        idx = torch.linspace(0, target_rad.shape[0] - 1, cfg.max_target_points, device=device).long()
        target_rad = target_rad.index_select(0, idx)
    target_eval_rad = target_rad.detach().cpu()

    was_training = model.training
    requires_grad_state = [p.requires_grad for p in model.parameters()]
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    plot_steps = set(int(s) for s in cfg.plot_steps) if cfg.plot_steps is not None else set()
    plot_dir = cfg.plot_dir or "."
    saved_heatmaps = []
    saved_best_heatmap = None
    saved_loss_surface = None
    saved_best_heatmap_1 = None
    saved_loss_surface_1 = None
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

    def run_phase(start_pair: torch.Tensor, steps: int, lr: float, add_noise: bool, suffix: str = "", save_steps: bool = False):
        eps_start = start_pair[opt_indices_tensor].clone().detach()
        if cfg.clamp_eps:
            eps_start = eps_start.clamp(lo, hi)
        eps_opt = eps_start.requires_grad_(True)
        optimizer = torch.optim.Adam([eps_opt], lr=lr)
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
            target_loss = target_rad[..., -2:]
            generated_loss = generated_rad[..., -2:]

            loss = kde_kl_target_to_generated(
                target_loss,
                generated_loss,
                bandwidth=cfg.kde_bandwidth,
                kl_lambda=cfg.kl_lambda,
            )

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

            if save_steps and step in plot_steps:
                from plotters import plot_target_vs_generated_heatmaps

                save_path = os.path.join(plot_dir, f"traj_{cfg.traj_id}_opt_step_{step:04d}_heatmap{suffix}.png")
                plot_target_vs_generated_heatmaps(
                    model=model,
                    frame_ds=frame_ds,
                    traj_id=cfg.traj_id,
                    eps_raw=(history["eps1"][-1], history["eps2"][-1]),
                    generated_rad=generated_rad.detach(),
                    bins=cfg.plot_bins,
                    save_path=save_path,
                    dpi=cfg.plot_dpi,
                    show=False,
                    close=True,
                    step=step,
                    kl=history["loss"][-1],
                )
                saved_heatmaps.append(save_path)
                print(f"saved heatmap: {save_path}")

            if need_print:
                phase_msg = f"{suffix[1:]} " if suffix else ""
                metric_msg = "" if current_metrics is None else " | " + format_basin_metrics(current_metrics)
                print(
                    f"{phase_msg}step {step:4d} | KL {loss.item():.6f} | "
                    f"eps1={eps_pair_raw[0].item():.6f}, eps2={eps_pair_raw[1].item():.6f} | "
                    f"{_format_distance_means(target_rad, generated_rad)}"
                    f"{metric_msg}"
                )

            if step == steps:
                break

            optimizer.step()
            with torch.no_grad():
                if add_noise:
                    base_noise_std = cfg.phase0_eps_noise_std
                    if bool((start_pair >= cfg.high_high_eps_threshold).all()):
                        base_noise_std = cfg.phase0_high_high_eps_noise_std
                    noise_std = base_noise_std * (1.0 - step / max(steps, 1))
                    eps_opt.add_(noise_std * torch.randn_like(eps_opt))
                if cfg.clamp_eps:
                    eps_opt.clamp_(lo, hi)

        return {
            "history": history,
            "final_pair": (history["eps1"][-1], history["eps2"][-1]),
            "best_pair": best_pair,
            "best_step": best_step,
            "best_loss": best_loss,
            "best_generated_rad": best_generated_rad,
            "best_metrics": best_metrics,
            "initial_pair": initial_pair_out,
            "initial_loss": initial_loss,
            "initial_generated_rad": initial_generated_rad,
            "initial_metrics": initial_metrics,
        }

    def save_phase_outputs(best_pair, best_step, best_loss, best_generated_rad, suffix: str = ""):
        from plotters import (
            plot_target_vs_generated_heatmaps,
            plot_epsilon_loss_surface,
            plot_target_vs_generated_distance_angle_fes,
        )

        best_heatmap = os.path.join(plot_dir, f"best_traj_{cfg.traj_id}_opt_step_{best_step:04d}_heatmap{suffix}.png")
        plot_target_vs_generated_heatmaps(
            model=model,
            frame_ds=frame_ds,
            traj_id=cfg.traj_id,
            eps_raw=best_pair,
            generated_rad=best_generated_rad,
            bins=cfg.plot_bins,
            save_path=best_heatmap,
            dpi=cfg.plot_dpi,
            show=False,
            close=True,
            step=best_step,
            kl=best_loss,
        )
        saved_heatmaps.append(best_heatmap)
        print(f"saved best heatmap: {best_heatmap}")

        best_distance_angle_fes = os.path.join(
            plot_dir,
            f"best_traj_{cfg.traj_id}_opt_step_{best_step:04d}_r12_theta123_fes{suffix}.png",
        )
        plot_target_vs_generated_distance_angle_fes(
            model=model,
            frame_ds=frame_ds,
            traj_id=cfg.traj_id,
            eps_raw=best_pair,
            generated_rad=best_generated_rad,
            bins=cfg.plot_bins,
            save_path=best_distance_angle_fes,
            dpi=cfg.plot_dpi,
            show=False,
            close=True,
            step=best_step,
            kl=best_loss,
        )
        saved_heatmaps.append(best_distance_angle_fes)
        print(f"saved best distance-angle FES: {best_distance_angle_fes}")

        loss_surface = None
        if cfg.plot_loss_surface:
            loss_surface = os.path.join(plot_dir, f"traj_{cfg.traj_id}_epsilon_loss_surface{suffix}.png")

            def loss_surface_fn(surface_target_rad, surface_generated_rad):
                return kde_kl_target_to_generated(
                    surface_target_rad[..., -2:],
                    surface_generated_rad[..., -2:],
                    bandwidth=cfg.kde_bandwidth,
                    kl_lambda=cfg.kl_lambda,
                )

            _fig, _ax, surface = plot_epsilon_loss_surface(
                model=model,
                frame_ds=frame_ds,
                traj_id=cfg.traj_id,
                loss_fn=loss_surface_fn,
                grid_size=cfg.loss_surface_grid_size,
                num_samples=cfg.loss_surface_num_samples,
                sample_steps=cfg.sample_steps,
                max_target_points=cfg.loss_surface_max_target_points,
                seed=cfg.seed,
                save_path=loss_surface,
                dpi=cfg.plot_dpi,
                show=False,
                close=True,
                initial_eps=(phase0["history"]["eps1"][0], phase0["history"]["eps2"][0]),
                optimized_eps=best_pair,
                title=f"trajectory {cfg.traj_id}: KL loss surface{suffix}",
            )
            print(
                f"saved loss surface: {loss_surface} | "
                f"best grid eps1={surface['best_eps'][0]:.6f}, "
                f"eps2={surface['best_eps'][1]:.6f}, "
                f"KL={surface['best_loss']:.6f}"
            )
        return best_heatmap, loss_surface

    try:
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

        from plotters import plot_target_vs_generated_free_energy

        saved_initial_fes = os.path.join(plot_dir, f"traj_{cfg.traj_id}_target_eps_fes_before_optimization.png")
        plot_target_vs_generated_free_energy(
            model=model,
            frame_ds=frame_ds,
            traj_id=cfg.traj_id,
            eps_raw=tuple(float(v) for v in target_pair.detach().cpu()),
            num_samples=cfg.num_samples,
            sample_steps=cfg.sample_steps,
            bins=cfg.plot_bins,
            seed=cfg.seed,
            save_path=saved_initial_fes,
            dpi=cfg.plot_dpi,
            show=False,
            close=True,
            free_energy_cap=5.0,
        )
        print(f"saved target-eps FES before optimization: {saved_initial_fes}")

        phase0 = run_phase(initial_pair, cfg.steps, cfg.lr, True, save_steps=True)
        final_pair = phase0["final_pair"]
        print(f"optimized: eps1={final_pair[0]:.6f}, eps2={final_pair[1]:.6f}")
        print(
            f"best KL: step={phase0['best_step']}, eps1={phase0['best_pair'][0]:.6f}, "
            f"eps2={phase0['best_pair'][1]:.6f}, KL={phase0['best_loss']:.6f}"
        )
        saved_best_heatmap, saved_loss_surface = save_phase_outputs(
            phase0["best_pair"], phase0["best_step"], phase0["best_loss"], phase0["best_generated_rad"]
        )

        phase1 = None
        if phase0["best_loss"] > 0.2:
            print(f"refining from best phase-0 KL={phase0['best_loss']:.6f}")
            phase1 = run_phase(torch.as_tensor(phase0["best_pair"], device=device), max(1, cfg.steps // 1), cfg.lr_refine, False, "_1")
            final_pair = phase1["final_pair"]
            print(f"refined: eps1={final_pair[0]:.6f}, eps2={final_pair[1]:.6f}")
            print(
                f"best KL_1: step={phase1['best_step']}, eps1={phase1['best_pair'][0]:.6f}, "
                f"eps2={phase1['best_pair'][1]:.6f}, KL={phase1['best_loss']:.6f}"
            )
            saved_best_heatmap_1, saved_loss_surface_1 = save_phase_outputs(
                phase1["best_pair"], phase1["best_step"], phase1["best_loss"], phase1["best_generated_rad"], "_1"
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

        return {
            "optimized": opt_name,
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
            "eval_metrics": eval_metrics,
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


if __name__ == "__main__":
    from plotters import plot_target_vs_generated_heatmaps

    cfg = TrainConfig(root_dir="/home/danman-ub/projects/lammps/SimpleChain/4bead_LJ_2eps_val")
    checkpoint_path = f"ddpm_best_epoch_{cfg.epochs:04d}.pt"
    retrain = False

    if retrain or not os.path.isfile(checkpoint_path):
        model, frame_ds = train(cfg)
        save_checkpoint(checkpoint_path, model, frame_ds, cfg)
        print(f"saved best epoch checkpoint: {checkpoint_path}")

    model, frame_ds, cfg = load_checkpoint(
        checkpoint_path=checkpoint_path,
        root_dir=cfg.root_dir,
        device=cfg.device,
    )
    print(f"loaded checkpoint without training: {checkpoint_path}")

    results_path = "./results_evaluation.csv"
    results_header = [
        "trajID",
        "start_id",
        "start_eps1",
        "start_eps2",
        "clamp_eps",
        "eps1_target",
        "eps2_target",
        "final_eps1",
        "final_eps2",
        "phase0_best_eps1",
        "phase0_best_eps2",
        "phase0_best_loss",
        "phase1_best_eps1",
        "phase1_best_eps2",
        "phase1_best_loss",
        "eval_phase",
        "target_type",
        "active_names",
        "missing_active_basin",
        "missing_names",
        "total_variation",
        "df_ref_name",
        "df_basin_names",
        "df_target",
        "df_generated",
        "df_abs_error",
        "df_mae",
        "df_median_abs_error",
        "dominant_basin_retention",
        "leakage",
    ]
    write_header = True
    if os.path.isfile(results_path) and os.path.getsize(results_path) > 0:
        with open(results_path, "r", newline="") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header == results_header:
            write_header = False
        else:
            backup_path = results_path + ".old"
            os.replace(results_path, backup_path)
            print(f"existing results CSV header differs; moved old file to {backup_path}")

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_header)
        if write_header:
            writer.writeheader()

    def metric_value(metrics, key, default=""):
        if metrics is None:
            return default
        value = metrics.get(key, default)
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return ";".join(str(v) for v in value)
        return value

    def eps_value(pair, idx, default=""):
        if pair is None:
            return default
        return pair[idx]

    START_EPS_PAIRS = [
      #  (0.25, 0.25),
        (0.25, 1.4),
       # (1.4, 0.25),
       # (1.4, 1.4),
    ]
    TARGET_TRAJ_IDS = range(4, 5)
    CLAMP_EPS_OPTIONS = [True]  # set to [False, True] to compare unclamped and clamped optimization
    OPTIMIZE_EPS = "both"  # use "eps1", "eps2", or "both"

    for clamp_eps in CLAMP_EPS_OPTIONS:
        clamp_label = "clamped" if clamp_eps else "unclamped"
        for start_id, (eps1_raw, eps2_raw) in enumerate(START_EPS_PAIRS):
            start_label = f"start_{start_id}_eps1_{eps1_raw:.2f}_eps2_{eps2_raw:.2f}".replace(".", "p")
            for trajj in TARGET_TRAJ_IDS:
                TARGET_TRAJ_ID = trajj
                plot_dir = f"traj_{TARGET_TRAJ_ID}_{start_label}_{clamp_label}_optimization_heatmaps"

                result = optimize_epsilon_condition(
                    model,
                    frame_ds,
                    EpsilonOptConfig(
                        traj_id=TARGET_TRAJ_ID,
                        optimize_eps=OPTIMIZE_EPS,
                        kde_bandwidth=0.5,
                        sample_steps=cfg.sample_steps,
                        init_optimized_raw_1=eps1_raw,
                        init_optimized_raw_2=eps2_raw,
                        fixed_eps_raw=0.5,
                        seed=cfg.seed,
                        steps=201,
                        lr=5e-2,
                        lr_refine=5e-2,
                        plot_steps=[0, 10, 50, 100],
                        plot_dir=plot_dir,
                        plot_loss_surface=True,
                        loss_surface_grid_size=25,
                        num_samples=3500,
                        loss_surface_num_samples=3500,
                        clamp_eps=clamp_eps,
                        kl_lambda=1.0,
                    ),
                )
                print(result["eval_eps"])

                eps1_target, eps2_target = result["target_eps"]
                final_eps1, final_eps2 = result["optimized_eps"]
                eval_eps1, eval_eps2 = result["eval_eps"]
                metrics = result["eval_metrics"]

                row = {
                    "trajID": TARGET_TRAJ_ID,
                    "start_id": start_id,
                    "start_eps1": eps1_raw,
                    "start_eps2": eps2_raw,
                    "clamp_eps": clamp_eps,
                    "eps1_target": eps1_target,
                    "eps2_target": eps2_target,
                    "final_eps1": final_eps1,
                    "final_eps2": final_eps2,
                    "phase0_best_eps1": eps_value(result["best_eps"], 0),
                    "phase0_best_eps2": eps_value(result["best_eps"], 1),
                    "phase0_best_loss": result["best_loss"],
                    "phase1_best_eps1": eps_value(result["best_eps_1"], 0),
                    "phase1_best_eps2": eps_value(result["best_eps_1"], 1),
                    "phase1_best_loss": result["best_loss_1"] or "",
                    "eval_phase": result["eval_phase"],
                    "target_type": metric_value(metrics, "target_type"),
                    "missing_active_basin": metric_value(metrics, "missing_active_basin"),
                    "df_ref_name": metric_value(metrics, "df_ref_name"),
                    "df_basin_names": metric_value(metrics, "df_basin_names"),
                    "df_target": metric_value(metrics, "df_target"),
                    "df_generated": metric_value(metrics, "df_generated"),
                    "df_abs_error": metric_value(metrics, "df_abs_error"),
                    "df_mae": metric_value(metrics, "df_mae"),
                    "df_median_abs_error": metric_value(metrics, "df_median_abs_error"),
                    "leakage": metric_value(metrics, "leakage"),
                }
                with open(results_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=results_header).writerow(row)

                plot_target_vs_generated_heatmaps(
                    model=model,
                    frame_ds=frame_ds,
                    traj_id=TARGET_TRAJ_ID,
                    eps_raw=result["eval_eps"],
                    num_samples=4000,
                    sample_steps=cfg.sample_steps,
                    bins=80,
                    seed=cfg.seed,
                    save_path=os.path.join(plot_dir, f"traj_{TARGET_TRAJ_ID}_target_vs_generated_heatmap.png"),
                )