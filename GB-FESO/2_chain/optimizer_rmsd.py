"""Optimize five chain-1 epsilons against joint RMSD samples using MMD squared."""

import argparse
import copy
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cv_maker import (
    CHAIN_SYMMETRY_PERMUTATIONS, RMSD_REFERENCE_DISTANCES, RMSD_REFERENCE_NAMES,
    rmsd_cvs,
)
from dataset import LammpsDistanceDataset
from loss import mmd_loss
from model import FiLMDDPM
from plot_fes import save_fes_comparison
from reparameterized_condition import ReparameterizedCondition
from sampler import DifferentiableDDIMSampler, symmetric_noise


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset_path", type=Path, help="Folder containing numbered target trajectories")
    parser.add_argument("--checkpoint", type=Path, default=Path("ddpm.pt"), help="Trained DDPM checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("optimization_rmsd"), help="Results folder")
    parser.add_argument("--trajectory-ids", type=int, nargs="+", help="Target IDs; omit for all trajectories")
    parser.add_argument("--initial-epsilons", type=float, nargs=5, help="Five fixed starting epsilons; omit to draw from the list")
    parser.add_argument("--epsilon-list", type=Path, default=Path("list.txt"), help="Allowed random starting values")
    parser.add_argument("--iterations", type=int, default=250, help="Phase-1 updates; phase 2 gets half as many")
    parser.add_argument("--learning-rate", type=float, default=0.08, help="Initial Adam learning rate")
    parser.add_argument("--generated-frames", type=int, default=1024, help="Fixed optimization ensemble size")
    parser.add_argument("--target-frames", type=int, default=2048, help="Maximum evenly spaced retained target frames")
    parser.add_argument("--evaluation-frames", type=int, default=2048, help="Independent generated ensemble size")
    parser.add_argument("--ddim-steps", type=int, default=50, help="Deterministic DDIM steps")
    parser.add_argument("--guidance-scale", type=float, default=0.99, help="Classifier-free guidance scale")
    parser.add_argument("--mmd-bandwidth", type=float, default=0.32, help="Gaussian sigma in RMSD length units")
    parser.add_argument("--seed", type=int, default=0, help="Sampling and optimization-noise seed; random starts use fresh entropy")
    parser.add_argument("--plot-every", type=int, default=50, help="Plot interval within each phase")
    parser.add_argument("--reparam-hidden-dim", type=int, default=16, help="Affine condition-map width")
    parser.add_argument("--phase1-lr-final-factor", type=float, default=0.1, help="Final phase-1 LR fraction")
    parser.add_argument("--phase1-eps-noise-std", type=float, default=0.05, help="Initial physical-epsilon noise sigma")
    parser.add_argument("--phase1-high-eps-noise-std", type=float, default=0.005, help="Noise sigma when all starts exceed the threshold")
    parser.add_argument("--high-eps-threshold", type=float, default=0.9, help="Threshold for the smaller exploration noise")
    parser.add_argument("--refinement-lr-factor", type=float, default=0.1, help="Phase-2 LR relative to the initial LR")
    parser.add_argument("--refinement-loss-threshold", type=float, default=0.1, help="Run phase 2 when best MMD squared exceeds this")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Condition-map gradient clipping norm")
    parser.add_argument("--max-update-retries", type=int, default=12, help="Retries after a nonfinite update")
    parser.add_argument("--lr-backoff", type=float, default=0.5, help="LR and exploration-noise multiplier on retry")
    args = parser.parse_args(arguments)
    if any(isinstance(value, float) and not np.isfinite(value) for value in vars(args).values()):
        parser.error("Numeric parameters must be finite.")
    if args.iterations < 0 or args.max_update_retries < 0:
        parser.error("Iteration and retry counts must be nonnegative.")
    if min(args.generated_frames, args.target_frames, args.evaluation_frames,
           args.plot_every, args.ddim_steps, args.reparam_hidden_dim) < 1:
        parser.error("Frame counts, plot interval, DDIM steps, and hidden width must be positive.")
    if min(args.learning_rate, args.mmd_bandwidth, args.max_grad_norm) <= 0:
        parser.error("Learning rate, bandwidth, and gradient limit must be positive.")
    if not 0 < args.phase1_lr_final_factor <= 1:
        parser.error("Phase-1 final LR factor must be in (0, 1].")
    if not 0 < args.refinement_lr_factor < 1 or not 0 < args.lr_backoff < 1:
        parser.error("Refinement LR and retry factors must be in (0, 1).")
    if min(args.phase1_eps_noise_std, args.phase1_high_eps_noise_std,
           args.refinement_loss_threshold) < 0:
        parser.error("Noise scales and refinement threshold must be nonnegative.")
    return args


class RMSDOptimization:
    """Bounded Adam with decaying exploration and rollback of nonfinite updates."""

    def __init__(self, sampler, target, fixed_epsilons, initial_epsilons, bounds,
                 latent, config, *, learning_rate, iterations, explore=True):
        self.config = config
        self.sampler, self.target = sampler, target
        self.fixed_epsilons, self.latent = fixed_epsilons, latent
        self.ddim_steps, self.bandwidth = config.ddim_steps, config.mmd_bandwidth
        self.guidance_scale = config.guidance_scale
        self.condition_map = ReparameterizedCondition(
            initial_epsilons, bounds[:5, 0], bounds[:5, 1], hidden_dim=config.reparam_hidden_dim,
        )
        self.initial_learning_rate = learning_rate
        self.iterations = iterations
        self.lr_final_factor = config.phase1_lr_final_factor if explore else 1.0
        self.initial_noise_std = 0.0
        if explore:
            self.initial_noise_std = (config.phase1_high_eps_noise_std
                                      if bool((initial_epsilons >= config.high_eps_threshold).all())
                                      else config.phase1_eps_noise_std)
        if (self.iterations < 0 or not 0 < self.lr_final_factor <= 1
                or not np.isfinite(self.initial_noise_std) or self.initial_noise_std < 0):
            raise ValueError("Invalid iteration budget, LR decay factor, or epsilon noise scale.")
        self.noise_generator = torch.Generator(device=initial_epsilons.device)
        self.noise_generator.manual_seed(config.seed)
        self.noise_std = 0.0  # Noise used to reach the current state; none at initialization.
        self.optimizer = torch.optim.Adam(self.condition_map.parameters(), lr=learning_rate)
        self.iteration = 0
        self.retries = 0
        self.nonfinite_retries = 0
        self.current = self.evaluate()

    def evaluate(self):
        self.optimizer.zero_grad(set_to_none=True)
        free_epsilons = self.condition_map()
        free_epsilons.retain_grad()
        epsilons = torch.cat((free_epsilons, self.fixed_epsilons))
        if (not torch.isfinite(epsilons).all()
                or any(not torch.isfinite(p).all() for p in self.condition_map.parameters())):
            raise FloatingPointError("Nonfinite condition-map parameters or epsilons.")
        matrices = self.sampler.sample(
            epsilons=epsilons, x_T=self.latent, num_steps=self.ddim_steps,
            guidance_scale=self.guidance_scale, checkpoint_steps=True,
        )
        if not torch.isfinite(matrices).all():
            raise FloatingPointError("Nonfinite generated distances.")
        generated = rmsd_cvs(matrices)
        loss = mmd_loss(generated, self.target, bandwidth=self.bandwidth)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite MMD loss.")
        loss.backward()
        parameters = list(self.condition_map.parameters())
        if (free_epsilons.grad is None or not torch.isfinite(free_epsilons.grad).all()
                or any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters)):
            raise FloatingPointError("Missing or nonfinite optimization gradients.")
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite condition-map gradient norm.")
        return (epsilons.detach(), generated.detach(), loss.item(),
                free_epsilons.grad.detach().clone())

    def step(self):
        parameters = list(self.condition_map.parameters())
        saved_parameters = copy.deepcopy(self.condition_map.state_dict())
        saved_optimizer = copy.deepcopy(self.optimizer.state_dict())
        saved_gradients = [p.grad.detach().clone() for p in parameters]
        saved_rng = self.noise_generator.get_state()
        progress = min(self.iteration / max(self.iterations - 1, 1), 1.0)
        learning_rate = self.initial_learning_rate * (1 - (1 - self.lr_final_factor) * progress)
        noise_std = self.initial_noise_std * (1 - progress)
        noise = torch.zeros_like(self.condition_map.x)
        if noise_std > 0:
            noise = torch.randn(noise.shape, device=noise.device, dtype=noise.dtype,
                                generator=self.noise_generator)
        error_message = ""
        for attempt in range(self.config.max_update_retries + 1):
            self.condition_map.load_state_dict(saved_parameters)
            # Adam mutates moment tensors in place; do not alias the saved snapshot.
            self.optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
            for parameter, gradient in zip(parameters, saved_gradients):
                parameter.grad = gradient.clone()
            self.optimizer.param_groups[0]["lr"] = learning_rate
            self.optimizer.step()
            with torch.no_grad():
                raw = self.condition_map.layers(self.condition_map.x)
                bounded = (raw + noise_std * noise).clamp(self.condition_map.lower,
                                                         self.condition_map.upper)
                self.condition_map.layers[-1].bias.add_(bounded - raw)
            try:
                current = self.evaluate()
            except FloatingPointError as error:
                error_message = str(error)
                self.nonfinite_retries += 1
            else:
                self.current = current
                self.noise_std = noise_std
                self.iteration += 1
                return
            self.retries += 1
            if attempt < self.config.max_update_retries:
                learning_rate *= self.config.lr_backoff
                noise_std *= self.config.lr_backoff
                print(f"Step {self.iteration + 1}: {error_message} "
                      f"Retrying from the last valid state with lr={learning_rate:g}, "
                      f"epsilon noise std={noise_std:g}.",
                      flush=True)
        self.condition_map.load_state_dict(saved_parameters)
        self.optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
        self.noise_generator.set_state(saved_rng)
        for parameter, gradient in zip(parameters, saved_gradients):
            parameter.grad = gradient
        raise FloatingPointError(f"Step {self.iteration + 1} failed after retries: {error_message}")


def load_target_cvs(dataset_path, trajectory_id, frame_count):
    """Return target CVs and ten labels; labels 6..10 fix the second chain."""
    dataset = LammpsDistanceDataset(
        dataset_path, trajectory_id=trajectory_id,
    )
    indices = np.linspace(0, len(dataset) - 1, min(frame_count, len(dataset)),
                          dtype=int)
    matrices = torch.stack([dataset[int(index)]["distances"] for index in indices])
    metadata = dataset.trajectories[0]
    return rmsd_cvs(matrices), metadata["epsilons"]


def main(args):
    trajectory_ids = args.trajectory_ids
    if trajectory_ids is None:
        trajectory_ids = sorted(
            int(path.name) for path in args.dataset_path.iterdir()
            if path.is_dir() and path.name.isdigit()
            and (path / "in.in").is_file() and (path / "dump.lammpstrj").is_file()
        )
    if not trajectory_ids:
        raise FileNotFoundError(f"No numbered trajectories found under {args.dataset_path}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if (saved.get("condition_atom_ids") != list(range(1, 11))
            or saved.get("spatial_dimensions") != 2):
        raise ValueError("Use an alleps 2D checkpoint with ten epsilons in atom-ID order.")
    model = FiLMDDPM().to(device)
    model.load_state_dict(saved["model"])
    model.eval().requires_grad_(False)
    sampler = DifferentiableDDIMSampler(
        model,
        diffusion_steps=saved["diffusion_steps"],
        beta_start=saved["beta_start"],
        beta_end=saved["beta_end"],
    )
    if args.ddim_steps > sampler.diffusion_steps:
        raise ValueError("--ddim-steps cannot exceed the checkpoint's diffusion steps.")

    bounds = torch.tensor(
        saved["dataset_ranges"]["epsilons"],
        device=device, dtype=sampler.dtype,
    )
    if (bounds.shape != (10, 2) or not torch.isfinite(bounds).all()
            or not (bounds[:, 0] > 0).all()
            or not (bounds[:, 0] <= bounds[:, 1]).all()):
        raise ValueError("Checkpoint epsilon ranges must be ten positive [lower, upper] pairs.")
    free_bounds = bounds[:5]
    if args.initial_epsilons is None:
        rng = np.random.default_rng()  # Fresh entropy, independent of args.seed.
        values = torch.as_tensor(np.loadtxt(args.epsilon_list, ndmin=1), device=device, dtype=sampler.dtype)
        if (values.ndim != 1 or values.numel() == 0 or not torch.isfinite(values).all()
                or not (values > 0).all()):
            raise ValueError("--epsilon-list must contain finite positive values.")
        initial_choices = [values[(values >= low) & (values <= high)] for low, high in free_bounds]
        if any(len(choices) == 0 for choices in initial_choices):
            raise ValueError("--epsilon-list needs an allowed starting value for every optimized bead.")
    else:
        initial_epsilons = torch.tensor(args.initial_epsilons, device=device, dtype=sampler.dtype)
        if (initial_epsilons.shape != (5,) or not torch.isfinite(initial_epsilons).all()
                or not ((initial_epsilons >= free_bounds[:, 0])
                        & (initial_epsilons <= free_bounds[:, 1])).all()):
            raise ValueError("--initial-epsilons needs five finite values within the checkpoint ranges.")

    for trajectory_id in trajectory_ids:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        run_started = time.perf_counter()
        if args.initial_epsilons is None:
            initial_epsilons = torch.stack([choices[rng.integers(len(choices))]
                                            for choices in initial_choices])
        torch.manual_seed(args.seed)
        target, target_epsilons = load_target_cvs(args.dataset_path, trajectory_id, args.target_frames)
        target = target.to(device=device, dtype=sampler.dtype).detach()
        fixed_epsilons = torch.tensor(target_epsilons[5:], device=device, dtype=sampler.dtype)
        if (fixed_epsilons.shape != (5,) or not torch.isfinite(fixed_epsilons).all()
                or not ((fixed_epsilons >= bounds[5:, 0])
                        & (fixed_epsilons <= bounds[5:, 1])).all()):
            raise ValueError("The target's fixed chain 2 must lie inside the checkpoint's ranges.")
        latent = symmetric_noise(args.generated_frames, device, sampler.dtype)
        best_loss = float("inf")
        output_dir = args.output_dir / f"target_{trajectory_id:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "status.txt").write_text("Running\n")
        for name in ("result.pt", "evaluation_failed.pt"):
            (output_dir / name).unlink(missing_ok=True)
        print(f"Optimizing target {trajectory_id:04d} on {device}; "
              f"chain 1 bounds={free_bounds.tolist()}; "
              f"fixed chain 2={fixed_epsilons.tolist()}", flush=True)
        print(f"Adam on a 5 -> {args.reparam_hidden_dim} -> 5 condition map; "
              "finite uphill moves allowed. Phase 1 decays LR and epsilon noise; "
              f"gradient norm <= {args.max_grad_norm:g}; nonfinite updates retry "
              f"up to {args.max_update_retries} times.", flush=True)
        phases = []
        iterations_completed = 0
        with (output_dir / "history.csv").open("w", newline="") as history:
            writer = csv.writer(history)
            writer.writerow(("iteration", *(f"eps{i}" for i in range(1, 11)), "mmd2",
                             *(f"grad_eps{i}" for i in range(1, 6)), "learning_rate", "retries",
                             "phase", "phase_iteration", "epsilon_noise_std"))
            for phase, budget in enumerate((args.iterations, args.iterations // 2), start=1):
                if phase == 2 and budget == 0:
                    break
                if phase == 2 and best_loss <= args.refinement_loss_threshold:
                    print(f"Skipping phase 2: best MMD²={best_loss:.6g} <= "
                          f"threshold={args.refinement_loss_threshold:g}.", flush=True)
                    break
                starting_epsilons = initial_epsilons if phase == 1 else best_epsilons[:5].clone()
                learning_rate = args.learning_rate * (1.0 if phase == 1 else args.refinement_lr_factor)
                phase_result = dict(
                    phase=phase, requested_iterations=budget, iterations_completed=0,
                    initial_epsilons=torch.cat((starting_epsilons, fixed_epsilons)).cpu(),
                    initial_learning_rate=learning_rate, final_learning_rate=learning_rate,
                    rejected_updates=0, nonfinite_retries=0, stop_reason="iteration_limit",
                )
                phases.append(phase_result)
                print(f"Phase {phase}: {budget} updates, initial lr={learning_rate:g}, "
                      f"chain 1={starting_epsilons.tolist()}", flush=True)
                try:
                    run = RMSDOptimization(
                        sampler, target, fixed_epsilons, starting_epsilons, bounds, latent,
                        args, learning_rate=learning_rate, iterations=budget, explore=(phase == 1),
                    )
                except FloatingPointError as error:
                    phase_result["stop_reason"] = f"Initialization failed: {error}"
                    print(f"Target {trajectory_id:04d}, phase {phase}: {error}", flush=True)
                    break
                phase_result.update(initial_epsilon_noise_std=run.initial_noise_std,
                                    lr_final_factor=run.lr_final_factor)
                for phase_iteration in range(budget + 1):
                    iteration = iterations_completed + phase_iteration
                    epsilons, generated, value, gradient = run.current
                    values = epsilons.cpu().tolist()
                    gradients = gradient.cpu().tolist()
                    writer.writerow((iteration, *values, value, *gradients,
                                     run.optimizer.param_groups[0]["lr"], run.retries,
                                     phase, phase_iteration, run.noise_std))
                    history.flush()
                    print(f"Phase {phase} {phase_iteration:04d}  chain 1={values[:5]}  "
                          f"MMD²={value:.6g}  grad={gradients}  "
                          f"lr={run.optimizer.param_groups[0]['lr']:.6g}  "
                          f"noise_std={run.noise_std:.6g}  rejected={run.retries}",
                          flush=True)
                    if value < best_loss:
                        best_loss, best_iteration = value, iteration
                        best_phase, best_phase_iteration = phase, phase_iteration
                        best_epsilons = epsilons.detach().clone()

                    if phase_iteration % args.plot_every == 0 or phase_iteration == budget:
                        filename = (f"fes_{phase_iteration:04d}.png" if phase == 1
                                    else f"fes_phase2_{phase_iteration:04d}.png")
                        save_fes_comparison(
                            target, generated, output_dir / filename,
                            f"Phase {phase}, step {phase_iteration}",
                            bandwidth=args.mmd_bandwidth,
                            target_epsilons=target_epsilons, generated_epsilons=values,
                        )
                    if phase_iteration < budget:
                        try:
                            run.step()
                        except FloatingPointError as error:
                            phase_result["stop_reason"] = str(error)
                            print(f"Target {trajectory_id:04d}, phase {phase}: {error} "
                                  "Keeping the best valid condition.", flush=True)
                            break
                phase_result.update(
                    iterations_completed=run.iteration,
                    final_learning_rate=run.optimizer.param_groups[0]["lr"],
                    rejected_updates=run.retries, nonfinite_retries=run.nonfinite_retries,
                )
                iterations_completed += run.iteration
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        execution_seconds = time.perf_counter() - run_started
        stop_reason = "; ".join(
            f"Phase {p['phase']}: {p['stop_reason']}" for p in phases
            if p["stop_reason"] != "iteration_limit"
        ) or "iteration_limit"
        (output_dir / "status.txt").write_text(stop_reason + "\n")
        with (args.output_dir / "optimization_summary.csv").open("a", newline="") as summary:
            writer = csv.writer(summary)
            if summary.tell() == 0:
                writer.writerow(("trajectory_id", "initial_epsilons", "final_epsilons",
                                 "optimization_mmd2", "execution_time_seconds", "status"))
            writer.writerow((
                trajectory_id, json.dumps(torch.cat((initial_epsilons, fixed_epsilons)).cpu().tolist()),
                json.dumps(best_epsilons.cpu().tolist()) if np.isfinite(best_loss) else "",
                best_loss if np.isfinite(best_loss) else "",
                execution_seconds, stop_reason,
            ))
        if best_loss == float("inf"):
            print(f"Skipping target {trajectory_id:04d}: no valid optimization state.", flush=True)
            continue

        # Evaluate with fresh latents, independently of the optimization ensemble.
        with torch.no_grad():
            evaluation = []
            for start in range(0, args.evaluation_frames, args.generated_frames):
                matrices = sampler.sample(
                    epsilons=best_epsilons,
                    batch_size=min(args.generated_frames, args.evaluation_frames - start),
                    num_steps=args.ddim_steps, guidance_scale=args.guidance_scale,
                )
                if not torch.isfinite(matrices).all():
                    break
                evaluation.append(rmsd_cvs(matrices))
            if (not evaluation or sum(len(batch) for batch in evaluation) != args.evaluation_frames
                    or any(not torch.isfinite(batch).all() for batch in evaluation)):
                torch.save({"epsilons": best_epsilons.cpu(), "best_iteration": best_iteration,
                            "best_phase": best_phase, "best_phase_iteration": best_phase_iteration,
                            "optimization_mmd2": best_loss, "stop_reason": stop_reason,
                            "mmd_bandwidth": args.mmd_bandwidth,
                            "refinement_loss_threshold": args.refinement_loss_threshold,
                            "phases": phases},
                           output_dir / "evaluation_failed.pt")
                (output_dir / "status.txt").write_text(
                    "Independent evaluation failed; best condition is in evaluation_failed.pt.\n"
                )
                print(f"Target {trajectory_id:04d}: nonfinite independent evaluation; "
                      "saved best epsilons and continuing to the next target.", flush=True)
                continue
            evaluation = torch.cat(evaluation)
            evaluation_loss = save_fes_comparison(
                target, evaluation, output_dir / "fes_best.png",
                "Independent ensemble", bandwidth=args.mmd_bandwidth,
                target_epsilons=target_epsilons, generated_epsilons=best_epsilons.tolist(),
            )

        # Reverse only the displayed best chain; saved values retain atom-ID order.
        figure, axes = plt.subplots(
            1, 2, figsize=(12, 4.5), sharey=True, layout="constrained",
        )
        beads = np.arange(1, 6)
        best_chain = best_epsilons[:5].cpu().tolist()
        for axis, best_values, title in (
            (axes[0], best_chain, "Best: normal order"),
            (axes[1], best_chain[::-1], "Best: reversed order"),
        ):
            for values, label, style in (
                (initial_epsilons.cpu().tolist(), "Initial", "o--"),
                (best_values, "Best", "s-"),
                (target_epsilons[:5], "Target", "^-"),
            ):
                axis.plot(beads, values, style, label=label)
            axis.set_xticks(beads)
            axis.set_xlabel("Optimized bead position")
            axis.set_ylabel(r"$\epsilon$ (LJ units)")
            axis.set_title(title)
            axis.legend()
        figure.suptitle(f"Trajectory {trajectory_id:04d}: chain 1 epsilons")
        figure.savefig(output_dir / "epsilon_comparison.png", dpi=180)
        plt.close(figure)

        torch.save({
            "epsilons": best_epsilons.cpu(),
            "best_iteration": best_iteration,
            "best_phase": best_phase,
            "best_phase_iteration": best_phase_iteration,
            "optimization_mmd2": best_loss,
            "evaluation_mmd2": evaluation_loss,
            "target_cvs": target.cpu(),
            "generated_cvs": evaluation.cpu(),
            "target_trajectory_id": trajectory_id,
            "target_epsilons": target_epsilons,
            "initial_epsilons": torch.cat((initial_epsilons, fixed_epsilons)).cpu(),
            "fixed_chain_epsilons": fixed_epsilons.cpu(),
            "optimized_atom_ids": list(range(1, 6)),
            "parameterization": "affine_reparameterized_epsilons",
            "optimizer": "Adam",
            "reparam_hidden_dim": args.reparam_hidden_dim,
            "phase1_lr_final_factor": args.phase1_lr_final_factor,
            "phase1_eps_noise_std": args.phase1_eps_noise_std,
            "phase1_high_eps_noise_std": args.phase1_high_eps_noise_std,
            "high_eps_threshold": args.high_eps_threshold,
            "lr_backoff": args.lr_backoff,
            "epsilon_bounds": bounds.cpu(),
            "ddim_steps": args.ddim_steps,
            "guidance_scale": args.guidance_scale,
            "mmd_bandwidth": args.mmd_bandwidth,
            "seed": args.seed,
            "stop_reason": stop_reason,
            "phases": phases,
            "refinement_lr_factor": args.refinement_lr_factor,
            "refinement_loss_threshold": args.refinement_loss_threshold,
            "iterations_completed": iterations_completed,
            "final_learning_rate": phases[-1]["final_learning_rate"],
            "rejected_updates": sum(p["rejected_updates"] for p in phases),
            "nonfinite_retries": sum(p["nonfinite_retries"] for p in phases),
            "max_grad_norm": args.max_grad_norm,
            "cv_type": "chain_reversal_distance_rmsd",
            "reference_names": RMSD_REFERENCE_NAMES,
            "reference_distances": torch.from_numpy(RMSD_REFERENCE_DISTANCES.copy()),
            "cv_permutations": torch.from_numpy(CHAIN_SYMMETRY_PERMUTATIONS.copy()),
        }, output_dir / "result.pt")
        print(f"Best epsilons={best_epsilons.tolist()} in phase {best_phase}, "
              f"step {best_phase_iteration}; "
              f"optimization/independent MMD²={best_loss:.6g}/{evaluation_loss:.6g}\n"
              f"Saved history, FES plots, epsilon comparison, and result.pt to {output_dir}",
              flush=True)


if __name__ == "__main__":
    main(parse_arguments())
