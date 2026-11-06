"""Plot optimization/evaluation MMD histograms and A/C basin free-energy errors."""

import csv
from pathlib import Path
from pickle import UnpicklingError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch

from cv_maker import CHAIN_SYMMETRY_PERMUTATIONS, RMSD_REFERENCE_DISTANCES, RMSD_REFERENCE_NAMES
from loss import mmd_loss


RUNS_DIR = Path("optimization_rmsd")
OUTPUT_DIR = Path("analysis")
MMD_BIN_WIDTH = 0.002
HISTOGRAM_BINS = 20
# Each row gives [[x_min, x_max], [y_min, y_max]], in RMSD length units.
BASIN_BOUNDS = np.array([
    [[0.0, 0.7], [0.95, 1.65]],  # A: near (0, 1.4)
    [[1.05, 1.75], [0.65, 1.35]],  # C: near (1.4, 1)
])
MIN_BASIN_POPULATION = 0.10


def basin_populations(cvs):
    """Count samples in A/C, normalized by the entire ensemble size."""
    cvs = np.asarray(cvs, dtype=float)
    if (cvs.ndim != 2 or cvs.shape[1] != 2 or len(cvs) == 0
            or not np.isfinite(cvs).all() or np.any(cvs < 0)):
        raise ValueError("Expected finite, nonnegative RMSD samples shaped (N, 2)")
    inside = ((cvs[:, None, :] >= BASIN_BOUNDS[:, :, 0])
              & (cvs[:, None, :] <= BASIN_BOUNDS[:, :, 1]))
    return inside.all(axis=2).mean(axis=0)


def analyze_delta_g(folder, runs):
    """Compare saved best-condition basin free energies for completed runs."""
    pairs, targets, rows = [], [], []
    target_excluded = generated_excluded = 0
    print(f"\nBasin A bounds: {BASIN_BOUNDS[0].tolist()}")
    print(f"Basin C bounds: {BASIN_BOUNDS[1].tolist()}")
    print(f"Required population: {MIN_BASIN_POPULATION:.0%} in each basin, in both ensembles")
    print("Delta G = G_C - G_A; error = optimized - target; units: kBT")
    for run_dir, result in runs:
        target = np.asarray(result["target_cvs"], dtype=float)
        populations = np.array([basin_populations(target),
                                basin_populations(result["generated_cvs"])])
        targets.append(target)
        target_ok, generated_ok = (populations >= MIN_BASIN_POPULATION).all(axis=1)
        note = (f"{run_dir.name}: P(A,C) target={populations[0].round(4).tolist()}, "
                f"optimized={populations[1].round(4).tolist()}")
        row = dict(target_id=result["target_trajectory_id"],
                   target_p_a=populations[0, 0], target_p_c=populations[0, 1],
                   generated_p_a=populations[1, 0], generated_p_c=populations[1, 1],
                   included=bool(target_ok and generated_ok),
                   target_delta_g="", generated_delta_g="", error="")
        rows.append(row)
        if not target_ok:
            target_excluded += 1
            print(f"{note}; excluded: target basin below cutoff")
        elif not generated_ok:
            generated_excluded += 1
            print(f"{note}; excluded: optimized basin below cutoff")
        else:
            delta_g = -np.log(populations[:, 1] / populations[:, 0])
            pairs.append(delta_g)
            row.update(target_delta_g=delta_g[0], generated_delta_g=delta_g[1],
                       error=delta_g[1] - delta_g[0])
            print(f"{note}; Delta G target={delta_g[0]:.6g}, optimized={delta_g[1]:.6g}, "
                  f"error={delta_g[1] - delta_g[0]:+.6g}")

    print(f"Delta G comparisons: {len(pairs)}/{len(runs)} runs")
    print(f"Excluded: {target_excluded} target populations below cutoff; "
          f"{generated_excluded} optimized populations below cutoff despite a qualifying target")
    with (folder / "delta_g.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if pairs:
        delta_g = np.array(pairs)
        errors = delta_g[:, 1] - delta_g[:, 0]
        print(f"Average target Delta G: {delta_g[:, 0].mean():.6g} kBT")
        print(f"Average optimized Delta G: {delta_g[:, 1].mean():.6g} kBT")
        print(f"Mean signed Delta G error: {errors.mean():.6g} kBT")
        print(f"Mean absolute Delta G error: {np.abs(errors).mean():.6g} kBT")
        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 13,
                             "mathtext.fontset": "dejavusans"}):
            figure, axis = plt.subplots(figsize=(4, 5), layout="constrained")
            absolute_errors = np.abs(errors)
            if len(absolute_errors) > 1 and np.ptp(absolute_errors) > 0:
                violin = axis.violinplot(absolute_errors, positions=[1], widths=0.6,
                                         showmedians=True, showextrema=False,
                                         bw_method="scott", points=200)
                violin["bodies"][0].set(facecolor="#ff7f0e", edgecolor="#ff7f0e",
                                        alpha=0.65, linewidth=1.3)
                violin["cmedians"].set(color="#333333", linewidth=2)
            else:
                axis.hlines(np.median(absolute_errors), 0.85, 1.15,
                            color="#333333", linewidth=2)
            axis.set_xticks([1], ["MMD optimization"])
            axis.set_ylabel(r"$|\Delta G_{\mathrm{opt}}-\Delta G_{\mathrm{target}}|/(k_BT)$",
                            fontsize=15)
            axis.yaxis.set_label_position("right")
            axis.yaxis.tick_right()
            axis.set(xlim=(0.4, 1.6), ylim=(0, max(0.1, absolute_errors.max() * 1.05)))
            axis.tick_params(direction="out", length=4, width=1)
            output = folder / "delta_g_error_violin.png"
            figure.savefig(output, dpi=300)
            print(f"Saved Delta G error violin: {output}")
            plt.close(figure)
    else:
        (folder / "delta_g_error_violin.png").unlink(missing_ok=True)
        print("No runs meet the population cutoff; Delta G errors are undefined.")

    figure, axes = plt.subplots(1, 3, figsize=(17, 5), layout="constrained")
    if targets:
        pooled = np.concatenate(targets)
        extent = max(1.9, float(pooled.max()))
        heatmap = axes[0].hist2d(*pooled.T, bins=80, range=((0, extent), (0, extent)),
                                 norm=LogNorm())[3]
        figure.colorbar(heatmap, ax=axes[0], label="Target frames (pooled)", shrink=0.8)
    for bounds, name, color in zip(BASIN_BOUNDS, ("A", "C"), ("#e45756", "#f2cf5b")):
        axes[0].add_patch(Rectangle(bounds[:, 0], *np.diff(bounds, axis=1).ravel(),
                                    fill=False, edgecolor=color, linewidth=2))
        axes[0].text(bounds[0, 0] + 0.03, bounds[1, 1] - 0.12, name, color=color,
                     weight="bold", bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))
    axes[0].text(1.4, 0.1, "B", color="white",
                 bbox=dict(facecolor="black", alpha=0.6, edgecolor="none"))
    axes[0].set(xlim=(0, 1.9), ylim=(0, 1.9), xlabel="RMSD 1 (LJ length)",
                ylabel="RMSD 2 (LJ length)", title="Integration squares (target region view)")
    axes[0].set_aspect("equal")
    if pairs:
        bins = np.histogram_bin_edges(delta_g.ravel(), bins=HISTOGRAM_BINS)
        axes[1].hist([delta_g[:, 0], delta_g[:, 1]], bins=bins, label=["Target", "Optimized"],
                     color=["#4c78a8", "#f58518"], edgecolor="white")
        axes[1].legend()
        axes[2].hist(errors, bins=HISTOGRAM_BINS, color="#4c78a8", edgecolor="white")
    else:
        for axis in axes[1:]:
            axis.text(0.5, 0.5, "No qualifying runs", transform=axis.transAxes, ha="center")
    axes[1].set(xlabel=r"$\Delta G/(k_BT)$", title="Basin free-energy difference (C − A)")
    axes[2].set(xlabel=r"$(\Delta G_{optimized}-\Delta G_{target})/(k_BT)$", title="Signed error")
    for axis in axes[1:]:
        axis.set_ylabel("Number of runs")
        axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    output = folder / "delta_g_histogram.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(f"Saved Delta G histogram: {output}")


@torch.no_grad()
def main(folder=RUNS_DIR, output=OUTPUT_DIR):
    folder, output = Path(folder), Path(output)
    histories = sorted(folder.glob("target_*/history.csv"))
    rows, runs = [], []
    for history in histories:
        run_dir = history.parent
        try:
            status = run_dir / "status.txt"
            if status.exists() and status.read_text().strip().lower().startswith("running"):
                print(f"Skipping {run_dir.name}: still running.")
                continue
            with history.open(newline="") as stream:
                losses = np.array([float(row["mmd2"]) for row in csv.DictReader(stream)])
            if len(losses) == 0 or not np.isfinite(losses).all() or (losses < 0).any():
                raise ValueError("Empty or invalid MMD history.")
            result = torch.load(run_dir / "result.pt", map_location="cpu", weights_only=True)
            if (result["cv_type"] != "chain_reversal_distance_rmsd"
                    or tuple(result["reference_names"]) != RMSD_REFERENCE_NAMES
                    or not np.allclose(result["reference_distances"], RMSD_REFERENCE_DISTANCES)
                    or not np.array_equal(result["cv_permutations"], CHAIN_SYMMETRY_PERMUTATIONS)):
                raise ValueError("Saved RMSD definitions differ from cv_maker.py.")
            if not np.isclose(result["optimization_mmd2"], losses.min(), rtol=1e-5, atol=1e-8):
                raise ValueError("Saved best result does not match the current history.")
            target, generated = result["target_cvs"].double(), result["generated_cvs"].double()
            for samples in (target, generated):
                basin_populations(samples)
            bandwidth = float(result["mmd_bandwidth"])
            score = mmd_loss(generated, target, bandwidth=bandwidth).item()
            if not np.isfinite(score):
                raise ValueError("Nonfinite independent MMD score.")
        except (OSError, EOFError, UnpicklingError, RuntimeError, KeyError, TypeError, ValueError) as error:
            print(f"Skipping {run_dir.name}: {error}")
            continue
        rows.append(dict(target_id=result["target_trajectory_id"], mmd_bandwidth=bandwidth,
                         initial_mmd2=losses[0], optimization_mmd2=losses.min(),
                         evaluation_mmd2=score, target_n=len(target), generated_n=len(generated)))
        runs.append((run_dir, result))

    if not rows:
        raise ValueError(f"No completed RMSD results with matching histories under {folder}")
    if not np.allclose([row["mmd_bandwidth"] for row in rows], rows[0]["mmd_bandwidth"]):
        raise ValueError("Analyze runs with the same MMD bandwidth together.")
    sample_sizes = {(row["target_n"], row["generated_n"]) for row in rows}
    print(f"Runs analyzed: {len(rows)}; skipped: {len(histories) - len(rows)}")
    print(f"MMD bandwidth: {rows[0]['mmd_bandwidth']:g}; sample counts: {sorted(sample_sizes)}")
    if len(sample_sizes) > 1:
        print("Sample counts vary; biased MMD scores have different finite-sample bias and precision.")
    before = np.array([row["initial_mmd2"] for row in rows])
    after = np.array([row["optimization_mmd2"] for row in rows])
    scores = np.array([row["evaluation_mmd2"] for row in rows])
    print(f"Mean MMD²: initial={before.mean():.6g}, best={after.mean():.6g}, "
          f"independent={scores.mean():.6g}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "mmd_scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    bins = np.histogram_bin_edges(np.concatenate((before, after)), bins=HISTOGRAM_BINS)
    figure, axis = plt.subplots(figsize=(8, 5), layout="constrained")
    axis.hist([before, after], bins=bins, label=["Initial", "Best"],
              color=["#4c78a8", "#f58518"], edgecolor="white")
    axis.set(xlabel="Optimization MMD²", ylabel="Number of runs")
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))
    axis.legend()
    figure.savefig(output / "mmd_optimization_histogram.png", dpi=200)
    plt.close(figure)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 18, "axes.labelsize": 24}):
        figure, axis = plt.subplots(figsize=(10, 5.5), layout="constrained")
        upper = max(0.1 + MMD_BIN_WIDTH, scores.max())
        bins = np.arange(int(np.ceil(upper / MMD_BIN_WIDTH)) + 1) * MMD_BIN_WIDTH
        axis.hist(scores, bins=bins, weights=np.full(len(scores), 1 / len(scores)),
                  histtype="stepfilled", color="#1f77b4", edgecolor="#1f77b4",
                  alpha=0.5, linewidth=1.3, zorder=3)
        axis.set(xlabel=r"Independent-ensemble MMD$^2$", ylabel="Probability per bin",
                 xlim=(0, bins[-1]))
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#e5e5e5", linewidth=1)
        axis.spines[["top", "right"]].set_visible(False)
        figure.savefig(output / "mmd_loss_histogram.png", dpi=200)
        plt.close(figure)
    analyze_delta_g(output, runs)
    print(f"Saved PNG plots and CSV metrics to {output}")


if __name__ == "__main__":
    main()

