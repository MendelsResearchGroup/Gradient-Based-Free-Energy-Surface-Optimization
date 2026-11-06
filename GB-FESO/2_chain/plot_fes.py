"""Optimizer diagnostic: paired RMSD histograms as F/(kBT) = -ln(P_bin)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from loss import mmd_loss


FES_BINS = 80


def free_energy_surface(values, bins):
    """Use all samples, with no smoothing, pseudocounts, or minimum shift."""
    values = np.asarray(values)
    if (values.ndim != 2 or values.shape[1] != 2 or len(values) == 0
            or not np.isfinite(values).all()):
        raise ValueError("Expected finite, nonempty (N, 2) RMSD samples.")
    counts = np.histogram2d(*values.T, bins=bins)[0]
    if counts.sum() != len(values):
        raise ValueError("Bin edges must cover every sample.")
    occupied = counts > 0
    surface = np.full_like(counts, np.inf)
    surface[occupied] = -np.log(counts[occupied] / len(values))
    return surface


def save_fes_comparison(target, generated, output_file, title, bandwidth=0.32,
                        target_epsilons=None, generated_epsilons=None):
    """Save PNG panels with common bins/color scale and raw-sample MMD squared."""
    target = target.detach().cpu().double()
    generated = generated.detach().cpu().double()
    score = mmd_loss(generated, target, bandwidth=bandwidth).item()
    values = (target.numpy(), generated.numpy())
    pooled = np.concatenate(values)
    bins = [np.histogram_bin_edges(pooled[:, i], bins=FES_BINS) for i in range(2)]
    surfaces = [free_energy_surface(samples, bins) for samples in values]
    finite = np.concatenate([surface[np.isfinite(surface)] for surface in surfaces])
    vmin, vmax = finite.min(), finite.max()
    if vmax == vmin:
        vmax = vmin + 1
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e5e5")
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    figure.subplots_adjust(left=0.08, right=0.94, bottom=0.17, top=0.77, wspace=0.08)
    for axis, surface, samples, label, epsilons in zip(
        axes, surfaces, values, ("Target", "Generated"),
        (target_epsilons, generated_epsilons),
    ):
        heatmap = axis.pcolormesh(*bins, np.ma.masked_invalid(surface.T),
                                  cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
        panel_title = f"{label} (N={len(samples):,})"
        if epsilons is not None:
            if len(epsilons) != 10:
                raise ValueError("Plot labels need ten epsilons in atom-ID order.")
            for chain, start in enumerate((0, 5), start=1):
                labels = ", ".join(f"{value:.4f}" for value in epsilons[start:start + 5])
                panel_title += f"\nChain {chain}: ε=[{labels}]"
        axis.set_title(panel_title, fontsize=10)
        axis.set_xlabel("RMSD to extended interface (LJ length)")
        axis.set_aspect("equal")
    axes[0].set_ylabel("RMSD to end-anchored dimer (LJ length)")
    figure.colorbar(heatmap, ax=axes, label=r"$F/(k_BT)=-\ln P_{\mathrm{bin}}$")
    figure.suptitle(f"{title}\nMMD² = {score:.5g} | Gaussian σ = {bandwidth:g} LJ length")
    figure.supxlabel("Gray: unsampled bins (F = +∞). MMD² uses the biased estimator.",
                    fontsize=10, y=0.03)
    figure.savefig(output_file, dpi=180)
    plt.close(figure)
    return score
