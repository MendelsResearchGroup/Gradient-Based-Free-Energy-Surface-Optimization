"""Plot final 5D losses from recovery.csv files. Every row has equal weight.

Groups may contain different target/start pairs; this is an unpaired descriptive
comparison. Repeated starts share targets and are not independent replicates.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_ROOT = Path("results")
OUTPUT_DIR = RESULTS_ROOT / "plots"


def plot_loss_histogram(results_paths, output_dir=OUTPUT_DIR):
    groups = {}
    settings = set()
    for path in results_paths:
        with Path(path).open(newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["evaluation_dim"]) != 5 or int(row["loss_dim"]) not in (2, 5):
                    raise ValueError(f"{path}: expected 2D/5D optimization with final 5D evaluation")
                label = f"{row['loss_dim']}D / {row['loss_type'].upper()}"
                values = (float(row["KL_5D"]), float(row["MMD2_5D"]))
                if not np.isfinite(values).all():
                    raise ValueError(f"{path}: missing or nonfinite final 5D losses")
                groups.setdefault(label, []).append(values)
                settings.add(tuple(float(row[key]) for key in
                                   ("kde_bandwidth", "mmd_bandwidth", "kl_lambda", "num_samples")))
    if not groups:
        raise ValueError("No recovery rows found")
    if len(settings) != 1:
        raise ValueError("Compare runs with the same bandwidths, KL weight, and generated sample count")
    groups = {label: np.asarray(values) for label, values in sorted(groups.items())}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, (name, label) in enumerate((("kl", "Final 5D KDE–KL loss"), ("mmd", r"Final 5D MMD$^2$"))):
        pooled = np.concatenate([values[:, index] for values in groups.values()])
        lo, hi = min(0.0, pooled.min()), max(0.0, pooled.max())
        bins = np.histogram_bin_edges(pooled, bins="fd", range=(lo, hi if hi > lo else lo + 1))
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for (group, samples), color in zip(groups.items(), plt.rcParams["axes.prop_cycle"].by_key()["color"]):
            values = samples[:, index]
            ax.hist(values, bins=bins, weights=np.full(len(values), 1 / len(values)),
                    histtype="step", linewidth=1.8, color=color,
                    label=f"{group} (n={len(values)}, mean={values.mean():.4g})")
            ax.axvline(values.mean(), color=color, linestyle=":", linewidth=1)
        ax.set(xlabel=label, ylabel="Probability per bin", ylim=(0, None))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        ax.grid(axis="y", color="0.9")
        fig.supxlabel("All available runs; selected-step samples; dotted lines show means", fontsize=9)
        path = output_dir / f"{name}_loss_histogram.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)
        print(f"Saved {path}")
    return paths


if __name__ == "__main__":
    plot_loss_histogram(sorted(RESULTS_ROOT.glob("*d_*/recovery.csv")))
