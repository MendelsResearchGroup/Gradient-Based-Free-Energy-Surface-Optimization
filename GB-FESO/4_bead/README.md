# Four-Bead DDPM Training and Optimization

Train a conditional DDPM for a four-bead chain in 2D and optimize its two Lennard-Jones epsilons using KL or MMD². The model generates three distances and two signed angles.

## Requirements

Python 3.10+, PyTorch, NumPy, Matplotlib, and SciPy. Dependencies are listed in `requirements.txt`. Run commands from this folder.

## Dataset

Provide training and validation folders:

```text
data/
  train/001/in.in
  train/001/dump.lammpstrj
  validation/001/in.in
  validation/001/dump.lammpstrj
```

Each folder contains one trajectory. Inputs require `pair_coeff 1 2` and `pair_coeff 2 4`; dumps require atoms 1–4, `id x y z` columns, and orthogonal periodic boxes. The example trajectory is in the test folder.

All frames are used. Trajectory IDs are zero-based, sorted by folder name. Simulations must share temperature and physical parameters apart from epsilons. 

## Training

Set the data path and settings in `train.py`:

```bash
python train.py
```

The best training-loss checkpoint is saved to `ddpm_best_epoch_XXXX.pt`. Loading restores its training distance scaling and epsilon bounds, including for a single validation trajectory.

## Optimization

Set `DATA_ROOT`, `CHECKPOINT_PATH`, starting pairs, trajectory IDs, and configuration at the bottom of `run_feso.py`:

```bash
python run_feso.py
```

| Setting | Options |
| --- | --- |
| `loss_dim` | `2`: angles; `5`: distances and angles |
| `loss_type` | `"kl"` or `"mmd"` |
| `optimize_eps` | `"eps1"`, `"eps2"`, or `"both"` |
| `fixed_eps_raw` | `None`: keep the other epsilon at its target value; a number overrides it. Ignored for `"both"`. |
| `clamp_eps` | Restrict epsilons to the checkpoint's training range |

Learning rates, steps, bandwidths, and sample counts are set in the same configuration. Losses use raw distance units and radians.

Selection and refinement use the chosen dimension. After optimization finishes, both KL and MMD² are evaluated in **5D** on the saved best-step samples, without resampling.

Use a new `OUTPUT_ROOT` to repeat a mode.

## Loss and ΔG Analysis

Set `RESULTS_ROOT` and `OUTPUT_DIR` in the plotting scripts:

```bash
python plot_loss_histogram.py
python plot_delta_g.py
```

Analysis uses saved recovery CSVs and final 5D scores. Basin bounds are defined in `utils.py`. ΔG/kBT = ln(P_A/P_B); A and B must each contain >5% of frames in both ensembles by default. This assumes equilibrium sampling at a common temperature.

For target/generated FES plots, set paths in `sample_fes.py` and run `python sample_fes.py`.

## Outputs

| Folder | Contents |
| --- | --- |
| `results/{2d,5d}_{kl,mmd}/` | Recovery CSV, optimized epsilons, 5D scores, and optional progress archives/PNGs |
| `results/plots/` | Loss histograms and ΔG PNGs |
| `sample_fes/` | Target/generated FES PNGs |
