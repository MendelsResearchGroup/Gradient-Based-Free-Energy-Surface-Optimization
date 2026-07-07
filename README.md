# Toy Four-Bead DDPM Inverse Optimization

This repository contains scripts used to generate LAMMPS trajectories for a four-bead Lennard-Jones toy system and to perform DDPM-based inverse optimization of the interaction parameters (epsilon_1) and (epsilon_2).

## Repository Structure

```bash
lammps_generation/
  data.polyLJ
  gener.py
  prepare_simulation_runs.sh
  run_lammps.sh
  plot_distributions.py

ddpm_inverse_optimization/
  dataset.py
  model.py
  train.py
  loss.py
  sampler.py
  utils.py
  plotters.py
  ddpm_to_publish.py
```

Rename folders or files if needed, but keep the same relative structure when running the commands below.

## Requirements

The LAMMPS dataset generation requires a working LAMMPS executable.

The DDPM inverse optimization requires Python with:

```bash
torch
numpy
matplotlib
```

## LAMMPS Dataset Generation

This section generates multiple randomized LAMMPS trajectories.

Move to the LAMMPS generation folder:

```bash
cd lammps_generation
```

Create trajectory folders. Replace `100` with the desired number of trajectories:

```bash
mkdir {001..100}
```

Generate `in.in` and copy `data.polyLJ` into each trajectory folder:

```bash
./prepare_simulation_runs.sh .
```

Run LAMMPS in every trajectory folder:

```bash
./run_lammps.sh /home/user/lammps/build/lmp -in in.in
```

Optional: plot angle distributions after the simulations finish:

```bash
python3 plot_distributions.py .
```

Each trajectory folder should contain at least:

```bash
in.in
data.polyLJ
dump.lammpstrj
```

## DDPM Inverse Optimization

This section trains or loads the DDPM model and optimizes epsilon values against each trajectory in the dataset.

Move to the DDPM method folder:

```bash
cd ddpm_inverse_optimization
```

Run optimization over the full dataset. Replace the dataset path and starting epsilons:

```bash
python3 ddpm_to_publish.py \
  --path ../lammps_generation \
  --eps1 0.25 \
  --eps2 1.4
```

By default, the script uses:

```bash
optimize_eps=both
kde_bandwidth=0.5
steps=201
lr=5e-2
lr_refine=5e-2
plot_steps=0,10,50,100
plot_loss_surface=True
loss_surface_grid_size=25
num_samples=3500
loss_surface_num_samples=3500
clamp_eps=True
kl_lambda=1.0
```

Optional example with changed settings:

```bash
python3 ddpm_to_publish.py \
  --path ../lammps_generation \
  --eps1 0.25 \
  --eps2 1.4 \
  --steps 100 \
  --kde_bandwidth 0.5 \
  --kl_lambda 0.0 \
  --no-plot-loss-surface
```

## Outputs

The optimization writes trajectory-specific folders containing free-energy plots and optimization diagnostics.

The summary table is written to:

```bash
results_evaluation.csv
```

The CSV contains:

```bash
trajID
start_id
start_eps1
start_eps2
clamp_eps
eps1_target
eps2_target
final_eps1
final_eps2
phase0_best_eps1
phase0_best_eps2
phase0_best_loss
phase1_best_eps1
phase1_best_eps2
phase1_best_loss
eval_phase
target_type
active_names
missing_active_basin
df_target
df_generated
df_abs_error
df_mae
df_median_abs_error
```

## Reproducibility Notes

The random seed can be changed with:

```bash
--seed <integer>
```

The starting epsilon values are set with:

```bash
--eps1 <value> --eps2 <value>
```

The symmetric KDE loss weight is controlled by:

```bash
--kl_lambda <value>
```

Use `--kl_lambda 0.0` for the one-sided loss.

## Data and Code Availability

The code and generated datasets supporting this work are provided with this repository for review. The final archived version will be deposited in a persistent public repository, and the DOI will be added to this README and to the manuscript Data Availability Statement before publication.
