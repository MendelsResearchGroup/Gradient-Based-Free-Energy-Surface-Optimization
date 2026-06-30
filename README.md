# LAMMPS Dataset Generation

This section generates multiple randomized LAMMPS trajectories.

Expected files in the repo root:

```bash
data.polyLJ
gener.py
prepare_simulation_runs.sh
run_lammps.sh
plot_distributions.py
```

Rename files if needed.

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

# DDPM Inverse Optimization

This section trains or loads the DDPM model and optimizes epsilon values against each trajectory in the dataset.

Expected files in the DDPM method folder:

```bash
dataset.py
model.py
train.py
loss.py
sampler.py
utils.py
plotters.py
ddpm_to_publish.py
```

Run optimization over the full dataset. Replace the dataset path and starting epsilons:

```bash
python3 ddpm_to_publish.py \
  --path /path/to/lammps_dataset \
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
  --path /path/to/lammps_dataset \
  --eps1 0.25 \
  --eps2 1.4 \
  --steps 100 \
  --kde_bandwidth 0.5 \
  --kl_lambda 0.0 \
  --no-plot-loss-surface
```

Outputs are written to trajectory-specific optimization folders and to:

```bash
results_evaluation.csv
```

