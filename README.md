# LAMMPS Dataset Generation

Generate multiple LAMMPS trajectories from randomized input files.


## 1. Create trajectory folders

Replace `100` with the number of trajectories you want:

```bash
mkdir {001..100}
```

## 2. Generate input files

```bash
./prepare_simulation_runs.sh .
```

This creates one `in.in` per trajectory folder and copies `data.polyLJ` into each folder.

## 3. Run LAMMPS

```bash
./run_lammps.sh /home/user/lammps/build/lmp -in in.in
```

The script runs the command inside every trajectory folder.

## 4. Optional: plot distributions

After simulations finish:

```bash
python3 plot_distributions.py .
```

This creates angle heatmap PNG files in the root directory.
