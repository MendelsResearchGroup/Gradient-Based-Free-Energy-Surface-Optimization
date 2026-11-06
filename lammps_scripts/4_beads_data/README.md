# Four-bead LAMMPS dataset generation

Generate trajectories for a four-bead chain with randomized Lennard-Jones
interaction strengths. Each run produces the same file format with newly sampled
parameters; it does not recreate a particular previously generated dataset.

## Requirements

- Bash and Python 3.10 or newer.
- A working LAMMPS executable supporting the commands in `gener.py`.
  The original simulations used LAMMPS 29 Aug 2024 – Update 4.
- NumPy and Matplotlib for the optional distribution plots.

## Generate a dataset

Run these commands in the directory containing this README, `gener.py`,
`data.polyLJ`, and the two shell scripts. The directory itself can have any name.

Create three-digit trajectory folders, for example 100 runs:

```bash
mkdir {001..100}
```

Generate a randomized `in.in` and copy the initial structure into each folder:

```bash
bash prepare_simulation_runs.sh .
```

The scripts process only three-digit folders such as `001`. Preparation
overwrites existing input files with newly sampled parameters, so use fresh
folders for a new dataset.

Run the simulations sequentially, replacing the executable path below:

```bash
bash run_lammps.sh /absolute/path/to/lmp -in in.in
```

If LAMMPS is on your `PATH`, use `lmp` instead of the absolute path.
Each completed trajectory folder contains:

```text
001/
  in.in           # simulation settings and sampled epsilon values
  data.polyLJ     # initial coordinates, atom types, masses, and bonds
  dump.lammpstrj  # trajectory: id type x y z vx vy vz
  log.lammps     # simulation log and thermodynamic output
```

Optionally plot the joint distribution of the signed angles 1–2–3 and 2–3–4:

```bash
python3 plot_distributions.py .
```

This writes `001_angle_heatmap.png`, etc., alongside the trajectory folders.
On a machine without a display, prefix the command with `MPLBACKEND=Agg`.

## Simulation settings

`gener.py` defines the simulation in reduced Lennard-Jones units:

- Four unit-mass beads in a periodic 2D box, connected by three harmonic bonds
  with coefficient `0.15` and equilibrium length `1.5`.
- Sigma is `1.0` and the LJ cutoff is `4.0`. LJ interactions include bonded pairs.
- Epsilon 1 is sampled uniformly from `[0.21, 1.55]` for pairs 1–2, 2–3, and 3–4.
- Epsilon 2 is sampled independently from `[0.21, 1.55]` for pairs 1–3 and 2–4.
- The 1–4 epsilon is fixed at `0.5`; same-type epsilons are zero.
- Langevin temperature `0.09`, damping time `1.0`, and timestep `0.005`.
- 4,000,000 steps, with a frame every 200 steps: 20,001 frames including step zero.

The initial velocities and Langevin thermostat use the fixed seed `12345`;
epsilon sampling is unseeded. The sampled values are saved in each `in.in`.
The plots use all frames with minimum-image bond vectors and no equilibration
discard. Equilibrium or independent sampling is not established by these plots.
