# Two-Chain LAMMPS Dataset Generation

These scripts generate trajectories of two five-bead chains in 2D using reduced Lennard-Jones units, harmonic bonds, and harmonic angles. Each bead independently samples its epsilon from `list.txt`, with replacement.

## Folder Structure

```text
2_chain_data/
  data.polyLJ
  list.txt
  gener.py
  generall.sh
  run_lammps.sh
```

## Requirements

Bash, Python 3 (standard library only), and a working LAMMPS executable.

## Dataset Generation

Move to the generation folder:

```bash
cd 2_chain_data
```

Create trajectory folders. Replace `100` with the desired number of trajectories:

```bash
mkdir {0001..0100}
```

Keep only trajectory folders as subdirectories, since the scripts process every immediate subdirectory.

Generate `in.in` and copy `data.polyLJ` into each trajectory folder:

```bash
bash generall.sh
```

Run LAMMPS in every trajectory folder, replacing the executable path:

```bash
bash run_lammps.sh /path/to/lmp -in in.in
```

Each simulation runs for 100,000,000 steps with timestep `0.005`, Langevin temperature `0.13`, and trajectory output every 2,000 steps. These settings are defined in `gener.py`.

## Outputs

Each trajectory folder contains:

```text
in.in
data.polyLJ
dump.lammpstrj
log.lammps
```

Run remove_no_contact.py keep only frames in contact and removes runs with frame count of less than 2048 frames:

```bash
python remove_no_contact.py
```
