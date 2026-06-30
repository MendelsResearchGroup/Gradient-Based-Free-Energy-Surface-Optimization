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

Rename files if needed:

```bash
mv generate_input_files.py gener.py
mv 'data(3).polyLJ' data.polyLJ
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
