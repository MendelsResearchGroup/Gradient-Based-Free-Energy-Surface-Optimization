# Two-Chain DDPM Training and Optimization

Train a conditional DDPM for two five-bead chains in 2D and optimize chain-1 epsilons using MMD² on pair-distance RMSDs. A trained `ddpm.pt` is included.

## Requirements

Python 3, PyTorch, NumPy, Matplotlib, and tqdm. Versions are listed in `requirements.txt`.

Run commands from this folder:

```bash
cd publication
```

## Dataset

Provide separate training and validation folders:

```text
data/
  train/0001/in.in
  train/0001/dump.lammpstrj
  val/0001/in.in
  val/0001/dump.lammpstrj
```

Each numbered folder contains one trajectory. Atom types equal atom IDs: 1–5 for chain 1, 6–10 for chain 2. Inputs require ten self-interaction `pair_coeff` entries; dumps require `id type x y` columns and orthogonal boxes.

The loader retains timesteps ≥1000 with an inter-chain distance ≤2.25, using 2D minimum-image distances. Simulations must share temperature and other physical parameters apart from epsilons.

## Training

Set `DATA_ROOT`, `VAL_ROOT`, and training settings in `train.py`:

```bash
python train.py
```

The best validation checkpoint is saved to `ddpm.pt`. Change `CHECKPOINT` to retain the included weights.

## Optimization

```bash
python optimizer_rmsd.py /path/to/dataset
```

The dataset path is required; all other arguments are optional. Chain 1's five epsilons are optimized while chain 2 remains fixed. Initial values are sampled from `list.txt`.

List available parameters:

```bash
python optimizer_rmsd.py --help
```

## MMD and ΔG Analysis

Set `RUNS_DIR` and `OUTPUT_DIR` in `analyze_results.py`:

```bash
python analyzer.py
```

Analysis uses saved optimization results. Basin bounds are defined in `analyzer.py`. ΔG/(kBT) = −ln(P_C/P_A); both basins must contain ≥10% of frames in both ensembles. The free-energy interpretation assumes equilibrium sampling of the contact-conditioned ensemble.

## Outputs

| Folder | Contents |
| --- | --- |
| `optimization_rmsd/` | Optimized epsilons, CV samples, histories, summary CSV, and diagnostic PNGs |
| `validation_maps/` | Target/generated contact-map PNGs |
| `analysis/` | MMD histograms, ΔG plots, and CSV metrics |
