# Gradient-Based Free Energy Surface Optimization (GB-FESO)

GB-FESO finds parameters that provide Free Energy Surface close to the target ensemble using conditional
denoising diffusion model (DDPM)  as a surrogate. In the showcase DDPM is trained on molecular dynamics trajectories
labelled by their Lennard-Jones epsilons. The epsilons are then optimized by backpropagating a distribution loss (KL or MMD²) through the DDIM sampler, so that generated samples reproduce the target free-energy surface.


## Repository structure

```text
Gaussian_example/              Toy Gaussian example
GB-FESO/
  4_bead/                      Four-bead chain: train DDPM, optimize 2 epsilons (KL or MMD²)
    ddpm_best_epoch_0080.pt    Pretrained checkpoint
    test/003/                  Example target trajectory
  2_chain/                     Two five-bead chains: train DDPM, optimize 5 epsilons (MMD² on RMSDs)
    ddpm.pt                    Pretrained checkpoint
    test/0001/                 Example target trajectory
lammps_scripts/
  4_beads_data/                LAMMPS data generation for GB-FESO/4_bead
  2_chain_data/                LAMMPS data generation for GB-FESO/2_chain
```

## Installation

Python 3.10 or newer. A CUDA GPU is strongly recommended for training and
optimization; CPU works for small test runs.

```bash
pip install torch numpy scipy matplotlib tqdm
```

LAMMPS is needed only to generate new trajectories. The original data was
produced with LAMMPS 29 Aug 2024 (Update 4)


## License

MIT, see [LICENSE](LICENSE).
