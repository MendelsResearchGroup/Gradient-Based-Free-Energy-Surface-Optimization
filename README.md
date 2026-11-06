# Gradient-Based Free Energy Surface Optimization (GB-FESO)

GB-FESO finds parameters that provide Free Energy Surface close to the target ensemble using conditional
denoising diffusion model (DDPM)  as a surrogate. In the showcase DDPM is trained on molecular dynamics trajectories
labelled by their Lennard-Jones epsilons. The epsilons are then optimized by backpropagating a distribution loss (KL or MMD²) through the DDIM sampler, so that generated samples reproduce the target free-energy surface.


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
