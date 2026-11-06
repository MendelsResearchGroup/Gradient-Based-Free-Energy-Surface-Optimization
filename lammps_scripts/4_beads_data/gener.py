import argparse
import os
import random

TEMPLATE = """# 4 bead types LJ + weak bond, proper 2D, very low temperature
# same-type epsilon is 0
# eps1 is randomized in [{eps_min}, {eps1_max}]
# eps2 is randomized in [{eps_min}, {eps2_max}]

units           lj
dimension       2
atom_style      bond
boundary        p p p          # in 2D, z must be periodic

read_data       data.polyLJ

# Keep LJ between bonded neighbors
special_bonds lj 1.0 1.0 1.0

# Neighbor list
neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes

# LJ: explicit pair coefficients
pair_style      lj/cut 4.0
#pair_modify     mix geometric
pair_coeff      1 1 {eps:.6f} 1.0
pair_coeff      2 2 {eps:.6f} 1.0
pair_coeff      3 3 {eps:.6f} 1.0
pair_coeff      4 4 {eps:.6f} 1.0
pair_coeff      1 2 {eps1:.6f} 1.0
pair_coeff      1 3 {eps2:.6f} 1.0
pair_coeff      1 4 {eps3:.6f} 1.0
pair_coeff      2 3 {eps1:.6f} 1.0
pair_coeff      2 4 {eps2:.6f} 1.0
pair_coeff      3 4 {eps1:.6f} 1.0

# Weak bond to prevent separation
bond_style      harmonic
bond_coeff      1 0.15 1.5

# Very low temperature
velocity        all create 0.09 12345 mom yes rot no dist gaussian
fix             thermo all langevin 0.09 0.09 1.0 12345
timestep        0.005
fix             integ all nve
fix             3 all enforce2d

# Output
thermo          100
thermo_style    custom step temp pe ke etotal
dump            traj all custom 200 dump.lammpstrj id type x y z vx vy vz

run             4000000
"""

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a LAMMPS input file with randomized LJ epsilons."
    )
    p.add_argument(
        "outdir",
        help="Output directory where in.in will be written.",
    )
    p.add_argument(
        "--eps1-max",
        type=float,
        default=1.55,
        help="Maximum value for randomized eps1.",
    )
    p.add_argument(
        "--eps2-max",
        type=float,
        default=1.55,
        help="Maximum value for randomized eps2.",
    )

    args = p.parse_args()

    eps_min = 0.21

    eps = 0.0
    eps1 = random.uniform(eps_min, args.eps1_max)
    eps2 = random.uniform(eps_min, args.eps2_max)
    eps3 = 0.50000

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, "in.in")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(
            TEMPLATE.format(
                eps=eps,
                eps1=eps1,
                eps2=eps2,
                eps3=eps3,
                eps_min=eps_min,
                eps1_max=args.eps1_max,
                eps2_max=args.eps2_max,
            )
        )

    print(f"wrote: {out_path}")
    print(f"eps:  {eps:.6f}")
    print(f"eps1: {eps1:.6f}")
    print(f"eps2: {eps2:.6f}")
    print(f"eps3: {eps3:.6f}")

if __name__ == "__main__":
    main()
