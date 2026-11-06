from pathlib import Path
from random import choice, randint


# Each bead independently samples its self-interaction epsilon, with replacement.
epsilon_values = [float(value) for value in Path(__file__).with_name("list.txt").read_text().split()]
epsilons = [choice(epsilon_values) for _ in range(10)]
langevin_seed = randint(1, 900_000_000)

input_text = f"""units           lj
dimension       2
boundary        p p p

atom_style      angle

# Define interaction styles BEFORE reading topology
pair_style      lj/cut 4.0
bond_style      harmonic
angle_style     harmonic

# Read atoms, bonds, and angles; this creates the simulation box
read_data       data.polyLJ

# Coefficients AFTER read_data
pair_modify     mix geometric
pair_coeff      1 1 {epsilons[0]:.6f} 1.5
pair_coeff      2 2 {epsilons[1]:.6f} 1.5
pair_coeff      3 3 {epsilons[2]:.6f} 1.5
pair_coeff      4 4 {epsilons[3]:.6f} 1.5
pair_coeff      5 5 {epsilons[4]:.6f} 1.5
pair_coeff      6 6 {epsilons[5]:.6f} 1.5
pair_coeff      7 7 {epsilons[6]:.6f} 1.5
pair_coeff      8 8 {epsilons[7]:.6f} 1.5
pair_coeff      9 9 {epsilons[8]:.6f} 1.5
pair_coeff      10 10 {epsilons[9]:.6f} 1.5

bond_coeff      1 80.15 1.5

# Small bending stiffness toward straight configuration
angle_coeff     1 0.25 180.0

# Keep LJ between bonded neighbors
special_bonds   lj 1.0 1.0 1.0

# Neighbor list
neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes

# Temperature / dynamics
velocity        all create 0.08 12345 mom yes rot no dist gaussian
fix             thermo all langevin 0.13 0.13 1.0 {langevin_seed}
timestep        0.005
fix             integ all nve
fix             3 all enforce2d

# Output
thermo          2000
thermo_style    custom step temp pe ke etotal
dump            traj all custom 2000 dump.lammpstrj id type x y z vx vy vz

run             100000000
"""

Path("in.in").write_text(input_text)
