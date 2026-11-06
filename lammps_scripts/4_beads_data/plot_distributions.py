#!/usr/bin/env python3
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


IDS = (1, 2, 3, 4)


def _min_image(dx: float, L: float) -> float:
    return dx - L * round(dx / L)


def _pbc_vec(a, b, Lx, Ly):
    dx = _min_image(a[0] - b[0], Lx)
    dy = _min_image(a[1] - b[1], Ly)
    return dx, dy


def _signed_angle(u, v) -> float:
    ux, uy = u
    vx, vy = v
    cross = ux * vy - uy * vx
    dot = ux * vx + uy * vy
    return math.atan2(cross, dot)


def read_epsilons(in_path: str):
    """
    Reads eps1, eps2, eps3 from in.in:

    eps1 = epsilon between atom types 1 and 2
    eps2 = epsilon between atom types 2 and 4
    eps3 = epsilon between atom types 1 and 4
    """
    wanted = {
        (1, 2): "eps1",
        (2, 4): "eps2",
        (1, 4): "eps3",
    }
    eps = {}

    if not os.path.isfile(in_path):
        return eps

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5 or parts[0] != "pair_coeff":
                continue

            try:
                i = int(parts[1])
                j = int(parts[2])
                value = float(parts[3])
            except ValueError:
                continue

            pair = tuple(sorted((i, j)))
            if pair in wanted:
                eps[wanted[pair]] = value

    return eps


def format_eps_legend(eps: dict) -> str:
    parts = []
    for key in ("eps1", "eps2", "eps3"):
        if key in eps:
            parts.append(f"{key} = {eps[key]:.6f}")
        else:
            parts.append(f"{key} = missing")
    return "\n".join(parts)


def iter_angles_from_dump(dump_path: str):
    """
    Yields (theta_123, theta_234) for each frame in a LAMMPS dump custom file.
    Angles are signed in [-pi, pi], computed with minimum-image convention.
    """
    with open(dump_path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                return

            if not line.startswith("ITEM: TIMESTEP"):
                continue

            _timestep = f.readline()

            line = f.readline()
            if not line:
                return
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Unexpected format near: {line.strip()}")
            natoms = int(f.readline().strip())

            line = f.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Unexpected format near: {line.strip()}")
            xlo, xhi = map(float, f.readline().split()[:2])
            ylo, yhi = map(float, f.readline().split()[:2])
            _zlo, _zhi = map(float, f.readline().split()[:2])
            Lx = xhi - xlo
            Ly = yhi - ylo

            line = f.readline()
            if not line.startswith("ITEM: ATOMS"):
                raise ValueError(f"Unexpected format near: {line.strip()}")

            cols = line.strip().split()[2:]
            try:
                id_i = cols.index("id")
                x_i = cols.index("x")
                y_i = cols.index("y")
            except ValueError as e:
                raise ValueError(f"Dump is missing required columns (id,x,y): {cols}") from e

            pos = {}
            for _ in range(natoms):
                parts = f.readline().split()
                if not parts:
                    return

                aid = int(parts[id_i])
                if aid in IDS:
                    pos[aid] = (float(parts[x_i]), float(parts[y_i]))

            if any(i not in pos for i in IDS):
                continue

            r1, r2, r3, r4 = pos[1], pos[2], pos[3], pos[4]

            v21 = _pbc_vec(r1, r2, Lx, Ly)
            v23 = _pbc_vec(r3, r2, Lx, Ly)
            th123 = _signed_angle(v21, v23)

            v32 = _pbc_vec(r2, r3, Lx, Ly)
            v34 = _pbc_vec(r4, r3, Lx, Ly)
            th234 = _signed_angle(v32, v34)

            yield th123, th234


def find_dump_file(sim_dir: str) -> str | None:
    cand = os.path.join(sim_dir, "dump.lammpstrj")
    if os.path.isfile(cand):
        return cand

    for name in sorted(os.listdir(sim_dir)):
        if name.endswith(".lammpstrj"):
            path = os.path.join(sim_dir, name)
            if os.path.isfile(path):
                return path

    return None


def make_heatmap(angles1, angles2, out_png: str, title: str, eps: dict):
    bins = 200
    rng = [[-math.pi, math.pi], [-math.pi, math.pi]]

    H, _xedges, _yedges = np.histogram2d(
        angles1,
        angles2,
        bins=bins,
        range=rng,
        density=True,
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[-math.pi, math.pi, -math.pi, math.pi],
        aspect="equal",
        interpolation="nearest",
    )

    ax.set_xlabel("angle(1-2-3) [rad]")
    ax.set_ylabel("angle(2-3-4) [rad]")
    ax.set_title(title)

    legend_label = format_eps_legend(eps)
    legend_handle = Line2D([], [], linestyle="none", label=legend_label)
    ax.legend(
        handles=[legend_handle],
        loc="upper right",
        frameon=True,
        handlelength=0,
        handletextpad=0,
        title="LJ epsilons",
    )

    fig.colorbar(im, ax=ax, label="Probability density")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(root_dir: str):
    root_dir = os.path.abspath(root_dir)

    for name in sorted(os.listdir(root_dir)):
        sim_dir = os.path.join(root_dir, name)
        if not os.path.isdir(sim_dir):
            continue

        dump_path = find_dump_file(sim_dir)
        if not dump_path:
            continue

        in_path = os.path.join(sim_dir, "in.in")
        eps = read_epsilons(in_path)

        a1 = []
        a2 = []
        for th123, th234 in iter_angles_from_dump(dump_path):
            a1.append(th123)
            a2.append(th234)

        if not a1:
            continue

        out_png = os.path.join(root_dir, f"{name}_angle_heatmap.png")
        make_heatmap(
            np.array(a1, dtype=float),
            np.array(a2, dtype=float),
            out_png,
            title=name,
            eps=eps,
        )
        print(f"saved: {out_png}")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    main(root)
