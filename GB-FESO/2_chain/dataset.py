"""Read four-bead LAMMPS trajectories as three distances and two signed angles.

Distances use minimum-image displacements in an orthogonal periodic box.
Angles are measured in the xy plane, in radians. Trajectory directories are
sorted by name, so use zero-padded names such as 001, 002, ...
"""

import bisect
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_PAIR_COEFF_RE = re.compile(
    r"^\s*pair_coeff\s+(\d+)\s+(\d+)\s+([0-9eE+\-\.]+)\s+([0-9eE+\-\.]+)"
)

TARGET_COLUMNS = ("r12", "r23", "r34", "theta123", "theta234")
DISTANCE_COLUMNS = TARGET_COLUMNS[:3]
ANGLE_COLUMNS = TARGET_COLUMNS[3:]


def _min_image(delta: float, length: float) -> float:
    return delta - length * round(delta / length)


def _pbc_vec(
    a_xyz: tuple[float, float, float],
    b_xyz: tuple[float, float, float],
    Lx: float,
    Ly: float,
    Lz: float,
) -> tuple[float, float, float]:
    """Minimum-image vector from b to a."""
    dx = _min_image(a_xyz[0] - b_xyz[0], Lx)
    dy = _min_image(a_xyz[1] - b_xyz[1], Ly)
    dz = _min_image(a_xyz[2] - b_xyz[2], Lz)
    return dx, dy, dz


def _norm3(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _signed_angle_xy(u, v) -> float:
    """Signed angle from u to v in the xy plane, in [-pi, pi]."""
    ux, uy = u[0], u[1]
    vx, vy = v[0], v[1]
    cross = ux * vy - uy * vx
    dot = ux * vx + uy * vy
    return math.atan2(cross, dot)


def read_epsilons_from_in(
    in_path: str,
    eps_pairs=((1, 2), (2, 4)),
) -> tuple[float, float]:
    """Read eps1 and eps2 from the selected pair_coeff lines in in.in."""
    wanted = [tuple(sorted(pair)) for pair in eps_pairs]
    found = {}

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            m = _PAIR_COEFF_RE.match(line)
            if not m:
                continue

            i = int(m.group(1))
            j = int(m.group(2))
            pair = tuple(sorted((i, j)))
            if pair not in wanted:
                continue

            eps = float(m.group(3))
            sigma = float(m.group(4))
            if sigma == 0.0:
                raise ValueError(f"Parsed sigma=0 from {in_path}, line: {line.strip()}")
            found[pair] = eps

    missing = [pair for pair in wanted if pair not in found]
    if missing:
        missing_str = ", ".join(f"{i} {j}" for i, j in missing)
        raise ValueError(f"Could not find required pair_coeff epsilon(s) for pair(s) {missing_str} in: {in_path}")

    return tuple(found[pair] for pair in wanted)


def parse_dump_geometry(
    dump_path: str,
    atom_ids=(1, 2, 3, 4),
) -> tuple[np.ndarray, np.ndarray]:
    """Return timesteps and [r12, r23, r34, theta123, theta234] per frame.

    Requires id, x, y, z columns and an orthogonal box. theta123 runs from
    2->1 to 2->3; theta234 runs from 3->2 to 3->4. Frames missing any selected
    atom are skipped.
    """
    a1, a2, a3, a4 = atom_ids
    timesteps = []
    targets = []

    with open(dump_path, "r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not line.startswith("ITEM: TIMESTEP"):
                continue

            ts_line = f.readline()
            if not ts_line:
                break
            ts = int(ts_line.strip())

            line = f.readline()
            if not line.startswith("ITEM: NUMBER OF ATOMS"):
                raise ValueError(f"Unexpected dump format near: {line.strip()}")
            natoms = int(f.readline().strip())

            line = f.readline()
            if not line.startswith("ITEM: BOX BOUNDS"):
                raise ValueError(f"Unexpected dump format near: {line.strip()}")
            xlo, xhi = map(float, f.readline().split()[:2])
            ylo, yhi = map(float, f.readline().split()[:2])
            zlo, zhi = map(float, f.readline().split()[:2])

            Lx = xhi - xlo
            Ly = yhi - ylo
            Lz = zhi - zlo
            if Lx <= 0 or Ly <= 0 or Lz <= 0:
                raise ValueError(f"Non-positive box length in {dump_path}: Lx={Lx}, Ly={Ly}, Lz={Lz}")

            line = f.readline()
            if not line.startswith("ITEM: ATOMS"):
                raise ValueError(f"Unexpected dump format near: {line.strip()}")

            cols = line.strip().split()[2:]
            try:
                id_i = cols.index("id")
                x_i = cols.index("x")
                y_i = cols.index("y")
                z_i = cols.index("z")
            except ValueError as e:
                raise ValueError(f"Dump missing required columns (id,x,y,z). Found: {cols}") from e

            pos = {}
            wanted = {a1, a2, a3, a4}

            for _ in range(natoms):
                parts = f.readline().split()
                if not parts:
                    break
                aid = int(parts[id_i])
                if aid in wanted:
                    pos[aid] = (float(parts[x_i]), float(parts[y_i]), float(parts[z_i]))

            if any(a not in pos for a in (a1, a2, a3, a4)):
                continue

            r1, r2, r3, r4 = pos[a1], pos[a2], pos[a3], pos[a4]
            v21 = _pbc_vec(r1, r2, Lx, Ly, Lz)
            v23 = _pbc_vec(r3, r2, Lx, Ly, Lz)
            v32 = _pbc_vec(r2, r3, Lx, Ly, Lz)
            v34 = _pbc_vec(r4, r3, Lx, Ly, Lz)

            r12 = _norm3(v21)
            r23 = _norm3(v23)
            r34 = _norm3(v34)
            th123 = _signed_angle_xy(v21, v23)
            th234 = _signed_angle_xy(v32, v34)

            timesteps.append(ts)
            targets.append((r12, r23, r34, th123, th234))

    return np.asarray(timesteps, dtype=np.int64), np.asarray(targets, dtype=np.float32).reshape(-1, 5)


def parse_dump_angles(
    dump_path: str,
    atom_ids=(1, 2, 3, 4),
) -> tuple[np.ndarray, np.ndarray]:
    """Read only the signed angles [theta123, theta234]."""
    timesteps, target = parse_dump_geometry(dump_path, atom_ids=atom_ids)
    return timesteps, target[:, 3:5]


@dataclass(frozen=True)
class TrajectoryInfo:
    traj_id: int
    name: str
    dump_path: str
    in_path: str
    eps1: float
    eps2: float
    nframes: int


class LammpsTrajectoryGeometryDataset(Dataset):
    """One item per trajectory, with raw geometry shaped [frames, 5]."""

    target_columns = TARGET_COLUMNS

    def __init__(
        self,
        root_dir: str,
        dump_filename: str = "dump.lammpstrj",
        in_filename: str = "in.in",
        atom_ids=(1, 2, 3, 4),
    ):
        self.root_dir = str(Path(root_dir).resolve())
        self.atom_ids = atom_ids

        traj_dirs = []
        for path in sorted(Path(self.root_dir).iterdir()):
            if not path.is_dir():
                continue
            dump_path = path / dump_filename
            in_path = path / in_filename
            if dump_path.is_file() and in_path.is_file():
                traj_dirs.append((path.name, str(dump_path), str(in_path)))

        if not traj_dirs:
            raise FileNotFoundError(
                f"No trajectories found in {self.root_dir} (expected subdirs with {dump_filename} and {in_filename})."
            )

        self._info = []
        self._name_to_id = {}
        self._timesteps = []
        self._targets = []
        self._eps = []

        for traj_id, (name, dump_path, in_path) in enumerate(traj_dirs):
            eps1, eps2 = read_epsilons_from_in(in_path)
            ts_np, target_np = parse_dump_geometry(dump_path, atom_ids=atom_ids)

            self._info.append(
                TrajectoryInfo(
                    traj_id=traj_id,
                    name=name,
                    dump_path=dump_path,
                    in_path=in_path,
                    eps1=eps1,
                    eps2=eps2,
                    nframes=int(target_np.shape[0]),
                )
            )
            self._name_to_id[name] = traj_id
            self._eps.append((eps1, eps2))
            self._timesteps.append(torch.from_numpy(ts_np))
            self._targets.append(torch.from_numpy(target_np))

    def __len__(self) -> int:
        return len(self._info)

    def __getitem__(self, traj_id: int) -> dict:
        info = self._info[traj_id]
        eps = torch.tensor([info.eps1, info.eps2], dtype=torch.float32)
        target = self._targets[traj_id]
        return {
            "traj_id": info.traj_id,
            "name": info.name,
            "eps1": eps[0],
            "eps2": eps[1],
            "eps": eps,
            "timesteps": self._timesteps[traj_id],
            "target": target,
            "geometry": target,
            "distances": target[:, 0:3],
            "angles": target[:, 3:5],
            "target_columns": TARGET_COLUMNS,
        }

    def infos(self) -> list[TrajectoryInfo]:
        return list(self._info)

    def name_to_id(self, name: str) -> int:
        return self._name_to_id[name]

    def get_epsilons(self, traj_id: int) -> tuple[float, float]:
        return self._eps[traj_id]

    def dump_path(self, traj_id: int) -> str:
        return self._info[traj_id].dump_path

    def in_path(self, traj_id: int) -> str:
        return self._info[traj_id].in_path


class GeometryFrameDataset(Dataset):
    """Flatten trajectories into (normalized geometry, raw epsilons, trajectory ID).

    Distances are min/max scaled to [-1, 1]; angles are divided by pi.
    Ranges come from these trajectories unless supplied explicitly.
    """

    target_columns = TARGET_COLUMNS
    distance_indices = (0, 1, 2)
    angle_indices = (3, 4)
    target_dim = 5

    def __init__(self, traj_ds, eps_range=None, distance_range=None):
        self.traj_ds = traj_ds
        self.offsets = [0]
        self.eps_raw = []
        all_targets = []

        for tid in range(len(traj_ds)):
            s = traj_ds[tid]
            target = s["target"]
            if torch.is_tensor(target):
                target_np = target.detach().cpu().numpy().astype(np.float32)
            else:
                target_np = np.asarray(target, dtype=np.float32)
            if target_np.ndim != 2 or target_np.shape[1] != self.target_dim:
                raise ValueError(f"Expected target shape [T, 5], got {target_np.shape}")

            self.offsets.append(self.offsets[-1] + len(target_np))
            all_targets.append(target_np)

            e = s["eps"]
            e_np = e.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(e) else np.asarray(e, dtype=np.float32)
            if e_np.shape != (2,):
                raise ValueError(f"Expected eps shape [2], got {e_np.shape}")
            self.eps_raw.append(e_np)

        self.total = self.offsets[-1]
        if self.total <= 0:
            raise ValueError("No frames found.")

        eps_values = np.stack(self.eps_raw, axis=0)
        if eps_range is None:
            self.eps_min = eps_values.min(axis=0).astype(np.float32)
            self.eps_max = eps_values.max(axis=0).astype(np.float32)
        else:
            r = np.asarray(eps_range, dtype=np.float32)
            if r.shape == (2,):
                self.eps_min = np.array([r[0], r[0]], dtype=np.float32)
                self.eps_max = np.array([r[1], r[1]], dtype=np.float32)
            elif r.shape == (2, 2):
                self.eps_min = r[0].astype(np.float32)
                self.eps_max = r[1].astype(np.float32)
            else:
                raise ValueError("eps_range must be (min, max) or ((eps1_min, eps2_min), (eps1_max, eps2_max)).")

        if np.any(self.eps_max <= self.eps_min):
            raise ValueError("Invalid epsilon range.")

        targets = np.concatenate(all_targets, axis=0)
        if distance_range is None:
            self.distance_min = targets[:, 0:3].min(axis=0).astype(np.float32)
            self.distance_max = targets[:, 0:3].max(axis=0).astype(np.float32)
        else:
            dr = np.asarray(distance_range, dtype=np.float32)
            if dr.shape != (2, 3):
                raise ValueError("distance_range must have shape [(r12_min,r23_min,r34_min), (r12_max,r23_max,r34_max)].")
            self.distance_min = dr[0].astype(np.float32)
            self.distance_max = dr[1].astype(np.float32)

        span = self.distance_max - self.distance_min
        bad = span <= 0
        if np.any(bad):
            self.distance_min = self.distance_min.copy()
            self.distance_max = self.distance_max.copy()
            self.distance_min[bad] -= 0.5
            self.distance_max[bad] += 0.5

        self.target_min = np.concatenate([self.distance_min, np.array([-math.pi, -math.pi], dtype=np.float32)])
        self.target_max = np.concatenate([self.distance_max, np.array([math.pi, math.pi], dtype=np.float32)])

    def __len__(self):
        return self.total

    def _locate(self, idx: int):
        tid = bisect.bisect_right(self.offsets, idx) - 1
        frame = idx - self.offsets[tid]
        return tid, frame

    def normalize_target(self, target) -> torch.Tensor:
        target_t = torch.as_tensor(target, dtype=torch.float32)
        device = target_t.device
        dmin = torch.as_tensor(self.distance_min, device=device, dtype=target_t.dtype)
        dmax = torch.as_tensor(self.distance_max, device=device, dtype=target_t.dtype)

        out = torch.empty_like(target_t)
        out[..., 0:3] = 2.0 * (target_t[..., 0:3] - dmin) / (dmax - dmin) - 1.0
        out[..., 3:5] = target_t[..., 3:5] / math.pi
        return out

    def denormalize_target(self, x_norm: torch.Tensor, clamp: bool = True) -> torch.Tensor:
        x = torch.as_tensor(x_norm, dtype=torch.float32, device=x_norm.device if torch.is_tensor(x_norm) else None)
        if clamp:
            x = x.clamp(-1.0, 1.0)
        dmin = torch.as_tensor(self.distance_min, device=x.device, dtype=x.dtype)
        dmax = torch.as_tensor(self.distance_max, device=x.device, dtype=x.dtype)

        out = torch.empty_like(x)
        out[..., 0:3] = dmin + 0.5 * (x[..., 0:3] + 1.0) * (dmax - dmin)
        angles = math.pi * x[..., 3:5]
        out[..., 3:5] = torch.atan2(torch.sin(angles), torch.cos(angles))
        return out

    def target_angles(self, target: torch.Tensor) -> torch.Tensor:
        return target[..., 3:5]

    def get_traj_eps_raw(self, tid: int) -> tuple[float, float]:
        eps = self.eps_raw[tid]
        return float(eps[0]), float(eps[1])

    def __getitem__(self, idx: int):
        tid, frame = self._locate(idx)
        s = self.traj_ds[tid]
        target = s["target"][frame].to(torch.float32)
        x0 = self.normalize_target(target)
        eps = torch.from_numpy(np.asarray(self.eps_raw[tid], dtype=np.float32))
        return x0, eps, torch.tensor(tid, dtype=torch.int64)
