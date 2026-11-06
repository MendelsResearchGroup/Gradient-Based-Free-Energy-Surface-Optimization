from bisect import bisect_right
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


START_STEP = 1000
CONTACT_CUTOFF = 2.25
ATOM_IDS = np.arange(1, 11)


class LammpsDistanceDataset(Dataset):
    """Labeled 2D distances and ten epsilons from numbered LAMMPS folders.

    Retain frames at/after start_step with an inter-chain distance <= cutoff.
    Use minimum-image x/y distances in orthogonal boxes; atom types equal IDs.
    Items contain frame_id, trajectory_id, epsilons (10,), distances (10, 10).
    Ranges are reported without normalization. Incomplete frames are skipped.
    """

    def __init__(self, root, trajectory_id=None,
                 start_step=START_STEP, contact_cutoff=CONTACT_CUTOFF):
        self.root = Path(root)
        self.start_step = start_step
        self.contact_cutoff = contact_cutoff

        if trajectory_id is None:
            directories = sorted(
                p for p in self.root.iterdir()
                if p.is_dir() and p.name.isdigit()
                and (p / "dump.lammpstrj").is_file()
                and (p / "in.in").is_file()
            )
        else:
            directory = self.root / f"{int(trajectory_id):04d}"
            if not (directory / "dump.lammpstrj").is_file():
                raise FileNotFoundError(directory / "dump.lammpstrj")
            if not (directory / "in.in").is_file():
                raise FileNotFoundError(directory / "in.in")
            directories = [directory]

        if not directories:
            raise FileNotFoundError(
                f"No trajectory folders with dump.lammpstrj and in.in under {self.root}"
            )

        self.trajectories = []
        self._ends = []
        self.excluded_frames = 0
        total = 0
        offdiag_min = np.inf
        distance_max = -np.inf

        for directory in directories:
            epsilons = self._read_eps(directory / "in.in")
            offsets, traj_min, traj_max, excluded = self._index_retained_frames(
                directory / "dump.lammpstrj"
            )
            self.excluded_frames += excluded
            if len(offsets) == 0:
                continue

            self.trajectories.append({
                "id": int(directory.name),
                "dump": directory / "dump.lammpstrj",
                "epsilons": epsilons,
                "offsets": offsets,
                "excluded_frames": excluded,
            })
            total += len(offsets)
            self._ends.append(total)
            offdiag_min = min(offdiag_min, traj_min)
            distance_max = max(distance_max, traj_max)

        if not self.trajectories:
            raise ValueError("No trajectory reached the contact criterion.")

        eps_values = np.array([traj["epsilons"] for traj in self.trajectories])
        self.ranges = {
            "epsilons": tuple(
                (float(values.min()), float(values.max())) for values in eps_values.T
            ),
            "distance": (0.0, float(distance_max)),
            "distance_off_diagonal": (float(offdiag_min), float(distance_max)),
        }

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        traj_index = bisect_right(self._ends, index)
        start = 0 if traj_index == 0 else self._ends[traj_index - 1]
        local_index = index - start
        traj = self.trajectories[traj_index]

        step, positions, lengths, periodic = self._read_frame(
            traj["dump"], int(traj["offsets"][local_index])
        )
        distances = self._distance_matrix(positions, lengths, periodic)

        return {
            "frame_id": step,
            "trajectory_id": traj["id"],
            "epsilons": torch.tensor(traj["epsilons"], dtype=torch.float32),
            "distances": torch.from_numpy(distances.astype(np.float32, copy=False)),
        }

    def trajectory(self, trajectory_id):
        """Return a Subset containing only frames from one trajectory ID."""
        trajectory_id = int(trajectory_id)
        for i, traj in enumerate(self.trajectories):
            if traj["id"] == trajectory_id:
                start = 0 if i == 0 else self._ends[i - 1]
                return Subset(self, range(start, self._ends[i]))
        raise KeyError(f"Trajectory {trajectory_id:04d} is not in this dataset.")

    @staticmethod
    def _read_eps(in_file):
        """Read explicit self coefficients; cross interactions use geometric mixing."""
        eps = {}
        for line in in_file.read_text().splitlines():
            fields = line.split("#", 1)[0].split()
            if not fields or fields[0] != "pair_coeff":
                continue
            if (len(fields) < 4 or not fields[1].isdigit()
                    or fields[1] != fields[2] or not 1 <= int(fields[1]) <= 10):
                raise ValueError(
                    f"Expected explicit pair_coeff i i epsilon for types 1..10 in {in_file}"
                )
            value = float(fields[3])
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Epsilons must be finite and positive in {in_file}")
            eps[int(fields[1])] = value

        missing = set(range(1, 11)) - eps.keys()
        if missing:
            raise ValueError(f"Missing pair_coeff for types {sorted(missing)} in {in_file}")
        return tuple(eps[i] for i in range(1, 11))

    def _index_retained_frames(self, dump_file):
        offsets = []
        excluded = 0
        offdiag_min = np.inf
        distance_max = -np.inf
        offdiag_mask = ~np.eye(10, dtype=bool)

        with dump_file.open("rb") as dump:
            while True:
                offset = dump.tell()
                line = dump.readline()
                if not line:
                    break
                if line.strip() != b"ITEM: TIMESTEP":
                    if (not line.endswith(b"\n")
                            and b"ITEM: TIMESTEP".startswith(line.strip())):
                        warnings.warn(
                            f"Skipping truncated final frame header in {dump_file} "
                            f"at byte offset {offset}.",
                            stacklevel=2,
                        )
                        break
                    warnings.warn(
                        f"Skipping data outside a frame in {dump_file} at byte offset "
                        f"{offset}: expected ITEM: TIMESTEP, got {line.strip()[:100]!r}.",
                        stacklevel=2,
                    )
                    while line and b"ITEM: TIMESTEP" not in line:
                        line = dump.readline()
                    if line:
                        dump.seek(line.index(b"ITEM: TIMESTEP") - len(line), 1)
                    continue

                header = []
                for _ in range(8):
                    line = dump.readline()
                    marker = line.find(b"ITEM: TIMESTEP")
                    if marker >= 0:
                        warnings.warn(
                            f"Skipping incomplete frame header in {dump_file} "
                            f"at byte offset {offset}, timestep "
                            f"{header[0].strip() if header else b'unknown'!r}: "
                            f"next timestep marker encountered in {line.strip()!r}.",
                            stacklevel=2,
                        )
                        dump.seek(marker - len(line), 1)
                        break
                    header.append(line)
                if len(header) < 8:
                    continue
                if any(not line.endswith(b"\n") for line in header):
                    warnings.warn(
                        f"Skipping truncated final frame header in {dump_file} "
                        f"at byte offset {offset}, timestep {header[0].strip()!r}.",
                        stacklevel=2,
                    )
                    break
                step = int(header[0])
                if header[1].strip() != b"ITEM: NUMBER OF ATOMS":
                    raise ValueError(
                        f"Expected ITEM: NUMBER OF ATOMS in {dump_file} at step {step}, "
                        f"frame byte offset {offset}; got {header[1].strip()!r}"
                    )
                number_of_atoms = int(header[2])

                box_header = header[3].decode().split()
                if box_header[:3] != ["ITEM:", "BOX", "BOUNDS"] or len(box_header) != 6:
                    raise ValueError("Only orthogonal simulation boxes are supported.")
                periodic = np.array([flag == "pp" for flag in box_header[3:5]])
                bounds = np.array([
                    list(map(float, line.split())) for line in header[4:7]
                ])
                lengths = bounds[:2, 1] - bounds[:2, 0]

                columns = header[7].decode().split()[2:]
                id_column = columns.index("id")
                type_column = columns.index("type")
                xy_columns = [columns.index(axis) for axis in ("x", "y")]

                need_positions = step >= self.start_step
                positions = {} if need_positions else None

                complete = True
                for atom_index in range(number_of_atoms):
                    line = dump.readline()
                    fields = line.split()
                    if (len(fields) != len(columns) or fields[0] == b"ITEM:"
                            or b"ITEM: TIMESTEP" in line):
                        warnings.warn(
                            f"Skipping incomplete atom block in {dump_file} at step {step}: "
                            f"expected {number_of_atoms} rows with {len(columns)} columns, "
                            f"read {atom_index} complete rows.",
                            stacklevel=2,
                        )
                        # Resume at the next frame, keeping its header unread.
                        while line and b"ITEM: TIMESTEP" not in line:
                            line = dump.readline()
                        if line:
                            dump.seek(line.index(b"ITEM: TIMESTEP") - len(line), 1)
                        complete = False
                        break
                    if need_positions:
                        atom_id = int(fields[id_column])
                        if 1 <= atom_id <= 10:
                            if int(fields[type_column]) != atom_id:
                                raise ValueError(
                                    f"Atom type must equal atom ID 1..10 in {dump_file} at step {step}"
                                )
                            positions[atom_id] = [
                                float(fields[column]) for column in xy_columns
                            ]

                if not complete or step < self.start_step:
                    continue
                if len(positions) < 10:
                    raise ValueError(f"Expected atom IDs 1..10 in {dump_file} at step {step}")

                ordered = np.array([positions[i] for i in ATOM_IDS], dtype=float)

                distances = self._distance_matrix(ordered, lengths, periodic)
                if not (distances[:5, 5:] <= self.contact_cutoff).any():
                    excluded += 1
                    continue
                offdiag_min = min(offdiag_min, float(distances[offdiag_mask].min()))
                distance_max = max(distance_max, float(distances.max()))
                offsets.append(offset)

        return np.asarray(offsets, dtype=np.uint64), offdiag_min, distance_max, excluded

    @staticmethod
    def _read_frame(dump_file, offset):
        with dump_file.open("rb") as dump:
            dump.seek(offset)
            if dump.readline().strip() != b"ITEM: TIMESTEP":
                raise ValueError(f"Bad frame offset {offset} in {dump_file}")
            step = int(dump.readline())

            dump.readline()  # ITEM: NUMBER OF ATOMS
            number_of_atoms = int(dump.readline())

            box_header = dump.readline().decode().split()
            if box_header[:3] != ["ITEM:", "BOX", "BOUNDS"] or len(box_header) != 6:
                raise ValueError("Only orthogonal simulation boxes are supported.")
            periodic = np.array([flag == "pp" for flag in box_header[3:5]])
            bounds = np.array([
                list(map(float, dump.readline().split())) for _ in range(3)
            ])
            lengths = bounds[:2, 1] - bounds[:2, 0]

            columns = dump.readline().decode().split()[2:]
            id_column = columns.index("id")
            type_column = columns.index("type")
            xy_columns = [columns.index(axis) for axis in ("x", "y")]

            positions = np.empty((10, 2), dtype=float)
            seen = np.zeros(10, dtype=bool)
            for atom_index in range(number_of_atoms):
                fields = dump.readline().split()
                if len(fields) != len(columns) or fields[0] == b"ITEM:":
                    raise ValueError(
                        f"Incomplete atom block in {dump_file} at step {step}: "
                        f"expected {number_of_atoms} rows with {len(columns)} columns, "
                        f"read {atom_index} complete rows."
                    )
                atom_id = int(fields[id_column])
                if 1 <= atom_id <= 10:
                    if int(fields[type_column]) != atom_id:
                        raise ValueError(
                            f"Atom type must equal atom ID 1..10 in {dump_file} at step {step}"
                        )
                    positions[atom_id - 1] = [
                        float(fields[column]) for column in xy_columns
                    ]
                    seen[atom_id - 1] = True

            if not seen.all():
                raise ValueError(f"Expected atom IDs 1..10 in {dump_file} at step {step}")

        return step, positions, lengths, periodic

    @staticmethod
    def _distance_matrix(positions, lengths, periodic):
        delta = positions[:, None, :] - positions[None, :, :]
        delta[..., periodic] -= lengths[periodic] * np.rint(
            delta[..., periodic] / lengths[periodic]
        )
        return np.sqrt(np.sum(delta * delta, axis=-1))

