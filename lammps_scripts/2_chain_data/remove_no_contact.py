from itertools import islice
from math import isfinite
from pathlib import Path
from shutil import rmtree
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parent
CONTACT_CUTOFF = 2.2
MIN_FRAMES = 2048


def has_contact(frame):
    """Validate one frame in the local 10-atom dump format and check contact."""
    headers = {
        0: b"ITEM: TIMESTEP",
        2: b"ITEM: NUMBER OF ATOMS",
        3: b"10",
        4: b"ITEM: BOX BOUNDS pp pp pp",
        8: b"ITEM: ATOMS id type x y z vx vy vz",
    }
    if len(frame) != 19:
        raise ValueError("Incomplete frame")
    if any(frame[i].strip() != expected for i, expected in headers.items()):
        raise ValueError("Unexpected dump format")
    int(frame[1])

    box = []
    for line in frame[5:8]:
        lo, hi = map(float, line.split())
        length = hi - lo
        if not all(map(isfinite, (lo, hi, length))) or length <= 0:
            raise ValueError("Invalid periodic box")
        box.append(length)

    points = {}
    for line in frame[9:]:
        fields = line.split()
        if len(fields) != 8:
            raise ValueError("Incomplete atom record")
        atom_id = int(fields[0])
        x, y, z = map(float, fields[2:5])
        if not all(map(isfinite, (x, y, z))) or z != 0:
            raise ValueError("Expected finite 2D coordinates")
        points[atom_id] = (x, y)
    if set(points) != set(range(1, 11)):
        raise ValueError("Expected atom IDs 1--10 exactly once")

    lx, ly = box[:2]
    for i in range(1, 6):
        for j in range(6, 11):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dx -= lx * round(dx / lx)
            dy -= ly * round(dy / ly)
            if dx * dx + dy * dy < CONTACT_CUTOFF ** 2:
                return True
    return False


def filter_trajectory(path):
    """Replace a validated dump only if at least MIN_FRAMES contact frames remain."""
    total = kept = 0
    output = NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".contact-", delete=False)
    temporary = Path(output.name)
    try:
        with output, path.open("rb") as source:
            while frame := list(islice(source, 19)):
                total += 1
                try:
                    contact = has_contact(frame)
                except ValueError as error:
                    raise ValueError(f"{path}, frame {total}: {error}") from error
                if contact:
                    output.writelines(frame)
                    kept += 1
        if kept >= MIN_FRAMES:
            temporary.chmod(path.stat().st_mode)
            temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return total, kept


def main():
    directories = sorted(
        path for path in ROOT.glob("[0-9][0-9][0-9][0-9]") if path.is_dir()
    )
    if not directories:
        raise FileNotFoundError(f"No numbered run directories in {ROOT}")
    # Check all paths before modifying any runs; do not follow linked data.
    for directory in directories:
        path = directory / "dump.lammpstrj"
        if directory.is_symlink() or path.is_symlink():
            raise ValueError(f"Expected a regular run directory and dump: {directory}")
        if not path.is_file():
            raise FileNotFoundError(path)

    remaining = []
    total_frames = kept_frames = 0
    for directory in directories:
        total, kept = filter_trajectory(directory / "dump.lammpstrj")
        total_frames += total
        if kept >= MIN_FRAMES:
            kept_frames += kept
            remaining.append(directory)
            print(f"{directory.name}: kept {kept}/{total} frames", flush=True)
        else:
            rmtree(directory)
            print(
                f"{directory.name}: deleted ({kept}/{total} contact frames, "
                f"fewer than {MIN_FRAMES})",
                flush=True,
            )

    # With positive, four-digit names, ascending moves only fill earlier gaps.
    for number, directory in enumerate(remaining, 1):
        target = ROOT / f"{number:04d}"
        if directory != target:
            if target.exists() or target.is_symlink():
                raise FileExistsError(f"Cannot rename {directory}: {target} already exists")
            directory.rename(target)
            print(f"{directory.name} -> {target.name}", flush=True)

    print(
        f"Removed {total_frames - kept_frames} frames and "
        f"{len(directories) - len(remaining)} directories; "
        f"kept {kept_frames} frames in {len(remaining)} directories."
    )


if __name__ == "__main__":
    main()
