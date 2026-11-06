#!/usr/bin/env bash
set -euo pipefail

root="$(pwd)"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <lammps_command_or_path> [args...]" >&2
  echo "Example: $0 lmp -in in.in" >&2
  echo "Example: $0 /path/to/lmp_mpi -in in.in" >&2
  exit 1
fi

found=0

for dir in "$root"/[0-9][0-9][0-9]/; do
  [[ -d "$dir" ]] || continue
  found=1

  echo "Running in: $dir"
  (
    cd "$dir"
    "$@"
  )
done

if [[ "$found" -eq 0 ]]; then
  echo "No three-digit trajectory folders found under $root" >&2
fi
