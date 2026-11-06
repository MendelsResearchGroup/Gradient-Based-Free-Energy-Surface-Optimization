#!/usr/bin/env bash
set -euo pipefail

base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

for directory in "$base"/*/; do
    [[ -f "$directory/in.in" ]] || continue
    echo "Running LAMMPS in $directory"
    (cd -- "$directory" && bash "$base/lmp.sh")
done
