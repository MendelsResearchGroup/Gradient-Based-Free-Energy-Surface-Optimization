#!/usr/bin/env bash
set -euo pipefail

base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

for directory in "$base"/*/; do
    cp -p -- "$base/data.polyLJ" "$directory/"
    echo "Generating $directory/in.in"
    (cd -- "$directory" && python3 "$base/gener.py")
done
