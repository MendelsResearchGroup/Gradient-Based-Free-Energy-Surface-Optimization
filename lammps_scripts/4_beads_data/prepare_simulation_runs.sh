#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"
ROOT="$(cd "$ROOT" && pwd)"

GENERATOR="$ROOT/gener.py"
DATA_FILE="$ROOT/data.polyLJ"

if [[ ! -f "$GENERATOR" ]]; then
    echo "Error: generator not found: $GENERATOR"
    exit 1
fi

if [[ ! -f "$DATA_FILE" ]]; then
    echo "Error: data file not found: $DATA_FILE"
    exit 1
fi

for dir in "$ROOT"/*/; do
    [[ -d "$dir" ]] || continue

    echo "Processing: $dir"

    cp "$DATA_FILE" "$dir/data.polyLJ"

    (
        cd "$dir"
        python3 "$GENERATOR" .
    )
done

echo "Done."
