#!/usr/bin/env bash
# Train the ActProbe probe (paper main method) with N random seeds.
#
# Each seed is a fresh random train/val/test split AND a freshly trained probe;
# eval auto-detects whatever seed checkpoints are present, so re-running this
# gives a new independent sample for the mean +/- std. Override the count with
# N_SEEDS=5 bash scripts/train_all.sh
#
# (Baseline detectors have their own train_*.py under code/train/ and additionally
#  need the hidden-state features noted in the README; they are not trained here.)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
source "$ROOT/env.sh"

N_SEEDS="${N_SEEDS:-3}"
SEEDS=()
while [ "${#SEEDS[@]}" -lt "$N_SEEDS" ]; do SEEDS+=( "$RANDOM" ); done

echo "=== Train ActProbe | random seeds: ${SEEDS[*]} ==="
"$GROOT_PYTHON" "$ROOT/code/train/train_actprobe.py" --seeds "${SEEDS[@]}"

echo ""
echo "Done. checkpoints -> checkpoints/actprobe/  (eval auto-detects these seeds)"
