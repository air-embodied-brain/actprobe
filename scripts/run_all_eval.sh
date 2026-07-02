#!/usr/bin/env bash
# Run main eval for all benchmarks. Prints numbers to stdout.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for bm in $ROOT/benchmarks/*/; do
  name=$(basename $bm)
  if [ -f "$bm/code/eval/eval_main_table.py" ]; then
    echo "================================"
    echo "  $name"
    echo "================================"
    cd $bm
    source env.sh 2>/dev/null
    python code/eval/eval_main_table.py --out /tmp/${name}_main_table.json
    cd $ROOT
  fi
done
