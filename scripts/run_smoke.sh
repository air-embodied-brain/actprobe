#!/usr/bin/env bash
# Smoke test for actprobe-public: verifies all 5 benchmarks can run end-to-end.
#
# Strategy:
#   1. Symlink data/ + checkpoints/ from $WORK_ROOT/<bm>/ into public
#   2. For each benchmark: source env.sh, import-check, run STAC allseen seed 0
#   3. Print compact PASS/FAIL summary
#   4. (Optional) clean symlinks with --cleanup
#
# Usage:
#   bash scripts/run_smoke.sh           # run smoke test, leave symlinks
#   bash scripts/run_smoke.sh --cleanup # only remove existing symlinks
#
# Exit code 0 if all benchmarks pass; 1 if any fail.

set -u

PUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Set WORK_ROOT to the directory holding each benchmark's data/ and checkpoints/.
WORK_ROOT="${WORK_ROOT:-/path/to/actprobe_data}"

# (pub_name : work_name) mapping
declare -A SRC_NAME=(
  [groot_robocasa]=groot_robocasa
  [pi0_libero]=pi0_libero
  [openvla_libero]=openvla_libero
  [pi05_robocasa]=pi05_robocasa
  [pi05_robocasa_multistage]=pi05_robocasa_multistage
)

# Each benchmark: which method to smoke-test with (fastest reliable choice)
declare -A SMOKE_METHOD=(
  [groot_robocasa]=stac
  [pi0_libero]=actprobe
  [openvla_libero]=actprobe
  [pi05_robocasa]=actprobe
  [pi05_robocasa_multistage]=actprobe
)

PY="${PYTHON:-python3}"

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup mode
# ─────────────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--cleanup" ]]; then
  echo "=== Cleanup mode: removing data/ and checkpoints/ symlinks ==="
  for bm in "${!SRC_NAME[@]}"; do
    for sub in data checkpoints; do
      tgt="$PUB_ROOT/benchmarks/$bm/$sub"
      if [[ -L "$tgt" ]]; then
        rm "$tgt" && echo "  removed: $bm/$sub"
      fi
    done
  done
  exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Symlink data/checkpoints
# ─────────────────────────────────────────────────────────────────────────────
echo "===================================================================="
echo "  Step 1: Linking data/ and checkpoints/ from $WORK_ROOT"
echo "===================================================================="
for bm in "${!SRC_NAME[@]}"; do
  src_name="${SRC_NAME[$bm]}"
  src="$WORK_ROOT/$src_name"
  dst="$PUB_ROOT/benchmarks/$bm"

  for sub in data checkpoints; do
    src_path="$src/$sub"
    dst_path="$dst/$sub"
    if [[ -e "$dst_path" || -L "$dst_path" ]]; then
      rm -f "$dst_path"
    fi
    if [[ -d "$src_path" ]]; then
      ln -s "$src_path" "$dst_path"
      echo "  ✓ $bm/$sub → $src_path"
    else
      echo "  ✗ $bm/$sub: source missing ($src_path)"
    fi
  done
done

# ─────────────────────────────────────────────────────────────────────────────
# Step 2+3: Per-benchmark smoke test
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "===================================================================="
echo "  Step 2+3: Import-check + 1-method eval per benchmark"
echo "===================================================================="

declare -A RESULTS
N_PASS=0
N_FAIL=0

for bm in "${!SRC_NAME[@]}"; do
  echo ""
  echo "--- $bm ---"
  cd "$PUB_ROOT/benchmarks/$bm"

  # source env.sh
  set +u
  source env.sh > /tmp/_smoke_envsh 2>&1
  rc=$?
  set -u
  if [[ $rc -ne 0 ]]; then
    echo "  [env.sh] FAIL"; cat /tmp/_smoke_envsh
    RESULTS[$bm]="FAIL: env.sh"; N_FAIL=$((N_FAIL+1)); continue
  fi
  echo "  [env.sh]    OK"

  # import check
  imp=$($PY -c "
import sys; sys.path.insert(0, 'code')
from lib import data, splits, metrics
" 2>&1)
  if [[ -n "$imp" ]]; then
    echo "  [import]    FAIL: $imp"
    RESULTS[$bm]="FAIL: import"; N_FAIL=$((N_FAIL+1)); continue
  fi
  echo "  [import]    OK"

  # eval main_table with smoke method
  meth="${SMOKE_METHOD[$bm]}"
  out_json="/tmp/smoke_${bm}.json"
  ev_log="/tmp/smoke_${bm}.log"
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
    timeout 300 $PY code/eval/eval_main_table.py \
      --methods $meth --splits allseen --seeds 0 \
      --out "$out_json" > "$ev_log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    tail -5 "$ev_log"
    echo "  [eval $meth] FAIL (rc=$rc)"
    RESULTS[$bm]="FAIL: eval $meth"; N_FAIL=$((N_FAIL+1)); continue
  fi
  qauc=$(grep -oE "$meth\s+allseen\s+[0-9.]+" "$ev_log" | awk '{print $NF}' | head -1)
  echo "  [eval $meth] OK  q-AUC=$qauc"
  RESULTS[$bm]="PASS ($meth q-AUC=$qauc)"; N_PASS=$((N_PASS+1))
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "===================================================================="
echo "  SUMMARY  ($N_PASS pass / $N_FAIL fail / ${#SRC_NAME[@]} total)"
echo "===================================================================="
for bm in groot_robocasa pi0_libero openvla_libero pi05_robocasa pi05_robocasa_multistage; do
  printf "  %-30s %s\n" "$bm" "${RESULTS[$bm]:-not-run}"
done
echo ""
echo "Logs: /tmp/smoke_<benchmark>.{json,log}"
echo "Cleanup: bash scripts/run_smoke.sh --cleanup"

[[ $N_FAIL -eq 0 ]] && exit 0 || exit 1
