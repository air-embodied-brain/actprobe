# Source this from this directory:
#   source env.sh
# Auto-derives GROOT_ROOT from the location of this file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GROOT_ROOT="$SCRIPT_DIR"
export GROOT_DATA="$GROOT_ROOT/data"
export GROOT_CKPT="$GROOT_ROOT/checkpoints"

# Python interpreter (override with PYTHON=... if needed)
export GROOT_PYTHON="${PYTHON:-python3}"

# Default GPU (override per-script with CUDA_VISIBLE_DEVICES=...)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Make code/lib importable
# actprobe-public top-level shared lib
ACTPROBE_LIB_SHARED="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$ACTPROBE_LIB_SHARED:$GROOT_ROOT/code:$PYTHONPATH"

echo "GROOT_ROOT = $GROOT_ROOT"
