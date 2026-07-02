# Source this from this directory:
#   source env.sh
# Auto-derives PI0_ROOT from the location of this file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PI0_ROOT="$SCRIPT_DIR"
export PI0_DATA="$PI0_ROOT/data"
export PI0_CKPT="$PI0_ROOT/checkpoints"

# Python interpreter (override with PYTHON=... if needed)
export PI0_PYTHON="${PYTHON:-python3}"

# Default GPU (override per-script with CUDA_VISIBLE_DEVICES=...)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Make code/lib importable
# actprobe-public top-level shared lib
ACTPROBE_LIB_SHARED="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$ACTPROBE_LIB_SHARED:$PI0_ROOT/code:$PYTHONPATH"

echo "PI0_ROOT = $PI0_ROOT"
