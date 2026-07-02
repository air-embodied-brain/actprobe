# Source this from this directory:
#   source env.sh
# Auto-derives OPENVLA_LIBERO_ROOT from the location of this file.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OPENVLA_LIBERO_ROOT="$SCRIPT_DIR"
export OPENVLA_LIBERO_DATA="$OPENVLA_LIBERO_ROOT/data"
export OPENVLA_LIBERO_CKPT="$OPENVLA_LIBERO_ROOT/checkpoints"

# Python interpreter (override with PYTHON=... if needed)
export OPENVLA_LIBERO_PYTHON="${PYTHON:-python3}"

# Default GPU (override per-script with CUDA_VISIBLE_DEVICES=...)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Make code/lib importable
# actprobe-public top-level shared lib
ACTPROBE_LIB_SHARED="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$ACTPROBE_LIB_SHARED:$OPENVLA_LIBERO_ROOT/code:$PYTHONPATH"

echo "OPENVLA_LIBERO_ROOT = $OPENVLA_LIBERO_ROOT"
