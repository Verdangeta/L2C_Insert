#!/usr/bin/env bash
set -euo pipefail

# Generic wrapper around run_layout_experiments.py.
# Keeps shell UX stable while supporting multiple instance layouts.

ACTIVE_CONFIG="${ACTIVE_CONFIG:-explosion_2k_default}"
REGENERATE_INSTANCES=0
EXTRA_ARGS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      ACTIVE_CONFIG="${2:-}"
      shift 2
      ;;
    --regenerate)
      REGENERATE_INSTANCES="${2:-0}"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${CONDA_DEFAULT_ENV:-}" != "TDA_L2C" ]] && command -v conda >/dev/null 2>&1; then
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate TDA_L2C
  set -u
fi

echo "Using config preset: $ACTIVE_CONFIG"
python -u "$SCRIPT_DIR/run_layout_experiments.py" \
  --config "$ACTIVE_CONFIG" \
  --regenerate "$REGENERATE_INSTANCES" \
  "${EXTRA_ARGS[@]}"
