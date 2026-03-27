#!/usr/bin/env bash
set -euo pipefail

# Wrapper around run_tsp_tests_experiments.py.
# Keeps the same UX as run_explosion_pair.sh with preset/config switching.

ACTIVE_CONFIG="${ACTIVE_CONFIG:-tsp_default}"
REGENERATE=0
BASELINE_ONLY=0
TASKS="${TASKS:-all}"
EXTRA_ARGS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      ACTIVE_CONFIG="${2:-}"
      shift 2
      ;;
    --tasks)
      TASKS="${2:-all}"
      shift 2
      ;;
    --regenerate)
      REGENERATE="${2:-0}"
      shift 2
      ;;
    --baseline-only)
      BASELINE_ONLY="${2:-1}"
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
echo "Tasks: $TASKS"
python -u "$SCRIPT_DIR/run_tsp_tests_experiments.py" \
  --config "$ACTIVE_CONFIG" \
  --tasks "$TASKS" \
  --regenerate "$REGENERATE" \
  --baseline-only "$BASELINE_ONLY" \
  "${EXTRA_ARGS[@]}"
