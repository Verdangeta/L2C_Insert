#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_explosion_pair.sh [--regenerate 0|1]
# Environment overrides (optional):
#   MODEL_PATH, PROBLEM_SIZES, NUM_INSTANCES, LAYOUT, NUM_CENTERS,
#   RANGE_MIN, RANGE_MAX, RATE, POOL_ROOT, POOL_NAME,
#   BASELINE_MODEL_NAME, ADV_MODEL_NAME, CUDA_DEVICE_NUM, RRC_BUDGET, RRC_RANGE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REGENERATE_INSTANCES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --regenerate)
      REGENERATE_INSTANCES="${2:-0}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: bash run_explosion_pair.sh [--regenerate 0|1]"
      exit 1
      ;;
  esac
done

# Activate conda env (as requested in workspace preferences)
if [[ "${CONDA_DEFAULT_ENV:-}" != "TDA_L2C" ]] && command -v conda >/dev/null 2>&1; then
  # conda activate hooks may reference unset vars (e.g. MKL_INTERFACE_LAYER),
  # so we temporarily disable nounset around activation.
  set +u
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate TDA_L2C
  set -u
fi

MODEL_PATH="${MODEL_PATH:-/trinity/home/alexander.mironenko/TDA_tsp/L2C_Insert/L2C_Insert/TSP/Test/result/pretrain/tsp_model.pt}"
PROBLEM_SIZES="${PROBLEM_SIZES:-500}"
NUM_INSTANCES="${NUM_INSTANCES:-50}"
LAYOUT="${LAYOUT:-explosion}"
NUM_CENTERS="${NUM_CENTERS:-6}"
RANGE_MIN="${RANGE_MIN:-0.1}"
RANGE_MAX="${RANGE_MAX:-0.5}"
RATE="${RATE:-10}"
POOL_ROOT="${POOL_ROOT:-./shared_instances}"
POOL_NAME="${POOL_NAME:-}"
BASELINE_MODEL_NAME="${BASELINE_MODEL_NAME:-baseline}"
ADV_MODEL_NAME="${ADV_MODEL_NAME:-advance_sampling_baseline}"
CUDA_DEVICE_NUM="${CUDA_DEVICE_NUM:-0}"
RRC_BUDGET="${RRC_BUDGET:-100}"
RRC_RANGE="${RRC_RANGE:-50}"

COMMON_ARGS=(
  --model_path "$MODEL_PATH"
  --problem_sizes "$PROBLEM_SIZES"
  --num_instances "$NUM_INSTANCES"
  --layout "$LAYOUT"
  --num_centers "$NUM_CENTERS"
  --range_min "$RANGE_MIN"
  --range_max "$RANGE_MAX"
  --rate "$RATE"
  --cuda_device_num "$CUDA_DEVICE_NUM"
  --RRC_budget "$RRC_BUDGET"
  --RRC_range "$RRC_RANGE"
  --instances_pool_root "$POOL_ROOT"
  --optimal_cost_method concorde
)
if [[ -n "$POOL_NAME" ]]; then
  COMMON_ARGS+=(--instances_pool_name "$POOL_NAME")
fi

echo "=== 1/3 Baseline run ==="
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_LOG_DIR="./pair_run_logs/${RUN_TS}"
mkdir -p "$RUN_LOG_DIR"
BASELINE_LOG="${RUN_LOG_DIR}/baseline.log"
python -u test_explosion.py \
  "${COMMON_ARGS[@]}" \
  --with_RTDL 0 \
  --use_rtdl_sampling 0 \
  --model_name "$BASELINE_MODEL_NAME" \
  --regenerate_instances "$REGENERATE_INSTANCES" | tee "$BASELINE_LOG"

BASELINE_DIR="$(awk -F': ' '/Results will be saved to:/ {print $2}' "$BASELINE_LOG" | tail -n1)"
if [[ -z "$BASELINE_DIR" ]]; then
  echo "Failed to detect baseline result folder from logs."
  exit 2
fi

echo "=== 2/3 Advanced sampling run ==="
ADV_LOG="${RUN_LOG_DIR}/advanced.log"
python -u test_explosion.py \
  "${COMMON_ARGS[@]}" \
  --with_RTDL 0 \
  --use_rtdl_sampling 1 \
  --model_name "$ADV_MODEL_NAME" \
  --regenerate_instances 0 | tee "$ADV_LOG"

ADV_DIR="$(awk -F': ' '/Results will be saved to:/ {print $2}' "$ADV_LOG" | tail -n1)"
if [[ -z "$ADV_DIR" ]]; then
  echo "Failed to detect advanced result folder from logs."
  exit 3
fi

echo "=== 3/3 Pair analysis ==="
python analyze_explosion_pair.py \
  --baseline_dir "$BASELINE_DIR" \
  --advanced_dir "$ADV_DIR"

echo
echo "Done."
echo "Baseline dir : $BASELINE_DIR"
echo "Advanced dir : $ADV_DIR"
echo "Compare dir  : $ADV_DIR/compare_with_baseline"
echo "Logs dir     : $RUN_LOG_DIR"

