#!/usr/bin/env bash
# Run the full adapter × scenario matrix against a local LM Studio endpoint.
#
# Usage:
#   ./harness/run_matrix.sh                    # default: 20 runs per cell
#   ./harness/run_matrix.sh --runs 5           # smaller sample
#   ./harness/run_matrix.sh --runs 1 --no-capture   # quickest smoke
#
# Must be run from the repo root with .venv activated or discoverable.

set -euo pipefail

RUNS=20
CAPTURE="--capture-traces"
API_BASE="${LMSTUDIO_BASE:-http://127.0.0.1:1234/v1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2 ;;
    --no-capture) CAPTURE=""; shift ;;
    -h|--help)
      sed -n '2,10p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

cd "$(dirname "$0")/.."

if [[ -z "${VIRTUAL_ENV:-}" ]] && [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo "Checking LM Studio at ${API_BASE}..."
if ! curl -s --max-time 3 "${API_BASE}/models" | grep -q '"id"'; then
  echo "ERROR: no model loaded at ${API_BASE}." >&2
  echo "Load Qwen 3.5 in LM Studio, then re-run." >&2
  exit 1
fi

ADAPTERS=(chat json qwen35)
SCENARIOS=(s1 s3 s10)
TOTAL=$(( ${#ADAPTERS[@]} * ${#SCENARIOS[@]} ))
CELL=0
START=$(date +%s)

for ADAPTER in "${ADAPTERS[@]}"; do
  for SCENARIO in "${SCENARIOS[@]}"; do
    CELL=$((CELL + 1))
    echo ""
    echo "=== [${CELL}/${TOTAL}] adapter=${ADAPTER} scenario=${SCENARIO} runs=${RUNS} ==="
    python -m harness.run_eval \
      --adapter "${ADAPTER}" \
      --scenario "${SCENARIO}" \
      --runs "${RUNS}" \
      ${CAPTURE}
  done
done

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "=== matrix complete in ${ELAPSED}s; writing summary ==="

python -m harness.analyze > harness/results/summary.md
echo ""
cat harness/results/summary.md
echo ""
echo "Summary saved to harness/results/summary.md"
