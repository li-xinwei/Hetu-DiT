#!/usr/bin/env bash
# Drive the warm-pool eval matrix.
#
# Reads cells from a YAML config (--matrix), generates a workload trace per
# cell, runs it (sim mode = eval_sim.py; real mode = TODO Phase 5 K8s wiring),
# captures cstrace.log + summary.txt into a run subdir, then aggregates +
# renders the report.
#
# Usage:
#   ./scripts/run_eval_matrix.sh --matrix scripts/eval_matrix.yaml \
#       --mode sim --set small --out results/eval-$(date +%s)
#
# The YAML parsing is intentionally minimal: pure-bash, no python_yaml dep on
# the orchestrator side. Each cell line under the chosen set must look like:
#   - {pattern: burst, n: 8, head_size: 1, warm_replicas: 0}
# Trailing comments tolerated, ordering preserved.

set -euo pipefail

MATRIX="scripts/eval_matrix.yaml"
MODE="sim"
SET_NAME="small"
OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --matrix) MATRIX="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --set) SET_NAME="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="results/eval-$(date +%Y%m%d-%H%M%S)"
fi

if [[ "$MODE" != "sim" && "$MODE" != "real" ]]; then
  echo "--mode must be sim or real (got: $MODE)" >&2
  exit 2
fi

if [[ ! -f "$MATRIX" ]]; then
  echo "matrix file not found: $MATRIX" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
echo "eval matrix: $MATRIX [set=$SET_NAME mode=$MODE] -> $OUT_DIR"

# Extract cell lines from YAML using awk: find the section header `${SET_NAME}:`,
# then collect indented `- {...}` lines until the next top-level key.
CELLS_FILE="$OUT_DIR/cells.txt"
awk -v set_name="$SET_NAME" '
  /^[a-zA-Z_]+:/ {
    in_set = ($1 == set_name ":")
    next
  }
  in_set && /^[[:space:]]*-[[:space:]]*{/ {
    sub(/^[[:space:]]*-[[:space:]]*/, "", $0)
    print $0
  }
' "$MATRIX" > "$CELLS_FILE"

N_CELLS=$(wc -l < "$CELLS_FILE" | tr -d ' ')
if [[ "$N_CELLS" -eq 0 ]]; then
  echo "no cells parsed for set=$SET_NAME from $MATRIX" >&2
  exit 2
fi
echo "  parsed $N_CELLS cells"

# Helper: extract a single key from a {a: b, c: d} line. Returns "" if missing.
extract_field() {
  local line="$1" key="$2"
  python3 - "$line" "$key" <<'PY'
import sys, re, json
line, key = sys.argv[1], sys.argv[2]
# Convert YAML inline mapping {a: b, c: d} into a JSON object (string-safe enough for our values).
stripped = line.strip().lstrip("{").rstrip("}")
pairs = {}
for item in stripped.split(","):
    if ":" not in item:
        continue
    k, _, v = item.partition(":")
    pairs[k.strip()] = v.strip()
print(pairs.get(key, ""))
PY
}

CELL_IDX=0
while IFS= read -r raw_line; do
  CELL_IDX=$((CELL_IDX + 1))
  pattern=$(extract_field "$raw_line" pattern)
  n=$(extract_field "$raw_line" n)
  rate=$(extract_field "$raw_line" rate)
  duration=$(extract_field "$raw_line" duration)
  head_size=$(extract_field "$raw_line" head_size)
  warm_replicas=$(extract_field "$raw_line" warm_replicas)

  : "${head_size:=1}"
  : "${warm_replicas:=0}"
  : "${pattern:=cold}"

  cell_id=$(printf "cell-%02d-%s-h%s-w%s" "$CELL_IDX" "$pattern" "$head_size" "$warm_replicas")
  cell_dir="$OUT_DIR/$cell_id"
  trace_file="$cell_dir/workload.trace"
  mkdir -p "$cell_dir"

  echo ""
  echo "[$CELL_IDX/$N_CELLS] $cell_id (pattern=$pattern head=$head_size warm=$warm_replicas)"

  # Workload generation arg assembly.
  gen_args=(--pattern "$pattern" --seed 0 --model sd3 --out "$trace_file")
  if [[ -n "$n" && "$pattern" == "burst" ]]; then gen_args+=(--n "$n"); fi
  if [[ -n "$rate" ]]; then gen_args+=(--rate "$rate"); fi
  if [[ -n "$duration" ]]; then gen_args+=(--duration "$duration"); fi

  python3 scripts/workload_gen.py "${gen_args[@]}"

  # Run either simulator or real K8s lifecycle.
  if [[ "$MODE" == "sim" ]]; then
    python3 scripts/eval_sim.py \
      --workload "$trace_file" \
      --out "$cell_dir" \
      --head-size "$head_size" \
      --warm-replicas "$warm_replicas" \
      --pattern "$pattern" \
      --model sd3 \
      --seed 0
  else
    bash scripts/runpod_drive_cell.sh \
      --workload "$trace_file" \
      --out "$cell_dir" \
      --head-size "$head_size" \
      --warm-replicas "$warm_replicas" \
      --pattern "$pattern" \
      --model sd3 \
      --seed 0
  fi
done < "$CELLS_FILE"

echo ""
echo "all cells done; aggregating..."
python3 scripts/aggregate_runs.py --root "$OUT_DIR"

echo ""
echo "generating report..."
python3 scripts/gen_report.py --root "$OUT_DIR"

echo ""
echo "done -> $OUT_DIR/report/report.md"
