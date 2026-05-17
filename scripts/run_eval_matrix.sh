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

# Helper: extract a single key from a {a: b, c: d, seeds: [0,1,2]} line.
# Returns "" if missing. Lists are returned as comma-separated strings.
extract_field() {
  local line="$1" key="$2"
  python3 - "$line" "$key" <<'PY'
import sys, re
line, key = sys.argv[1], sys.argv[2]
stripped = line.strip().lstrip("{").rstrip("}")
# Split top-level commas (handling nested brackets for lists).
pairs = {}
depth = 0
buf = ""
parts = []
for ch in stripped:
    if ch in "[{":
        depth += 1
    elif ch in "]}":
        depth -= 1
    if ch == "," and depth == 0:
        parts.append(buf)
        buf = ""
    else:
        buf += ch
if buf:
    parts.append(buf)
for item in parts:
    if ":" not in item:
        continue
    k, _, v = item.partition(":")
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        # list literal -> comma-joined string
        v = v[1:-1].replace(" ", "")
    pairs[k.strip()] = v
print(pairs.get(key, ""))
PY
}

CELL_IDX=0
TOTAL_RUNS=0
while IFS= read -r raw_line; do
  CELL_IDX=$((CELL_IDX + 1))
  pattern=$(extract_field "$raw_line" pattern)
  n=$(extract_field "$raw_line" n)
  rate=$(extract_field "$raw_line" rate)
  duration=$(extract_field "$raw_line" duration)
  head_size=$(extract_field "$raw_line" head_size)
  warm_replicas=$(extract_field "$raw_line" warm_replicas)
  seeds=$(extract_field "$raw_line" seeds)

  : "${head_size:=1}"
  : "${warm_replicas:=0}"
  : "${pattern:=cold}"
  : "${seeds:=0}"

  echo ""
  echo "[$CELL_IDX/$N_CELLS] pattern=$pattern head=$head_size warm=$warm_replicas seeds=$seeds"

  IFS=',' read -ra SEED_ARRAY <<< "$seeds"
  for seed in "${SEED_ARRAY[@]}"; do
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    run_id=$(printf "cell-%02d-%s-h%s-w%s-s%s" "$CELL_IDX" "$pattern" "$head_size" "$warm_replicas" "$seed")
    run_dir="$OUT_DIR/$run_id"
    trace_file="$run_dir/workload.trace"
    mkdir -p "$run_dir"

    # Build a rich pattern tag so aggregator can distinguish e.g. poisson@λ=0.5
    # from poisson@λ=1.0 when grouping cells. Tags become workload_pattern in
    # summary.txt and aggregator groups by this exact string.
    rich_pattern="$pattern"
    if [[ -n "$rate" ]]; then rich_pattern="${rich_pattern}-r${rate}"; fi
    if [[ -n "$n" && "$pattern" == "burst" ]]; then rich_pattern="${rich_pattern}-n${n}"; fi

    # Workload generation arg assembly.
    gen_args=(--pattern "$pattern" --seed "$seed" --model sd3 --out "$trace_file")
    if [[ -n "$n" && "$pattern" == "burst" ]]; then gen_args+=(--n "$n"); fi
    if [[ -n "$rate" ]]; then gen_args+=(--rate "$rate"); fi
    if [[ -n "$duration" ]]; then gen_args+=(--duration "$duration"); fi

    python3 scripts/workload_gen.py "${gen_args[@]}" > "$run_dir/workload_gen.log" 2>&1

    if [[ "$MODE" == "sim" ]]; then
      python3 scripts/eval_sim.py \
        --workload "$trace_file" \
        --out "$run_dir" \
        --head-size "$head_size" \
        --warm-replicas "$warm_replicas" \
        --pattern "$rich_pattern" \
        --model sd3 \
        --seed "$seed" > "$run_dir/sim.log" 2>&1
    else
      bash scripts/runpod_drive_cell.sh \
        --workload "$trace_file" \
        --out "$run_dir" \
        --head-size "$head_size" \
        --warm-replicas "$warm_replicas" \
        --pattern "$rich_pattern" \
        --model sd3 \
        --seed "$seed"
    fi
  done
done < "$CELLS_FILE"
echo ""
echo "total runs executed: $TOTAL_RUNS"

echo ""
echo "all cells done; aggregating..."
python3 scripts/aggregate_runs.py --root "$OUT_DIR"

echo ""
echo "generating report..."
python3 scripts/gen_report.py --root "$OUT_DIR"

echo ""
echo "done -> $OUT_DIR/report/report.md"
