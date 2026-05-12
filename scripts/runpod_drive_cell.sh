#!/usr/bin/env bash
# Drive ONE eval cell against a real K3s/RunPod cluster.
#
# Lifecycle: mutate raycluster-runpod.yaml -> kubectl apply -> wait /readyz ->
# kubectl-exec the trace-replay client inside the head pod -> poll /metrics
# once after drain -> dump api.log + cstrace.log + summary.txt -> teardown.
#
# Called by scripts/run_eval_matrix.sh in --mode real. Designed to be
# resumable / idempotent: every cell tears down its own RC at the end.
#
# Required env / args:
#   --workload PATH           path to workload trace (must already exist on the pod, or scp'd in)
#   --out DIR                 output directory (will get cstrace.log + summary.txt + final_metrics.json)
#   --head-size INT           always 1 today
#   --warm-replicas INT
#   --pattern STR             workload_pattern tag for summary.txt
#   --model STR               sd3|flux|cogvideox|hunyuanvideo
#   --seed INT
#
# Optional env (with defaults):
#   REPO=/root/Hetu-DiT
#   K8S_NAMESPACE=default
#   TEMPLATE=${REPO}/k8s/raycluster-runpod.yaml
#   TIMEOUT_READY_S=600
#   TIMEOUT_TRACE_S=1800

set -euo pipefail

WORKLOAD=""
OUT_DIR=""
HEAD_SIZE=1
WARM_REPLICAS=0
PATTERN="unknown"
MODEL="sd3"
SEED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workload) WORKLOAD="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --head-size) HEAD_SIZE="$2"; shift 2 ;;
    --warm-replicas) WARM_REPLICAS="$2"; shift 2 ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

: "${REPO:=/root/Hetu-DiT}"
: "${K8S_NAMESPACE:=default}"
: "${TEMPLATE:=${REPO}/k8s/raycluster-runpod.yaml}"
: "${TIMEOUT_READY_S:=600}"
: "${TIMEOUT_TRACE_S:=1800}"

if [[ -z "$WORKLOAD" || -z "$OUT_DIR" ]]; then
  echo "usage: $0 --workload PATH --out DIR [--head-size N --warm-replicas N ...]" >&2
  exit 2
fi
if [[ ! -f "$WORKLOAD" ]]; then
  echo "ERROR: workload file not found: $WORKLOAD" >&2
  exit 2
fi
if [[ ! -f "$TEMPLATE" ]]; then
  echo "ERROR: K8s template not found: $TEMPLATE (override via TEMPLATE env)" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
YAML="$OUT_DIR/raycluster.yaml"

# --- Step 1: mutate RC yaml ---
python3 - "$TEMPLATE" "$YAML" "$WARM_REPLICAS" <<'PYEOF'
import sys, yaml, pathlib
src, dst, warm_replicas = sys.argv[1], sys.argv[2], int(sys.argv[3])
rc = yaml.safe_load(open(src))

def find_container(spec, name):
    return next(c for c in spec["template"]["spec"]["containers"] if c["name"] == name)

api = find_container(rc["spec"]["headGroupSpec"], "hetudit-api")
# Always enable L2 pool for eval — needed for warm/L2 hit_type classification.
if "--l2_pool_enabled" not in api.get("args", []):
    api.setdefault("args", []).append("--l2_pool_enabled")

# Toggle warm worker group size.
for wg in rc["spec"].get("workerGroupSpecs", []):
    if wg.get("groupName") == "gpu-workers-warm":
        wg["replicas"] = warm_replicas
        wg["minReplicas"] = warm_replicas
        wg["maxReplicas"] = max(warm_replicas, wg.get("maxReplicas", 1))

# Adjust HETUDIT_EXPECTED_GPUS / HETUDIT_MACHINE_NUMS for warm > 0.
if warm_replicas > 0:
    head_gpus = 8
    warm_gpus = 8 * warm_replicas
    total_gpus = head_gpus + warm_gpus
    for e in api.get("env", []):
        if e.get("name") == "HETUDIT_EXPECTED_GPUS":
            e["value"] = str(total_gpus)
        elif e.get("name") == "HETUDIT_MACHINE_NUMS":
            e["value"] = str(1 + warm_replicas)

pathlib.Path(dst).write_text(yaml.safe_dump(rc, sort_keys=False))
print(f"wrote {dst}")
PYEOF

# --- Step 2: apply + wait /readyz ---
echo "[$(date +%H:%M:%S)] applying RC (warm_replicas=$WARM_REPLICAS) ..."
T_APPLY=$(date +%s.%N)
kubectl apply -n "$K8S_NAMESPACE" -f "$YAML" >/dev/null

# Path-5 deadlock fix from run-runpod-paths.sh: publishNotReadyAddresses.
if [[ "$WARM_REPLICAS" -gt 0 ]]; then
    for _ in $(seq 1 30); do
        kubectl patch svc -n "$K8S_NAMESPACE" hetudit-head-svc --type=json \
            -p='[{"op":"add","path":"/spec/publishNotReadyAddresses","value":true}]' >/dev/null 2>&1 && break
        sleep 2
    done
fi

HEAD=""
for _ in $(seq 1 60); do
    HEAD=$(kubectl get pod -n "$K8S_NAMESPACE" \
        -l ray.io/cluster=hetudit,ray.io/node-type=head \
        -o jsonpath="{.items[0].metadata.name}" 2>/dev/null || true)
    [[ -n "$HEAD" ]] && break
    sleep 1
done
[[ -n "$HEAD" ]] || { echo "ERROR: head pod never appeared" >&2; exit 1; }

T_READY=""
for _ in $(seq 1 "$TIMEOUT_READY_S"); do
    STATUS=$(kubectl get pod -n "$K8S_NAMESPACE" "$HEAD" \
        -o jsonpath='{.status.containerStatuses[?(@.name=="hetudit-api")].ready}' 2>/dev/null || true)
    if [[ "$STATUS" == "true" ]]; then T_READY=$(date +%s.%N); break; fi
    sleep 1
done
if [[ -z "$T_READY" ]]; then
    echo "ERROR: /readyz never green within ${TIMEOUT_READY_S}s" >&2
    kubectl logs -n "$K8S_NAMESPACE" "$HEAD" -c hetudit-api > "$OUT_DIR/api.fail.log" 2>&1 || true
    kubectl delete -f "$YAML" --wait=false >/dev/null 2>&1 || true
    exit 2
fi
DT_READY=$(awk -v a="$T_APPLY" -v r="$T_READY" 'BEGIN{printf "%.2f", r-a}')
echo "[$(date +%H:%M:%S)] /readyz green after ${DT_READY}s on $HEAD"

# --- Step 3: copy workload into pod + drive trace via aiohttp client ---
kubectl cp "$WORKLOAD" "$K8S_NAMESPACE/$HEAD:/tmp/workload.trace" -c hetudit-api

T_TRACE_START=$(date +%s.%N)
# Trace replay client lives in examples/api_benchmark_example_with_trace.py inside the image.
kubectl exec -n "$K8S_NAMESPACE" "$HEAD" -c hetudit-api -- bash -c \
    "cd /workspace/Hetu-DiT && timeout ${TIMEOUT_TRACE_S} python3 examples/api_benchmark_example_with_trace.py --trace /tmp/workload.trace --url http://127.0.0.1:8000" \
    > "$OUT_DIR/client.log" 2>&1 || echo "WARN: trace client returned non-zero — partial results may exist"
T_TRACE_END=$(date +%s.%N)
DT_TRACE=$(awk -v a="$T_TRACE_START" -v d="$T_TRACE_END" 'BEGIN{printf "%.2f", d-a}')
echo "[$(date +%H:%M:%S)] trace replay finished in ${DT_TRACE}s"

# Grace period for last task to drain before /metrics snapshot.
sleep 5

# --- Step 4: snapshot /metrics ---
kubectl exec -n "$K8S_NAMESPACE" "$HEAD" -c hetudit-api -- \
    python3 -c "import urllib.request,sys; sys.stdout.write(urllib.request.urlopen('http://127.0.0.1:8000/metrics', timeout=10).read().decode())" \
    > "$OUT_DIR/final_metrics.json" 2>/dev/null || echo "{}" > "$OUT_DIR/final_metrics.json"

# --- Step 5: capture cstrace + logs ---
kubectl logs -n "$K8S_NAMESPACE" "$HEAD" -c hetudit-api > "$OUT_DIR/api.log" 2>&1 || true
kubectl logs -n "$K8S_NAMESPACE" "$HEAD" -c ray-head > "$OUT_DIR/ray-head.log" 2>&1 || true
if [[ "$WARM_REPLICAS" -gt 0 ]]; then
    for warm_pod in $(kubectl get pod -n "$K8S_NAMESPACE" \
        -l ray.io/cluster=hetudit,ray.io/group=gpu-workers-warm \
        -o jsonpath='{.items[*].metadata.name}' 2>/dev/null); do
        kubectl logs -n "$K8S_NAMESPACE" "$warm_pod" >> "$OUT_DIR/warm.log" 2>&1 || true
    done
fi
grep -hE '^\[CSTRACE\]' "$OUT_DIR/api.log" "$OUT_DIR/ray-head.log" "$OUT_DIR/warm.log" 2>/dev/null > "$OUT_DIR/cstrace.log" || true

# --- Step 6: extended summary.txt for aggregator ---
N_PLANNED=$(grep -c '^Request(' "$WORKLOAD" 2>/dev/null || echo 0)
WALL_CLOCK=$(awk -v a="$T_APPLY" -v d="$T_TRACE_END" 'BEGIN{printf "%.3f", d-a}')
POOL_SIZE=$((HEAD_SIZE + WARM_REPLICAS))
cat > "$OUT_DIR/summary.txt" <<SUM
workload_pattern=${PATTERN}
pool_size=${POOL_SIZE}
head_size=${HEAD_SIZE}
warm_replicas=${WARM_REPLICAS}
model=${MODEL}
n_requests_planned=${N_PLANNED}
seed=${SEED}
wall_clock_s=${WALL_CLOCK}
mode=real
apply_to_ready_s=${DT_READY}
trace_replay_s=${DT_TRACE}
head_pod=${HEAD}
SUM

# --- Step 7: teardown ---
kubectl delete -f "$YAML" --wait=false >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
    kubectl get pod -n "$K8S_NAMESPACE" "$HEAD" >/dev/null 2>&1 || break
    sleep 2
done

echo "[$(date +%H:%M:%S)] cell done -> $OUT_DIR"
