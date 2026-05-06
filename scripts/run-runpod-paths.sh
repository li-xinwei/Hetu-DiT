#!/usr/bin/env bash
# Run cold-start path matrix on RunPod K3s (single 8-GPU node).
#
# Paths:
#   path1-default          baseline, no D1, no L2
#   path2-nixl_broadcast   D1 broadcast (rank 0 fully loads, peers fetch via NIXL)
#   path3-nixl_pipelined   D1 pipelined (rank 0 streams blocks)
#   path4-l2_only          --l2_pool_enabled, no warm pool
#   path5-prewarm          --l2_pool_enabled + warm replicas=1 (PR3-lite dispatcher)
#
# Each path:
#   - kubectl apply RC variant (generated from /root/Hetu-DiT/k8s/raycluster-runpod.yaml)
#   - patch hetudit-head-svc with publishNotReadyAddresses=true (path 5 only — head pod
#     and warm pod are siblings in the same RayCluster, warm pod's wait-gcs-ready needs
#     to reach GCS via svc before head /readyz is green; without this, deadlock)
#   - wait /readyz green
#   - send /generate, poll /status until completed
#   - capture cstrace + image path + per-pod logs + summary.txt
#
# Output dir per run: /root/results/cold-<path>-<timestamp>/

set -euo pipefail

PATH_NAME="${1:?usage: $0 <path1-default|path2-nixl_broadcast|path3-nixl_pipelined|path4-l2_only|path5-prewarm>}"

REPO=/root/Hetu-DiT
TEMPLATE="${REPO}/k8s/raycluster-runpod.yaml"
YAML="/tmp/rc-${PATH_NAME}.yaml"
OUTDIR="/root/results/cold-${PATH_NAME}-$(date +%Y%m%d-%H%M%S)"

[[ -f "${TEMPLATE}" ]] || { echo "ERROR: template ${TEMPLATE} missing" >&2; exit 1; }
mkdir -p "${OUTDIR}"

# ---------- generate per-path yaml from template ----------
python3 - "${TEMPLATE}" "${YAML}" "${PATH_NAME}" <<'PYEOF'
import sys, copy, yaml, pathlib
src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
rc = yaml.safe_load(open(src))
def find(spec, n):
    return next(c for c in spec["template"]["spec"]["containers"] if c["name"] == n)
api = find(rc["spec"]["headGroupSpec"], "hetudit-api")

if name == "path1-default":
    pass  # template defaults
elif name == "path2-nixl_broadcast":
    api["args"].append("--init_strategy=nixl_broadcast")
elif name == "path3-nixl_pipelined":
    api["args"].append("--init_strategy=nixl_pipelined")
elif name == "path4-l2_only":
    api["args"].append("--l2_pool_enabled")
elif name == "path5-prewarm":
    api["args"].append("--l2_pool_enabled")
    # 16 GPUs total: 8 head + 8 warm
    for e in api["env"]:
        if e["name"] == "HETUDIT_EXPECTED_GPUS":
            e["value"] = "16"
        elif e["name"] == "HETUDIT_MACHINE_NUMS":
            e["value"] = "2"
        elif e["name"] in ("HETUDIT_SEARCH_MODE", "HETUDIT_SCHEDULER_STRATEGY"):
            e["value"] = "multi_machine_efficient_ilp"
    for wg in rc["spec"]["workerGroupSpecs"]:
        if wg["groupName"] == "gpu-workers-warm":
            wg["replicas"] = 1
            wg["minReplicas"] = 1
else:
    print(f"ERROR: unknown path {name}", file=sys.stderr); sys.exit(1)

# UCX env so NIXL paths (2/3) actually use cuda_copy/cuda_ipc on NVLink
if name in ("path2-nixl_broadcast", "path3-nixl_pipelined", "path5-prewarm"):
    UCX_ENV = [
        {"name": "UCX_TLS", "value": "cuda_copy,cuda_ipc,sm,tcp"},
        {"name": "UCX_MEMTYPE_CACHE", "value": "n"},
        {"name": "UCX_LOG_LEVEL", "value": "warn"},
    ]
    head_spec = rc["spec"]["headGroupSpec"]["template"]["spec"]
    for c in head_spec["containers"]:
        c.setdefault("env", []).extend(UCX_ENV)
    for wg in rc["spec"]["workerGroupSpecs"]:
        for c in wg["template"]["spec"]["containers"]:
            c.setdefault("env", []).extend(UCX_ENV)

pathlib.Path(dst).write_text(yaml.safe_dump(rc, sort_keys=False))
print(f"wrote {dst}")
PYEOF

echo "[$(date +%H:%M:%S)] [${PATH_NAME}] applying ${YAML} ..."
T_APPLY=$(date +%s.%N)
kubectl apply -f "${YAML}" >/dev/null

# Path 5 deadlock break: KubeRay's head svc skips not-ready endpoints by
# default; warm pod's wait-gcs-ready can't reach GCS via svc until head
# is /readyz green, which can't happen until warm pod has joined PG.
# Patch publishNotReadyAddresses=true so kube-proxy routes svc traffic
# even when head is not ready.
if [[ "${PATH_NAME}" == "path5-prewarm" ]]; then
    for _ in $(seq 1 30); do
        kubectl patch svc -n default hetudit-head-svc --type=json \
            -p='[{"op":"add","path":"/spec/publishNotReadyAddresses","value":true}]' 2>/dev/null && break
        sleep 2
    done
fi

echo "[$(date +%H:%M:%S)] [${PATH_NAME}] waiting for head pod ..."
HEAD=""
for _ in $(seq 1 60); do
    HEAD=$(kubectl get pod -n default -l ray.io/cluster=hetudit,ray.io/node-type=head -o jsonpath="{.items[0].metadata.name}" 2>/dev/null || true)
    [[ -n "${HEAD}" ]] && break
    sleep 1
done
[[ -n "${HEAD}" ]] || { echo "ERROR: head pod never appeared" >&2; exit 1; }
echo "[$(date +%H:%M:%S)] [${PATH_NAME}] head=${HEAD}"

WARM=""
if [[ "${PATH_NAME}" == "path5-prewarm" ]]; then
    for _ in $(seq 1 60); do
        WARM=$(kubectl get pod -n default -l ray.io/cluster=hetudit,ray.io/node-type=worker,ray.io/group=gpu-workers-warm -o jsonpath="{.items[0].metadata.name}" 2>/dev/null || true)
        [[ -n "${WARM}" ]] && break
        sleep 2
    done
    echo "[$(date +%H:%M:%S)] [${PATH_NAME}] warm=${WARM:-NONE}"
fi

# Wait /readyz green
T_READY=""
for _ in $(seq 1 90); do
    STATUS=$(kubectl get pod -n default "${HEAD}" -o jsonpath='{.status.containerStatuses[?(@.name=="hetudit-api")].ready}' 2>/dev/null || true)
    if [[ "${STATUS}" == "true" ]]; then T_READY=$(date +%s.%N); break; fi
    sleep 5
done
if [[ -z "${T_READY}" ]]; then
    echo "ERROR: /readyz never green" >&2
    kubectl logs -n default "${HEAD}" -c hetudit-api > "${OUTDIR}/api.fail.log" 2>&1 || true
    kubectl logs -n default "${HEAD}" -c ray-head > "${OUTDIR}/ray-head.fail.log" 2>&1 || true
    [[ -n "${WARM}" ]] && kubectl logs -n default "${WARM}" > "${OUTDIR}/warm.fail.log" 2>&1 || true
    kubectl delete -f "${YAML}" --wait=false >/dev/null 2>&1 || true
    exit 2
fi
DT_READY=$(awk -v a="$T_APPLY" -v r="$T_READY" 'BEGIN{printf "%.2f", r-a}')
echo "[$(date +%H:%M:%S)] [${PATH_NAME}] /readyz green after ${DT_READY}s"

# Send /generate
echo "[$(date +%H:%M:%S)] [${PATH_NAME}] sending /generate ..."
T_REQ=$(date +%s.%N)
RESP=$(kubectl exec -n default "${HEAD}" -c hetudit-api -- python3 -c '
import urllib.request,json,sys
body=json.dumps({"req_id":sys.argv[1],"prompt":"a serene mountain landscape at sunset","height":1024,"width":1024,"seed":42,"num_inference_steps":20}).encode()
r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8000/generate",body,{"Content-Type":"application/json"}),timeout=600)
sys.stdout.write(r.read().decode())
' "${PATH_NAME}" 2>&1 || echo "PYERR")
T_REQ_RET=$(date +%s.%N)
DT_REQ=$(awk -v a="$T_REQ" -v d="$T_REQ_RET" 'BEGIN{printf "%.2f", d-a}')
echo "[$(date +%H:%M:%S)] [${PATH_NAME}] /generate ack ${DT_REQ}s: ${RESP}"
echo "${RESP}" > "${OUTDIR}/response.json"

# Poll /status until completed
TASK=$(echo "${RESP}" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("task_id",""))' 2>/dev/null || echo "")
DT_TOTAL="N/A"
if [[ -n "${TASK}" ]]; then
    echo "[$(date +%H:%M:%S)] [${PATH_NAME}] polling /status/${TASK} ..."
    POLL=$(kubectl exec -n default "${HEAD}" -c hetudit-api -- python3 -c '
import urllib.request,json,sys,time
task=sys.argv[1]
last=""
for _ in range(180):
    try:
        r=urllib.request.urlopen(f"http://127.0.0.1:8000/status/{task}",timeout=10)
        d=json.loads(r.read())
        last=json.dumps(d)
        if d.get("status") in ("completed","failed"): break
    except Exception:
        pass
    time.sleep(2)
sys.stdout.write(last)
' "${TASK}" 2>&1 || echo "POLL_ERR")
    T_DONE=$(date +%s.%N)
    echo "${POLL}" > "${OUTDIR}/final_status.json"
    DT_TOTAL=$(awk -v a="$T_REQ" -v d="$T_DONE" 'BEGIN{printf "%.2f", d-a}')
    echo "[$(date +%H:%M:%S)] [${PATH_NAME}] image done in ${DT_TOTAL}s"
fi

# Capture logs
kubectl logs -n default "${HEAD}" -c hetudit-api > "${OUTDIR}/api.log" 2>&1 || true
kubectl logs -n default "${HEAD}" -c ray-head > "${OUTDIR}/ray-head.log" 2>&1 || true
[[ -n "${WARM}" ]] && kubectl logs -n default "${WARM}" > "${OUTDIR}/warm.log" 2>&1 || true
grep -hE '^\[CSTRACE\]' "${OUTDIR}/api.log" "${OUTDIR}/ray-head.log" "${OUTDIR}/warm.log" 2>/dev/null > "${OUTDIR}/cstrace.log" || true

# Tear down RC
echo "[$(date +%H:%M:%S)] [${PATH_NAME}] tearing down RC ..."
kubectl delete -f "${YAML}" --wait=false >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
    kubectl get pod -n default "${HEAD}" >/dev/null 2>&1 || break
    sleep 2
done

cat > "${OUTDIR}/summary.txt" <<SUM
path:           ${PATH_NAME}
head_pod:       ${HEAD}
warm_pod:       ${WARM:-NONE}
apply_to_ready: ${DT_READY}s
generate_ack:   ${DT_REQ}s
end_to_end:     ${DT_TOTAL}s
cstrace_lines:  $(wc -l < "${OUTDIR}/cstrace.log" 2>/dev/null || echo 0)
SUM
cat "${OUTDIR}/summary.txt"
echo "[$(date +%H:%M:%S)] [${PATH_NAME}] DONE outdir=${OUTDIR}"
