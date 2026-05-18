#!/usr/bin/env bash
# One-command provisioned real-hardware Hetu Benchmark run.
# Launches a Lambda A100-40GB, sets up the env (committed fastsetup
# recipe), downloads sd3+flux, starts the multimodel server, runs the
# full Hetu Benchmark, pulls HETU_REPORT.json back, and TERMINATES the
# instance (no ongoing cost). Idempotent-safe: always tears down.
#
# Prereqs (never hardcoded — read from env / files):
#   ~/.lambdalabs/key   Lambda Cloud API key
#   $HF_TOKEN           HuggingFace token (gated SD3)        REQUIRED
#   ~/.ssh/<KEY>        SSH private key whose name is $SSH_KEY_NAME
#
# Usage:
#   HF_TOKEN=hf_xxx SSH_KEY_NAME=my-key SSH_KEY=~/.ssh/my-key \
#     bash scripts/run_hetu_benchmark.sh [--mode all] [--region us-west-2]
set -euo pipefail
: "${HF_TOKEN:?set HF_TOKEN (do NOT hardcode it)}"
LK=$(cat ~/.lambdalabs/key)
SSH_KEY_NAME="${SSH_KEY_NAME:?set SSH_KEY_NAME (Lambda-registered key name)}"
SSH_KEY="${SSH_KEY:?set SSH_KEY (path to the matching private key)}"
REGION="${REGION:-us-west-2}"
BRANCH="${BRANCH:-xinwei/multimodel-base}"
MODE="${HETU_MODE:-all}"
SSHO="-i $SSH_KEY -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
API=https://cloud.lambdalabs.com/api/v1

echo "== launch =="
IID=$(curl -s -u "$LK:" -X POST "$API/instance-operations/launch" \
  -H "Content-Type: application/json" \
  -d "{\"region_name\":\"$REGION\",\"instance_type_name\":\"gpu_1x_a100_sxm4\",\"ssh_key_names\":[\"$SSH_KEY_NAME\"],\"name\":\"hetu-bench\"}" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"]["instance_ids"][0])')
echo "instance=$IID"
trap 'echo "== teardown =="; curl -s -u "$LK:" -X POST \
  "$API/instance-operations/terminate" -H "Content-Type: application/json" \
  -d "{\"instance_ids\":[\"$IID\"]}" >/dev/null; echo terminated $IID' EXIT

echo "== wait SSH =="
IP=""
for _ in $(seq 1 60); do
  IP=$(curl -s -u "$LK:" "$API/instances/$IID" \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"].get("ip") or "")')
  [ -n "$IP" ] && ssh $SSHO -o ConnectTimeout=8 ubuntu@"$IP" 'echo ok' \
    2>/dev/null && break
  sleep 25
done
echo "ip=$IP"

echo "== bootstrap (committed fastsetup one-pass) =="
ssh $SSHO ubuntu@"$IP" "HF_TOKEN='$HF_TOKEN' bash -s" <<EOF
set -eo pipefail
curl -fsSL https://raw.githubusercontent.com/li-xinwei/Hetu-DiT/$BRANCH/scripts/fastsetup/lambda_setup.sh -o ls.sh
rm -rf ~/venv; bash ls.sh $BRANCH
source ~/venv/bin/activate
bash ~/work/Hetu-DiT/scripts/fastsetup/lambda_download_models.sh
echo BOOTSTRAP_DONE
EOF

echo "== launch server =="
ssh $SSHO ubuntu@"$IP" "HF_TOKEN='$HF_TOKEN' bash -s" <<'EOF'
source ~/venv/bin/activate
export HF_HOME=/home/ubuntu/hf_cache HETU_COLDSTART_TRACE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd ~/work/Hetu-DiT
tmux new -d -s srv "python -m hetu_dit.entrypoint.api_server \
  --host 0.0.0.0 --port 8000 \
  --models sd3=stabilityai/stable-diffusion-3-medium-diffusers \
  --models flux=black-forest-labs/FLUX.1-dev --default-model sd3 \
  --tensor_parallel_degree 1 --ulysses_degree 1 --ring_degree 1 \
  --machine_nums 1 --search-mode random 2>&1 | tee ~/server.log"
for _ in $(seq 1 40); do
  grep -q 'Uvicorn running' ~/server.log 2>/dev/null && { echo SRV_UP; break; }
  sleep 10
done
EOF

echo "== run Hetu Benchmark (mode=$MODE) =="
ssh $SSHO ubuntu@"$IP" "bash -s" <<EOF
source ~/venv/bin/activate; cd ~/work/Hetu-DiT
python3 scripts/hetu_benchmark.py --base-url http://localhost:8000 \
  --mode $MODE --out-dir ~/hetu_bench --drain-s 200 2>&1 | tail -50
EOF

echo "== pull report =="
mkdir -p ./hetu_bench_results
scp $SSHO ubuntu@"$IP":~/hetu_bench/HETU_REPORT.json \
  ./hetu_bench_results/ 2>/dev/null || echo "(report scp failed)"
echo "HETU_REPORT -> ./hetu_bench_results/HETU_REPORT.json"
echo "DONE (teardown on exit)"
