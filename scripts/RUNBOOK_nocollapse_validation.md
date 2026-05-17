# No-collapse validation runbook (PKU 4090 / any GPU box)

Purpose: prove the §8.32 throughput-collapse is fixed by the serial
dispatcher (commit `f6e4ec6`). This is the only remaining step of the
goal; code fix + local TDD are done. RunPod was abandoned for this run
due to repeated cold-node image-pull stalls (infra, not our code).

## 0. Prereqs (once)
```
git fetch && git checkout xinwei/multimodel-base && git pull   # tip >= f6e4ec6
# env recipe (context.md §8.30): on a fresh box, in the hetu_dit venv:
rm -f /usr/lib/python3/dist-packages/blinker*.egg-info 2>/dev/null || true
pip install -q --upgrade blinker
pip install -e .
pip install -q scipy matplotlib 'diffusers==0.32.0' 'transformers<4.50'
# flash-attn: use the wheel matching the box's torch/cuda/py (the
# 2.6.3 cu123 torch2.4 cp311 wheel worked on the A100 runs); on PKU
# 4090 pick the matching prebuilt wheel or the cluster's existing one.
python3 -c 'import flash_attn'
python3 -m pytest tests/unit/test_serial_dispatcher.py -q   # must be 5/5
```

## 1. Local sanity (no GPU) — already green here
`tests/unit/test_serial_dispatcher.py` 5/5 incl. the meta-test proving
the progress assertion has teeth. If this fails, stop — the fix
regressed.

## 2. Launch the multi-model server (registers TridentServe model set)
4090 = 24 GB: pick two models that each fit in 24 GB individually (the
dispatcher keeps ONE resident at a time and swaps). sd3 (~14 GB) +
hunyuandit (~15 GB) both fit; flux (~24 GB) is too tight on a single
4090 (use it only on >=40 GB cards or multi-4090). cogvideox /
hunyuanvideo are registered for paper coverage but need bigger VRAM.

```
export HF_TOKEN=...                # gated SD3
export HF_HOME=/path/to/hf_cache
export HETU_COLDSTART_TRACE=1
python -m hetu_dit.entrypoint.api_server --host 0.0.0.0 --port 8000 \
  --models sd3=stabilityai/stable-diffusion-3-medium-diffusers \
  --models hunyuandit=Tencent-Hunyuan/HunyuanDiT-Diffusers \
  --models flux=black-forest-labs/FLUX.1-dev \
  --models cogvideox=THUDM/CogVideoX-2b \
  --models hunyuanvideo=hunyuanvideo-community/HunyuanVideo \
  --default-model sd3 \
  --tensor_parallel_degree 1 --ulysses_degree 1 --ring_degree 1 \
  --machine_nums 1 --search-mode random
```
Wait for `Uvicorn running on http://0.0.0.0:8000`.
Coverage check: `curl -s localhost:8000/models` must list all 5 ids.

## 3. The no-collapse open-loop test (exact §8.32 config)
Use the SAME knobs as the §8.32 baseline so it is an apples-to-apples
before/after. Substitute the model pair to the two that fit (sd3 +
hunyuandit on a single 4090; sd3 + flux on >=40 GB):
```
python3 scripts/coldstart_openloop_bench.py --base-url http://localhost:8000 \
  --models sd3,hunyuandit --rate-scale 0.05 --duration 60 \
  --max-requests 120 --poll-interval 0.5 --sample-interval 1.0 \
  --drain-s 240 --slo-interactive 10 --slo-relaxed 30 \
  --out openloop_fixed.json
```
(The harness merges per-model traces; if a model has no
`data/<id>_trace.txt`, pass `--models sd3,flux` etc. matching the
available traces. sd3+flux traces both exist.)

## 4. PASS / FAIL criteria (vs §8.32 baseline)

| signal | §8.32 BASELINE (broken) | FIXED must show |
|---|---|---|
| completed | 8 / 96 (8.3%) | **>= ~90%** of admitted reqs complete (rest may be admission-shed, NOT silently stuck) |
| sd3 outstanding vs t | ramps to 84 then **frozen 180+s** | rises then **drains** (monotone-down after arrivals stop) |
| GPU utilization | **0%** while queue full | **>0** (busy) whenever work is queued |
| done-count over time | frozen at 7 from t~=36s | strictly increasing until drained |
| nvidia-smi during drain | 0% util, queue stuck | active util, queue shrinking |

Also confirm in `server.log`: serial dispatcher started
(`[engine] serial dispatcher started`), and NO permanent freeze (the
`fire seq=` / completion lines keep advancing; `_serial_execute`
proceeds task after task).

PASS = throughput stays > 0 and the queue drains (graceful degradation:
latency may be high under overload, admission may shed — that is
correct, NOT a collapse). FAIL = GPU 0% with a non-empty queue / done
count frozen / >~50% silently never complete (i.e. §8.32 reproduced).

## 5. Record
On PASS: append result to `~/Desktop/Hetu-DiT-context.md` §8.34 with the
per-model timeseries + completion %, and flip memory
`project_coldstart_finding.md` to "fixed, validated on <hw> <date>".
On FAIL: capture server.log + openloop_fixed.json, return to
systematic-debugging Phase 1 (the dispatcher contract is unit-proven, so
a FAIL means the integration — _serial_execute reusing
get_ready_executor_or_reconfigure — still hits a Ray-side stall; next
suspect: the reused Path-C reconfigure under serial still wedging).
```
```
