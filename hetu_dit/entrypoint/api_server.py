import argparse
import ssl
from typing import Any, Optional
import time
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, FileResponse, HTMLResponse
import glob as _glob
import uvicorn
import asyncio
import copy
import os
import math
import traceback
import random
from hetu_dit.engine.async_serving_engine import AsyncServingEngine
from hetu_dit.config import hetuDiTArgs, EngineConfig, InputConfig, ServingConfig
from hetu_dit.config.config import ModelEntry
from diffusers import StableDiffusion3Pipeline
from diffusers import CogVideoXPipeline
from diffusers import FluxPipeline
from diffusers import HunyuanDiTPipeline
from diffusers import HunyuanVideoPipeline


# Map short model-class tokens (CLI/request) → diffusers Pipeline class. Used
# by both legacy --model-class and the new --models multi-model registration.
MODEL_CLASS_REGISTRY = {
    "sd3": StableDiffusion3Pipeline,
    "sd3.5": StableDiffusion3Pipeline,
    "cogvideox": CogVideoXPipeline,
    "flux": FluxPipeline,
    "hunyuandit": HunyuanDiTPipeline,
    "hunyuanvideo": HunyuanVideoPipeline,
}


def _resolve_pipeline_class(class_name: str):
    cls = MODEL_CLASS_REGISTRY.get(class_name)
    if cls is None:
        raise ValueError(
            f"Invalid model class: {class_name!r}; "
            f"available: {sorted(MODEL_CLASS_REGISTRY)}"
        )
    return cls
from hetu_dit.logger import init_logger
from hetu_dit.config.config import EngineConfig, InputConfig, ParallelConfig
from hetu_dit.core.request_manager.scheduler import Scheduler
import threading
from hetu_dit.model_profiler import ModelProfiler
from hetu_dit.profiler import global_profiler
from tqdm.asyncio import tqdm
from hetu_dit.utils import create_new_config, make_profile_key

from hetu_dit.entrypoint.utils import get_loopback_host

logger = init_logger(__name__)
TIMEOUT_KEEP_ALIVE = 10  # seconds.
app = FastAPI()
engine = None

results_store = {}

# Single-file manual test UI served at GET /ui. Talks to /models, /generate,
# /status/{id}, /image/{id}. Kept inline (no static dir) so a single
# api_server process is fully self-contained for RunPod-proxy testing.
# All dynamic text goes through textContent / DOM nodes (no innerHTML) so
# there is no injection surface even though model_ids are server-controlled.
_WEB_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Hetu-DiT multi-model playground</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
         background:#0b0d10; color:#d7dce2; }
  header { padding:14px 20px; border-bottom:1px solid #1e2329;
           font-weight:600; letter-spacing:.3px; }
  header span { color:#6ee7b7; }
  main { display:grid; grid-template-columns:380px 1fr; gap:0; height:calc(100vh - 51px); }
  form { padding:20px; border-right:1px solid #1e2329; overflow-y:auto; }
  label { display:block; margin:14px 0 4px; color:#8b95a1; font-size:12px;
          text-transform:uppercase; letter-spacing:.5px; }
  select,textarea,input { width:100%; background:#13171c; color:#d7dce2;
          border:1px solid #2a313a; border-radius:6px; padding:9px 10px;
          font:inherit; }
  textarea { resize:vertical; min-height:74px; }
  .row { display:flex; gap:10px; }
  .row > div { flex:1; }
  button { margin-top:20px; width:100%; padding:11px; background:#1f6feb;
          color:#fff; border:0; border-radius:6px; font:inherit;
          font-weight:600; cursor:pointer; }
  button:disabled { background:#30363d; cursor:not-allowed; }
  .panel { padding:20px; display:flex; flex-direction:column; align-items:center;
          justify-content:center; overflow:auto; }
  .panel img { max-width:100%; max-height:100%; border-radius:8px;
          border:1px solid #1e2329; }
  #stat { color:#8b95a1; font-size:13px; white-space:pre-wrap; text-align:center; }
  #spin { display:none; width:14px; height:14px; border:2px solid #30363d;
          border-top-color:#6ee7b7; border-radius:50%; animation:s .8s linear infinite;
          vertical-align:-2px; margin-right:6px; }
  #spin.on { display:inline-block; }
  @keyframes s { to { transform:rotate(360deg);} }
</style>
</head>
<body>
<header>Hetu-DiT <span>multi-model</span> playground</header>
<main>
  <form id="f">
    <label>Model</label>
    <select id="model"></select>
    <label>Prompt</label>
    <textarea id="prompt">a photorealistic red panda astronaut, studio lighting</textarea>
    <label>Negative prompt</label>
    <textarea id="neg"></textarea>
    <div class="row">
      <div><label>Width</label><input id="w" type="number" value="512" step="64"/></div>
      <div><label>Height</label><input id="h" type="number" value="512" step="64"/></div>
    </div>
    <div class="row">
      <div><label>Steps</label><input id="steps" type="number" value="20"/></div>
      <div><label>Seed</label><input id="seed" type="number" value="42"/></div>
    </div>
    <button id="go" type="submit">Generate</button>
  </form>
  <div class="panel">
    <div id="stat"><span id="spin"></span><span id="msg">Pick a model and hit Generate.</span></div>
    <img id="img" style="display:none"/>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
let polling = null;

function say(text, busy){
  $('msg').textContent = text;
  $('spin').classList.toggle('on', !!busy);
}

async function loadModels() {
  try {
    const r = await fetch('/models'); const d = await r.json();
    const sel = $('model');
    sel.textContent = '';
    d.models.forEach(m => {
      const o = document.createElement('option');
      o.textContent = m; o.value = m;
      if (m === d.default) o.selected = true;
      sel.appendChild(o);
    });
  } catch(e) { say('Failed to load /models: ' + e, false); }
}

function setBusy(b){ $('go').disabled=b; $('go').textContent=b?'Generating…':'Generate'; }

async function poll(taskId, t0){
  try {
    const r = await fetch('/status/' + encodeURIComponent(taskId));
    const d = await r.json();
    const secs = ((Date.now()-t0)/1000).toFixed(1);
    if (d.image_url) {
      clearInterval(polling); polling=null; setBusy(false);
      $('img').src = d.image_url + '?t=' + Date.now();
      $('img').style.display='block';
      say(`done in ${secs}s  (${taskId})`, false);
    } else {
      say(`${d.status} · ${secs}s · ${taskId}`, true);
    }
  } catch(e) { say('poll error: ' + e, false); }
}

$('f').addEventListener('submit', async ev => {
  ev.preventDefault();
  if (polling) clearInterval(polling);
  $('img').style.display='none';
  setBusy(true);
  say('queuing…', true);
  const body = {
    model: $('model').value,
    prompt: $('prompt').value,
    negative_prompt: $('neg').value,
    width: +$('w').value, height: +$('h').value,
    num_inference_steps: +$('steps').value, seed: +$('seed').value,
    req_id: 'ui' + Date.now()
  };
  try {
    const r = await fetch('/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const d = await r.json();
    if (r.status >= 400) {
      setBusy(false);
      say('error ' + r.status + ': ' + JSON.stringify(d), false);
      return;
    }
    const t0 = Date.now();
    polling = setInterval(() => poll(d.task_id, t0), 1500);
  } catch(e) { setBusy(false); say('request failed: ' + e, false); }
});

loadModels();
</script>
</body>
</html>"""

PROFILE_ON_STARTUP = False
PROFILE_REPEAT_TIMES = 1


def _update_result(task_id: str, *, status: str, output: Optional[str] = None) -> None:
    results_store.setdefault(task_id, {})
    results_store[task_id]["status"] = status
    if output is not None:
        results_store[task_id]["output"] = output


_parallel_strategy_cache = None
_parallel_strategy_lock = threading.Lock()


def _log_parallel_parameters(task_id: str, parallel_config: ParallelConfig) -> None:
    logger.info(
        "[API Server] Task %s parallel parameters: dp=%s, ulysses=%s, ring=%s, tp=%s, pp=%s",
        task_id,
        parallel_config.dp_config.dp_degree,
        parallel_config.sp_config.ulysses_degree,
        parallel_config.sp_config.ring_degree,
        parallel_config.tp_config.tp_degree,
        parallel_config.pp_config.pp_degree,
    )


def _schedule_task(coro: Any, *, description: str) -> asyncio.Task:
    task = asyncio.create_task(coro)
    logger.debug("Scheduled %s; total tasks=%d", description, len(asyncio.all_tasks()))
    return task


def find_machine_ilde_num(
    detect_meta: dict, constrained_worker_ids: list[int] = None
) -> int:
    from collections import defaultdict

    machine_workers = defaultdict(list)

    # Assign workers that meet the constraints to the corresponding machine
    for worker_id, info in detect_meta.items():
        num = int(worker_id.replace("Worker", ""))

        if constrained_worker_ids is not None and num not in constrained_worker_ids:
            continue

        machine_id = num // 8
        machine_workers[machine_id].append(info["state"])

    if not machine_workers:
        raise ValueError("No valid workers found in constrained_worker_ids.")

    # Count the number of busy workers in each machine
    busy_counts = {}
    idle_count = []
    for machine_id, states in machine_workers.items():
        busy_count = sum(1 for state in states if state == "busy")
        # busy_counts[machine_id] = 8 - busy_count
        idle_count.append(8 - busy_count)

    return idle_count


@app.on_event("startup")
async def startup():
    logger.info("Server starting up...")
    # init
    global engine
    await engine.init_all_executors()
    await engine.init_monitor()
    if PROFILE_ON_STARTUP:
        logger.info("Startup profiling enabled (repeat=%d)", PROFILE_REPEAT_TIMES)
        await profile_task(repeat_times=PROFILE_REPEAT_TIMES)
    logger.info("Server initialization complete")
    _schedule_task(process_queue(), description="process_queue")
    if engine.search_mode == "fix":
        scanner_task = _schedule_task(
            engine._scan_worker_queues(), description="queue_scanner"
        )
        scanner_task.set_name("queue_scanner")
    elif engine.search_mode == "greedy_ilp":
        logger.info("enter greedy_ilp scan-request-queue")
        scanner_task = _schedule_task(
            engine._scan_request_queues(), description="queue_scanner"
        )
        scanner_task.set_name("queue_scanner")


async def process_queue():
    """Process tasks in the request queue."""
    processed = False
    global engine
    if engine.search_mode == "random" or engine.search_mode == "greedy_ilp":
        while True:
            try:
                """
                if not scheduler._queue and processed:
                    avg_latency = scheduler.get_average_latency()
                    logger.info(
                        f"[API Server] All tasks completed. Average latency: {avg_latency:.2f} seconds")
                    processed = False
                """

                priority, (task_id, input_config, engine_config) = await scheduler.get()

                _update_result(task_id, status="processing")
                processed = True

                logger.info(
                    f"[API Server] Processing task {task_id} (height={input_config.height}, width={input_config.width})"
                )

                _log_parallel_parameters(task_id, engine_config.parallel_config)

                # Generate image

                output_path = await generate_image(
                    input_config,
                    engine_config,
                    task_id=task_id,
                    search_mode=engine.search_mode,
                )
                _update_result(task_id, status="completed", output=output_path)

                logger.info(f"The remaining tasks left:{len(scheduler._queue)}")

            except asyncio.QueueEmpty:
                processed = False
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"[API Server] Error processing task: {e}")
                traceback.print_exc()
                await asyncio.sleep(0.1)

    elif engine.search_mode == "efficient_ilp":
        while True:
            try:
                """
                if not scheduler._queue and processed:
                    avg_latency = scheduler.get_average_latency()
                    logger.info(
                        f"[API Server] All tasks completed. Average latency: {avg_latency:.2f} seconds")
                    processed = False
                """
                await engine.monitor.refresh()
                meta = engine.detect_meta()  # use monitoring results
                free = sum(1 for info in meta.values() if info["state"] != "busy")
                busy_machine_idle_time = [
                    info["estimated_idle_time"]
                    for info in meta.values()
                    if info["state"] == "busy"
                ]
                logger.debug(
                    f"free:{free}, len busy_machine is {len(busy_machine_idle_time)}, busy_machine_idle_time:{busy_machine_idle_time}"
                )
                busy_machine_idle_time = None
                (task_id, input_config, engine_config) = await scheduler.get(
                    free, busy_machine_idle_time
                )

                _update_result(task_id, status="processing")
                processed = True

                logger.info(
                    f"[API Server] Processing task {task_id} (height={input_config.height}, width={input_config.width})"
                )

                _log_parallel_parameters(task_id, engine_config.parallel_config)

                # Generate image
                logger.debug(
                    f"before enter await generate_image, parallel_config.ulysses_degree is {engine_config.parallel_config.ulysses_degree}"
                )
                output_path = await generate_image(
                    input_config,
                    copy.deepcopy(engine_config),
                    task_id=task_id,
                    search_mode=engine.search_mode,
                )
                _update_result(task_id, status="completed", output=output_path)

                logger.info(f"The remaining tasks left:{len(scheduler._queue)}")

            except asyncio.QueueEmpty:
                processed = False
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"[API Server] Error processing task: {e}")
                traceback.print_exc()
                await asyncio.sleep(0.1)

    elif (
        engine.search_mode == "multi_machine_efficient_ilp"
        or engine.search_mode == "greedy_splitk"
    ):
        while True:
            try:
                """
                if not scheduler._queue and processed:
                    avg_latency = scheduler.get_average_latency()
                    logger.info(
                        f"[API Server] All tasks completed. Average latency: {avg_latency:.2f} seconds")
                    processed = False
                """
                await engine.monitor.refresh()
                meta = engine.detect_meta()  # use monitoring results
                free = find_machine_ilde_num(meta)
                busy_machine_idle_time = None
                (
                    task_id,
                    input_config,
                    engine_config,
                    machine_id,
                ) = await scheduler.get(free, busy_machine_idle_time)

                _update_result(task_id, status="processing")
                processed = True

                logger.info(
                    f"[API Server] Processing task {task_id} (height={input_config.height}, width={input_config.width})"
                )

                _log_parallel_parameters(task_id, engine_config.parallel_config)

                # Generate image
                logger.debug(
                    f"before enter await generate_image, parallel_config.ulysses_degree is {engine_config.parallel_config.ulysses_degree}"
                )
                output_path = await generate_image(
                    input_config,
                    copy.deepcopy(engine_config),
                    task_id=task_id,
                    search_mode=engine.search_mode,
                    machine_id=machine_id,
                )
                _update_result(task_id, status="completed", output=output_path)

                logger.info(f"The remaining tasks left:{len(scheduler._queue)}")

            except asyncio.QueueEmpty:
                processed = False
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"[API Server] Error processing task: {e}")
                traceback.print_exc()
                await asyncio.sleep(0.1)
    elif engine.search_mode == "fix":
        while True:
            try:
                if not scheduler._queue and processed:
                    avg_latency = scheduler.get_average_latency()
                    logger.info(
                        f"[API Server] All tasks completed. Average latency: {avg_latency:.2f} seconds"
                    )
                    processed = False

                (
                    priority,
                    (task_id, input_config, engine_config, future, worker_ids),
                ) = await scheduler.get()
                # engine_config is potentially modified by the scheduler.
                # Create a deep copy to ensure a fresh object is passed via Ray,
                # mitigating issues if Ray's serialization doesn't handle the mutated object as expected.

                _update_result(task_id, status="processing")
                processed = True

                logger.info(
                    "[API Server] Task %s worker_ids=%s",
                    task_id,
                    worker_ids,
                )
                _log_parallel_parameters(task_id, engine_config.parallel_config)

                logger.info(
                    f"[API Server] Processing task {task_id} (height={input_config.height}, width={input_config.width})"
                )

                output_path = await generate_image(
                    input_config,
                    engine_config,
                    search_mode="fix",
                    worker_ids=worker_ids,
                    future=future,
                    task_id=task_id,
                )
                _update_result(task_id, status="completed", output=output_path)

                logger.info(f"The remaining tasks left:{len(scheduler._queue)}")
            except asyncio.QueueEmpty:
                processed = False
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"[API Server] Error processing task: {e}")
                await asyncio.sleep(0.1)
    else:
        raise ValueError(f"Invalid search mode: {engine.search_mode}")


@app.get("/")
async def root():
    return {"message": "API is running. Please use the defined endpoints."}


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


def _find_result_image(task_id: str) -> Optional[str]:
    """Locate the PNG a worker wrote for this task.

    Workers save model-specific filenames under results/ (e.g.
    stable_diffusion_3_result_{task_id}.png, flux_result_{task_id}.png,
    hunyuandit_result_{task_id}.png). We glob on the task_id suffix so the
    lookup is model-agnostic and naturally compatible with PR-2 multi-model
    routing (task_id already embeds the model_id).
    """
    matches = _glob.glob(f"results/*{task_id}.png")
    if not matches:
        return None
    # Newest match wins if a task_id somehow recurs.
    return max(matches, key=os.path.getmtime)


@app.get("/models")
async def list_models():
    """Expose the registered multi-model registry for the web UI dropdown."""
    return {
        "models": list(engine.models),
        "default": engine.default_model_id,
    }


@app.get("/status/{task_id}")
async def status(task_id: str):
    """Poll-able task status. Adds image_url once the worker has written the PNG.

    The serving engine is async (scheduler queue + background process_queue),
    so /generate is fire-and-forget. The web UI polls this endpoint until
    status == 'completed' and an image is available.
    """
    entry = results_store.get(task_id)
    if entry is None:
        return JSONResponse({"status": "unknown", "task_id": task_id}, status_code=404)
    resp = {"task_id": task_id, "status": entry.get("status", "unknown")}
    img = _find_result_image(task_id)
    if img is not None:
        resp["image_url"] = f"/image/{task_id}"
        # The result file exists even if the queue callback hasn't flipped the
        # status yet; treat presence of the PNG as completion for the UI.
        if resp["status"] != "completed":
            resp["status"] = "completed"
    return resp


@app.get("/image/{task_id}")
async def image(task_id: str):
    """Serve the generated PNG for a finished task."""
    path = _find_result_image(task_id)
    if path is None:
        return JSONResponse(
            {"error": "image not ready", "task_id": task_id}, status_code=404
        )
    return FileResponse(path, media_type="image/png")


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    """Minimal single-file web UI for manual multi-model testing."""
    return HTMLResponse(_WEB_UI_HTML)


def find_least_busy_machine(
    detect_meta: dict, constrained_worker_ids: list[int] = None
) -> int:
    from collections import defaultdict

    machine_workers = defaultdict(list)

    # allocate workers that meet the constraints to the corresponding machine

    for worker_id, info in detect_meta.items():
        num = int(worker_id.replace("Worker", ""))

        if constrained_worker_ids is not None and num not in constrained_worker_ids:
            continue

        machine_id = num // 8
        machine_workers[machine_id].append(info["state"])

    if not machine_workers:
        raise ValueError("No valid workers found in constrained_worker_ids.")

    # count the number of busy workers in each machine
    busy_counts = {}
    for machine_id, states in machine_workers.items():
        busy_count = sum(1 for state in states if state == "busy")
        busy_counts[machine_id] = busy_count

    # find the machine with the least busy workers; if tied, choose the one with the highest ID
    logger.debug(f"busy_counts: {busy_counts}")
    min_busy = min(busy_counts.values())
    candidate_machines = [
        mid for mid, count in busy_counts.items() if count == min_busy
    ]
    return max(candidate_machines)


@app.post("/generate")
async def generate(request: Request):
    """Generate image for the request."""
    request_dict = await request.json()
    logger.info("enter generate")
    req_id = request_dict.get("req_id", None)
    prompt = request_dict.get("prompt")
    negative_prompt = request_dict.get("negative_prompt", "")
    height = request_dict.get("height", 1024)
    width = request_dict.get("width", 1024)
    num_frames = request_dict.get("num_frames", 0)
    num_inference_steps = request_dict.get("num_inference_steps", 20)
    seed = request_dict.get("seed", 42)

    # M1 multi-model routing: pick which registered model serves this request.
    # Default falls back to engine.default_model_id so old clients without a
    # `model` field keep working unchanged.
    model_id = request_dict.get("model") or engine.default_model_id
    if model_id not in engine.models:
        return JSONResponse(
            {
                "error": "unknown model",
                "requested": model_id,
                "available": list(engine.models),
            },
            status_code=400,
        )

    # Generate unique task ID (model_id embedded for cstrace/grep ergonomics)
    task_id = f"task-{req_id}_{model_id}_{width}x{height}"
    global_profiler.start(task_id, request_dict.copy())
    # Get number of visible GPUs from environment
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices.strip() == "":
        num_visible_gpus = 8  # fallback to 1 if not set
    else:
        num_visible_gpus = len(cuda_visible_devices.split(","))

    # The parallelism parameter is automatically determined based on the input size.
    (
        data_parallel_degree,
        use_cfg_parallel,
        ulysses_degree,
        ring_degree,
        tensor_parallel_degree,
        pipefusion_parallel_degree,
        text_encoder_tensor_parallel_degree,
    ) = determine_parallel_degrees(
        height,
        width,
        num_frames,
        num_visible_gpus,
        engine.engine_config.runtime_config.use_parallel_text_encoder,
        model_class=engine.models[model_id].model_class,
    )
    logger.debug(
        f"The parallel degree is determined as: data_parallel_degree={data_parallel_degree}, use_cfg_parallel={use_cfg_parallel}, ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, tensor_parallel_degree={tensor_parallel_degree}, pipefusion_parallel_degree={pipefusion_parallel_degree}"
    )
    if engine.use_disaggregated_encode_decode:
        random_machine_id = random.randrange(0, engine.engine_config.machine_num)
    else:
        random_machine_id = random.randrange(0, engine.engine_config.machine_num)

    machine_id = find_least_busy_machine(
        engine.detect_meta(), engine.diffusion_worker_ids
    )
    engine_config, input_config = create_new_config(
        old_engine_config=engine.engine_config,
        data_parallel_degree=data_parallel_degree,
        use_cfg_parallel=use_cfg_parallel,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        tensor_parallel_degree=tensor_parallel_degree,
        pipefusion_parallel_degree=pipefusion_parallel_degree,
        use_parallel_text_encoder=engine.engine_config.runtime_config.use_parallel_text_encoder,
        text_encoder_tensor_parallel_degree=text_encoder_tensor_parallel_degree,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        seed=seed,
        task_id=task_id,
        machine_id=machine_id,
    )
    # Tag the per-request input_config with the resolved model_id so that
    # AsyncServingEngine.run_task can route to the right registered model.
    input_config.model_id = model_id

    # Add request to the queue
    scheduler.record_start_time(task_id)
    priority = (
        (height * width) // (16 * 16) * (num_frames - 1) // 4
        if num_frames > 1
        else (height * width) // (16 * 16)
    )
    start_time = time.time()
    if (
        engine.search_mode == "efficient_ilp"
        or engine.search_mode == "multi_machine_efficient_ilp"
        or engine.search_mode == "greedy_splitk"
    ):
        # Profiler entries are per-model; route the lookup to the request's
        # model_id rather than the engine default so multi-model deployments
        # don't share a single profile table.
        model_name = model_id
        profile_key = make_profile_key(input_config)
        t_dict = engine.model_profiler.get_performance_data(model_name, profile_key)
        t_dict = {k: v * input_config.num_inference_steps for k, v in t_dict.items()}
        await scheduler.put(priority, (task_id, input_config, engine_config), t_dict)
    else:
        await scheduler.put(priority, (task_id, input_config, engine_config))
    results_store[task_id] = {"status": "queued"}

    return JSONResponse(
        {
            "message": "Request queued",
            "task_id": task_id,
            "status_url": f"/status/{task_id}",
        },
        status_code=202,
    )


total_request = 0


@app.post("/generate_with_workers")
async def generate_with_workers(request: Request):
    """Generate image for the request."""
    global total_request
    request_dict = await request.json()

    # Extract parameters from request
    req_id = request_dict.get("req_id", None)
    prompt = request_dict.get("prompt")
    negative_prompt = request_dict.get("negative_prompt", "")
    height = request_dict.get("height", 1024)
    width = request_dict.get("width", 1024)
    num_frames = request_dict.get("num_frames", 0)
    num_inference_steps = request_dict.get("num_inference_steps", 20)
    seed = request_dict.get("seed", 42)

    # Generate unique task ID
    task_id = f"task-{req_id}_{width}x{height}"
    # Get number of visible GPUs from environment
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible_devices.strip() == "":
        num_visible_gpus = 8  # fallback to 1 if not set
    else:
        num_visible_gpus = len(cuda_visible_devices.split(","))

    # The parallelism parameter is automatically determined based on the input size.
    (
        data_parallel_degree,
        use_cfg_parallel,
        ulysses_degree,
        ring_degree,
        tensor_parallel_degree,
        pipefusion_parallel_degree,
        text_encoder_tensor_parallel_degree,
        worker_ids,
    ) = determine_parallel_degrees_with_worker_ids(
        height,
        width,
        num_frames,
        num_visible_gpus,
        total_request,
        engine.engine_config.runtime_config.use_parallel_text_encoder,
    )
    logger.debug(
        f"The parallel degree is determined as: data_parallel_degree={data_parallel_degree}, use_cfg_parallel={use_cfg_parallel}, ulysses_degree={ulysses_degree}, ring_degree={ring_degree}, tensor_parallel_degree={tensor_parallel_degree}, pipefusion_parallel_degree={pipefusion_parallel_degree}"
    )

    # Create new engine_config and input_config based on input size and parallelism parameters
    engine_config, input_config = create_new_config(
        old_engine_config=engine.engine_config,
        data_parallel_degree=data_parallel_degree,
        use_cfg_parallel=use_cfg_parallel,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        tensor_parallel_degree=tensor_parallel_degree,
        pipefusion_parallel_degree=pipefusion_parallel_degree,
        use_parallel_text_encoder=engine.engine_config.runtime_config.use_parallel_text_encoder,
        text_encoder_tensor_parallel_degree=text_encoder_tensor_parallel_degree,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        seed=seed,
        task_id=task_id,
    )

    # Queue the request and get future
    future = asyncio.Future()
    logger.debug(f"worker_ids = {worker_ids}")
    priority = (
        (height * width) // (16 * 16) * (num_frames - 1) // 4
        if num_frames > 1
        else (height * width) // (16 * 16)
    )
    await scheduler.put(
        priority, (task_id, input_config, engine_config, future, worker_ids)
    )
    total_request += 1
    results_store[task_id] = {"status": "queued"}
    try:
        result = await future
        results_store[task_id] = result
        return {"request_id": task_id, "status": "completed"}
    except Exception as e:
        logger.error(f"Error processing request {task_id}: {str(e)}")
        return {"request_id": task_id, "status": "error", "message": str(e)}


async def generate_image(
    input_config: InputConfig,
    engine_config: EngineConfig,
    search_mode="random",
    worker_ids=None,
    future=None,
    task_id=None,
    machine_id=None,
) -> str:
    """Generate an image using the engine."""
    logger.debug(
        f"Starting image generation for prompt: {input_config.prompt}, height = {input_config.height}, width = {input_config.width}"
    )
    torch.cuda.empty_cache()

    # Run the engine
    if search_mode == "random" and engine.use_disaggregated_encode_decode:
        logger.debug("before run_task_disaggregated generate_image")
        await engine.run_task_disaggregated(
            input_config=input_config, engine_config=engine_config, task_id=task_id
        )
    elif search_mode == "random" and engine.stage_level:
        logger.debug("enter random stage_level")
        await engine.run_task_downscale_vae(
            input_config=input_config, engine_config=engine_config, task_id=task_id
        )
    elif (
        search_mode == "random"
        and not engine.stage_level
        and not engine.use_disaggregated_encode_decode
    ):
        logger.debug("enter random ")
        await engine.run_task(
            input_config=input_config, engine_config=engine_config, task_id=task_id
        )
    elif (
        search_mode == "greedy_ilp"
        and not engine.stage_level
        and not engine.use_disaggregated_encode_decode
    ):
        logger.debug("enter greedy_ilp")
        await engine.add_task_greedy_ilp(input_config, engine_config, task_id=task_id)
    elif (
        search_mode == "efficient_ilp"
        and not engine.stage_level
        and not engine.use_disaggregated_encode_decode
    ):
        logger.debug(
            f"enter efficient_ilp, engine_config.ulysses_degree = {engine_config.parallel_config.ulysses_degree}"
        )
        await engine.run_task(
            input_config=input_config, engine_config=engine_config, task_id=task_id
        )
    elif search_mode == "efficient_ilp" and engine.stage_level:
        logger.debug(
            f"enter efficient_ilp, stage_level, engine_config.ulysses_degree = {engine_config.parallel_config.ulysses_degree}"
        )
        await engine.run_task_downscale_vae(
            input_config=input_config, engine_config=engine_config, task_id=task_id
        )
    elif (
        search_mode == "multi_machine_efficient_ilp"
        and not engine.stage_level
        and not engine.use_disaggregated_encode_decode
    ):
        logger.debug(
            f"enter multi_machine_efficient_ilp, engine_config.ulysses_degree = {engine_config.parallel_config.ulysses_degree}"
        )
        await engine.run_task(
            input_config=input_config,
            engine_config=engine_config,
            task_id=task_id,
            machine_id=machine_id,
        )
    elif search_mode == "multi_machine_efficient_ilp" and engine.stage_level:
        logger.debug(
            f"enter multi_machine_efficient_ilp, stage_level, engine_config.ulysses_degree = {engine_config.parallel_config.ulysses_degree}"
        )
        await engine.run_task_downscale_vae(
            input_config=input_config,
            engine_config=engine_config,
            task_id=task_id,
            machine_id=machine_id,
        )
    elif (
        search_mode == "greedy_splitk"
        and not engine.stage_level
        and not engine.use_disaggregated_encode_decode
    ):
        logger.debug(
            f"enter greedy_splitk, engine_config.ulysses_degree = {engine_config.parallel_config.ulysses_degree}"
        )
        await engine.run_task(
            input_config=input_config,
            engine_config=engine_config,
            task_id=task_id,
            machine_id=machine_id,
        )
    elif search_mode == "greedy_splitk" and engine.stage_level:
        logger.debug(
            f"enter greedy_splitk, stage_level, engine_config.ulysses_degree = {engine_config.parallel_config.ulysses_degree}"
        )
        await engine.run_task_downscale_vae(
            input_config=input_config,
            engine_config=engine_config,
            task_id=task_id,
            machine_id=machine_id,
        )
    elif search_mode == "fix":
        logger.debug("enter fix")
        logger.debug(f"in generate_image worker_ids = {worker_ids}")
        result = await engine.add_task(
            worker_ids, input_config, engine_config, task_id=task_id
        )
        logger.info("Request completed")
        future.set_result(result)
    return "success"


def _parse_models_cli(raw_models: list[str]) -> list[ModelEntry]:
    """Parse repeated --models tokens of form 'model_id=path' into ModelEntry.

    The model_id token must exist in MODEL_CLASS_REGISTRY (short names: sd3,
    flux, hunyuandit, ...). Custom IDs would require an extra --model-class-of
    override mechanism (deferred to a future PR).
    """
    entries: list[ModelEntry] = []
    seen_ids = set()
    for token in raw_models:
        if "=" not in token:
            raise ValueError(
                f"--models expects 'model_id=path', got {token!r}"
            )
        model_id, _, model_path = token.partition("=")
        model_id = model_id.strip()
        model_path = model_path.strip()
        if not model_id or not model_path:
            raise ValueError(f"--models malformed entry: {token!r}")
        if model_id in seen_ids:
            raise ValueError(f"--models duplicate model_id: {model_id!r}")
        seen_ids.add(model_id)
        entries.append(
            ModelEntry(
                model_id=model_id,
                model_class=_resolve_pipeline_class(model_id),
                model_path=model_path,
            )
        )
    return entries


def create_engine(args: argparse.Namespace) -> AsyncServingEngine:
    """Create AsyncServingEngine from arguments."""
    # M1 multi-model registration. If --models is provided, build the full
    # registry; otherwise fall back to legacy --model-class single-entry path
    # so existing deployments keep working byte-for-byte.
    raw_models = getattr(args, "models", None) or []
    if raw_models:
        models_list = _parse_models_cli(raw_models)
        default_model_id = args.default_model or models_list[0].model_id
        if default_model_id not in {m.model_id for m in models_list}:
            raise ValueError(
                f"--default-model {default_model_id!r} not in --models "
                f"{[m.model_id for m in models_list]}"
            )
        default_entry = next(m for m in models_list if m.model_id == default_model_id)
        primary_model_class = default_entry.model_class
        primary_model_path = default_entry.model_path
        primary_model_class_name = default_model_id
    else:
        models_list = None
        default_model_id = None
        primary_model_class = _resolve_pipeline_class(args.model_class)
        primary_model_path = args.model
        primary_model_class_name = args.model_class

    # Convert args to hetuDiTArgs; engine_config.model_config.model carries the
    # *default* model path. _resolve_model_id swaps it per request when serving
    # multiple models.
    engine_args = hetuDiTArgs(
        model=primary_model_path,
        download_dir=args.download_dir,
        trust_remote_code=args.trust_remote_code,
        # Parallel configs
        data_parallel_degree=args.data_parallel_degree,
        use_cfg_parallel=args.use_cfg_parallel,
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        tensor_parallel_degree=args.tensor_parallel_degree,
        pipefusion_parallel_degree=args.pipefusion_parallel_degree,
        use_parallel_text_encoder=args.use_parallel_text_encoder,
        text_encoder_tensor_parallel_degree=args.text_encoder_tensor_parallel_degree,
        # Runtime configs
        warmup_steps=args.warmup_steps,
        use_parallel_vae=args.use_parallel_vae,
        use_torch_compile=args.use_torch_compile,
        use_onediff=args.use_onediff,
        adjust_strategy=args.adjust_strategy,
        # machine id
        machine_num=args.machine_nums,
        use_disaggregated_encode_decode=args.use_disaggregated_encode_decode,
        encode_worker_ids=args.encode_worker_ids,
        decode_worker_ids=args.decode_worker_ids,
        stage_level=args.stage_level,
    )

    # Create engine configs
    engine_config, input_config = engine_args.create_config(is_serving=True)
    serving_config = ServingConfig(
        engine_config=engine_config,
        input_config=input_config,
        model_class=primary_model_class,
        models=models_list,
        default_model_id=default_model_id,
    )

    # Create engine
    return AsyncServingEngine.from_engine_args(
        serving_config,
        args.search_mode,
        args.use_disaggregated_encode_decode,
        args.stage_level,
        args.encode_worker_ids,
        args.decode_worker_ids,
        primary_model_class_name,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--ssl-ca-certs", type=str, default=None, help="The CA certificates file"
    )
    parser.add_argument(
        "--ssl-cert-reqs",
        type=int,
        default=int(ssl.CERT_NONE),
        help="Whether client certificate is required (see stdlib ssl module's)",
    )
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy",
    )
    # ========== for scheduler =========
    parser.add_argument(
        "--scheduler-strategy",
        type=str,
        default="fifo",
        help="Scheduling strategy to use (priority, fifo, ilp_fix, or ilp_random)",
    )
    # ========== for scheduler =========
    # Add hetuDiTArgs for model class (legacy single-model path; superseded by
    # --models when both are present, but kept for back-compat with old launch
    # scripts).
    parser.add_argument("--model-class", type=str, default="sd3")

    # M1 multi-model serving registry. Repeat once per model:
    #   --models sd3=stabilityai/stable-diffusion-3-medium-diffusers
    #   --models flux=black-forest-labs/FLUX.1-dev
    # The model_id (token before =) must match a known short name in
    # MODEL_CLASS_REGISTRY; the path is forwarded to Pipeline.from_pretrained.
    parser.add_argument(
        "--models",
        action="append",
        default=None,
        help="Repeated entries of 'model_id=hf_path_or_local_path'. When set, "
             "/generate accepts a 'model' field naming any registered model_id; "
             "unknown ids return HTTP 400.",
    )
    parser.add_argument(
        "--default-model",
        type=str,
        default=None,
        help="The model_id used when /generate omits the 'model' field. "
             "Required when --models has more than one entry; defaults to the "
             "first --models entry otherwise.",
    )

    # Add hetuDiTArgs for search mode
    parser.add_argument("--search-mode", type=str, default="random")

    # Add hetuDiTArgs for multi machine serving
    parser.add_argument(
        "--machine_nums",
        type=int,
        default=1,
        help="the number of machines in the cluster",
    )
    parser.add_argument(
        "--use_disaggregated_encode_decode",
        action="store_true",
        help="Whether to use disaggregated encode decode, and default only use machine_id=0 to do encode and decode",
    )
    parser.add_argument(
        "--encode_worker_ids",
        type=int,
        nargs="*",
        default=None,
        help="machines to do text encode.",
    )
    parser.add_argument(
        "--decode_worker_ids",
        type=int,
        nargs="*",
        default=None,
        help="machines to do vae decode.",
    )
    parser.add_argument(
        "--stage_level", action="store_true", help="Whether to use vae downscale"
    )
    parser.add_argument(
        "--profile-on-startup",
        action="store_true",
        help="Run the profiler before serving requests",
    )
    parser.add_argument(
        "--profile-repeat",
        type=int,
        default=1,
        help="How many times to repeat each profiled configuration",
    )
    # Add hetuDiTArgs specific arguments
    parser = hetuDiTArgs.add_cli_args(parser)
    args = parser.parse_args()

    # Create engine
    global engine, PROFILE_ON_STARTUP, PROFILE_REPEAT_TIMES
    engine = create_engine(args)
    engine.post_init()

    PROFILE_ON_STARTUP = args.profile_on_startup
    PROFILE_REPEAT_TIMES = max(1, args.profile_repeat)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"

    engine.model_profiler = ModelProfiler(
        model_name=engine.model_class_name,
        device=device_name,
    )

    # ========== init scheduler =========
    global scheduler
    scheduler = Scheduler(
        strategy=args.scheduler_strategy,
        search_mode=args.search_mode,
        model_profiler=engine.model_profiler,
    )
    logger.info(f"[API Server] use schedule strategy '{args.scheduler_strategy}')")

    # Start server. Default keeps the historical loopback bind (single-box
    # local testing); pass --host 0.0.0.0 to expose through the RunPod HTTP
    # proxy so the /ui playground is reachable from a browser.
    app.root_path = args.root_path
    bind_host = args.host if args.host else get_loopback_host()
    logger.info(f"[API Server] binding uvicorn on {bind_host}:{args.port}")
    uvicorn.run(
        app,
        host=bind_host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        ssl_ca_certs=args.ssl_ca_certs,
        ssl_cert_reqs=args.ssl_cert_reqs,
    )


def determine_parallel_degrees_with_worker_ids(
    height: int,
    width: int,
    num_frames: int,
    max_degrees: int = 8,
    request_num=0,
    use_text_encoder_parallel=False,
):
    """
    Determine parallelism parameters based on input image resolution.
    Only apply the first rule whose resulting degree product does not exceed max_degrees.

    Return values:
    (data_parallel_degree, use_cfg_parallel, ulysses_degree, ring_degree,
     tensor_parallel_degree, pipefusion_parallel_degree)
    """
    img_size = height * width * num_frames

    # Defaults
    data_parallel_degree = 1
    use_cfg_parallel = False
    ulysses_degree = 1
    ring_degree = 1
    tensor_parallel_degree = 1
    pipefusion_parallel_degree = 1

    # Random ruled candidates in order: (condition, new degrees tuple)
    if num_frames <= 1:
        rules = [
            (img_size <= 256 * 128, (2, 1, 1, 1)),  # default
            (img_size <= 256 * 256, (2, 1, 1, 1)),  # ulysses=2 2111
            (img_size <= 256 * 512, (1, 2, 1, 1)),  # ulysses=2, pipefusion=2 2112
            (img_size <= 512 * 512, (1, 1, 2, 1)),  # ulysses=2, ring=2 2211
            (img_size <= 1024 * 512, (1, 2, 1, 1)),  # tensor=2, pipefusion=2 1122
            (True, (2, 1, 1, 1)),  # ulysses=2, ring=2, pipefusion=2 2212
        ]
    else:
        rules = [
            (img_size <= 192 * 912 * 9, (2, 1, 1, 1)),  # default
            (img_size <= 192 * 384 * 9, (2, 1, 1, 1)),  # ulysses=2 2111
            (img_size <= 384 * 384 * 9, (1, 2, 1, 1)),  # ulysses=4, 4111
            (img_size <= 768 * 384 * 9, (1, 2, 1, 1)),  # ulysses=2, ring=2 2211
            (img_size <= 768 * 768 * 9, (1, 1, 2, 1)),  # tensor=4, 1141
            (True, (1, 1, 2, 1)),  # ulysses=2, ring=4, pipefusion=2 2411
        ]

    for condition, (uly, ring, tensor, pipe) in rules:
        if condition:
            total_degree = data_parallel_degree * uly * ring * tensor * pipe
            if total_degree <= max_degrees:
                ulysses_degree = uly
                ring_degree = ring
                tensor_parallel_degree = tensor
                pipefusion_parallel_degree = pipe
            break  # Only apply the first matching rule
    needed_workers = (
        data_parallel_degree
        * ulysses_degree
        * ring_degree
        * tensor_parallel_degree
        * pipefusion_parallel_degree
    )
    worker_ids = sorted(random.sample(range(max_degrees - 1), needed_workers))
    logger.debug(
        f" in determine_parallel_degrees_with_worker_ids worker_ids = {worker_ids}"
    )
    if use_text_encoder_parallel:

        def random_power_of_two_fast(num):
            max_exp = int(math.log2(num))
            exp = random.randint(0, max_exp)
            return 2**exp

        parallel_world_size = (
            data_parallel_degree
            * ulysses_degree
            * ring_degree
            * tensor_parallel_degree
            * pipefusion_parallel_degree
        )
        text_encoder_tensor_parallel_degree = random_power_of_two_fast(
            parallel_world_size
        )
        return (
            data_parallel_degree,
            use_cfg_parallel,
            ulysses_degree,
            ring_degree,
            tensor_parallel_degree,
            pipefusion_parallel_degree,
            text_encoder_tensor_parallel_degree,
            worker_ids,
        )
    else:
        return (
            data_parallel_degree,
            use_cfg_parallel,
            ulysses_degree,
            ring_degree,
            tensor_parallel_degree,
            pipefusion_parallel_degree,
            1,
            worker_ids,
        )


def determine_parallel_degrees(
    height: int,
    width: int,
    num_frames: int,
    max_degrees: int = 8,
    use_text_encoder_parallel: bool = False,
    model_class=None,
):
    """
    Determine parallelism parameters based on input image resolution.
    Only apply the first rule whose resulting degree product does not exceed max_degrees.

    Return values:
    (data_parallel_degree, use_cfg_parallel, ulysses_degree, ring_degree,
     tensor_parallel_degree, pipefusion_parallel_degree)

    `model_class` selects the rule table for the request's model. Falls back to
    the engine's default model class for legacy single-model callers.
    """
    img_size = height * width * num_frames

    # Defaults
    data_parallel_degree = 1
    use_cfg_parallel = False
    ulysses_degree = 1
    ring_degree = 1
    tensor_parallel_degree = 1
    pipefusion_parallel_degree = 1

    if model_class is None:
        model_class = engine.model_class

    # Random Ruled candidates in order: (condition, new degrees tuple)
    if model_class == HunyuanDiTPipeline:
        rules = [
            (img_size <= 768 * 768, (1, 1, 1, 1)),  # default
            (img_size <= 1024 * 1024, (2, 1, 1, 1)),  # ulysses=2, ring=2 2211
            (img_size <= 2048 * 2048, (4, 1, 1, 1)),  # tensor=2, pipefusion=2 1122
            (True, (8, 1, 1, 1)),  # ulysses=2, ring=2, pipefusion=2 2212
        ]
    elif model_class == StableDiffusion3Pipeline:
        rules = [
            (img_size <= 1024 * 1536, (1, 1, 1, 1)),
            (True, (2, 1, 1, 1)),
        ]
    elif model_class == FluxPipeline:
        rules = [
            (img_size <= 512 * 512, (1, 1, 1, 1)),  # default
            (img_size <= 1536 * 1536, (2, 1, 1, 1)),  # ulysses=2 2111
            (True, (4, 1, 1, 1)),  # ulysses=2, ring=2, pipefusion=2 2212
        ]
    elif model_class == CogVideoXPipeline:
        rules = [
            (img_size <= 768 * 1024 * 33, (1, 1, 1, 1)),
            (img_size <= 1024 * 1024 * 65, (2, 1, 1, 1)),
            (True, (4, 1, 1, 1)),
        ]

    elif model_class == HunyuanVideoPipeline:
        rules = [
            (img_size <= 720 * 1280 * 17, (1, 1, 1, 1)),
            (img_size <= 720 * 1280 * 33, (2, 1, 1, 1)),  # default
            (img_size <= 720 * 1280 * 65, (4, 1, 1, 1)),  # ulysses=2 2111
            (True, (8, 1, 1, 1)),  # ulysses=2, ring=2, pipefusion=2 2212
        ]
    elif num_frames <= 1:
        rules = [
            (img_size <= 256 * 128, (1, 1, 1, 1)),  # default
            (img_size <= 256 * 256, (2, 1, 1, 1)),  # ulysses=2 2111
            (img_size <= 256 * 512, (1, 4, 1, 1)),  # ulysses=2, pipefusion=2 2112
            (img_size <= 512 * 512, (2, 2, 1, 1)),  # ulysses=2, ring=2 2211
            (img_size <= 1024 * 512, (1, 1, 2, 2)),  # tensor=2, pipefusion=2 1122
            (True, (2, 2, 1, 2)),  # ulysses=2, ring=2, pipefusion=2 2212
        ]
    else:
        rules = [
            (img_size <= 192 * 912 * 9, (1, 1, 1, 1)),  # default
            (img_size <= 192 * 384 * 9, (2, 1, 1, 1)),  # ulysses=2 2111
            (img_size <= 384 * 384 * 9, (4, 1, 1, 1)),  # ulysses=2, pipefusion=2 2112
            (img_size <= 768 * 384 * 9, (2, 2, 1, 1)),  # ulysses=2, ring=2 2211
            (img_size <= 768 * 768 * 9, (1, 1, 4, 1)),  # tensor=2, pipefusion=2 1122
            (True, (2, 4, 1, 1)),  # ulysses=2, ring=2, pipefusion=2 2212
        ]

    for condition, (uly, ring, tensor, pipe) in rules:
        if condition:
            total_degree = data_parallel_degree * uly * ring * tensor * pipe
            if total_degree <= max_degrees:
                ulysses_degree = uly
                ring_degree = ring
                tensor_parallel_degree = tensor
                pipefusion_parallel_degree = pipe
            break  # Only apply the first matching rule

    if use_text_encoder_parallel:

        def random_power_of_two_fast(num):
            max_exp = int(math.log2(num))
            exp = random.randint(0, max_exp)
            return 2**exp

        parallel_world_size = (
            data_parallel_degree
            * ulysses_degree
            * ring_degree
            * tensor_parallel_degree
            * pipefusion_parallel_degree
        )
        text_encoder_tensor_parallel_degree = random_power_of_two_fast(
            parallel_world_size
        )
        return (
            data_parallel_degree,
            use_cfg_parallel,
            ulysses_degree,
            ring_degree,
            tensor_parallel_degree,
            pipefusion_parallel_degree,
            text_encoder_tensor_parallel_degree,
        )
    else:
        return (
            data_parallel_degree,
            use_cfg_parallel,
            ulysses_degree,
            ring_degree,
            tensor_parallel_degree,
            pipefusion_parallel_degree,
            1,
        )


async def profile_task(repeat_times=1):
    if not engine.model_profiler.load_cache():
        task_configs, max_bs = engine.model_profiler.generate_task_config()
        for i, (parallel_config, height, width, num_frames) in enumerate(
            tqdm(task_configs, desc="Profiling tasks")
        ):
            # Only batch size 1 for parallel configs other than (1,1,1,1)
            if parallel_config != (1, 1, 1, 1):
                batchsizes = [1]
            else:
                batchsizes = [i for i in range(1, max_bs + 1) if i & (i - 1) == 0]
            for j in batchsizes:
                engine_config, input_config = create_new_config(
                    old_engine_config=engine.engine_config,
                    ulysses_degree=parallel_config[0],
                    ring_degree=parallel_config[1],
                    tensor_parallel_degree=parallel_config[2],
                    pipefusion_parallel_degree=parallel_config[3],
                    prompt="Model_Profiler" if j == 1 else ["Model_Profiler"] * j,
                    negative_prompt="nagative prompt"
                    if j == 1
                    else ["nagative prompt"] * j,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    num_inference_steps=engine.model_profiler.steps,
                )
                task_id = f"parallel_{parallel_config}_height_{height}_width_{width}_frame_{num_frames}_batchsize_{j}"

                for _ in range(
                    4 * repeat_times if i == len(task_configs) - 1 else repeat_times
                ):
                    await engine.run_task(
                        input_config=input_config,
                        engine_config=engine_config,
                        task_id=task_id,
                    )

        engine.model_profiler.save_cache()
    logger.info("Profiling completed...")


if __name__ == "__main__":
    main()
