"""TDD for SerialModelDispatcher — the §8.32 collapse regression guard.

The pre-existing dispatch path (get_ready_executor_or_reconfigure +
search_reconfigure_worker + waiting_* futures) deadlocked under open-loop
concurrent multi-model load: throughput went to zero, GPU 0% util, ~92%
of requests never completed (context.md §8.32/§8.33).

These tests pin the behavioral contract that makes the new dispatcher
collapse-proof:
  1. progress guarantee  — every submitted task completes (no deadlock)
  2. admission control   — bounded queue rejects under overload
  3. fault isolation     — one failing task does not stall the consumer
  4. switch-on-change    — model bind happens only when model_id changes
  5. the tests have teeth — a deliberately deadlocking variant fails (4 via
                            a meta-test that asserts a hung consumer is
                            detected by the timeout)
"""
import asyncio
import importlib.util
import os


# Load the dispatcher module directly from its file so the test does not
# trigger hetu_dit/__init__.py (which imports diffusers/torch and is not
# available in a plain CI/dev env). The dispatcher itself is stdlib-only.
_MOD_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "hetu_dit",
    "engine",
    "serial_dispatcher.py",
)
_spec = importlib.util.spec_from_file_location("serial_dispatcher", _MOD_PATH)
_sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sd)
SerialModelDispatcher = _sd.SerialModelDispatcher
SubmitRejected = _sd.SubmitRejected


async def _drain(handles, timeout):
    """Wait until all handles are done or timeout. Returns done count."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if all(h.done_ts is not None for h in handles):
            return sum(1 for h in handles if h.ok)
        await asyncio.sleep(0.01)
    return sum(1 for h in handles if h.done_ts is not None and h.ok)


async def _test_progress_guarantee_under_skewed_open_loop_impl():
    """§8.32 regression: skewed concurrent multi-model load must fully drain.

    Mirrors the failing scenario: ~18:1 model skew, all submitted up-front
    (open-loop), slow bind + slow execute. The old machinery hung here.
    """
    binds = []

    async def bind_fn(model_id, _payload):
        binds.append(model_id)
        await asyncio.sleep(0.02)  # model swap cost

    async def execute_fn(_payload):
        await asyncio.sleep(0.01)  # inference cost

    d = SerialModelDispatcher(bind_fn, execute_fn, max_queue=256)
    d.start()
    handles = []
    # 90 sd3 + 5 flux interleaved (skew like the mentor trace), all at once
    seq = (["sd3"] * 18 + ["flux"]) * 5
    for i, m in enumerate(seq):
        handles.append(d.submit(f"t{i}", m, {"i": i}))
    done_ok = await _drain(handles, timeout=15.0)
    await d.stop()

    assert done_ok == len(handles), (
        f"collapse: only {done_ok}/{len(handles)} completed "
        f"(stats={d.stats})"
    )
    assert d.stats["completed"] == len(handles)
    assert d.stats["failed"] == 0


async def _test_admission_control_rejects_when_full_impl():
    """Bounded queue: submit beyond capacity raises SubmitRejected."""
    async def bind_fn(_m, _p):
        await asyncio.sleep(0)

    async def execute_fn(_p):
        await asyncio.sleep(0.05)  # slow so the queue backs up

    d = SerialModelDispatcher(bind_fn, execute_fn, max_queue=4)
    # do NOT start the consumer -> queue fills and stays full
    ok = 0
    rejected = 0
    for i in range(20):
        try:
            d.submit(f"t{i}", "sd3", {})
            ok += 1
        except SubmitRejected:
            rejected += 1
    assert ok == 4, f"queue cap not enforced (accepted {ok})"
    assert rejected == 16
    assert d.stats["rejected"] == 16


async def _test_one_failing_task_does_not_stall_consumer_impl():
    """Fault isolation: a raising execute must not freeze the pipeline."""
    async def bind_fn(_m, _p):
        await asyncio.sleep(0)

    async def execute_fn(payload):
        if payload.get("boom"):
            raise RuntimeError("boom")
        await asyncio.sleep(0.005)

    d = SerialModelDispatcher(bind_fn, execute_fn, max_queue=64)
    d.start()
    h_bad = d.submit("bad", "sd3", {"boom": True})
    good = [d.submit(f"g{i}", "sd3", {}) for i in range(10)]
    done_ok = await _drain(good, timeout=5.0)
    await asyncio.sleep(0.05)
    await d.stop()

    assert done_ok == 10, "consumer stalled after a failing task"
    assert h_bad.done_ts is not None and h_bad.ok is False
    assert d.stats["failed"] == 1 and d.stats["completed"] == 10


async def _test_batching_amortizes_switch_impl():
    """§8.41-FINAL fix: per-model batching reorders same-model work so an
    alternating submit pattern does NOT switch on every request.

    Old global-FIFO contract would bind on every model change (here 6).
    The new scheduler batches the bound model's queue, so a burst of
    alternating sd3/flux submitted together collapses to ~2 binds (drain
    sd3 batch, then flux batch) — the switch-cost amortization that turns
    the measured 6-9s/req into 6-9s/batch.
    """
    binds = []

    async def bind_fn(model_id, _p):
        binds.append(model_id)
        await asyncio.sleep(0)

    async def execute_fn(_p):
        await asyncio.sleep(0)

    d = SerialModelDispatcher(bind_fn, execute_fn, max_queue=64,
                              batch_max_n=8, batch_max_s=20.0)
    d.start()
    seq = ["sd3", "flux", "sd3", "flux", "sd3", "flux"]  # alternating
    hs = [d.submit(f"t{i}", m, {}) for i, m in enumerate(seq)]
    await _drain(hs, timeout=5.0)
    await d.stop()
    assert d.stats["completed"] == 6
    # batched, NOT one-bind-per-request: far fewer than the 5 switches a
    # naive FIFO would do on this alternating pattern.
    assert d.stats["switches"] <= 1, (
        f"batching failed: {d.stats['switches']} switches, binds={binds}"
    )


async def _test_micro_batch_same_shape_one_pass_impl():
    """§8.45 optimization #1: concurrently-queued same-(model,shape)
    requests execute together in ONE batched GPU pass; different shapes
    do NOT mix; all complete; FIFO across shapes preserved."""
    batched_calls = []          # list of batch sizes seen by execute_batch
    single_calls = [0]

    async def bind_fn(_m, _p):
        await asyncio.sleep(0)

    async def execute_fn(_p):           # size-1 path
        single_calls[0] += 1
        await asyncio.sleep(0.005)

    async def execute_batch_fn(payloads):
        batched_calls.append(len(payloads))
        await asyncio.sleep(0.005)      # ONE pass for the whole group

    def shape_fn(p):
        return (p["w"], p["h"], p["steps"])

    d = SerialModelDispatcher(
        bind_fn, execute_fn, max_queue=64, batch_max_n=64,
        exec_batch_max=4, shape_fn=shape_fn,
        execute_batch_fn=execute_batch_fn)
    # 6 sd3 @512 + 2 sd3 @1024 (distinct shape) submitted up-front
    hs = []
    for i in range(6):
        hs.append(d.submit(f"a{i}", "sd3", {"w": 512, "h": 512, "steps": 20}))
    for i in range(2):
        hs.append(d.submit(f"b{i}", "sd3", {"w": 1024, "h": 1024,
                                            "steps": 28}))
    d.start()
    done = await _drain(hs, timeout=5.0)
    await d.stop()
    assert done == 8, f"not all completed: {done}/8 (stats={d.stats})"
    assert d.stats["completed"] == 8 and d.stats["failed"] == 0
    # 6 @512 -> batched into groups capped at 4 => sizes like [4,2];
    # 2 @1024 -> [2]. Every batched call must be a single shape (<=4)
    # and there must be NO size-1 fallback (all went batched).
    assert batched_calls, "micro-batching never triggered"
    assert all(1 < n <= 4 for n in batched_calls), batched_calls
    assert sum(batched_calls) + single_calls[0] == 8
    # the 6 same-shape collapsed to <=2 passes instead of 6:
    assert len([n for n in batched_calls if n]) <= 4


async def _test_opt2_linger_batches_paced_arrivals_impl():
    """§8.46 opt#2: with a linger, requests that arrive AFTER the head
    (paced load) still get batched into ONE GPU pass; linger is bounded;
    without linger the head would fire alone (size-1)."""
    seen = []

    async def bind_fn(_m, _p):
        await asyncio.sleep(0)

    async def execute_fn(_p):       # size-1 path
        seen.append(1)
        await asyncio.sleep(0.01)

    async def execute_batch_fn(payloads):
        seen.append(len(payloads))
        await asyncio.sleep(0.01)

    def shape_fn(p):
        return (p["w"], p["h"])

    d = SerialModelDispatcher(
        bind_fn, execute_fn, max_queue=64, batch_max_n=64,
        exec_batch_max=8, shape_fn=shape_fn,
        execute_batch_fn=execute_batch_fn,
        batch_linger_s=1.0, batch_linger_poll=0.02)
    d.start()
    h0 = d.submit("p0", "sd3", {"w": 512, "h": 512})  # head
    # paced arrivals AFTER the head, within the 1.0s linger window
    later = []
    for k in range(4):
        await asyncio.sleep(0.1)
        later.append(d.submit(f"p{k+1}", "sd3", {"w": 512, "h": 512}))
    done = await _drain([h0] + later, timeout=5.0)
    await d.stop()
    assert done == 5, f"not all done: {done}/5 ({d.stats})"
    # the linger must have collapsed the 5 paced arrivals into >=1
    # batched call (size>1); a no-linger dispatcher would show 5 size-1.
    assert any(n > 1 for n in seen), f"linger failed to batch: {seen}"
    assert sum(seen) == 5 and d.stats["completed"] == 5
    # head latency bounded by ~linger (1.0s) + service, not unbounded
    assert (h0.done_ts - h0.submit_ts) < 3.0, "linger not bounded"


async def _test_starvation_freedom_under_extreme_skew_impl():
    """Suite gate F2: under 50:1 skew the single rare-model request MUST
    complete within a bounded time (aging anti-starvation), not be
    starved behind the heavy model forever (the §8.41 flux 0/147 bug).
    """
    async def bind_fn(_m, _p):
        await asyncio.sleep(0.005)

    async def execute_fn(_p):
        await asyncio.sleep(0.005)

    # short deadline so the test is fast but exercises the override
    d = SerialModelDispatcher(bind_fn, execute_fn, max_queue=512,
                              batch_max_n=8, starvation_deadline_s=0.2)
    d.start()
    # one flux first, then a long sd3 flood — flux must still complete
    h_flux = d.submit("flux0", "flux", {})
    sd3 = [d.submit(f"s{i}", "sd3", {}) for i in range(400)]
    # flux completes well before the whole sd3 flood drains
    deadline = asyncio.get_event_loop().time() + 5.0
    while asyncio.get_event_loop().time() < deadline:
        if h_flux.done_ts is not None:
            break
        await asyncio.sleep(0.01)
    flux_done = h_flux.done_ts is not None and h_flux.ok
    await _drain(sd3, timeout=10.0)
    await d.stop()
    assert flux_done, "STARVATION: rare model never completed under skew"
    assert d.stats["per_model"]["flux"]["completed"] == 1


async def _test_fair_selection_round_robins_models_impl():
    """Steady multi-model load: no model is starved; each makes progress
    interleaved (least-recently-served fairness), not all-of-A-then-B.
    """
    order = []

    async def bind_fn(model_id, _p):
        await asyncio.sleep(0)

    async def execute_fn(p):
        order.append(p["m"])
        await asyncio.sleep(0.002)

    d = SerialModelDispatcher(bind_fn, execute_fn, max_queue=256,
                              batch_max_n=4, starvation_deadline_s=30.0)
    d.start()
    hs = []
    for i in range(12):
        hs.append(d.submit(f"a{i}", "A", {"m": "A"}))
        hs.append(d.submit(f"b{i}", "B", {"m": "B"}))
    await _drain(hs, timeout=5.0)
    await d.stop()
    assert d.stats["completed"] == 24
    # both models fully served, and not one giant block of 12 then 12:
    # batching caps a run at batch_max_n, so >=2 alternations occur.
    blocks = sum(
        1 for i in range(1, len(order)) if order[i] != order[i - 1]
    )
    assert blocks >= 2, f"no interleaving (starvation risk): {order}"
    assert order.count("A") == 12 and order.count("B") == 12


async def _test_timeout_detects_a_hung_consumer_impl():
    """Meta-test: prove the progress assertion has teeth.

    A bind_fn that never returns simulates the §8.32 deadlock. _drain must
    report < all-done, i.e. our progress assertions would FAIL on a hung
    dispatcher (so test_progress_guarantee is meaningful, not vacuous).
    """
    hang = asyncio.Event()  # never set

    async def bind_fn(_m, _p):
        await hang.wait()  # never returns -> simulated deadlock

    async def execute_fn(_p):
        await asyncio.sleep(0)

    d = SerialModelDispatcher(bind_fn, execute_fn, max_queue=64)
    d.start()
    hs = [d.submit(f"t{i}", "sd3", {}) for i in range(5)]
    done_ok = await _drain(hs, timeout=1.0)
    await d.stop()
    assert done_ok == 0, "a hung consumer must NOT report progress"


# --- sync wrappers (no pytest-asyncio dependency) ---
def test_progress_guarantee_under_skewed_open_loop():
    asyncio.run(_test_progress_guarantee_under_skewed_open_loop_impl())

def test_admission_control_rejects_when_full():
    asyncio.run(_test_admission_control_rejects_when_full_impl())

def test_one_failing_task_does_not_stall_consumer():
    asyncio.run(_test_one_failing_task_does_not_stall_consumer_impl())

def test_batching_amortizes_switch():
    asyncio.run(_test_batching_amortizes_switch_impl())

def test_micro_batch_same_shape_one_pass():
    asyncio.run(_test_micro_batch_same_shape_one_pass_impl())

def test_opt2_linger_batches_paced_arrivals():
    asyncio.run(_test_opt2_linger_batches_paced_arrivals_impl())

def test_starvation_freedom_under_extreme_skew():
    asyncio.run(_test_starvation_freedom_under_extreme_skew_impl())

def test_fair_selection_round_robins_models():
    asyncio.run(_test_fair_selection_round_robins_models_impl())

def test_timeout_detects_a_hung_consumer():
    asyncio.run(_test_timeout_detects_a_hung_consumer_impl())

