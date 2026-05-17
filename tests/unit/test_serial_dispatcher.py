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


async def _test_bind_only_on_model_change_impl():
    """Switch-on-change: contiguous same-model runs cause one bind."""
    binds = []

    async def bind_fn(model_id, _p):
        binds.append(model_id)
        await asyncio.sleep(0)

    async def execute_fn(_p):
        await asyncio.sleep(0)

    d = SerialModelDispatcher(bind_fn, execute_fn, max_queue=64)
    d.start()
    seq = ["sd3", "sd3", "sd3", "flux", "flux", "sd3"]
    hs = [d.submit(f"t{i}", m, {}) for i, m in enumerate(seq)]
    await _drain(hs, timeout=5.0)
    await d.stop()
    # sd3 -> flux -> sd3 == 3 binds, not 6
    assert binds == ["sd3", "flux", "sd3"], f"over-binding: {binds}"
    assert d.stats["binds"] == 3


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

def test_bind_only_on_model_change():
    asyncio.run(_test_bind_only_on_model_change_impl())

def test_timeout_detects_a_hung_consumer():
    asyncio.run(_test_timeout_detects_a_hung_consumer_impl())

