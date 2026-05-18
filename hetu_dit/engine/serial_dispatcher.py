"""Serial model dispatcher — deadlock-free, fair, switch-cost-aware.

Replaces the get_ready_executor_or_reconfigure / search_reconfigure_worker /
waiting_reconfigure_tasks machinery for the common (single executor pool)
case. See context.md §8.32/§8.33 (collapse root cause) and §8.41-FINAL
(the fairness/starvation bottleneck this scheduler fixes).

Why collapse-proof (unchanged invariant)
-----------------------------------------
Exactly ONE consumer coroutine owns the worker pool. Its loop is
unconditionally progress-making:

    pick task -> (rebind model if changed) -> execute -> mark done

It never awaits an inter-coroutine future whose resolution depends on the
consumer itself. The only thing it waits on is `_work` — an Event set by
the *producer* (submit), never by the consumer — so there is no circular
wait. The queue therefore always drains at the single-GPU service rate.

Why fair + switch-cost-aware (the §8.41-FINAL fix)
--------------------------------------------------
The old design was a single global FIFO. Under industrial load imbalance
(one model ~95% of traffic) a rare model's request sits behind thousands
of heavy-model requests and is *starved indefinitely* (measured: flux
0/147 completions). And FIFO interleaving of two models forces a 6-9s
model switch on nearly every request (switch-thrash).

This dispatcher keeps **per-model FIFO queues** and a scheduler that:

  1. **Batches** the bound model: once serving model M, keep serving M
     (no switch) until M's queue drains OR a batch quantum is hit
     (`batch_max_n` requests or `batch_max_s` seconds). This amortizes
     the expensive model switch over a whole batch.
  2. **Aging anti-starvation (hard guarantee):** every model's oldest
     waiting task has an age. If any *other* model has a task older than
     `starvation_deadline_s`, the scheduler force-switches to the
     longest-waiting model next, regardless of batch quantum. This makes
     starvation-freedom a *bounded-time guarantee* (suite gate F2): no
     model with pending work waits longer than
     ~(starvation_deadline_s + one batch service time).
  3. **Fair selection** among eligible models: least-recently-served
     (round-robin-ish) so steady skew still gives the rare model its
     turn without waiting for the deadline.

Admission control: a bounded *total* depth across all per-model queues.
submit() beyond it rejects immediately (backpressure) — graceful
degradation, never unbounded buildup.

Ray/GPU-free by design → unit-testable in isolation; the engine injects
real `bind_fn` / `execute_fn` coroutines.
"""
import asyncio
import time
from collections import deque
from typing import Any, Awaitable, Callable, Deque, Dict, Optional, Tuple


class SubmitRejected(Exception):
    """Raised when the admission bound is hit (backpressure)."""


class TaskHandle:
    """Tracks one submitted task's lifecycle for status polling."""

    __slots__ = ("task_id", "model_id", "submit_ts", "start_ts", "done_ts",
                 "ok", "error", "bind_s", "infer_s", "switched")

    def __init__(self, task_id: str, model_id: str):
        self.task_id = task_id
        self.model_id = model_id
        self.submit_ts = time.time()
        self.start_ts: Optional[float] = None
        self.done_ts: Optional[float] = None
        self.ok: bool = False
        self.error: Optional[str] = None
        # per-request latency decomposition of the service phase, filled
        # by the engine's _serial_execute once known (the multimodel
        # switch tax vs pure inference — the heart of "is switching slow")
        self.bind_s: Optional[float] = None     # model-switch wall (0 if hot)
        self.infer_s: Optional[float] = None    # pure inference wall
        self.switched: Optional[bool] = None    # did this request switch?

    def set_switch_cost(self, bind_s, infer_s, switched):
        self.bind_s = bind_s
        self.infer_s = infer_s
        self.switched = switched


class SerialModelDispatcher:
    """One consumer; per-model fair, switch-cost-aware scheduling.

    Parameters
    ----------
    bind_fn(model_id, payload) -> Awaitable
        Make the worker pool ready to serve `model_id` (model load/swap).
        Called only when the model actually changes.
    execute_fn(payload) -> Awaitable
        Run one request on the currently-bound model.
    max_queue:
        Total admission bound across all per-model queues.
    batch_max_n / batch_max_s:
        Keep serving the bound model up to this many requests / seconds
        before considering a switch (switch-cost amortization).
    starvation_deadline_s:
        Hard fairness bound: a model with work waiting longer than this is
        force-scheduled next regardless of batching.
    """

    def __init__(
        self,
        bind_fn: Callable[[str, Any], Awaitable[None]],
        execute_fn: Callable[[Any], Awaitable[Any]],
        max_queue: int = 256,
        batch_max_n: int = 8,
        batch_max_s: float = 20.0,
        starvation_deadline_s: float = 30.0,
        exec_batch_max: int = 1,
        shape_fn: Optional[Callable[[Any], Any]] = None,
        execute_batch_fn: Optional[Callable[[list], Awaitable[Any]]] = None,
        batch_linger_s: float = 0.0,
        batch_linger_poll: float = 0.1,
    ):
        self._bind = bind_fn
        self._execute = execute_fn
        self._max_queue = max_queue
        self._batch_max_n = batch_max_n
        self._batch_max_s = batch_max_s
        self._starv = starvation_deadline_s
        # SOTA micro-batching (§8.45 optimization #1): execute up to
        # exec_batch_max concurrently-queued requests of the SAME model
        # AND SAME shape (shape_fn(payload) key) in ONE GPU pass via the
        # diffusion batch dim. Default 1 + no fns = EXACT prior behaviour
        # (zero-risk; the optimization is opt-in & benchmark-gated).
        self._exec_batch_max = max(1, exec_batch_max)
        self._shape_fn = shape_fn
        self._execute_batch = execute_batch_fn
        # opt#2 (§8.46): dynamic batch formation. Under PACED industrial
        # arrivals (e.g. 1 req/s) only ~1 same-shape request is queued at
        # any instant, so opt#1's gather yields size-1 batches and the
        # GPU stays serial-pass-bound (steady_skew: 79% queue, 57.5s p95).
        # Linger up to batch_linger_s accumulating same-shape arrivals
        # before firing the GPU pass (vLLM/Sarathi-style continuous
        # batching) — trades <=linger added latency for N× throughput so
        # the queue collapses. 0 = off (safe default / A-B baseline).
        self._batch_linger_s = max(0.0, batch_linger_s)
        self._batch_linger_poll = max(0.01, batch_linger_poll)

        # per-model FIFO queues of (task_id, payload, enqueue_ts)
        self._queues: Dict[str, Deque[Tuple[str, Any, float]]] = {}
        self._depth = 0  # total across all queues (admission accounting)
        self._bound_model: Optional[str] = None
        self._served_in_batch = 0
        self._batch_started_ts = 0.0
        self._last_served_ts: Dict[str, float] = {}

        self._handles: Dict[str, TaskHandle] = {}
        self._work = asyncio.Event()       # set by producer only
        self._consumer: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

        self.stats = {
            "submitted": 0, "rejected": 0, "completed": 0, "failed": 0,
            "binds": 0, "switches": 0,
            "per_model": {},          # model -> {submitted,completed,failed}
            "max_wait_s": 0.0,        # worst observed enqueue->start wait
        }

    # ---- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if self._consumer is None:
            self._consumer = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopped.set()
        self._work.set()  # unblock the consumer so it can observe stop
        if self._consumer is not None:
            self._consumer.cancel()
            try:
                await self._consumer
            except asyncio.CancelledError:
                pass

    # ---- admission -----------------------------------------------------
    def submit(self, task_id: str, model_id: str, payload: Any) -> TaskHandle:
        """Non-blocking admission. Rejects when total depth is full.

        Never blocks the caller, so the HTTP layer cannot head-of-line
        stall on a slow model.
        """
        if self._depth >= self._max_queue:
            self.stats["rejected"] += 1
            raise SubmitRejected(
                f"admission full (depth={self._depth}/{self._max_queue}); "
                f"shedding load"
            )
        h = TaskHandle(task_id, model_id)
        self._handles[task_id] = h
        self._queues.setdefault(model_id, deque()).append(
            (task_id, payload, time.time())
        )
        self._depth += 1
        self.stats["submitted"] += 1
        pm = self.stats["per_model"].setdefault(
            model_id, {"submitted": 0, "completed": 0, "failed": 0}
        )
        pm["submitted"] += 1
        self._work.set()
        return h

    def get_handle(self, task_id: str) -> Optional[TaskHandle]:
        return self._handles.get(task_id)

    @property
    def queue_depth(self) -> int:
        return self._depth

    # ---- scheduling ----------------------------------------------------
    def _oldest_age(self, model_id: str, now: float) -> float:
        q = self._queues.get(model_id)
        if not q:
            return -1.0
        return now - q[0][2]  # head enqueue_ts

    def _pick_model(self, now: float) -> Optional[str]:
        """Choose which model to serve next.

        Priority:
          (1) starvation override — any model whose head task waited
              longer than starvation_deadline_s; pick the oldest such.
          (2) stay on the bound model while it has work AND its batch
              quantum is not exhausted (switch-cost amortization).
          (3) otherwise least-recently-served among models with work
              (fair round-robin).
        """
        nonempty = [m for m, q in self._queues.items() if q]
        if not nonempty:
            return None

        # (1) hard anti-starvation
        starving = [
            (self._oldest_age(m, now), m)
            for m in nonempty
            if m != self._bound_model
            and self._oldest_age(m, now) > self._starv
        ]
        if starving:
            starving.sort(reverse=True)  # longest-waiting first
            return starving[0][1]

        # (2) keep batching the currently bound model
        if self._bound_model in self._queues and self._queues[
            self._bound_model
        ]:
            within_n = self._served_in_batch < self._batch_max_n
            within_s = (now - self._batch_started_ts) < self._batch_max_s
            if within_n and within_s:
                return self._bound_model

        # (3) fair: least-recently-served model with work
        return min(
            nonempty, key=lambda m: self._last_served_ts.get(m, 0.0)
        )

    async def _run(self) -> None:
        """Single, unconditionally-progressing consumer loop."""
        while not self._stopped.is_set():
            if self._depth == 0:
                self._work.clear()
                try:
                    await self._work.wait()
                except asyncio.CancelledError:
                    break
                if self._stopped.is_set():
                    break
            now = time.time()
            model_id = self._pick_model(now)
            if model_id is None:
                continue
            q = self._queues[model_id]
            task_id, payload, enq_ts = q.popleft()
            self._depth -= 1
            # ---- micro-batch gather: same model + same shape, FIFO,
            # cap exec_batch_max. Disabled (size-1) by default => exact
            # prior one-at-a-time behaviour.
            batch = [(task_id, payload, enq_ts)]
            if (self._exec_batch_max > 1 and self._shape_fn is not None
                    and self._execute_batch is not None):
                try:
                    key = self._shape_fn(payload)
                    keep = deque()
                    while q and len(batch) < self._exec_batch_max:
                        t2, p2, e2 = q.popleft()
                        if self._shape_fn(p2) == key:
                            batch.append((t2, p2, e2))
                            self._depth -= 1
                        else:
                            keep.append((t2, p2, e2))
                    # restore the non-matching ones we skipped, in order,
                    # ahead of the remaining queue (FIFO preserved)
                    while keep:
                        q.appendleft(keep.pop())
                except Exception:  # noqa: BLE001 — never wedge on shape_fn
                    batch = [(task_id, payload, enq_ts)]
                # opt#2: dynamic linger — keep accumulating same-shape
                # arrivals for up to batch_linger_s. Bounded (adds <=
                # linger to the head request only) and fairness-safe:
                # abort the moment another model is starving so the
                # §8.41 starvation guarantee is preserved.
                if (self._batch_linger_s > 0.0
                        and len(batch) < self._exec_batch_max):
                    try:
                        skey = self._shape_fn(payload)
                        _lt0 = time.time()
                        while (len(batch) < self._exec_batch_max
                               and (time.time() - _lt0)
                               < self._batch_linger_s
                               and not self._stopped.is_set()):
                            # fairness: don't linger if another model has
                            # a task older than the starvation deadline.
                            if any(self._oldest_age(m, time.time())
                                   > self._starv
                                   for m in self._queues
                                   if m != model_id):
                                break
                            await asyncio.sleep(self._batch_linger_poll)
                            keep2 = deque()
                            while q and len(batch) < self._exec_batch_max:
                                t3, p3, e3 = q.popleft()
                                if self._shape_fn(p3) == skey:
                                    batch.append((t3, p3, e3))
                                    self._depth -= 1
                                else:
                                    keep2.append((t3, p3, e3))
                            while keep2:
                                q.appendleft(keep2.pop())
                    except asyncio.CancelledError:
                        break
                    except Exception:  # noqa: BLE001
                        pass
            hs = [self._handles.get(t) for t, _, _ in batch]
            try:
                if model_id != self._bound_model:
                    await self._bind(model_id, payload)
                    if self._bound_model is not None:
                        self.stats["switches"] += 1
                    self._bound_model = model_id
                    self.stats["binds"] += 1
                    self._served_in_batch = 0
                    self._batch_started_ts = time.time()
                self._served_in_batch += len(batch)
                self._last_served_ts[model_id] = time.time()
                _now = time.time()
                for (t, _p, e), hh in zip(batch, hs):
                    if hh is not None:
                        hh.start_ts = _now
                        wait_s = _now - e
                        if wait_s > self.stats["max_wait_s"]:
                            self.stats["max_wait_s"] = wait_s
                if len(batch) == 1:
                    await self._execute(batch[0][1])
                else:
                    await self._execute_batch([p for _, p, _ in batch])
                _d = time.time()
                pm = self.stats["per_model"].setdefault(
                    model_id, {"submitted": 0, "completed": 0, "failed": 0})
                for hh in hs:
                    if hh is not None:
                        hh.done_ts = _d
                        hh.ok = True
                self.stats["completed"] += len(batch)
                pm["completed"] += len(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001 — one bad batch must not
                # kill the consumer (that would reintroduce a stall).
                _d = time.time()
                for hh in hs:
                    if hh is not None:
                        hh.done_ts = _d
                        hh.ok = False
                        hh.error = repr(e)
                self.stats["failed"] += len(batch)
                self.stats["per_model"].setdefault(
                    model_id, {"submitted": 0, "completed": 0, "failed": 0}
                )["failed"] += len(batch)
