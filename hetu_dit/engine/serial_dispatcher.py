"""Serial model dispatcher — deadlock-free multi-model dispatch.

Replaces the get_ready_executor_or_reconfigure / search_reconfigure_worker /
waiting_reconfigure_tasks / waiting_worker_tasks machinery for the common
(single executor pool) case. See context.md §8.32/§8.33 for the root cause
of the throughput collapse this fixes.

Why this design is collapse-proof
---------------------------------
The old path had a circular wait: the single serial consumer blocked on a
future that could only be resolved by a task completion, but task
completions could only happen if the consumer kept dispatching. Once the
in-flight set drained, nothing could ever resolve the futures -> GPU 0%,
zero throughput.

This dispatcher has exactly ONE consumer coroutine that owns the worker
pool. Its loop is unconditionally progress-making:

    pop task  ->  (rebind model if changed)  ->  execute  ->  mark done

It never awaits an inter-coroutine future whose resolution depends on the
consumer itself. `bind` and `execute` are bounded operations (a model load
+ an inference). Therefore the queue always drains at the single-GPU
service rate: under overload latency grows and admission control sheds
load, but the GPU stays busy and the system never deadlocks.

Admission control: a bounded queue. When full, submit() rejects
immediately (backpressure) instead of accepting unbounded work that can
never complete.

The class is intentionally Ray/GPU-free so it is unit-testable in
isolation; the engine injects real `bind_fn` / `execute_fn` coroutines.
"""
import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, Optional


class SubmitRejected(Exception):
    """Raised/returned when the admission queue is full (backpressure)."""


class TaskHandle:
    """Tracks one submitted task's lifecycle for status polling."""

    __slots__ = ("task_id", "model_id", "submit_ts", "start_ts", "done_ts",
                 "ok", "error")

    def __init__(self, task_id: str, model_id: str):
        self.task_id = task_id
        self.model_id = model_id
        self.submit_ts = time.time()
        self.start_ts: Optional[float] = None
        self.done_ts: Optional[float] = None
        self.ok: bool = False
        self.error: Optional[str] = None


class SerialModelDispatcher:
    """One consumer, one worker pool, model-switch on demand, never deadlocks.

    Parameters
    ----------
    bind_fn(model_id, payload) -> Awaitable
        Make the worker pool ready to serve `model_id` (model load / swap).
        Called only when the requested model differs from the bound one.
    execute_fn(payload) -> Awaitable
        Run one request on the currently-bound model.
    max_queue:
        Admission bound. submit() beyond this rejects (backpressure).
    """

    def __init__(
        self,
        bind_fn: Callable[[str, Any], Awaitable[None]],
        execute_fn: Callable[[Any], Awaitable[Any]],
        max_queue: int = 256,
    ):
        self._bind = bind_fn
        self._execute = execute_fn
        self._q: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._bound_model: Optional[str] = None
        self._handles: Dict[str, TaskHandle] = {}
        self._consumer: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()
        # observability
        self.stats = {
            "submitted": 0,
            "rejected": 0,
            "completed": 0,
            "failed": 0,
            "binds": 0,
        }

    def start(self) -> None:
        if self._consumer is None:
            self._consumer = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._stopped.set()
        if self._consumer is not None:
            self._consumer.cancel()
            try:
                await self._consumer
            except asyncio.CancelledError:
                pass

    def submit(self, task_id: str, model_id: str, payload: Any) -> TaskHandle:
        """Non-blocking admission. Rejects (raises) when the queue is full.

        Returns a TaskHandle the caller can poll for completion. Never
        blocks the caller, so the HTTP/queue layer cannot head-of-line
        stall on a slow model.
        """
        if self._q.full():
            self.stats["rejected"] += 1
            raise SubmitRejected(
                f"admission queue full ({self._q.maxsize}); shedding load"
            )
        h = TaskHandle(task_id, model_id)
        self._handles[task_id] = h
        # put_nowait is safe: we just checked .full() and we are the only
        # producer path that matters; if it races, treat as rejection.
        try:
            self._q.put_nowait((task_id, model_id, payload))
        except asyncio.QueueFull:
            del self._handles[task_id]
            self.stats["rejected"] += 1
            raise SubmitRejected("admission queue full (race); shedding load")
        self.stats["submitted"] += 1
        return h

    def get_handle(self, task_id: str) -> Optional[TaskHandle]:
        return self._handles.get(task_id)

    @property
    def queue_depth(self) -> int:
        return self._q.qsize()

    async def _run(self) -> None:
        """The single, unconditionally-progressing consumer loop."""
        while not self._stopped.is_set():
            try:
                task_id, model_id, payload = await self._q.get()
            except asyncio.CancelledError:
                break
            h = self._handles.get(task_id)
            try:
                if model_id != self._bound_model:
                    # model switch — bounded op (load/swap). Reusing the
                    # resident executor; NO destroy, NO reconfigure futures.
                    await self._bind(model_id, payload)
                    self._bound_model = model_id
                    self.stats["binds"] += 1
                if h is not None:
                    h.start_ts = time.time()
                await self._execute(payload)
                if h is not None:
                    h.done_ts = time.time()
                    h.ok = True
                self.stats["completed"] += 1
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001 — one bad task must not
                # kill the consumer (that would reintroduce a stall).
                if h is not None:
                    h.done_ts = time.time()
                    h.ok = False
                    h.error = repr(e)
                self.stats["failed"] += 1
            finally:
                self._q.task_done()
