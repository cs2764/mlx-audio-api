import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


class QueueFullError(Exception):
    pass


@dataclass
class QueueItem:
    # Callable that accepts one positional arg: the engine instance
    request: Callable[[Any], Any]
    future: asyncio.Future
    enqueue_time: float = field(default_factory=time.monotonic)


class RequestQueue:
    def __init__(
        self,
        engines: list | None = None,
        max_size: int = 32,
        timeout: float = 300.0,
        batch_window_ms: int = 0,
        max_batch_size: int = 1,
    ):
        """
        engines: list of TTSEngine instances, one per worker.
        batch_window_ms: if > 0 and engine supports batching, collect requests
                         for this many ms before dispatching as a single batch.
        max_batch_size: maximum number of requests per batch.

        Load balancing: shared asyncio.Queue (work-stealing).
        All workers compete on the same queue — whichever worker finishes first
        immediately picks up the next request. This is optimal for variable-length
        workloads (e.g. audiobook) because a fast worker never sits idle waiting
        for a slow one to finish.
        """
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._engines = engines or []
        self._timeout = timeout
        self._batch_window_ms = batch_window_ms
        self._max_batch_size = max(1, max_batch_size)

    async def enqueue(self, handler: Callable[[Any], Any]) -> Any:
        """
        Enqueue a callable handler(engine) -> result and wait for its result.
        Raises QueueFullError (→ 503) or asyncio.TimeoutError (→ 408).
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        item = QueueItem(request=handler, future=future)

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            raise QueueFullError("Request queue is full")

        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout=self._timeout)
        except asyncio.TimeoutError:
            future.cancel()
            raise

    def start_workers(self) -> list[asyncio.Task]:
        """Spawn one worker task per engine and return the task list."""
        tasks = []
        for idx, engine in enumerate(self._engines):
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"tts-worker-{idx}")
            supports_batch = getattr(engine, "supports_batch", False)
            if supports_batch and self._batch_window_ms > 0:
                task = asyncio.create_task(
                    self._batch_worker(idx, engine, executor)
                )
                logger.info(
                    f"[worker-{idx}] dynamic batching enabled "
                    f"(window={self._batch_window_ms}ms, max_batch={self._max_batch_size})"
                )
            else:
                task = asyncio.create_task(self._worker(idx, engine, executor))
            tasks.append(task)
        logger.info(f"Started {len(tasks)} inference worker(s)")
        return tasks

    # ── legacy single-worker (tests / compat) ────────────────────────────────

    async def worker(self):
        """Legacy single-worker coroutine (no-engine mode, for tests/compat)."""
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-worker-legacy")
        while True:
            item: QueueItem = await self._queue.get()
            if item.future.cancelled():
                self._queue.task_done()
                continue
            wait_time = time.monotonic() - item.enqueue_time
            logger.info(f"[worker-legacy] dequeued after {wait_time:.3f}s wait")
            start_time = time.monotonic()
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, item.request
                )
                if not item.future.cancelled():
                    item.future.set_result(result)
            except Exception as e:
                if not item.future.cancelled():
                    item.future.set_exception(e)
            finally:
                logger.info(f"[worker-legacy] done in {time.monotonic()-start_time:.3f}s")
                self._queue.task_done()

    # ── standard single-item worker ───────────────────────────────────────────

    async def _worker(self, idx: int, engine: Any, executor: ThreadPoolExecutor):
        """Single worker: pulls items from the shared queue and runs inference."""
        while True:
            item: QueueItem = await self._queue.get()

            if item.future.cancelled():
                self._queue.task_done()
                continue

            wait_time = time.monotonic() - item.enqueue_time
            logger.info(f"[worker-{idx}] dequeued after {wait_time:.3f}s wait")

            start_time = time.monotonic()
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    executor, item.request, engine
                )
                if not item.future.cancelled():
                    item.future.set_result(result)
            except Exception as e:
                if not item.future.cancelled():
                    item.future.set_exception(e)
            finally:
                inference_time = time.monotonic() - start_time
                logger.info(f"[worker-{idx}] inference done in {inference_time:.3f}s")
                self._queue.task_done()


    # ── dynamic batching worker (Qwen3 only, requires --batch-window-ms > 0) ──

    def _drain_nowait(self, batch: list[QueueItem]) -> list[QueueItem]:
        """Non-blocking: drain as many items as possible from the shared queue."""
        while len(batch) < self._max_batch_size:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item.future.cancelled():
                self._queue.task_done()
                continue
            batch.append(item)
        return batch

    async def _collect_batch(self, first_item: QueueItem) -> list[QueueItem]:
        """Greedy batch collection: drain immediately, then wait up to batch_window_ms."""
        batch = [first_item]
        self._drain_nowait(batch)
        if len(batch) < self._max_batch_size and self._batch_window_ms > 0:
            deadline = time.monotonic() + self._batch_window_ms / 1000.0
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    if item.future.cancelled():
                        self._queue.task_done()
                        continue
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
        return batch

    async def _batch_worker(self, idx: int, engine: Any, executor: ThreadPoolExecutor):
        """Batching worker: collects a batch then dispatches as a single inference call."""
        loop = asyncio.get_event_loop()
        while True:
            first_item: QueueItem = await self._queue.get()
            if first_item.future.cancelled():
                self._queue.task_done()
                continue

            batch = await self._collect_batch(first_item)
            batch_size = len(batch)
            avg_wait = sum(time.monotonic() - i.enqueue_time for i in batch) / batch_size
            logger.info(f"[worker-{idx}] dispatching batch of {batch_size} (avg wait {avg_wait:.3f}s)")
            start_time = time.monotonic()

            requests = [item.request for item in batch]

            def _run_batch(eng, reqs=requests):
                return eng.generate_batch(reqs)

            try:
                results: list = await loop.run_in_executor(executor, _run_batch, engine)
            except Exception as exc:
                results = [exc] * batch_size

            inference_time = time.monotonic() - start_time
            logger.info(
                f"[worker-{idx}] batch of {batch_size} done in {inference_time:.3f}s "
                f"({inference_time / batch_size:.3f}s/req)"
            )

            for item, result in zip(batch, results):
                if not item.future.cancelled():
                    if isinstance(result, Exception):
                        item.future.set_exception(result)
                    else:
                        item.future.set_result(result)
                self._queue.task_done()
