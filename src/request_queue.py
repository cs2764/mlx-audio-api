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
    def __init__(self, engines: list | None = None, max_size: int = 32, timeout: float = 300.0):
        """
        engines: list of TTSEngine instances, one per worker.
        If None or empty, falls back to legacy single-worker mode where
        the handler callable takes no arguments (for backward compatibility).
        """
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._engines = engines or []
        self._timeout = timeout

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
            task = asyncio.create_task(self._worker(idx, engine, executor))
            tasks.append(task)
        logger.info(f"Started {len(tasks)} inference worker(s)")
        return tasks

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
                # Legacy: handler takes no args
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
