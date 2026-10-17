from __future__ import annotations

import asyncio
import logging
import os
import resource
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, contextmanager
from typing import Any, Coroutine, Generator

import structlog
from structlog.contextvars import bound_contextvars

from nanoeval._aiomonitor import start_aiomonitor
from nanoeval.library_config import get_library_config

logger = structlog.stdlib.get_logger(component=__name__)


global_exit_stack = AsyncExitStack()
"""
The global exit stack can be used to represent closable global state which will be
cleaned up when the program exits.
"""


def nanoeval_logging() -> None:
    logging.captureWarnings(True)
    get_library_config().setup_logging()


@contextmanager
def properly_closed_thread_pool(n_threads: int = 10_000) -> Generator[None, None, None]:
    # Use a very large thread pool executor so we can saturate engines
    executor = ThreadPoolExecutor(n_threads)
    try:
        asyncio.get_running_loop().set_default_executor(executor)
        yield
    finally:
        # Properly close the thread pool executor
        # See the great post here: https://www.roguelynn.com/words/asyncio-sync-and-threaded/
        logger.info("Shutting down executor")
        executor.shutdown(wait=False)

        logging.info(f"Releasing {len(executor._threads)} threads from executor")
        for thread in executor._threads:
            try:
                thread._tstate_lock.release()  # type: ignore
            except Exception:
                pass


def _raise_open_file_limit(target: int = 131_072) -> None:
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit == resource.RLIM_INFINITY:
        return

    target_soft_limit = (
        target
        if hard_limit == resource.RLIM_INFINITY
        else min(target, hard_limit)
    )
    if soft_limit < target_soft_limit:
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (target_soft_limit, hard_limit),
        )


async def _main_process_async_entrypoint(entry: Coroutine[Any, Any, None]) -> None:
    with properly_closed_thread_pool():
        nanoeval_logging()
        async with global_exit_stack:
            with (
                # Must be inside the thread pool executor, because we don't
                # want to close the thread pool executor before uploading logs
                bound_contextvars(pid=os.getpid()),
                start_aiomonitor(),
            ):
                await entry


def nanoeval_entrypoint(entry: Coroutine[Any, Any, None]) -> None:
    """
    Use at the beginning of your eval file. This configures nanoeval in high
    performance + easy debugging mode. It sets up:

    * default asyncio thread pool executor of 10_000 threads
    """

    # Raise the open file limit as far as the host allows, up to the target
    # needed by nanoeval's default concurrency. Do not attempt to raise the
    # process hard limit, which is not permitted on many constrained hosts.
    _raise_open_file_limit()

    asyncio.run(_main_process_async_entrypoint(entry))
