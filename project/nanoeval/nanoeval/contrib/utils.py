import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerConfiguration,
    ComputerInterface,
    ComputerRuntime,
)


@asynccontextmanager
async def run_with_startup_timeout(
    runtime: ComputerRuntime,
    task: ComputerConfiguration,
    timeout: float,
) -> AsyncGenerator[ComputerInterface, None]:
    """
    Same as computer_runtime.run, but with a timeout on the context entry.

    Wraps only the context entry (runtime.run) in `asyncio.wait_for`, so that
    if the underlying VM/container takes too long to start, an asyncio.TimeoutError
    is raised. On both success or failure, the runtime context is properly closed.
    """
    cm = runtime.run(task)
    # timeout only the startup phase
    computer = await asyncio.wait_for(cm.__aenter__(), timeout)
    try:
        yield computer
    finally:
        await cm.__aexit__(None, None, None)
