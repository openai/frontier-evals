import asyncio

import pytest

from evmbench.agents.run import execute_agent_in_computer, run_agent_in_computer


class BlockingComputer:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.block = asyncio.Event()

    async def send_shell_command(self, _cmd):
        self.started.set()
        await self.block.wait()
        raise AssertionError("blocking command should not complete")


@pytest.mark.asyncio
async def test_agent_timeout_is_handled_locally() -> None:
    computer = BlockingComputer()

    await execute_agent_in_computer(
        computer,
        timeout=0.01,
        run_group_id="group",
        run_id="run",
        runs_dir="runs",
    )


@pytest.mark.asyncio
async def test_external_cancellation_propagates_from_agent_run() -> None:
    computer = BlockingComputer()
    task = asyncio.create_task(
        run_agent_in_computer(
            computer,
            timeout=3600,
            run_group_id="group",
            run_id="run",
            runs_dir="runs",
        )
    )

    await computer.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
