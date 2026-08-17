from contextlib import asynccontextmanager

import pytest

from paperbench.computer_utils import start_computer_with_retry


class FakeRuntime:
    def __init__(self, startup_failures: int = 0) -> None:
        self.startup_failures = startup_failures
        self.enter_count = 0
        self.exit_count = 0
        self.computer = object()

    @asynccontextmanager
    async def run(self, _config):
        self.enter_count += 1
        if self.enter_count <= self.startup_failures:
            raise ValueError("startup failed")
        try:
            yield self.computer
        finally:
            self.exit_count += 1


@pytest.mark.asyncio
async def test_start_computer_retries_only_startup_failures() -> None:
    runtime = FakeRuntime(startup_failures=2)

    async with start_computer_with_retry(
        runtime,
        object(),
        exception_types=ValueError,
        max_attempts=3,
    ) as computer:
        assert computer is runtime.computer

    assert runtime.enter_count == 3
    assert runtime.exit_count == 1


@pytest.mark.asyncio
async def test_start_computer_does_not_retry_caller_body() -> None:
    runtime = FakeRuntime()

    with pytest.raises(ValueError, match="caller failed"):
        async with start_computer_with_retry(
            runtime,
            object(),
            exception_types=ValueError,
            max_attempts=3,
        ):
            raise ValueError("caller failed")

    assert runtime.enter_count == 1
    assert runtime.exit_count == 1
