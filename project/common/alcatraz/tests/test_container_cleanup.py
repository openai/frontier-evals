import asyncio
import threading
from unittest.mock import MagicMock

import pytest

from alcatraz.clusters import local


@pytest.mark.asyncio
async def test_ports_remain_leased_until_container_removal_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    available_ports = {10000, 10001}
    monkeypatch.setattr(local, "_FREE_PORTS", available_ports.copy())
    cluster = local.LocalCluster(image="alpine:latest")
    cluster.docker_client = MagicMock()
    container = cluster.docker_client.containers.run.return_value

    loop = asyncio.get_running_loop()
    removal_started = asyncio.Event()
    finish_removal = threading.Event()

    def remove(*, force: bool) -> None:
        assert force
        loop.call_soon_threadsafe(removal_started.set)
        assert finish_removal.wait(timeout=10), "Container removal was not released"

    container.remove.side_effect = remove
    host_ports, _ = await cluster._create_container(container_ports=[80, 81])
    assert set(host_ports) == available_ports
    assert not local._FREE_PORTS

    cleanup = asyncio.create_task(cluster._exit_stack.aclose())
    try:
        await asyncio.wait_for(removal_started.wait(), timeout=5)
        # Another coroutine can run while Docker removal is pending. It must
        # not be able to allocate any of the container's host ports yet.
        assert not cleanup.done()
        assert not local._FREE_PORTS
    finally:
        finish_removal.set()
        await asyncio.wait_for(cleanup, timeout=5)

    container.remove.assert_called_once_with(force=True)
    assert local._FREE_PORTS == available_ports
