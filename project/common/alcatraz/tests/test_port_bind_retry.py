import random
from unittest.mock import MagicMock, call

import pytest
from docker.errors import APIError, ImageNotFound, NotFound
from requests import Response

from alcatraz.clusters import local


def docker_error(message: str, status: int = 500) -> APIError:
    response = Response()
    response.status_code = status
    return APIError(message, response=response, explanation=message)


@pytest.fixture
def cluster(monkeypatch: pytest.MonkeyPatch) -> local.LocalCluster:
    monkeypatch.setattr(local, "_FREE_PORTS", set(range(10000, 10020)))
    monkeypatch.setattr(random, "choice", min)
    result = local.LocalCluster(image="alpine:latest")
    result.docker_client = MagicMock()
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Bind for 0.0.0.0:10000 failed: port is already allocated",
        "failed to bind host port 0.0.0.0:10000/tcp: address already in use",
        "Error starting userland proxy: listen tcp4 0.0.0.0:10000: bind: address already in use",
    ],
)
async def test_binding_conflict_retries_with_new_ports(
    cluster: local.LocalCluster, message: str
) -> None:
    client = MagicMock()
    cluster.docker_client = client
    failed, started = MagicMock(), MagicMock()
    client.containers.create.side_effect = [failed, started]
    failed.start.side_effect = docker_error(message)
    operations = MagicMock()
    operations.attach_mock(client.containers.create, "create")
    operations.attach_mock(failed.remove, "remove_failed")

    ports, container = await cluster._create_container(
        container_ports=[80, 81], image="alpine:latest", name="same-name", detach=True, remove=True
    )

    assert ports == [10002, 10003]
    assert container is started
    assert [entry[0] for entry in operations.mock_calls] == ["create", "remove_failed", "create"]
    failed.remove.assert_called_once_with(force=True)
    for index, create_call in enumerate(client.containers.create.call_args_list):
        assert create_call.kwargs["name"] == "same-name"
        assert create_call.kwargs["auto_remove"] is True
        assert "remove" not in create_call.kwargs
        assert create_call.kwargs["ports"] == {
            "80/tcp": 10000 + 2 * index,
            "81/tcp": 10001 + 2 * index,
        }
    # Failed candidates are returned; successful candidates stay leased.
    assert local._FREE_PORTS == set(range(10000, 10020)) - set(ports)
    await cluster._exit_stack.aclose()
    started.remove.assert_called_once_with(force=True)
    assert local._FREE_PORTS == set(range(10000, 10020))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message,status",
    [
        ("network not found", 404),
        ("Conflict. The container name is already in use", 409),
        ("OCI runtime create failed", 500),
        ("address already in use", 500),
        ("port is already allocated", 403),
    ],
)
async def test_other_start_errors_are_not_retried(
    cluster: local.LocalCluster, message: str, status: int
) -> None:
    client = MagicMock()
    cluster.docker_client = client
    container = client.containers.create.return_value
    failure = docker_error(message, status)
    container.start.side_effect = failure
    with pytest.raises(APIError) as caught:
        await cluster._create_container(container_ports=[80], image="alpine:latest", detach=True)
    assert caught.value is failure
    client.containers.create.assert_called_once()
    container.remove.assert_called_once_with(force=True)
    assert local._FREE_PORTS == set(range(10000, 10020))


@pytest.mark.asyncio
async def test_create_error_does_not_remove_an_existing_container(
    cluster: local.LocalCluster,
) -> None:
    client = MagicMock()
    cluster.docker_client = client
    failure = docker_error("Conflict. The container name is already in use", 409)
    client.containers.create.side_effect = failure
    with pytest.raises(APIError) as caught:
        await cluster._create_container(
            container_ports=[80], image="alpine:latest", name="existing", detach=True
        )
    assert caught.value is failure
    client.containers.create.assert_called_once()
    client.containers.get.assert_not_called()
    client.containers.create.return_value.remove.assert_not_called()
    assert local._FREE_PORTS == set(range(10000, 10020))


@pytest.mark.asyncio
@pytest.mark.parametrize("pool_size", [1, 20])
async def test_conflicts_stop_at_pool_exhaustion_or_retry_limit(
    cluster: local.LocalCluster, monkeypatch: pytest.MonkeyPatch, pool_size: int
) -> None:
    pool = set(range(10000, 10000 + pool_size))
    monkeypatch.setattr(local, "_FREE_PORTS", pool.copy())
    client = MagicMock()
    cluster.docker_client = client
    failures = [docker_error("port is already allocated") for _ in range(5)]
    containers = [MagicMock() for _ in failures]
    for container, failure in zip(containers, failures):
        container.start.side_effect = failure
    client.containers.create.side_effect = containers
    attempts = min(pool_size, 5)
    with pytest.raises(APIError) as caught:
        await cluster._create_container(container_ports=[80], image="alpine:latest", detach=True)
    assert caught.value is failures[attempts - 1]
    assert client.containers.create.call_count == attempts
    assert [c.kwargs["ports"]["80/tcp"] for c in client.containers.create.call_args_list] == list(
        range(10000, 10000 + attempts)
    )
    for container in containers[:attempts]:
        container.remove.assert_called_once_with(force=True)
    assert local._FREE_PORTS == pool


@pytest.mark.asyncio
async def test_auto_removed_failed_container_can_be_retried(cluster: local.LocalCluster) -> None:
    client = MagicMock()
    cluster.docker_client = client
    failed, started = MagicMock(), MagicMock()
    failed.start.side_effect = docker_error("port is already allocated")
    failed.remove.side_effect = NotFound("already removed")
    client.containers.create.side_effect = [failed, started]
    ports, container = await cluster._create_container(
        container_ports=[80], image="alpine:latest", detach=True, remove=True
    )
    assert ports == [10001]
    assert container is started
    await cluster._exit_stack.aclose()


@pytest.mark.asyncio
async def test_cleanup_error_stops_retries(cluster: local.LocalCluster) -> None:
    client = MagicMock()
    cluster.docker_client = client
    failed = client.containers.create.return_value
    failed.start.side_effect = docker_error("port is already allocated")
    cleanup_error = docker_error("removal failed")
    failed.remove.side_effect = cleanup_error
    with pytest.raises(APIError) as caught:
        await cluster._create_container(container_ports=[80], image="alpine:latest", detach=True)
    assert caught.value is cleanup_error
    client.containers.create.assert_called_once()


@pytest.mark.asyncio
async def test_missing_image_is_pulled_before_create(cluster: local.LocalCluster) -> None:
    client = MagicMock()
    cluster.docker_client = client
    container = MagicMock()
    client.containers.create.side_effect = [ImageNotFound("missing"), container]
    ports, _ = await cluster._create_container(
        container_ports=[80], image="alpine:latest", platform="linux/amd64", detach=True
    )
    assert ports == [10000]
    client.images.pull.assert_called_once_with("alpine:latest", platform="linux/amd64")
    assert client.containers.create.call_args_list[0] == client.containers.create.call_args_list[1]
    container.start.assert_called_once_with()
    await cluster._exit_stack.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("forced", [None, [12345]])
async def test_fixed_or_unpublished_ports_keep_run_behavior(
    cluster: local.LocalCluster, forced: list[int] | None
) -> None:
    client = MagicMock()
    cluster.docker_client = client
    failure = docker_error("port is already allocated")
    client.containers.run.side_effect = failure
    with pytest.raises(APIError) as caught:
        await cluster._create_container(
            container_ports=[80] if forced else [],
            force_host_ports=forced,
            image="alpine:latest",
            detach=True,
        )
    assert caught.value is failure
    assert client.containers.run.call_args_list == [
        call(image="alpine:latest", detach=True, ports={"80/tcp": 12345} if forced else {})
    ]
    client.containers.create.assert_not_called()
    assert local._FREE_PORTS == set(range(10000, 10020))
