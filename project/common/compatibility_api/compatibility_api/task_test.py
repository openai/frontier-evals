"""Validate exact Docker registry boundaries for portable computer tasks."""

from types import SimpleNamespace

import pytest
from compatibility_api.task import CompatibleComputerTask

from nanoeval.solvers.computer_tasks.code_execution_interface import NetworkMode


def _fake_task(image: str) -> SimpleNamespace:
    return SimpleNamespace(
        docker_image=image,
        network_mode=NetworkMode.NONE,
        volumes_config={},
        volume_mounts=[],
        side_images=[],
        enable_cyber_only_features=False,
        docker_compose_yaml=None,
        mem_limit=None,
        shm_size=None,
        timeout=None,
        alcatraz_limits=None,
        jupyter_setup=None,
        disk_mount_path=None,
    )


@pytest.mark.parametrize(
    "image",
    [
        "docker.io/library/python:3.11",
        "docker.io/namespace/image:tag",
        "jupyter/base-notebook:3.11",
    ],
)
def test_compatible_registry_images_remain_supported(image: str) -> None:
    task = _fake_task(image)
    assert CompatibleComputerTask._check_for_compatibility(task) is task


@pytest.mark.parametrize(
    "image", ["docker.io.example.com/malicious", "docker.ioevil/image", "example.org/image"]
)
def test_lookalike_registry_names_are_rejected(image: str) -> None:
    with pytest.raises(AssertionError, match="shared CR"):
        CompatibleComputerTask._check_for_compatibility(_fake_task(image))
