import platform
from pathlib import Path

import pandas as pd
from docker import DockerClient
from docker.errors import DockerException


def get_root() -> Path:
    """Returns the absolute path to the root directory of the project."""

    root = Path(__file__).parent.parent.parent.resolve()

    assert root.is_dir(), f"Root directory {root} does not exist or is not a directory."
    assert root.name == "swelancer", "Expected root directory to be named 'swelancer'."
    assert root.parent.name == "project", "Expected root directory to be in 'project' directory."

    return root


PATH_TO_SWE_LANCER_TASKS = get_root() / Path("all_swelancer_tasks.csv")


def is_linux_machine() -> bool:
    """Returns `True` iff running on a Linux machine."""

    return platform.system() == "Linux"


def get_tasks(split: str | None = None, task_type: str | None = None) -> list[str]:
    """Get all available tasks, optionally filtered by split and task type."""

    tasks = pd.read_csv(PATH_TO_SWE_LANCER_TASKS)

    if split is not None:
        tasks = tasks[tasks["set"] == split]

    if task_type is not None:
        tasks = tasks[tasks["variant"] == task_type]

    return tasks["question_id"].tolist()


def is_docker_running(timeout: float = 10.0) -> bool:
    """Return `True` if and only if Docker is running."""

    try:
        return DockerClient(timeout=timeout).ping()
    except DockerException:
        return False


def is_docker_image(image_name: str) -> bool:
    """Return `True` if and only if the specified Docker image exists locally."""

    try:
        client = DockerClient()
        client.images.get(image_name)
    except DockerException:
        return False

    return True


def get_runs_dir() -> Path:
    """Returns the absolute path to the runs directory."""

    return get_root() / "runs"
