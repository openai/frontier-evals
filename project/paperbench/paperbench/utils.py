import io
import logging
import os
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import blobfile as bf
import openai
import structlog.stdlib
import tenacity
import yaml
from docker import DockerClient
from docker.errors import DockerException

logger = structlog.stdlib.get_logger(component=__name__)


def in_ci() -> bool:
    """Checks if the tests are running in CI."""

    return os.environ.get("CI") == "true"


def purple(text: str) -> str:
    return f"\033[1;35m{text}\033[0m"


def is_docker_running(timeout: float = 10.0) -> bool:
    """Return `True` if and only if Docker is running."""

    try:
        return DockerClient(timeout=timeout).ping()
    except DockerException:
        return False


def load_yaml(fpath: Path) -> dict:
    """Loads a YAML file and returns its contents as a dictionary."""

    assert isinstance(fpath, Path), f"Expected a `Path`, but got `{type(fpath)}`."
    assert fpath.exists(), f"File `{fpath}` does not exist."
    assert fpath.is_file(), f"Expected a file, but got `{fpath}`."
    assert fpath.suffix == ".yaml", f"Expected a YAML file, but got `{fpath}`."

    with open(fpath, "r") as file:
        contents = yaml.safe_load(file)

    return contents


def get_root() -> Path:
    """Returns an absolute path to the root of the PaperBench module."""

    path = Path(__file__).parent.resolve()

    assert (
        path.name == "paperbench"
    ), f"Expected the module directory to be `paperbench`, but got `{path.name}`."

    return path


def get_paperbench_data_dir() -> Path:
    """Returns an absolute path to the paperbench data directory."""

    return get_root().parent / "data"


def get_experiments_dir() -> Path:
    """Returns an absolute path to the paperbench data directory."""

    return get_root().parent / "experiments"


def get_timestamp() -> str:
    """Returns the current timestamp in the format `YYYY-MM-DDTHH-MM-SS-Z`."""

    return time.strftime("%Y-%m-%dT%H-%M-%S-%Z", time.gmtime())


def create_run_id(
    paper_id: str,
) -> str:
    """Creates a run ID."""

    return f"{paper_id}_{str(uuid.uuid4())}"


def create_run_dir(
    run_group: str,
    run_id: str,
    runs_dir: str,
) -> str:
    """Creates a directory for the run."""

    run_dir = bf.join(runs_dir, run_group, run_id)
    bf.makedirs(run_dir)
    return run_dir


def get_default_runs_dir() -> str:
    """Returns an absolute path to the directory storing runs."""

    return str(get_root().parent / "runs")


def path_to_tar(source_path: Path, arcname: str) -> io.BytesIO:
    """Tars a file or directory and returns the tar stream."""
    tar_stream = io.BytesIO()
    with tarfile.open(fileobj=tar_stream, mode="w") as tar:
        tar.add(source_path, arcname=arcname)
    tar_stream.seek(0)
    return tar_stream


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=1, max=300),  # Max wait time of 5 minutes
    stop=tenacity.stop_after_delay(3600 * 2),  # Retry for up to 2 hours
    retry=tenacity.retry_if_exception_type(OPENAI_TIMEOUT_EXCEPTIONS),
    before_sleep=(
        tenacity.before_sleep_log(logger._logger, logging.WARNING) if logger._logger else None
    ),
    reraise=True,
)
async def oai_completion_with_retry_async(method: Callable, *args, **kwargs) -> Any:
    return await method(*args, **kwargs)


@tenacity.retry(
    wait=tenacity.wait_random_exponential(min=1, max=300),  # Max wait time of 5 minutes
    stop=tenacity.stop_after_delay(3600 * 2),  # Retry for up to 2 hours
    retry=tenacity.retry_if_exception_type(OPENAI_TIMEOUT_EXCEPTIONS),
    before_sleep=(
        tenacity.before_sleep_log(logger._logger, logging.WARNING) if logger._logger else None
    ),
    reraise=True,
)
def oai_completion_with_retry(method: Callable, *args, **kwargs) -> Any:
    return method(*args, **kwargs)
