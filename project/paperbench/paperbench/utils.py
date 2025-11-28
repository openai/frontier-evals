import io
import logging
import os
import subprocess
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, ParamSpec, Sequence, TypeVar

import blobfile as bf
import numpy as np
import openai
import structlog.stdlib
import tenacity
import yaml
from docker import DockerClient
from docker.errors import DockerException

logger = structlog.stdlib.get_logger(component=__name__)

P = ParamSpec("P")
R = TypeVar("R")

DATA_DIR_ENV_VAR = "PAPERBENCH_DATA_DIR"


class PaperbenchDataNotFoundError(FileNotFoundError):
    """Raised when we cannot locate the PaperBench data directory."""

    def __init__(self, candidate: Path):
        message = (
            "Unable to locate the PaperBench data directory.\n"
            f"Checked: {candidate}\n"
            "If you installed PaperBench from a git dependency, the dataset is not pulled "
            "automatically. Set the PAPERBENCH_DATA_DIR environment variable to point to a "
            "clone of `frontier-evals` where you've run `git lfs fetch --include "
            '"project/paperbench/data/**" --exclude ""` (and `git lfs checkout`), or hydrate the data in '
            "another location and point PAPERBENCH_DATA_DIR at it."
        )
        super().__init__(message)
        self.candidate = candidate


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


def load_yaml_dict(fpath: Path) -> dict[str, Any]:
    """Loads a YAML file and returns its contents as a dictionary."""

    assert isinstance(fpath, Path), f"Expected a `Path`, but got `{type(fpath)}`."
    assert fpath.exists(), f"File `{fpath}` does not exist."
    assert fpath.is_file(), f"Expected a file, but got `{fpath}`."
    assert fpath.suffix == ".yaml", f"Expected a YAML file, but got `{fpath}`."

    with open(fpath, "r") as file:
        contents = yaml.safe_load(file)

    assert isinstance(contents, dict), (
        f"Expected to load a dictionary from YAML file, but got `{type(contents)}`."
    )

    return contents


def get_root() -> Path:
    """Returns an absolute path to the root of the PaperBench module."""

    path = Path(__file__).parent.resolve()

    assert path.name == "paperbench", (
        f"Expected the module directory to be `paperbench`, but got `{path.name}`."
    )

    return path


def get_paperbench_data_dir(*, require_exists: bool = True) -> Path:
    """Returns an absolute path to the PaperBench data directory.

    Args:
        require_exists: When True (default) we raise a PaperbenchDataNotFoundError if the
            resolved directory is missing. When False we return the candidate directory and
            create it if an explicit override was provided.
    """

    override = os.environ.get(DATA_DIR_ENV_VAR)
    if override:
        override_path = Path(override).expanduser()
        if override_path.exists():
            return override_path
        if not require_exists:
            override_path.mkdir(parents=True, exist_ok=True)
            return override_path
        raise PaperbenchDataNotFoundError(override_path)

    default_path = get_root().parent / "data"
    if default_path.exists():
        return default_path
    if require_exists:
        raise PaperbenchDataNotFoundError(default_path)
    return default_path


def build_canonical_sub_path(run_dir: Path | str, timestamp: str) -> str:
    """
    The canonical place where we expect submission.tar.gz to be for repro and grading
    """
    return bf.join(run_dir, "submissions", timestamp, "submission.tar.gz")


def get_experiments_dir() -> Path:
    """Returns an absolute path to the paperbench data directory."""

    return get_root().parent / "experiments"


def find_dotenv() -> Path:
    """Returns an absolute path to the .env file."""

    return get_root().parent / ".env"


def get_timestamp() -> str:
    """Returns the current timestamp in the format `YYYY-MM-DDTHH-MM-SS-Z`."""

    return time.strftime("%Y-%m-%dT%H-%M-%S-%Z", time.gmtime())


def _get_env_var_name(value: str) -> str | None:
    """Extract `${{ secrets.NAME }}` -> `NAME`; return None if not matching."""
    if not isinstance(value, str):
        return None
    import re

    env_var_pattern = r"\$\{\{\s*secrets\.(\w+)\s*\}\}"
    match = re.match(env_var_pattern, value)
    if not match:
        return None
    return match.group(1)


def parse_env_var_values(dictionary: dict[str, str]) -> dict[str, str]:
    """Replace values matching `${{ secrets.NAME }}` with `os.environ[NAME]`.

    Mutates and returns the same dict for convenience.
    """
    for key, value in dictionary.items():
        env_var = _get_env_var_name(value)
        if not env_var:
            continue
        env_val = os.getenv(env_var)
        if env_val is None:
            raise ValueError(f"Environment variable `{env_var}` is not set!")
        dictionary[key] = env_val
    return dictionary


def safe_mean(values: Sequence[float | int], default: float = np.nan) -> float:
    """Return the mean or a default when no values are provided."""

    assert isinstance(values, Sequence), "`values` must be a sequence"
    assert all(isinstance(v, (int, float)) for v in values), "`values` must be numeric list"
    assert isinstance(default, (int, float)), "`default` must be numeric"

    if not values:
        return float(default)

    result = float(np.mean(values))

    assert isinstance(result, (int, float)), (
        f"Expected the mean to be a number, but got `{type(result)}`"
    )

    return result


def get_commit_hash() -> str:
    """Returns the current Git commit hash."""

    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("ascii")


def create_run_id(paper_id: str) -> str:
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
async def oai_completion_with_retry_async(
    method: Callable[P, Awaitable[R]], *args: P.args, **kwargs: P.kwargs
) -> R:
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
def oai_completion_with_retry(method: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    return method(*args, **kwargs)
