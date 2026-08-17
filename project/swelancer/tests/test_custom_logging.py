from pathlib import Path

from swelancer.utils.custom_logging import get_default_runs_dir
from swelancer.utils.general import get_runs_dir


def test_default_runs_dir_uses_project_runs_directory() -> None:
    default_runs_dir = Path(get_default_runs_dir())

    assert default_runs_dir == get_runs_dir()
    assert default_runs_dir.is_absolute()
