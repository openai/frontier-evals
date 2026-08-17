import pytest

from nanoeval.eval import RunnerArgs


def test_runner_args_reject_zero_num_processes() -> None:
    with pytest.raises(AssertionError):
        RunnerArgs(
            experimental_use_multiprocessing=True,
            num_processes=0,
        )
