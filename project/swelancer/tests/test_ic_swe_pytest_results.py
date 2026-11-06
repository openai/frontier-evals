"""Prevent missing pytest statuses from being recorded as model failures."""

import pytest
from swelancer.eval import _has_valid_pytest_result


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ([], False),
        ([-1.0], False),
        ([-1.0, -1.0], False),
        ([-1.0, 0.0], True),
        ([0.0], True),
        ([1.0], True),
        ([-1.0, 1.0], True),
    ],
)
def test_valid_pytest_run_detection(results: list[float], expected: bool) -> None:
    assert _has_valid_pytest_result(results) is expected
