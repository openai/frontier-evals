import math
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from nanoeval.eval import RolloutSystemError
from nanoeval.metrics.standard import (
    compute_default_metrics,
    compute_default_metrics_on_correctness_without_answer_groups,
    handle_system_errors_and_compute_metrics,
)


def test_metrics_handle_ragged_results() -> None:
    """
    Metrics calculator should weight instances equally regardless of each number of attempts.
    """
    metrics = compute_default_metrics(
        pd.DataFrame(
            [
                {
                    "instance": "a",
                    "attempt": 0,
                    "answer_group_id": 0,
                },
                {
                    "instance": "a",
                    "attempt": 1,
                    "answer_group_id": 0,
                },
                {
                    "instance": "b",
                    "attempt": 0,
                    "answer_group_id": 1,
                },
            ]
        ),
        pd.DataFrame(
            [
                {
                    "instance": "a",
                    "answer_group_id": 0,
                    "is_correct": True,
                },
                {
                    "instance": "b",
                    "answer_group_id": 1,
                    "is_correct": False,
                },
            ]
        ),
    )
    assert math.isclose(cast(float, metrics["accuracy"]), 0.5)


@pytest.mark.parametrize(
    "data, expected",
    [
        # Simple
        (
            [
                {
                    "instance": "a",
                    "attempt": 0,
                    "is_correct": True,
                },
                {
                    "instance": "b",
                    "attempt": 0,
                    "is_correct": False,
                },
            ],
            {"accuracy": 0.5},
        ),
        # 2/3
        (
            [
                {
                    "instance": "a",
                    "attempt": 0,
                    "is_correct": True,
                },
                {
                    "instance": "b",
                    "attempt": 0,
                    "is_correct": False,
                },
                {
                    "instance": "c",
                    "attempt": 0,
                    "is_correct": True,
                },
            ],
            {"accuracy": 2 / 3},
        ),
        # Ragged results
        (
            [
                {
                    "instance": "a",
                    "attempt": 0,
                    "is_correct": True,
                },
                {
                    "instance": "a",
                    "attempt": 1,
                    "is_correct": False,
                },
                {
                    "instance": "c",
                    "attempt": 0,
                    "is_correct": True,
                },
                {
                    "instance": "c",
                    "attempt": 1,
                    "is_correct": True,
                },
                {
                    "instance": "c",
                    "attempt": 2,
                    "is_correct": True,
                },
            ],
            {"accuracy": (1 / 2 + 3 / 3) / 2},
        ),
        # Empty case
        ([], {}),
    ],
)
def test_compute_metrics_on_correctness_without_answer_groups(
    data: list[dict[str, Any]], expected: dict[str, float]
) -> None:
    metrics = compute_default_metrics_on_correctness_without_answer_groups(pd.DataFrame(data))
    for key, value in expected.items():
        assert math.isclose(cast(float, metrics[key]), value)


async def _count_metrics(results: list[tuple[Any, Any]]) -> dict[str, int]:
    return {"count": len(results)}


def _process_invalid(task: Any) -> bool:
    del task
    return False


@pytest.mark.asyncio
async def test_partial_instance_system_error_marks_top_level_metrics_invalid() -> None:
    task0 = SimpleNamespace(question_id="q0", attempt_id=0)
    task1 = SimpleNamespace(question_id="q0", attempt_id=1)

    summary = await handle_system_errors_and_compute_metrics(
        _count_metrics,
        [
            (task0, True),
            (task1, RolloutSystemError("worker failed")),
        ],
        _process_invalid,
    )

    assert summary["count"] == 1
    assert summary["num_tasks"] == 1
    assert summary["is_valid"] is False
    assert summary["metrics_including_errors"]["count"] == 2
    assert summary["metrics_including_errors"]["num_tasks"] == 1


@pytest.mark.asyncio
async def test_complete_multi_attempt_instance_remains_valid() -> None:
    task0 = SimpleNamespace(question_id="q0", attempt_id=0)
    task1 = SimpleNamespace(question_id="q0", attempt_id=1)

    summary = await handle_system_errors_and_compute_metrics(
        _count_metrics,
        [(task0, True), (task1, False)],
        _process_invalid,
    )

    assert summary["count"] == 2
    assert summary["is_valid"] is True
