"""Ensure distinct sets of MCQ answers remain distinct in the metric input."""

from typing import Any

import pandas as pd
import pytest

import chz
import nanoeval.solvers.mcq as mcq
from nanoeval.solvers.mcq import Answer, MCQEval, MCQSolver, MCQTask, Question


@chz.chz
class _StubSolver(MCQSolver[Answer]):
    async def solve(self, task: MCQTask) -> Answer:
        raise AssertionError("not called")


@chz.chz
class _StubEval(MCQEval):
    async def _get_tasks(self) -> list[Question]:
        return []


@pytest.mark.asyncio
async def test_distinct_multi_answer_sets_have_distinct_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, pd.DataFrame] = {}

    def capture(samples_df: pd.DataFrame, correctness_df: pd.DataFrame) -> dict[str, float]:
        seen["samples"] = samples_df.copy()
        seen["correctness"] = correctness_df.copy()
        return {"accuracy": 0.0}

    monkeypatch.setattr(mcq, "compute_default_metrics", capture)
    question = Question(
        question="Select all valid options",
        answers=["alpha", "beta", "gamma", "delta"],
        correct_indices={0, 3},
        allow_multiple_choices=True,
    )
    inputs: list[tuple[MCQTask, Answer]] = []
    for attempt, picked in enumerate(({0, 3}, {1, 2}, {3, 0})):
        inputs.append(
            (
                MCQTask(question=question, question_id="q.0", attempt_id=attempt),
                Answer(picked=picked, correct=picked == question.correct_indices),
            )
        )
    result: dict[str, Any] = await _StubEval(solver=_StubSolver()).get_summary(inputs)
    assert "accuracy" in result
    ids = seen["samples"]["answer_group_id"].tolist()
    assert ids[0] != ids[1]
    assert ids[0] == ids[2]
    assert seen["correctness"]["answer_group_id"].nunique() == 2
