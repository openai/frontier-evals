from __future__ import annotations

from typing import Any

import pytest

import chz
from nanoeval.solvers.mcq import Answer, MCQEval, MCQSolver, MCQTask, Question
from nanoeval.solvers.mcq_api import OpenAIAPIMCQSolver


@chz.chz
class DummySolver(MCQSolver[Answer]):
    async def solve(self, task: MCQTask) -> Answer:  # type: ignore
        raise AssertionError("not used")


@chz.chz
class ConcreteMCQEval(MCQEval):
    async def _get_tasks(self) -> list[Question]:  # type: ignore
        raise AssertionError("not used")


def test_extract_picked_letters_uses_final_single_answer() -> None:
    sampled = "Answer: A\nAfter checking the reasoning, I need to correct this.\nAnswer: B"

    picked = OpenAIAPIMCQSolver._extract_picked_letters(
        sampled, {"A", "B", "C", "D"}, allow_multiple_choices=False
    )

    assert picked == {"B"}


def test_extract_picked_letters_uses_final_multi_answer() -> None:
    sampled = "Answer: A, C\nAfter checking the reasoning, I need to correct this.\nAnswer: B, D"

    picked = OpenAIAPIMCQSolver._extract_picked_letters(
        sampled, {"A", "B", "C", "D"}, allow_multiple_choices=True
    )

    assert picked == {"B", "D"}


def test_extract_picked_letters_handles_adjacent_multi_answer_correction() -> None:
    sampled = "Answer: A, C\nAnswer: B, D"

    picked = OpenAIAPIMCQSolver._extract_picked_letters(
        sampled, {"A", "B", "C", "D"}, allow_multiple_choices=True
    )

    assert picked == {"B", "D"}


@pytest.mark.asyncio
async def test_multi_answer_random_guess_satisfies_eval_contract() -> None:
    question = Question(
        question="Pick all that apply.",
        answers=["alpha", "beta", "gamma", "delta"],
        correct_indices={0, 2},
        allow_multiple_choices=True,
    )
    task = MCQTask(question=question, question_id="q.0", attempt_id=0)

    answer = OpenAIAPIMCQSolver._random_guess(question)

    assert isinstance(answer.picked, set)
    assert len(answer.picked) == 1
    assert answer.correct == (answer.picked == question.correct_indices)
    assert answer.metadata == {"random_guess": True}

    eval_instance = ConcreteMCQEval(solver=DummySolver())
    summary: dict[str, Any] = await eval_instance.get_full_summary([(task, answer)])
    assert "accuracy" in summary


def test_single_answer_random_guess_keeps_int_contract() -> None:
    question = Question(
        question="Pick one.",
        answers=["alpha", "beta"],
        correct_indices={0},
        allow_multiple_choices=False,
    )

    answer = OpenAIAPIMCQSolver._random_guess(question)

    assert isinstance(answer.picked, int)
    assert answer.correct == (answer.picked in question.correct_indices)
    assert answer.metadata == {"random_guess": True}
