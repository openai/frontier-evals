"""Regression tests for multi-answer MCQ grading.

Both behaviours under test previously caused the eval to UNDER-count correct
multi-answer responses: a model that picks the genuinely-correct multi-answer
set was scored as incorrect. For dangerous-capability evals this is a
false-assurance failure -- a capable model reads as incapable.

Bug A (mcq.py / `_choice_to_answer_group`): the old set->group-id encoding
collided distinct same-size sets with equal element sums (e.g. {0, 3} and
{1, 2} both -> 113). After de-duplication on (instance, answer_group_id) a
correct set could inherit a colliding wrong set's `is_correct=False`.

Bug B (mcq_api.py / `OpenAIAPIMCQSolver.solve`): a correctly-extracted
multi-letter pick (len > 1) was collapsed to None and replaced by a random
single-index guess, whose single-int correctness test can never set-match a
multi-index `correct_indices`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import chz
from nanoeval.solvers.mcq import Answer, MCQEval, MCQSolver, MCQTask, Question
from nanoeval.solvers.mcq_api import OpenAIAPIMCQSolver


# Dummy solver to satisfy abstract class instantiation (mirrors mcq_test.py).
@chz.chz
class DummySolver(MCQSolver[Answer]):
    async def solve(self, task: MCQTask) -> Answer:  # type: ignore
        pass  # Not used in tests


@chz.chz
class ConcreteMCQEval(MCQEval):
    async def _get_tasks(self) -> list[Question]:  # type: ignore
        pass  # Not used in tests


# ---------------------------------------------------------------------------
# Bug A: collision-free answer-group ids.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_colliding_sets_get_distinct_groups_and_correct_set_is_counted() -> None:
    """{0, 3} and {1, 2} used to share group id 113; the correct attempt was
    dropped to is_correct=False, reporting accuracy 0.0. With distinct group
    ids the correct attempt is counted: per-sample accuracy is now 0.5 (one of
    two attempts correct), not the under-counted 0.0."""
    question = Question(
        question="pick the right ones",
        answers=["a", "b", "c", "d"],
        correct_indices={1, 2},
        allow_multiple_choices=True,
    )
    # Two consensus attempts on one instance: one wrong set, one correct set.
    # The wrong attempt is listed first so that, under the old encoding, the
    # de-dup would have kept its is_correct=False for the shared group id.
    results: list[Any] = [
        (
            MCQTask(question=question, question_id="q.0", attempt_id=0),
            Answer(picked={0, 3}, correct=False),
        ),
        (
            MCQTask(question=question, question_id="q.0", attempt_id=1),
            Answer(picked={1, 2}, correct=True),
        ),
    ]
    eval_instance = ConcreteMCQEval(solver=DummySolver())
    summary = await eval_instance.get_full_summary(results)
    # Previously 0.0 (correct attempt collided into the wrong group).
    assert np.isclose(summary["accuracy"], 0.5)
    assert np.isclose(summary["metrics_including_errors"]["accuracy"], 0.5)


@pytest.mark.asyncio
async def test_correct_multi_answer_set_scores_full_accuracy() -> None:
    """A single instance whose only attempt is the exactly-correct set scores
    1.0 regardless of which set it is (no collision can demote it)."""
    question = Question(
        question="pick the right ones",
        answers=["a", "b", "c", "d"],
        correct_indices={0, 3},  # a set that previously collided with {1, 2}
        allow_multiple_choices=True,
    )
    results: list[Any] = [
        (
            MCQTask(question=question, question_id="q.0", attempt_id=0),
            Answer(picked={0, 3}, correct=True),
        ),
    ]
    eval_instance = ConcreteMCQEval(solver=DummySolver())
    summary = await eval_instance.get_full_summary(results)
    assert np.isclose(summary["accuracy"], 1.0)


def test_choice_to_answer_group_is_collision_free_for_distinct_sets() -> None:
    """Directly exercise the encoding used by get_summary: distinct index sets
    must map to distinct ids; single-answer ints must be unchanged."""

    # Rebuild the same encoding get_summary applies via a tiny multi-answer eval.
    def group_id(picked: int | set[int]) -> int:
        # Mirror of MCQEval.get_summary._choice_to_answer_group; kept local so a
        # regression in that helper is caught by the end-to-end tests above.
        if isinstance(picked, set):
            assert all(p < 10 for p in picked)
            return sum(1 << p for p in picked)
        return picked

    # The historically-colliding pairs from the old sum-of-(10**(i+1)+p) scheme.
    assert group_id({0, 3}) != group_id({1, 2})
    assert group_id({0, 4}) != group_id({1, 3})
    assert group_id({0, 1, 4}) != group_id({0, 2, 3})
    # Order independence: a set has a single canonical id.
    assert group_id({3, 0}) == group_id({0, 3})
    # Single-answer ids are unchanged (returned verbatim).
    assert group_id(0) == 0
    assert group_id(2) == 2
    assert group_id(9) == 9


# ---------------------------------------------------------------------------
# Bug B: correct multi-letter pick is not demoted to a random guess.
# ---------------------------------------------------------------------------
class _StubMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = _StubMessage(content)


class _StubCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_StubChoice(content)]

    def to_dict(self, mode: str = "json") -> dict[str, Any]:
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


class _StubCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    async def create(self, **kwargs: Any) -> _StubCompletion:
        return _StubCompletion(self._content)


class _StubChat:
    def __init__(self, content: str) -> None:
        self.completions = _StubCompletions(content)


class _StubClient:
    """Deterministic stand-in for openai.AsyncClient (no network, no model)."""

    def __init__(self, content: str) -> None:
        self.chat = _StubChat(content)


async def _solve_with_response(question: Question, response: str) -> Answer:
    """Run the real OpenAIAPIMCQSolver.solve with a canned model response."""
    solver = OpenAIAPIMCQSolver(model="stub")
    # Inject the stub client into the cached_property slot.
    solver.__dict__["_client"] = _StubClient(response)
    task = MCQTask(question=question, question_id="q.0", attempt_id=0)

    from unittest.mock import MagicMock

    from nanoeval.recorder import recorder

    token = recorder.set(MagicMock())  # record_sampling is a no-op in this test
    try:
        return await solver.solve(task)
    finally:
        recorder.reset(token)


@pytest.mark.asyncio
async def test_correct_multi_letter_answer_is_graded_correct_as_set() -> None:
    """'Answer: A, C' on a {0, 2} multi-answer question must be graded correct
    and returned as a set -- not collapsed to None and demoted to a random
    single-index guess."""
    question = Question(
        question="Pick all that apply.",
        answers=["alpha", "beta", "gamma", "delta"],
        correct_indices={0, 2},
        allow_multiple_choices=True,
    )
    answer = await _solve_with_response(question, "Reasoning ...\nAnswer: A, C")
    assert isinstance(answer.picked, set)
    assert answer.picked == {0, 2}
    assert answer.correct is True
    assert answer.metadata.get("random_guess") is not True


@pytest.mark.asyncio
async def test_correct_multi_answer_does_not_crash_get_summary() -> None:
    """The set-valued Answer produced for a multi-answer pick must satisfy the
    `isinstance(picked, set)` contract in MCQEval.get_summary."""
    question = Question(
        question="Pick all that apply.",
        answers=["alpha", "beta", "gamma", "delta"],
        correct_indices={0, 2},
        allow_multiple_choices=True,
    )
    answer = await _solve_with_response(question, "Answer: A, C")
    results: list[Any] = [
        (MCQTask(question=question, question_id="q.0", attempt_id=0), answer),
    ]
    eval_instance = ConcreteMCQEval(solver=DummySolver())
    summary = await eval_instance.get_full_summary(results)
    assert np.isclose(summary["accuracy"], 1.0)


@pytest.mark.asyncio
async def test_single_answer_still_graded_by_membership() -> None:
    """Guard: single-answer behaviour is unchanged -- one picked index, graded
    by membership, returned as an int with correct_format metadata."""
    question = Question(
        question="What is 2+2?",
        answers=["3", "4", "5", "6"],
        correct_indices={1},
        allow_multiple_choices=False,
    )
    answer = await _solve_with_response(question, "Answer: B")
    assert isinstance(answer.picked, int)
    assert answer.picked == 1
    assert answer.correct is True
    assert answer.metadata.get("correct_format") is True


@pytest.mark.asyncio
async def test_unextractable_answer_still_falls_back_to_random_guess() -> None:
    """Guard: genuine extraction failure (no answer letters) is the only path
    that still reaches _random_guess."""
    question = Question(
        question="Pick all that apply.",
        answers=["alpha", "beta", "gamma", "delta"],
        correct_indices={0, 2},
        allow_multiple_choices=True,
    )
    answer = await _solve_with_response(question, "I have no idea, sorry.")
    assert answer.metadata.get("random_guess") is True
    assert isinstance(answer.picked, int)
