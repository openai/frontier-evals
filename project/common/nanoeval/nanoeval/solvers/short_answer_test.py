from unittest.mock import MagicMock

import pytest

import chz
import nanoeval.solvers.short_answer as short_answer_module
from nanoeval.solvers.short_answer import (
    Answer,
    Question,
    ShortAnswerEval,
    ShortAnswerSolver,
    ShortAnswerTask,
)


@chz.chz
class FixedCorrectnessSolver(ShortAnswerSolver):
    is_correct: bool | None

    async def solve(self, task: ShortAnswerTask) -> Answer:
        return Answer(
            answer="sampled answer",
            is_correct=self.is_correct,
            metadata={"source": "test"},
        )


@chz.chz
class ConcreteShortAnswerEval(ShortAnswerEval):
    async def _get_tasks(self) -> list[Question]:
        return []


@pytest.mark.asyncio
async def test_evaluate_records_explicit_correctness(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = MagicMock()
    monkeypatch.setattr(short_answer_module, "get_recorder", lambda: recorder)

    eval_instance = ConcreteShortAnswerEval(solver=FixedCorrectnessSolver(is_correct=True))
    task = ShortAnswerTask(
        question=Question(question="test question"),
        question_id="short-answer.0",
        attempt_id=0,
    )

    result = await eval_instance.evaluate(task)

    assert result.is_correct is True
    recorder.record_match.assert_called_once_with(
        correct=True,
        metadata={"source": "test"},
    )


@pytest.mark.asyncio
async def test_evaluate_keeps_ungraded_samples_recorder_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = MagicMock()
    monkeypatch.setattr(short_answer_module, "get_recorder", lambda: recorder)

    eval_instance = ConcreteShortAnswerEval(solver=FixedCorrectnessSolver(is_correct=None))
    task = ShortAnswerTask(
        question=Question(question="test question"),
        question_id="short-answer.0",
        attempt_id=0,
    )

    result = await eval_instance.evaluate(task)

    assert result.is_correct is None
    recorder.record_match.assert_called_once_with(
        correct=False,
        metadata={"source": "test"},
    )
