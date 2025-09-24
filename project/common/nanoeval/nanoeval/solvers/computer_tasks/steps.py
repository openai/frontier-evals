from __future__ import annotations

from abc import ABC
from datetime import timedelta
from typing import Any, ClassVar, Literal

from pydantic import ConfigDict, SerializeAsAny
from pydantic.v1.json import timedelta_isoformat

from nanoeval.solvers.computer_tasks._versioning import Migration, VersionedModel
from nanoeval.solvers.computer_tasks.task import Grade


def _step_0_to_1(values: dict[str, Any]) -> dict[str, Any]:
    # Migrate correct, autograder_result to new Grade field
    if grader_log := values.pop("autograder_result"):
        values["grade"] = Grade(score=1 if values.pop("correct") else 0, grader_log=grader_log)
    else:
        assert not values.pop("correct")
        values["grade"] = None

    return values


class Step(VersionedModel):
    """
    A single step in the conversation. Contains results from the autograder.

    Extend this class to add more information about the step, e.g. the model's
    output or other metadata you'd like to include in your solver.
    """

    schema_version: int = 1
    _migrations: ClassVar[dict[int, Migration]] = {
        0: _step_0_to_1,
    }

    convo: Any

    # If none, the step wasn't graded.
    grade: SerializeAsAny[Grade] | None

    elapsed: timedelta

    type: Literal["step"] = "step"

    @property
    def correct(self) -> bool:
        return self.grade is not None and self.grade.score == 1

    model_config = ConfigDict(json_encoders={timedelta: timedelta_isoformat})


def _final_result_successful_0_to_1(values: dict[str, Any]) -> dict[str, Any]:
    # Migrate correct, autograder_result to new Grade field
    if grader_log := values.pop("grader_log"):
        values["grade"] = Grade(score=1 if values.pop("correct") else 0, grader_log=grader_log)
    else:
        assert not values.pop("correct")
        values["grade"] = None

    return values


class FinalResult(ABC, VersionedModel):
    # Allow subclasses of Grade to also dump their fields
    # https://docs.pydantic.dev/latest/concepts/serialization/#serializing-with-duck-typing
    grade: SerializeAsAny[Grade]
    convo: Any = None

    finish_status: Literal["finished-successfully"] = "finished-successfully"
    max_steps_reached: bool = False
    max_tokens_reached: bool = False
    max_time_reached: bool = False
    type: Literal["final_result_successful"] = "final_result_successful"

    schema_version: int = 1
    _migrations: ClassVar[dict[int, Migration]] = {
        0: _final_result_successful_0_to_1,
    }

    @property
    def correct(self) -> bool:
        return self.grade.score == 1
