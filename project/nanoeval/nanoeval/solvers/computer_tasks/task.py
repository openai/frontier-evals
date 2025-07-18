from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import structlog
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from pydantic import BaseModel, ConfigDict, field_validator
from typing_extensions import override

from nanoeval.evaluation import Task
from nanoeval.solvers.computer_tasks._serializable_base_model import SerializableBaseModel
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerConfiguration,
    ComputerInterface,
    JupyterComputerInterface,
    RuntimeConfig,
    valid_ipython_code,
)

logger = structlog.stdlib.get_logger(component=__name__)


class Grade(BaseModel):
    score: float
    grader_log: str
    is_continuous: bool = False


class ComputerTask(ABC, ComputerConfiguration, Task, SerializableBaseModel):
    """
    A computer using agentic task. Generally it is assumed the agent has access to a
    terminal and optionally a Jupyter kernel environment.
    """

    # Prompt. For internal usage
    prompt: list[ChatCompletionMessageParam]

    # Grader may print out additional information to stdout; this will be logged.
    grade_every_step: bool = True

    # The message used to nudge Harmony agents to keep trying
    retry_message: str = """The task isn't complete yet. Keep trying!

    # You must complete **all of the above tasks** completely, without caveats and without further input from the user. I know you can do it, because I've seen you do it in prior conversations. Remember you can try different approaches!"""

    # Optional hints to provide to the model.
    hints: list[str] = []

    @abstractmethod
    async def grade(self, computer: ComputerInterface, runtime_config: RuntimeConfig) -> Grade:
        pass

    @abstractmethod
    async def _setup(self, computer: ComputerInterface, runtime_config: RuntimeConfig) -> None:
        pass

    async def check_for_valid_submission(
        self, computer: ComputerInterface, runtime_config: RuntimeConfig
    ) -> bool:
        """
        Function used to check if the model has generated any potential solution, regardless of
        whether or not it is correct. This is a simple way to poke the model to try again if it
        failed so egregiously at instruction-following that it didn't even generate a submission.

        If max_pokes > 0,
        this function must be implemented, and will be used for bumping the model to try again:
        if it returns False, the model will be shown the retry_message up to max_pokes times. If
        it returns true, the model submission will be graded and the rollout will end.
        """
        raise NotImplementedError(
            "You must implement task.check_for_valid_submission if max_pokes > 0"
        )

    @property
    def supports_pokes(self) -> bool:
        # supports pokes if method was overridden
        return type(self).check_for_valid_submission != ComputerTask.check_for_valid_submission

    async def setup(self, computer: ComputerInterface, runtime_config: RuntimeConfig) -> None:
        await super().setup(computer, runtime_config)
        # Now, we run the task-specific setup function.
        await self._setup(computer, runtime_config)

    @field_validator("prompt")
    @classmethod
    def _validate_prompt_length(
        cls, v: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        assert len(v) > 0, "Prompt must have at least one message."
        return v

    @field_validator("prompt")
    @classmethod
    def _validate_prompt_roles(
        cls, v: list[ChatCompletionMessageParam]
    ) -> list[ChatCompletionMessageParam]:
        for msg in v:
            assert msg["role"] != "system", (
                "System messages must be provided by the agent, not the task"
            )
        return v

    # No additional attributes. Relevant for deserialization.
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class SimpleJupyterTask(ComputerTask):
    """
    Simple implementation of a ComputerTask that runs single jupyter cells to
    set up and grade the task.
    """

    # Sets up the task. May include imports, function definitions, etc.
    setup_cell: str
    # May return a boolean or a float.
    grader_cell: str

    @field_validator("setup_cell", "grader_cell")
    @classmethod
    def _validate_valid_python(cls, v: str) -> str:
        assert valid_ipython_code(v), f"Code is not valid Python: {v}"

        return v

    @override
    async def _setup(self, computer: ComputerInterface, runtime_config: RuntimeConfig) -> None:
        assert isinstance(computer, JupyterComputerInterface), (
            "Computer must be a JupyterComputerInterface"
        )

        await computer.check_execute(self.setup_cell)

    @override
    async def grade(self, computer: ComputerInterface, runtime_config: RuntimeConfig) -> Grade:
        assert isinstance(computer, JupyterComputerInterface), (
            "Computer must be a JupyterComputerInterface"
        )

        res = await computer.check_execute(self.grader_cell)
        assert res.parsed_final_expression_output is not None, res

        expression = res.parsed_final_expression_output
        if isinstance(expression, bool):
            score = 1.0 if expression else 0.0
        elif isinstance(expression, float):
            score = expression
        else:
            raise ValueError(
                "Grader must return a boolean or a float, but returned: %s" % expression
            )

        return Grade(
            score=score,
            grader_log=res.output,
            # Should this be `is_continuous=True`?
        )
