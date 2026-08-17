from abc import ABC, abstractmethod
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal, Self

import structlog.stdlib
from nanoeval.solvers.computer_tasks.task import Grade
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface

from evmbench.audit import Audit
from evmbench.agents.agent import AgentOutput

logger = structlog.stdlib.get_logger(component=__name__)

class JudgeResult(BaseModel):
    """Represents an LLM judge's decision on a single vulnerability."""
    detected: bool = Field(..., description="Whether the vulnerability was detected.")
    reasoning: str = Field(..., description="The reasoning for the judge decision.")

class TestResult(BaseModel):
    """
    Represents the result of running one or more tests.
    Can represent either a full test suite (invariant) run or a single vulnerability test.
    """
    n_total: int
    n_failures: int
    n_errors: int
    failures: list[str]
    vulnerability_id: str | None = None
    score: int = 0 # set manually by the grader

class VulnerabilityResult(BaseModel):
    """Represents a per-vulnerability grading outcome."""
    vulnerability_id: str
    score: int | float
    max_score: int | float
    passed: bool

class GradeResult(BaseModel):
    """
    Represents the grade output for a single exploit task.
    """
    score: float = 0.0
    max_score: float = 0.0

class EVMbenchResult(BaseModel):
    audit_id: str
    score: int | float
    max_score: int |float
    agent_output: AgentOutput | None = None
    details: dict[str, Any] = {}

class EVMbenchDetectResult(EVMbenchResult):
    detect_award: float = 0.0
    detect_max_award: float = float("inf")

class EVMbenchGrade(Grade):
    """Wrapper class for interfacting with nanoeval."""
    evmbench_result: EVMbenchResult

    @model_validator(mode="after")
    def normalize_nanoeval_score(self) -> Self:
        max_score = float(self.evmbench_result.max_score)
        self.score = (
            float(self.evmbench_result.score) / max_score
            if max_score > 0
            else 0.0
        )
        self.is_continuous = True
        return self

class GraderContext(BaseModel):
    audit: Audit
    mode: Literal["detect", "patch", "exploit"]
    agent_output_path: Path | None = None
    run_group_id: str
    run_id: str
    runs_dir: str
    apply_gold_solution: bool = False
    changed_test_files: list[str] = []

class EVMbenchGrader(ABC):
    def __init__(self, computer: ComputerInterface):
        self.computer = computer

    async def grade(self, ctx: GraderContext) -> EVMbenchGrade:
        ctx_logger = logger.bind(run_group_id=ctx.run_group_id, run_id=ctx.run_id, runs_dir=ctx.runs_dir)

        ctx_logger.info(
            f"[{ctx.audit.id}] Grading...",
            destinations=["group", "run"],
            _print=True,
        )

        grade = await self._grade(ctx)

        ctx_logger.info(
            f"[{ctx.audit.id}] Graded. Score: {grade.evmbench_result.score}/{grade.evmbench_result.max_score}",
            destinations=["group", "run"],
            _print=True,
        )

        ctx_logger.info(
            f"[{ctx.audit.id}] Grade:\n",
            grade=grade.dict(),
            destinations=["run"],
        )

        return grade

    @abstractmethod
    async def _grade(self, context: GraderContext) -> EVMbenchGrade:
        pass
