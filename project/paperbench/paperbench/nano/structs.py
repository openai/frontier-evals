from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from paperbench.monitor.monitor import MonitorResult

load_dotenv()
import structlog.stdlib
from alcatraz.clusters.local import LocalConfig
from nanoeval.solvers.computer_tasks.task import Grade
from preparedness_turn_completer.oai_turn_completer import OpenAITurnCompleter
from preparedness_turn_completer.turn_completer import TurnCompleter
from pydantic import BaseModel

from paperbench.agents.utils import (
    AgentOutput,
)
from paperbench.grade import JudgeOutput
from paperbench.scripts.run_reproduce import ReproductionMetadata

GRADER_OPENAI_API_KEY = os.getenv("GRADER_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

logger = structlog.stdlib.get_logger(component=__name__)


class ReproductionOutput(BaseModel):
    executed_submission: Path | str | None = None
    metadata: ReproductionMetadata | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReproductionOutput:
        metadata_exists = data.get("metadata") is not None

        if metadata_exists:
            metadata = ReproductionMetadata.from_dict(data["metadata"])
        else:
            metadata = None

        try:
            return cls(
                executed_submission=data.get("executed_submission"),
                metadata=metadata,
            )
        except KeyError as e:
            raise ValueError("Missing required field in reproduction output") from e

    def to_dict(self) -> dict[str, Any]:
        return {
            "executed_submission": self.executed_submission,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }

    @property
    def success(self) -> bool:
        return self.metadata is not None


@dataclass(frozen=False)
class PaperBenchResult:
    paper_id: str
    run_id: str
    submission_exists: bool
    skipped_reproduction: bool
    code_only: bool
    resources_provided: bool
    agent_output: AgentOutput | None = None
    judge_output: JudgeOutput | None = None
    reproduction_output: ReproductionOutput | None = None
    monitor_result: MonitorResult | None = None
    monitor_ran: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = {
            "paper_id": self.paper_id,
            "run_id": self.run_id,
            "submission_exists": self.submission_exists,
            "skipped_reproduction": self.skipped_reproduction,
            "code_only": self.code_only,
            "resources_provided": self.resources_provided,
            "agent_output": None,
            "judge_output": None,
            "reproduction_output": None,
            "monitor_result": None,
            "monitor_ran": self.monitor_ran,
        }

        if self.agent_output:
            data["agent_output"] = self.agent_output.to_dict()

        if self.judge_output:
            data["judge_output"] = self.judge_output.to_dict()

        if self.reproduction_output:
            data["reproduction_output"] = self.reproduction_output.to_dict()

        if self.monitor_result:
            data["monitor_result"] = self.monitor_result.to_dict()

        return data


class ReproductionConfig(BaseModel):
    timeout: int = 100 * 3600
    retry_threshold: float = 600
    overwrite_existing_output: bool = False
    skip_reproduction: bool = False
    cluster_config: LocalConfig = LocalConfig(
        image="pb-reproducer:latest",
        pull_from_registry=False,
    )


class JudgeConfig(BaseModel):
    grade: bool = True
    grade_locally: bool = True
    grade_id: int = 0
    overwrite_existing_output: bool = False
    scaffold: str = "simple"
    completer_config: TurnCompleter.Config = OpenAITurnCompleter.Config(
        model="o3-mini-2025-01-31",
        reasoning_effort="high",
    )
    code_only: bool = False
    resources_provided: bool = False
    cluster_config: LocalConfig = LocalConfig(
        image="pb-env:latest",
        pull_from_registry=False,
        environment={"OPENAI_API_KEY": GRADER_OPENAI_API_KEY},
    )


class PaperBenchGrade(Grade):
    paperbench_result: PaperBenchResult
    is_continuous: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "paperbench_result": self.paperbench_result.to_dict(),
            "score": self.score,
            "grader_log": self.grader_log,
        }
