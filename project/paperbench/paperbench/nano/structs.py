from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Self

from dotenv import load_dotenv

import chz
from alcatraz.clusters.interface import AlcatrazException

load_dotenv()
import structlog.stdlib
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerRuntime
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)
from preparedness_turn_completer.turn_completer import TurnCompleter
from pydantic import BaseModel, ConfigDict, model_validator

from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerConfiguration,
    ComputerRuntime,
    NetworkMode,
    RuntimeConfig,
)
from nanoeval.solvers.computer_tasks.task import Grade
from paperbench.grade import JudgeOutput
from paperbench.monitor.monitor import MonitorResult

GRADER_OPENAI_API_KEY = os.getenv("GRADER_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

logger = structlog.stdlib.get_logger(component=__name__)


@dataclass(frozen=True)
class ReproductionMetadata:
    is_valid_git_repo: bool
    git_log: str
    repro_script_exists: bool
    files_before_reproduce: str
    files_after_reproduce: str
    timedout: bool
    repro_log: str
    retried_results: list[ReproScriptRunOutcome] = field(default_factory=list)
    repro_execution_time: float | None = None
    git_status_after_reproduce: str | None = None
    executed_submission: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReproductionMetadata:
        try:
            retried_results = [
                ReproScriptRunOutcome.from_dict(r) for r in data.get("retried_results", [])
            ]

            return cls(
                is_valid_git_repo=data["is_valid_git_repo"],
                git_log=data["git_log"],
                repro_script_exists=data["repro_script_exists"],
                files_before_reproduce=data["files_before_reproduce"],
                files_after_reproduce=data["files_after_reproduce"],
                timedout=data["timedout"],
                repro_log=data["repro_log"],
                retried_results=retried_results,
                repro_execution_time=data.get("repro_execution_time"),
                git_status_after_reproduce=data.get("git_status_after_reproduce"),
                executed_submission=data.get("executed_submission"),
            )
        except KeyError as e:
            raise ValueError("Missing required field in reproduction metadata") from e

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReproScriptRunOutcome:
    repro_execution_time: float
    timedout: bool
    repro_log: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReproScriptRunOutcome:
        try:
            return cls(
                repro_execution_time=data["repro_execution_time"],
                timedout=data["timedout"],
                repro_log=data["repro_log"],
            )
        except KeyError as e:
            raise ValueError("Missing required field in reproduction script run outcome") from e


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
    reproduction_metadata: ReproductionMetadata | None = None
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
            "reproduction_metadata": None,
            "monitor_result": None,
            "monitor_ran": self.monitor_ran,
        }

        if self.agent_output:
            data["agent_output"] = self.agent_output.model_dump()

        if self.judge_output:
            data["judge_output"] = self.judge_output.to_dict()

        if self.reproduction_metadata:
            data["reproduction_metadata"] = self.reproduction_metadata.to_dict()

        if self.monitor_result:
            data["monitor_result"] = self.monitor_result.to_dict()

        return data


class AgentOutput(BaseModel):
    """Summary of an agent rollout for a single run."""

    run_id: str
    time_start: float
    time_end: float
    error_msg: str | None = None
    runtime_in_seconds: float
    status_exists: bool


@chz.chz
class PBRuntimeConfig(RuntimeConfig):
    """
    Light wrapper around RuntimeConfig that defines PaperBench-specific attributes
    RuntimeConfigs used by PaperBench PBSolvers should inherit from this class.
    """

    exception_types: tuple[type[Exception]] | type[Exception] | None = chz.field(
        default=None, doc="Exception types that trigger a retry on task setup or grading"
    )


@chz.chz
class AlcatrazPBRuntimeConfig(PBRuntimeConfig):
    """
    Implementation of PBRuntimeConfig for Alcatraz.
    """

    exception_types: tuple[type[Exception]] | type[Exception] | None = chz.field(
        default=AlcatrazException,
        doc="Exception types that trigger a retry on task setup or grading",
    )


class ReproductionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    timeout: int = 100 * 3600
    # if the reproduce.sh runs for less than this, it will be retried with salvaging fixes
    retry_threshold: float = 600
    overwrite_existing_output: bool = False
    skip_reproduction: bool = False
    computer_runtime: ComputerRuntime = AlcatrazComputerRuntime()
    computer_config: ComputerConfiguration = ComputerConfiguration(
        docker_image="pb-reproducer:latest", network_mode=NetworkMode.UNPROXIED
    )
    runtime_config: PBRuntimeConfig = AlcatrazPBRuntimeConfig()

    @model_validator(mode="after")
    def _validate_timeout_and_retry_threshold(self) -> Self:
        if self.retry_threshold >= self.timeout:
            logger.warning(
                "ReproductionConfig.retry_threshold >= ReproductionConfig.timeout, so reproduce.sh salvaging is disabled.",
            )
        return self


class JudgeConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grade: bool = True
    grade_locally: bool = True
    grade_id: int = 0
    overwrite_existing_output: bool = False
    scaffold: str = "simple"
    completer_config: TurnCompleter.Config = OpenAICompletionsTurnCompleter.Config(
        model="o3-mini-2025-01-31",
        reasoning_effort="high",
    )
    code_only: bool = False
    resources_provided: bool = False
    computer_runtime: ComputerRuntime = AlcatrazComputerRuntime()
    computer_config: ComputerConfiguration = ComputerConfiguration(
        docker_image="pb-env:latest",
        environment={"OPENAI_API_KEY": GRADER_OPENAI_API_KEY},
        network_mode=NetworkMode.UNPROXIED,
    )
    runtime_config: PBRuntimeConfig = AlcatrazPBRuntimeConfig()


class PaperBenchGrade(Grade):
    paperbench_result: PaperBenchResult
    is_continuous: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "paperbench_result": self.paperbench_result.to_dict(),
            "score": self.score,
            "grader_log": self.grader_log,
        }


@dataclass
class AgentDirConfig:
    agent_dir: str
    directories_to_save: list[str]