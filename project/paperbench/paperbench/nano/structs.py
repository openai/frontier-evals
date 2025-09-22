from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Self

from dotenv import load_dotenv

from paperbench.monitor.monitor import MonitorResult

load_dotenv()
import structlog.stdlib
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)
from preparedness_turn_completer.google_completions_turn_completer import (
    GoogleCompletionsTurnCompleter,
)
from preparedness_turn_completer.turn_completer import TurnCompleter
from pydantic import BaseModel, model_validator

from alcatraz.clusters.local import LocalConfig
from nanoeval.solvers.computer_tasks.task import Grade
from paperbench.agents.utils import (
    AgentOutput,
)
from paperbench.grade import JudgeOutput
from paperbench.scripts.run_reproduce import ReproductionMetadata

GRADER_OPENAI_API_KEY = os.getenv("GRADER_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
GRADER_GOOGLE_API_KEY = os.getenv("GRADER_GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

logger = structlog.stdlib.get_logger(component=__name__)


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
            data["agent_output"] = self.agent_output.to_dict()

        if self.judge_output:
            data["judge_output"] = self.judge_output.to_dict()

        if self.reproduction_metadata:
            data["reproduction_metadata"] = self.reproduction_metadata.to_dict()

        if self.monitor_result:
            data["monitor_result"] = self.monitor_result.to_dict()

        return data


class ReproductionConfig(BaseModel):
    timeout: int = 100 * 3600
    # if the reproduce.sh runs for less than this, it will be retried with salvaging fixes
    retry_threshold: float = 600
    overwrite_existing_output: bool = False
    skip_reproduction: bool = False
    cluster_config: LocalConfig = LocalConfig(
        image="pb-reproducer:latest",
        pull_from_registry=False,
    )

    @model_validator(mode="after")
    def _validate_timeout_and_retry_threshold(self) -> Self:
        if self.retry_threshold >= self.timeout:
            logger.warning(
                "ReproductionConfig.retry_threshold >= ReproductionConfig.timeout, so reproduce.sh salvaging is disabled.",
            )
        return self


class JudgeConfig(BaseModel):
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
    cluster_config: LocalConfig = LocalConfig(
        image="pb-env:latest",
        pull_from_registry=False,
        environment={"OPENAI_API_KEY": GRADER_OPENAI_API_KEY},
    )

    # New fields for Gemini support
    use_gemini: bool = False
    gemini_api_key: str | None = None

    @model_validator(mode="after")
    def setup_gemini_if_needed(self) -> Self:
        if self.use_gemini:
            # Get API key
            if not self.gemini_api_key:
                self.gemini_api_key = GRADER_GOOGLE_API_KEY
                if not self.gemini_api_key:
                    raise ValueError("GOOGLE_API_KEY or GRADER_GOOGLE_API_KEY required when use_gemini=True")

            # Replace completer config with Gemini configuration
            self.completer_config = GoogleCompletionsTurnCompleter.Config(
                model="gemini-2.5-pro",
                api_key=self.gemini_api_key,
                temperature=0.1,
            )

            # Update Docker environment to include Google API key
            new_env = self.cluster_config.environment.copy()
            new_env["GOOGLE_API_KEY"] = self.gemini_api_key

            # Create new cluster config with updated environment
            self.cluster_config = LocalConfig(
                image=self.cluster_config.image,
                pull_from_registry=self.cluster_config.pull_from_registry,
                environment=new_env,
            )
        return self


class PaperBenchGrade(Grade):
    paperbench_result: PaperBenchResult
    is_continuous: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "paperbench_result": self.paperbench_result.to_dict(),
            "score": self.score,
            "grader_log": self.grader_log,
        }
