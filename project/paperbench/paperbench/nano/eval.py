import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncGenerator, Literal, Optional, Sequence

import chz
import dotenv
import numpy as np
import structlog
from alcatraz.clusters.local import ClusterConfig, LocalConfig
from nanoeval.eval import RetryableSystemError
from nanoeval.recorder import get_recorder
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from nanoeval.solvers.computer_tasks.solver import PythonCodingEval, PythonCodingSolver
from nanoeval.solvers.computer_tasks.steps import FinalResult, FinalResultSuccessful, Step
from nanoeval.solvers.computer_tasks.task import ComputerTask, Grade
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerInterface
from nanoeval_alcatraz.task_to_alcatraz_config import task_to_alcatraz_config
from paperbench.agents.registry import registry as agent_registry
from paperbench.agents.run import (
    AgentOutput,
    prepare_computer,
    run_agent_in_computer,
    upload_heavy_logs,
    upload_status,
)
from paperbench.agents.utils import AgentDirConfig, prepare_agent_dir_config
from paperbench.constants import AGENT_DIR, CODE_DIR, LOGS_DIR, SUBMISSION_DIR, WORKSPACE_BASE
from paperbench.metrics import compute_agg_stats, per_paper_results
from paperbench.nano.utils import (
    SPLIT_TO_EXPECTED_PAPERS,
    file_processor,
    filter_processor,
    gather_eval_runs,
    get_file_at_duration,
)
from paperbench.paper_registry import paper_registry
from paperbench.scripts.alcatraz_services import grade_on_computer, reproduce_on_computer
from paperbench.scripts.run_judge import JudgeOutput
from paperbench.scripts.run_reproduce import ReproductionMetadata
from paperbench.utils import (
    create_run_dir,
    create_run_id,
    get_default_runs_dir,
    get_experiments_dir,
    get_timestamp,
    purple,
)
from pydantic import BaseModel
from typing_extensions import override

dotenv.load_dotenv()


GRADER_OPENAI_API_KEY = os.getenv("GRADER_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

structlog.configure(
    processors=[
        # Standard processors
        structlog.processors.TimeStamper(fmt="%Y-%m-%dT%H:%M:%S.%fZ"),
        structlog.processors.add_log_level,
        # Custom processors
        file_processor,
        filter_processor,
        # Rendering
        structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.stdlib.get_logger().bind(component=__name__)


async def check_submission_exists(computer: ComputerInterface, logger: logging.Logger):
    """
    Checks if there is at least one file in the submission directory in the cluster.

    Args:
        cluster: The Alcatraz cluster instance.
        save_dir: The directory where the output folder/file will be saved.
    """
    res = await computer.send_shell_command(f"ls -A {SUBMISSION_DIR} | wc -l")
    num_files = int(res.output.decode("utf-8").strip())
    if res.exit_code != 0 or num_files <= 1:  # we expect the initial .git file
        logger.error(f"No files found in submission directory\n{num_files}")
        return False
    return True


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
    timeout: int = 4 * 3600
    grade: bool = True
    grade_id: int = 0
    overwrite_existing_output: bool = False
    scaffold: str = "simple"
    model: str = "o3-mini-2025-01-31"
    code_only: bool = False
    reasoning_effort: Optional[str] = "high"
    cluster_config: LocalConfig = LocalConfig(
        image="pb-grader:latest",
        pull_from_registry=False,
        environment={"OPENAI_API_KEY": GRADER_OPENAI_API_KEY},
    )


class ReproductionOutput(BaseModel):
    executed_submission: Path | str | None = None
    metadata: ReproductionMetadata | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "ReproductionOutput":
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
            raise ValueError(f"Missing required field in reproduction output: {e}")

    def to_dict(self) -> dict:
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
    agent_output: Optional[AgentOutput] = None
    judge_output: Optional[JudgeOutput] = None
    reproduction_output: Optional[ReproductionOutput] = None

    def to_dict(self) -> dict:
        data = {
            "paper_id": self.paper_id,
            "skipped_reproduction": self.skipped_reproduction,
            "code_only": self.code_only,
            "agent_output": None,
            "judge_output": None,
            "reproduction_output": None,
        }

        if self.agent_output:
            data["agent_output"] = self.agent_output.to_dict()

        if self.judge_output:
            data["judge_output"] = self.judge_output.to_dict()

        if self.reproduction_output:
            data["reproduction_output"] = self.reproduction_output.to_dict()

        return data


class PaperBenchGrade(Grade):
    paperbench_result: PaperBenchResult
    is_continuous: bool = True

    def to_dict(self) -> dict:
        return {
            "paperbench_result": self.paperbench_result.to_dict(),
            "score": self.score,
            "grader_log": self.grader_log,
        }


class PBTask(ComputerTask):
    paper_id: str
    run_id: str  # Unique identifier for task
    run_group_id: str  # The group of runs this task belongs to
    run_dir: str  # Directory where logs will be saved for this task
    local_runs_dir: str
    target_duration_hr: Optional[int]
    reproduction: ReproductionConfig
    judge: JudgeConfig

    @asynccontextmanager
    async def _start_computer(
        self,
        cluster_config: LocalConfig,
    ) -> AsyncGenerator[ComputerInterface, None]:
        # This is a total hack to get around the fact that `ComputerTask` only supports a single
        # Docker image. Should we run the agent and judge in separate `ComputerTask`s? Should we
        # refactor `ComputerTask` to support multiple computer configurations? Or something
        # else?
        task = self.model_copy(
            update={
                "docker_image": cluster_config.image,
                "environment": {
                    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", None),
                    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", None),
                    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", None),
                    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", None),
                    "SUBMISSION_DIR": SUBMISSION_DIR,
                    "LOGS_DIR": LOGS_DIR,
                    "CODE_DIR": CODE_DIR,
                    "AGENT_DIR": AGENT_DIR,
                },
            }
        )

        alcatraz_env = task_to_alcatraz_config(task, cluster_config)

        async with alcatraz_env.build() as cluster:
            yield AlcatrazComputerInterface(cluster_value=cluster)

    @override
    async def _setup(self, computer: ComputerInterface) -> None:
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id, run_id=self.run_id, runs_dir=Path(self.local_runs_dir)
        )

        paper = paper_registry.get_paper(self.paper_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            instructions = Path(tmp_dir) / "instructions.txt"
            instructions.write_text(self.prompt[0]["content"])

            await computer.check_shell_command(f"mkdir -p {WORKSPACE_BASE}/paper/assets")

            for src, dst in [
                (instructions, f"{WORKSPACE_BASE}/instructions.txt"),
                (paper.paper_pdf, f"{WORKSPACE_BASE}/paper/paper.pdf"),
                (paper.paper_md, f"{WORKSPACE_BASE}/paper/paper.md"),
                (paper.addendum, f"{WORKSPACE_BASE}/paper/addendum.md"),
                (paper.blacklist, f"{WORKSPACE_BASE}/paper/blacklist.txt"),
            ]:
                with open(src, "rb") as f:
                    await computer.upload(f.read(), dst)

            agent_env_path = agent_registry.get_agents_dir() / "agent.env"
            if agent_env_path.exists():
                with open(agent_env_path, "rb") as f:
                    await computer.upload(f.read(), f"{WORKSPACE_BASE}/agent.env")
            else:
                ctx_logger.warning(
                    f"agent.env not found at: {agent_env_path} agent.env should be created by copying agent.env.example"
                    f"to agent.env and populating the necessary keys. agent.env is necessary for replicating some papers when using a non-dummy agent",
                    destinations=["console", "run"],
                )

        for asset in paper.assets.glob("*"):
            with open(asset, "rb") as f:
                await computer.upload(f.read(), f"{WORKSPACE_BASE}/paper/assets/{asset.name}")

        for dir in [SUBMISSION_DIR, CODE_DIR]:
            await computer.check_shell_command(f"mkdir -p {dir}")

    @override
    async def grade(
        self, computer: ComputerInterface, agent_output: AgentOutput
    ) -> PaperBenchGrade:
        """
        Runs the grading process for a single submission.
        """

        ctx_logger = logger.bind(
            run_group_id=self.run_group_id,
            run_id=self.run_id,
            runs_dir=Path(self.local_runs_dir),
        )

        # We need one final upload before grading, for solvers which do not take care of this step.
        if not agent_output.skipped_rollout:
            ctx_logger.info("Attempting to upload logs before grading...", destinations=["run"])

            try:
                await upload_heavy_logs(
                    computer=computer,
                    agent_start_time=int(time.time()),
                    agent_dir_config=prepare_agent_dir_config(),
                    run_dir=Path(self.run_dir),
                    logger=ctx_logger.bind(destinations=["run"]),
                )

                await upload_status(
                    start_time=int(time.time()),
                    run_dir=Path(self.run_dir),
                    status="done",
                    logger=ctx_logger.bind(destinations=["run"]),
                )
            except Exception as e:
                ctx_logger.error(
                    f"Exception uploading final logs before grading: {e}", destinations=["run"]
                )

        checkpoint = await self._select_checkpoint()
        if not checkpoint:
            ctx_logger.error("No checkpoint exists, skipping grading!", destinations=["run"])

            return PaperBenchGrade(
                paperbench_result=PaperBenchResult(
                    paper_id=self.paper_id,
                    run_id=self.run_id,
                    submission_exists=False,
                    skipped_reproduction=self.reproduction.skip_reproduction,
                    code_only=self.judge.code_only,
                    agent_output=agent_output,
                    judge_output=None,
                    reproduction_output=None,
                ),
                score=0.0,
                grader_log="No checkpoint exists, skipping grading!",
            )

        ctx_logger.info("Successfully uploaded logs before grading!", destinations=["run"])

        prelim_submission_path, _ = checkpoint
        reproduction_output = None

        if self.reproduction.skip_reproduction or self.judge.code_only:
            submission_to_grade_path = prelim_submission_path
        else:
            submission_to_grade_path = prelim_submission_path.replace(".tar.gz", "_repro.tar.gz")

            ctx_logger.info(
                f"Starting the reproduction process for `{self.question_id}.{self.attempt_id}`...",
                destinations=["console", "group", "run"],
            )

            async with self._start_computer(
                self.reproduction.cluster_config
            ) as reproduction_computer:
                reproduction_output = await self._reproduce(reproduction_computer)

            ctx_logger.info(
                f"The reproduction process for {self.question_id}.{self.attempt_id} has finished!",
                destinations=["console", "group", "run"],
            )

            get_recorder().record_extra(
                {
                    "run_group_id": self.run_group_id,
                    "run_id": self.run_id,
                    "repro_metadata": (
                        reproduction_output.metadata.to_dict()
                        if reproduction_output.metadata
                        else {}
                    ),
                }
            )

        # Grade reproduction
        judge_output = None

        if (
            reproduction_output
            and reproduction_output.success
            or self.reproduction.skip_reproduction
            or self.judge.code_only
        ):
            ctx_logger.info(
                f"Grading the submission for {self.question_id}.{self.attempt_id}...",
                destinations=["console", "group", "run"],
            )

            ctx_logger.info(
                f"Grading submission {submission_to_grade_path}...", destinations=["run"]
            )

            judge_output = await self.grade_submission(
                submission_to_grade_path,
                self.paper_id,
                Path(self.run_dir),
            )

            ctx_logger.info(
                f"Grading for {self.question_id}.{self.attempt_id} finished!",
                destinations=["console", "group", "run"],
            )

            get_recorder().record_extra(
                {
                    "run_group_id": self.run_group_id,
                    "run_id": self.run_id,
                    "judge_output": judge_output.to_dict() if judge_output else None,
                }
            )

        grade = PaperBenchGrade(
            paperbench_result=PaperBenchResult(
                paper_id=self.paper_id,
                run_id=self.run_id,
                submission_exists=True,
                skipped_reproduction=self.reproduction.skip_reproduction,
                code_only=self.judge.code_only,
                agent_output=agent_output,
                judge_output=judge_output,
                reproduction_output=reproduction_output,
            ),
            score=judge_output.score if judge_output else 0.0,
            grader_log=(
                json.dumps(judge_output.to_dict()) if judge_output else "Log file not found!"
            ),
        )

        get_recorder().record_extra(
            {
                "run_group_id": self.run_group_id,
                "run_id": self.run_id,
                "pb_result": grade.to_dict(),
            }
        )

        with open(Path(self.run_dir) / "pb_result.json", "w") as f:
            json.dump(grade.to_dict(), f, indent=2)

        ctx_logger.info(
            purple(f"Grades saved to {Path(self.run_dir) / 'pb_result.json'}"),
            destinations=["console", "group", "run"],
        )

        return grade

    async def _reproduce(self, computer: ComputerInterface) -> ReproductionOutput:
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id,
            run_id=self.run_id,
            runs_dir=Path(self.local_runs_dir),
        )

        checkpoint = await self._select_checkpoint()

        if not checkpoint:
            ctx_logger.error("No checkpoint exists, skipping reproduction!", destinations=["run"])

            return ReproductionOutput(
                executed_submission=None,
                metadata=None,
            )

        submission, _ = checkpoint
        reproduce_output_path = submission.replace(".tar.gz", "_repro.tar.gz")
        repro_metadata_path = submission.replace(".tar.gz", "_repro_metadata.json")
        tmp_out_path = Path("/tmp") / repro_metadata_path

        ctx_logger.info(f"Reproducing submission {reproduce_output_path}...", destinations=["run"])

        # If the reproduction output already exists, we can skip reproduction
        if not self.reproduction.overwrite_existing_output:
            repro_output_exists = os.path.exists(reproduce_output_path)
            repro_metadata_exists = os.path.exists(repro_metadata_path)
            if repro_output_exists and repro_metadata_exists:
                ctx_logger.info(
                    f"Reproduction output already exists, skipping reproduction: {reproduce_output_path}, {repro_metadata_path}",
                    destinations=["run"],
                )
                with open(tmp_out_path, "r") as f:
                    data = json.load(f)

                metadata = ReproductionMetadata.from_dict(data)

                return ReproductionOutput(
                    executed_submission=reproduce_output_path,
                    metadata=metadata,
                )
        # Reproduce on alcatraz
        await reproduce_on_computer(
            computer=computer,
            submission_path=submission,
            logger=ctx_logger.bind(destinations=["run"]),
            run_dir=Path(self.run_dir),
            timeout=self.reproduction.timeout,
            retry_threshold=self.reproduction.retry_threshold,
        )

        # Now the result should exist
        repro_output_exists = os.path.exists(reproduce_output_path)
        repro_metadata_exists = os.path.exists(repro_metadata_path)
        if not repro_output_exists:
            ctx_logger.error(
                f"Reproduction failed to produce output: {reproduce_output_path}",
                destinations=["console", "group", "run"],
            )
            return ReproductionOutput(
                executed_submission=reproduce_output_path,
                metadata=None,
            )
        if not repro_metadata_exists:
            ctx_logger.error(
                f"Reproduction failed to produce metadata: {repro_metadata_path}",
                destinations=["console", "group", "run"],
            )
            return ReproductionOutput(
                executed_submission=reproduce_output_path,
                metadata=None,
            )

        with open(repro_metadata_path, "r") as f:
            data = json.load(f)

        metadata = ReproductionMetadata.from_dict(data)

        return ReproductionOutput(
            executed_submission=reproduce_output_path,
            metadata=metadata,
        )

    async def _select_checkpoint(self) -> Optional[tuple[str, timedelta]]:
        """Identify the submission tarball to use for reproduction/grading."""
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id,
            run_id=self.run_id,
            runs_dir=Path(self.local_runs_dir),
        )

        # First, identify the submission that we want to grade
        # (each run_id can have multiple timestamped submissions, we need to select one)
        # Runs are organized as {run_group_id}/{run_id}/{log_timestamp}.tar.gz
        submission_checkpoints = [
            str(p)
            for p in Path(self.run_dir).glob("*.tar.gz")
            if not p.name.endswith("_repro.tar.gz")
        ]

        if not submission_checkpoints:
            return None
        # Get the submission at the target duration if it is set, otherwise get the latest submission
        target_duration_hr = self.target_duration_hr if self.target_duration_hr else 10000
        submission_path, submission_duration = get_file_at_duration(
            submission_checkpoints,
            target_duration_hr,
            ctx_logger.bind(destinations=["run"]),
        )

        return submission_path, submission_duration

    async def grade_submission(
        self,
        submission_path: str,
        paper_id: str,
        run_dir: Path,
    ) -> Optional[JudgeOutput]:
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id,
            run_id=self.run_id,
            runs_dir=Path(self.local_runs_dir),
        )

        grader_upload_path = submission_path.replace(
            ".tar.gz", f"_grader_output_{self.judge.grade_id}.json"
        )
        tmp_result_path = Path("/tmp") / grader_upload_path

        # If grader output already exists, we can skip grading
        if not self.judge.overwrite_existing_output:
            grader_output_exists = os.path.exists(grader_upload_path)
            if grader_output_exists:
                with open(tmp_result_path, "r") as f:
                    grader_output = json.load(f)

                ctx_logger.info(
                    f"Skipping grading for {self.question_id}.{self.attempt_id} because an existing grader output was found.",
                    destinations=["console", "group", "run"],
                )

                ctx_logger.info(
                    f"Found existing grader output {grader_upload_path}, skipping grading",
                    destinations=["run"],
                )

                return JudgeOutput.from_dict(grader_output)

        if not self.judge.grade:
            ctx_logger.info(
                f"Skipping grading for {self.question_id}.{self.attempt_id} because grade is set to `False`",
                destinations=["console", "group", "run"],
            )
            return None

        async with self._start_computer(self.judge.cluster_config) as computer:
            judge_output = await grade_on_computer(
                computer=computer,
                submission_path=submission_path,
                grader_upload_path=grader_upload_path,
                paper_id=paper_id,
                judge_type=self.judge.scaffold,
                model_name=self.judge.model,
                logger=ctx_logger.bind(destinations=["run"]),
                run_dir=run_dir,
                timeout=self.judge.timeout,
                code_only=self.judge.code_only,
                reasoning_effort=self.judge.reasoning_effort,
            )

        return judge_output


@chz.chz
class ExternalPythonCodingSolver(PythonCodingSolver):
    name: str = "PaperBenchSolver"
    agent_id: str  # dummy, or select ID from project/paperbench/paperbench/agents/aisi-basic-agent/config.yaml
    is_nvidia_gpu_env: bool = chz.field(
        default=False, doc="Whether to make the local NVIDIA GPU available to the agent"
    )

    upload_interval_messages: int = chz.field(
        default=None,
        doc="Upload interval in agent steps for heavy logs",
    )
    upload_interval_seconds: int = chz.field(
        default=1800,
        doc="Upload interval in time for heavy logs",
    )

    # This just sets an upper limit on agent runtime; in practice we rely on agent-side timeouts for actual rollout termination
    # (e.g. `MAX_TIME_IN_HOURS` in `project/paperbench/paperbench/agents/aisi-basic-agent/config.yaml`)
    timeout: int = chz.field(
        default=100 * 3600,
        doc="Upper limit on agent runtime. In practice, we rely on agent-side timeouts for actual rollout termination.",
    )
    agent_dir_config: AgentDirConfig = chz.field(default_factory=prepare_agent_dir_config)
    privileged: bool = chz.field(default=True)
    mount_docker_socket: bool = chz.field(default=True)
    cluster_config: ClusterConfig = chz.field(
        default_factory=lambda: LocalConfig(
            image="aisi-basic-agent:latest",
            pull_from_registry=False,
        )
    )

    iterative: bool = chz.field(default=False)

    def shortname(self) -> str:
        return self.agent_id

    @asynccontextmanager
    async def _start_computer(self, task: PBTask) -> AsyncGenerator[ComputerInterface, None]:
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            run_id=task.run_id,
            runs_dir=Path(task.local_runs_dir),
        )

        if task.reproduction.timeout < self.timeout:
            ctx_logger.warning(
                f"Reproduction timeout ({task.reproduction.timeout}) should be at least as large as agent timeout ({self.solver.timeout}), is this a mistake?",
                destinations=["console", "group", "run"],
            )

        if isinstance(self.cluster_config, LocalConfig):
            assert isinstance(
                task.reproduction.cluster_config, LocalConfig
            ), "Reproduction cluster config must be a LocalConfig if the agent's cluster config is a LocalConfig"

        ctx_logger.info(
            f"cluster_config: {json.dumps(self.cluster_config, indent=4, sort_keys=True, default=str)}",
            destinations=["run"],
        )
        ctx_logger.info(
            f"Attempting to start a cluster instance. This may take a while...",
            destinations=["run"],
        )

        agent = agent_registry.get_agent(self.agent_id)
        alcatraz_config = task_to_alcatraz_config(task, self.cluster_config)

        # TODO: Move this to the `get_instances` method in `PaperBench`.
        alcatraz_config = prepare_computer(
            alcatraz_config=alcatraz_config,
            agent=agent,
            is_nvidia_gpu_env=self.is_nvidia_gpu_env,
        )

        async with alcatraz_config.build() as cluster:
            yield AlcatrazComputerInterface(cluster_value=cluster)

    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        assert isinstance(task, PBTask)

        try:
            async with self._start_computer(task) as computer:
                # 1. Run the task setup
                await task.setup(computer)

                # 2. Run the agent
                agent_output = await self._run_agent(computer, task)

                # 3. Grade the submission
                grade = await task.grade(computer, agent_output)
                if grade.paperbench_result.judge_output is None:
                    grade = PaperBenchGrade(
                        paperbench_result=PaperBenchResult(
                            paper_id=task.paper_id,
                            run_id=task.run_id,
                            submission_exists=grade.paperbench_result.submission_exists,
                            skipped_reproduction=task.reproduction.skip_reproduction,
                            code_only=task.judge.code_only,
                            agent_output=agent_output,
                            judge_output=None,
                            reproduction_output=None,
                        ),
                        score=0.0,
                        grader_log="",
                    )

            yield FinalResultSuccessful(grade=grade)
        except Exception as e:
            raise RetryableSystemError(f"Run failed with error: {str(e)}") from e

    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            run_id=task.run_id,
            runs_dir=Path(task.local_runs_dir),
        )

        ctx_logger.info(
            f"Agent `{self.agent_id}` is attempting to replicate the `{task.paper_id}` paper...",
            destinations=["console", "group", "run"],
        )

        ctx_logger.info(
            purple(
                f"Writing logs for run to {Path(task.local_runs_dir) / task.run_group_id / task.run_id / 'run.log'}"
            ),
            destinations=["console", "group"],
        )

        if self.agent_id == "human":
            get_recorder().record_extra(
                {
                    "run_group_id": task.run_group_id,
                    "run_id": task.run_id,
                    "rollout_metadata": {},
                }
            )

            get_recorder().record_match(correct=True)

            return AgentOutput(
                run_id=task.run_id,
                time_start=time.time(),
                time_end=time.time(),
                runtime_in_seconds=0,
                error_msg=None,
                status_exists=False,  # Humans don't produce status.json files :)
            )

        ctx_logger.info(
            f"Starting evaluation for task {task.question_id}.{task.attempt_id}",
            destinations=["run"],
        )

        get_recorder().record_sampling(
            prompt="",
            sampled=f"Rolling out task {task.question_id}.{task.attempt_id}",
        )

        # If agent logs already exist, we can skip running the agent
        tmp_out_dir = Path("/tmp") / f"{task.run_group_id}/{task.run_id}"
        tmp_out_dir.mkdir(parents=True, exist_ok=True)
        tmp_out_path = tmp_out_dir / "status.json"
        status_path = Path(task.run_dir) / "status.json"
        at_least_one_tar = len([file for file in Path(task.run_dir).glob("*.tar.gz")]) >= 1
        status_exists = os.path.exists(status_path)
        # we expect at least one tar if the run was successful: one gets uploaded at the start,
        # and one more at the end (you would have more at every half-hour checkpoint in between)
        if status_exists and at_least_one_tar:
            with open(status_path, "r") as f:
                status = json.load(f)

            ctx_logger.info(
                f"Agent logs already exist, skipping rollouts for {task.run_id}",
                destinations=["run"],
            )

            ctx_logger.info(f"status: {status}", destinations=["run"])
            start_time = status.get("created_at") if status.get("created_at") else time.time()
            if status.get("agent_finished_at"):
                end_time = status.get("agent_finished_at")
            elif status.get("last_updated"):
                end_time = status.get("last_updated")
            else:
                end_time = time.time()
            runtime = end_time - start_time
            agent_output = AgentOutput(
                run_id=task.run_id,
                time_start=start_time,
                time_end=end_time,
                runtime_in_seconds=runtime,
                error_msg=None,  # No error if we have status.json and tars
                status_exists=True,  # We already checked status_exists
                skipped_rollout=True,
            )

            get_recorder().record_extra(
                {
                    "run_group_id": task.run_group_id,
                    "run_id": task.run_id,
                    "agent_output": agent_output.to_dict(),
                    "status": status,
                }
            )

            return agent_output

        # Run the agent
        agent = agent_registry.get_agent(self.agent_id)
        paper = paper_registry.get_paper(task.paper_id)
        start = time.time()

        try:
            agent_output = await run_agent_in_computer(
                computer=computer,
                task=task,
                paper=paper,
                agent=agent,
                run_dir=Path(task.run_dir),
                logger=ctx_logger.bind(destinations=["run"]),
                agent_dir_config=self.agent_dir_config,
                timeout=self.timeout,
                upload_interval_messages=self.upload_interval_messages,
                upload_interval_seconds=self.upload_interval_seconds,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            end = time.time()
            ctx_logger.error(
                f"Agent timed out for task {task.question_id}.{task.attempt_id}",
                destinations=["console", "group", "run"],
            )
            status_path = f"{task.run_group_id}/{task.run_id}/status.json"
            status_exists = os.path.exists(status_path)
            get_recorder().record_match(correct=False)

            return AgentOutput(
                run_id=task.run_id,
                time_start=start,
                time_end=end,
                runtime_in_seconds=end - start,
                error_msg=str(e),
                status_exists=status_exists,
            )

        ctx_logger.info(
            f"Agent `{self.agent_id}` finished running for `{task.question_id}.{task.attempt_id}`!",
            destinations=["console", "group", "run"],
        )

        with open(Path(task.run_dir) / "metadata.json", "w") as f:
            json.dump(agent_output.to_dict(), f, indent=4)

        # Now the result should exist
        num_tars = len([file for file in Path(task.run_dir).glob("*.tar.gz")])
        ctx_logger.info(f"Found {num_tars} tars for {task.run_id}", destinations=["run"])

        if num_tars < 1:
            ctx_logger.error(
                f"Rollout failed to produce at least one tar for {task.run_id}",
                destinations=["console", "group", "run"],
            )
            get_recorder().record_extra(
                {
                    "run_group_id": task.run_group_id,
                    "run_id": task.run_id,
                    "agent_output": agent_output.to_dict(),
                }
            )

            return agent_output

        if not agent_output.status_exists:
            ctx_logger.error(
                f"Rollout failed to produce status.json for {task.run_id}",
                destinations=["console", "group", "run"],
            )
            get_recorder().record_extra(
                {
                    "run_group_id": task.run_group_id,
                    "run_id": task.run_id,
                    "agent_output": agent_output.to_dict(),
                }
            )

            return agent_output

        with open(status_path, "r") as f:
            status = json.load(f)

        get_recorder().record_extra(
            {
                "run_group_id": task.run_group_id,
                "run_id": task.run_id,
                "agent_output": agent_output.to_dict(),
                "status": status,
            }
        )

        return agent_output


@chz.chz
class PaperBench(PythonCodingEval):
    reproduction: ReproductionConfig = chz.field(default_factory=ReproductionConfig)
    judge: JudgeConfig = chz.field(default_factory=JudgeConfig)

    # task args
    paper_split: Literal["debug", "dev", "human", "all"] = chz.field(
        default="all",
        doc="Paper split to use. One of 'debug' (rice only), 'dev' (two papers), 'human' (papers used in human baseline), 'all' (full set)",
        # should match what is in experiments/splits/
    )
    resume_run_group_id: Optional[str] = chz.field(default=None)
    resume_no_extend: bool = chz.field(
        default=False,
        doc="If true, resume only existing run_ids without creating new ones.",
    )
    target_duration_hr: int | None = chz.field(
        default=None,
        doc="If provided, reproduce and grade the agent's submission at specific checkpoint.",
    )

    # other args
    local_runs_dir: Path = chz.field(default=get_default_runs_dir(), blueprint_cast=Path)
    allow_internet: bool = chz.field(default=True)

    @chz.init_property
    def run_group_id(self) -> str:
        if self.resume_run_group_id is not None:
            return self.resume_run_group_id
        else:
            return f"{get_timestamp()}_run-group_{self.solver.shortname()}"

    @override
    def get_name(self) -> str:
        return f"PaperBench-{self.solver.shortname()}"

    @chz.validate
    def _validate_args(self):
        if self.resume_run_group_id is not None:
            assert (
                self.resume_run_group_id.strip() != ""
            ), "resume_run_group_id is empty, did you set it correctly?"

    @override
    async def get_instances(self) -> list[PBTask]:
        """Tasks are papers * seeds, with the paper list from `self.paper_split` and seeds from `self.n_tries`."""

        assert GRADER_OPENAI_API_KEY, "Environment variable `GRADER_OPENAI_API_KEY` is not set."

        ctx_logger = logger.bind(run_group_id=self.run_group_id, runs_dir=self.local_runs_dir)

        ctx_logger.info(
            purple(
                f"Writing run group logs to {self.local_runs_dir / self.run_group_id / 'group.log'}"
            ),
            destinations=["console"],
        )

        tasks = []
        paper_split_path = get_experiments_dir() / "splits" / f"{self.paper_split}.txt"
        with open(paper_split_path, "r") as f:
            paper_ids = [line.strip() for line in f.read().splitlines() if line.strip()]

        existing_run_ids = []
        if self.resume_run_group_id is not None:
            existing_run_ids = await self.get_existing_run_ids(self.run_group_id)

        agents_dir = agent_registry.get_agents_dir()

        if self.judge.code_only:
            instructions = agents_dir / "instructions" / "code_only_instructions.txt"
        elif "iterative" in self.run_group_id:
            # TODO: This is a hack to load the iterative instructions. We should refactor this
            #  such that it occurs agent-side e.g., update the agent start.py to use the instructions needed.
            instructions = agents_dir / "instructions" / "instructions_iterative.txt"
        else:
            instructions = agents_dir / "instructions" / "instructions.txt"

        # populate tasks with all the run_ids
        tasks = []
        for attempt_idx in range(self.n_tries):
            for paper_id in paper_ids:
                # See if there is an existing run_id containing the paper_id we want
                run_id = next((run_id for run_id in existing_run_ids if paper_id in run_id), None)

                if run_id is not None:
                    # if we're using an existing run_id, pop it from the set
                    existing_run_ids.remove(run_id)
                    run_dir = create_run_dir(
                        self.run_group_id, run_id, self.local_runs_dir.as_posix()
                    )
                elif self.resume_no_extend:
                    continue  # Purely resuming existing runs, don't add new ones!
                else:
                    # if none found, create a new run_id
                    run_id = create_run_id(paper_id)
                    run_dir = create_run_dir(
                        self.run_group_id, run_id, self.local_runs_dir.as_posix()
                    )

                tasks.append(
                    PBTask(
                        question_id=run_id,
                        attempt_id=attempt_idx,
                        prompt=[{"role": "user", "content": instructions.read_text()}],
                        paper_id=paper_id,
                        run_id=run_id,
                        run_group_id=self.run_group_id,
                        run_dir=str(run_dir),
                        local_runs_dir=str(self.local_runs_dir),
                        target_duration_hr=self.target_duration_hr,
                        judge=self.judge,
                        reproduction=self.reproduction,
                        allow_internet=self.allow_internet,
                        cwd="/home",
                    )
                )

        ctx_logger.info(
            f"Preparing to run {len(tasks)} tasks...", destinations=["console", "group"]
        )

        return tasks

    @override
    async def get_tasks(self) -> Sequence[PBTask]:
        # we handle the n_tries in get_instances, since we're creating run_ids and run_dirs there
        # so we can simply return the result of get_instances here
        return await self.get_instances()

    @override
    async def get_full_summary(
        self, tasks_and_results: list[tuple[PBTask, FinalResult | RetryableSystemError]]
    ) -> dict[str, Any]:
        tasks = [t for t, _ in tasks_and_results]
        results = [
            r.grade.paperbench_result
            for _, r in tasks_and_results
            if isinstance(r, FinalResult) and isinstance(r.grade, PaperBenchGrade)
        ]

        # params
        params = {
            "paper_split": self.paper_split,
            "n_tries": self.n_tries,
            "n_samples": len(tasks_and_results),
            "skip_reproduction": self.reproduction.skip_reproduction,
            "code_only": self.judge.code_only,
            "agent": self.solver.shortname(),
        }

        # health
        results_clean = [r for r in results if not isinstance(r, RetryableSystemError)]
        run_health = {
            "n_rollouts_failed": len(
                [r for r in results_clean if not r.agent_output or not r.submission_exists]
            ),
            "n_reproductions_failed": len(
                [
                    r
                    for r in results_clean
                    if not r.reproduction_output or not r.reproduction_output.success
                ]
            ),
            "n_gradings_failed": len(
                [r for r in results_clean if not r.judge_output or not r.judge_output.success]
            ),
        }

        eval_runs = gather_eval_runs(results_clean, self.n_tries)
        expected_papers = SPLIT_TO_EXPECTED_PAPERS[self.paper_split]
        overall_results = compute_agg_stats(eval_runs, expected_papers=expected_papers)
        mean_score_by_paper = per_paper_results(eval_runs, self.n_tries)

        metrics = {
            "mean_score": overall_results.mean,
            "std_err": overall_results.std_err,
            "n_complete_tries": overall_results.n_runs,
            "mean_score_by_paper": mean_score_by_paper,
        }

        other_stats = {
            "repro_mean_time": np.mean(
                [
                    r.reproduction_output.metadata.repro_execution_time  # type: ignore
                    for r in results_clean
                    if r.reproduction_output and r.reproduction_output.success
                ]
            ),
            "n_is_valid_git_repo": len(
                [
                    r
                    for r in results_clean
                    if r.reproduction_output
                    and r.reproduction_output.success
                    and r.reproduction_output.metadata.is_valid_git_repo  # type: ignore
                ]
            ),
            "n_nontrivial_git_log": len(
                [
                    r
                    for r in results_clean
                    if r.reproduction_output
                    and r.reproduction_output.success
                    and len(r.reproduction_output.metadata.git_log.strip().splitlines()) > 1  # type: ignore
                ]
            ),
            "n_repro_script_exists": len(
                [
                    r
                    for r in results_clean
                    if r.reproduction_output
                    and r.reproduction_output.success
                    and r.reproduction_output.metadata.repro_script_exists  # type: ignore
                ]
            ),
        }

        final_report = {
            "params": params,
            "run_health": run_health,
            "metrics": metrics,
            "other_stats": other_stats,
            "run_group_id": tasks[0].run_group_id,
        }

        return final_report

    async def get_existing_run_ids(self, run_group_id: str) -> list[str]:
        """
        Existing run_ids will be resumed (we'll skip any steps that have already been done).
        """
        ctx_logger = logger.bind(run_group_id=run_group_id, runs_dir=self.local_runs_dir)

        run_ids = set(os.listdir(Path(self.local_runs_dir) / run_group_id))

        ctx_logger.info(
            f"Found {len(run_ids)} existing run_ids in {run_group_id}",
            destinations=["console", "group"],
        )

        return run_ids
