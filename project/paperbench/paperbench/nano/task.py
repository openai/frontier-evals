from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import asynccontextmanager, nullcontext
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncGenerator

import blobfile as bf
from dotenv import load_dotenv

load_dotenv()
import structlog.stdlib
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerInterface
from nanoeval_alcatraz.task_to_alcatraz_config import task_to_alcatraz_config
from typing_extensions import override

from alcatraz.clusters.local import LocalConfig
from nanoeval.recorder import get_recorder
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    RuntimeConfig,
)
from nanoeval.solvers.computer_tasks.task import ComputerTask
from paperbench.agents.upload import (
    upload_heavy_logs,
    upload_status,
)
from paperbench.constants import (
    AGENT_DIR,
    AGENT_DIR_CONFIG,
    LOGS_DIR,
    SUBMISSION_DIR,
    WORKSPACE_BASE,
)
from paperbench.grade import JudgeOutput, grade_submission
from paperbench.monitor.create_monitor import create_monitor
from paperbench.monitor.monitor import MonitorResult
from paperbench.nano.structs import (
    JudgeConfig,
    PaperBenchGrade,
    PaperBenchResult,
    ReproductionConfig,
)
from paperbench.nano.utils import get_file_at_duration
from paperbench.paper_registry import paper_registry
from paperbench.scripts.run_reproduce import (
    ReproductionMetadata,
    reproduce_on_computer_with_salvaging,
)
from paperbench.utils import get_agents_dir, purple

GRADER_OPENAI_API_KEY = os.getenv("GRADER_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

logger = structlog.stdlib.get_logger(component=__name__)


class PBTask(ComputerTask):
    paper_id: str
    run_id: str  # Unique identifier for task
    run_group_id: str  # The group of runs this task belongs to
    run_dir: str  # Directory where logs will be saved for this task
    runs_dir: str
    target_duration_hr: int | None
    reproduction: ReproductionConfig
    judge: JudgeConfig
    skipped_rollout: bool = False  # whether rollouts were skipped (e.g. if we're resuming)

    save_cluster_output_to_host: bool

    @asynccontextmanager
    async def _start_computer(
        self,
        cluster_config: LocalConfig,
    ) -> AsyncGenerator[ComputerInterface, None]:
        """
        Helper function for starting an AlcatrazComputerInterface with a particular cluster_config.
        """
        # here we are ensuring task-level configuration overrides cluster configuration
        task = self.model_copy(
            update={
                # one exception: want to keep the cluster_config'd image
                "docker_image": cluster_config.image,
                # override the env variables so it's the same for any computer in PBTask
                "environment": {
                    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", None),
                    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", None),
                    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY", None),
                    "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", None),
                    "SUBMISSION_DIR": SUBMISSION_DIR,
                    "LOGS_DIR": LOGS_DIR,
                    "AGENT_DIR": AGENT_DIR,
                },
            }
        )
        alcatraz_env = task_to_alcatraz_config(task, cluster_config)

        async with alcatraz_env.build() as cluster:
            yield AlcatrazComputerInterface(cluster_value=cluster)

    @override
    async def _setup(self, computer: ComputerInterface, runtime_config: RuntimeConfig) -> None:
        """
        Called by ComputerTask.setup(), implements the setup logic specific to PBTask instances.
        Specifically, sets up the files and folders necessary for attempting the PBTask.
        """
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id,
            runs_dir=self.runs_dir,
            run_id=self.run_id,
            destinations=["run"],
        )

        paper = paper_registry.get_paper(self.paper_id)

        with tempfile.TemporaryDirectory() as tmp_dir:
            instructions = Path(tmp_dir) / "instructions.txt"
            instructions_str = self.prompt[0].get("content", None)
            if instructions_str is None:
                raise ValueError(f"Missing prompt content in run {self.run_id!r}")
            if not isinstance(instructions_str, str):
                raise TypeError(
                    f"Expected prompt content to be str, got {type(instructions_str).__name__!r}"
                )
            instructions.write_text(instructions_str)

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

            agent_env_path = get_agents_dir() / "agent.env"
            if agent_env_path.exists():
                with open(agent_env_path, "rb") as f:
                    await computer.upload(f.read(), f"{WORKSPACE_BASE}/agent.env")
            else:
                ctx_logger.warning(
                    f"agent.env not found at: {agent_env_path} agent.env should be created by copying agent.env.example"
                    f"to agent.env and populating the necessary keys. agent.env is necessary for replicating some papers when using a non-dummy agent",
                    destinations=["run"],
                    _print=True,
                )

        for asset in paper.assets.glob("*"):
            with open(asset, "rb") as f:
                await computer.upload(f.read(), f"{WORKSPACE_BASE}/paper/assets/{asset.name}")

        for dir in [SUBMISSION_DIR]:
            await computer.check_shell_command(f"mkdir -p {dir}")

    async def _ensure_files_available_for_grading(self, computer: ComputerInterface) -> None:
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id, run_id=self.run_id, runs_dir=self.runs_dir
        )
        # We need one final upload before grading, for solvers which do not take care of this step.
        if not self.skipped_rollout:
            ctx_logger.info("Attempting to upload logs before grading...", destinations=["run"])

            try:
                await upload_heavy_logs(
                    computer=computer,
                    agent_start_time=int(time.time()),
                    agent_dir_config=AGENT_DIR_CONFIG,
                    run_dir=self.run_dir,
                    run_group_id=self.run_group_id,
                    runs_dir=self.runs_dir,
                    run_id=self.run_id,
                )
                await upload_status(
                    start_time=int(time.time()),
                    run_dir=self.run_dir,
                    status="done",
                )
                ctx_logger.info("Successfully uploaded logs before grading!", destinations=["run"])
            except Exception as e:
                ctx_logger.exception(
                    f"Exception uploading final logs before grading: {e}", destinations=["run"]
                )

    def early_exit_grade(
        self,
        grader_log: str,
        monitor_ran: bool = False,
        monitor_result: MonitorResult | None = None,
    ) -> PaperBenchGrade:
        """Helper function to create a PaperBenchGrade when early exiting for some reason."""
        grade = PaperBenchGrade(
            paperbench_result=PaperBenchResult(
                paper_id=self.paper_id,
                run_id=self.run_id,
                submission_exists=False,
                skipped_reproduction=self.reproduction.skip_reproduction,
                code_only=self.judge.code_only,
                resources_provided=self.judge.resources_provided,
                agent_output=None,
                judge_output=None,
                reproduction_metadata=None,
                monitor_result=monitor_result,
                monitor_ran=monitor_ran,
            ),
            score=0.0,
            grader_log=grader_log,
        )
        self._record_extra({"pb_result": grade.to_dict()})
        self._save_grade(grade)
        return grade

    def _should_grade(self, reproduction_metadata: ReproductionMetadata | None) -> bool:
        """
        We can proceed with grading if reproduction was successful
        OR we are in a reproduction-free setup
        """
        return (reproduction_metadata is not None) or (
            self.reproduction.skip_reproduction or self.judge.code_only
        )

    def _save_grade(self, grade: PaperBenchGrade) -> None:
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id, run_id=self.run_id, runs_dir=self.runs_dir
        )

        path_to_grade = bf.join(self.run_dir, "grade.json")
        bf.write_bytes(path_to_grade, json.dumps(grade.to_dict(), indent=2).encode("utf-8"))
        ctx_logger.info(
            purple(f"Grades saved to {path_to_grade}"), destinations=["group", "run"], _print=True
        )

    def _record_extra(self, extra: dict[str, Any]) -> None:
        get_recorder().record_extra(
            {
                "run_group_id": self.run_group_id,
                "run_id": self.run_id,
                **extra,
            }
        )

    def _should_reproduce(self) -> bool:
        """Check if we should run the reproduction step"""
        return not self.reproduction.skip_reproduction and not self.judge.code_only

    @override
    async def grade(
        self, computer: ComputerInterface, runtime_config: RuntimeConfig
    ) -> PaperBenchGrade:
        """Defines the grading process for a PBTask."""
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id, run_id=self.run_id, runs_dir=self.runs_dir
        )

        # 1. setup
        await self._ensure_files_available_for_grading(computer)
        checkpoint = await self._select_checkpoint()
        if not checkpoint:
            ctx_logger.exception("No checkpoint exists, skipping grading!", destinations=["run"])
            return self.early_exit_grade("No checkpoint exists, skipping grading!")
        path_to_submission, _ = checkpoint
        path_to_executed_submission = path_to_submission.replace(".tar.gz", "_executed.tar.gz")

        # 2. run monitor
        agent_log_file_path = bf.join(self.run_dir, "agent.log")
        mon_result = None
        mon_ran = False
        if self._should_monitor(agent_log_file_path):
            mon_result = self._run_monitor(agent_log_file_path)
            mon_ran = True
            self._record_extra({"monitor_result": asdict(mon_result)})
            if mon_result.violations:
                ctx_logger.warning(
                    f"Submission for {self.run_id} flagged by monitor",
                    destinations=["group", "run"],
                )
                return self.early_exit_grade(
                    "Submission flagged by monitor", monitor_ran=mon_ran, monitor_result=mon_result
                )

        # 3. run reproduction
        repro_metadata = None
        submission_to_grade_path = path_to_submission
        if self._should_reproduce():
            repro_metadata = await self._run_reproduce(path_to_submission)
            repro_metadata_dict = repro_metadata.to_dict() if repro_metadata else {}
            submission_to_grade_path = path_to_executed_submission
            self._record_extra({"repro_metadata": repro_metadata_dict})

        # 4. run judge
        judge_output = None
        if self._should_grade(repro_metadata):
            judge_output = await self._run_judge(submission_to_grade_path, self.paper_id)
            self._record_extra({"judge_output": judge_output.to_dict() if judge_output else None})

        # 5. wrap up
        grade = PaperBenchGrade(
            paperbench_result=PaperBenchResult(
                paper_id=self.paper_id,
                run_id=self.run_id,
                submission_exists=True,
                skipped_reproduction=self.reproduction.skip_reproduction,
                code_only=self.judge.code_only,
                resources_provided=self.judge.resources_provided,
                judge_output=judge_output,
                reproduction_metadata=repro_metadata,
                monitor_ran=mon_ran,
                monitor_result=mon_result,
            ),
            score=judge_output.score if judge_output else 0.0,
            grader_log="Grading completed successfully" if judge_output else "Log file not found!",
        )
        self._record_extra({"pb_result": grade.to_dict()})
        self._save_grade(grade)

        return grade

    def _should_monitor(self, log_file_path: str) -> bool:
        """Check if we should run the monitor on the agent log file"""
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id, run_id=self.run_id, runs_dir=self.runs_dir
        )

        if bf.exists(log_file_path):
            return True
        else:
            ctx_logger.info(f"{log_file_path} not found, skipping monitor", destinations=["run"])
            return False

    def _run_monitor(self, log_file_path: str) -> MonitorResult:
        """
        Runs the monitor on an given log file
        TODO: make this configurable through chz in `PaperBenchEval`
        """
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id, run_id=self.run_id, runs_dir=self.runs_dir
        )

        ctx_logger.info(
            f"Running monitor on {self.run_id} agent.log", destinations=["run"], _print=True
        )
        paper = paper_registry.get_paper(self.paper_id)
        monitor = create_monitor(
            monitor_type="basic",
            paper=paper,
            monitor_kwargs={},
        )
        monitor_result = monitor.check_log(log_file_path)
        return monitor_result

    async def _run_reproduce(self, submission: str) -> ReproductionMetadata | None:
        """Runs the reproduction process for the submission associated with the PBTask."""
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id, run_id=self.run_id, runs_dir=self.runs_dir
        )
        ctx_logger.info(
            f"Starting the reproduction process for `{self.run_id}`...",
            destinations=["group", "run"],
            _print=True,
        )

        metadata: ReproductionMetadata | None = None
        reproduce_output_path = submission.replace(".tar.gz", "_executed.tar.gz")
        repro_metadata_path = submission.replace(".tar.gz", "_executed_metadata.json")

        # If the reproduction output already exists, we can skip reproduction
        if not self.reproduction.overwrite_existing_output:
            repro_output_exists = bf.exists(reproduce_output_path)
            repro_metadata_exists = bf.exists(repro_metadata_path)
            if repro_output_exists and repro_metadata_exists:
                ctx_logger.info(
                    f"Reproduction output already exists, skipping reproduction: {reproduce_output_path}, {repro_metadata_path}",
                    destinations=["run"],
                )
                with bf.BlobFile(repro_metadata_path, "r") as f:
                    metadata = ReproductionMetadata.from_dict(json.loads(f.read()))
                metadata = replace(metadata, executed_submission=reproduce_output_path)
                return metadata

        try:
            metadata = await reproduce_on_computer_with_salvaging(
                cluster_config=self.reproduction.cluster_config,
                submission_path=submission,
                run_group_id=self.run_group_id,
                runs_dir=self.runs_dir,
                run_id=self.run_id,
                run_dir=self.run_dir,
                timeout=self.reproduction.timeout,
                retry_threshold=self.reproduction.retry_threshold,
            )
        except Exception as e:
            logger.exception(f"Reproduction failed with error:\n{str(e)}")

        if metadata is None:
            ctx_logger.exception(
                f"Reproduction failed to produce metadata: {repro_metadata_path}",
                destinations=["group", "run"],
                _print=True,
            )

        ctx_logger.info(
            f"The reproduction process for {self.run_id} has finished!",
            destinations=["group", "run"],
            _print=True,
        )

        return metadata

    async def _select_checkpoint(self) -> tuple[str, timedelta] | None:
        """Identifies the submission tarball to use for reproduction/grading."""

        # First, identify the submission that we want to grade
        # (each run_id can have multiple timestamped submissions, we need to select one)
        # Runs are organized as {run_group_id}/{run_id}/submissions/{timestamp}/submission.tar.gz
        pattern = bf.join(self.run_dir, "**/*.tar.gz")
        submission_checkpoints = [
            i
            for i in bf.glob(pattern)
            if i.endswith(".tar.gz") and not i.endswith("_executed.tar.gz")
        ]

        if not submission_checkpoints:
            return None

        # Get the submission at the target duration if it is set, otherwise get the latest submission
        target_duration_hr = self.target_duration_hr if self.target_duration_hr else 10000
        checkpoint = get_file_at_duration(submission_checkpoints, target_duration_hr)

        path, duration = checkpoint["path"], checkpoint["duration"]

        assert isinstance(path, str)
        assert isinstance(duration, timedelta)

        return path, duration

    async def _run_judge(
        self,
        submission_path: str,
        paper_id: str,
    ) -> JudgeOutput | None:
        """Runs the judge on the given submission for the PBTask."""
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id, run_id=self.run_id, runs_dir=self.runs_dir
        )
        ctx_logger.info(
            f"Grading the submission for {self.question_id}.{self.attempt_id}"
            f" from {submission_path}...",
            destinations=["group", "run"],
            _print=True,
        )

        grader_upload_path = submission_path.replace(
            ".tar.gz", f"_grader_output_{self.judge.grade_id}.json"
        )

        # If grader output already exists, we can skip grading
        if not self.judge.overwrite_existing_output:
            grader_output_exists = bf.exists(grader_upload_path)
            if grader_output_exists:
                with bf.BlobFile(grader_upload_path, "r") as f:
                    grader_output = json.loads(f.read())
                ctx_logger.info(
                    f"Skipping grading for {self.question_id}.{self.attempt_id} because an existing grader output was found.",
                    destinations=["group", "run"],
                    _print=True,
                )

                ctx_logger.info(
                    f"Found existing grader output {grader_upload_path}, skipping grading",
                    destinations=["run"],
                )

                return JudgeOutput.from_dict(grader_output, self.judge.completer_config.__class__)

        if not self.judge.grade:
            ctx_logger.info(
                f"Skipping grading for {self.question_id}.{self.attempt_id} because grade is set to `False`",
                destinations=["group", "run"],
                _print=True,
            )
            return None

        computer_ctx = (
            nullcontext()
            if self.judge.grade_locally
            else self._start_computer(self.judge.cluster_config)
        )
        async with computer_ctx as computer:
            judge_output = await grade_submission(
                submission_path=submission_path,
                grader_upload_path=grader_upload_path,
                paper_id=paper_id,
                judge_type=self.judge.scaffold,
                completer_config=self.judge.completer_config,
                run_group_id=self.run_group_id,
                runs_dir=self.runs_dir,
                run_id=self.run_id,
                code_only=self.judge.code_only,
                computer=computer,
                resources_provided=self.judge.resources_provided,
            )

        ctx_logger.info(
            f"Grading for {self.question_id}.{self.attempt_id} finished!",
            destinations=["group", "run"],
            _print=True,
        )

        return judge_output
