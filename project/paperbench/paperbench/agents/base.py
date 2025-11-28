import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import blobfile as bf
import structlog
from nanoeval_alcatraz.task_to_alcatraz_config import task_to_alcatraz_config
from typing_extensions import override

import chz
from alcatraz.clusters.local import LocalConfig
from nanoeval.eval import RolloutSystemError
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    RuntimeConfig,
)
from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from nanoeval.solvers.computer_tasks.steps import FinalResult, Step
from nanoeval.solvers.computer_tasks.task import ComputerTask
from paperbench.agents.utils import MAX_CLUSTER_START_ATTEMPTS
from paperbench.nano.structs import AgentOutput, PaperBenchGrade, PaperBenchResult
from paperbench.nano.task import PBTask
from paperbench.nano.utils import prepare_computer
from paperbench.scripts.alcatraz_services import start_alcatraz_computer

logger = structlog.stdlib.get_logger(component=__name__)


@chz.chz
class BasePBSolver(PythonCodingSolver, ABC):
    """
    Helper base class for PaperBench solvers.
    Not necessary to inherit from this class to evaluate your solver on PaperBench
    But provides useful functionality, such as a standard PythonCodingSolver.run method
    that handles the task setup and grading.
    """

    cluster_config: LocalConfig = chz.field(
        default_factory=lambda: LocalConfig(
            image="pb-env:latest",
            pull_from_registry=False,
        )
    )
    mount_docker_socket: bool = chz.field(default=True)
    runtime_config: RuntimeConfig = chz.field(default_factory=RuntimeConfig)

    @abstractmethod
    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        """
        Run the agent on the computer, to be implemented by subclasses.
        """
        pass

    async def _setup_computer(self, computer: ComputerInterface, task: PBTask) -> None:
        """Solver-specific setup, run right after task has been setup. By default does nothing."""
        return

    @asynccontextmanager
    async def _start_computer(self, task: PBTask) -> AsyncGenerator[ComputerInterface, None]:
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )

        ctx_logger.info(
            f"cluster_config: {json.dumps(self.cluster_config, indent=4, sort_keys=True, default=str)}",
            destinations=["run"],
        )
        ctx_logger.info(
            "Attempting to start a cluster instance. This may take a while...",
            destinations=["run"],
        )
        alcatraz_config = task_to_alcatraz_config(task, self.cluster_config)
        alcatraz_config = prepare_computer(
            alcatraz_config=alcatraz_config, mount_docker_socket=self.mount_docker_socket
        )

        async with start_alcatraz_computer(
            cluster_config=alcatraz_config, max_attempts=MAX_CLUSTER_START_ATTEMPTS
        ) as computer:
            yield computer

    async def _sanity_check_submission(self, task: PBTask) -> None:
        """
        Sanity checks that at least one tar was produced.
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )

        pattern = bf.join(task.run_dir, "**/*.tar.gz")
        num_tars = len(list(bf.glob(pattern)))
        ctx_logger.info(f"Found {num_tars} tars for {task.run_id}", destinations=["run"])

        if num_tars == 0:
            ctx_logger.exception(
                f"Expected BasicAgent to produce a tar for {task.run_id} (should always attempt to upload a tar at the end of the run), but found 0 tars",
                destinations=["group", "run"],
                _print=True,
            )

    async def _setup(self, computer: ComputerInterface, task: PBTask) -> None:
        """Wrapper around setup functions, for readability."""
        await task.setup(computer, self.runtime_config)
        await self._setup_computer(computer, task)

    async def _run_save_and_check(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        """Wrapper around _run_agent that also saves metadata.json and runs sanity checks."""
        agent_output = await self._run_agent(computer, task)

        with bf.BlobFile(bf.join(task.run_dir, "metadata.json"), "w") as f:
            json.dump(agent_output.model_dump(), f, indent=4)
        await self._sanity_check_submission(task)

        return agent_output

    async def _grade(
        self, computer: ComputerInterface, task: PBTask, agent_output: AgentOutput
    ) -> PaperBenchGrade:
        """Wrapper around task.grade with additional handling, for readability."""
        grade: PaperBenchGrade = await task.grade(computer, self.runtime_config)
        if grade.paperbench_result.judge_output is None:
            grade = PaperBenchGrade(
                paperbench_result=PaperBenchResult(
                    paper_id=task.paper_id,
                    run_id=task.run_id,
                    submission_exists=grade.paperbench_result.submission_exists,
                    skipped_reproduction=task.reproduction.skip_reproduction,
                    code_only=task.judge.code_only,
                    resources_provided=task.judge.resources_provided,
                    judge_output=None,
                    reproduction_metadata=None,
                ),
                score=0.0,
                grader_log="",
            )
        grade.paperbench_result.agent_output = agent_output
        return grade

    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        assert isinstance(task, PBTask), f"Task must be a PBTask, got {type(task)}"
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            run_id=task.run_id,
            runs_dir=task.runs_dir,
            destinations=["run"],
        )

        try:
            async with self._start_computer(task) as computer:
                # 1. Run the task and solver setup
                await self._setup(computer, task)
                # 2. Run the agent, serialize the output to metadata.json
                agent_output = await self._run_save_and_check(computer, task)
                # 3. Grade the submission
                grade = await self._grade(computer, task, agent_output)
            yield FinalResult(grade=grade)
        except Exception as e:
            # TODO rolloutsystem errors only on actual system errors;
            # agent errors should be handled and/or reported as a FinalResult with error.
            error_msg = f"Run failed with error: {str(e)}"
            ctx_logger.exception(error_msg)
            raise RolloutSystemError(error_msg) from e
