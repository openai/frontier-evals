import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import chz
import structlog
from alcatraz.clusters.local import LocalConfig
from nanoeval.eval import RolloutSystemError
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    RuntimeConfig,
)
from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from nanoeval.solvers.computer_tasks.steps import FinalResult, Step
from nanoeval.solvers.computer_tasks.task import ComputerTask
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerInterface
from nanoeval_alcatraz.task_to_alcatraz_config import task_to_alcatraz_config
from typing_extensions import override

from paperbench.agents.run import prepare_computer
from paperbench.agents.utils import AgentOutput
from paperbench.nano.structs import PaperBenchGrade, PaperBenchResult
from paperbench.nano.task import PBTask

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
    runtime_config: RuntimeConfig

    @abstractmethod
    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        """
        Run the agent on the computer, to be implemented by subclasses.
        """
        pass

    @asynccontextmanager
    async def _start_computer(self, task: PBTask) -> AsyncGenerator[ComputerInterface, None]:
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            run_id=task.run_id,
            runs_dir=task.runs_dir,
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
            alcatraz_config=alcatraz_config,
            mount_docker_socket=self.mount_docker_socket,
        )

        async with alcatraz_config.build() as cluster:
            yield AlcatrazComputerInterface(cluster_value=cluster)

    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        assert isinstance(task, PBTask), f"Task must be a PBTask, got {type(task)}"

        try:
            async with self._start_computer(task) as computer:
                # 1. Run the task setup
                await task.setup(computer, self.runtime_config)

                # 2. Run the agent
                agent_output = await self._run_agent(computer, task)

                # 3. Grade the submission
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
            yield FinalResult(grade=grade)
        except Exception as e:
            raise RolloutSystemError(f"Run failed with error: {str(e)}") from e
