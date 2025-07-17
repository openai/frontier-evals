import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Literal, Sequence

import blobfile as bf
import numpy as np
import structlog.stdlib
from dotenv import load_dotenv

load_dotenv()
import chz
from alcatraz.clusters.local import ClusterConfig, LocalConfig
from nanoeval.eval import RetryableSystemError
from nanoeval.recorder import get_recorder
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from nanoeval.solvers.computer_tasks.solver import PythonCodingEval, PythonCodingSolver
from nanoeval.solvers.computer_tasks.steps import FinalResult, FinalResultSuccessful, Step
from nanoeval.solvers.computer_tasks.task import ComputerTask
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerInterface
from nanoeval_alcatraz.task_to_alcatraz_config import task_to_alcatraz_config
from typing_extensions import override

from paperbench.agents.registry import registry as agent_registry
from paperbench.agents.run import (
    prepare_computer,
    run_agent_in_computer,
)
from paperbench.agents.utils import AgentDirConfig, AgentOutput, prepare_agent_dir_config
from paperbench.metrics import compute_agg_stats, per_paper_results
from paperbench.nano.structs import (
    JudgeConfig,
    PaperBenchGrade,
    PaperBenchResult,
    ReproductionConfig,
)
from paperbench.nano.task import PBTask
from paperbench.nano.utils import SPLIT_TO_EXPECTED_PAPERS, gather_eval_runs
from paperbench.paper_registry import paper_registry
from paperbench.utils import (
    create_run_dir,
    create_run_id,
    get_default_runs_dir,
    get_experiments_dir,
    get_paperbench_data_dir,
    get_timestamp,
    is_docker_running,
    purple,
)

GRADER_OPENAI_API_KEY = os.getenv("GRADER_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


logger = structlog.stdlib.get_logger(component=__name__)


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
            runs_dir=task.runs_dir,
        )

        if task.reproduction.timeout < self.timeout:
            ctx_logger.warning(
                f"Reproduction timeout ({task.reproduction.timeout}) should be at least as large as agent timeout ({self.timeout}), is this a mistake?",
                destinations=["group", "run"],
                _print=True,
            )

        if isinstance(self.cluster_config, LocalConfig):
            assert isinstance(task.reproduction.cluster_config, LocalConfig), (
                "Reproduction cluster config must be a LocalConfig if the agent's cluster config is a LocalConfig"
            )

        ctx_logger.info(
            f"cluster_config: {json.dumps(self.cluster_config, indent=4, sort_keys=True, default=str)}",
            destinations=["run"],
        )
        ctx_logger.info(
            "Attempting to start a cluster instance. This may take a while...",
            destinations=["run"],
        )

        if self.upload_interval_messages and self.upload_interval_messages <= 5:
            ctx_logger.warning(
                "Uploading artifacts every five messages or less is untested. "
                "Consider setting `upload_interval_messages` to a higher value.",
                destinations=["run"],
            )

        if self.upload_interval_seconds and self.upload_interval_seconds < 1800:
            ctx_logger.warning(
                "Uploading artifacts more frequently than every 1800 seconds is untested. "
                "Consider setting `upload_interval_seconds` to a higher value.",
                destinations=["run"],
            )

        agent = agent_registry.get_agent(self.agent_id)
        alcatraz_config = task_to_alcatraz_config(task, self.cluster_config)

        # TODO: Move this to the `get_instances` method in `PaperBench`.
        alcatraz_config = prepare_computer(
            alcatraz_config=alcatraz_config,
            env_vars=agent.env_vars,
            privileged=agent.privileged,
            mount_docker_socket=self.mount_docker_socket,
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
                grade: PaperBenchGrade = await task.grade(computer)
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
                            reproduction_output=None,
                        ),
                        score=0.0,
                        grader_log="",
                    )
                grade.paperbench_result.agent_output = agent_output

            yield FinalResultSuccessful(grade=grade)
        except Exception as e:
            raise RetryableSystemError(f"Run failed with error: {str(e)}") from e

    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            run_id=task.run_id,
            runs_dir=task.runs_dir,
        )

        ctx_logger.info(
            f"Agent `{self.agent_id}` is attempting to replicate the `{task.paper_id}` paper...",
            destinations=["group", "run"],
            _print=True,
        )

        ctx_logger.info(
            purple(
                f"Writing logs for run to {bf.join(task.runs_dir, task.run_group_id, task.run_id, 'run.log')}"
            ),
            destinations=["group"],
            _print=True,
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
        pattern = bf.join(task.run_dir, "**/*.tar.gz")
        num_tars = len(list(bf.glob(pattern)))

        status_path = bf.join(task.run_dir, "status.json")
        status_exists = bf.exists(status_path)

        # we expect at least one tar if the run was successful: one gets uploaded at the start,
        # and one more at the end (you would have more at every half-hour checkpoint in between)
        if status_exists and num_tars >= 1:
            with bf.BlobFile(status_path, "r") as f:
                status = json.loads(f.read())

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
            )
            task.skipped_rollout = True

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
                run_dir=task.run_dir,
                logger=ctx_logger.bind(destinations=["run"]),
                agent_dir_config=self.agent_dir_config,
                timeout=self.timeout,
                upload_interval_messages=self.upload_interval_messages,
                upload_interval_seconds=self.upload_interval_seconds,
                save_cluster_output_to_host=task.save_cluster_output_to_host,
            )
        except (asyncio.TimeoutError, asyncio.CancelledError) as e:
            end = time.time()
            ctx_logger.exception(
                f"Agent timed out for task {task.question_id}.{task.attempt_id}",
                destinations=["group", "run"],
                _print=True,
            )
            get_recorder().record_match(correct=False)

            return AgentOutput(
                run_id=task.run_id,
                time_start=start,
                time_end=end,
                runtime_in_seconds=end - start,
                error_msg=str(e),
                status_exists=bf.exists(status_path),
            )

        ctx_logger.info(
            f"Agent `{self.agent_id}` finished running for `{task.question_id}.{task.attempt_id}`!",
            destinations=["group", "run"],
            _print=True,
        )

        with bf.BlobFile(bf.join(task.run_dir, "metadata.json"), "w") as f:
            json.dump(agent_output.to_dict(), f, indent=4)

        # Now the result should exist
        pattern = bf.join(task.run_dir, "**/*.tar.gz")
        num_tars = len(list(bf.glob(pattern)))
        ctx_logger.info(f"Found {num_tars} tars for {task.run_id}", destinations=["run"])

        if num_tars < 1:
            ctx_logger.exception(
                f"Rollout failed to produce at least one tar for {task.run_id}",
                destinations=["group", "run"],
                _print=True,
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
            ctx_logger.exception(
                f"Rollout failed to produce status.json for {task.run_id}",
                destinations=["group", "run"],
                _print=True,
            )
            get_recorder().record_extra(
                {
                    "run_group_id": task.run_group_id,
                    "run_id": task.run_id,
                    "agent_output": agent_output.to_dict(),
                }
            )

            return agent_output

        with bf.BlobFile(status_path, "r") as f:
            status = json.loads(f.read())

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
    resume_run_group_id: str | None = chz.field(default=None)
    resume_no_extend: bool = chz.field(
        default=False,
        doc="If true, resume only existing run_ids without creating new ones.",
    )
    target_duration_hr: int | None = chz.field(
        default=None,
        doc="If provided, reproduce and grade the agent's submission at specific checkpoint.",
    )
    save_cluster_output_to_host: bool = chz.field(
        default=False,
        doc="If true, save cluster output to host machine at the end of a run.",
    )

    # other args
    runs_dir: str = chz.field(default=get_default_runs_dir())
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
    def _validate_args(self) -> None:
        if self.resume_run_group_id is not None:
            assert self.resume_run_group_id.strip() != "", (
                "resume_run_group_id is empty, did you set it correctly?"
            )

    @override
    async def get_instances(self) -> list[PBTask]:
        """Tasks are papers * seeds, with the paper list from `self.paper_split` and seeds from `self.n_tries`."""

        assert GRADER_OPENAI_API_KEY, "Environment variable `GRADER_OPENAI_API_KEY` is not set."

        ctx_logger = logger.bind(run_group_id=self.run_group_id, runs_dir=self.runs_dir)

        ctx_logger.info(
            purple(
                f"Writing run group logs to {bf.join(self.runs_dir, self.run_group_id, 'group.log')}"
            ),
            _print=True,
        )

        paper_split_path = get_experiments_dir() / "splits" / f"{self.paper_split}.txt"
        with open(paper_split_path, "r") as f:
            paper_ids = [line.strip() for line in f.read().splitlines() if line.strip()]

        existing_run_ids = set()
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
                run_id = next(
                    (run_id for run_id in existing_run_ids if run_id.startswith(f"{paper_id}_")),
                    None,
                )

                if run_id is not None:
                    # if we're using an existing run_id, pop it from the set
                    existing_run_ids.remove(run_id)
                    run_dir = create_run_dir(self.run_group_id, run_id, self.runs_dir)
                elif self.resume_no_extend:
                    continue  # Purely resuming existing runs, don't add new ones!
                else:
                    # if none found, create a new run_id
                    run_id = create_run_id(paper_id)
                    run_dir = create_run_dir(self.run_group_id, run_id, self.runs_dir)

                tasks.append(
                    PBTask(
                        question_id=run_id,
                        attempt_id=attempt_idx,
                        prompt=[{"role": "user", "content": instructions.read_text()}],
                        paper_id=paper_id,
                        run_id=run_id,
                        run_group_id=self.run_group_id,
                        run_dir=run_dir,
                        runs_dir=self.runs_dir,
                        target_duration_hr=self.target_duration_hr,
                        judge=self.judge,
                        reproduction=self.reproduction,
                        save_cluster_output_to_host=self.save_cluster_output_to_host,
                        allow_internet=self.allow_internet,
                        cwd="/home",
                    )
                )

        ctx_logger.info(
            f"Preparing to run {len(tasks)} tasks...", destinations=["group"], _print=True
        )

        return tasks

    @override
    async def get_tasks(self) -> Sequence[PBTask]:
        self.run_sanity_checks()
        # we handle the n_tries in get_instances, since we're creating run_ids and run_dirs there
        # so we can simply return the result of get_instances here
        return await self.get_instances()

    @override
    async def get_full_summary(
        self, results: list[tuple[ComputerTask, FinalResult | RetryableSystemError]]
    ) -> dict[str, Any]:
        tasks = [t for t, _ in results]
        final_results = [
            r.grade.paperbench_result
            for _, r in results
            if isinstance(r, FinalResult) and isinstance(r.grade, PaperBenchGrade)
        ]

        # params
        params = {
            "paper_split": self.paper_split,
            "n_tries": self.n_tries,
            "n_samples": len(results),
            "skip_reproduction": self.reproduction.skip_reproduction,
            "code_only": self.judge.code_only,
            "resources_provided": self.judge.resources_provided,
            "agent": self.solver.shortname(),
        }

        # health
        results_clean = [r for r in final_results if not isinstance(r, RetryableSystemError)]
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
            "run_group_id": getattr(tasks[0], "run_group_id", "undefined"),
        }

        return final_report

    async def get_existing_run_ids(self, run_group_id: str) -> set[str]:
        """
        Existing run_ids will be resumed (we'll skip any steps that have already been done).
        """
        ctx_logger = logger.bind(run_group_id=run_group_id, runs_dir=self.runs_dir)

        run_id_dirs = bf.listdir(bf.join(self.runs_dir, run_group_id))
        run_ids = {i for i in run_id_dirs if bf.isdir(bf.join(self.runs_dir, run_group_id, i))}

        ctx_logger.info(
            f"Found {len(run_ids)} existing unique run_ids in {run_group_id}",
            destinations=["group"],
            _print=True,
        )

        return run_ids

    def uses_local_config(self) -> bool:
        """
        Check if any of paperbench.solver.cluster_config, paperbench.reproduction.cluster_config,
        or paperbench.judge.cluster_config is an instance of LocalConfig.

        Args:
            paperbench: A PaperBench PythonCodingEval instance

        Returns:
            bool: True if any of the cluster configs is a LocalConfig, False otherwise
        """

        # PythonCodingSolver may not have a cluster_config, just ExternalPythonCodingSolver does for now
        if hasattr(self.solver, "cluster_config"):
            if isinstance(self.solver.cluster_config, LocalConfig):
                return True

        # Check reproduction's cluster_config
        if isinstance(self.reproduction.cluster_config, LocalConfig):
            return True

        # Check judge's cluster_config
        if isinstance(self.judge.cluster_config, LocalConfig):
            return True

        return False

    def run_sanity_checks(self) -> None:
        self.check_for_docker()
        self.check_for_lfs()
        self.check_for_computer_grading()

    def check_for_docker(self) -> None:
        if self.uses_local_config():
            assert is_docker_running(), (
                "Docker is not running, but a local config requested."
                " Please ensure Docker is running if you wish to use `LocalConfig` for any of the `cluster_config`s."
            )

    def check_for_lfs(self) -> None:
        # Check dataset has been pulled from git lfs
        papers_dir = get_paperbench_data_dir() / "papers"
        papers = list(papers_dir.glob("**/paper.md"))

        for paper in papers:
            with open(paper, "r") as f:
                assert len(f.readlines()) > 5, f"Paper at {paper} should be pulled from git lfs"

    def check_for_computer_grading(self) -> None:
        if not self.judge.grade_locally:
            raise NotImplementedError("Grading on computer is not fully supported yet.")
