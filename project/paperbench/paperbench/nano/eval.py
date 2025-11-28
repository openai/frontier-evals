import os
from typing import Any, Literal, Sequence

import blobfile as bf
import structlog.stdlib
from dotenv import load_dotenv

load_dotenv()
from typing_extensions import override

import chz
from alcatraz.clusters.local import LocalConfig
from nanoeval.eval import RolloutSystemError
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    NetworkMode,
)
from nanoeval.solvers.computer_tasks.solver import PythonCodingEval
from nanoeval.solvers.computer_tasks.steps import FinalResult
from nanoeval.solvers.computer_tasks.task import ComputerTask
from paperbench.metrics import compute_agg_stats, per_paper_results
from paperbench.nano.structs import (
    JudgeConfig,
    PaperBenchGrade,
    ReproductionConfig,
)
from paperbench.nano.task import PBTask
from paperbench.nano.utils import SPLIT_TO_EXPECTED_PAPERS, gather_eval_runs
from paperbench.utils import (
    create_run_dir,
    create_run_id,
    get_default_runs_dir,
    get_experiments_dir,
    get_paperbench_data_dir,
    get_root,
    get_timestamp,
    is_docker_running,
    purple,
    safe_mean,
)

GRADER_OPENAI_API_KEY = os.getenv("GRADER_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")


logger = structlog.stdlib.get_logger(component=__name__)


@chz.chz
class PaperBench(PythonCodingEval):
    reproduction: ReproductionConfig = chz.field(default_factory=ReproductionConfig)
    judge: JudgeConfig = chz.field(default_factory=JudgeConfig)

    # task args
    paper_split: Literal["debug", "dev", "human", "testing", "all"] = chz.field(
        default="all",
        doc="Paper split to use. One of 'testing' (lca-on-the-line only), 'debug' (rice only), 'dev' (two papers), 'human' (papers used in human baseline), 'all' (full set)",
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

        if self.judge.code_only:
            instructions = get_root() / "instructions" / "code_only_instructions.txt"
        else:
            instructions = get_root() / "instructions" / "instructions.txt"

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
                        network_mode=NetworkMode.UNPROXIED
                        if self.allow_internet
                        else NetworkMode.NONE,
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
        self, results: list[tuple[ComputerTask, FinalResult | RolloutSystemError]]
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
        results_clean = [r for r in final_results if not isinstance(r, RolloutSystemError)]
        run_health = {
            "n_rollouts_failed": len(
                [r for r in results_clean if not r.agent_output or not r.submission_exists]
            ),
            "n_reproductions_failed": len(
                [r for r in results_clean if not r.reproduction_metadata]
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
            "repro_mean_time": safe_mean(
                [
                    r.reproduction_metadata.repro_execution_time
                    for r in results_clean
                    if r.reproduction_metadata and r.reproduction_metadata.repro_execution_time
                ]
            ),
            "n_is_valid_git_repo": len(
                [
                    r
                    for r in results_clean
                    if r.reproduction_metadata and r.reproduction_metadata.is_valid_git_repo
                ]
            ),
            "n_nontrivial_git_log": len(
                [
                    r
                    for r in results_clean
                    if r.reproduction_metadata
                    and r.reproduction_metadata.git_log is not None
                    and len(r.reproduction_metadata.git_log.strip().splitlines()) > 1
                ]
            ),
            "n_repro_script_exists": len(
                [
                    r
                    for r in results_clean
                    if r.reproduction_metadata and r.reproduction_metadata.repro_script_exists
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
        """Return True if any cluster config uses LocalConfig."""

        # PythonCodingSolver may not have a cluster_config
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
