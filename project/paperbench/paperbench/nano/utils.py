from datetime import datetime, timedelta
from typing import Tuple

import structlog.stdlib
from alcatraz.clusters.local import LocalConfig
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.constants import AGENT_DIR, SUBMISSION_DIR, WORKSPACE_BASE
from paperbench.metrics import EvaluationRun, PaperEvaluation
from paperbench.utils import get_experiments_dir, get_paperbench_data_dir, is_docker_running
from structlog.stdlib import BoundLogger

logger = structlog.stdlib.get_logger(component=__name__)


def get_split_to_expected_papers() -> dict[str, int]:
    """
    Reads the split files in experiments/splits and returns a dictionary
    mapping split names to the number of papers in each split.
    """
    split_to_expected_papers = {}
    splits_dir = get_experiments_dir() / "splits"

    for split_file in splits_dir.glob("*.txt"):
        split_name = split_file.stem
        with open(split_file, "r") as f:
            # Count non-empty lines in the file
            papers = [line.strip() for line in f if line.strip()]
            split_to_expected_papers[split_name] = len(papers)

    return split_to_expected_papers


SPLIT_TO_EXPECTED_PAPERS = get_split_to_expected_papers()


def gather_eval_runs(results: list["PaperBenchResult"], n_runs: int) -> list[EvaluationRun]:
    """
    Gathers succesfully graded results of nano/eval into a list of n_runs EvaluationRuns
    where a single EvaluationRun does not contain more than one evaluation of the same paper.
    """
    seed_to_eval_run = {
        seed: EvaluationRun(seed=seed, paper_evaluations={}) for seed in range(n_runs)
    }
    paper_to_cur_seed = {}

    if not results:
        return list(seed_to_eval_run.values())

    for result in results:
        if result.judge_output is None or result.judge_output.graded_task_tree is None:
            continue
        paper_id = result.paper_id
        if paper_id not in paper_to_cur_seed:
            paper_to_cur_seed[paper_id] = 0
        seed = paper_to_cur_seed[paper_id]
        paper_to_cur_seed[paper_id] += 1
        paper_eval = PaperEvaluation(
            paper_id=paper_id,
            graded_task_node=result.judge_output.graded_task_tree,
            paper_run_id=result.run_id,
        )
        seed_to_eval_run[seed].paper_evaluations[paper_id] = paper_eval

    return list(seed_to_eval_run.values())


def uses_local_config(paperbench: "PaperBench") -> bool:  # type: ignore
    """
    Check if any of paperbench.solver.cluster_config, paperbench.reproduction.cluster_config,
    or paperbench.judge.cluster_config is an instance of LocalConfig.

    Args:
        paperbench: A PaperBench PythonCodingEval instance

    Returns:
        bool: True if any of the cluster configs is a LocalConfig, False otherwise
    """

    # PythonCodingSolver may not have a cluster_config, just ExternalPythonCodingSolver does for now
    if hasattr(paperbench.solver, "cluster_config"):
        if isinstance(paperbench.solver.cluster_config, LocalConfig):
            return True

    # Check reproduction's cluster_config
    if isinstance(paperbench.reproduction.cluster_config, LocalConfig):
        return True

    # Check judge's cluster_config
    if isinstance(paperbench.judge.cluster_config, LocalConfig):
        return True

    return False


def build_agent_command(agent: "Agent") -> str:  # type: ignore
    """Builds the command to run the agent."""

    cmd = ["bash", f"{AGENT_DIR}/start.sh"]

    if agent.kwargs_type == "argparse":
        for key, value in agent.kwargs.items():
            cmd += [f"--{key}", str(value)]

    if agent.kwargs_type == "omegaconf":
        cmd += [f"{key}={value}" for key, value in agent.kwargs.items()]

    return " ".join(cmd)


def build_reproduce_command(task: "PaperBenchTask") -> str:  # type: ignore
    """Builds the command to run the reproduction."""

    cmd = [
        f"python3 {WORKSPACE_BASE}/run_reproduce.py",
        f"--submission-path {SUBMISSION_DIR}",
        "--out-path /output/reproduction_metadata.json",
    ]

    if task.reproduction.timeout:
        cmd.extend(["--timeout", str(task.reproduction.timeout)])

    return " ".join(map(str, cmd))


async def check_submission_exists(computer: ComputerInterface, logger: BoundLogger) -> bool:
    """Checks if there is at least one file in the submission directory in the cluster."""

    result = await computer.send_shell_command(f"ls -A {SUBMISSION_DIR} | wc -l")
    num_files = int(result.output.decode("utf-8").strip())
    submission_exists = result.exit_code == 0 and num_files > 0

    if not submission_exists:
        logger.error(f"No files found in submission directory!")

    return submission_exists


def get_file_at_duration(
    files: list[str], duration_hr: int, logger: BoundLogger
) -> Tuple[str, timedelta]:
    """
    Given a list of files with timestamped names, return the file closest to `duration_hr`-hours
    after the earliest file in the list.
    e.g.
    ```
    files = [
        "path/to/file/2024-12-07T10-19-52-GMT.tar.gz",
        "path/to/file/2024-12-07T10-49-55-GMT.tar.gz",
        "path/to/file/2024-12-07T11-19-56-GMT.tar.gz",
        "path/to/file/2024-12-07T11-49-56-GMT_step_10.tar.gz",
        "path/to/file/2024-12-07T12-19-58-GMT.tar.gz",
    ]
    get_file_at_duration(files, 1)
    > "path/to/file/2024-12-07T11-19-56-GMT.tar.gz",
    ```
    """
    # Extract timestamps from filenames
    timestamps = []
    for file in files:
        # Extract extract timestamp from the path
        ts_str = file.split("/")[-2].replace(".tar.gz", "")
        if "step" in ts_str:
            ts_str = ts_str.split("_step_")[0]
        # Parse timestamp string into datetime
        try:
            # Try parsing with timezone
            dt = datetime.strptime(ts_str, "%Y-%m-%dT%H-%M-%S-%Z")
        except ValueError:
            # Fallback to GMT if no timezone specified
            dt = datetime.strptime(ts_str, "%Y-%m-%dT%H-%M-%S-GMT")
        timestamps.append(dt)

    earliest = min(timestamps)
    target = earliest + timedelta(hours=duration_hr)

    # Find file with timestamp closest to target
    closest_file = min(zip(files, timestamps), key=lambda x: abs((x[1] - target).total_seconds()))
    logger.info(
        f"Closest file to {duration_hr} hours after earliest file ({earliest}) is {closest_file[0]}"
    )
    retrieved_file = closest_file[0]
    retrieved_duration = closest_file[1] - earliest
    return retrieved_file, retrieved_duration


def run_sanity_checks(paperbench: "PaperBench"):
    check_for_docker(paperbench)
    check_for_lfs()
    check_for_computer_grading(paperbench)


def check_for_docker(paperbench: "PaperBench"):
    if uses_local_config(paperbench):
        assert is_docker_running(), (
            "Docker is not running, but a local config requested."
            " Please ensure Docker is running if you wish to use `LocalConfig` for any of the `cluster_config`s."
        )


def check_for_lfs():
    # Check dataset has been pulled from git lfs
    papers_dir = get_paperbench_data_dir() / "papers"
    papers = [i for i in papers_dir.glob("**/paper.md")]

    for paper in papers:
        with open(paper, "r") as f:
            assert len(f.readlines()) > 5, f"Paper at {paper} should be pulled from git lfs"


def check_for_computer_grading(paperbench: "PaperBench"):
    if not paperbench.judge.grade_locally:
        raise NotImplementedError("Grading on computer is not fully supported yet.")
