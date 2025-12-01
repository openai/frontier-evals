from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import blobfile as bf
import structlog.stdlib

from alcatraz.clusters.local import ClusterConfig
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.infra.alcatraz import tar_and_extract_from_computer
from paperbench.scripts.alcatraz_services import put_submission_in_computer, start_alcatraz_computer
from paperbench.utils import get_agents_env_vars

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
            retried_results = [ReproScriptRunOutcome.from_dict(r) for r in data["retried_results"]]

            return cls(
                is_valid_git_repo=data["is_valid_git_repo"],
                git_log=data["git_log"],
                repro_script_exists=data["repro_script_exists"],
                files_before_reproduce=data["files_before_reproduce"],
                files_after_reproduce=data["files_after_reproduce"],
                timedout=data["timedout"],
                repro_log=data["repro_log"],
                retried_results=retried_results,
                repro_execution_time=data["repro_execution_time"],
                git_status_after_reproduce=data["git_status_after_reproduce"],
                executed_submission=data["executed_submission"],
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


async def run_reproduce_script(
    computer: ComputerInterface,
    submission_path: Path,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
    timeout: float | None,
    use_py3_11: bool = False,  # defaults to 3.12
    make_venv: bool = False,
) -> ReproScriptRunOutcome:
    """
    Runs the reproduce.sh script in the submission directory.
    Optionally sets python3 to python3.11 and/or creates a virtual environment.
    """

    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )
    # log rough start time of reproduction to a file
    await computer.check_shell_command(f"date +%s > {submission_path}/reproduce.log.creation_time")

    cmd_str = f"bash -c 'cd {submission_path}"
    # remove existing venvs since they contain broken symlinks to python3
    cmd_str += " && rm -rf venv .venv"
    if use_py3_11:
        cmd_str += " && update-alternatives --set python3 /usr/bin/python3.11"
    if make_venv:
        cmd_str += " && python3 -m venv venv && source venv/bin/activate"
    cmd_str += " && bash reproduce.sh 2>&1 | tee reproduce.log'"

    repro_start_time = time.time()
    # run reproduce.sh with timeout
    timedout = False
    try:
        result = await asyncio.wait_for(computer.send_shell_command(cmd_str), timeout=timeout)
        ctx_logger.info(f"Reproduction script output: {result.output.decode('utf-8')}")
    except asyncio.TimeoutError:
        timedout = True
    repro_end_time = time.time()

    repro_execution_time = repro_end_time - repro_start_time

    # Read the log file directly
    result = await computer.send_shell_command(f"cat {submission_path}/reproduce.log")
    if result.exit_code != 0:
        repro_log = "reproduce.log file not found"
    else:
        repro_log = result.output.decode("utf-8")

    return ReproScriptRunOutcome(repro_execution_time, timedout, repro_log)


async def reproduce(
    computer: ComputerInterface,
    submission_path: Path,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
    timeout: float | None = None,
    use_py3_11: bool = False,
    make_venv: bool = False,
) -> ReproductionMetadata:
    """
    args:
        computer: ComputerInterface on which the reproduce.sh script will be run
        submission_path: Path to the submission directory
        timeout: (optional) Timeout for the reproduce.sh script
        use_py3_11: (optional) Whether to switch python3 to python3.11 before running
        make_venv: (optional) Whether to create and use a virtualenv before running
    """
    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )
    # get git history for interest
    cmd_str = f"bash -c 'cd {submission_path} && git --no-pager log'"
    result = await computer.send_shell_command(cmd_str)
    exit_code, git_log = result.exit_code, result.output.decode("utf-8")
    # fine if this fails (e.g. if not a git repo), we just store that error message if so
    is_valid_git_repo = True
    if exit_code != 0:
        # typical errors: "not a git repository" or "does not have any commits"
        is_valid_git_repo = False

    result = await computer.check_shell_command(f"ls -la {submission_path}")
    files_before_reproduce = result.output.decode("utf-8")

    # does reproduce.sh exist?
    reproduce_script_path = submission_path / "reproduce.sh"
    result = await computer.send_shell_command(f"ls {reproduce_script_path}")
    repro_script_exists = result.exit_code == 0
    ctx_logger.info(f"Reproduce script exists at {reproduce_script_path}: {repro_script_exists}")
    if not repro_script_exists:
        return ReproductionMetadata(
            is_valid_git_repo=is_valid_git_repo,
            git_log=git_log,
            repro_script_exists=repro_script_exists,
            repro_log="reproduce.sh not found",
            files_before_reproduce=files_before_reproduce,
            files_after_reproduce=files_before_reproduce,
            timedout=False,
        )

    # sometimes git complains about `detected dubious ownership in repository` due to mismatching file ownership
    cmd_str = f"bash -c 'git config --global --add safe.directory {submission_path}'"
    await computer.send_shell_command(cmd_str)

    repro_outcome = await run_reproduce_script(
        computer=computer,
        submission_path=submission_path,
        run_group_id=run_group_id,
        runs_dir=runs_dir,
        run_id=run_id,
        timeout=timeout,
        use_py3_11=use_py3_11,
        make_venv=make_venv,
    )

    result = await computer.check_shell_command(f"ls -la {submission_path}")
    files_after_reproduce = result.output.decode("utf-8")

    result = await computer.send_shell_command(f"cd {submission_path} && git status")
    git_status = result.output.decode("utf-8")

    return ReproductionMetadata(
        is_valid_git_repo=is_valid_git_repo,
        git_log=git_log,
        repro_script_exists=repro_script_exists,
        repro_execution_time=repro_outcome.repro_execution_time,
        repro_log=repro_outcome.repro_log,
        files_before_reproduce=files_before_reproduce,
        files_after_reproduce=files_after_reproduce,
        git_status_after_reproduce=git_status,
        timedout=repro_outcome.timedout,
        # will populate retried_results and executed_submission later
    )


async def reproduce_on_computer(
    cluster_config: ClusterConfig,  # TODO: don't assume AlcatrazComputerInterface
    submission_path: str,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
    run_dir: str,
    submission_cluster_path: Path = Path("/submission"),
    output_cluster_path: Path = Path("/output"),
    timeout: float | None = None,
    use_py3_11: bool = False,
    make_venv: bool = False,
) -> ReproductionMetadata:
    """
    Reproduce a single submission on a computer.

    This script will spin up a pb-reproducer container on the computer to do the following:
    - Download the submission
    - Run the reproduction script
    - Save and return the reproduction results
    """
    # we're ensuring the env variables defined in agent.env are set on the computer
    cluster_config = cluster_config.model_copy(
        update={"environment": {**cluster_config.environment, **get_agents_env_vars()}}
    )
    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )
    async with start_alcatraz_computer(cluster_config) as computer:
        time_start = time.time()

        await computer.check_shell_command(
            f"mkdir -p {output_cluster_path} {submission_cluster_path}"
        )

        # Step 1: Put submission to /submission
        await put_submission_in_computer(
            computer,
            submission_path,
            run_group_id,
            runs_dir,
            run_id,
        )

        # Step 2: Run reproduction
        repro_metadata = await reproduce(
            computer=computer,
            submission_path=submission_cluster_path,
            run_group_id=run_group_id,
            runs_dir=runs_dir,
            run_id=run_id,
            timeout=timeout,
            use_py3_11=use_py3_11,
            make_venv=make_venv,
        )

        # Step 3: Save metadata
        path_to_output = submission_path.replace(".tar.gz", "_executed_metadata.json")
        bf.write_bytes(path_to_output, json.dumps(repro_metadata.to_dict()).encode("utf-8"))

        # Step 4: Save the reproduced submission itself
        timestamp = Path(submission_path).parts[-2]
        upload_from = output_cluster_path / "submission_executed.tar.gz"
        upload_to = bf.join(run_dir, "submissions", timestamp, "submission_executed.tar.gz")
        await tar_and_extract_from_computer(
            computer=computer,
            dir_path_on_computer=submission_cluster_path,
            tar_path_on_computer=upload_from,
            tar_path_on_target=upload_to,
            run_group_id=run_group_id,
            runs_dir=runs_dir,
            run_id=run_id,
            max_file_size="10M",
        )

        ctx_logger.info(f"Reproduced dir has been written: {upload_to}")

        time_end = time.time()
        ctx_logger.info(f"Reproduction completed in {time_end - time_start:.2f} seconds.")

        return repro_metadata


async def reproduce_on_computer_with_salvaging(
    cluster_config: ClusterConfig,
    submission_path: str,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
    run_dir: str,
    submission_cluster_path: Path = Path("/submission"),
    output_cluster_path: Path = Path("/output"),
    timeout: float | None = None,
    retry_threshold: float = 0,
) -> ReproductionMetadata:
    """
    Reproduce a single submission on a computer,
    salvaging reproduce attempts by retrying with slightly different configurations.
    """
    valid_threshold = True if timeout is None else retry_threshold < timeout
    retries_enabled = retry_threshold > 0 and valid_threshold

    retry_options = [{"use_py3_11": False, "make_venv": False}]
    if retries_enabled:
        retry_options.extend(
            [
                {"use_py3_11": True, "make_venv": False},
                {"use_py3_11": False, "make_venv": True},
                {"use_py3_11": True, "make_venv": True},
            ]
        )

    repro_attempts: list[ReproductionMetadata] = []

    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )

    for opts in retry_options:
        ctx_logger.info(
            f"Executing reproduce.sh with py3_11={opts['use_py3_11']}"
            f" and make_venv={opts['make_venv']}"
        )
        repro_attempt = await reproduce_on_computer(
            cluster_config=cluster_config,
            submission_path=submission_path,
            run_group_id=run_group_id,
            runs_dir=runs_dir,
            run_id=run_id,
            run_dir=run_dir,
            submission_cluster_path=submission_cluster_path,
            output_cluster_path=output_cluster_path,
            timeout=timeout,
            use_py3_11=opts["use_py3_11"],
            make_venv=opts["make_venv"],
        )
        repro_attempts.append(repro_attempt)
        if _should_retry(retries_enabled, repro_attempt, retry_threshold):
            ctx_logger.info(
                f"Reproduction attempt ran for less than {retry_threshold} seconds,"
                " retrying with different configuration."
            )
            continue  # retry, with next configuration
        else:
            break  # this last attempt was it

    repro_metadata = repro_attempts[-1]
    repro_metadata = _populate_retried_results(repro_metadata, repro_attempts[:-1])

    return repro_metadata


def _should_retry(
    retries_enabled: bool, repro_attempt: ReproductionMetadata, retry_threshold: float
) -> bool:
    """helper for determining whether we should retry to run reproduce.sh"""
    execution_time = repro_attempt.repro_execution_time or 0
    return retries_enabled and execution_time < retry_threshold


def _populate_retried_results(
    repro_metadata: ReproductionMetadata, repro_attempts: list[ReproductionMetadata]
) -> ReproductionMetadata:
    """Populates a ReproductionMetadata.retried_results with info from previous attempts"""
    if len(repro_attempts) >= 1:
        retried = [
            ReproScriptRunOutcome(float(m.repro_execution_time or 0), m.timedout, m.repro_log)
            for m in repro_attempts
        ]
        repro_metadata = replace(repro_metadata, retried_results=retried)
    return repro_metadata
