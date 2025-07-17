import asyncio
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog.stdlib
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from structlog.stdlib import BoundLogger

logger = structlog.stdlib.get_logger(component=__name__)


@dataclass(frozen=True)
class ReproductionMetadata:
    is_valid_git_repo: bool
    git_log: str
    repro_script_exists: bool
    repro_execution_time: float
    files_before_reproduce: str
    files_after_reproduce: str
    git_status_after_reproduce: str
    timedout: bool
    repro_log: str
    retried_results: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReproductionMetadata":
        try:
            return cls(
                is_valid_git_repo=data["is_valid_git_repo"],
                git_log=data["git_log"],
                repro_script_exists=data["repro_script_exists"],
                repro_execution_time=data["repro_execution_time"],
                repro_log=data["repro_log"],
                files_before_reproduce=data["files_before_reproduce"],
                files_after_reproduce=data["files_after_reproduce"],
                git_status_after_reproduce=data["git_status_after_reproduce"],
                timedout=data["timedout"],
                retried_results=data["retried_results"],
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


async def run_reproduce_script(
    computer: ComputerInterface,
    logger: BoundLogger,
    submission_path: Path,
    timeout: float | None,
    use_py3_11: bool = False,  # defaults to 3.12
    make_venv: bool = False,
) -> ReproScriptRunOutcome:
    """
    Runs the reproduce.sh script in the submission directory.
    Optionally sets python3 to python3.11 and/or creates a virtual environment.
    """

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
        logger.info(f"Reproduction script output: {result.output.decode('utf-8')}")
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
    logger: BoundLogger,
    timeout: float | None = None,
    retry_threshold: float = 0,
) -> dict[str, Any]:
    """
    args:
        computer: ComputerInterface on which the reproduce.sh script will be run
        submission_path: Path to the submission directory
        logger: Logger object to log messages
        timeout: (optional) Timeout for the reproduce.sh script
        retry_threshold: (optional) If greater than 0 and timeout, when the reproduce.sh runs for
            less than this threshold of seconds, it is retried with series of arbitrary/generic fixes
    """
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
    logger.info(f"Reproduce script exists at {reproduce_script_path}: {repro_script_exists}")
    if not repro_script_exists:
        return asdict(
            ReproductionMetadata(
                is_valid_git_repo=is_valid_git_repo,
                git_log=git_log,
                repro_script_exists=repro_script_exists,
                repro_execution_time=0,
                repro_log="reproduce.sh not found",
                files_before_reproduce=files_before_reproduce,
                files_after_reproduce=files_before_reproduce,
                git_status_after_reproduce="",
                timedout=False,
            )
        )

    # sometimes git complains about `detected dubious ownership in repository` due to mismatching file ownership
    cmd_str = f"bash -c 'git config --global --add safe.directory {submission_path}'"
    await computer.send_shell_command(cmd_str)

    repro_outcomes: list[ReproScriptRunOutcome] = []
    repro_outcome = await run_reproduce_script(computer, logger, submission_path, timeout)
    repro_outcomes.append(repro_outcome)

    valid_threshold = True if timeout is None else retry_threshold < timeout
    retries_enabled = retry_threshold > 0 and valid_threshold
    script_ran_quickly = repro_outcome.repro_execution_time <= retry_threshold

    # only ran shortly, something trivial might be broken: maybe trivial fixes help, so retry
    if retries_enabled and script_ran_quickly:
        logger.info("Reproduce.sh ran for <= 10 minutes, retrying with small fixes")
        retry_options = [
            {"use_py3_11": True, "make_venv": False},
            {"use_py3_11": False, "make_venv": True},
            {"use_py3_11": True, "make_venv": True},
        ]
        for retry_opts in retry_options:
            repro_outcome = await run_reproduce_script(
                computer, logger, submission_path, timeout, **retry_opts
            )
            repro_outcomes.append(repro_outcome)
            if repro_outcome.repro_execution_time > retry_threshold:
                logger.info("Reproduce.sh ran for more than 10 minutes, breaking out of retry loop")
                break
        if repro_outcome.repro_execution_time <= retry_threshold:
            logger.info("Reproduce.sh still ran for <= 10 minutes, giving up")
    final_outcome = repro_outcomes[-1]

    result = await computer.check_shell_command(f"ls -la {submission_path}")
    files_after_reproduce = result.output.decode("utf-8")

    result = await computer.send_shell_command(f"cd {submission_path} && git status")
    git_status = result.output.decode("utf-8")

    return asdict(
        ReproductionMetadata(
            is_valid_git_repo=is_valid_git_repo,
            git_log=git_log,
            repro_script_exists=repro_script_exists,
            repro_execution_time=final_outcome.repro_execution_time,
            repro_log=final_outcome.repro_log,
            files_before_reproduce=files_before_reproduce,
            files_after_reproduce=files_after_reproduce,
            git_status_after_reproduce=git_status,
            timedout=final_outcome.timedout,
            retried_results=[asdict(r) for r in repro_outcomes[:-1]],
        )
    )
