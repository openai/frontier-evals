import json
import os
import shutil
import tarfile
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import blobfile as bf
import pytest
import structlog.stdlib
from configs import (
    LOCAL_BASICAGENT_SOLVER_CONFIG,
    LOCAL_DUMMY_SOLVER_CONFIG,
    LOCAL_JUDGE_CONFIG,
    LOCAL_REPRODUCTION_CONFIG,
    LOCAL_REPRODUCTION_WITHOUT_SKIP_CONFIG,
)
from dotenv import load_dotenv
from pytest import fixture

from nanoeval.eval import EvalSpec
from nanoeval.evaluation import run
from nanoeval.setup import global_exit_stack
from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from paperbench.judge.graded_task_node import GradedTaskNode
from paperbench.nano.entrypoint import DefaultRunnerArgs
from paperbench.nano.eval import (
    PaperBench,
)
from paperbench.nano.logging import PaperBenchLibraryConfig, setup_logging
from paperbench.nano.structs import (
    JudgeConfig,
    ReproductionConfig,
)
from paperbench.utils import find_dotenv, get_commit_hash, in_ci, is_docker_running

load_dotenv(find_dotenv())

logger = structlog.stdlib.get_logger(component=__name__)

LOCAL_RUNS_DIR = Path(__file__).parent / "runs"
GIT_COMMIT_HASH = get_commit_hash()


@dataclass(frozen=True)
class Rollout:
    """A dataclass representing a rollout."""

    run_group_id: str
    run_group: str
    run_id: str
    run: str
    most_recent_submission: str | None


@fixture
def local_dummy_rollout() -> Iterator[Rollout]:
    """Builds a minimal dummy rollout structure."""

    with tempfile.TemporaryDirectory() as runs_dir:
        run_group_id = f"{datetime.now(timezone.utc):%Y-%m-%dT%H-%M-%S-UTC}_run-group_dummy"
        run_id = f"lca-on-the-line_{uuid.uuid4().hex}"

        run_group = bf.join(runs_dir, run_group_id)
        run = bf.join(run_group, run_id)

        submission_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-UTC")
        submission = bf.join(run, "submissions", submission_time, "submission")
        logs = bf.join(submission, "logs")

        for dirpath in [
            run_group,
            run,
            submission,
            logs,
        ]:
            bf.makedirs(dirpath)

        for fpath, content in [
            (bf.join(run_group, "group.log"), ""),
            (bf.join(run, "run.log"), ""),
            (bf.join(logs, "docker.log"), "Hello from Docker!"),
            (bf.join(logs, "run.log"), ""),
        ]:
            with bf.BlobFile(fpath, "w") as f:
                f.write(content)

        for fname in ["grade.json", "metadata.json", "status.json"]:
            with bf.BlobFile(bf.join(run, fname), "w") as f:
                json.dump({}, f)

        tarball = bf.join(run, "submissions", submission_time, "submission.tar.gz")

        with tarfile.open(tarball, "w:gz"):
            pass

        yield Rollout(
            run_group_id=run_group_id,
            run_group=run_group,
            run_id=run_id,
            run=run,
            most_recent_submission=submission,
        )


async def _generate_rollout(
    solver_config: PythonCodingSolver,
    judge_config: JudgeConfig,
    reproduction_config: ReproductionConfig,
    runs_dir: str,
    resume_run_group_id: str | None = None,
    n_tries: int = 1,
    exist_ok: bool = False,
) -> str:
    """Creates a rollout for an agent and returns the run directory. Assumes `runs_dir` is unique."""

    if not exist_ok:
        assert not bf.exists(runs_dir), f"Expected runs dir to not exist at {runs_dir}"

    bf.makedirs(runs_dir)

    async with global_exit_stack:
        setup_logging(PaperBenchLibraryConfig())

        paperbench = PaperBench(
            paper_split="testing",
            solver=solver_config,
            judge=judge_config,
            reproduction=reproduction_config,
            runs_dir=runs_dir,
            resume_run_group_id=resume_run_group_id,
            n_tries=n_tries,
        )

        eval_spec = EvalSpec(
            eval=paperbench,
            runner=DefaultRunnerArgs(max_retries=0),
        )

        await run(eval_spec)

    return runs_dir


def _get_rollout(runs_dir: str) -> Rollout:
    """A layer of indirection for a rollout to make testing easier. Assumes `runs_dir` contains a single run group and run."""

    assert bf.exists(runs_dir), f"Expected runs dir to exist at {runs_dir}"
    assert bf.isdir(runs_dir), f"Expected there to be a directory at {runs_dir}"

    run_groups = list(bf.listdir(runs_dir))

    assert len(run_groups) == 1, "Expected exactly one run group directory."

    run_group = bf.join(runs_dir, run_groups[0])
    runs = [d for d in bf.listdir(run_group) if bf.isdir(bf.join(run_group, d))]

    assert len(runs) == 1, "Expected exactly one run directory."

    run = bf.join(run_group, runs[0])
    submissions = bf.join(run, "submissions")

    assert bf.exists(submissions), f"Expected a submissions directory at {submissions}"

    timestamps = sorted(d for d in bf.listdir(submissions))

    assert len(timestamps) >= 1, "Expected rollout to generate at least one submission."

    latest = timestamps[-1]
    submission_folder = bf.join(submissions, latest)
    submission = bf.join(submission_folder, "submission.tar.gz")

    assert bf.exists(submission), f"Submission tarball {submission} does not exist."

    most_recent_submission = bf.join(submission_folder, "submission")

    if bf.exists(most_recent_submission):
        return Rollout(
            run_group_id=run_groups[0],
            run_group=run_group,
            run_id=runs[0],
            run=run,
            most_recent_submission=most_recent_submission,
        )

    bf.makedirs(most_recent_submission)

    # Tarball handling:
    # 1) Copy the tarball (possibly remote) to a local tmp file so `tarfile` can read it.
    # 2) Extract locally into a tmp directory, stripping the timestamp prefix.
    # 3) Copy the extracted tree back to `most_recent_submission` (local or remote) via `blobfile`.
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmpfile:
        local_submission = tmpfile.name
        bf.copy(submission, local_submission, overwrite=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            with tarfile.open(local_submission, "r:gz") as tf:
                for member in tf.getmembers():
                    assert member.name.startswith(latest), (
                        f"Expected one extracted folder called {latest} in submission.tar.gz file, "
                        f"but found {member.name}!"
                    )

                    member.name = member.name[len(latest) + 1 :]
                    tf.extract(member, path=tmpdir)

            for root, _, files in os.walk(tmpdir):
                for fname in files:
                    src_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(src_path, tmpdir)
                    dst_path = bf.join(most_recent_submission, rel_path)
                    dst_dir = os.path.dirname(dst_path)

                    if not bf.exists(dst_dir):
                        bf.makedirs(dst_dir)

                    bf.copy(src_path, dst_path)

    return Rollout(
        run_group=run_group,
        run_group_id=run_groups[0],
        run=run,
        run_id=runs[0],
        most_recent_submission=most_recent_submission,
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running")
@pytest.mark.parametrize(
    "solver_config, reproduction_config, judge_config, runs_dir",
    [
        (
            LOCAL_DUMMY_SOLVER_CONFIG,
            LOCAL_REPRODUCTION_CONFIG,
            LOCAL_JUDGE_CONFIG,
            LOCAL_RUNS_DIR,
        ),
    ],
)
async def test_dummy_agent_generates_expected_files(
    solver_config: PythonCodingSolver,
    reproduction_config: ReproductionConfig,
    judge_config: JudgeConfig,
    runs_dir: str,
) -> None:
    # Given
    unique_runs_dir = bf.join(runs_dir, GIT_COMMIT_HASH, uuid.uuid4().hex)

    # When
    await _generate_rollout(
        solver_config=solver_config,
        judge_config=judge_config,
        reproduction_config=reproduction_config,
        runs_dir=unique_runs_dir,
    )

    rollout = _get_rollout(unique_runs_dir)

    # Then
    assert bf.exists(bf.join(rollout.run_group, "group.log"))
    assert bf.exists(bf.join(rollout.run, "run.log"))
    assert rollout.most_recent_submission is not None
    assert bf.exists(bf.join(rollout.most_recent_submission, "logs", "run.log"))
    assert bf.exists(bf.join(rollout.most_recent_submission, "logs", "docker.log"))
    assert bf.exists(bf.join(rollout.most_recent_submission, "submission", "reproduce.sh"))


@pytest.mark.asyncio
@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running.")
@pytest.mark.parametrize(
    "solver_config, reproduction_config, judge_config, runs_dir",
    [
        (
            LOCAL_DUMMY_SOLVER_CONFIG,
            LOCAL_REPRODUCTION_CONFIG,
            LOCAL_JUDGE_CONFIG,
            LOCAL_RUNS_DIR,
        ),
    ],
)
async def test_dummy_agents_docker_log_contains_hello_from_docker(
    solver_config: PythonCodingSolver,
    reproduction_config: ReproductionConfig,
    judge_config: JudgeConfig,
    runs_dir: str,
) -> None:
    # Given
    unique_runs_dir = bf.join(runs_dir, GIT_COMMIT_HASH, uuid.uuid4().hex)

    # When
    await _generate_rollout(
        solver_config=solver_config,
        judge_config=judge_config,
        reproduction_config=reproduction_config,
        runs_dir=unique_runs_dir,
    )

    rollout = _get_rollout(unique_runs_dir)

    # Then
    assert rollout.most_recent_submission is not None
    assert bf.exists(bf.join(rollout.most_recent_submission, "logs", "docker.log"))

    docker_log = bf.join(rollout.most_recent_submission, "logs", "docker.log")

    with bf.BlobFile(docker_log, "rb") as f:
        log_content = f.read().decode("utf-8")

        assert "Hello from Docker!" in log_content


@pytest.mark.asyncio
@pytest.mark.slow()
@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running.")
@pytest.mark.skipif(in_ci(), reason="Skip non-dummy runs in CI.")
@pytest.mark.parametrize(
    "solver_config, reproduction_config, judge_config, runs_dir",
    [
        (
            LOCAL_BASICAGENT_SOLVER_CONFIG,
            LOCAL_REPRODUCTION_CONFIG,
            LOCAL_JUDGE_CONFIG,
            LOCAL_RUNS_DIR,
        ),
    ],
)
async def test_basicagent_generates_expected_files(
    solver_config: PythonCodingSolver,
    reproduction_config: ReproductionConfig,
    judge_config: JudgeConfig,
    runs_dir: str,
) -> None:
    # Given
    unique_runs_dir = bf.join(runs_dir, GIT_COMMIT_HASH, uuid.uuid4().hex)

    # When
    await _generate_rollout(
        solver_config=solver_config,
        judge_config=judge_config,
        reproduction_config=reproduction_config,
        runs_dir=unique_runs_dir,
    )

    rollout = _get_rollout(unique_runs_dir)

    # Then
    assert bf.exists(bf.join(rollout.run_group, "group.log"))
    assert bf.exists(bf.join(rollout.run, "run.log"))
    assert bf.exists(bf.join(rollout.run, "agent.log"))
    assert rollout.most_recent_submission is not None
    assert bf.exists(bf.join(rollout.most_recent_submission, "logs", "docker.log"))


@pytest.mark.asyncio
@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running")
@pytest.mark.parametrize(
    "solver_config, reproduction_config, judge_config, runs_dir",
    [
        (
            LOCAL_DUMMY_SOLVER_CONFIG,
            LOCAL_REPRODUCTION_CONFIG,
            LOCAL_JUDGE_CONFIG,
            LOCAL_RUNS_DIR,
        ),
        pytest.param(
            LOCAL_BASICAGENT_SOLVER_CONFIG,
            LOCAL_REPRODUCTION_CONFIG,
            LOCAL_JUDGE_CONFIG,
            LOCAL_RUNS_DIR,
            marks=pytest.mark.slow(),
        ),
    ],
)
async def test_agent_group_log_does_not_contain_errors(
    solver_config: PythonCodingSolver,
    reproduction_config: ReproductionConfig,
    judge_config: JudgeConfig,
    runs_dir: str,
) -> None:
    # Given
    unique_runs_dir = bf.join(runs_dir, GIT_COMMIT_HASH, uuid.uuid4().hex)

    # When
    await _generate_rollout(
        solver_config=solver_config,
        judge_config=judge_config,
        reproduction_config=reproduction_config,
        runs_dir=unique_runs_dir,
    )

    rollout = _get_rollout(unique_runs_dir)
    group_log = bf.join(rollout.run_group, "group.log")

    # Then
    assert bf.exists(bf.join(rollout.run_group, "group.log"))

    with bf.BlobFile(group_log, "rb") as f:
        log_content = f.read().decode("utf-8").lower()
        assert "error" not in log_content


@pytest.mark.asyncio
@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running")
@pytest.mark.parametrize(
    "solver_config, reproduction_config, judge_config, runs_dir",
    [
        (
            LOCAL_DUMMY_SOLVER_CONFIG,
            LOCAL_REPRODUCTION_CONFIG,
            LOCAL_JUDGE_CONFIG,
            LOCAL_RUNS_DIR,
        ),
        pytest.param(
            LOCAL_BASICAGENT_SOLVER_CONFIG,
            LOCAL_REPRODUCTION_CONFIG,
            LOCAL_JUDGE_CONFIG,
            LOCAL_RUNS_DIR,
            marks=pytest.mark.slow(),
        ),
    ],
)
async def test_agent_receives_valid_grade_for_its_rollout(
    solver_config: PythonCodingSolver,
    reproduction_config: ReproductionConfig,
    judge_config: JudgeConfig,
    runs_dir: str,
) -> None:
    # Given
    unique_runs_dir = bf.join(runs_dir, GIT_COMMIT_HASH, uuid.uuid4().hex)

    # When
    await _generate_rollout(
        solver_config=solver_config,
        judge_config=judge_config,
        reproduction_config=reproduction_config,
        runs_dir=unique_runs_dir,
    )

    rollout = _get_rollout(unique_runs_dir)
    fpath = bf.join(rollout.run, "grade.json")

    # Then
    assert bf.exists(fpath)

    data = json.load(bf.BlobFile(fpath, "r"))  # Check it can run without error

    assert "paperbench_result" in data
    assert isinstance(data["paperbench_result"], dict)
    assert "judge_output" in data["paperbench_result"]
    assert isinstance(data["paperbench_result"]["judge_output"], dict)
    assert "graded_task_tree" in data["paperbench_result"]["judge_output"]
    assert isinstance(data["paperbench_result"]["judge_output"]["graded_task_tree"], dict)

    graded_task_tree = data["paperbench_result"]["judge_output"]["graded_task_tree"]
    GradedTaskNode.from_dict(graded_task_tree)  # Check it can run without error


@pytest.mark.asyncio
@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running")
@pytest.mark.skipif(in_ci(), reason="Skip execution of agent submissions in CI.")
@pytest.mark.parametrize(
    "solver_config, reproduction_config, judge_config, runs_dir",
    [
        (
            LOCAL_DUMMY_SOLVER_CONFIG,
            LOCAL_REPRODUCTION_WITHOUT_SKIP_CONFIG,
            LOCAL_JUDGE_CONFIG,
            LOCAL_RUNS_DIR,
        ),
    ],
)
async def test_reproduced_rollout_produces_valid_execution_metadata(
    solver_config: PythonCodingSolver,
    reproduction_config: ReproductionConfig,
    judge_config: JudgeConfig,
    runs_dir: str,
) -> None:
    # Given
    unique_runs_dir = bf.join(runs_dir, GIT_COMMIT_HASH, uuid.uuid4().hex)

    # When
    await _generate_rollout(
        solver_config=solver_config,
        judge_config=judge_config,
        reproduction_config=reproduction_config,
        runs_dir=unique_runs_dir,
    )

    rollout = _get_rollout(unique_runs_dir)
    pattern = bf.join(rollout.run, "**/submission_executed_metadata.json")
    metadata_paths = list(bf.glob(pattern))

    # Then
    assert len(list(metadata_paths)) == 1

    metadata_path = metadata_paths[0]

    assert bf.exists(metadata_path)

    metadata = json.load(bf.BlobFile(metadata_path, "r"))

    assert "repro_script_exists" in metadata
    assert isinstance(metadata["repro_script_exists"], bool)
    assert metadata["repro_script_exists"]

    assert "repro_execution_time" in metadata
    assert isinstance(metadata["repro_execution_time"], (int, float))
    assert metadata["repro_execution_time"] > 0


@pytest.mark.asyncio
@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running")
@pytest.mark.skipif(in_ci(), reason="Skip execution of agent submissions in CI.")
@pytest.mark.parametrize(
    "solver_config, reproduction_config, judge_config, runs_dir",
    [
        (
            LOCAL_DUMMY_SOLVER_CONFIG,
            LOCAL_REPRODUCTION_WITHOUT_SKIP_CONFIG,
            LOCAL_JUDGE_CONFIG,
            LOCAL_RUNS_DIR,
        ),
        pytest.param(
            LOCAL_BASICAGENT_SOLVER_CONFIG,
            LOCAL_REPRODUCTION_WITHOUT_SKIP_CONFIG,
            LOCAL_JUDGE_CONFIG,
            LOCAL_RUNS_DIR,
            marks=pytest.mark.slow(),
        ),
    ],
)
async def test_rollout_with_reproduction_step_produces_an_executed_submission(
    solver_config: PythonCodingSolver,
    reproduction_config: ReproductionConfig,
    judge_config: JudgeConfig,
    runs_dir: str,
) -> None:
    # Given
    unique_runs_dir = bf.join(runs_dir, GIT_COMMIT_HASH, uuid.uuid4().hex)

    # When
    await _generate_rollout(
        solver_config=solver_config,
        judge_config=judge_config,
        reproduction_config=reproduction_config,
        runs_dir=unique_runs_dir,
    )

    rollout = _get_rollout(unique_runs_dir)
    executed_submissions = bf.glob(bf.join(rollout.run, "**/submission_executed.tar.gz"))

    # Then
    assert len(list(executed_submissions)) >= 1


@pytest.mark.asyncio
@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running")
@pytest.mark.parametrize(
    "solver_config, reproduction_config, judge_config, runs_dir",
    [
        (
            LOCAL_DUMMY_SOLVER_CONFIG,
            LOCAL_REPRODUCTION_CONFIG,
            LOCAL_JUDGE_CONFIG,
            LOCAL_RUNS_DIR,
        ),
    ],
)
async def test_resuming_dummy_agent_rollouts_generates_more_rollouts(
    solver_config: PythonCodingSolver,
    reproduction_config: ReproductionConfig,
    judge_config: JudgeConfig,
    runs_dir: str,
    local_dummy_rollout: Rollout,
) -> None:
    # Given
    n_tries = 2
    unique_runs_dir = bf.join(runs_dir, GIT_COMMIT_HASH, uuid.uuid4().hex)

    shutil.copytree(  # Copy over a single run
        local_dummy_rollout.run_group,
        bf.join(unique_runs_dir, local_dummy_rollout.run_group_id),
        dirs_exist_ok=True,
    )

    # When
    await _generate_rollout(
        solver_config=solver_config,
        judge_config=judge_config,
        reproduction_config=reproduction_config,
        runs_dir=unique_runs_dir,
        resume_run_group_id=local_dummy_rollout.run_group_id,
        n_tries=n_tries,
        exist_ok=True,
    )

    # Then
    run_groups = [d for d in bf.listdir(unique_runs_dir) if bf.isdir(bf.join(unique_runs_dir, d))]

    assert len(run_groups) == 1

    run_group = bf.join(unique_runs_dir, run_groups[0])
    runs = [d for d in bf.listdir(run_group) if bf.isdir(bf.join(run_group, d))]
    group_log = bf.join(run_group, "group.log")

    assert len(runs) == n_tries
    assert bf.exists(group_log)

    with bf.BlobFile(group_log, "rb") as f:
        log_content = f.read().decode("utf-8")

        assert log_content.count("Grades saved to") == n_tries


@pytest.mark.asyncio
@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running")
@pytest.mark.parametrize(
    "solver_config, reproduction_config, judge_config, runs_dir",
    [
        (
            LOCAL_DUMMY_SOLVER_CONFIG,
            LOCAL_REPRODUCTION_CONFIG,
            LOCAL_JUDGE_CONFIG,
            LOCAL_RUNS_DIR,
        ),
    ],
)
async def test_rollout_runs_when_varying_grade_locally_parameter(
    solver_config: PythonCodingSolver,
    reproduction_config: ReproductionConfig,
    judge_config: JudgeConfig,
    runs_dir: str,
) -> None:
    # Given
    unique_runs_dir = bf.join(runs_dir, GIT_COMMIT_HASH, uuid.uuid4().hex)

    # When
    await _generate_rollout(
        solver_config=solver_config,
        judge_config=judge_config,
        reproduction_config=reproduction_config,
        runs_dir=unique_runs_dir,
    )

    rollout = _get_rollout(unique_runs_dir)

    # Then
    assert bf.exists(bf.join(rollout.run, "grade.json"))
