import json
import tarfile
import tempfile
import uuid
from contextlib import contextmanager, nullcontext

import blobfile as bf
import pytest
import structlog.stdlib
from nanoeval.eval import EvalSpec
from nanoeval.evaluation import run
from nanoeval.setup import global_exit_stack
from paperbench.judge.judge import GradedTaskNode
from paperbench.nano.entrypoint import DefaultRunnerArgs
from paperbench.nano.eval import PaperBench
from paperbench.nano.logging import PaperBenchLibraryConfig, setup_logging
from paperbench.utils import create_run_dir, create_run_id, in_ci, is_docker_running
from utils import (
    assert_rollout_files_exist,
    check_group_log_for_errors,
    create_fake_submission,
    setup_judge_config,
    setup_reproduction_config,
    setup_solver,
)

logger = structlog.stdlib.get_logger(component=__name__)


@contextmanager
def run_dir_ctx_manager():
    runs_dir = None
    with tempfile.TemporaryDirectory() as runs_dir:
        yield runs_dir


@pytest.mark.asyncio
@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running")
@pytest.mark.parametrize(
    "agent_id",
    [
        pytest.param(
            agent_id,
            marks=(
                pytest.mark.skipif(
                    in_ci() and (agent_id != "dummy"),
                    reason="Only running LocalCluster dummy rollouts in CI",
                )
            ),
        )
        for agent_id in ("dummy", "aisi-basic-agent-openai-dev")
    ],
)
async def test_rollout(agent_id: str):
    """
    Test that executing an agent rollout runs without errors and produces expected output. We do not
    perform reproduction or grading in this test.
    """
    with run_dir_ctx_manager() as runs_dir:
        runs_dir = bf.join(runs_dir, uuid.uuid4().hex)

        solver = setup_solver(agent_id)
        judge_config = setup_judge_config()
        reproduction_config = setup_reproduction_config()
        async with global_exit_stack:
            setup_logging(PaperBenchLibraryConfig())
            runner_args = DefaultRunnerArgs(max_retries=0)
            paperbench = PaperBench(
                paper_split="debug",
                solver=solver,
                judge=judge_config,
                local_runs_dir=runs_dir,
                reproduction=reproduction_config,
            )

            # Run evaluation; we run like this to avoid setting rlimit via nanoeval_entrypoint
            await run(EvalSpec(eval=paperbench, runner=runner_args))

        run_dirs = [i for i in bf.listdir(runs_dir) if bf.isdir(bf.join(runs_dir, i))]
        assert (
            len(run_dirs) == 1
        ), f"Expected exactly one run group directory in {runs_dir}, found {len(run_dirs)}"
        run_group_dir = bf.join(runs_dir, run_dirs[0])

        check_group_log_for_errors(run_group_dir)

        paper_dirs = [i for i in bf.listdir(run_group_dir) if bf.isdir(bf.join(run_group_dir, i))]
        assert (
            len(paper_dirs) == 1
        ), f"Expected exactly one paper directory in {run_group_dir}, found {len(paper_dirs)}"
        paper_dir = bf.join(run_group_dir, paper_dirs[0])

        assert_rollout_files_exist(paper_dir, agent_id)


@pytest.mark.asyncio
@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running")
@pytest.mark.skipif(in_ci(), reason="Not running resume tests in CI")
async def test_resuming():
    """
    Test that we can resume a partially-completed run group using the dummy agent. We construct a run group that has one
    fake "rice" run, and resume it, using n_tries=3, i.e., we expect 2 additional runs to be executed. We
    do not perform reproduction or grading
    """
    agent_id = "dummy"
    with run_dir_ctx_manager() as runs_dir:
        # Create a partially-completed run group, containing one "rice" run
        run_group_id = uuid.uuid4().hex
        runs_dir = bf.join(runs_dir, run_group_id)
        run_id = create_run_id("rice")
        run_dir = create_run_dir(run_group_id, run_id, runs_dir)
        create_fake_submission(run_dir)

        solver = setup_solver(agent_id)
        judge_config = setup_judge_config()
        reproduction_config = setup_reproduction_config()
        async with global_exit_stack:
            setup_logging(PaperBenchLibraryConfig())
            runner_args = DefaultRunnerArgs(max_retries=0)
            paperbench = PaperBench(
                paper_split="debug",
                solver=solver,
                judge=judge_config,
                local_runs_dir=runs_dir,
                reproduction=reproduction_config,
                resume_run_group_id=run_group_id,
                n_tries=3,
            )

            # Run evaluation; we run like this to avoid setting rlimit via nanoeval_entrypoint
            await run(EvalSpec(eval=paperbench, runner=runner_args))

        run_dirs = [i for i in bf.listdir(runs_dir) if bf.isdir(bf.join(runs_dir, i))]
        assert (
            len(run_dirs) == 1
        ), f"Expected exactly one run group directory in {runs_dir}, found {len(run_dirs)}"
        run_group_dir = bf.join(runs_dir, run_dirs[0])

        check_group_log_for_errors(run_group_dir)

        paper_dirs = [i for i in bf.listdir(run_group_dir) if bf.isdir(bf.join(run_group_dir, i))]
        assert (
            len(paper_dirs) == 3
        ), f"Expected exactly 3 paper directories in {run_group_dir}, found {len(paper_dirs)}"

        # Check the existing run was skipped
        paper_dir = bf.join(run_group_dir, run_id)
        with bf.BlobFile(bf.join(paper_dir, "run.log"), "r") as f:
            run_log = f.read()
        assert "skipping rollouts" in run_log.lower(), f"Expected run {run_id} to be skipped"

        # Check that two runs were executed
        for paper_dir in paper_dirs:
            if paper_dir == run_id:  # Skip the original fake run
                continue
            assert_rollout_files_exist(bf.join(run_group_dir, paper_dir), agent_id)


@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running")
@pytest.mark.skipif(in_ci(), reason="Not running reproduction tests in CI")
async def test_reproduction():
    """
    Test reproduction produces the expected output when using the dummy agent. We create a fake result of a rollout.
    Grading is not performed in this test.
    """
    with run_dir_ctx_manager() as runs_dir:
        # Create a partially-completed run group, containing one "rice" run
        run_group_id = uuid.uuid4().hex
        runs_dir = bf.join(runs_dir, run_group_id)
        run_id = create_run_id("rice")
        run_dir = create_run_dir(run_group_id, run_id, runs_dir)
        create_fake_submission(run_dir)

        solver = setup_solver("dummy")
        judge_config = setup_judge_config()
        reproduction_config = setup_reproduction_config(skip_reproduction=False)
        async with global_exit_stack:
            setup_logging(PaperBenchLibraryConfig())
            runner_args = DefaultRunnerArgs(max_retries=0)
            paperbench = PaperBench(
                paper_split="debug",
                solver=solver,
                judge=judge_config,
                local_runs_dir=runs_dir,
                reproduction=reproduction_config,
                resume_run_group_id=run_group_id,
            )

            # Run evaluation; we run like this to avoid setting rlimit via nanoeval_entrypoint
            await run(EvalSpec(eval=paperbench, runner=runner_args))

        run_dirs = [i for i in bf.listdir(runs_dir) if bf.isdir(bf.join(runs_dir, i))]
        assert (
            len(run_dirs) == 1
        ), f"Expected exactly one run group directory in {runs_dir}, found {len(run_dirs)}"
        run_group_dir = bf.join(runs_dir, run_dirs[0])

        check_group_log_for_errors(run_group_dir)

        paper_dirs = [i for i in bf.listdir(run_group_dir) if bf.isdir(bf.join(run_group_dir, i))]
        assert (
            len(paper_dirs) == 1
        ), f"Expected exactly one paper directory in {run_group_dir}, found {len(paper_dirs)}"
        paper_dir = bf.join(run_group_dir, paper_dirs[0])

        # Check repro metadata
        repro_metadata_files = [
            i for i in bf.listdir(paper_dir) if i.endswith("_repro_metadata.json")
        ]
        assert (
            len(repro_metadata_files) == 1
        ), f"Expected one repro metadata file in {paper_dir}, found {len(repro_metadata_files)}"
        repro_metadata_file = bf.join(paper_dir, repro_metadata_files[0])
        repro_metadata = json.load(bf.BlobFile(repro_metadata_file, "r"))
        assert repro_metadata[
            "repro_script_exists"
        ], f"repro_script_exists is False in {repro_metadata_file}"
        assert (
            repro_metadata["repro_execution_time"] > 0
        ), f"repro_execution_time expected to be greater than 0 in {repro_metadata_file}"
        assert (
            "hello_world" not in repro_metadata["files_before_reproduce"]
        ), f"hello_world not expected to exist before reproduce.sh in {repro_metadata_file}"
        assert (
            "hello_world" in repro_metadata["files_after_reproduce"]
        ), f"hello_world expected to be created by reproduce.sh in {repro_metadata_file}"

        # Check reproduced submission
        repro_tar_files = [i for i in bf.listdir(paper_dir) if i.endswith("_repro.tar.gz")]
        assert (
            len(repro_tar_files) == 1
        ), f"Expected one repro tar.gz file in {paper_dir}, found {len(repro_tar_files)}"
        repro_tar_file = bf.join(paper_dir, repro_tar_files[0])

        with tempfile.TemporaryDirectory() as tmp_submission_dir:
            with bf.BlobFile(repro_tar_file, "rb") as f:
                with tarfile.open(fileobj=f) as tar:
                    tar.extractall(path=tmp_submission_dir)

            checkpoint_dirs = [
                i
                for i in bf.listdir(tmp_submission_dir)
                if bf.isdir(bf.join(tmp_submission_dir, i))
            ]
            assert (
                len(checkpoint_dirs) == 1
            ), f"Expected exactly one checkpoint directory in {tmp_submission_dir}, found {len(checkpoint_dirs)}"
            checkpoint_dir = bf.join(tmp_submission_dir, checkpoint_dirs[0])

            assert bf.exists(
                bf.join(checkpoint_dir, "reproduce.sh")
            ), f"reproduce.sh not found in {checkpoint_dir}"
            assert bf.exists(
                bf.join(checkpoint_dir, "hello_world")
            ), f"hello_world not found in {checkpoint_dir}"


@pytest.mark.skipif(not is_docker_running(), reason="Docker is not running")
@pytest.mark.parametrize(
    "judge_scaffold",
    [
        pytest.param(
            judge_scaffold,
            marks=(pytest.mark.skipif(in_ci(), reason="Not running grading tests in CI")),
        )
        for judge_scaffold in ("dummy", "simple")
    ],
)
async def test_grading(
    judge_scaffold: str,
):
    """
    Test grading produces the expected output when using the dummy agent. We create a fake result of a rollout,
    skip reproduction, and then grade the submission.
    """
    with run_dir_ctx_manager() as runs_dir:
        # Create a partially-completed run group, containing one "rice" run
        run_group_id = uuid.uuid4().hex
        runs_dir = bf.join(runs_dir, run_group_id)
        run_id = create_run_id("rice")
        run_dir = create_run_dir(run_group_id, run_id, runs_dir)
        create_fake_submission(run_dir)

        solver = setup_solver("dummy")
        judge_config = setup_judge_config(
            skip_grading=False,
        )
        reproduction_config = setup_reproduction_config()
        async with global_exit_stack:
            setup_logging(PaperBenchLibraryConfig())
            runner_args = DefaultRunnerArgs(max_retries=0)
            paperbench = PaperBench(
                paper_split="debug",
                solver=solver,
                judge=judge_config,
                local_runs_dir=runs_dir,
                reproduction=reproduction_config,
                resume_run_group_id=run_group_id,
            )

            # Run evaluation; we run like this to avoid setting rlimit via nanoeval_entrypoint
            await run(EvalSpec(eval=paperbench, runner=runner_args))

        run_dirs = [i for i in bf.listdir(runs_dir) if bf.isdir(bf.join(runs_dir, i))]
        assert (
            len(run_dirs) == 1
        ), f"Expected exactly one run group directory in {runs_dir}, found {len(run_dirs)}"
        run_group_dir = bf.join(runs_dir, run_dirs[0])

        check_group_log_for_errors(run_group_dir)

        paper_dirs = [i for i in bf.listdir(run_group_dir) if bf.isdir(bf.join(run_group_dir, i))]
        assert (
            len(paper_dirs) == 1
        ), f"Expected exactly one paper directory in {run_group_dir}, found {len(paper_dirs)}"
        paper_dir = bf.join(run_group_dir, paper_dirs[0])

        # Check grader output
        grader_output_files = [
            i for i in bf.listdir(paper_dir) if i.endswith("_grader_output_0.json")
        ]
        assert (
            len(grader_output_files) == 1
        ), f"Expected one grader output file in {paper_dir}, found {len(grader_output_files)}"
        grader_output_file = bf.join(paper_dir, grader_output_files[0])
        grader_output = json.load(bf.BlobFile(grader_output_file, "r"))  # check we can load

        if judge_scaffold == "dummy":
            assert (
                grader_output["score"] == 1.0
            ), f"score expected to be 1.0 in {grader_output_file} when using dummy judge scaffold"

        graded_task_tree = GradedTaskNode.from_dict(
            grader_output["graded_task_tree"]
        )  # Check graded task tree can be loaded

        # Check pb_result
        assert bf.exists(
            bf.join(paper_dir, "pb_result.json")
        ), f"pb_result.json not found in {paper_dir}"
