import re
from typing import Literal

import pytest
from nanoeval_alcatraz.alcatraz_computer_interface import (
    AlcatrazComputerRuntime,
)

import nanoeval
from alcatraz.clusters.local import LocalConfig
from nanoeval.evaluation import EvalSpec, RunnerArgs
from nanoeval.recorder import dummy_recorder
from nanoeval.setup import global_exit_stack
from nanoeval.solvers.computer_tasks.code_execution_interface import RuntimeConfig
from swelancer.eval import SWELancerEval
from swelancer.solvers.dummy.solver import DummySolver
from swelancer.utils.custom_logging import setup_logging, swelancer_library_config
from swelancer.utils.general import (
    get_runs_dir,
    get_tasks,
    is_docker_image,
    is_docker_running,
    is_linux_machine,
)

# Docker Configuration
DOCKER_IMAGE_TAG = "latest"
LOCAL_DOCKER_IMAGE_PREFIX = "swelancer_x86"
LOCAL_DOCKER_IMAGE = f"{LOCAL_DOCKER_IMAGE_PREFIX}_monolith:{DOCKER_IMAGE_TAG}"

# Test Configuration
SPLIT: Literal["diamond", "nondiamond", "all"] = "diamond"
TASK_TYPE: Literal["ic_swe", "swe_manager", "all"] = "ic_swe"
N_TEST_RUNS = 3

# Runner Configuration
LOCAL_CONCURRENCY = 1


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.skipif(
    not is_linux_machine(),
    reason="Disabling internet access on Alcatraz is only supported on Linux machines.",
)
@pytest.mark.skipif(
    not is_docker_running(),
    reason="Docker must be running when using LocalCluster.",
)
@pytest.mark.skipif(
    not is_docker_image(LOCAL_DOCKER_IMAGE),
    reason=f"Required Docker image '{LOCAL_DOCKER_IMAGE}' not found.",
)
@pytest.mark.parametrize("issue_id", get_tasks(split=SPLIT, task_type=TASK_TYPE))
async def test_gold_patch_passes_tests_using_local_cluster(issue_id: str) -> None:
    # Given
    runner = RunnerArgs(
        concurrency=LOCAL_CONCURRENCY,
        experimental_use_multiprocessing=False,
        enable_slackbot=False,
        recorder=dummy_recorder(),
        max_retries=2,
    )

    solver = DummySolver(
        runtime_config=RuntimeConfig(),
        test_user_tool=False,
        apply_gold_solution=True,  # Apply gold solution -- should pass
        computer_runtime=AlcatrazComputerRuntime(
            env=LocalConfig(
                pull_from_registry=False,
            )
        ),
    )

    eval = SWELancerEval(
        solver=solver,
        split=SPLIT,
        task_type=TASK_TYPE,
        taskset=[issue_id],
        runs_dir=str(get_runs_dir()),
        docker_image_prefix=LOCAL_DOCKER_IMAGE_PREFIX,
        docker_image_tag=DOCKER_IMAGE_TAG,
        use_single_image=True,
        n_test_runs=N_TEST_RUNS,
    )

    setup_logging(swelancer_library_config)

    # When
    async with global_exit_stack:
        summary = await nanoeval.run(EvalSpec(eval=eval, runner=runner))

    # Then
    assert "aggregations" in summary
    assert "num_correct" in summary["aggregations"]
    assert isinstance(summary["aggregations"]["num_correct"], int)
    assert summary["aggregations"]["num_correct"] == 1


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.skipif(
    not is_linux_machine(),
    reason="Disabling internet access on Alcatraz is only supported on Linux machines.",
)
@pytest.mark.skipif(
    not is_docker_running(),
    reason="Docker must be running when using LocalCluster.",
)
@pytest.mark.skipif(
    not is_docker_image(LOCAL_DOCKER_IMAGE),
    reason=f"Required Docker image '{LOCAL_DOCKER_IMAGE}' not found.",
)
@pytest.mark.parametrize("issue_id", get_tasks(split=SPLIT, task_type=TASK_TYPE))
async def test_without_gold_patch_fails_tests_using_local_cluster(issue_id: str) -> None:
    # Given
    runner = RunnerArgs(
        concurrency=LOCAL_CONCURRENCY,
        experimental_use_multiprocessing=False,
        enable_slackbot=False,
        recorder=dummy_recorder(),
        max_retries=0,
    )

    solver = DummySolver(
        runtime_config=RuntimeConfig(),
        test_user_tool=False,
        apply_gold_solution=False,  # Don't apply gold solution -- should fail
        computer_runtime=AlcatrazComputerRuntime(
            env=LocalConfig(
                pull_from_registry=False,
            )
        ),
    )

    eval = SWELancerEval(
        solver=solver,
        split=SPLIT,
        task_type=TASK_TYPE,
        taskset=[issue_id],
        runs_dir=str(get_runs_dir()),
        docker_image_prefix=LOCAL_DOCKER_IMAGE_PREFIX,
        docker_image_tag=DOCKER_IMAGE_TAG,
        use_single_image=True,
        n_test_runs=N_TEST_RUNS,
    )

    setup_logging(swelancer_library_config)

    # When
    async with global_exit_stack:
        summary = await nanoeval.run(EvalSpec(eval=eval, runner=runner))

    # Then
    assert "aggregations" in summary
    assert "num_correct" in summary["aggregations"]
    assert isinstance(summary["aggregations"]["num_correct"], int)
    assert summary["aggregations"]["num_correct"] == 0


@pytest.mark.slow
@pytest.mark.asyncio
@pytest.mark.skipif(
    not is_linux_machine(),
    reason="Disabling internet access on Alcatraz is only supported on Linux machines.",
)
@pytest.mark.skipif(
    not is_docker_running(),
    reason="Docker must be running when using LocalCluster.",
)
@pytest.mark.skipif(
    not is_docker_image(LOCAL_DOCKER_IMAGE),
    reason=f"Required Docker image '{LOCAL_DOCKER_IMAGE}' not found.",
)
@pytest.mark.parametrize("issue_id", get_tasks(split=SPLIT, task_type=TASK_TYPE))
async def test_user_tool_logs_using_local_cluster(issue_id: str) -> None:
    # Given
    runner = RunnerArgs(
        concurrency=LOCAL_CONCURRENCY,
        experimental_use_multiprocessing=False,
        enable_slackbot=False,
        recorder=dummy_recorder(),
        max_retries=0,
    )

    solver = DummySolver(
        runtime_config=RuntimeConfig(),
        test_user_tool=True,
        apply_gold_solution=False,
        computer_runtime=AlcatrazComputerRuntime(
            env=LocalConfig(
                pull_from_registry=False,
            )
        ),
    )

    eval = SWELancerEval(
        solver=solver,
        split=SPLIT,
        task_type=TASK_TYPE,
        taskset=[issue_id],
        runs_dir=str(get_runs_dir()),
        docker_image_prefix=LOCAL_DOCKER_IMAGE_PREFIX,
        docker_image_tag=DOCKER_IMAGE_TAG,
        use_single_image=True,
        n_test_runs=1,
    )

    setup_logging(swelancer_library_config)

    # When
    async with global_exit_stack:
        _ = await nanoeval.run(EvalSpec(eval=eval, runner=runner))

    # Then: locate the single run entry via summary mapping
    runs_dir = get_runs_dir()
    run_group_dir = runs_dir / eval.run_group_id
    run_dirs = [d for d in run_group_dir.iterdir() if d.is_dir()]
    assert len(run_dirs) == 1, f"Expected one run, got {len(run_dirs)}"
    run_dir = run_dirs[0]
    user_tool_log = run_dir / "user_tool.log"
    assert user_tool_log.exists(), f"user_tool.log not found in run {run_dir}"

    content = user_tool_log.read_text()
    assert re.search(r"^ls -al /app/expensify/user_tool", content, re.MULTILINE)
    assert re.search(r"^Exit code: 0$", content, re.MULTILINE)
    assert re.search(r"^Output: total \d+", content, re.MULTILINE)
    assert content.count("output_") >= 1
