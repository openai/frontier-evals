import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from dotenv import load_dotenv
from nanoeval_alcatraz.alcatraz_computer_interface import AlcatrazComputerInterface
from tenacity import AsyncRetrying, RetryCallState, retry_if_exception_type, stop_after_attempt

from alcatraz.clusters.interface import AlcatrazException
from alcatraz.clusters.local import ClusterConfig
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.infra.alcatraz import put_file_in_computer
from paperbench.utils import find_dotenv

logger = structlog.stdlib.get_logger(component=__name__)
load_dotenv(find_dotenv())


async def put_submission_in_computer(
    computer: ComputerInterface,
    submission_path: str,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
) -> None:
    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )
    ctx_logger.info(f"Placing submission in computer from: {submission_path}")
    tar_gz_on_computer = "/tmp/logs.tar.gz"
    # Put the tar.gz to the container
    await put_file_in_computer(
        computer=computer,
        blobfile_path=submission_path,
        dest_path=tar_gz_on_computer,
        run_group_id=run_group_id,
        runs_dir=runs_dir,
        run_id=run_id,
    )

    # Extract tar.gz into a unique temp dir to avoid collisions.
    extract_dir = f"/tmp/pb_extract_{uuid.uuid4().hex}"
    cmd = f"mkdir -p {extract_dir} && tar -xzf {tar_gz_on_computer} -C {extract_dir}"
    ctx_logger.info(f"Extracting submission: {cmd}")
    result = await computer.check_shell_command(cmd)

    # Move the submission directory into /submission deterministically
    cmd = f"rm -rf /submission && mv {extract_dir}/submission /submission"
    ctx_logger.info(f"Placing submission to /submission: {cmd}")
    await computer.check_shell_command(cmd)

    # list files in /submission
    result = await computer.check_shell_command("ls -la /submission")
    ctx_logger.info(f"Files in /submission: {result.output.decode('utf-8')}")


@asynccontextmanager
async def start_alcatraz_computer(
    cluster_config: ClusterConfig,
    max_attempts: int = 5,
) -> AsyncGenerator[ComputerInterface, None]:
    """Helper method for starting an AlcatrazComputerInterface given a ClusterConfig."""

    def before_sleep(state: RetryCallState) -> None:
        exception = state.outcome.exception() if state.outcome else None
        logger.warning(
            f"Cluster start failed on attempt {state.attempt_number} out of {max_attempts}; "
            f"retrying due to '{exception}'"
        )

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception_type(AlcatrazException),
        before_sleep=before_sleep,
    ):
        with attempt:
            async with cluster_config.build() as cluster:
                yield AlcatrazComputerInterface(cluster_value=cluster)
