import json
import time
from pathlib import Path

import blobfile as bf
import structlog
from dotenv import load_dotenv
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.infra.alcatraz import put_file_in_computer, tar_and_extract_from_computer
from paperbench.scripts.run_reproduce import reproduce
from structlog.stdlib import BoundLogger

logger = structlog.stdlib.get_logger(component=__name__)
load_dotenv()


async def put_submission_in_computer(
    computer: ComputerInterface,
    submission_path: str,
    logger: BoundLogger,
):
    logger.info(f"Placing submission in computer from: {submission_path}")
    # Put the tar.gz to the container
    await put_file_in_computer(
        computer=computer,
        blobfile_path=submission_path,
        dest_path="/tmp/logs.tar.gz",
        logger=logger,
    )

    # Extract tar.gz
    cmd = "tar -xzf /tmp/logs.tar.gz -C /tmp"
    logger.info(f"Extracting submission: {cmd}")
    result = await computer.check_shell_command(cmd)

    # Move submission subdir to /submission
    # TODO: this is a hack because sometimes the submission dir is nested several levels deep
    # (e.g. from the agent's tar.gz, you get `{unzip_location}/{timestamp}/submission/` and
    # from the reproducer, you just get `{unzip_location}/submission/`), so we have to `find`
    # the submission dir and move it to /submission. We should fix this by always uploading
    # `submission` at the top level in the tar
    cmd = "find /tmp/ -type d -name submission -print0 | xargs -0 -I{} mv {} /"
    logger.info(f"Moving submission to /submission: {cmd}")
    result = await computer.check_shell_command(cmd)

    # list files in /submission
    result = await computer.check_shell_command("ls -la /submission")
    logger.info(f"Files in /submission: {result.output.decode('utf-8')}")


async def reproduce_on_computer(
    computer: ComputerInterface,
    submission_path: str,
    logger: BoundLogger,
    run_dir: str,
    submission_cluster_path: Path = Path("/submission"),
    output_cluster_path: Path = Path("/output"),
    timeout: float | None = None,
    retry_threshold: float = 0,
):
    """
    Reproduce a single submission on a computer.

    This script will spin up a pb-reproducer container on the computer to do the following:
    - Download the submission
    - Run the reproduction script
    - Upload the reproduction results
    """
    time_start = time.time()
    # TODO Don't override existing env vars if they're already set
    # TODO verify the API keys passed to the solver are set

    error_msg = None
    try:
        await computer.check_shell_command(
            f"mkdir -p {output_cluster_path} {submission_cluster_path}"
        )

        # install prerequisites
        cmd_str = "apt-get update && apt-get install -y git python3 curl sudo"
        result = await computer.send_shell_command(cmd_str)
        assert result.exit_code == 0, f"Install failed: {result}"

        # Get the submission stem which will be the prefix for various outputs
        # e.g. /path/to/2024-12-03T17-47-25-GMT.tar.gz -> 2024-12-03T17-47-25-GMT
        submission_stem = Path(submission_path).stem.split(".tar")[0]

        # Step 1: Put submission to /submission
        await put_submission_in_computer(computer, submission_path, logger)

        # Step 2: Kick off reproduction runner
        repro_metadata = await reproduce(
            computer=computer,
            submission_path=submission_cluster_path,
            logger=logger,
            timeout=timeout,
            retry_threshold=retry_threshold,
        )

        # Step 3: Save outputs
        bf.write_bytes(
            bf.join(run_dir, f"{submission_stem}_repro_metadata.json"),
            json.dumps(repro_metadata).encode("utf-8"),
        )

        # extract tar of the submission
        tar_path = output_cluster_path / f"{submission_stem}_repro.tar.gz"
        upload_to_path = bf.join(run_dir, f"{submission_stem}_repro.tar.gz")

        await tar_and_extract_from_computer(
            computer=computer,
            dir_path_on_computer=submission_cluster_path,
            tar_path_on_computer=tar_path,
            tar_path_on_target=upload_to_path,
            max_file_size="10M",
            logger=logger,
        )

        logger.info(f"Reproduced dir has been written: {upload_to_path}")
    except Exception as e:
        error_msg = str(e)
        logger.exception(f"Reproduction failed with error:\n{error_msg}")
    finally:
        time_end = time.time()
        logger.info(f"Run completed in {time_end - time_start:.2f} seconds.")

    time_end = time.time()
    logger.info(f"Reproduction completed in {time_end - time_start:.2f} seconds.")
