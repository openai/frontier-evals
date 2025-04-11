import asyncio
import json
import logging
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Optional

import blobfile as bf
from dotenv import load_dotenv
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.agents.registry import Agent
from paperbench.infra.alcatraz import tar_and_extract_from_computer
from paperbench.judge.create_judge import create_judge, handle_judge_kwargs
from paperbench.judge.judge import Judge
from paperbench.paper_registry import paper_registry
from paperbench.rubric.tasks import TaskNode
from paperbench.scripts.run_judge import JudgeOutput, get_total_token_usage
from paperbench.scripts.run_reproduce import reproduce
from paperbench.utils import get_timestamp

load_dotenv()


async def download_submission_to_container(
    computer: ComputerInterface, submission_path: str, logger: logging.Logger
):
    logger.info(f"Downloading submission from: {submission_path}")
    # Download the tar.gz to the container
    with open(submission_path, "rb") as f:
        await computer.upload(f.read(), "/tmp/logs.tar.gz")

    # Extract tar.gz
    cmd = "tar -xzf /tmp/logs.tar.gz -C /tmp"
    logger.info(f"Extracting submission: {cmd}")
    result = await computer.send_shell_command(cmd)
    assert result.exit_code == 0, f"Extract submission failed: {result}"

    # Move submission subdir to /submission
    # TODO: this is a hack because sometimes the submission dir is nested several levels deep
    # (e.g. from the agent's tar.gz, you get `{unzip_location}/{timestamp}/submission/` and
    # from the reproducer, you just get `{unzip_location}/submission/`), so we have to `find`
    # the submission dir and move it to /submission. We should fix this by always uploading
    # `submission` at the top level in the tar
    cmd = "find /tmp/ -type d -name submission -print0 | xargs -0 -I{} mv {} /"
    logger.info(f"Moving submission to /submission: {cmd}")
    result = await computer.send_shell_command(cmd)
    assert result.exit_code == 0, f"Move submission failed: {result}"

    # list files in /submission
    result = await computer.send_shell_command("ls -la /submission")
    assert result.exit_code == 0, f"List files in /submission failed: {result}"
    logger.info(f"Files in /submission: {result.output.decode('utf-8')}")


async def grade_on_cluster(
    judge: Judge, computer: ComputerInterface, logger: logging.Logger, code_only: bool
) -> JudgeOutput:
    # If the judge has a custom grade_on_cluster method, use it
    if hasattr(judge, "grade_on_cluster") and callable(getattr(judge, "grade_on_cluster", None)):
        return await judge.grade_on_cluster(computer)

    result = await computer.send_shell_command(f"test -f {judge.reproduce_sh_path}")

    if result.exit_code != 0:
        logger.warning(
            f"Reproduction script {judge.reproduce_sh_path} does not exist on the computer"
        )

    result = await computer.send_shell_command(f"test -f {judge.reproduce_log_path}")

    if result.exit_code != 0:
        logger.warning(
            f"Reproduction log {judge.reproduce_log_path} does not exist on the computer"
        )

    # Otherwise, just run the judge on the cluster using the installed judge script
    paper_id = judge.paper_path.parent.name
    judge_type = judge.judge_type
    model_name = judge.model if judge.judge_type not in ["dummy", "random"] else None
    logger.info(f"Grading paper {paper_id} on cluster")
    python_path = "/opt/conda/envs/grader/bin/python"  # use the conda env we installed in pb-grader
    code_only_string = " --code-only" if code_only else ""
    judge_completion_kwargs = getattr(judge, "completion_kwargs", None)
    reasoning_effort_str = (
        f" --reasoning-effort {judge_completion_kwargs['reasoning_effort']}"
        if judge_completion_kwargs and judge_completion_kwargs.get("reasoning_effort")
        else ""
    )
    cmd_str = (
        f"{python_path} paperbench/scripts/run_judge.py --submission-path /submission"
        + f" --paper-id {paper_id} --judge {judge_type} --model {model_name} --out-dir /output"
        + reasoning_effort_str
        + code_only_string
    )

    logger.info(f"Running judge with cmd: {cmd_str}")
    judge_result = await computer.send_shell_command(cmd_str)
    assert judge_result.exit_code == 0, f"Judge failed: {judge_result}"
    logger.info(f"Judge result: {judge_result.output.decode('utf-8')}")

    grader_output_path = Path("/output/grader_output.json")
    grader_output_bytes: bytes = await computer.download(str(grader_output_path))
    grader_output_dict = json.loads(grader_output_bytes.decode("utf-8"))

    return JudgeOutput.from_dict(grader_output_dict)


async def grade_on_computer(
    computer: ComputerInterface,
    submission_path: str,
    grader_upload_path: str,
    paper_id: str,
    judge_type: str,
    model_name: str,
    logger: logging.Logger,
    run_dir: str,
    code_only: bool = False,
    reasoning_effort: str | None = None,
) -> JudgeOutput | None:
    """
    Grade a single submission on a computer.

    This script will spin up a pb-grader container on the computer to do the following:
    - Download the submission
    - Run the Judge on the submission
    - Upload the judge results
    """

    time_start = time.time()
    logger.info(f"Grading {submission_path} for paper {paper_id}")

    error_msg = None
    grader_output = None
    try:

        # Step 1: Download submission to /submission
        await download_submission_to_container(computer, submission_path, logger)
        logger.info("Downloaded submission to /submission")

        # Step 2: Run the judge
        paper = paper_registry.get_paper(paper_id)
        with open(paper.rubric, "r") as f:
            task_tree = TaskNode.from_dict(json.load(f))
        if code_only:
            task_tree = task_tree.code_only() or task_tree.set_task_category(
                "Code Development"
            ).set_sub_tasks([])
        judge_kwargs = handle_judge_kwargs(
            judge_type, code_only, paper, model_name, reasoning_effort
        )
        judge = create_judge(
            judge_type=judge_type,
            judge_kwargs=judge_kwargs,
            paper_path=paper.paper_pdf,
            rubric=task_tree,
            addendum=paper.addendum.read_text() if paper.addendum else None,
            judge_addendum=(
                paper.judge_addendum.read_text() if paper.judge_addendum.exists() else None
            ),
            submission_dir=Path("/submission"),
            paper_md=paper.paper_md,
        )
        logger.info(f"Judge created: {judge}")
        grader_output = await grade_on_cluster(judge, computer, logger, code_only)

        logger.info(f"Graded {paper_id} at {submission_path}")

        # Step 3: Upload /output/grader_output.json
        grader_result_on_container = Path("/output/grader_output.json")
        grader_result = await computer.download(str(grader_result_on_container))
        with open(grader_upload_path, "wb") as f:
            f.write(grader_result)
        logger.info(f"Grading results have been written to local file: {grader_upload_path}")
    except Exception as e:
        error_msg = str(e)
        logger.exception(f"Grading failed with error:\n{error_msg}")
    finally:
        time_end = time.time()
        logger.info(f"Run completed in {time_end - time_start:.2f} seconds.")

    time_end = time.time()
    logger.info(f"Grading completed in {time_end - time_start:.2f} seconds.")

    return grader_output


async def grade_locally(
    submission_path: str,
    grader_upload_path: str,
    paper_id: str,
    judge_type: str,
    model_name: str,
    logger: logging.Logger,
    code_only: bool = False,
    reasoning_effort: str | None = None,
) -> Optional[JudgeOutput]:
    """
    Grade a single submission locally

    This script will:
    - Extract the submission to a temporary directory
    - Run the Judge on the submission
    - Return/Upload the judge results
    """

    time_start = time.time()
    logger.info(f"Grading {submission_path} for paper {paper_id}")

    error_msg = token_usage = None
    try:
        # Step 1: Unzip submission from submission_path to tmp dir
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)

            logger.info(f"Unzipping submission to {tmp_dir}")
            with bf.BlobFile(submission_path, "rb") as f:
                with tarfile.open(fileobj=f, mode="r") as tar:
                    tar.extractall(path=tmp_dir)

            # Step 2: Run the judge
            paper = paper_registry.get_paper(paper_id)
            with open(paper.rubric, "r") as f:
                task_tree = TaskNode.from_dict(json.load(f))
            if code_only:
                task_tree = task_tree.code_only() or task_tree.set_task_category(
                    "Code Development"
                ).set_sub_tasks([])
            judge_kwargs = handle_judge_kwargs(
                judge_type, code_only, paper, model_name, reasoning_effort
            )
            judge = create_judge(
                judge_type=judge_type,
                judge_kwargs=judge_kwargs,
                paper_path=paper.paper_pdf,
                rubric=task_tree,
                addendum=paper.addendum.read_text() if paper.addendum else None,
                judge_addendum=(
                    paper.judge_addendum.read_text() if paper.judge_addendum.exists() else None
                ),
                submission_dir=tmp_dir,
                paper_md=paper.paper_md,
            )
            logger.info(f"Judge created: {judge}")
            graded_task_tree = await judge.grade()
            if judge_type == "simple":
                token_usage = get_total_token_usage(graded_task_tree)

            logger.info(f"Graded {paper_id} at {submission_path}")

            # Step 3: Upload /output/grader_output.json
            bf.write_bytes(
                grader_upload_path,
                json.dumps(graded_task_tree.to_dict(), indent=4).encode("utf-8"),
            )
            logger.info(f"Grading results have been written to file: {grader_upload_path}")
    except Exception as e:
        error_msg = str(e)
        logger.exception(f"Grading failed with error:\n{error_msg}")
    finally:
        time_end = time.time()
        logger.info(f"Grading completed in {time_end - time_start:.2f} seconds.")

        judge_output = JudgeOutput(
            judge_type=judge_type,
            model_name=model_name,
            score=graded_task_tree.score,
            num_leaf_nodes=len(graded_task_tree.get_leaf_nodes()),
            num_invalid_leaf_nodes=(
                len([node for node in graded_task_tree.get_leaf_nodes() if not node.valid_score])
            ),
            graded_at=get_timestamp(),
            graded_task_tree=graded_task_tree,
            token_usage=token_usage,
        )
        return judge_output


async def reproduce_on_computer(
    computer: ComputerInterface,
    submission_path: str,
    logger: logging.Logger,
    run_dir: str,
    submission_cluster_path: Path = Path("/submission"),
    output_cluster_path: Path = Path("/output"),
    timeout: float | None = None,
    retry_threshold: float = 0,
):
    """
    Reproduce a single submission on a computer.

    This script will spin up a pb-grader container on the computer to do the following:
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

        # Step 1: Download submission to /submission
        await download_submission_to_container(computer, submission_path, logger)

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
        logger.error(f"Reproduction failed with error:\n{error_msg}")
    finally:
        time_end = time.time()
        logger.info(f"Run completed in {time_end - time_start:.2f} seconds.")

    time_end = time.time()
    logger.info(f"Reproduction completed in {time_end - time_start:.2f} seconds.")
