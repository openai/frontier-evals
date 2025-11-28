from __future__ import annotations

import asyncio
import shutil
import tarfile
import tempfile
import time

import blobfile as bf
import structlog
from typing_extensions import Iterable, override

import chz
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.agents.base import BasePBSolver
from paperbench.nano.structs import AgentOutput
from paperbench.nano.task import PBTask
from paperbench.paper_registry import paper_registry
from paperbench.utils import build_canonical_sub_path

logger = structlog.stdlib.get_logger(component=__name__)


def _iter_dir(path: str) -> Iterable[str]:
    """
    Yield the immediate children of *path*, returned as full paths.
    """
    prefix = path.rstrip("/")
    for name in bf.listdir(path):
        yield f"{prefix}/{name}"


def validate_submissions_dir_structure(
    solver: PBDirectSubmissionSolver, submissions_dir_field_name: str
) -> None:
    """
    Validates that the contents of the submissions_dir follow the correct structure.
    Namely
    ```
    submission_dir/
        paper_id/
            submission/
            other_submission/
            ...
            (i.e. multiple submissions for the same paper. single submission is also
            fine. Missing submissions for a try will be treated as missing submissions)
        ...
        (does not need a folder for each paper, missing papers will be treated as
        missing submissions)
    ```
    """
    submissions_dir: str = getattr(solver, submissions_dir_field_name)
    assert bf.exists(submissions_dir), f"{submissions_dir!r} does not exist"
    assert bf.isdir(submissions_dir), f"{submissions_dir!r} is not a directory"
    paper_dirs = [entry for entry in _iter_dir(submissions_dir) if bf.isdir(entry)]
    for paper_dir in paper_dirs:
        parsed_paper_id = paper_dir.split("/")[-1]
        assert parsed_paper_id in paper_registry.list_paper_ids(), (
            f"{paper_dir!r} is not a valid paper id"
        )


def _put_folder_submission_in_canonical_path(submission_dir: str, canonical_run_dir: str) -> None:
    """Puts a submission folder into the canonical .tar.gz path expected for grading"""
    # 1. make a temporarily file
    with tempfile.TemporaryFile() as tmp:
        # 2. populate it with the tarred submission
        with tarfile.open(mode="w:gz", fileobj=tmp) as tar:
            tar.add(submission_dir, arcname="submission")
        # 3. prepare the destination path
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S-%Z", time.gmtime())
        destination_path = build_canonical_sub_path(run_dir=canonical_run_dir, timestamp=timestamp)
        bf.makedirs(bf.dirname(destination_path))
        # 4. copy the tarred submission to the destination path
        tmp.seek(0)
        with bf.BlobFile(destination_path, "wb") as out_blob:
            shutil.copyfileobj(tmp, out_blob)
    # 5. the temporary file is automatically cleaned up by the context manager


@chz.chz
class PBDirectSubmissionSolver(BasePBSolver):
    submissions_dir: str = chz.field(validator=validate_submissions_dir_structure)

    @override
    def shortname(self) -> str:
        return "direct_submission_solver"

    @chz.init_property
    def paper_to_submissions(self) -> dict[str, list[str]]:
        """
        Returns a mapping from paper_id to a list of submission .tar.gz files.
        Run once after init and then cached.
        """
        paper_to_submissions = {}
        paper_dirs = [entry for entry in _iter_dir(self.submissions_dir) if bf.isdir(entry)]
        for paper_dir in paper_dirs:
            paper_id = paper_dir.split("/")[-1]
            paper_to_submissions[paper_id] = [f for f in _iter_dir(paper_dir) if bf.isdir(f)]
        return paper_to_submissions

    @chz.init_property
    def _pop_lock(self) -> asyncio.Lock:
        """Initializing an instance-level lock to guard popping submissions."""
        return asyncio.Lock()

    def _produce_direct_submission_agent_output(
        self, start_time: float, task: PBTask
    ) -> AgentOutput:
        return AgentOutput(
            run_id=task.run_id,
            time_start=start_time,
            time_end=time.time(),
            error_msg=None,
            runtime_in_seconds=time.time() - start_time,
            status_exists=False,  # irrelevant to direct submission solver
        )

    @override
    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        """
        Takes a submission for the current paper_id and places it in the correct
        location in the run dir
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
        )

        start_time = time.time()
        # technically we're skipping the rollout and just directly placing the submission in the correct location;
        # this informs task.grade not to try extracting files from the computer
        task.skipped_rollout = True

        current_paper_id = task.paper_id
        if current_paper_id not in self.paper_to_submissions:
            ctx_logger.warning(
                f"No submissions found for {task.paper_id!r},",
                " a score of 0 will be returned for this paper.",
                destinations=["group", "run"],
            )
        async with self._pop_lock:
            submission_dir = (
                # pop a submission if the paper still has submissions to grade
                self.paper_to_submissions[current_paper_id].pop()
                if self.paper_to_submissions.get(current_paper_id, [])
                else None
            )

        if not submission_dir:
            return self._produce_direct_submission_agent_output(start_time, task)

        _put_folder_submission_in_canonical_path(submission_dir, task.run_dir)

        return self._produce_direct_submission_agent_output(start_time, task)
