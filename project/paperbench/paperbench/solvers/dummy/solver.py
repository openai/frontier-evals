import json
import time
from pathlib import Path

import blobfile as bf
import chz
import structlog
from nanoeval.recorder import get_recorder
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
)
from typing_extensions import override

from paperbench.agents.utils import AgentOutput
from paperbench.constants import AGENT_DIR, LOGS_DIR, SUBMISSION_DIR, WORKSPACE_BASE
from paperbench.nano.task import PBTask
from paperbench.solvers.base import BasePBSolver

logger = structlog.stdlib.get_logger(component=__name__)


DOCKER_COMMANDS = [
    # docker tests
    "docker --version",
    "docker run --rm hello-world",
    "docker ps -a",
]

BASH_COMMANDS = [
    # make dummy files
    f"conda run -n agent python {AGENT_DIR}/main.py",
    f"touch {LOGS_DIR}/run.log",
    f"touch {AGENT_DIR}/agent_was_here.txt",
    f"cat {WORKSPACE_BASE}/instructions.txt",
    f"touch {SUBMISSION_DIR}/reproduce.sh",
]


@chz.chz
class PaperBenchDummySolver(BasePBSolver):
    @override
    def shortname(self) -> str:
        return "dummy"

    async def _check_for_existing_run(self, task: PBTask) -> AgentOutput | None:
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            run_id=task.run_id,
            runs_dir=task.runs_dir,
        )

        # If agent logs already exist, we can skip running the agent
        pattern = bf.join(task.run_dir, "**/*.tar.gz")
        num_tars = len(list(bf.glob(pattern)))

        status_path = bf.join(task.run_dir, "status.json")
        status_exists = bf.exists(status_path)

        # we expect at least one tar if the run was successful: one gets uploaded at the start,
        # and one more at the end (you would have more at every half-hour checkpoint in between)
        if status_exists and num_tars >= 1:
            with bf.BlobFile(status_path, "r") as f:
                status = json.loads(f.read())

            ctx_logger.info(
                f"Agent logs already exist, skipping rollouts for {task.run_id}",
                destinations=["run"],
            )

            ctx_logger.info(f"status: {status}", destinations=["run"])
            start_time = status.get("created_at") if status.get("created_at") else time.time()
            if status.get("agent_finished_at"):
                end_time = status.get("agent_finished_at")
            elif status.get("last_updated"):
                end_time = status.get("last_updated")
            else:
                end_time = time.time()
            runtime = end_time - start_time
            agent_output = AgentOutput(
                run_id=task.run_id,
                time_start=start_time,
                time_end=end_time,
                runtime_in_seconds=runtime,
                error_msg=None,  # No error if we have status.json and tars
                status_exists=True,  # We already checked status_exists
            )
            task.skipped_rollout = True

            get_recorder().record_extra(
                {
                    "run_group_id": task.run_group_id,
                    "run_id": task.run_id,
                    "agent_output": agent_output.to_dict(),
                    "status": status,
                }
            )

            return agent_output
        return None

    @override
    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        # Don't run the agent if we already have agent logs
        agent_output = await self._check_for_existing_run(task)
        if agent_output:
            return agent_output

        start_time = time.time()
        with open(Path(__file__).parent / "main.py", "rb") as f:
            main_py = f.read()
        await computer.upload(main_py, f"{AGENT_DIR}/main.py")

        res = ""
        for cmd in DOCKER_COMMANDS:
            result = await computer.send_shell_command(cmd)
            res += f"{result.output.decode('utf-8')}\n"
        await computer.upload(res.encode("utf-8"), f"{LOGS_DIR}/docker.log")
        res = ""
        for cmd in BASH_COMMANDS:
            result = await computer.send_shell_command(cmd)
            res += f"{result.output.decode('utf-8')}\n"
        await computer.upload(res.encode("utf-8"), f"{LOGS_DIR}/run.log")

        agent_output = AgentOutput(
            run_id=task.run_id,
            time_start=start_time,
            time_end=time.time(),
            error_msg=None,
            runtime_in_seconds=time.time() - start_time,
            status_exists=True,
        )
        with bf.BlobFile(bf.join(task.run_dir, "metadata.json"), "w") as f:
            json.dump(agent_output.to_dict(), f, indent=4)

        with bf.BlobFile(bf.join(task.run_dir, "agent.log"), "w") as f:
            f.write(
                "This is a dummy agent that runs some basic debug commands and uploads the results.\n"
            )

        return agent_output
