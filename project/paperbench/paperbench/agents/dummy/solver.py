import json
import time
from pathlib import Path

import blobfile as bf
import structlog
from typing_extensions import override

import chz
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.agents.base import BasePBSolver
from paperbench.agents.utils import check_for_existing_run, sanity_check_docker
from paperbench.constants import AGENT_DIR, LOGS_DIR, SUBMISSION_DIR, WORKSPACE_BASE
from paperbench.nano.structs import AgentOutput
from paperbench.nano.task import PBTask

logger = structlog.stdlib.get_logger(component=__name__)


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

    async def _run_agent(self, computer: ComputerInterface, task: PBTask) -> AgentOutput:
        # Don't run the agent if we already have agent logs
        agent_output = await check_for_existing_run(task)
        if agent_output:
            return agent_output

        start_time = time.time()
        with open(Path(__file__).parent / "main.py", "rb") as f:
            main_py = f.read()
        await computer.upload(main_py, f"{AGENT_DIR}/main.py")

        await sanity_check_docker(computer)
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
            json.dump(agent_output.model_dump(), f, indent=4)

        with bf.BlobFile(bf.join(task.run_dir, "agent.log"), "w") as f:
            f.write(
                "This is a dummy agent that runs some basic debug commands and uploads the results.\n"
            )

        return agent_output
