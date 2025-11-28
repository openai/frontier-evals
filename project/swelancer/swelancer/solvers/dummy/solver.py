import asyncio
import json
from pathlib import Path
from typing import AsyncGenerator

import structlog
from nanoeval_alcatraz.alcatraz_computer_interface import (
    AlcatrazComputerRuntime,
)
from typing_extensions import override

import chz
from nanoeval.contrib.utils import run_with_startup_timeout
from nanoeval.eval import RolloutSystemError
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    ComputerRuntime,
    RuntimeConfig,
)
from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from nanoeval.solvers.computer_tasks.steps import (
    FinalResult,
    Step,
)
from nanoeval.solvers.computer_tasks.task import ComputerTask
from swelancer.eval import SWELancerTask

logger = structlog.stdlib.get_logger(component=__name__)

STARTUP_TIMEOUT = 1200  # seconds (20 mins)


@chz.chz
class DummySolver(PythonCodingSolver):
    computer_runtime: ComputerRuntime = chz.field(default_factory=AlcatrazComputerRuntime)

    log_dir: str = chz.field(default="/tmp/dummy_logs/")  # Dir on claimed VM to write dummy logs to
    runtime_config: RuntimeConfig = chz.field(default_factory=RuntimeConfig)

    test_user_tool: bool = chz.field(default=True)

    apply_gold_solution: bool = chz.field(default=False)

    def shortname(self) -> str:
        return "dummy"

    async def _test_user_tool(self, computer: ComputerInterface, task: SWELancerTask) -> None:
        """
        Executes the user tool and writes the output to a log file on the computer, and downloads the log file to the local machine
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            runs_dir=task.runs_dir,
            run_id=task.run_id,
        )
        user_tool_log_path = Path(self.log_dir) / "user_tool.log"
        ctx_logger.info(
            f"Testing user tool and writing logs to {user_tool_log_path}",
            destinations=["run"],
        )

        await computer.check_shell_command(
            "unzip -q -P 'secretpasswordhere' /app/tests.zip -d /app/tests"
        )

        res = await computer.send_shell_command("bash -c -i 'user-tool'")
        output = f"bash -c -i 'user-tool'\nExit code: {res.exit_code}\nOutput: {res.output.decode('utf-8')}"
        await computer.check_shell_command(f"echo '{output}' >> {user_tool_log_path}")

        res = await computer.send_shell_command("ls -al /app/expensify/user_tool")
        output = f"ls -al /app/expensify/user_tool\nExit code: {res.exit_code}\nOutput: {res.output.decode('utf-8')}"
        await computer.check_shell_command(f"echo '{output}' >> {user_tool_log_path}")

        log_bytes = await computer.download(str(user_tool_log_path))
        with open(Path(task.run_dir) / "user_tool.log", "wb") as f:
            f.write(log_bytes)

    async def _apply_gold_solution(self, computer: ComputerInterface, task: SWELancerTask) -> None:
        """
        Applies the gold solution to the task.
        """
        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            runs_dir=task.runs_dir,
            run_id=task.run_id,
        )

        if task.variant == "ic_swe":
            ctx_logger.info("Applying gold patch", destinations=["run"])

            # Unzip the tests dir containing the patch
            await computer.check_shell_command(
                "unzip -q -P 'secretpasswordhere' /app/tests.zip -d /app/tests"
            )

            await computer.check_shell_command(
                f"git apply -R /app/tests/issues/{task.issue_id}/bug_reintroduce.patch"
            )
        elif task.variant == "swe_manager":
            assert task.manager_data is not None
            correct_option: int = task.manager_data["game"]["correct_proposal"]["id"]
            gold_manager_decision = {
                "proposal_id": correct_option,
            }
            await computer.upload(
                json.dumps(gold_manager_decision).encode("utf-8"),
                "/app/expensify/manager_decisions.json",
            )
        else:
            raise NotImplementedError(
                f"Gold solution not implemented for task variant {task.variant}"
            )

    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        assert isinstance(task, SWELancerTask), (
            f"DummySolver only supports SWELancerTasks, got {type(task)}"
        )

        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            runs_dir=task.runs_dir,
            run_id=task.run_id,
        )

        try:
            ctx_logger.info("Starting computer...", destinations=["run"])
            # only timeout the startup phase; everything else runs un-timed
            async with run_with_startup_timeout(
                self.computer_runtime, task, STARTUP_TIMEOUT
            ) as computer:
                await computer.check_shell_command(f"mkdir -p {self.log_dir}")

                # 1. Run dummy checks
                if self.test_user_tool:
                    await self._test_user_tool(computer, task)

                # 2. Optionally apply the gold solution
                if self.apply_gold_solution:
                    await self._apply_gold_solution(computer, task)

                # 3. Grade and yield the final result
                try:
                    grade = await task.grade(computer, self.runtime_config)
                except Exception as e:
                    raise RolloutSystemError(f"Error during grading: {e}") from e

                yield FinalResult(grade=grade)
        except asyncio.TimeoutError as e:
            ctx_logger.exception("Computer startup timed out", destinations=["run"])
            raise RolloutSystemError("Computer startup timed out") from e
        except Exception as e:
            if isinstance(e, RolloutSystemError):
                ctx_logger.exception(
                    f"RolloutSystemError in {task.run_id}: {e}", destinations=["run"]
                )
                raise
            ctx_logger.exception(
                f"Unexpected error in task {task.run_id}: {e}", destinations=["run"]
            )
            raise RolloutSystemError(f"Unexpected error: {e}") from e
