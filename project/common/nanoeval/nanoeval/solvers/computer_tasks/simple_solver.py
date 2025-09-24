import asyncio
import sys
import traceback
from abc import abstractmethod
from typing import AsyncGenerator

import structlog.stdlib
from typing_extensions import override

import chz
from nanoeval.eval import RolloutSystemError
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    ComputerRuntime,
)
from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from nanoeval.solvers.computer_tasks.steps import (
    FinalResult,
    Step,
)
from nanoeval.solvers.computer_tasks.task import ComputerTask

logger = structlog.stdlib.get_logger(component=__name__)


@chz.chz
class ComputerRuntimeSolver(PythonCodingSolver):
    """
    A solver dependent on an environment-agnostic computer runtime.
    """

    runtime: ComputerRuntime


@chz.chz
class SimpleSolver(ComputerRuntimeSolver):
    timeout: int = 3600

    @abstractmethod
    async def solve(self, task: ComputerTask, computer: ComputerInterface) -> None:
        pass

    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        """
        Runs the solver, then yields a FinalResult.
        """
        try:
            async with asyncio.timeout(self.timeout):
                async with self.runtime.run(task) as computer:
                    # 1. Run the solver on the task
                    await self.solve(task, computer)

                    # 2. Grade and yield the final result
                    grade = await task.grade(computer, self.runtime.runtime_config)
                    yield FinalResult(grade=grade)
        except Exception as e:
            raise RolloutSystemError(f"Solver failed with error: {str(e)}") from e


@chz.chz
class DummySolver(SimpleSolver):
    name: str = "DummySolver"

    def shortname(self) -> str:
        return "dummy-solver"

    @override
    async def solve(self, task: ComputerTask, computer: ComputerInterface) -> None:
        # Dummy code execution
        res = await computer.check_shell_command("echo 'Hello, world!' > test.txt")
        assert res.exit_code == 0
        logger.info("Completed dummy code execution", res=res)


# A solver that, after task startup, sets up an interactive terminal from standard i/o.
# Running more than one concurrent task with this solver will cause issues, only use this in debug mode.
@chz.chz
class UserSolver(SimpleSolver):
    name: str = "UserSolver"

    def shortname(self) -> str:
        return "user-solver"

    async def _ainput(self, prompt: str) -> str:
        # helper to read a line of input from stdin in an async context
        sys.stdout.write(prompt)
        sys.stdout.flush()
        return await asyncio.to_thread(sys.stdin.readline)

    @override
    async def solve(self, task: ComputerTask, computer: ComputerInterface) -> None:
        print(
            "\nEntering interactive computer REPL.\n"
            "Commands will be run in a fresh shell each time; side effects like cd's won't persist.\n"
            "Type commands to run inside the container; press Ctrl-D or enter an empty line to finish.\n"
            "Your prompt is as follows:\n"
        )
        print("\n".join([str(message.get("content", "")) for message in task.prompt]))

        while True:
            try:
                cmd = await self._ainput("> ")
            except EOFError:
                break
            cmd = cmd.strip()
            if cmd in ["exit", "quit"]:
                break
            try:
                res = await computer.send_shell_command(cmd)
                print("Exit code:", res.exit_code)
                print(res.output.decode("utf-8", errors="replace"))
            except Exception as e:
                print(f"Error: {e}")
                traceback.print_exc()
