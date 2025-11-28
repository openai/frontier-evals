import os
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Literal, cast

import async_lru
import structlog.stdlib
from pydantic import BaseModel
from typing_extensions import override

import chz
from alcatraz.clusters.local import BaseAlcatrazCluster, ClusterConfig, LocalConfig
from alcatraz.utils.cmds import run_command_streaming
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerConfiguration,
    ComputerInterface,
    ComputerRuntime,
    ExecutionResult,
    JupyterComputerInterface,
    JupyterExecutionResult,
)
from nanoeval_alcatraz.task_to_alcatraz_config import (
    SUPPORTED_NETWORK_MODES,
    task_to_alcatraz_config,
)

logger = structlog.stdlib.get_logger(component=__name__)

ALCATRAZ_TIMEOUT = int(os.getenv("ALCATRAZ_TIMEOUT", 120))


class Python3ExceptionDict(BaseModel):
    """A pydantic model for serializing a Python 3.x exception.

    Attrs:
        name: The type of the exception, e.g. ValueError.
        traceback: The traceback. Every line ends with a \n.
        args: The args passed to the exception's ctor.
        notes: Future-proofing for PEP 678.
    """

    name: str
    traceback: list[str]
    args: tuple[Any, ...]
    notes: list[str]


class BaseAlcatrazComputerInterfaceNoJupyter(ComputerInterface, ABC):
    """
    Override this class to create a custom AlcatrazComputerInterface without Jupyter support.
    """

    @property
    @abstractmethod
    def _cluster(self) -> BaseAlcatrazCluster:
        pass

    async def disable_internet(self) -> None:
        res = await self.send_shell_command("hostname")
        cid_prefix = res.output.decode().strip()

        await self.cluster.add_container_network_block_via_ip_tables(cid_prefix)

        # Verify
        logger.info("Post-setup network access disabled")
        try:
            from alcatraz.utils.network import assert_internet_disabled  # type: ignore

            await assert_internet_disabled(self._cluster)
            logger.info("Verified network access successfully disabled")
        except ImportError:
            pass

    async def upload(self, file: bytes, destination: str) -> None:
        return await self._cluster.upload(file, destination)

    async def download(self, file: str) -> bytes:
        return await self._cluster.download(file)

    async def send_shell_command(self, cmd: str, *, idempotent: bool = False) -> ExecutionResult:
        res = await run_command_streaming(self._cluster, cmd)
        return ExecutionResult(output=res["result"], exit_code=res["exit_code"])

    async def fetch_container_names(self) -> list[str]:
        return await self._cluster.fetch_container_names()

    async def stop(self) -> None:
        await self._cluster._stop()


@chz.chz
class BaseAlcatrazComputerInterface(
    BaseAlcatrazComputerInterfaceNoJupyter, JupyterComputerInterface
):
    """
    Override this class to add Jupyter-specific functionality to the AlcatrazComputerInterface.
    """

    @override
    async def execute(self, code: str, timeout: int = ALCATRAZ_TIMEOUT) -> JupyterExecutionResult:
        await self._start_jupyter_kernel_once()

        messages = await self._cluster.send_kernel_command(code, timeout=timeout)

        # Parse the messages into a final execution result
        # TODO(kevinliu) - this may not be a perfect parsing, but it should only really be used for setup and grade so hopefully it's good enough
        output = ""
        status: Literal["success", "failed_with_in_kernel_exception"] = "success"
        exception = None
        final_expression_output = None
        for msg in messages:
            if msg["msg_type"] == "error":
                # Extract the exception
                exception = Python3ExceptionDict(
                    name=msg["content"]["ename"],
                    traceback=msg["content"]["traceback"],
                    args=(msg["content"]["evalue"],),
                    notes=[],
                )
                status = "failed_with_in_kernel_exception"
            elif msg["msg_type"] == "@timeout":
                status = "failed_with_in_kernel_exception"
            elif msg["msg_type"] == "execute_result":
                final_expression_output = msg["content"]["data"].get("text/plain", "")
            elif msg["msg_type"] == "stream":
                output += msg["content"].get("text", "")

        return JupyterExecutionResult(
            status=status,
            output=output,
            final_expression_output=final_expression_output,
            in_kernel_exception=cast(Any, exception),
            system_exception=None,
        )

    @async_lru.alru_cache(maxsize=1)
    async def _start_jupyter_kernel_once(self) -> None:
        if not await self._cluster.is_kernel_started():
            await self._cluster.create_kernel_on_machine()


@chz.chz
class AlcatrazComputerInterfaceNoJupyter(BaseAlcatrazComputerInterfaceNoJupyter):
    cluster_value: BaseAlcatrazCluster

    @property
    def _cluster(self) -> BaseAlcatrazCluster:
        return self.cluster_value


@chz.chz
class AlcatrazComputerInterface(AlcatrazComputerInterfaceNoJupyter, BaseAlcatrazComputerInterface):
    pass


@chz.chz
class AlcatrazComputerRuntimeNoJupyter(ComputerRuntime):
    env: ClusterConfig = chz.field(default_factory=LocalConfig)

    async def _do_runtime_setup(
        self, task: ComputerConfiguration, computer: ComputerInterface
    ) -> None:
        assert isinstance(computer, BaseAlcatrazComputerInterfaceNoJupyter)
        # Alcatraz must do this dynamically since it doesn't have the ability to remove
        # a container's internet access at creation time.
        # Other runtimes should do this by configuring the container object itself.
        if not task.allow_internet:
            logger.info("Disabling internet (since allow_internet is False)")
            await computer.disable_internet()

        if task.network_mode not in SUPPORTED_NETWORK_MODES:
            raise ValueError(
                f"The {task.network_mode} network mode is not supported on Alcatraz, and will never be. Please change it, or stop using Alcatraz."
            )

    @asynccontextmanager
    async def _start_computer(
        self, task: ComputerConfiguration
    ) -> AsyncGenerator[AlcatrazComputerInterfaceNoJupyter, None]:
        async with task_to_alcatraz_config(task, self.env).build() as _cluster:
            computer = AlcatrazComputerInterfaceNoJupyter(cluster_value=_cluster)
            yield computer


@chz.chz
class AlcatrazComputerRuntime(ComputerRuntime):
    env: ClusterConfig = chz.field(default_factory=LocalConfig)

    async def _do_runtime_setup(
        self, task: ComputerConfiguration, computer: ComputerInterface
    ) -> None:
        assert isinstance(computer, BaseAlcatrazComputerInterface)

        # Run a jupyter command; this will force Jupyter to start up and be installed.
        # This is an Alcatraz only feature. It must be done before Internet access is disabled, because
        # Alcatraz supports live-installing Jupyter.
        # TODO(kevinliu) we should catalog which evals rely on this and remove it.
        logger.info("Running initial Jupyter command to ensure Jupyter is installed")
        await computer.execute("print('hi')")
        logger.info("Jupyter working!")

        # Alcatraz must do this dynamically since it doesn't have the ability to remove
        # a container's internet access at creation time.
        # Other runtimes should do this by configuring the container object itself.
        if not task.allow_internet:
            logger.info("Disabling internet (since allow_internet is False)")
            await computer.disable_internet()

        if task.network_mode not in SUPPORTED_NETWORK_MODES:
            raise ValueError(
                f"The {task.network_mode} network mode is not supported on Alcatraz, and will never be. Please change it, or stop using Alcatraz."
            )

    @asynccontextmanager
    async def _start_computer(
        self, task: ComputerConfiguration
    ) -> AsyncGenerator[AlcatrazComputerInterface, None]:
        async with task_to_alcatraz_config(task, self.env).build() as _cluster:
            computer = AlcatrazComputerInterface(cluster_value=_cluster)
            yield computer


