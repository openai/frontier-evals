import ast
import shlex
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from enum import StrEnum, auto
from functools import cached_property
from pathlib import Path
from typing import Any, AsyncContextManager, AsyncGenerator, Literal, Sequence

import structlog.stdlib
from IPython.core.inputtransformer2 import TransformerManager
from pydantic import BaseModel, Field, field_validator
from typing_extensions import deprecated, final

import chz

logger = structlog.stdlib.get_logger(component=__name__)


class JupyterExecutionResult(BaseModel):
    status: Literal[
        "running",
        "success",
        "cancelled",
        "failed_with_in_kernel_exception",
        "failed_with_system_exception",
        "failed_with_kernel_death",
        "failed_with_response_too_large",
    ]
    output: str
    final_expression_output: str | None
    in_kernel_exception: Any | None = None
    system_exception: Any | None = None

    @cached_property
    def parsed_final_expression_output(self) -> Any:
        if not self.final_expression_output:
            return None
        return ast.literal_eval(self.final_expression_output)


class ExecutionResult(BaseModel):
    output: bytes
    exit_code: int

    @property
    def unicode_output_best_effort(self) -> str:
        """
        Not all command outputs are valid Unicode, because the unix command line is binary. However, it is very common
        to want to treat the output as Unicode. This is a utility function that decodes the output as best as possible
        and replaces any invalid characters with the Unicode replacement character.
        """
        return self.output.decode("utf-8", errors="replace")


class NetworkMode(StrEnum):
    """
    The mode specifying the network configuration for the container.
    """

    # No network access
    NONE = auto()
    # Webcache access, only GET requests allowed
    WEBCACHE_GET_ONLY = auto()
    # Unproxied access to the internet
    UNPROXIED = auto()
    # full GET access + whitelisted POST access to LLM provider APIs
    API_ACCESS = auto()


class ComputerInterface(ABC):
    """
    Represents a rollout's connection to a computer. This represents the "runtime state"
    of a Task, and is an abstraction that lets you write assertions and tests against a computer without regard to the
    backend.

    Note that a ComputerInterface does not "own" the underlying resource, as in when a ComputerInterface for a cluster
    resource gets deleted, the cluster itself does not necessarily stop. It is the responsibility of the code that
    created the underlying resource to manage its lifecycle and dispose of it.
    """

    @abstractmethod
    async def disable_internet(self) -> None:
        """
        Disables outbound internet access for the container. Any other containers on the local network
        should still be accessible.
        """
        pass

    @abstractmethod
    async def upload(self, file: bytes, destination: str) -> None:
        pass

    @abstractmethod
    async def download(self, file: str) -> bytes:
        pass

    @abstractmethod
    async def send_shell_command(self, cmd: str, *, idempotent: bool = False) -> ExecutionResult:
        """
        Executes the shell command in a new bash shell. Every command should be unique in terms of environment.

        If ``idempotent`` is ``True`` the call may transparently retry on failure. Only pass ``True`` if the command
        can be safely repeated without side effects.
        """

    async def check_shell_command(self, cmd: str, *, idempotent: bool = False) -> ExecutionResult:
        res = await self.send_shell_command(cmd, idempotent=idempotent)
        assert res.exit_code == 0, (
            f"Command failed with {res.exit_code=}\n\n{cmd=}\n\n{res.output.decode(errors='ignore')}"
        )
        return res

    @abstractmethod
    @deprecated("Only CTFs should use this function")
    async def fetch_container_names(self) -> list[str]:
        """
        TODO(kevinliu): This function really shouldn't need to be here; it's a layering
        violation. We should not require eval tasks to manage individual containers in code.

        Remove once CTFs no longer use this in their prompts.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        # Stops the computer. If used, the computer will no longer be accessible after this and the behavior
        # of all other functions is undefined.
        pass


class JupyterComputerInterface(ComputerInterface):
    """
    A computer upon which a jupyter kernel has been started. It can accept commands.
    """

    @abstractmethod
    async def execute(self, code: str, timeout: int = 120) -> JupyterExecutionResult:
        """
        Execute the given Python code and return the result.
        """
        pass

    async def check_execute(self, code: str) -> JupyterExecutionResult:
        """
        Execute the given Python code and return the result.
        """
        res = await self.execute(code)
        assert res.status == "success", (res, code)
        return res


# TODO(kevinliu) gpus?
class ContainerResources(BaseModel):
    """
    Resources requested for a container. The defaults are set pretty low to encourage conservation
    and make downstream evals more explicit about their resource requirements.
    """

    # number of cpus (default)
    cpu: float = 0.2
    # amount of memory in MB
    memory: int = 1024


class VolumeMount(BaseModel):
    """
    Mounts files from the host onto the container.
    """

    host_path: str
    container_path: str


@chz.chz
class RuntimeConfig:
    """Configuration for runtime-specific resources."""


class ComputerConfiguration(BaseModel):
    """
    Very roughly, this class represents how to construct a Docker Compose-style workload for a
    model rollout. It is written mostly generically, such that people can write their own
    `ComputerRuntime` that provides a `ComputerInterface` for downstream code and solvers.
    """

    # cwd: the working directory for the task. Typically, a solver can use this
    # to cd into the directory where the data for the task is located.
    cwd: str = "/root"

    # Custom docker image to start execution in. Must have Python installed.
    # If not specified, depends on the execution environment - so I highly recommend
    # specifying the image if your task has any dependencies at all.
    docker_image: str | None = None

    # Side containers to run alongside the main container.
    side_images: list[str] = []
    docker_compose_yaml: str | None = None

    # Resource requirements for the main container.
    azure_vm_sku: str | None = None
    disk_mount_path: str | None = None

    azure_files_config: dict[str, str] | None = None
    azure_container_config: dict[str, str] | None = None
    volumes_config: dict[str, Any] = Field(
        default={},
        deprecated=True,
        description="For backwards compatibility with datasets that explicitly specify volumes. Note that volumes_config now does absolutely nothing, you should use volume_mounts instead.",
    )
    shm_size: str | None = None
    # TODO(kevinliu) remove this
    mem_limit: str | None = None
    timeout: int | None = None
    alcatraz_limits: dict[str, Any] | None = None

    # Code to run the jupyter kernel in conda
    jupyter_setup: Sequence[str] | None = None

    # Allow setting custom environment variables
    environment: dict[str, Any] = {}

    resources: ContainerResources = ContainerResources()
    limits: ContainerResources = ContainerResources()
    volume_mounts: list[VolumeMount] = Field(
        default_factory=list,
        description="Mounts files from the host onto the container. Volume mounts are read only.",
    )
    network_mode: NetworkMode = Field(
        default=NetworkMode.NONE,
        description="The mode specifying the network configuration for the container. If set to NONE, container will not be attached to any Internet route. It will still have access to side containers on the internal network.  Due to runtime restrictions, this is DIFFERENT from computer.disable_internet() which also disconnects the container from the internal network.",
    )
    num_gpus: int = 0

    def validate_runtime_config(self, runtime_config: RuntimeConfig) -> RuntimeConfig:
        """Validate runtime-specific resources."""
        assert type(runtime_config) is RuntimeConfig, (
            "Task must override validate_resources to use runtime-specific resources"
        )
        return runtime_config

    async def setup(self, computer: ComputerInterface, runtime_config: RuntimeConfig) -> None:
        """
        This setup function enforces and verifies the ComputerConfiguration invariants on the computer. All tasks
        should call this function in their setup function.
        """

        logger.info("Beginning setup")

        # Sanity checks
        if self.num_gpus > 0:
            num_gpus = await computer.check_shell_command("nvidia-smi -L | wc -l", idempotent=True)
            num_gpus = int(num_gpus.output.decode())
            assert self.num_gpus == num_gpus, f"{self.num_gpus=} != {num_gpus=}"

        if isinstance(computer, JupyterComputerInterface):
            await computer.check_execute(f"%cd {self.cwd}")
            res = await computer.check_execute("import os; os.getcwd()")
            assert res.parsed_final_expression_output == self.cwd, (
                f"Jupyter execution environment didn't properly set up cwd: {res}"
            )

        # TODO(kevinliu) move this out and validate on Alcatraz
        # cd to the proper root folder. Try to cover all bases with .bashrc, .profile, and .bash_profile;
        # note that the existence of .bash_profile overrides .profile so be careful to gate on existence.
        await computer.send_shell_command(
            f"""
            echo cd {shlex.quote(self.cwd)} >> ~/.bashrc
            [ -f ~/.profile ] && echo cd {shlex.quote(self.cwd)} >> ~/.profile
            [ -f ~/.bash_profile ] && echo cd {shlex.quote(self.cwd)} >> ~/.bash_profile
            """,
            idempotent=True,
        )
        res = await computer.check_shell_command("bash -lc 'pwd'", idempotent=True)
        if res.unicode_output_best_effort.strip() != self.cwd:
            raise RuntimeError(f"bash -lc didn't set up cwd properly: {res}")

        # Check the volumes are setup properly
        for volume_mount in self.volume_mounts:
            await computer.check_shell_command(
                f"test -d {shlex.quote(volume_mount.container_path)} || test -f {shlex.quote(volume_mount.container_path)} || test -S {shlex.quote(volume_mount.container_path)}",
                idempotent=True,
            )

    @property
    def allow_internet(self) -> bool:
        return self.network_mode != NetworkMode.NONE

    # Validators
    @field_validator("cwd")
    @classmethod
    def _validate_cwd(cls, v: str) -> str:
        assert Path(v).is_absolute(), "cwd must be an absolute path."
        return v


@chz.chz
class ComputerRuntime(ABC):
    """
    Represents a runtime to run a ComputerTask. You can use this class to build simple portable Solvers that
    are environment-agnostic.

    Many realistic solvers may be implementation-specific and will not use this interface directly, but may use
    a concrete downstream implementation.

    Generally, you want to call `runtime.run` to start a computer and then use the `ComputerInterface`
    to interact with the computer.
    """

    runtime_config: RuntimeConfig = chz.field(default_factory=RuntimeConfig)

    @final
    async def setup_computer(
        self, task: ComputerConfiguration, computer: ComputerInterface
    ) -> None:
        """
        General setup flow:

        - runtime.run -> creates computer
        - runtime.setup_computer
            - runtime.runtime_presetup -> sets up the computer with any runtime-specific parameters
            - task.setup -> sets up the task with any task-specific parameters
        """

        # TODO(kevinliu): In the future, due to different model harnesses, we may want to have 4 stages instead of 3:
        # 1. Create computer
        # 2. Runtime setup
        # 3. Model specific setup (e.g., tools)
        # 4. Task setup
        # For now, runtime setup also includes model-specific setup.
        runtime_config = task.validate_runtime_config(self.runtime_config)
        await self._do_runtime_setup(task, computer)
        await task.setup(computer, runtime_config=runtime_config)

    @final
    @asynccontextmanager
    async def run(self, task: ComputerConfiguration) -> AsyncGenerator[ComputerInterface, None]:
        """
        Provides an interface to a computer that is ready to run agent code.

        Internally, this function creates a new computer via the appropriate runtime backend, runs runtime setup, and
        runs task setup.
        """
        async with self._start_computer(task) as computer:
            await self.setup_computer(task, computer)
            yield computer

    # Override the functions below to implement your own runtime
    @abstractmethod
    async def _do_runtime_setup(
        self, task: ComputerConfiguration, computer: ComputerInterface
    ) -> None:
        """
        This function is called before the task.setup function. It is used to set up any runtime-specific
        settings on the computer - e.g., overriding proxies, installing packages, etc.
        """
        pass

    @abstractmethod
    def _start_computer(
        self, task: ComputerConfiguration
    ) -> AsyncContextManager[ComputerInterface]:
        """
        Starts the computer and returns a context manager that owns the underlying container.
        The context manager will stop the underlying resources when the context manager exits.
        """
        pass


def valid_ipython_code(code: str) -> bool:
    # Parse the code using ast to check for syntax errors.
    # NOTE: technically, might be unsound if the ACE environment
    # has a newer Python than the one we're using here. But atm
    # (4/3/24), this is not the case.
    tm = TransformerManager()  # type: ignore
    transformed_code = tm.transform_cell(code)
    try:
        ast.parse(transformed_code)
        return True
    except SyntaxError:
        return False
    except Exception as e:
        raise ValueError("Error parsing code") from e
