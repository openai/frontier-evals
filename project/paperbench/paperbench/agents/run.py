from __future__ import annotations

import asyncio
import io
import json
import os
import tarfile
import time

import blobfile as bf
from alcatraz.clusters.local import ClusterConfig, VolumesConfig
from nanoeval.eval import RetryableSystemError
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from structlog.stdlib import BoundLogger

from paperbench.agents.registry import Agent
from paperbench.agents.upload import (
    start_periodic_heavy_log_upload,
    start_periodic_light_log_upload,
    upload_light_and_heavy_logs,
)
from paperbench.agents.utils import AgentDirConfig, AgentOutput
from paperbench.nano.task import PBTask
from paperbench.paper_registry import Paper
from paperbench.utils import purple


async def run_agent_in_computer(
    computer: ComputerInterface,
    task: PBTask,
    paper: Paper,
    agent: Agent,
    run_dir: str,
    logger: BoundLogger,
    agent_dir_config: AgentDirConfig,
    timeout: int,
    upload_interval_seconds: int = 1800,
    upload_interval_messages: int | None = None,
    save_cluster_output_to_host: bool = True,
) -> AgentOutput:
    start = time.time()
    logger.info(purple(f"Run for `{agent.id}` agent attempting `{paper.id}`: {run_dir}"))

    error: Exception | None = None

    try:
        await execute_agent_in_computer(
            computer=computer,
            agent=agent,
            agent_dir_config=agent_dir_config,
            run_dir=run_dir,
            timeout=timeout,
            logger=logger,
            upload_interval_seconds=upload_interval_seconds,
            upload_interval_messages=upload_interval_messages,
        )
        logger.info("Done running agent in cluster")

        if save_cluster_output_to_host:
            await save_computer_output(
                computer, run_dir, agent_dir_config.directories_to_save, logger=logger
            )
    except Exception as e:
        error = e
        logger.exception(f"Run failed with error:\n{str(error)}")
    finally:
        # re-raise retryable errors
        if isinstance(error, RetryableSystemError):
            raise error

        end = time.time()
        logger.info(f"Run completed in {end - start:.2f} seconds.")

        status_exists = bf.exists(bf.join(run_dir, "status.json"))

        agent_output = AgentOutput(
            run_id=task.run_id,
            time_start=start,
            time_end=end,
            runtime_in_seconds=end - start,
            error_msg=str(error) if error else None,
            status_exists=status_exists,
        )

        with bf.BlobFile(bf.join(run_dir, "metadata.json"), "w") as f:
            json.dump(agent_output.model_dump(mode="json"), f, indent=4)

    return agent_output


def prepare_computer(
    alcatraz_config: ClusterConfig,
    agent: Agent,
    is_nvidia_gpu_env: bool = False,
) -> ClusterConfig:
    """
    Prepares computer for a run by processing its Cluster Config
    """
    if agent.mount_docker_socket:
        assert hasattr(alcatraz_config, "volumes_config"), (
            "ClusterConfig instance must support a `volumes_config` attribute for `mount_docker_socket` agents"
        )
        volumes = (
            alcatraz_config.volumes_config if alcatraz_config.volumes_config else VolumesConfig()
        )
        # Alcatraz uses Named Volumes syntax
        # Practically though the name is ignored, and anonymous bind mounts are used
        volumes["dockersocket"] = {
            "bind_source": "/var/run/docker.sock",
            "bind_dest": "/var/run/docker.sock",
            "mode": "rw",
        }
        alcatraz_config.volumes_config = volumes

    alcatraz_config.is_nvidia_gpu_env = is_nvidia_gpu_env
    if agent.privileged:
        raise ValueError("Cannot set privileged=True for LocalConfig")
    alcatraz_config.pull_from_registry = False
    alcatraz_config.environment.update(agent.env_vars)

    return alcatraz_config


async def execute_agent_in_computer(
    computer: ComputerInterface,
    agent: Agent,
    agent_dir_config: AgentDirConfig,
    run_dir: str,
    timeout: int,
    logger: BoundLogger,
    upload_interval_seconds: int | None = 1800,
    upload_interval_messages: int | None = None,
) -> None:
    """Initiates the agent via its start script inside the cluster container."""
    cmd_str = build_agent_command(agent, agent_dir_config.agent_dir)
    logger.info(f"Running agent with command: {cmd_str}")

    heavy_periodic_upload_task: asyncio.Task[None] | None = None
    light_periodic_upload_task: asyncio.Task[None] | None = None
    initial_upload_task: asyncio.Task[None] | None = None
    initial_upload_complete: asyncio.Event | None = None
    async with asyncio.timeout(timeout):
        try:
            agent_start_time = int(time.time())
            agent_task = asyncio.create_task(computer.send_shell_command(cmd_str))

            initial_upload_task, initial_upload_complete = await upload_light_and_heavy_logs(
                computer=computer,
                agent_start_time=agent_start_time,
                agent_dir_config=agent_dir_config,
                run_dir=run_dir,
                logger=logger,
            )
            light_periodic_upload_task = await start_periodic_light_log_upload(
                agent_start_time=agent_start_time,
                run_dir=run_dir,
                logger=logger,
            )
            heavy_periodic_upload_task = await start_periodic_heavy_log_upload(
                computer=computer,
                agent_dir_config=agent_dir_config,
                agent_start_time=agent_start_time,
                run_dir=run_dir,
                upload_interval_messages=upload_interval_messages,
                upload_interval_seconds=upload_interval_seconds,
                logger=logger,
            )

            while not agent_task.done():
                logger.info("Waiting for agent to finish...")
                for task in [
                    initial_upload_task,
                    heavy_periodic_upload_task,
                    light_periodic_upload_task,
                ]:
                    if task and task.done():
                        exc = task.exception()
                        if exc:
                            raise exc
                await asyncio.sleep(60)

            output = await agent_task
            decoded_result = output.output.decode("utf-8")
            if output.exit_code != 0:
                raise Exception(
                    f"Agent exited with code: {output.exit_code}, output: \n{decoded_result}"
                )
            logger.info(f"Agent done! exit_code: {output.exit_code}")
        except asyncio.TimeoutError as e:
            logger.warning(
                f"Agent run timed out after {time.time() - agent_start_time} second (timeout: {timeout}): {e}"
            )
        except asyncio.CancelledError as e:
            logger.warning(
                f"Agent run cancelled after {time.time() - agent_start_time} seconds: {e}"
            )
        finally:
            if not agent_task.done():
                agent_task.cancel()
            if initial_upload_complete is not None:
                await initial_upload_complete.wait()
            for upload_task in [
                initial_upload_task,
                heavy_periodic_upload_task,
                light_periodic_upload_task,
            ]:
                if upload_task is not None and not upload_task.done():
                    upload_task.cancel()
                    try:
                        await upload_task
                    except (asyncio.CancelledError, Exception) as e:
                        if not isinstance(e, asyncio.CancelledError):
                            logger.exception(f"Task failed with error: {e}")


def build_agent_command(agent: Agent, agent_dir: str) -> str:
    """Builds the command to run the agent."""

    cmd = ["bash", f"{agent_dir}/start.sh"]

    if agent.kwargs_type == "argparse":
        for key, value in agent.kwargs.items():
            cmd += [f"--{key}", str(value)]

    if agent.kwargs_type == "omegaconf":
        cmd += [f"{key}={value}" for key, value in agent.kwargs.items()]

    return " ".join(cmd)


async def save_computer_output(
    computer: ComputerInterface,
    save_dir: str,
    directories_to_save: list[str],
    logger: BoundLogger,
) -> None:
    """
    Extracts the submission, logs, and code directories from the cluster container to the host
    machine and saves them to the specified directory.

    Args:
        computer: The computer instance.
        save_dir: The directory where the output folder/file will be saved.
        directories_to_save: The directories to save from the container.
    """
    for dir_to_save in directories_to_save:
        await extract_dir_from_computer(computer, dir_to_save, save_dir, logger=logger)


async def extract_dir_from_computer(
    computer: ComputerInterface,
    path_on_cluster: str,
    save_dir: str,
    logger: BoundLogger,
) -> None:
    """
    Extracts a directory from a computer to a specified local directory.

    Args:
        computer: The computer instance.
        path_on_cluster: The path to the directory on the computer.
        save_dir: The local directory where the file or directory will be saved.
    """
    res = await computer.send_shell_command(f"ls -l {path_on_cluster}")
    if res.exit_code != 0:
        logger.exception(
            f"Directory {path_on_cluster} does not exist\n{res.output.decode('utf-8')}"
        )
        return

    target_dir_name = os.path.basename(path_on_cluster.rstrip("/"))
    parent_path = os.path.dirname(path_on_cluster)
    target_dir_tar_path = f"/tmp/{target_dir_name}.tar"
    # make the target dir into a tar file, so we can use computer.download
    res = await computer.send_shell_command(
        f"tar -cf {target_dir_tar_path} -C {parent_path} {target_dir_name}"
    )

    tar_bytes = await computer.download(target_dir_tar_path)

    tar_stream = io.BytesIO(tar_bytes)

    try:
        with tarfile.open(fileobj=tar_stream, mode="r") as tar:
            tar.extractall(path=save_dir)
        logger.info(f"Extracted contents to: {save_dir}/")
    except tarfile.TarError as e:
        logger.exception(f"Error extracting tar file: {e}")
        return
    finally:
        # cleanup
        await computer.send_shell_command(f"rm {target_dir_tar_path}")
