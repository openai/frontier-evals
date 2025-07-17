import asyncio
import io
import json
import os
import tarfile
import time
from typing import Optional

import blobfile as bf
from alcatraz.clusters.local import LocalConfig, VolumesConfig
from nanoeval.eval import RetryableSystemError
from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.agents.registry import Agent
from paperbench.agents.utils import AgentDirConfig
from paperbench.constants import SUBMISSION_DIR
from paperbench.infra.alcatraz import (
    compute_aisi_basic_agent_runtime,
    count_aisi_basic_agent_messages,
    upload_sources,
)
from paperbench.nano.utils import check_submission_exists
from paperbench.paper_registry import Paper
from paperbench.utils import purple
from pydantic import BaseModel
from structlog.stdlib import BoundLogger


class AgentOutput(BaseModel):
    run_id: str
    time_start: float
    time_end: float
    error_msg: Optional[str] = None
    runtime_in_seconds: float
    status_exists: bool
    skipped_rollout: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "AgentOutput":
        try:
            return AgentOutput(
                run_id=data["run_id"],
                time_start=data["time_start"],
                time_end=data["time_end"],
                error_msg=data.get("error_msg"),
                runtime_in_seconds=data["runtime_in_seconds"],
                status_exists=data["status_exists"],
                skipped_rollout=data["skipped_rollout"],
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in agent output: {e}")

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "error_msg": self.error_msg,
            "runtime_in_seconds": self.runtime_in_seconds,
            "status_exists": self.status_exists,
            "skipped_rollout": self.skipped_rollout,
        }


async def run_agent_in_computer(
    computer: ComputerInterface,
    task: "PBTask",  # type: ignore
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

    error: Optional[Exception] = None

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
    alcatraz_config: LocalConfig, agent: Agent, is_nvidia_gpu_env: bool
) -> LocalConfig:
    """
    Prepares computer for a run by processing its Cluster Config
    """
    if agent.mount_docker_socket:
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


async def start_periodic_heavy_log_upload(
    computer: ComputerInterface,
    agent_dir_config: AgentDirConfig,
    agent_start_time: int,
    run_dir: str,
    upload_interval_messages: int | None,
    upload_interval_seconds: int | None,
    logger: BoundLogger,
) -> asyncio.Task:
    """
    Uploads heavy logs periodically. Returns the periodic upload task
    """

    async def upload_task():
        try:
            last_message_upload = 0
            last_time_upload = 0
            while True:
                # Every 30s, compute the number of messages and productive runtime (in parallel)
                await asyncio.sleep(30)
                num_messages, (runtime, productive_runtime, retry_time) = await asyncio.gather(
                    count_aisi_basic_agent_messages(computer),
                    compute_aisi_basic_agent_runtime(computer),
                )
                # If at step or time interval, upload heavy logs
                over_step_interval = (
                    upload_interval_messages is not None
                    and num_messages - last_message_upload > upload_interval_messages
                )
                over_time_interval = (
                    upload_interval_seconds is not None
                    and productive_runtime is not None
                    and productive_runtime - last_time_upload > upload_interval_seconds
                )
                if over_step_interval or over_time_interval:
                    if over_step_interval:
                        last_message_upload = (
                            num_messages // upload_interval_messages
                        ) * upload_interval_messages
                    if over_time_interval:
                        last_time_upload = (
                            productive_runtime // upload_interval_seconds
                        ) * upload_interval_seconds
                    await upload_heavy_logs(
                        computer=computer,
                        agent_start_time=agent_start_time,
                        agent_dir_config=agent_dir_config,
                        run_dir=run_dir,
                        logger=logger,
                        runtime=runtime,
                        productive_runtime=productive_runtime,
                        retry_time=retry_time,
                        num_messages=num_messages,
                    )
                    logger.info(f"Uploaded heavy logs for run {run_dir}")
        except Exception as e:
            logger.exception(f"Exception in upload_task: {e}")
            raise

    return asyncio.create_task(upload_task())


async def start_periodic_light_log_upload(
    agent_start_time: int,
    run_dir: str,
    logger: BoundLogger,
) -> asyncio.Task:
    """
    Uploads light logs periodically. Returns the periodic upload task
    """

    async def upload_task():
        try:
            while True:
                await asyncio.sleep(300)
                await upload_light_logs(
                    agent_start_time=agent_start_time,
                    run_dir=run_dir,
                    logger=logger,
                )
        except Exception as e:
            logger.exception(f"Exception in upload_task: {e}")
            raise

    return asyncio.create_task(upload_task())


async def upload_heavy_logs(
    computer: ComputerInterface,
    agent_start_time: int,
    agent_dir_config: AgentDirConfig,
    run_dir: str,
    logger: BoundLogger,
    runtime: float | None = None,
    productive_runtime: float | None = None,
    retry_time: float | None = None,
    num_messages: int | None = None,
):
    timestamp = f"{time.strftime('%Y-%m-%dT%H-%M-%S-%Z', time.gmtime())}"

    await upload_sources(
        computer=computer,
        sources=agent_dir_config.directories_to_save,
        run_dir=run_dir,
        timestamp=timestamp,
        logger=logger,
    )
    if runtime is None or productive_runtime is None or retry_time is None:
        runtime, productive_runtime, retry_time = await compute_aisi_basic_agent_runtime(computer)
    if num_messages is None:
        num_messages = await count_aisi_basic_agent_messages(computer)
    await upload_log_info(
        start_time=agent_start_time,
        run_dir=run_dir,
        timestamp=timestamp,
        num_messages=num_messages,
        runtime=runtime,
        productive_runtime=productive_runtime,
        retry_time=retry_time,
    )
    logger.info(f"Uploaded periodic heavy logs for run {run_dir}")


async def upload_light_logs(
    agent_start_time: int,
    run_dir: str,
    logger: BoundLogger,
):
    await upload_status(
        start_time=agent_start_time,
        run_dir=run_dir,
        status="running",
    )
    logger.info(f"Uploaded periodic light logs for run {run_dir}")


async def upload_light_and_heavy_logs(
    computer: ComputerInterface,
    agent_start_time: int,
    agent_dir_config: AgentDirConfig,
    run_dir: str,
    logger: BoundLogger,
):
    initial_upload_complete = asyncio.Event()

    async def upload_task():
        try:
            await upload_light_logs(
                agent_start_time=agent_start_time,
                run_dir=run_dir,
                logger=logger,
            )
            await upload_heavy_logs(
                computer=computer,
                agent_start_time=agent_start_time,
                agent_dir_config=agent_dir_config,
                run_dir=run_dir,
                logger=logger,
            )
            logger.info(f"Uploaded light and heavy logs for run {run_dir}")
        except Exception as e:
            logger.exception(f"Exception in upload_task: {e}")
            raise
        finally:
            initial_upload_complete.set()

    return asyncio.create_task(upload_task()), initial_upload_complete


async def upload_status(
    start_time: int,
    run_dir: str,
    status: str,
    end_time: int | None = None,
):
    status_obj = {
        "status": status,
        "created_at": start_time,
        "agent_finished_at": end_time,
        "last_updated": int(time.time()),
    }
    bf.write_bytes(
        bf.join(run_dir, "status.json"),
        json.dumps(status_obj, indent=4).encode("utf-8"),
    )


async def upload_log_info(
    start_time: int,
    run_dir: str,
    timestamp: str,
    num_messages: int,
    runtime: str,
    productive_runtime: str,
    retry_time: str,
):
    log_info = {
        "created_at": start_time,
        "num_messages": num_messages,
        "runtime": runtime,
        "productive_runtime": productive_runtime,
        "retry_time": retry_time,
    }
    bf.write_bytes(
        bf.join(run_dir, "submissions", timestamp, "log.json"),
        json.dumps(log_info, indent=4).encode("utf-8"),
    )


async def execute_agent_in_computer(
    computer: ComputerInterface,
    agent: Agent,
    agent_dir_config: AgentDirConfig,
    run_dir: str,
    timeout: int,
    logger: BoundLogger,
    upload_interval_seconds: int | None = 1800,
    upload_interval_messages: int | None = None,
):
    """Initiates the agent via its start script inside the cluster container."""
    cmd_str = build_agent_command(agent, agent_dir_config.agent_dir)
    logger.info(f"Running agent with command: {cmd_str}")

    heavy_periodic_upload_task = None
    light_periodic_upload_task = None
    initial_upload_task, initial_upload_complete = None, None
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
            for task in [
                initial_upload_task,
                heavy_periodic_upload_task,
                light_periodic_upload_task,
            ]:
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await task
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
):
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
):
    """
    Extracts a directory from a computer to a specified local directory.

    Args:
        computer: The computer instance.
        path_on_cluster: The path to the directory on the computer.
        save_dir: The local directory where the file or directory will be saved.
    """
    res = await computer.send_shell_command(f"ls -l {path_on_cluster}")
    if res.exit_code != 0:
        logger.exception(f"Directory {path_on_cluster} does not exist\n{res.output}")
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
