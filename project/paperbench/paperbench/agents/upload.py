from __future__ import annotations

import asyncio
import json
import time

import blobfile as bf
import structlog.stdlib

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.agents.utils import AgentDirConfig
from paperbench.infra.alcatraz import (
    compute_aisi_basic_agent_runtime,
    count_aisi_basic_agent_messages,
    upload_sources,
)

logger = structlog.stdlib.get_logger(component=__name__)


async def start_periodic_heavy_log_upload(
    computer: ComputerInterface,
    agent_dir_config: AgentDirConfig,
    agent_start_time: int,
    run_dir: str,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
    upload_interval_messages: int | None,
    upload_interval_seconds: int | None,
) -> asyncio.Task[None]:
    """
    Uploads heavy logs periodically. Returns the periodic upload task
    """

    async def upload_task() -> None:
        ctx_logger = logger.bind(
            run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
        )
        try:
            last_message_upload = 0
            last_time_upload: float = 0
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
                    if over_step_interval and upload_interval_messages:
                        last_message_upload = (
                            num_messages // upload_interval_messages
                        ) * upload_interval_messages
                    if over_time_interval and upload_interval_seconds and productive_runtime:
                        last_time_upload = (
                            productive_runtime // upload_interval_seconds
                        ) * upload_interval_seconds
                    await upload_heavy_logs(
                        computer=computer,
                        agent_start_time=agent_start_time,
                        agent_dir_config=agent_dir_config,
                        run_dir=run_dir,
                        run_group_id=run_group_id,
                        runs_dir=runs_dir,
                        run_id=run_id,
                        runtime=runtime,
                        productive_runtime=productive_runtime,
                        retry_time=retry_time,
                        num_messages=num_messages,
                    )
                    ctx_logger.info(f"Uploaded heavy logs for run {run_dir}")
        except Exception as e:
            ctx_logger.exception(f"Exception in upload_task: {e}")
            raise

    return asyncio.create_task(upload_task())


async def start_periodic_light_log_upload(
    agent_start_time: int,
    run_dir: str,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
) -> asyncio.Task[None]:
    """
    Uploads light logs periodically. Returns the periodic upload task
    """

    async def upload_task() -> None:
        ctx_logger = logger.bind(
            run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
        )
        try:
            while True:
                await asyncio.sleep(300)
                await upload_light_logs(
                    agent_start_time=agent_start_time,
                    run_dir=run_dir,
                    run_group_id=run_group_id,
                    runs_dir=runs_dir,
                    run_id=run_id,
                )
        except Exception as e:
            ctx_logger.exception(f"Exception in upload_task: {e}")
            raise

    return asyncio.create_task(upload_task())


async def upload_heavy_logs(
    computer: ComputerInterface,
    agent_start_time: int,
    agent_dir_config: AgentDirConfig,
    run_dir: str,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
    runtime: float | None = None,
    productive_runtime: float | None = None,
    retry_time: float | None = None,
    num_messages: int | None = None,
) -> None:
    timestamp = f"{time.strftime('%Y-%m-%dT%H-%M-%S-%Z', time.gmtime())}"
    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )
    await upload_sources(
        computer=computer,
        sources=agent_dir_config.directories_to_save,
        run_dir=run_dir,
        run_group_id=run_group_id,
        runs_dir=runs_dir,
        run_id=run_id,
        timestamp=timestamp,
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
    ctx_logger.info(f"Uploaded periodic heavy logs for run {run_dir}")


async def upload_light_logs(
    agent_start_time: int,
    run_dir: str,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
) -> None:
    ctx_logger = logger.bind(
        run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
    )
    await upload_status(
        start_time=agent_start_time,
        run_dir=run_dir,
        status="running",
    )
    ctx_logger.info(f"Uploaded periodic light logs for run {run_dir}")


async def upload_light_and_heavy_logs(
    computer: ComputerInterface,
    agent_start_time: int,
    agent_dir_config: AgentDirConfig,
    run_dir: str,
    run_group_id: str,
    runs_dir: str,
    run_id: str,
) -> tuple[asyncio.Task[None], asyncio.Event]:
    initial_upload_complete = asyncio.Event()

    async def upload_task() -> None:
        ctx_logger = logger.bind(
            run_group_id=run_group_id, runs_dir=runs_dir, run_id=run_id, destinations=["run"]
        )
        try:
            await upload_light_logs(
                agent_start_time=agent_start_time,
                run_dir=run_dir,
                run_group_id=run_group_id,
                runs_dir=runs_dir,
                run_id=run_id,
            )
            await upload_heavy_logs(
                computer=computer,
                agent_start_time=agent_start_time,
                agent_dir_config=agent_dir_config,
                run_dir=run_dir,
                run_group_id=run_group_id,
                runs_dir=runs_dir,
                run_id=run_id,
            )
            ctx_logger.info(f"Uploaded light and heavy logs for run {run_dir}")
        except Exception as e:
            ctx_logger.exception(f"Exception in upload_task: {e}")
            raise
        finally:
            initial_upload_complete.set()

    return asyncio.create_task(upload_task()), initial_upload_complete


async def upload_status(
    start_time: int,
    run_dir: str,
    status: str,
    end_time: int | None = None,
) -> None:
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
    runtime: float | None,
    productive_runtime: float | None,
    retry_time: float | None,
) -> None:
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
