import asyncio
import json
import time

import blobfile as bf
import structlog.stdlib

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.infra.alcatraz import (
    upload_sources,
)
from paperbench.nano.structs import AgentDirConfig

logger = structlog.stdlib.get_logger(component=__name__)


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
    num_messages: int | None = None,
    runtime: float | None = None,
    productive_runtime: float | None = None,
    retry_time: float | None = None,
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
    num_messages: int | None,
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
