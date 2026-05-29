from __future__ import annotations

import asyncio
import datetime
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import structlog.stdlib
from tqdm.asyncio import tqdm_asyncio

import chz
from paperbench.monitor.monitor import BasicMonitor, Monitor
from paperbench.paper_registry import paper_registry
from paperbench.tar_utils import safe_extractall

logger = structlog.stdlib.get_logger(component=__name__)


def _describe_monitor_config(monitor_config: Monitor.Config) -> str:
    return f"{monitor_config.__class__.__module__}.{monitor_config.__class__.__qualname__}"


@chz.chz
class MonitorCLIArgs:
    """Monitor agent logs for violations."""

    logs_dir: Path = chz.field(
        doc="Directory containing multiple run groups.",
    )
    run_groups: list[str] = chz.field(
        default_factory=list,
        doc="List of run group IDs to monitor.",
    )
    monitor_config: Monitor.Config = chz.field(
        default_factory=BasicMonitor.Config,
        doc="Specify the monitor to use (default: BasicMonitor).",
    )
    out_dir: Path | None = chz.field(
        default=None,
        doc="Directory to save the monitor results JSON file (default: current directory).",
    )


def get_paper_id_from_run_id(run_id: str) -> str:
    """Extract paper ID from run ID (e.g. 'rice_508398cb-0825-4bf0-b647-a9200ac03d21' -> 'rice')"""
    return run_id.split("_")[0]


async def monitor_single_log(
    run_dir: Path,
    monitor_config: Monitor.Config,
) -> dict[str, Any] | None:
    """
    Monitor a single run's log with the specified monitor.
    """
    run_id = run_dir.name

    # look at latest checkpoint
    checkpoints = [
        i
        for i in list(run_dir.glob("submissions/*-GMT")) + list(run_dir.glob("submissions/*-UTC"))
        if i.is_dir()
    ]
    if len(checkpoints) == 0:
        logger.warning(f"No checkpoints found for {run_id}")
        return None

    latest_checkpoint = None

    for checkpoint in sorted(checkpoints, key=lambda x: x.stem, reverse=True):
        if (checkpoint / "submission.tar.gz").exists():
            latest_checkpoint = checkpoint
            break

    if not latest_checkpoint:
        logger.warning(f"No submission.tar.gz found for {run_id}")
        return None

    with tempfile.TemporaryDirectory() as extract_to:
        with tarfile.open(latest_checkpoint / "submission.tar.gz", "r:gz") as tar:
            safe_extractall(tar, extract_to)

        matches = list(Path(extract_to).glob("**/logs/agent.log"))
        assert len(matches) == 1, f"Expected exactly one agent.log file, found {len(matches)}"
        log_file = latest_checkpoint / "agent.log"
        shutil.copy(matches[0], log_file)

    paper_id = get_paper_id_from_run_id(run_id)

    if not log_file.exists():
        logger.warning(f"Log file not found at {log_file}")
        return None

    monitor_config_payload = monitor_config.model_dump(mode="json")
    logger.info(
        f"Running monitor on agent.log from {run_id}",
        monitor=_describe_monitor_config(monitor_config),
        monitor_config_json=json.dumps(monitor_config_payload, indent=2),
    )

    # Create monitor
    paper = paper_registry.get_paper(paper_id)
    monitor = monitor_config.build(paper=paper)

    # Run monitor on the log file
    result = await asyncio.to_thread(monitor.check_log, log_file.as_posix())

    return {
        "run_group_id": run_dir.parent.name,
        "monitor_type": _describe_monitor_config(monitor_config),
        "monitor_config": monitor_config_payload,
        "paper_id": paper_id,
        "log_file": str(log_file),
        "run_id": run_id,
        "results": {
            "violations": [
                {
                    "line_number": v.line_number,
                    "violation": v.violation,
                    "context": v.context,
                    "context_start": v.context_start,
                }
                for v in result.violations
            ],
            "explanation": result.explanation,
        },
    }


async def monitor_run_group(
    group_dir: Path,
    monitor_config: Monitor.Config,
) -> list[dict[str, Any] | None]:
    """Monitor all runs in a run group directory."""
    run_group_id = group_dir.name

    # Find all run directories
    run_dirs = [d for d in group_dir.iterdir() if d.is_dir()]
    logger.info(f"Found {len(run_dirs)} runs in group {run_group_id}")

    tasks = [
        monitor_single_log(
            run_dir=run_dir,
            monitor_config=monitor_config,
        )
        for run_dir in run_dirs
    ]

    results = await tqdm_asyncio.gather(*tasks, desc=f"Running monitor on {run_group_id}")
    return [r for r in results if r is not None]


async def monitor_multiple_run_groups(
    logs_dir: Path,
    monitor_config: Monitor.Config,
    run_groups: list[str] | None = None,
) -> dict[str, Any] | None:
    """Run monitor on multiple run groups that are in a directory of run groups."""
    if not logs_dir.exists():
        logger.exception(f"Logs directory {logs_dir} does not exist")
        return None

    # Get all available run groups in the logs directory
    available_run_groups = [
        d.name for d in logs_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]

    # If specific run groups were provided, filter to only those that exist
    if run_groups:
        run_groups = [rg for rg in run_groups if rg in available_run_groups]
        if not run_groups:
            logger.warning("None of the specified run groups were found in the logs directory")
            return None
    else:
        # If no run groups specified, use all available ones
        run_groups = available_run_groups

    logger.info(f"Running monitor on run groups: {run_groups}")

    tasks = [
        monitor_run_group(
            group_dir=logs_dir / run_group_id,
            monitor_config=monitor_config,
        )
        for run_group_id in run_groups
    ]

    # Collect all results
    all_results = []
    group_results = await tqdm_asyncio.gather(*tasks, desc="Running monitor on run groups")
    for results in group_results:
        all_results.extend(results)

    # Split results into flagged and other
    flagged_results = [result for result in all_results if len(result["results"]["violations"]) > 0]
    other_results = [result for result in all_results if len(result["results"]["violations"]) == 0]

    monitor_config_payload = monitor_config.model_dump(mode="json")

    # Create final output with results and summary
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "monitor_type": _describe_monitor_config(monitor_config),
        "monitor_config": monitor_config_payload,
        "logs_dir": str(logs_dir.absolute()),
        "run_groups": run_groups,
        "total_runs": len(all_results),
        "flagged_runs": len(flagged_results),
        "flagged_run_ids": [r["run_id"] for r in flagged_results],
        "flagged_results": flagged_results,
        "other_results": other_results,
    }


async def main(
    logs_dir: Path,
    monitor_config: Monitor.Config,
    run_groups: list[str] | None = None,
    out_dir: Path | None = None,
) -> None:
    """
    Main function to run the monitor on a directory of logs.
    """

    monitor_config = monitor_config.model_copy()

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = await monitor_multiple_run_groups(
        logs_dir=logs_dir,
        monitor_config=monitor_config,
        run_groups=run_groups,
    )

    if results:
        # Write results to disk
        filename = f"monitor_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file = out_dir / filename if out_dir else Path(filename)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        logger.info(f"All monitor results written to {output_file}")


async def _run_from_cli(args: MonitorCLIArgs) -> None:
    await main(
        logs_dir=args.logs_dir,
        monitor_config=args.monitor_config,
        run_groups=args.run_groups or None,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    asyncio.run(chz.nested_entrypoint(_run_from_cli))
