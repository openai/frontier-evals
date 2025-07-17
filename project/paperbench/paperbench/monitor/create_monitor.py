from typing import Any

from paperbench.monitor.monitor import BasicMonitor, Monitor
from paperbench.paper_registry import Paper


def create_monitor(
    monitor_type: str,
    paper: Paper,
    monitor_kwargs: dict[str, Any],
) -> Monitor:
    """Create and return appropriate monitor instance based on type.

    Args:
        monitor_type: Type of monitor to create ('basic')
        paper: Paper instance containing blacklist and other paper-specific data
        monitor_kwargs: Keyword arguments specific for the monitor

    Returns:
        An instance of the appropriate monitor class
    """

    if monitor_type == "basic":
        return BasicMonitor(paper=paper, **monitor_kwargs)
    else:
        raise ValueError(f"Invalid monitor type: {monitor_type}")
