"""Unit tests for BasicAgentSolver._construct_periodic_reminder.

Regression coverage for the missing-`elif` bug described in
https://github.com/openai/frontier-evals/issues/112: with the original
control flow, the wall-clock branch always ran after the
retry-adjusted branch and silently overwrote `elapsed_time`, so
`use_real_time_limit=True` was a no-op for the periodic reminder.
"""

from paperbench.solvers.basicagent import solver
from paperbench.solvers.basicagent.solver import BasicAgentSolver


def test_periodic_reminder_subtracts_retry_time_when_real_time_limit_enabled(
    monkeypatch,
) -> None:
    """With `use_real_time_limit=True`, retry waits must be excluded."""
    monkeypatch.setattr(solver.time, "time", lambda: 1_100.0)
    agent = BasicAgentSolver(time_limit=200, use_real_time_limit=True)

    message = agent._construct_periodic_reminder(
        start_time=1_000.0,
        total_retry_time=40.0,
    )

    # elapsed = (1100 - 1000) - 40 = 60s -> " 0:01:00"
    assert " 0:01:00 time elapsed out of  0:03:20" in message["content"]


def test_periodic_reminder_uses_wall_clock_when_real_time_limit_disabled(
    monkeypatch,
) -> None:
    """With `use_real_time_limit=False`, the reminder reports raw wall-clock."""
    monkeypatch.setattr(solver.time, "time", lambda: 1_100.0)
    agent = BasicAgentSolver(time_limit=200, use_real_time_limit=False)

    message = agent._construct_periodic_reminder(
        start_time=1_000.0,
        total_retry_time=40.0,
    )

    # elapsed = 1100 - 1000 = 100s -> " 0:01:40"
    assert " 0:01:40 time elapsed out of  0:03:20" in message["content"]


def test_periodic_reminder_falls_back_to_unlimited_message_when_no_time_limit(
    monkeypatch,
) -> None:
    """With `time_limit=None`, the reminder omits the deadline phrasing."""
    monkeypatch.setattr(solver.time, "time", lambda: 1_100.0)
    agent = BasicAgentSolver(time_limit=None, use_real_time_limit=True)

    message = agent._construct_periodic_reminder(
        start_time=1_000.0,
        total_retry_time=40.0,
    )

    content = message["content"]
    assert " 0:01:40 time elapsed" in content
    assert "out of" not in content
