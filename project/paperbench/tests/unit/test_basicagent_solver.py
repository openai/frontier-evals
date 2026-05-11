from paperbench.solvers.basicagent import solver
from paperbench.solvers.basicagent.solver import BasicAgentSolver


def test_periodic_reminder_uses_retry_adjusted_elapsed_time(monkeypatch):
    monkeypatch.setattr(solver.time, "time", lambda: 1_100.0)
    agent = BasicAgentSolver(time_limit=200, use_real_time_limit=True)

    message = agent._construct_periodic_reminder(
        start_time=1_000.0,
        total_retry_time=40.0,
    )

    assert "0:01:00 time elapsed out of  0:03:20" in message["content"]


def test_periodic_reminder_uses_wall_clock_when_real_time_disabled(monkeypatch):
    monkeypatch.setattr(solver.time, "time", lambda: 1_100.0)
    agent = BasicAgentSolver(time_limit=200, use_real_time_limit=False)

    message = agent._construct_periodic_reminder(
        start_time=1_000.0,
        total_retry_time=40.0,
    )

    assert "0:01:40 time elapsed out of  0:03:20" in message["content"]
