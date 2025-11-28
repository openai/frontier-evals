from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog.stdlib
import yaml

from paperbench.utils import parse_env_var_values

logger = structlog.stdlib.get_logger(component=__name__)


@dataclass(frozen=True)
class Agent:
    id: str
    name: str
    agents_dir: Path
    start: Path
    dockerfile: Path
    kwargs: dict[str, Any]
    env_vars: dict[str, str]
    privileged: bool = False
    mount_docker_socket: bool = False
    kwargs_type: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.start, Path), "Agent start script must be a pathlib.Path object."
        assert isinstance(self.dockerfile, Path), "Agent dockerfile must be a pathlib.Path object."
        assert isinstance(self.kwargs, dict), "Agent kwargs must be a dictionary."
        assert isinstance(self.privileged, bool), "Agent privileged must be a boolean."
        assert isinstance(self.mount_docker_socket, bool), (
            "Agent mount_docker_socket must be a boolean."
        )

        if self.kwargs_type is not None:
            assert isinstance(self.kwargs_type, str), "Agent kwargs_type must be a string."
        else:  # i.e., self.kwargs_type is None
            assert self.kwargs == {}, "Agent kwargs_type must be set if kwargs are provided."

        assert isinstance(self.env_vars, dict), "Agent env_vars must be a dictionary."

        assert self.start.exists(), f"start script {self.start} does not exist."
        assert self.dockerfile.exists(), f"dockerfile {self.dockerfile} does not exist."

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Agent:
        agents_dir = Path(data["agents_dir"])
        try:
            return Agent(
                id=data["id"],
                name=data["name"],
                agents_dir=agents_dir,
                start=agents_dir / data["start"],
                dockerfile=agents_dir / data["dockerfile"],
                kwargs=data.get("kwargs", {}),
                kwargs_type=data.get("kwargs_type", None),
                env_vars=data.get("env_vars", {}),
                privileged=data.get("privileged", False),
                mount_docker_socket=data.get("mount_docker_socket", False),
            )
        except KeyError as e:
            raise ValueError(f"Missing key {e} in agent config!") from e


class AgentRegistry:
    def get_agents_dir(self) -> Path:
        """Retrieves the agents directory within the registry."""

        return Path(__file__).parent

    def get_agent(self, agent_id: str) -> Agent:
        """Fetch the agent from the registry."""

        agents_dir = self.get_agents_dir()

        for fpath in agents_dir.glob("**/config.yaml"):
            with open(fpath, "r") as f:
                contents = yaml.safe_load(f)

            if agent_id not in contents:
                continue

            kwargs = contents[agent_id].get("kwargs", {})
            kwargs_type = contents[agent_id].get("kwargs_type", None)
            env_vars = contents[agent_id].get("env_vars", {})
            privileged = contents[agent_id].get("privileged", False)

            # env vars can be used both in kwargs and env_vars
            kwargs = parse_env_var_values(kwargs)
            env_vars = parse_env_var_values(env_vars)

            return Agent.from_dict(
                {
                    **contents[agent_id],
                    "id": agent_id,
                    "name": fpath.parent.name,
                    "agents_dir": agents_dir,
                    "kwargs": kwargs,
                    "kwargs_type": kwargs_type,
                    "env_vars": env_vars,
                    "privileged": privileged,
                }
            )

        raise ValueError(f"Agent with id {agent_id} not found")


registry = AgentRegistry()


def get_agents_env_vars(registry: AgentRegistry) -> dict[str, str]:
    """Parses agent.env txt file of KEY=VALUE into a dictionary"""
    agent_env_path = registry.get_agents_dir() / "agent.env"

    if not agent_env_path.exists():
        logger.warning(f"agent.env not found in {agent_env_path}")
        return {}
    env_vars = {}
    with open(agent_env_path, "r") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                env_vars[key] = value
    return env_vars
