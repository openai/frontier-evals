from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import structlog.stdlib
from pydantic import BaseModel

from paperbench.constants import AGENT_DIR, CODE_DIR, LOGS_DIR, SUBMISSION_DIR

logger = structlog.stdlib.get_logger(component=__name__)


@dataclass
class AgentDirConfig:
    agent_dir: str
    directories_to_save: list[str]


class AgentOutput(BaseModel):
    run_id: str
    time_start: float
    time_end: float
    machine_info: str | None = None
    error_msg: str | None = None
    runtime_in_seconds: float
    status_exists: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentOutput:
        try:
            return AgentOutput(
                run_id=data["run_id"],
                time_start=data["time_start"],
                time_end=data["time_end"],
                machine_info=data.get("machine_info"),
                error_msg=data.get("error_msg"),
                runtime_in_seconds=data["runtime_in_seconds"],
                status_exists=data["status_exists"],
            )
        except KeyError as e:
            raise ValueError("Missing required field in agent output") from e

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "machine_info": self.machine_info,
            "error_msg": self.error_msg,
            "runtime_in_seconds": self.runtime_in_seconds,
            "status_exists": self.status_exists,
        }


def prepare_agent_dir_config() -> AgentDirConfig:
    # TODO: Delete this; it's essentially wrapping a few constants in a function.

    return AgentDirConfig(
        directories_to_save=[
            SUBMISSION_DIR,
            LOGS_DIR,
            CODE_DIR,
        ],
        agent_dir=AGENT_DIR,
    )


def get_env_var(value: str) -> str | None:
    """Returns the name of the environment variable in the format `${secrets.<name>}`."""

    if not isinstance(value, str):
        return None

    env_var_pattern = r"\$\{\{\s*secrets\.(\w+)\s*\}\}"
    match = re.match(env_var_pattern, value)

    if not match:
        return None

    return match.group(1)


def is_env_var(value: str) -> bool:
    """Checks if the value is an environment variable."""
    return get_env_var(value) is not None


def parse_env_var_values(dictionary: dict[str, str]) -> dict[str, str]:
    """
    Parses any values in the dictionary that match the ${{ secrets.ENV_VAR }} pattern and replaces
    them with the value of the ENV_VAR environment variable.
    """
    for key, value in dictionary.items():
        if not is_env_var(value):
            continue

        env_var = get_env_var(value)
        if env_var is None:
            raise ValueError("Pattern did not yield an env-var name")

        env_val = os.getenv(env_var)
        if env_val is None:
            raise ValueError(f"Environment variable `{env_var}` is not set!")

        dictionary[key] = env_val

    return dictionary
