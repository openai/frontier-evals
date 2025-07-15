import os
import re
from dataclasses import dataclass

import structlog.stdlib
from paperbench.constants import AGENT_DIR, CODE_DIR, LOGS_DIR, SUBMISSION_DIR

logger = structlog.stdlib.get_logger(component=__name__)


@dataclass
class AgentDirConfig:
    agent_dir: str
    directories_to_save: list[str]


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


def parse_env_var_values(dictionary: dict) -> dict:
    """
    Parses any values in the dictionary that match the ${{ secrets.ENV_VAR }} pattern and replaces
    them with the value of the ENV_VAR environment variable.
    """
    for key, value in dictionary.items():
        if not is_env_var(value):
            continue

        env_var = get_env_var(value)

        if os.getenv(env_var) is None:
            raise ValueError(f"Environment variable `{env_var}` is not set!")

        dictionary[key] = os.getenv(env_var)

    return dictionary
