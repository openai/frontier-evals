from __future__ import annotations

from paperbench.nano.structs import AgentDirConfig

WORKSPACE_BASE = "/home"
SUBMISSION_DIR = WORKSPACE_BASE + "/submission"
LOGS_DIR = WORKSPACE_BASE + "/logs"
AGENT_DIR = WORKSPACE_BASE + "/agent"


AGENT_DIR_CONFIG: AgentDirConfig = AgentDirConfig(
    directories_to_save=[SUBMISSION_DIR, LOGS_DIR], agent_dir=AGENT_DIR
)
