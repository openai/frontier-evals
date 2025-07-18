import os

from dotenv import load_dotenv

from nanoeval.solvers.computer_tasks.code_execution_interface import RuntimeConfig
from paperbench.utils import find_dotenv

load_dotenv(find_dotenv())

from preparedness_turn_completer.oai_turn_completer import OpenAITurnCompleter

from alcatraz.clusters.local import LocalConfig
from paperbench.nano.eval import (
    ExternalPythonCodingSolver,
)
from paperbench.nano.structs import (
    JudgeConfig,
    ReproductionConfig,
)
from paperbench.solvers.dummy.solver import PaperBenchDummySolver

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
GRADER_OPENAI_API_KEY = os.getenv("GRADER_OPENAI_API_KEY", OPENAI_API_KEY)

LOCAL_DUMMY_SOLVER_CONFIG = PaperBenchDummySolver(runtime_config=RuntimeConfig())

LOCAL_AISI_SOLVER_CONFIG = ExternalPythonCodingSolver(
    runtime_config=RuntimeConfig(),
    agent_id="aisi-basic-agent-openai-dev",
    cluster_config=LocalConfig(
        image="aisi-basic-agent:latest",
        pull_from_registry=False,
    ),
)

LOCAL_REPRODUCTION_CONFIG = ReproductionConfig(
    timeout=100 * 3600,
    retry_threshold=600,
    overwrite_existing_output=False,
    skip_reproduction=True,
    cluster_config=LocalConfig(
        image="pb-reproducer:latest",
        pull_from_registry=False,
    ),
)

LOCAL_REPRODUCTION_WITHOUT_SKIP_CONFIG = ReproductionConfig(
    timeout=100 * 3600,
    retry_threshold=600,
    overwrite_existing_output=False,
    skip_reproduction=False,
    cluster_config=LocalConfig(
        image="pb-reproducer:latest",
        pull_from_registry=False,
    ),
)

LOCAL_JUDGE_CONFIG = JudgeConfig(
    grade=True,
    grade_locally=True,
    grade_id=0,
    overwrite_existing_output=False,
    scaffold="dummy",
    completer_config=OpenAITurnCompleter.Config(
        model="gpt-4o-mini",
    ),
    code_only=False,
    resources_provided=False,
    cluster_config=LocalConfig(
        image="pb-env:latest",
        pull_from_registry=False,
        environment={"OPENAI_API_KEY": GRADER_OPENAI_API_KEY},
    ),
)
