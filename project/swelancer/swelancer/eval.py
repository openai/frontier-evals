import ast
import asyncio
import json
import os
import random
import re
import string
import uuid
from pathlib import Path
from typing import Any, Literal, Sequence, cast

import blobfile as bf
import pandas as pd
import structlog.stdlib
from openai.types.chat import ChatCompletionMessageParam
from typing_extensions import TypedDict, override

import chz
from nanoeval.asyncio_utils import generator_with_cleanup
from nanoeval.eval import RolloutSystemError
from nanoeval.metrics.agents import get_summary_error_aware
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerInterface,
    JupyterComputerInterface,
    RuntimeConfig,
)
from nanoeval.solvers.computer_tasks.solver import PythonCodingEval
from nanoeval.solvers.computer_tasks.steps import (
    FinalResult,
)
from nanoeval.solvers.computer_tasks.task import ComputerTask, Grade
from swelancer.prompts import construct_task_prompt
from swelancer.utils.custom_logging import get_default_runs_dir, get_timestamp
from swelancer.utils.general import PATH_TO_SWE_LANCER_TASKS

logger = structlog.stdlib.get_logger(component=__name__)

PUSHER_APP_KEY = "".join(random.choices(string.ascii_letters + string.digits, k=20))
PUSHER_APP_SECRET = "".join(random.choices(string.ascii_letters + string.digits, k=20))
PUSHER_APP_ID = "".join(random.choices(string.digits, k=7))

SWELancerSplit = Literal[
    "diamond",
    "nondiamond",
    "all",
]
SWELancerTaskType = Literal[
    "ic_swe",
    "swe_manager",
    "all",
]


class SwelancerInstance(TypedDict):
    repo: str


def tests_failed_in_pytest_log(pytest_log: str) -> bool:
    lines = pytest_log.split("\n")
    summary_pattern = re.compile(r"=+\s+(\d+)\s+failed.*in\s+[\d\.]+s")

    return any(summary_pattern.search(line) for line in lines)


def create_run_id(
    task_id: str,
) -> str:
    """Creates a run ID."""
    return f"{task_id}_{str(uuid.uuid4())}"


def create_run_dir(
    run_group: str,
    run_id: str,
    runs_dir: str,
) -> str:
    """Creates a directory for the run."""
    run_dir = bf.join(runs_dir, run_group, run_id)
    bf.makedirs(run_dir)
    return run_dir


class SWELancerGrade(Grade):
    patch_path: str
    grade_report_path: str | None


class SWELancerTask(ComputerTask):
    instance: SwelancerInstance
    issue_id: str
    run_group_id: str
    run_id: str
    runs_dir: str
    run_dir: str

    price: float
    variant: str
    acceptable_folders: list[str] = []
    manager_data: dict[str, Any] | None = None
    manager_commit: str | None = None

    # retry_message: str = "The task isn't complete yet! You can submit your answer using apply_patch via the command-line."
    retry_message: str = "The task isn't complete yet! You must write your output to this path exactly: '/app/expensify/manager_decisions.json'. You cannot submit to /mnt/data!"
    # Install jupyter in the correct conda environment and start it within that environment.
    # Ensures the model can properly run unit tests.
    jupyter_setup: Sequence[str] | None = (
        "bash",
        "-c",
        "conda run -n testbed pip install jupyter && conda run -n testbed jupyter kernel --ip 0.0.0.0",
    )
    cwd: str = "/app/expensify"

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    n_test_runs: int
    disable_internet: bool

    @override
    async def _setup(self, computer: ComputerInterface, runtime_config: RuntimeConfig) -> None:
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id, runs_dir=self.runs_dir, run_id=self.run_id
        )
        async with asyncio.timeout(2400):
            try:
                # Assert various things about the environment
                ctx_logger.info("SETUP", destinations=["run"])
                if isinstance(computer, JupyterComputerInterface):
                    await computer.check_execute(
                        """import os; assert os.environ.get('CONDA_DEFAULT_ENV') == 'testbed', os.environ.get('CONDA_DEFAULT_ENV')""",
                    )
                    ctx_logger.info("Correct Jupyter environment is loaded.", destinations=["run"])

                if self.variant == "swe_manager":
                    # reset to the broken commit
                    await computer.check_shell_command(
                        f"bash -c 'git reset --hard {self.manager_commit}^'"
                    )
                else:
                    # instead of a time based check, check until the SETUP_DONE environment variable is set
                    ctx_logger.info("Waiting for setup to be done", destinations=["run"])
                    while True:
                        res = await computer.send_shell_command("cat /setup_done.txt")
                        if res.output.decode("utf-8").strip() == "done":
                            ctx_logger.info("Setup is done", destinations=["run"])
                            break
                        await asyncio.sleep(15)

                # power user flag: if using a stale image where zip and unzip are not installed
                if os.getenv("INSTALL_ZIP_UNZIP", "false").lower() == "true":
                    await computer.check_shell_command(
                        "apt-get update && apt-get install -y zip unzip"
                    )

                # Zip the tests directory with a password
                ctx_logger.info("Zipping tests", destinations=["run"])
                await computer.check_shell_command(
                    "cd /app/tests && zip -P 'secretpasswordhere' -r /app/tmp.zip . && mv /app/tmp.zip /app/tests.zip"
                )
                # Clear tests directory
                await computer.send_shell_command("rm -rf /app/tests")

                # Remove existing git repo and create a new one
                await computer.check_shell_command("rm -rf .git")
                await computer.check_shell_command("git init")

                # Creat a temp commit to mark where the model's changes begin
                if self.variant == "ic_swe":
                    await computer.check_shell_command("bash -c 'git add .'")
                    await computer.send_shell_command(
                        'bash -c \'git -c user.name="temp" -c user.email="temp@example.com" commit -m "temp"\''
                    )
                    ctx_logger.info("Temp commit created", destinations=["run"])

                if self.disable_internet:
                    await computer.disable_internet()
            except Exception as e:
                ctx_logger.exception("An error occurred during setup", destinations=["run"])
                raise RolloutSystemError(f"An error occurred during setup: {e}") from e
        ctx_logger.info("Setup complete", destinations=["run"])

    @override
    async def grade(
        self, computer: ComputerInterface, runtime_config: RuntimeConfig
    ) -> SWELancerGrade:
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id, runs_dir=self.runs_dir, run_id=self.run_id
        )

        timeout = 1200
        ctx_logger.info("Grading task...", destinations=["run"])
        try:
            async with asyncio.timeout(timeout):
                # Unzip locked testing files
                ctx_logger.info("Unzipping tests", destinations=["run"])
                res = await computer.check_shell_command(
                    "unzip -q -P 'secretpasswordhere' -o /app/tests.zip -d /app/tests"
                )
                ctx_logger.info(res.output.decode("utf-8"), destinations=["run"])
                if self.variant == "swe_manager":
                    return await self._grade_swe_manager(computer)
                elif self.variant == "ic_swe":
                    return await self._grade_swe_ic(computer)
                else:
                    raise ValueError(f"Unknown variant: {self.variant}")

        except asyncio.TimeoutError as e:
            ctx_logger.warning("Grading timed out", destinations=["run"])
            raise RolloutSystemError("Grading timed out") from e
        except Exception as e:
            ctx_logger.exception("An error occurred during grading", destinations=["run"])
            raise RolloutSystemError(f"An error occurred during grading: {e}") from e

    async def _grade_swe_manager(self, computer: ComputerInterface) -> SWELancerGrade:
        """Grades the SWE Manager variant tasks."""
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id, runs_dir=self.runs_dir, run_id=self.run_id
        )
        manager_decisions = await computer.send_shell_command(
            """bash -c 'cat /app/expensify/manager_decisions.json'"""
        )
        try:
            manager_decisions_json = json.loads(manager_decisions.output.decode("utf-8"))
            # check if the manager decisions are correct
            correct = 0.0
            correct_option = int(
                self.manager_data["game"]["correct_proposal"]["id"]  # type: ignore
            )
            submitted_option = int(list(manager_decisions_json.values())[0])
        except Exception as e:
            ctx_logger.warning(
                f"Failed to parse manager decisions: {e}",
                destinations=["run"],
                _print=True,
            )
            return SWELancerGrade(
                score=0,
                grader_log=json.dumps({}),
                patch_path="",
                grade_report_path="",
            )

        if submitted_option == correct_option:
            correct = 1.0
        else:
            correct = 0.0

        return SWELancerGrade(
            score=correct,
            grader_log=json.dumps(
                {
                    "earned": self.price * correct,
                    "available": self.price,
                    "variant": self.variant,
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "reasoning_tokens": self.reasoning_tokens,
                }
            ),
            patch_path="",
            grade_report_path="",
        )

    async def _grade_swe_ic(self, computer: ComputerInterface) -> SWELancerGrade:
        """Grades the IC SWE variant tasks"""
        ctx_logger = logger.bind(
            run_group_id=self.run_group_id, runs_dir=self.runs_dir, run_id=self.run_id
        )

        pytest_outputs = []
        model_patch = b""

        await computer.check_shell_command(
            "git -c core.fileMode=false diff --binary HEAD > /tmp/patch.diff"
        )
        model_patch = await computer.download("/tmp/patch.diff")

        # we simply re-use the computer for grading
        grading_computer = computer

        matching_folder = str(self.question_id)

        ctx_logger.info(
            f"self.question_id: {self.question_id}\n\n matching_folder: {matching_folder}",
            destinations=["run"],
        )
        ctx_logger.info(
            f"Running n={self.n_test_runs} tests for {self.question_id}.",
            destinations=["run"],
        )

        test_results = []
        for idx in range(self.n_test_runs):
            ctx_logger.info(
                f"Running test {self.question_id} ({idx + 1}/{self.n_test_runs})",
                destinations=["run"],
            )
            # Set the ISSUE_ID environment variable
            set_env_command = f"""echo "export ISSUE_ID={self.question_id}" >> ~/.bashrc"""
            await grading_computer.send_shell_command(f"bash -i -c '{set_env_command}'")
            ctx_logger.info(f"Making directory /app/tests/logs/{self.question_id}")
            await grading_computer.send_shell_command(
                f"""mkdir -p /app/tests/logs/{self.question_id}"""
            )
            await grading_computer.send_shell_command(
                f"""rm -rf /app/tests/logs/{self.question_id}/*"""
            )

            # Run tests against the model's code
            ctx_logger.info("running ansible playbook")
            res = await grading_computer.send_shell_command(
                f"""bash -i -c 'export ISSUE_ID={self.question_id}; ansible-playbook -i "localhost," --connection=local /app/tests/run_tests.yml'"""
            )

            # grab the exit code to use to determine if the tests passed, failed, or errored
            exit_code_raw = await grading_computer.send_shell_command(
                f"bash -i -c 'export ISSUE_ID={self.question_id}; cat /app/tests/logs/$ISSUE_ID/pytest_exit_code'"
            )
            raw = exit_code_raw.output.decode()
            pytest_exit = None
            for line in reversed(raw.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    pytest_exit = int(line)
                    break
                except ValueError:
                    continue
            if pytest_exit is None:
                pytest_exit = -1
            ctx_logger.info(f"pytest_exit: {pytest_exit}", destinations=["run"])

            ctx_logger.info(
                f"Ansible playbook output: {res.output.decode('utf-8')}",
                destinations=["run"],
            )

            # Get pytest.log
            pytest_log = await grading_computer.send_shell_command(
                f"""bash -i -c 'export ISSUE_ID={self.question_id}; cat /app/tests/logs/$ISSUE_ID/pytest.log'"""
            )

            pytest_output = pytest_log.output.decode("utf-8")
            pytest_outputs.append(pytest_output)
            ctx_logger.info(f"Pytest output: {pytest_output}", destinations=["run"])

            if pytest_exit == 0:
                test_results.append(1.0)
                # we found a succesful test, can exit from our retry loop
                break
            elif pytest_exit >= 2:
                raise Exception(f"pytest failed with exit code {pytest_exit}")
            elif pytest_exit == -1:
                test_results.append(-1.0)
            else:
                test_results.append(0.0)
        if not test_results:
            raise RolloutSystemError(
                f"No tests properly ran for {self.question_id}; matching_folder: {matching_folder}"
            )

        ctx_logger.info(f"Test results: {test_results}", destinations=["run"])
        correct = 1.0 if any(result == 1.0 for result in test_results) else 0.0

        return SWELancerGrade(
            score=correct,
            grader_log=json.dumps(
                {
                    "earned": self.price * correct,
                    "available": self.price,
                    "pytest_logs": pytest_outputs,
                    "variant": self.variant,
                    "model_patch": model_patch.decode("utf-8"),
                    "input_tokens": self.input_tokens,
                    "output_tokens": self.output_tokens,
                    "reasoning_tokens": self.reasoning_tokens,
                    "pytest_exit": pytest_exit,
                    "notes": "Tests ran smoothly, score is based on model performance.",
                }
            ),
            patch_path="",
            grade_report_path="",
        )


@chz.chz
class SWELancerEval(PythonCodingEval):
    split: SWELancerSplit = chz.field(default="all", doc="Set of tasks to evaluate on")
    task_type: SWELancerTaskType = chz.field(default="ic_swe", doc="Type of tasks to evaluate on")
    taskset: list[str] | None = chz.field(default=None)

    runs_dir: str = chz.field(default=get_default_runs_dir())

    docker_image_prefix: str = chz.field(
        default="swelancer_x86",
        doc="Prefix for the docker image used for running tasks. "
        "For a task with ISSUE_ID, the docker image used will be {docker_image_prefix}_{ISSUE_ID}:{tag}. "
        "You may wish to set this to your contain registry, e.g. 'docker.io/swelancer_x86'",
    )

    use_single_image: bool = chz.field(
        default=False,
        doc=(
            "If True, run the eval using a single monolithic docker image, "
            "moving the individual task setup to runtime. "
            "The image used will be {docker_image_prefix}_monolith:{tag}."
        ),
    )

    docker_image_tag: str = chz.field(
        default="latest", doc="Tag for the docker image used for running tasks."
    )

    n_test_runs: int = chz.field(
        default=3,
        doc=(
            "Tests will be run this many times until they don't fail,"
            " to account for occasional non-deterministic test failures."
        ),
    )
    disable_internet: bool = chz.field(
        default=True,
        doc=(
            "If True (default), internet will be disabled"
            " on the computer used for rollouts and grading"
        ),
    )

    @chz.validate
    def _task_type_must_be_specific(self) -> None:
        """
        `task_type='all'` was supported in older configs but is now
        deprecated.  Fail fast so callers see a clear error instead of
        silently falling back to the default filtering logic.
        """
        if self.task_type != "ic_swe" and self.task_type != "swe_manager":
            raise ValueError("task_type must be either 'ic_swe' or 'swe_manager'.")

    @chz.validate
    def _manager_requires_single_image(self) -> None:
        """
        swe_manager tasks are only supported with the monolith docker
        image.  Force callers to set `use_single_image=True`; otherwise
        fail fast with a clear message.
        """
        if self.task_type == "swe_manager" and not self.use_single_image:
            raise ValueError(
                "SWELancerEval configuration error: "
                "`use_single_image` must be True when `task_type=='swe_manager'` "
                "(manager tasks require the monolith image)."
            )

    @override
    def get_name(self) -> str:
        return "SWELancer"

    @chz.init_property
    def run_group_id(self) -> str:
        return f"{get_timestamp()}_run-group_{self.solver.shortname()}"

    @override
    async def get_instances(self) -> list[SWELancerTask]:
        ctx_logger = logger.bind(run_group_id=self.run_group_id, runs_dir=self.runs_dir)

        (Path(self.runs_dir) / self.run_group_id).mkdir(parents=True, exist_ok=True)
        ctx_logger.info(
            f"Writing run group logs to {bf.join(self.runs_dir, self.run_group_id, 'group.log')}",
            _print=True,
        )

        ctx_logger.info(
            f"Getting tasks for split {self.split} and task type {self.task_type}",
            destinations=["group"],
        )
        tasks = pd.read_csv(PATH_TO_SWE_LANCER_TASKS)
        tasks["proposals"] = tasks["proposals"].fillna("")  # pandas reads empty as float nan

        if self.split != "all":
            tasks = tasks[tasks["set"] == self.split]

        if self.task_type == "ic_swe" or self.task_type == "swe_manager":
            tasks = tasks[tasks["variant"] == self.task_type]

        SWEFL_ENV = {
            "PUSHER_APP_KEY": PUSHER_APP_KEY,
            "PUSHER_APP_SECRET": PUSHER_APP_SECRET,
            "PUSHER_APP_ID": PUSHER_APP_ID,
            "USE_WEB_PROXY": os.getenv("USE_WEB_PROXY"),
            "EXPENSIFY_URL": os.getenv("EXPENSIFY_URL"),
            "NEW_EXPENSIFY_URL": os.getenv("NEW_EXPENSIFY_URL"),
            "ISSUE_ID": "0",
            "LC_ALL": "C.UTF-8",
        }

        swelancer_tasks = []
        for attempt_idx in range(self.n_tries):
            for task in tasks.to_dict(orient="records"):
                task_env = SWEFL_ENV.copy()
                task_env["ISSUE_ID"] = task["question_id"]
                task_env["EVAL_VARIANT"] = task["variant"]

                if self.taskset and task["question_id"] not in self.taskset:
                    continue
                prompt_str = construct_task_prompt(
                    task["variant"],
                    task["title"].encode("utf-8").decode("unicode_escape"),
                    task["description"].encode("utf-8").decode("unicode_escape"),
                    task["price"],
                    task["proposals"].encode("utf-8").decode("unicode_escape"),
                )
                prompt = cast(
                    list[ChatCompletionMessageParam],
                    [{"role": "user", "content": prompt_str}],
                )
                acceptable_folders = ast.literal_eval(task["acceptable_folders"])
                if str(task["manager_data"]) == "nan":
                    manager_data = None
                else:
                    manager_data = ast.literal_eval(task["manager_data"])
                manager_commit = (
                    task["manager_commit"] if str(task["manager_commit"]) != "nan" else None
                )
                cwd = task["cwd"]

                run_id = create_run_id(task["question_id"])
                run_dir = create_run_dir(self.run_group_id, run_id, self.runs_dir)
                docker_image = (
                    f"{self.docker_image_prefix}_{task['question_id']}"
                    if not self.use_single_image
                    else f"{self.docker_image_prefix}_monolith"
                ) + f":{self.docker_image_tag}"

                swelancer_tasks.append(
                    SWELancerTask(
                        question_id=task["question_id"],
                        prompt=prompt,
                        price=task["price"],
                        variant=task["variant"],
                        acceptable_folders=acceptable_folders,
                        manager_data=manager_data,
                        manager_commit=manager_commit,
                        issue_id=task["question_id"],
                        cwd=cwd,
                        attempt_id=attempt_idx,
                        run_group_id=self.run_group_id,
                        run_id=run_id,
                        runs_dir=self.runs_dir,
                        run_dir=run_dir,
                        environment=task_env,
                        grade_every_step=False,
                        instance=SwelancerInstance(repo="expensify"),
                        docker_image=docker_image,
                        n_test_runs=self.n_test_runs,
                        disable_internet=self.disable_internet,
                    )
                )
        ctx_logger.info("Fetched tasks.", destinations=["group"])
        return swelancer_tasks

    @override
    async def get_tasks(self) -> Sequence[SWELancerTask]:
        # get_instances() already handles n_tries, so return it directly.
        return await self.get_instances()

    @override
    async def evaluate(self, task: ComputerTask) -> FinalResult:
        assert isinstance(task, SWELancerTask)

        ctx_logger = logger.bind(
            run_group_id=self.run_group_id, runs_dir=self.runs_dir, run_id=task.run_id
        )

        ctx_logger.info(f"Evaluating task with ID {task.question_id}", destinations=["run"])

        async with generator_with_cleanup(self.solver.run(task)) as gen:
            ctx_logger.info(f"Loaded generator {gen}", destinations=["run"])
            async for step in gen:
                ctx_logger.info(
                    f"Running step {step} for task {task.question_id}",
                    destinations=["run"],
                )
                if isinstance(step, (FinalResult)):
                    return step

        raise ValueError("Solver did not return a final result! This is a bug.")

    @override
    async def get_full_summary(
        self, results: list[tuple[ComputerTask, FinalResult | RolloutSystemError]]
    ) -> dict[str, Any]:
        for task, _ in results:
            assert isinstance(task, SWELancerTask)
        run_group_dir = bf.join(self.runs_dir, self.run_group_id)

        summary = await asyncio.to_thread(
            get_summary_error_aware,
            [
                (
                    task,
                    (result.correct if isinstance(result, (FinalResult)) else result),
                )
                for task, result in results
            ],
        )
        summary["length_stats"] = self._get_convo_len_stats(results)

        summary["total_input_tokens"] = 1
        summary["total_output_tokens"] = 1
        summary["total_reasoning_tokens"] = 1
        for _task, result in results:
            if isinstance(result, FinalResult):
                try:
                    grader_log = json.loads(result.grade.grader_log)
                except json.JSONDecodeError:
                    grader_log = {}
                summary["total_input_tokens"] += grader_log.get("input_tokens", 0)
                summary["total_output_tokens"] += grader_log.get("output_tokens", 0)
                summary["total_reasoning_tokens"] += grader_log.get("reasoning_tokens", 0)

        # Make csv of results
        input_tokens = []
        output_tokens = []
        reasoning_tokens = []
        for _task, result in results:
            if isinstance(result, FinalResult):
                try:
                    grader_log = json.loads(result.grade.grader_log)
                except json.JSONDecodeError:
                    grader_log = {}
                input_tokens.append(grader_log.get("input_tokens", 0))
                output_tokens.append(grader_log.get("output_tokens", 0))
                reasoning_tokens.append(grader_log.get("reasoning_tokens", 0))
            else:
                input_tokens.append(0)
                output_tokens.append(0)
                reasoning_tokens.append(0)
        earned = [_parse_earned(result) for _task, result in results]
        summary["total_earned"] = sum(earned)
        data = {
            "question_id": [task.question_id for task, _ in results],
            "attempt_id": [task.attempt_id for task, _ in results],
            "correct": [
                result.correct if isinstance(result, FinalResult) else False
                for task, result in results
            ],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "earned": earned,
        }
        df = pd.DataFrame(data)
        out_path = bf.join(run_group_dir, "results.csv")
        df.to_csv(out_path, index=False)
        return summary


def _parse_earned(result: FinalResult | RolloutSystemError) -> float:
    if not isinstance(result, FinalResult):
        return 0.0
    try:
        return json.loads(result.grade.grader_log).get("earned", 0.0)
    except json.JSONDecodeError:
        return 0.0
