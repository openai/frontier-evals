import asyncio
import json
import logging
import os
import re
import shlex
from textwrap import dedent
from typing import Any, AsyncGenerator, cast

import openai
import structlog
import tenacity
import tiktoken
from dotenv import load_dotenv
from nanoeval_alcatraz.alcatraz_computer_interface import (
    AlcatrazComputerRuntime,
)
from openai import AsyncOpenAI
from openai._types import NotGiven
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionReasoningEffort,
)
from typing_extensions import override

import chz
from nanoeval.contrib.utils import run_with_startup_timeout
from nanoeval.eval import RolloutSystemError
from nanoeval.solvers.computer_tasks.code_execution_interface import (
    ComputerRuntime,
    RuntimeConfig,
)
from nanoeval.solvers.computer_tasks.solver import PythonCodingSolver
from nanoeval.solvers.computer_tasks.steps import (
    FinalResult,
    Step,
)
from nanoeval.solvers.computer_tasks.task import ComputerTask
from swelancer.eval import SWELancerTask

logger = structlog.stdlib.get_logger(component=__name__)

STARTUP_TIMEOUT = 1200  # seconds (20 mins)
DEFAULT_CTX_LIMIT = 110000
MODEL_TO_CTX_LIMIT = {
    "openai/gpt-4o": 128000,
    "openai/gpt-4.1": 1047576,
    "openai/o1": 200000,
    "openai/o3": 200000,
    "openai/o3-mini": 200000,
    "openai/o4-mini": 200000,
    "anthropic/claude-3-5-sonnet": 200000,
    "anthropic/claude-3.5-sonnet-20240620": 200000,
    "deepseek/deepseek-coder": 128000,
    "meta-llama/llama-3-70b-instruct": 8000,
    "meta-llama/llama-3.3-70b-instruct": 128000,
}

load_dotenv()


def get_client_and_model(model: str) -> tuple[AsyncOpenAI, str]:
    if model.startswith("openai/"):
        model = model[len("openai/") :]
        base_url = None
        api_key = os.environ.get("OPENAI_API_KEY")
    else:  # Otherwise, use openrouter
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY")

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    ), model


def count_tokens(messages: list[dict[str, Any]], model: str) -> int:
    """
    Count the number of tokens in a list of messages. Uses tiktoken to count tokens.
    If the model is not found, use gpt-4.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4")

    num_tokens = 0
    for message in messages:
        # Every message follows format: {"role": role, "content": content}
        num_tokens += 4  # Every message follows format: <im_start>{role/name}\n{content}<im_end>\n
        for value in message.values():
            num_tokens += len(encoding.encode(str(value)))

    return num_tokens


def trim_messages(
    task: SWELancerTask, messages: list[dict[str, Any]], model: str
) -> list[dict[str, Any]]:
    """Trim messages to fit within token limit by removing older messages."""
    ctx_logger = logger.bind(
        run_group_id=task.run_group_id,
        runs_dir=task.runs_dir,
        run_id=task.run_id,
    )

    max_tokens = MODEL_TO_CTX_LIMIT[model] if model in MODEL_TO_CTX_LIMIT else DEFAULT_CTX_LIMIT
    while len(messages) > 1 and count_tokens(messages, model) > max_tokens:
        ctx_logger.info("Trimming message...", destinations=["run"])
        messages.pop(1)
    return messages


OPENROUTER_RETRY_EXCEPTIONS = (
    openai.BadRequestError,
    openai.InternalServerError,
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    json.JSONDecodeError,
)


async def get_model_response(
    task: SWELancerTask,
    messages: list[dict[str, Any]],
    model: str,
    reasoning_effort: str | None = None,
    logger: structlog.stdlib.BoundLogger | None = None,
) -> tuple[str, int, int, int | None]:
    client, model = get_client_and_model(model)

    messages = trim_messages(task, messages, model)

    @tenacity.retry(
        wait=tenacity.wait_random_exponential(min=1, max=300),  # Max wait time of 5 minutes
        # 20 attempts or 20 minutes max
        stop=(tenacity.stop_after_attempt(20) | tenacity.stop_after_delay(20 * 60)),
        retry=tenacity.retry_if_exception_type(OPENROUTER_RETRY_EXCEPTIONS),
        before_sleep=(tenacity.before_sleep_log(logger._logger, logging.INFO) if logger else None),
        reraise=True,
    )
    async def _get_model_response() -> ChatCompletion:
        chat_completion = await client.chat.completions.create(
            messages=cast(list[ChatCompletionMessageParam], messages),
            model=model,
            reasoning_effort=cast(ChatCompletionReasoningEffort, reasoning_effort)
            if reasoning_effort is not None
            else NotGiven(),
        )
        if chat_completion is None:
            raise json.JSONDecodeError(msg="chat_completion is None", doc="", pos=0)
        if chat_completion.choices is None or len(chat_completion.choices) == 0:
            raise json.JSONDecodeError(
                msg="chat_completion.choices is None or empty", doc="", pos=0
            )
        if chat_completion.choices[0] is None:
            raise json.JSONDecodeError(msg="chat_completion.choices[0] is None", doc="", pos=0)
        if chat_completion.choices[0].message is None:
            raise json.JSONDecodeError(
                msg="chat_completion.choices[0].message is None", doc="", pos=0
            )
        return chat_completion

    chat_completion = await _get_model_response()

    # get number of input and output tokens
    input_tokens = chat_completion.usage.prompt_tokens if chat_completion.usage is not None else 0
    output_tokens = (
        chat_completion.usage.completion_tokens if chat_completion.usage is not None else 0
    )
    reasoning_tokens = (
        chat_completion.usage.completion_tokens_details.reasoning_tokens
        if chat_completion.usage is not None
        and chat_completion.usage.completion_tokens_details is not None
        else 0
    )
    assert chat_completion.choices[0].message.content is not None
    return (
        chat_completion.choices[0].message.content,
        input_tokens,
        output_tokens,
        reasoning_tokens,
    )


@chz.chz
class SimpleAgentSolver(PythonCodingSolver):
    computer_runtime: ComputerRuntime = chz.field(default_factory=AlcatrazComputerRuntime)

    name: str = chz.field(default="SimpleAgentSolver")
    model: str = chz.field(
        default="openai/gpt-4o",
        doc="Assumes model to be in OpenRouter format of PROVIDER/MODEL",
    )
    reasoning_effort: str | None = chz.field(default=None)

    runtime_config: RuntimeConfig = chz.field(default_factory=RuntimeConfig)

    def shortname(self) -> str:
        return "simple-solver"

    @override
    async def run(self, task: ComputerTask) -> AsyncGenerator[Step | FinalResult, None]:
        assert isinstance(task, SWELancerTask), (
            f"SimpleAgentSolver only supports SWELancerTasks, got {type(task)}"
        )

        ctx_logger = logger.bind(
            run_group_id=task.run_group_id,
            runs_dir=task.runs_dir,
            run_id=task.run_id,
        )

        try:
            ctx_logger.info("Starting computer...", destinations=["run"])
            async with run_with_startup_timeout(
                self.computer_runtime, task, STARTUP_TIMEOUT
            ) as computer:
                # 1. Query the API / some agent
                messages = []

                assert "content" in task.prompt[0]
                assert isinstance(task.prompt[0]["content"], str)
                instructions = task.prompt[0]["content"] + "\n\n" + ADDITIONAL_AGENT_PROMPT
                messages.append({"role": "user", "content": instructions})

                max_turns = 40
                for remaining_turns in range(max_turns, 0, -1):
                    (
                        model_response,
                        input_tokens,
                        output_tokens,
                        reasoning_tokens,
                    ) = await get_model_response(
                        task, messages, self.model, self.reasoning_effort, ctx_logger
                    )
                    task.input_tokens += input_tokens
                    task.output_tokens += output_tokens
                    task.reasoning_tokens += reasoning_tokens if reasoning_tokens is not None else 0

                    messages.append({"role": "assistant", "content": model_response})

                    ctx_logger.info(
                        f"Message history on turn {remaining_turns}, number of messages: {len(messages)}: {messages}",
                        destinations=["run"],
                    )

                    execution_output = None

                    # Check for user-tool calls
                    timeout = 300  # 5 mins
                    try:
                        async with asyncio.timeout(timeout):
                            if "<user-tool>" in model_response:
                                ctx_logger.info(
                                    "User tool call detected",
                                    destinations=["run"],
                                )
                                try:
                                    await computer.check_shell_command(
                                        "unzip -q -P 'secretpasswordhere' /app/tests.zip -d /app/tests"
                                    )
                                    await computer.check_shell_command(
                                        "bash -c -i 'user-tool'; exit"
                                    )
                                    execution_output = (
                                        "User tool output is in /app/expensify/user_tool"
                                    )
                                except Exception as e:
                                    execution_output = f"Error: {e}"
                                await computer.send_shell_command("rm -rf /app/tests")
                                await computer.send_shell_command("rm -rf /app/tests")
                            elif model_response.lower() == "done":
                                break
                            else:
                                ctx_logger.info(
                                    "No user tool call detected; executing code",
                                    destinations=["run"],
                                )
                                # Extract Python code from the model response
                                python_blocks = re.findall(
                                    r"```python\s*\n(.*?)\n```",
                                    model_response,
                                    re.DOTALL,
                                )

                                if not python_blocks:
                                    messages.append(
                                        {
                                            "role": "user",
                                            "content": "Warning: No Python code blocks were found in the response. Please include Python code in your replies.",
                                        }
                                    )
                                    continue

                                code = dedent(python_blocks[0])

                                execution_res = await computer.send_shell_command(
                                    f"python -c {shlex.quote(code)}"
                                )
                                execution_output = execution_res.output.decode(
                                    "utf-8", errors="replace"
                                )
                    except asyncio.TimeoutError:
                        execution_output = (
                            f"Timeout: The code took too longer than {timeout} seconds to execute."
                        )
                        ctx_logger.info(
                            "Agent code timed out",
                            destinations=["run"],
                        )

                    if execution_output is None or execution_output == "":
                        execution_output = "No output or errors from the code. If you think you're done, send 'DONE' and we will grade your submission. Otherwise, continue working."

                    # Append the code and its output to the messages
                    messages.append(
                        {
                            "role": "user",
                            "content": f"{execution_output}\nTurns left: {remaining_turns - 1}",
                        }
                    )

                # 2. Grade and yield the final result
                try:
                    grade = await task.grade(computer, self.runtime_config)
                except Exception as e:
                    raise RolloutSystemError(f"Error during grading: {e}") from e

                yield FinalResult(grade=grade)
        except asyncio.TimeoutError as e:
            ctx_logger.exception("Computer startup timed out", destinations=["run"])
            raise RolloutSystemError("Computer startup timed out") from e
        except Exception as e:
            if isinstance(e, RolloutSystemError):
                ctx_logger.exception(
                    f"RolloutSystemError in {task.run_id}: {e}", destinations=["run"]
                )
                raise
            ctx_logger.exception(
                f"Unexpected error in task {task.run_id}: {e}", destinations=["run"]
            )
            raise RolloutSystemError(f"Unexpected error during rollout: {e}") from e


ADDITIONAL_AGENT_PROMPT = (
    """The repository is cloned in your CWD. You must send Python code in backticks in each response to me, and I will execute the code and send you back the result, for example:

```python
import os
print(os.listdir())
```
You can include other content in your message, as long as it has the Python code block. You must only submit one code block per message.
The python code block is your means of interacting with the environment; write Python code to view files, edit files, and run commands. Do not leave any steps incomplete. You must implement all fixes yourself.
When you are done, send "DONE" (and nothing else - you don't need a python code block), and we will grade your submission.
Please note that the Python code is not a Jupyter notebook; you must write a full Python script each time, including imports, and printing things if you want to see the output.
"""
    + """
You may find it useful to search for particular keywords in the codebase using the `grep` command via subprocess:
```python
import subprocess
print(subprocess.check_output(['grep', '-irn', '"foo"', '/app/expensify"]))
```
where `"foo"` is the keyword you want to search for.

Remember, you can go step by step. If you are unsure about a command, use a step to look up help e.g.
```python
import subprocess
print(subprocess.check_output(['grep', '--help']))
```

You should start by browsing the files in the codebase to understand the problem and formulate a plan.
"""
)
