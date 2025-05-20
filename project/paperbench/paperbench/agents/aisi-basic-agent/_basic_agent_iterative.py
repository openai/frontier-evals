import asyncio
import copy
import os
import time
from json import JSONDecodeError
from logging import getLogger
from typing import Callable, cast

from inspect_ai._util.format import format_progress_time
from inspect_ai.model import GenerateConfig
from inspect_ai.model._cache import CachePolicy
from inspect_ai.model._call_tools import call_tools
from inspect_ai.model._chat_message import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from inspect_ai.model._model import get_model
from inspect_ai.scorer._metric import Score, ValueToFloat, value_to_float
from inspect_ai.scorer._score import score
from inspect_ai.solver._chain import chain
from inspect_ai.solver._prompt import system_message
from inspect_ai.solver._solver import Generate, Solver, solver
from inspect_ai.solver._task_state import TaskState
from inspect_ai.solver._use_tools import use_tools
from inspect_ai.tool._tool import Tool, ToolResult, tool
from inspect_ai.tool._tool_with import tool_with
from openai import LengthFinishReasonError
from typing_extensions import TypedDict, Unpack
from utils import generate_patched, prune_messages

logger = getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = """You are a helpful agent attempting to solve a task.
- You have several functions available to help with solving the task. Each message may perform one function call. You will see the result of the function right after sending the message. If you need to perform multiple actions, you can always send more messages with subsequent function calls.
- Do some reasoning before your actions, describing what function calls you are going to use and how they fit into your plan.
- Use as much of the available time as possible to refine and optimize your solution: Your goal is to achieve the best possible score, not just to submit your first working solution. Keep working until the time limit expires.
- The file read tool is paginated, so you will need to call it multiple times to make sure that you have read all the relevant parts of the paper.
- This is a multi-step task. DO THINGS BIT BY BIT by slowly building up your solution with multiple tool calls, don't try to do everything in one go. Just reason about what is the best next step and execute it. Then take the next step after you receive the output of the previous step.
- You should try and create a clean codebase. When it makes sense you should edit/overwrite files rather than creating new ones.
- When writing code, do not write newlines as the string "\n", instead use the newline character."""

DEFAULT_INCORRECT_MESSAGE = """
Your submission was incorrect. Please proceed and attempt to find the correct answer.
"""
DEFAULT_CONTINUE_MESSAGE = (
    "ERROR; please use the available tools to write and execute the necessary code."
)
DEFAULT_SUBMIT_NAME = "end_task"
DEFAULT_SUBMIT_DESCRIPTION = (
    "Signal that you are completely finished and that you would like to end the task."
)

CONTINUE_USER_MESSAGE = """Now given the previous progress made by the agent, take the next step towards replicating the paper:
- You have a lot of time available, so don't try and do everything in one go. Just reason about what is the best next step and execute it.
- Use the available tools to write and execute the necessary code.
- Remember, you should try prioritize the most important parts of the paper to replicate first."""

CONTINUE_USER_MESSAGE_CODE_ONLY = """Now given the previous progress made by the agent, take the next step towards replicating the paper:
- You have a lot of time available, so don't try and do everything in one go. Just reason about what is the best next step and execute it.
- Use the available tools to write the necessary code.
- Remember, you should try prioritize the most important parts of the paper to replicate first."""


class BasicAgentDeprecatedArgs(TypedDict, total=False):
    max_messages: int | None


@solver
def basic_agent_iterative(
    *,
    init: Solver | list[Solver] | None = None,
    tools: list[Tool] | Solver | None = None,
    cache: bool | CachePolicy = False,
    max_attempts: int = 1,
    message_limit: int | None = None,
    token_limit: int | None = None,
    real_time_limit: int | None = None,
    max_tool_output: int | None = None,
    score_value: ValueToFloat | None = None,
    incorrect_message: str | Callable[[TaskState, list[Score]], str] = DEFAULT_INCORRECT_MESSAGE,
    continue_message: str = DEFAULT_CONTINUE_MESSAGE,
    submit_name: str = DEFAULT_SUBMIT_NAME,
    submit_description: str = DEFAULT_SUBMIT_DESCRIPTION,
    disallow_submit: bool = False,
    **kwargs: Unpack[BasicAgentDeprecatedArgs],
) -> Solver:
    """Basic ReAct agent.

    Agent that runs a tool use loop. Tailor the model's instructions by passing a `system_message()` and/or other steps to `init` (if no `init` is specified then a default system
    message will be used). Use `max_attempts` to support additional submissions if
    the initial submission(s) are incorrect.

    Submissions are evaluated using the task's main scorer, with value of 1.0
    indicating a correct answer. Scorer values are converted to float (e.g.
    "C" becomes 1.0) using the standard value_to_float() function. Provide an
    alternate conversion scheme as required via `score_value`.

    Args:
       init: (Solver | list[Solver] | None): Agent initialisation
         (defaults to system_message with basic ReAct prompt)
       tools (list[Tool | ToolDef] | Solver | None): Tools available for the agent. Either a
         list of tools or a Solver that can yield dynamic tools per-sample.
       cache: (bool | CachePolicy): Caching behaviour for generate responses
         (defaults to no caching).
       max_attempts (int): Maximum number of submissions to accept before terminating.
       message_limit (int | None): Limit on messages in sample before terminating agent.
          If not specified, will use limit_messages defined for the task. If there is none
          defined for the task, 50 will be used as a default.
       token_limit (int | None): Limit on tokens used in sample before terminating agent.
       max_tool_output (int | None): Maximum output length (in bytes).
          Defaults to max_tool_output from active GenerateConfig.
       score_value (ValueToFloat): Function used to extract float from scores (defaults
         to standard value_to_float())
       incorrect_message (str | Callable[[TaskState, list[Score]], str]): User message reply for an
         incorrect submission from the model. Alternatively, a function which returns a message.
       continue_message (str): User message to urge the model to continue when it
         doesn't make a tool call.
       **kwargs (Any): Deprecated arguments for backward compatibility.

    Returns:
        Plan for agent.
    """
    # resolve deprecated
    for arg, value in kwargs.items():
        if arg == "max_messages":
            # deprecated, don't warn yet
            message_limit = int(cast(int, value))

    # resolve init
    if init is None:
        init = system_message(DEFAULT_SYSTEM_MESSAGE)
    init = init if isinstance(init, list) else [init]

    continue_user_message = (
        CONTINUE_USER_MESSAGE_CODE_ONLY
        if os.environ.get("PB_CODE_ONLY", "false") == "true"
        else CONTINUE_USER_MESSAGE
    )

    # resolve tools
    if tools is None:
        tools = []
    tools = tools if isinstance(tools, Solver) else use_tools(tools)

    # resolve score_value function
    score_value_fn = score_value or value_to_float()

    # submission tool
    @tool
    def end_task() -> Tool:
        async def execute(end_message: str) -> ToolResult:
            """Signal that you are completely finished.

            Args:
              end_message (str): Final message to signal that you are finished.
            """
            return end_message

        return execute

    # solver that adds submission tool
    @solver
    def end_task_tool() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            state.tools.append(tool_with(end_task(), submit_name, submit_description))
            return state

        return solve

    # helper to extract a submitted answer
    def submission(tool_results: list[ChatMessageTool]) -> str | None:
        return next(
            (result.text for result in tool_results if result.function == submit_name),
            None,
        )

    # main agent loop
    @solver
    def basic_agent_loop() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # resolve message_limit -- prefer parameter then fall back to task
            state.message_limit = message_limit or state.message_limit

            # resolve token limit
            state.token_limit = token_limit or state.token_limit

            # track attempts
            attempts = 0

            num_steps = 0
            start_time = time.time()
            model = get_model()
            setattr(model, "total_retry_time", 0)
            setattr(model, "generate", generate_patched)
            if "o3" in model.api.model_name or "o4" in model.api.model_name:
                model.api.responses_api = True

            # main loop (state.completed checks message_limit and token_limit)
            while not state.completed:
                # Proactively prune messages if we're approaching 1000 messages for Claude 3.5 Sonnet on Anthropic API
                if "sonnet" in model.name.lower() and len(state.messages) > 900:
                    state.messages = prune_messages(state.messages)

                # Finish depending on step and time limits. If both are set, both must be met.
                num_steps += 1

                # NOTE log.warning is necessary here for this to appear in inspect.log, which is necessary so we can know the agents runtime
                logger.warning(
                    f"total runtime: {round(time.time() - start_time, 2)}, total productive runtime: {round(time.time() - start_time - model.total_retry_time, 2)}, total retry time: {round(model.total_retry_time, 2)}"
                )

                over_time_limit = (
                    (time.time() - start_time - model.total_retry_time > real_time_limit)
                    if real_time_limit is not None
                    else False
                )
                if real_time_limit is not None and over_time_limit:
                    state.completed = True
                    break

                # Send progress update every 5 steps
                if num_steps % 5 == 0:
                    if real_time_limit is not None:
                        elapsed_time = time.time() - start_time - model.total_retry_time
                        periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed out of {format_progress_time(real_time_limit)}. Remember, you only have to stop working when the time limit has been reached."
                    else:
                        elapsed_time = time.time() - start_time
                        periodic_msg = f"Info: {format_progress_time(elapsed_time)} time elapsed"
                    periodic_msg += "\n\nNote: Don't forget to git commit regularly!"
                    state.messages.append(ChatMessageUser(content=periodic_msg))

                length_finish_error = False
                prune_individual = False
                try:
                    # Re-prompt model with instructions, but do not include these instructions in the conversation history
                    conversation = copy.deepcopy(state.messages) + [
                        ChatMessageUser(content=continue_user_message)
                    ]

                    # Calculate timeout in seconds for model.generate if real_time_limit is set
                    generate_timeout = None
                    if real_time_limit is not None:
                        generate_timeout = int(
                            real_time_limit - (time.time() - start_time - model.total_retry_time)
                        )

                    state.output = await model.generate(
                        self=model,
                        input=conversation,
                        tools=state.tools,
                        cache=cache,
                        config=GenerateConfig(timeout=generate_timeout),
                    )
                    state.messages.append(state.output.message)
                except (LengthFinishReasonError, IndexError) as e:
                    length_finish_error = True
                    prune_individual = False
                    if "PRUNE_INDIVIDUAL_MESSAGES" in str(e):
                        prune_individual = True
                except JSONDecodeError:
                    state.messages.append(ChatMessageUser(content="The JSON returned was invalid."))
                    continue

                if length_finish_error or state.output.stop_reason == "model_length":
                    logger.warning("context length overflow")
                    state.messages = prune_messages(
                        state.messages, prune_individual=prune_individual
                    )
                    continue

                # resolve tools calls (if any)
                if state.output.message.tool_calls:
                    # For each tool call, use timeout equal to the time remaining on this task
                    timeout = None
                    if real_time_limit is not None:
                        timeout = int(
                            real_time_limit - (time.time() - start_time - model.total_retry_time)
                        )

                    # call tool functions
                    try:
                        async with asyncio.timeout(timeout):
                            tool_results = await call_tools(
                                state.output.message, state.tools, max_output=max_tool_output
                            )
                    except asyncio.TimeoutError:
                        state.messages.append(
                            ChatMessageUser(content="Timeout: The tool call timed out.")
                        )
                        state.completed = True
                        break
                    state.messages.extend(tool_results)
                # no tool calls, urge the model to continue
                else:
                    state.messages.append(ChatMessageUser(content=continue_message))

            return state

        return solve

    # return chain
    return chain(
        init
        + [
            tools,
            basic_agent_loop(),
        ]
    )
