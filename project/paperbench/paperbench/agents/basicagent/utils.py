import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import tiktoken
from openai.types.chat import (
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
)
from preparedness_turn_completer.type_helpers import is_chat_completion_assistant_message_param
from preparedness_turn_completer.utils import text_from_completion
from tiktoken.core import Encoding

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.agents.basicagent.completer import BasicAgentTurnCompleterConfig
from paperbench.agents.basicagent.logging import LoggableMessages
from paperbench.agents.basicagent.prompts.templates import (
    additional_notes_template,
    gpu_template,
    no_gpu_template,
    no_time_limit_template,
    time_limit_template,
)
from paperbench.agents.basicagent.tools.base import Tool, ToolCall
from paperbench.agents.upload import upload_heavy_logs
from paperbench.constants import AGENT_DIR_CONFIG, WORKSPACE_BASE
from paperbench.nano.task import PBTask
from paperbench.utils import get_root

logger = structlog.stdlib.get_logger(component=__name__)


async def cancel_async_task(async_task: asyncio.Task[Any] | None, pb_task: PBTask) -> None:
    """Cancel *async_task* and wait for it to finish.

    Any exception other than ``asyncio.CancelledError`` raised by the task is
    logged.
    """
    if async_task is None or async_task.done():
        return

    ctx_logger = logger.bind(
        run_group_id=pb_task.run_group_id,
        runs_dir=pb_task.runs_dir,
        run_id=pb_task.run_id,
        destinations=["run"],
    )
    async_task.cancel()
    try:
        await asyncio.shield(async_task)
    except asyncio.CancelledError:
        pass
    except Exception as e:  # pragma: no cover - exercised in tests
        ctx_logger.exception(f"Task failed with error: {e}")


def format_progress_time(time: float, pad_hours: bool = True) -> str:
    """
    Format seconds into hours, minutes, and seconds.
    If pad_hours is True, pad the hours with a leading zero.
    """
    minutes, seconds = divmod(time, 60)
    hours, minutes = divmod(minutes, 60)
    hours_fmt = f"{hours:2.0f}" if pad_hours else f"{hours:.0f}"
    return f"{hours_fmt}:{minutes:02.0f}:{seconds:02.0f}"


async def should_upload_periodic_heavy_logs(
    num_steps: int,
    upload_interval_messages: int | None,
    upload_interval_seconds: int | None,
    last_time_uploaded: float,
) -> bool:
    """
    Determines whether it is time to upload heavy logs based on step and/or time
    intervals.
    Note that time-based intervals always use wall-clock time, not real time
    """
    at_step_interval = (
        upload_interval_messages is not None and num_steps % upload_interval_messages == 0
    )
    at_time_interval = False
    if upload_interval_seconds is not None:
        at_time_interval = time.time() > last_time_uploaded + upload_interval_seconds
    return at_step_interval or at_time_interval


@dataclass(frozen=True)
class OptionalUploadOutcome:
    last_time_uploaded: float
    upload_task: asyncio.Task[None] | None


async def optionally_upload_heavy_logs(
    computer: ComputerInterface,
    task: PBTask,
    run_dir: str,
    num_steps: int,
    start_time: int,
    last_time_uploaded: float,
    upload_interval_messages: int | None,
    upload_interval_seconds: int | None,
    always_upload: bool = False,
) -> OptionalUploadOutcome:
    """
    Upload heavy logs if the agent has reached the step limit or the time limit. The uploading
    is non-blocking
    """
    if (
        not await should_upload_periodic_heavy_logs(
            num_steps=num_steps,
            upload_interval_messages=upload_interval_messages,
            upload_interval_seconds=upload_interval_seconds,
            last_time_uploaded=last_time_uploaded,
        )
        and not always_upload
    ):
        return OptionalUploadOutcome(last_time_uploaded, None)

    # make upload non-blocking
    upload_task = asyncio.create_task(
        upload_heavy_logs(
            computer=computer,
            agent_start_time=start_time,
            agent_dir_config=AGENT_DIR_CONFIG,
            run_dir=run_dir,
            run_group_id=task.run_group_id,
            runs_dir=task.runs_dir,
            run_id=task.run_id,
            num_messages=num_steps,
        )
    )
    last_time_uploaded = time.time()
    # Snap to start of current wall-clock interval to keep future uploads aligned (avoid drift)
    if upload_interval_seconds is not None:
        last_time_uploaded = (time.time() // upload_interval_seconds) * upload_interval_seconds
    return OptionalUploadOutcome(last_time_uploaded, upload_task)


async def get_gpu_generation(computer: ComputerInterface) -> str | None:
    """Return the GPU generation available on ``computer``."""
    try:
        result = await computer.send_shell_command(
            "nvidia-smi --query-gpu=name --format=csv,noheader"
        )
    except Exception:
        return None
    if result.exit_code != 0:
        return None

    generation = result.unicode_output_best_effort.strip().split("\n")
    if not generation or not generation[0]:
        return None
    return ", ".join([info.strip() for info in generation])


async def get_instructions(
    computer: ComputerInterface,
    task: PBTask,
    iterative_agent: bool,
    time_limit: float | None,
) -> str:
    """
    Prepares instructions for the agent. The instructions vary depending on whether:
    - The task is the default task or code-only
    - The agent is the default agent or iterative
    """
    if iterative_agent:
        if task.judge.code_only:
            instructions_path = (
                Path(__file__).parent.resolve() / "prompts" / "code_only_instructions_iterative.txt"
            )
        else:
            instructions_path = (
                Path(__file__).parent.resolve() / "prompts" / "instructions_iterative.txt"
            )
    else:
        if task.judge.code_only:
            instructions_path = get_root() / "instructions" / "code_only_instructions.txt"
        else:
            instructions_path = get_root() / "instructions" / "instructions.txt"

    instructions = instructions_path.read_text()

    # add additional notes
    type_of_processor = await get_gpu_generation(computer)
    if type_of_processor:
        type_of_processor_str = gpu_template.format(type_of_processor=type_of_processor)
    else:
        type_of_processor_str = no_gpu_template
    if time_limit:
        time_limit_in_hours = round(time_limit / 3600, 3)
        time_limit_str = time_limit_template.format(max_time_in_hours=time_limit_in_hours)
    else:
        time_limit_str = no_time_limit_template.format()
    instructions += additional_notes_template.format(
        type_of_processor_str=type_of_processor_str,
        time_limit_str=time_limit_str,
        workspace_base=WORKSPACE_BASE,
    )
    return instructions


def _handle_message_len(
    msg: ChatCompletionMessageParam, tokenizer: Encoding, max_tokens: int
) -> ChatCompletionMessageParam:
    """
    Truncates the content of a message to fit within max_tokens.
    Mostly ported verbatim from AISI basic agent implementation.
    """

    def truncate_string(input_string: str, input_tokens: list[int], max_tokens: int) -> str:
        n_tokens = len(input_tokens)
        if n_tokens > max_tokens:
            keep_tokens = max_tokens // 2
            first_half = tokenizer.decode(input_tokens[:keep_tokens])
            second_half = tokenizer.decode(input_tokens[-keep_tokens:])
            return first_half + "\n...[content truncated due to length]...\n" + second_half
        return input_string

    if isinstance(msg["content"], str):
        item_tokens = tokenizer.encode(msg["content"], disallowed_special=())
        msg["content"] = truncate_string(msg["content"], item_tokens, max_tokens)
    elif isinstance(msg["content"], list):
        token_lists: list[list[int]] = []  # tokenization of each message
        token_counts: list[int] = []  # number of tokens in each message
        for item in msg["content"]:
            if item["type"] == "text":
                item_tokens = tokenizer.encode(item["text"], disallowed_special=())
                token_lists.append(item_tokens)
                token_counts.append(len(item_tokens))
            else:
                # Non-text content types don't count towards token limit
                token_lists.append([])
                token_counts.append(0)

        total_tokens = sum(token_counts)
        if total_tokens == 0:
            return msg  # No text content to truncate edge case

        # Distribute max_tokens proportionally for text content
        tokens_per_item = [
            max(1, int((count / total_tokens) * max_tokens)) if count > 0 else 0
            for count in token_counts
        ]

        # Apply truncation while preserving content type
        new_content: list[ChatCompletionContentPartTextParam] = []
        for item, item_tokens, max_tokens_for_item in zip(
            msg["content"], token_lists, tokens_per_item
        ):
            if item["type"] == "text":
                item["text"] = truncate_string(item.text, item_tokens, max_tokens_for_item)
            new_content.append(item)

        msg["content"] = new_content

    return msg


def prune_messages(
    messages: LoggableMessages[ChatCompletionMessageParam],
    prune_individual: bool = False,
) -> LoggableMessages[ChatCompletionMessageParam]:
    """Prune messages to stay within API limits.

    Removes older messages while preserving conversation coherence by:
    - Keeping all system messages
    - Keeping the initial task instruction message
    - Removing the oldest 30% of conversation messages
    - Ensuring tool messages remain paired with their parent assistant messages
    Mostly ported verbatim from AISI basic agent implementation.

    Args:
        messages: List of chat messages to prune

    Returns:
        LoggableMessages: List of messages with preserved conversation flow
    """

    # Split messages into system and conversation parts
    system_msgs: list[ChatCompletionMessageParam] = [
        m for m in messages.data if "role" in m and m["role"] == "system"
    ]
    non_system_msgs = [m for m in messages.data if not ("role" in m and m["role"] == "system")]

    # Always preserve the first user message (task instructions)
    task_msg = next((m for m in non_system_msgs if "role" in m and m["role"] == "user"), None)

    # Remove oldest 30% of messages
    start_idx = max(1, int(len(non_system_msgs) * 0.3))  # Keep 70%

    preserved: list[ChatCompletionMessageParam] = [task_msg] if task_msg else []
    preserved.extend(non_system_msgs[start_idx:])
    conversation = preserved

    # OAI API requires any messages with `msg.role == "tool"` to be preceded by
    # an assistant message with corresponding tool_calls. Our pruning may violate this,
    # so we need to clean up tool messages that lost their parent assistant message
    valid_messages: list[ChatCompletionMessageParam] = []
    active_tool_ids = set()  # IDs from most recent assistant's tool_calls

    for msg in conversation:
        role = msg["role"]
        content_str = text_from_completion(msg)
        if "prompt is too long" in content_str:
            continue
        if role == "assistant":
            assert is_chat_completion_assistant_message_param(msg)
            # Track any tool calls from this assistant message
            active_tool_ids = {tc["id"] for tc in (msg.get("tool_calls") or [])}
            valid_messages.append(msg)
        elif role == "tool" and getattr(msg, "tool_call_id", None) in active_tool_ids:
            # Keep tool messages only if they match an active tool call
            valid_messages.append(msg)
        elif role == "user":
            # Reset tool tracking at user messages & keep the message
            active_tool_ids = set()
            valid_messages.append(msg)

    if prune_individual:
        # ensure individual messages are not over the context limit
        MAX_TOKENS_PER_MESSAGE = 190_000  # buffer under 200k context
        # use OAI's 200k tokenizer as an approximation
        tokenizer: Encoding = tiktoken.get_encoding("o200k_base")

        valid_messages = [
            _handle_message_len(msg, tokenizer, MAX_TOKENS_PER_MESSAGE) for msg in valid_messages
        ]

    # Reconstruct pruned conversation
    pruned_messages = system_msgs + valid_messages
    pruned_messages_loggable = LoggableMessages(
        pruned_messages, log_path=messages._log_path, logger=messages._logger
    )
    return pruned_messages_loggable


def handle_sonnet_limits(
    completer_config: BasicAgentTurnCompleterConfig,
    messages: LoggableMessages[ChatCompletionMessageParam],
) -> LoggableMessages[ChatCompletionMessageParam]:
    """
    Prunes messages if using a sonnet model and exceeding limits on number of messages.
    """
    if (
        hasattr(completer_config, "model")
        and "sonnet" in completer_config.model
        and len(messages) > 900
    ):
        messages = prune_messages(messages)
    return messages


async def handle_tool_call(
    basic_agent_tool_call: ToolCall, tools: list[Tool], task: PBTask, computer: ComputerInterface
) -> ChatCompletionToolMessageParam | None:
    """
    Parses and executes a tool call from the agent.
    Returns None if the tool call is "submit", otherwise returns the tool response.
    """
    ctx_logger = logger.bind(
        run_group_id=task.run_group_id, run_id=task.run_id, runs_dir=task.runs_dir
    )
    if basic_agent_tool_call.name == "submit":
        ctx_logger.info("Agent used submit tool. Finishing run....", destinations=["run"])
        return None
    else:
        tool = next(t for t in tools if t.name() == basic_agent_tool_call.name)
        try:
            result = await tool.execute(computer, **basic_agent_tool_call.arguments)
        except Exception as e:
            result = f"Error executing tool: {e}"
        return {
            "role": "tool",
            "content": str(result),
            "tool_call_id": basic_agent_tool_call.call_id,
        }
