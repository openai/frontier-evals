from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias

import openai
import structlog.stdlib
import tiktoken
from dotenv import load_dotenv

load_dotenv()
from openai.types import CompletionUsage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)
from preparedness_turn_completer.turn_completer import TurnCompleter
from pydantic import BaseModel
from typing_extensions import override

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.judge.base import Judge
from paperbench.judge.constants import (
    CRITERION_PROMPT,
    FILE_RANKING_PROMPT,
    GRADING_PROMPT,
    build_judge_task_prompt,
)
from paperbench.judge.graded_task_node import GradedTaskNode
from paperbench.judge.token_usage import TokenUsage
from paperbench.judge.utils import format_file, read_file_content, walk_dir_with_mtimes
from paperbench.rubric.tasks import TASK_CATEGORY_QUESTIONS, TaskNode

logger = structlog.stdlib.get_logger(component=__name__)

FileTree: TypeAlias = dict[str, "FileTree"]


class ParsedJudgeResponseFloat(BaseModel):
    valid_score: bool
    score: float
    explanation: str


class ParsedJudgeResponseInt(BaseModel):
    valid_score: bool
    score: int
    explanation: str


class ParseError(Exception):
    pass


@dataclass
class TreePrepOutcome:
    tree_structure: str
    within_token_budget: bool


class SimpleJudge(Judge):
    def __init__(
        self,
        paper_path: Path,
        rubric: TaskNode,
        addendum: str | None,
        judge_addendum: str | None,
        submission_dir: Path,
        paper_md: Path,
        completer_config: TurnCompleter.Config,
        int_completer_config: OpenAICompletionsTurnCompleter.Config | None = None,
        float_completer_config: OpenAICompletionsTurnCompleter.Config | None = None,
        log_path: Path | None = None,
        buffer_tokens: int = 10000,  # 10k tokens of buffer
        max_depth: int = 999,
        code_only: bool = False,
        max_prior_nodes: int | None = None,
        max_file_depth: int = 4,
        computer: ComputerInterface | None = None,
    ):
        super().__init__(
            paper_path=paper_path,
            rubric=rubric,
            addendum=addendum,
            judge_addendum=judge_addendum,
            submission_dir=submission_dir,
            log_path=log_path,
            max_depth=max_depth,
            code_only=code_only,
            computer=computer,
        )

        self.completer_config = completer_config
        self.completer = completer_config.build()
        self.token_encoder = tiktoken.get_encoding(self.completer.encoding_name)

        self.float_completer_conf, self.float_completer = self._init_structured_completer(
            float_completer_config, ParsedJudgeResponseFloat
        )
        self.int_completer_conf, self.int_completer = self._init_structured_completer(
            int_completer_config, ParsedJudgeResponseInt
        )

        self.paper_md = paper_md.read_text()
        self.rubric = rubric
        self.prompt = build_judge_task_prompt(code_only)
        self.buffer_tokens = buffer_tokens
        self.joined_addendum = f"{self.addendum if self.addendum else ''}\n{self.judge_addendum if self.judge_addendum else ''}".strip()
        self.leaf_semaphore = asyncio.Semaphore(100)
        self.max_prior_nodes = max_prior_nodes
        if self.joined_addendum == "":
            self.joined_addendum = "(NO ADDENDUM GIVEN)"
        self.reproduce_touched_files = True  # by default assume reproduce was functional
        self.max_file_depth = max_file_depth

    def _init_structured_completer(
        self, config: TurnCompleter.Config | None, response_format: type[BaseModel]
    ) -> tuple[TurnCompleter.Config, TurnCompleter]:
        """
        if `config` is provided, it is assumed that it will use `response_format` internally.
        If `config` is not provided,
        we fallback to a default OpenAICompletionsStructuredCompleter config
        """
        cfg = config or OpenAICompletionsTurnCompleter.Config(
            model="gpt-4o-2024-08-06",
            response_format=response_format,
        )
        return cfg, cfg.build()

    async def process_file_content(self) -> None:
        """
        Pre-emptively truncates reproduce.log, paper.md and the content of the files
        in the codebase to avoid running into context length issues downstream
        """
        # pre-emptively truncate the reproduce.log and paper.md (latter almost never happens)
        # to allow for space for additional context when prompting
        self.reproduce_log_tokens = self.token_encoder.encode(
            self.reproduce_log_content, disallowed_special=()
        )
        self.paper_md_tokens = self.token_encoder.encode(self.paper_md, disallowed_special=())
        self._truncate_input()

        self.avail_context_lens: dict[str, int] = {
            "Code Development": self._get_available_context("Code Development"),
            "Code Execution": self._get_available_context("Code Execution"),
            "Result Analysis": self._get_available_context("Result Analysis"),
            "Subtree": self._get_available_context("Subtree"),
        }

        self.tree_structures = {
            k: await self._prepare_tree_structure(k)
            for k in ["Code Development", "Code Execution", "Result Analysis", "Subtree"]
        }

    async def before_grading(self) -> None:
        await super().before_grading()
        await self.process_file_content()

    def _truncate_in_token_space(self, input_str: str, max_length_tokens: int) -> str:
        input_tokens = self.token_encoder.encode(input_str, disallowed_special=())
        truncated_tokens = input_tokens[:max_length_tokens]
        return self.token_encoder.decode(truncated_tokens)

    def _get_available_context(self, task_category: str) -> int:
        """number of input tokens available for use for each category"""
        reserved_context_lens = {
            "Code Development": len(self.paper_md_tokens),
            "Code Execution": len(self.paper_md_tokens) + len(self.reproduce_log_tokens),
            "Result Analysis": len(self.paper_md_tokens) + len(self.reproduce_log_tokens),
            "Subtree": len(self.paper_md_tokens) + len(self.reproduce_log_tokens),
        }
        model_context_length = self.completer.n_ctx

        return model_context_length - (reserved_context_lens[task_category] + self.buffer_tokens)

    def _truncate_input(self) -> None:
        """
        Truncates reproduce.log and paper.md until there is leeway for prompting.
        Truncates log files to be half of the context window length.
        e.g. 128k context window -> 64k token reproduce.log limit
        Assumes log reduction via reduce_log() has already been applied

        Further truncates log and paper until theres at least 5k tokens of space left
        Prioritizing log truncation over paper truncation
        """
        context_window_tokens = self.completer.n_ctx
        half_context_window = context_window_tokens // 2
        five_k_tokens = 5000

        # initial truncation
        self.reproduce_log_tokens = self.reproduce_log_tokens[:half_context_window]

        # further truncate the log if we're still over
        token_consumption = len(self.reproduce_log_tokens) + len(self.paper_md_tokens)
        avail_context = context_window_tokens - token_consumption
        if avail_context < 0:
            logger.warning("Paper + log content exceeds context window. Truncating log.")
            self.reproduce_log_tokens = self.reproduce_log_tokens[: avail_context - five_k_tokens]

        # if we're still over (reproduce.log wasnt the culprit), truncate the paper
        token_consumption = len(self.reproduce_log_tokens) + len(self.paper_md_tokens)
        avail_context = context_window_tokens - token_consumption
        if avail_context < 0:
            logger.warning("Paper + log content still exceeds context window. Truncating paper.")
            self.paper_md_tokens = self.paper_md_tokens[: avail_context - five_k_tokens]

        # update the content strings
        self.reproduce_log_content = self.token_encoder.decode(self.reproduce_log_tokens)
        self.paper_md = self.token_encoder.decode(self.paper_md_tokens)

    @property
    def judge_type(self) -> str:
        return "simple"

    def _create_tree_structure(self, files: list[Path]) -> str:
        """Creates a tree-like structure visualization of files."""
        tree: FileTree = {}
        for file in files:
            current = tree
            for part in file.parts:
                if part not in current:
                    current[part] = {}
                current = current[part]

        def _build_tree(node: FileTree, prefix: str = "") -> str:
            lines = []
            items = list(node.items())

            for i, (name, subtree) in enumerate(items):
                is_last_item = i == len(items) - 1
                connector = "└── " if is_last_item else "├── "
                lines.append(f"{prefix}{connector}{name}")

                if subtree:
                    extension = "    " if is_last_item else "│   "
                    subtree_lines = _build_tree(subtree, prefix + extension)
                    lines.append(subtree_lines)
            return "\n".join(lines)

        return _build_tree(tree)

    async def _get_whitelisted_files(
        self, task_category: str, max_file_depth: int | None = None
    ) -> list[Path]:
        """
        Returns any files in the codebase that are plaintext and relevant for the task.
        For code development and execution, docs and code are relevant.
        For result analysis, docs and tables are relevant.

        Note: this is unrelated to reproduce.sh and reproduce.log, which are handled separately.
        """
        # fmt: off
        blacklisted_base_dirs = {
            "venv", ".venv", ".env", "wandb", ".egg-info", ".git", ".github",
            "__pycache__", "node_modules",
        }
        whitelisted_docs = {".md", ".txt", ".rst"}
        whitelisted_code = {
            '.py', '.R', '.Rmd', '.m', '.jl',                              # common DS/ML langs
            '.c', '.h', '.cpp', '.hpp', '.cc', '.cxx', '.hxx',             # C/C++
            '.java', '.js', '.ts', '.scala', '.go', '.rs',                 # Other languages
            '.sh',                                                         # Shell
            '.config', '.cfg', '.json', '.yaml', '.yml', '.toml', '.ini'   # Config files
        }
        whitelisted_tables = {
            ".csv", ".tsv", ".psv", ".json", ".jsonl", ".html", ".xml", ".yaml", ".yml",
            ".toml", ".arff", ".tex", ".svm", ".libsvm"
        }
        # fmt: on

        extension_sets = {
            "Result Analysis": whitelisted_docs | whitelisted_tables,
            "Subtree": whitelisted_docs | whitelisted_code | whitelisted_tables,
            # Default for Code Development and Code Execution
            "default": whitelisted_docs | whitelisted_code,
        }
        whitelisted_extensions = extension_sets.get(task_category, extension_sets["default"])

        def should_include_file(path: Path, mtime: float) -> bool:
            if path.suffix not in whitelisted_extensions:
                return False

            if mtime != mtime:  # if mtime is nan, we can't trust it
                return False

            file_last_modified_time = datetime.fromtimestamp(mtime, tz=timezone.utc)

            if task_category == "Result Analysis":
                return (
                    path.suffix in whitelisted_docs
                    or file_last_modified_time >= self.reproduction_log_creation_time_utc
                )
            elif task_category == "Subtree":
                return (
                    path.suffix in whitelisted_docs
                    or path.suffix in whitelisted_code
                    or file_last_modified_time >= self.reproduction_log_creation_time_utc
                )
            else:
                return True

        whitelisted_files = []
        whitelisted_mtimes = []
        async for root, dirs, files, mtimes in walk_dir_with_mtimes(
            self.submission_dir, self.computer
        ):
            # Limit directory traversal based on max_file_depth
            current_depth = len(Path(root).relative_to(self.submission_dir).parts)
            if max_file_depth is not None and current_depth >= max_file_depth:
                dirs[:] = []  # stop traversing subdirectories if the depth limit is reached
            if any(
                blacklisted in part
                for blacklisted in blacklisted_base_dirs
                for part in Path(root).parts
            ):
                continue
            for file, mtime in zip(files, mtimes):
                full_path = Path(root) / file
                if full_path.suffix in whitelisted_extensions:
                    if should_include_file(full_path, mtime):
                        whitelisted_files.append(full_path)
                        whitelisted_mtimes.append(mtime)

        if task_category == "Result Analysis":
            mtimes_utc = [
                datetime.fromtimestamp(mtime, tz=timezone.utc) for mtime in whitelisted_mtimes
            ]
            if all(mtime < self.reproduction_log_creation_time_utc for mtime in mtimes_utc):
                self.reproduce_touched_files = False

        return whitelisted_files

    async def _attempt_preparing_tree_structure(
        self, task_category: str, max_depth: int | None = None
    ) -> TreePrepOutcome:
        whitelisted_files: list[Path] = await self._get_whitelisted_files(
            task_category, max_file_depth=max_depth
        )
        tree_structure: str = self._create_tree_structure(
            [p.relative_to(self.submission_dir) for p in whitelisted_files]
        )
        tree_structure_len = len(self.token_encoder.encode(tree_structure, disallowed_special=()))
        if tree_structure_len >= self.avail_context_lens[task_category]:
            return TreePrepOutcome(tree_structure=tree_structure, within_token_budget=False)
        return TreePrepOutcome(tree_structure=tree_structure, within_token_budget=True)

    async def _prepare_tree_structure(self, task_category: str) -> str:
        """
        Prepares the relevant tree directory structure for a given task category.
        Automatically limits file depth if necessary.
        Automatically truncates to the model context window if necessary.
        """
        # 1st try without limiting depth
        tree_attempt = await self._attempt_preparing_tree_structure(task_category)
        if tree_attempt.within_token_budget:
            return tree_attempt.tree_structure
        # 2nd attempt: limit depth to 4
        tree_attempt = await self._attempt_preparing_tree_structure(task_category, max_depth=4)
        if tree_attempt.within_token_budget:
            return tree_attempt.tree_structure
        # 3rd attempt: simply truncate the file strings, forcing 'success'
        truncated_tree_structure = self._truncate_in_token_space(
            tree_attempt.tree_structure, self.avail_context_lens[task_category]
        )
        return truncated_tree_structure

    async def _prepare_relevant_files(
        self,
        task: TaskNode,
        max_files: int | None = 10,
    ) -> str:
        leaf_logger = self.get_logger(task)
        """
        Returns the relevant files for judging the task.
        For everything except results analysis nodes:
            returns the entire codebase if it's within the model's context window (minus buffer_tokens)
            Otherwise has the model order files in the codebase by importance, and includes
            as many files as possible without exceeding the context window (minus buffer_tokens).
        For results analysis nodes:
            Returns all documentation, and any files in the submission directory that
            were touched (modified or created) during the reproduce.sh execution
            Context window is handled in the same way as above
        """
        tree_structure = self.tree_structures[task.task_category or "Subtree"]

        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": FILE_RANKING_PROMPT,
            },
            {
                "role": "user",
                "content": f"The paper is below:\n<paper>\n{self.paper_md}\n</paper>",
            },
            {
                "role": "user",
                "content": f"If included with the paper, you will now be shown an addendum which provides clarification for the paper and how to evaluate its reproduction:\n<addendum>\n{self.joined_addendum}\n</addendum>",
            },
            {
                "role": "user",
                "content": f"Here is the criterion that you are grading:\n<criterion>\n{task.requirements}\n</criterion>",
            },
            {
                "role": "user",
                "content": f"Here are the files in the submission attempt:\n\nDirectory structure:\n{tree_structure}\n\nNow return a list of the {str(max_files) + ' ' if max_files else ''}most relevant files in order of relevance (descending) to the resolution criteria, to be provided for your inspection. Your response must contain each filename separated by newlines, with each file containing the full path. Do not write anything else.",
            },
        ]
        model_response = await self.completer.async_completion(conversation=messages)
        selected_files = model_response.output_messages[0].content
        if selected_files is None:
            raise Exception("No response received from completer for file selection")
        leaf_logger.info(f"Model file selection raw output:\n{selected_files}")

        selected_files_tokens = []
        num_files = 0
        total_tokens = 0
        max_tokens = (
            self.avail_context_lens[task.task_category or "Subtree"] - 2000
        )  # Buffer of 2k tokens

        file_content_tasks = [
            read_file_content(
                self.submission_dir / rel_path.strip().strip("/"),
                self.computer,
            )
            for rel_path in selected_files.split("\n")[: max_files or None]
        ]

        file_contents: list[str | BaseException] = await asyncio.gather(
            *file_content_tasks, return_exceptions=True
        )

        for rel_path, content in zip(selected_files.split("\n"), file_contents):
            full_path = self.submission_dir / rel_path.strip()
            try:
                if isinstance(content, BaseException):
                    raise content
                file_content = format_file(full_path.relative_to(self.submission_dir), content)
                content_tokens = self.token_encoder.encode(
                    file_content + "\n\n", disallowed_special=()
                )

                # If this file would put us over the limit
                if total_tokens + len(content_tokens) > max_tokens:
                    # Truncate in token space
                    target_len = max_tokens - total_tokens
                    content_tokens = content_tokens[:target_len]
                    selected_files_tokens.extend(content_tokens)
                    num_files += 1
                    break

                selected_files_tokens.extend(content_tokens)
                num_files += 1
                total_tokens += len(content_tokens)

                if max_files and num_files >= max_files:
                    break

            except FileNotFoundError:
                leaf_logger.info(f"File {full_path} not found!")
            except IsADirectoryError:
                leaf_logger.info(f"File {full_path} is a directory!")
            except UnicodeDecodeError:
                leaf_logger.info(f"File {full_path} is not a text file!")
            except Exception as e:
                leaf_logger.info(f"File {full_path} is not readable! Error: {e}")

        # Decode once at the end, ensuring we end with complete lines
        return self.token_encoder.decode(selected_files_tokens).rsplit("\n", 1)[0]

    async def _construct_grade_leaf_messages(
        self, task: TaskNode
    ) -> list[ChatCompletionMessageParam]:
        relevant_files = await self._prepare_relevant_files(task)
        relevant_files_prompt = (
            f"Here are the most relevant files included in the submission attempt, concatenated:\n<files>\n{relevant_files}\n</files>"
            if task.task_category != "Result Analysis"
            else f"Here are the most relevant docs and the files touched (i.e. modified or created) during the reproduce.sh execution, concatenated:\n<files>\n{relevant_files}\n</files>"
        )

        relevant_rubric_nodes = task.get_prior_nodes(self.rubric, self.max_prior_nodes)
        relevant_rubric_context = ""
        for node in relevant_rubric_nodes:
            relevant_rubric_context += f" -> {node.requirements}\n"

        reproduce_files_messages: list[ChatCompletionMessageParam] = []
        if self.code_only:
            reproduce_files_messages = []
        elif task.task_category == "Code Development":
            reproduce_files_messages = [
                {
                    "role": "user",
                    "content": f"Here is the `reproduce.sh` provided in the submission, if any:\n<reproduce.sh>\n{self.reproduce_sh_content}\n</reproduce.sh>",
                }
            ]
        else:
            reproduce_files_messages = [
                {
                    "role": "user",
                    "content": f"Here is the `reproduce.sh` provided in the submission, if any:\n<reproduce.sh>\n{self.reproduce_sh_content}\n</reproduce.sh>",
                },
                {
                    "role": "user",
                    "content": f"Here is the `reproduce.log` provided in the submission, if any:\n<reproduce.log>\n{self.reproduce_log_content}\n</reproduce.log>",
                },
            ]

        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": self.prompt,
            },
            {
                "role": "user",
                "content": f"The paper is below:\n{self.paper_md}",
            },
            {
                "role": "user",
                "content": f"If included with the paper, you will now be shown an addendum which provides clarification for the paper and how to evaluate its reproduction:\n<addendum>\n{self.joined_addendum}\n</addendum>",
            },
            {
                "role": "user",
                "content": relevant_files_prompt,
            },
            *reproduce_files_messages,
            {
                "role": "user",
                "content": CRITERION_PROMPT.format(
                    preceding_criteria=relevant_rubric_context,
                    criterion=task.requirements,
                    task_category=task.task_category,
                    task_category_question=TASK_CATEGORY_QUESTIONS.get(
                        task.task_category,  # type: ignore
                        "Does the submission satisfy this criterion?",
                    ),
                ),
            },
            {
                "role": "user",
                "content": GRADING_PROMPT(continuous=(task.task_category == "Subtree")),
            },
        ]
        return messages

    @override
    async def grade_leaf(self, task: TaskNode) -> GradedTaskNode:
        async with self.leaf_semaphore:
            leaf_logger = self.get_logger(task)
            leaf_std_logger = leaf_logger._logger
            try:
                leaf_logger.info(f"Grading leaf: {task.requirements}")
                if task.task_category == "Result Analysis" and not self.reproduce_touched_files:
                    leaf_logger.info(
                        "reproduce.sh failed to modify or create any files."
                        " All result analysis tasks will be graded as 0."
                    )
                    graded_task_node = GradedTaskNode.from_task(
                        task,
                        score=0,
                        valid_score=True,
                        explanation="Reproduce.sh did not touch any files, so there are no reproduced results to analyze.",
                        judge_metadata=None,
                    )
                else:
                    judge_token_usage = None
                    messages = await self._construct_grade_leaf_messages(task)
                    response: TurnCompleter.Completion = await self.completer.async_completion(
                        conversation=messages
                    )

                    response_usage = response.usage if hasattr(response, "usage") else None
                    judge_token_usage = self._handle_usage(
                        self.completer, judge_token_usage, response_usage
                    )

                    model_response = response.output_messages[0].content
                    messages += [{"role": "assistant", "content": model_response}]

                    leaf_logger.info(f"model response: {model_response}")

                    continuous = task.task_category == "Subtree"
                    score_response, parse_usage = await self._parse_model_response(
                        model_response, continuous=continuous
                    )

                    parse_completer = self.float_completer if continuous else self.int_completer
                    judge_token_usage = self._handle_usage(
                        parse_completer, judge_token_usage, parse_usage
                    )

                    graded_task_node = GradedTaskNode.from_task(
                        task,
                        score=score_response.score,
                        valid_score=score_response.valid_score,
                        explanation=score_response.explanation,
                        judge_metadata={
                            "full_judge_response": model_response,
                            "token_usage": judge_token_usage.to_dict()
                            if judge_token_usage
                            else None,
                        },
                    )

                    # Dump full messages
                    if (
                        self.log_path
                        and leaf_std_logger is not None
                        and leaf_std_logger.handlers
                        and isinstance(leaf_std_logger.handlers[0], logging.FileHandler)
                    ):
                        log_file_path = leaf_std_logger.handlers[0].baseFilename
                        with open(
                            Path(log_file_path).parent / f"{task.id}_messages.jsonl", "w"
                        ) as f:
                            for message in messages:
                                f.write(json.dumps(message) + "\n")

                return graded_task_node
            finally:
                if leaf_std_logger is not None:
                    for handler in leaf_std_logger.handlers:
                        handler.close()
                        leaf_std_logger.removeHandler(handler)

    def _handle_usage(
        self,
        completer: TurnCompleter,
        existing_usage: TokenUsage | None,
        incoming_usage: CompletionUsage | None,
    ) -> TokenUsage | None:
        if isinstance(completer, OpenAICompletionsTurnCompleter):
            if existing_usage is None:
                existing_usage = TokenUsage()
            existing_usage.add_from_completion(completer.model, incoming_usage)

        return existing_usage

    @override
    async def grade_subtree(self, task: TaskNode) -> GradedTaskNode:
        logger.info(f"Grading subtree: {task.requirements}")

        def build_requirements_string(task: TaskNode, depth: int = 0) -> str:
            indent = "| " * depth
            requirements_str = f"{indent}{task.requirements} (weight: {task.weight})\n"
            for sub_task in task.sub_tasks:
                requirements_str += build_requirements_string(sub_task, depth + 1)
            return requirements_str

        requirements_string = build_requirements_string(task)

        leaf_shim = TaskNode(
            id=task.id,
            requirements=requirements_string,
            weight=task.weight,
            sub_tasks=[],
            task_category="Subtree",
        )
        graded_leaf_shim = await self.grade_leaf(leaf_shim)
        return graded_leaf_shim

    async def _parse_model_response(
        self, response: str | None, continuous: bool = False
    ) -> tuple[
        ParsedJudgeResponseFloat | ParsedJudgeResponseInt, openai.types.CompletionUsage | None
    ]:
        """Parses a model response as a `ParsedJudgeResponse`."""
        if response is None:
            raise ParseError("No response received")

        score_instruction = "(either 0 or 1)" if not continuous else "(between 0 and 1)"
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": f"You are given a response output from a judge which should contain a score and an explanation. Please parse the text into a structured object containing `valid_score` (boolean indicating whether the response contains a valid score), the `score` {score_instruction}, and an `explanation` (a short summary of the judge's reasoning). If the response does not contain a valid score, set `valid_score` to False and set the `score` to 0.0.",
            },
            {
                "role": "user",
                "content": response,
            },
        ]

        try:
            ParsedJudgeResponse = (
                ParsedJudgeResponseInt if not continuous else ParsedJudgeResponseFloat
            )
            completer = self.int_completer if not continuous else self.float_completer
            completion = await completer.async_completion(conversation=messages)

            usage = None
            if isinstance(completer, OpenAICompletionsTurnCompleter) and isinstance(
                completion, OpenAICompletionsTurnCompleter.Completion
            ):
                usage = completion.usage

            content = completion.output_messages[0].content
            judge_response = ParsedJudgeResponse.model_validate_json(content) if content else None

            if judge_response is None:
                raise ParseError(f"Response could not be parsed: {content}")
            elif not (0 <= judge_response.score <= 1):
                raise ParseError(f"Score is not between 0 and 1: {judge_response.score}")

            return judge_response, usage
        except Exception as e:
            raise ParseError(e) from e
