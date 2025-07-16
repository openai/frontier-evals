import asyncio
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

import openai
import structlog.stdlib
import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI
from paperbench.judge.constants import (
    CRITERION_PROMPT,
    FILE_RANKING_PROMPT,
    GRADING_PROMPT,
    build_judge_task_prompt,
)
from paperbench.judge.utils import format_file, get_model_context_window_length, reduce_log
from paperbench.rubric.tasks import TASK_CATEGORY_QUESTIONS, TaskNode
from paperbench.utils import oai_completion_with_retry_async
from pydantic import BaseModel
from structlog import wrap_logger
from structlog.stdlib import BoundLogger
from typing_extensions import override

logger = structlog.stdlib.get_logger(component=__name__)
load_dotenv()


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


RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


@dataclass
class FilesContentData:
    all_files_content: str
    tree_structure: str
    all_file_names: str


@dataclass
class FilesPreparationOutcome:
    success: bool
    all_files_fit: bool
    files_content_data: FilesContentData


@dataclass
class TokenUsage:
    """Tracks token usage across different OAI models."""

    def __init__(self):
        self.usage = {}

    def add_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Add token usage for a model."""
        if model not in self.usage:
            self.usage[model] = {"in": 0, "out": 0}
        self.usage[model]["in"] += input_tokens
        self.usage[model]["out"] += output_tokens

    def add_from_completion(self, model: str, usage: openai.types.CompletionUsage):
        """Add token usage from an OpenAI completion response."""
        self.add_usage(model, usage.prompt_tokens, usage.completion_tokens)

    def to_dict(self) -> dict:
        """Convert usage to a dictionary format."""
        return self.usage

    @classmethod
    def from_dict(cls, data: dict) -> "TokenUsage":
        """Create a TokenUsage instance from a dictionary."""
        token_usage = cls()
        for model, usage in data.items():
            token_usage.add_usage(model, usage["in"], usage["out"])
        return token_usage


@dataclass(frozen=True)
class GradedTaskNode(TaskNode):
    """
    Same as a `TaskNode`, but each node also has a `score` and an `explanation`.

    Attributes:
        score: Score between 0 and 1 (and exclusively 0 or 1 for leaf nodes)
        valid_score: Boolean indicating whether the grading is valid, i.e. in case of judge errors
        explanation: Explanation of the grading
        judge_metadata: Additional judge-specific metadata for this node
        sub_tasks: List of sub GradedTaskNodes
    """

    score: float = 0.0
    valid_score: bool = False
    explanation: str = "not yet graded"
    judge_metadata: dict | None = None
    sub_tasks: Sequence["GradedTaskNode"] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "GradedTaskNode":
        try:
            sub_tasks = [cls.from_dict(task) for task in data["sub_tasks"]]
            task = cls(
                id=data["id"],
                requirements=data["requirements"],
                weight=data["weight"],
                sub_tasks=sub_tasks,
                task_category=data["task_category"],
                score=data["score"],
                valid_score=data["valid_score"],
                explanation=data["explanation"],
                judge_metadata=(data["judge_metadata"] if "judge_metadata" in data else None),
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in task data: {e}")
        return task

    @override
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "requirements": self.requirements,
            "weight": self.weight,
            "score": self.score,
            "valid_score": self.valid_score,
            "task_category": self.task_category,
            "explanation": self.explanation,
            "judge_metadata": self.judge_metadata,
            "sub_tasks": [task.to_dict() for task in self.sub_tasks],
        }

    def set_score(self, score: float) -> "GradedTaskNode":
        return replace(self, score=score)

    @classmethod
    def from_task(
        cls,
        task: TaskNode,
        score: float,
        valid_score: bool,
        explanation: str,
        judge_metadata: dict | None = None,
    ) -> "GradedTaskNode":
        graded_sub_tasks = [
            cls.from_task(
                sub_task,
                score,
                valid_score,
                explanation=explanation,
                judge_metadata=judge_metadata,
            )
            for sub_task in task.sub_tasks
        ]
        return cls(
            id=task.id,
            requirements=task.requirements,
            weight=task.weight,
            sub_tasks=graded_sub_tasks,
            task_category=task.task_category,
            score=score,
            valid_score=valid_score,
            explanation=explanation,
            judge_metadata=judge_metadata,
        )

    def to_task(self) -> TaskNode:
        sub_tasks = [t.to_task() for t in self.sub_tasks]
        return TaskNode(
            id=self.id,
            requirements=self.requirements,
            weight=self.weight,
            sub_tasks=sub_tasks,
            task_category=self.task_category,
        )


def disqualify_leafs(node: GradedTaskNode) -> GradedTaskNode:
    """
    Sets all leaf scores to 0 and explanations to 'Disqualified'.
    Should be separately followed by `update_all_grades` to propagate the changes.
    """
    if node.is_leaf():
        disqualified_node = node.set_score(0.0)
        disqualified_node = disqualified_node.set_explanation("Submission has been disqualified")
        return disqualified_node

    new_sub_tasks = [disqualify_leafs(child) for child in node.sub_tasks]
    return node.set_sub_tasks(new_sub_tasks)


def disqualify(node: GradedTaskNode) -> GradedTaskNode:
    """
    Sets all leaf scores to 0 and explanations to 'Disqualified'. Updates all scores.
    """
    disqualified_node = disqualify_leafs(node)
    disqualified_node = update_all_grades(disqualified_node)
    return disqualified_node


def update_all_grades(node: GradedTaskNode) -> GradedTaskNode:
    """Recursively updates the scores for all nodes in a graded task tree.
    Leaf nodes retain their existing score, while internal nodes get a score
    computed from their children using score_from_children."""
    if node.is_leaf():
        return node
    new_sub_tasks = [update_all_grades(child) for child in node.sub_tasks]
    computed_score = score_from_children(new_sub_tasks)
    updated_node = node.set_sub_tasks(new_sub_tasks).set_score(computed_score)
    return updated_node


class Judge(ABC):
    def __init__(
        self,
        paper_path: Path,
        rubric: TaskNode,
        addendum: str | None,
        judge_addendum: str | None,
        submission_dir: Path,
        structured_output_model: str = "gpt-4o-2024-08-06",
        log_path: Path | None = None,
        max_depth: int = 999,
        code_only: bool = False,
        resources_provided: bool = False,
        *args,
        **kwargs,
    ):
        self.paper_path: Path = paper_path
        self.rubric: TaskNode = rubric
        self.addendum: str | None = addendum
        self.judge_addendum: str | None = judge_addendum
        self.submission_dir: Path = submission_dir
        self.structured_output_model: str = structured_output_model
        self.log_path: Path | None = log_path
        self.max_depth: int = max_depth
        self.code_only: bool = code_only
        self.resources_provided: bool = resources_provided

        # Reproduction script and log
        self.reproduce_sh_path: Path = self.submission_dir / "reproduce.sh"
        self.reproduce_log_path: Path = self.submission_dir / "reproduce.log"
        reproduce_log_creation_file_path: Path = self.submission_dir / "reproduce.log.creation_time"
        if self.reproduce_sh_path.exists():
            self.reproduce_sh_content: str = self.reproduce_sh_path.read_text()
        else:
            self.reproduce_sh_content = "(Does not exist)"
        if self.reproduce_log_path.exists():
            self.reproduce_log_content: str = self.reproduce_log_path.read_text()
            self.reproduce_log_content = reduce_log(self.reproduce_log_content)
            self.reproduction_log_creation_time_utc = datetime.fromtimestamp(
                (
                    int(reproduce_log_creation_file_path.read_text())
                    if reproduce_log_creation_file_path.exists()
                    else self.reproduce_log_path.stat().st_mtime  # fallback to next best thing
                ),
                tz=timezone.utc,
            )
        else:
            self.reproduce_log_content = "(Does not exist)"
            self.reproduction_log_creation_time_utc = datetime.now(tz=timezone.utc)

    @property
    @abstractmethod
    def judge_type(self) -> str:
        """Abstract property for judge type, to be implemented by sub-classes."""
        raise NotImplementedError()

    async def grade(
        self,
        task: TaskNode | None = None,
        grade_leaf_fn: Callable[[TaskNode], Awaitable[GradedTaskNode]] | None = None,
        current_depth: int = 1,
    ) -> GradedTaskNode:
        """
        Options:
        - task: The (sub)task to grade. If None, the entire rubric is graded.
        - grade_leaf_fn: The function to use to grade leaf nodes. If None, the default `grade_leaf` method is used.

        Returns a `GradedTaskNode` for the given `TaskNode`.
        If the task is a leaf, it calls `grade_leaf` to grade it.
        Otherwise, the task is graded by recursively grading its descendants bottom-up.
        """
        if task is None:
            task = self.rubric

        grade_leaf_fn = grade_leaf_fn or self.grade_leaf

        try:
            if current_depth >= self.max_depth and not task.is_leaf():
                logger.info(f"Max depth reached for task {task.id}. Approximating entire subtree.")
                return await self.grade_subtree(task)
            elif task.is_leaf():
                return await grade_leaf_fn(task)
        except openai.RateLimitError as e:
            logger.exception(f"Rate limit error while grading leaf {task.id}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Grading leaf {task.id} failed!\n{e}")
            return GradedTaskNode.from_task(
                task,
                score=0.0,
                valid_score=False,
                explanation=str(e),
                judge_metadata=None,
            )

        graded_sub_tasks = await asyncio.gather(
            *(self.grade(t, grade_leaf_fn, current_depth + 1) for t in task.sub_tasks)
        )
        weighted_score = score_from_children(graded_sub_tasks)

        return GradedTaskNode(
            id=task.id,
            requirements=task.requirements,
            weight=task.weight,
            sub_tasks=graded_sub_tasks,
            score=weighted_score,
            valid_score=True,
            explanation="Aggregated score from sub-tasks.",
            judge_metadata=None,
        )

    @abstractmethod
    async def grade_leaf(self, task: TaskNode) -> GradedTaskNode:
        """Grades a leaf task (to be implemented by sub-classes)."""

        raise NotImplementedError()

    @abstractmethod
    async def grade_subtree(self, task: TaskNode) -> GradedTaskNode:
        """Approximates the grade for an entire subtree when the max depth is reached."""
        raise NotImplementedError()

    def get_logger(self, task: TaskNode) -> BoundLogger:
        """Creates a logger for a specific task."""

        if not self.log_path:
            _logger = logging.getLogger("null_logger")
            _logger.addHandler(logging.NullHandler())
            return wrap_logger(_logger)

        run_logger = logging.getLogger(task.id)
        run_logger.setLevel(logging.DEBUG)
        log_file_handler = logging.FileHandler(self.log_path / f"{task.id}.log")
        run_logger.addHandler(log_file_handler)
        run_logger.propagate = False

        return wrap_logger(run_logger)


class DummyJudge(Judge):
    @property
    def judge_type(self) -> str:
        return "dummy"

    @override
    async def grade_leaf(self, task: TaskNode) -> GradedTaskNode:
        return GradedTaskNode.from_task(
            task,
            score=1.0,
            valid_score=True,
            explanation="This is a dummy judge that always gives a score of 1.0.",
            judge_metadata=None,
        )

    @override
    async def grade_subtree(self, task: TaskNode) -> GradedTaskNode:
        # For demonstration, we'll just assign a perfect score:
        return GradedTaskNode.from_task(
            task,
            score=1.0,
            valid_score=True,
            explanation="Dummy approximate subtree grading with a perfect score.",
            judge_metadata=None,
        )


class RandomJudge(Judge):
    @property
    def judge_type(self) -> str:
        return "random"

    @override
    async def grade_leaf(self, task: TaskNode) -> GradedTaskNode:
        return GradedTaskNode.from_task(
            task,
            score=random.randint(0, 1),
            valid_score=True,
            explanation="This is a random judge that gives a random score of 0 or 1.",
            judge_metadata=None,
        )

    @override
    async def grade_subtree(self, task: TaskNode) -> GradedTaskNode:
        return GradedTaskNode.from_task(
            task,
            score=random.randint(0, 1),
            valid_score=True,
            explanation="This is a random judge that gives a random score of 0 or 1.",
            judge_metadata=None,
        )


class SimpleJudge(Judge):
    def __init__(
        self,
        paper_path: Path,
        rubric: TaskNode,
        addendum: str | None,
        judge_addendum: str | None,
        submission_dir: Path,
        model: str,
        paper_md: Path,
        log_path: Path | None = None,
        buffer_tokens: int = 10000,  # 10k tokens of buffer
        max_depth: int = 999,
        code_only: bool = False,
        max_prior_nodes: int | None = None,
        completion_kwargs: dict | None = None,
        max_file_depth: int = 4,
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
        )

        self.model = model
        self.token_encoder = tiktoken.encoding_for_model(self.model)
        self.completion_kwargs = completion_kwargs or {}
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
        # pre-emptively truncate the reproduce.log and paper.md (latter almost never happens)
        # to allow for space for additional context when prompting
        self.reproduce_log_tokens = self.token_encoder.encode(
            self.reproduce_log_content, disallowed_special=()
        )
        self.paper_md_tokens = self.token_encoder.encode(self.paper_md, disallowed_special=())
        self._truncate_input()
        self.openai_client = AsyncOpenAI()

        self.avail_context_lens = {
            "Code Development": self._get_available_context("Code Development"),
            "Code Execution": self._get_available_context("Code Execution"),
            "Result Analysis": self._get_available_context("Result Analysis"),
            "Subtree": self._get_available_context("Subtree"),
        }

        self.files_string_prep_outcome = {
            k: self._prepare_relevant_files_strings(k)
            for k in ["Code Development", "Code Execution", "Result Analysis", "Subtree"]
        }

    def _get_available_context(self, task_category):
        reserved_context_lens = {
            "Code Development": len(self.paper_md_tokens),
            "Code Execution": len(self.paper_md_tokens) + len(self.reproduce_log_tokens),
            "Result Analysis": len(self.paper_md_tokens) + len(self.reproduce_log_tokens),
            "Subtree": len(self.paper_md_tokens) + len(self.reproduce_log_tokens),
        }
        model_context_length = get_model_context_window_length(self.model)

        return model_context_length - (reserved_context_lens[task_category] + self.buffer_tokens)

    def _truncate_input(self):
        """
        Truncates reproduce.log and paper.md until there is leeway for prompting.
        Truncates log files to be half of the context window length.
        e.g. 128k context window -> 64k token reproduce.log limit
        Assumes log reduction via reduce_log() has already been applied

        Further truncates log and paper until theres at least 5k tokens of space left
        Prioritizing log truncation over paper truncation
        """
        context_window_tokens = get_model_context_window_length(self.model)
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
        tree = {}
        for file in files:
            current = tree
            for part in file.parts:
                if part not in current:
                    current[part] = {}
                current = current[part]

        def _build_tree(node: dict, prefix: str = "") -> str:
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

    def _safe_read_file(self, file_path: Path) -> str:
        """
        Try to read a file, with robustness to different encodings.
        (Without this, we sometimes get `'utf-8' codec can't decode byte 0xa4 in position 64: invalid start byte`)
        """
        try:
            # Try utf-8 first
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Try latin1 if utf-8 fails
            return file_path.read_text(encoding="latin1")

    def _get_whitelisted_files(
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

        def should_include_file(path: Path) -> bool:
            if path.suffix not in whitelisted_extensions:
                return False

            if not path.exists() or path.is_symlink() and not path.resolve().exists():
                return False

            file_last_modified_time = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)

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
        for root, dirs, files in os.walk(self.submission_dir):
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
            for file in files:
                full_path = Path(root) / file
                if full_path.suffix in whitelisted_extensions:
                    if should_include_file(full_path):
                        whitelisted_files.append(full_path)

        if task_category == "Result Analysis" and all(
            datetime.fromtimestamp(file.stat().st_mtime, tz=timezone.utc)
            < self.reproduction_log_creation_time_utc
            for file in whitelisted_files
        ):
            self.reproduce_touched_files = False

        return whitelisted_files

    def _build_relevant_files_strings(
        self, task_category: str, max_file_depth: int | None = None
    ) -> FilesContentData:
        """
        Builds the file strings for the given task category.
        i.e. all the content of the files, all the names of the files, and a tree structure of the files
        """
        whitelisted_files: list[Path] = self._get_whitelisted_files(
            task_category, max_file_depth=max_file_depth
        )
        # can exit early if all the files in submission fit comfortably within context
        all_files_content = "\n\n".join(
            [
                format_file(
                    full_path.relative_to(self.submission_dir), self._safe_read_file(full_path)
                )
                for full_path in whitelisted_files
            ]
        )
        tree_structure: str = self._create_tree_structure(
            [p.relative_to(self.submission_dir) for p in whitelisted_files]
        )
        # TODO we might want to pass in function signatures as well
        all_file_names = "\n".join(
            [str(full_path.relative_to(self.submission_dir)) for full_path in whitelisted_files]
        )

        return FilesContentData(
            all_files_content=all_files_content,
            tree_structure=tree_structure,
            all_file_names=all_file_names,
        )

    def _attempt_preparing_files_strings(
        self, task_category: str, available_context: int, max_file_depth: int | None = None
    ) -> FilesPreparationOutcome:
        """
        Attempts preparing file strings for the given category
        Taking into account the available context window in its response
        Available context is in terms of tokens, not characters.
        """
        files_content_data = self._build_relevant_files_strings(task_category, max_file_depth)
        all_files_content = files_content_data.all_files_content
        tree_structure = files_content_data.tree_structure
        all_file_names = files_content_data.all_file_names

        all_files_content_len = len(
            self.token_encoder.encode(all_files_content, disallowed_special=())
        )
        if all_files_content_len < available_context:
            return FilesPreparationOutcome(
                success=True,
                all_files_fit=True,
                files_content_data=files_content_data,
            )

        tree_structure_len = len(self.token_encoder.encode(tree_structure, disallowed_special=()))
        all_file_names_len = len(self.token_encoder.encode(all_file_names, disallowed_special=()))
        if (
            all_file_names_len >= available_context
            or tree_structure_len >= available_context
            or all_file_names_len + tree_structure_len >= available_context
        ):
            return FilesPreparationOutcome(
                success=False,
                all_files_fit=False,
                files_content_data=files_content_data,
            )
        else:
            return FilesPreparationOutcome(
                success=True,
                all_files_fit=False,
                files_content_data=files_content_data,
            )

    def _truncate_files(self, tree_structure, all_file_names, available_context) -> tuple[str, str]:
        """
        Truncates both tree structure and file names to fit within available context.
        Distributes the available context roughly equally between the two strings.
        Available context is in terms of tokens, not characters.
        """
        all_file_names_toks = self.token_encoder.encode(all_file_names, disallowed_special=())
        tree_structure_toks = self.token_encoder.encode(tree_structure, disallowed_special=())

        all_file_names_len = len(all_file_names_toks)
        tree_structure_len = len(tree_structure_toks)
        total_len = all_file_names_len + tree_structure_len

        # If total length is already within context, return as is
        if total_len <= available_context:
            return all_file_names, tree_structure

        # Calculate proportional lengths to maintain relative sizes
        proportion = all_file_names_len / total_len
        target_file_names_len = int(available_context * proportion)
        target_tree_len = available_context - target_file_names_len

        truncated_file_names = all_file_names_toks[:target_file_names_len]
        truncated_tree = tree_structure_toks[:target_tree_len]

        # preserve complete lines when decoding where possible by dropping the last line
        truncated_file_names = self.token_encoder.decode(truncated_file_names).rsplit("\n", 1)[0]
        truncated_tree = self.token_encoder.decode(truncated_tree).rsplit("\n", 1)[0]

        return truncated_file_names, truncated_tree

    def _prepare_relevant_files_strings(self, task_category: str):
        """
        Prepares the relevant file strings necessary for judging specific task categories.
        Automatically limits file depth if necessary.
        Automatically truncates to the model context window if necessary.
        """
        available_context = self.avail_context_lens[task_category]

        # 1st try without limiting depth
        attempt_outcome = self._attempt_preparing_files_strings(
            task_category, available_context, max_file_depth=None
        )

        # 2nd attempt: limit depth to 4
        if not attempt_outcome.success:
            attempt_outcome = self._attempt_preparing_files_strings(
                task_category, available_context, max_file_depth=4
            )

        # 3rd attempt: simply truncate the file strings, forcing 'success'
        if not attempt_outcome.success:
            files_content_data = attempt_outcome.files_content_data
            all_files_content = files_content_data.all_files_content  # irrelevant here
            all_file_names, tree_structure = self._truncate_files(
                files_content_data.tree_structure,
                files_content_data.all_file_names,
                available_context,
            )

            attempt_outcome = FilesPreparationOutcome(
                success=True,
                all_files_fit=False,
                files_content_data=FilesContentData(
                    all_files_content=all_files_content,
                    tree_structure=tree_structure,
                    all_file_names=all_file_names,
                ),
            )

        return attempt_outcome

    async def _prepare_relevant_files(
        self,
        task: TaskNode,
        leaf_logger: BoundLogger,
        max_files: int | None = 10,
    ) -> str:
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
        files_prep_outcome = self.files_string_prep_outcome[task.task_category]

        if files_prep_outcome.all_files_fit:
            leaf_logger.info("Codebase is within context window, returning entire codebase")
            return files_prep_outcome.files_content_data.all_files_content

        tree_structure = files_prep_outcome.files_content_data.tree_structure
        all_files_names = files_prep_outcome.files_content_data.all_file_names

        messages = [
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
                "content": f"Here are the files in the submission attempt:\n\nDirectory structure:\n{tree_structure}\n\nFlat file list:\n{all_files_names}\n\nNow return a list of the {str(max_files) + ' ' if max_files else ''}most relevant files in order of relevance (descending) to the resolution criteria, to be provided for your inspection. Your response must contain each filename separated by newlines, with each file containing the full path. Use the exact paths from the flat file list. Do not write anything else.",
            },
        ]
        model_response = await oai_completion_with_retry_async(
            self.openai_client.chat.completions.create,
            model=self.model,
            messages=messages,
            **self.completion_kwargs,
        )
        selected_files = model_response.choices[0].message.content
        leaf_logger.info(f"Model file selection raw output:\n{selected_files}")

        selected_files_tokens = []
        num_files = 0
        total_tokens = 0
        max_tokens = self.avail_context_lens[task.task_category] - 2000  # Buffer of 2k tokens

        for rel_path in selected_files.split("\n"):
            full_path = self.submission_dir / rel_path.strip()
            try:
                file_content = format_file(
                    full_path.relative_to(self.submission_dir), self._safe_read_file(full_path)
                )
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
                leaf_logger.warning(f"File {full_path} not found!")
            except IsADirectoryError:
                leaf_logger.warning(f"File {full_path} is a directory!")
            except UnicodeDecodeError:
                leaf_logger.warning(f"File {full_path} is not a text file!")
            except Exception as e:
                leaf_logger.warning(f"File {full_path} is not readable! Error: {e}")

        # Decode once at the end, ensuring we end with complete lines
        return self.token_encoder.decode(selected_files_tokens).rsplit("\n", 1)[0]

    async def _construct_grade_leaf_messages(
        self, task: TaskNode, leaf_logger: BoundLogger
    ) -> list[dict[str, Any]]:
        relevant_files = await self._prepare_relevant_files(task, leaf_logger)
        relevant_files_prompt = (
            f"Here are the most relevant files included in the submission attempt, concatenated:\n<files>\n{relevant_files}\n</files>"
            if task.task_category != "Result Analysis"
            else f"Here are the most relevant docs and the files touched (i.e. modified or created) during the reproduce.sh execution, concatenated:\n<files>\n{relevant_files}\n</files>"
        )

        relevant_rubric_nodes = task.get_prior_nodes(self.rubric, self.max_prior_nodes)
        relevant_rubric_context = ""
        for node in relevant_rubric_nodes:
            relevant_rubric_context += f" -> {node.requirements}\n"

        reproduce_files_messages = []
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

        messages = [
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
                    judge_token_usage = TokenUsage()
                    messages = await self._construct_grade_leaf_messages(task, leaf_logger)
                    model_response = await oai_completion_with_retry_async(
                        self.openai_client.chat.completions.create,
                        model=self.model,
                        messages=messages,
                        **self.completion_kwargs,
                    )
                    judge_token_usage.add_from_completion(self.model, model_response.usage)
                    model_response = model_response.choices[0].message.content
                    messages += [{"role": "assistant", "content": model_response}]

                    leaf_logger.info(f"model response: {model_response}")
                    score_response, parse_usage = await self._parse_model_response(
                        model_response, continuous=(task.task_category == "Subtree")
                    )
                    judge_token_usage.add_from_completion(self.structured_output_model, parse_usage)
                    graded_task_node = GradedTaskNode.from_task(
                        task,
                        score=score_response.score,
                        valid_score=score_response.valid_score,
                        explanation=score_response.explanation,
                        judge_metadata={
                            "full_judge_response": model_response,
                            "token_usage": judge_token_usage.to_dict(),
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
                for handler in leaf_std_logger.handlers:
                    handler.close()
                    leaf_std_logger.removeHandler(handler)

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
        self, response: str, continuous: bool = False
    ) -> tuple[ParsedJudgeResponseFloat | ParsedJudgeResponseInt, openai.types.CompletionUsage]:
        """Parses a model response as a `ParsedJudgeResponse`."""

        score_instruction = "(either 0 or 1)" if not continuous else "(between 0 and 1)"
        messages = [
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
            response_format = ParsedJudgeResponseInt if not continuous else ParsedJudgeResponseFloat
            completion = await oai_completion_with_retry_async(
                self.openai_client.beta.chat.completions.parse,
                model=self.structured_output_model,
                messages=messages,
                response_format=response_format,
            )
            usage = completion.usage
            score_response = completion.choices[0].message
            if score_response.parsed:
                # check if score is between 0 and 1
                if not (0 <= score_response.parsed.score <= 1):
                    raise ParseError(f"Score is not between 0 and 1: {score_response.parsed.score}")
                return score_response.parsed, usage
            raise ParseError(
                f"Unexpected error - Response neither parsed nor refused: {score_response}"
            )
        except Exception as e:
            raise ParseError(e)


def score_from_children(children: list[GradedTaskNode]) -> float:
    """
    Calculate the weighted score accumulated from a list of graded children.
    """
    total_weight = sum(child.weight for child in children)
    if total_weight == 0:
        return 0.0
    weighted_score = sum(child.score * child.weight for child in children) / total_weight
    return weighted_score


def _get_leaf_node_token_usages(task: GradedTaskNode) -> list[TokenUsage]:
    """Recursively extract token usage from leaf nodes of the task tree"""

    if task.is_leaf():
        # need this check because judge_metadata may be malformed in case of node errors
        if task.judge_metadata is not None and "token_usage" in task.judge_metadata:
            return [TokenUsage.from_dict(task.judge_metadata["token_usage"])]
        else:
            return []

    token_usages = []

    for t in task.sub_tasks:
        t_usages = _get_leaf_node_token_usages(t)
        token_usages.extend(t_usages)
    return token_usages


def get_total_token_usage(graded_task_tree: GradedTaskNode) -> TokenUsage:
    """
    Gets the total token usage summed across all leaf nodes of the task tree
    Assumes the judge_metadata dict a `token_usage` key of type `TokenUsage`
    """
    token_usages = _get_leaf_node_token_usages(graded_task_tree)

    total_token_usage = TokenUsage()
    for token_usage in token_usages:
        for model, usage in token_usage.usage.items():
            total_token_usage.add_usage(model, usage["in"], usage["out"])

    return total_token_usage
