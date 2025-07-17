from dataclasses import dataclass

import openai
from paperbench.judge.graded_task_node import GradedTaskNode


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
