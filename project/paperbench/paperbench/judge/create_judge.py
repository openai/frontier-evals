# separate file to avoid circular imports when other files import base Judge class
import structlog.stdlib
from paperbench.judge.base import Judge
from paperbench.judge.dummyrandom import DummyJudge, RandomJudge
from paperbench.judge.simple import SimpleJudge
from paperbench.paper_registry import Paper

logger = structlog.stdlib.get_logger(component=__name__)


def can_model_reason(model_name: str) -> bool:
    reasoning_models = ["o1", "o1-mini", "o3-mini"]
    if any(model_name.startswith(rm) for rm in reasoning_models):
        return True
    return False


def handle_rubrics_for_simple_judge(judge_kwargs: dict, paper: Paper) -> dict:
    large_rubrics_to_handle = {"pinn"}
    if paper.id in large_rubrics_to_handle:
        judge_kwargs["max_prior_nodes"] = 5
    return judge_kwargs


def handle_reasoning_effort(judge_kwargs: dict, reasoning_effort: str | None) -> dict:
    if reasoning_effort is None:
        return judge_kwargs
    assert "model" in judge_kwargs, "Cannot set reasoning effort if no model specified"
    model_name = judge_kwargs["model"]
    if not can_model_reason(model_name):
        logger.warning(f"Ignoring `reasoning_effort`: not supported for {model_name}")
        return judge_kwargs  # ignore this param for models that can't reason
    if "completion_kwargs" not in judge_kwargs:
        judge_kwargs["completion_kwargs"] = {}
    judge_kwargs["completion_kwargs"]["reasoning_effort"] = reasoning_effort
    return judge_kwargs


def handle_judge_kwargs(
    judge_type: str,
    code_only: bool = False,
    paper: Paper | None = None,
    model_name: str | None = None,
    reasoning_effort: str | None = None,
) -> dict:
    """
    Prepares the right judge kwargs based on the judge type, model name and paper
    To be fed into `create_judge` typically.
    """
    judge_kwargs = {"code_only": code_only}
    if judge_type == "dummy":
        return judge_kwargs
    judge_kwargs["model"] = model_name
    if judge_type == "simple":
        if paper is not None:
            judge_kwargs = handle_rubrics_for_simple_judge(judge_kwargs, paper)
        judge_kwargs = handle_reasoning_effort(judge_kwargs, reasoning_effort)

    return judge_kwargs


def create_judge(
    judge_type: str,
    judge_kwargs: dict,
    **shared_kwargs,
) -> Judge:
    """Create and return appropriate judge instance based on type.

    Args:
        judge_type: Type of judge to create ('dummy', 'random', or 'simple')
        judge_kwargs: Keyword arguments specific for the judge
        shared_kwargs: Keyword arguments shared by all judges

    Returns:
        An instance of the appropriate judge class
    """

    if judge_type == "simple":
        return SimpleJudge(**{**judge_kwargs, **shared_kwargs})
    elif judge_type == "random":
        return RandomJudge(**{**judge_kwargs, **shared_kwargs})
    elif judge_type == "dummy":
        return DummyJudge(**shared_kwargs)
    else:
        raise ValueError(f"Invalid judge type: {judge_type}")
