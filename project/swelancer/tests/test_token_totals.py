import asyncio
from types import SimpleNamespace
from typing import Any, cast

from swelancer.eval import SWELancerEval


def test_empty_summary_reports_zero_token_totals(tmp_path) -> None:
    run_group_id = "token-total-regression"
    (tmp_path / run_group_id).mkdir()
    eval_stub = SimpleNamespace(
        runs_dir=str(tmp_path),
        run_group_id=run_group_id,
        _get_convo_len_stats=lambda results: {},
    )

    summary = asyncio.run(
        SWELancerEval.get_full_summary(cast(Any, eval_stub), [])
    )

    assert summary["total_input_tokens"] == 0
    assert summary["total_output_tokens"] == 0
    assert summary["total_reasoning_tokens"] == 0
