import pytest

from evmbench.nano.eval import EVMbench


@pytest.mark.asyncio
async def test_evmbench_full_summary_handles_empty_results() -> None:
    eval = EVMbench()

    summary = await eval.get_full_summary([])

    assert summary["params"]["n_samples"] == 0
    assert summary["metrics"]["score"] == 0
    assert summary["metrics"]["max_score"] == 0
    assert summary["run_group_id"] == eval.run_group_id
