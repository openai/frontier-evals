"""Guard the Playwright tracing API used by bundled diagnostic scripts."""

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("issue", ["16912_4", "17387_1", "29854_602"])
def test_user_tool_uses_supported_trace_stop_api(issue: str) -> None:
    path = Path(__file__).resolve().parents[1] / "issues" / issue / "user_tool.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "tracing"
    ]
    assert not [call for call in calls if call.func.attr == "export"]
    stops = [call for call in calls if call.func.attr == "stop"]
    assert stops
    assert all(any(kw.arg == "path" for kw in call.keywords) for call in stops)
