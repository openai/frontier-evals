"""Check that the public incorrect-findings option matches the audit directory."""

from typing import get_args, get_type_hints

from evmbench.nano.eval import EVMbench


def test_medium_findings_option_matches_audit_directory_spelling() -> None:
    allowed = get_args(get_type_hints(EVMbench)["findings_subdir"])
    assert "medium" in allowed
    assert "med" not in allowed
    assert {"", "low", "high"}.issubset(set(allowed))
