import evmbench.audit as audit_module
from evmbench.audit import Vulnerability


def test_vulnerability_text_content_uses_configured_incorrect_findings(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(audit_module, "get_audits_dir", lambda: tmp_path)

    findings_dir = tmp_path / "sample-audit" / "findings"
    incorrect_dir = findings_dir / "incorrect" / "medium"
    incorrect_dir.mkdir(parents=True)

    (findings_dir / "H-01.md").write_text("canonical finding")
    (incorrect_dir / "H-01.md").write_text("medium incorrect finding")

    canonical = Vulnerability(id="H-01", audit_id="sample-audit", title="Finding")
    incorrect = Vulnerability(
        id="H-01",
        audit_id="sample-audit",
        title="Finding",
        findings_subdir="medium",
    )

    assert canonical.text_content == "canonical finding"
    assert incorrect.text_content == "medium incorrect finding"
