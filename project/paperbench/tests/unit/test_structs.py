from paperbench.nano.structs import ReproductionMetadata


def _legacy_metadata() -> dict:
    return {
        "is_valid_git_repo": True,
        "git_log": "commit",
        "repro_script_exists": True,
        "files_before_reproduce": "before",
        "files_after_reproduce": "after",
        "timedout": False,
        "repro_log": "ok",
    }


def test_reproduction_metadata_loads_legacy_optional_schema() -> None:
    metadata = ReproductionMetadata.from_dict(_legacy_metadata())

    assert metadata.retried_results == []
    assert metadata.repro_execution_time is None
    assert metadata.git_status_after_reproduce is None
    assert metadata.executed_submission is None


def test_reproduction_metadata_preserves_optional_fields_when_present() -> None:
    data = {
        **_legacy_metadata(),
        "retried_results": [
            {
                "repro_execution_time": 12.5,
                "timedout": False,
                "repro_log": "retry ok",
            }
        ],
        "repro_execution_time": 20.0,
        "git_status_after_reproduce": "clean",
        "executed_submission": "/tmp/submission.tar.gz",
    }

    metadata = ReproductionMetadata.from_dict(data)

    assert len(metadata.retried_results) == 1
    assert metadata.retried_results[0].repro_execution_time == 12.5
    assert metadata.repro_execution_time == 20.0
    assert metadata.git_status_after_reproduce == "clean"
    assert metadata.executed_submission == "/tmp/submission.tar.gz"
