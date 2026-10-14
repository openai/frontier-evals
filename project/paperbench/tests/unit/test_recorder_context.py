from nanoeval.json_recorder import JsonRecorder
from nanoeval.recorder_protocol import BasicRunSpec
from paperbench.nano.logging import PaperBenchLibraryConfig


def test_paperbench_recorder_context_sets_attempt_group_id(tmp_path) -> None:
    recorder = JsonRecorder(
        run_spec=BasicRunSpec(run_id="run", run_set_id="run-set"),
        filename=tmp_path / "results.jsonl",
    )
    config = PaperBenchLibraryConfig()

    assert recorder.current_sample_id() is None
    assert recorder.current_group_id() is None

    with config.set_recorder_context(recorder, sample_id="paper-run", group_id="2.3"):
        assert recorder.current_sample_id() == "paper-run"
        assert recorder.current_group_id() == "2.3"

    assert recorder.current_sample_id() is None
    assert recorder.current_group_id() is None
