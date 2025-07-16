import subprocess

import pytest

scripts = [
    "paperbench/scripts/run_judge.py",
    "paperbench/scripts/run_judge_eval.py",
    "paperbench/scripts/run_reproduce.py",
]


@pytest.mark.parametrize("script", scripts)
def test_script_help(script):
    """Test that each script runs with -h without errors."""
    try:
        # Run the script with -h and check it doesn't throw an error
        result = subprocess.run(
            ["python", script, "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Assert that the process exits successfully
        assert result.returncode == 0, f"Script {script} failed with error:\n{result.stderr}"
    except Exception as e:
        pytest.fail(f"Exception occurred while running {script}: {e}")
