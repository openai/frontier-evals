import fnmatch
import json
import tarfile
import tempfile
import time
from pathlib import Path

import blobfile as bf
from alcatraz.clusters.local import LocalConfig
from preparedness_turn_completer.oai_turn_completer import OpenAITurnCompleter

from paperbench.nano.eval import ExternalPythonCodingSolver
from paperbench.nano.structs import JudgeConfig, ReproductionConfig

DEFAULT_AZURE_VM_SKU = "Standard_D2as_v4"

AGENT_ID_TO_IMAGE = {
    "dummy": "dummy",
    "aisi-basic-agent-openai-dev": "aisi-basic-agent:latest",
}

EXPECTED_FILES_BY_AGENT = {
    "dummy": [
        "logs/run.log",
        "logs/docker.log",
        "submission/reproduce.sh",
    ],
    "aisi-basic-agent": [
        "logs/agent.log",
        "logs/docker.log",
        "logs/*.eval",
    ],
    "aisi-basic-agent-openai-dev": [
        "logs/agent.log",
        "logs/docker.log",
        "logs/*.eval",
    ],
}


def assert_expected_files_exist(run_dir: str, agent_id: str) -> None:
    """Asserts that the expected files exist in the run directory. Allows for wildcards in the file pattern."""
    for file_pattern in EXPECTED_FILES_BY_AGENT[agent_id]:
        if "*" in file_pattern or "?" in file_pattern:
            pattern_dir = Path(file_pattern).parent
            pattern_name = Path(file_pattern).name
            search_path = bf.join(run_dir, str(pattern_dir))
            matching_files = [
                i for i in bf.listdir(search_path) if fnmatch.fnmatch(i, pattern_name)
            ]
            assert matching_files, (
                f"No files matching {file_pattern} found for agent {agent_id} in {search_path}"
            )
        else:
            assert bf.exists(bf.join(run_dir, file_pattern)), (
                f"Expected {file_pattern} to exist for agent {agent_id} at {bf.join(run_dir, file_pattern)}"
            )


def assert_rollout_files_exist(paper_dir: str, agent_id: str) -> None:
    assert bf.exists(bf.join(paper_dir, "metadata.json")), (
        f"metadata.json not found in paper directory at {paper_dir}"
    )

    pattern = paper_dir + "/**/submission.tar.gz"
    tar_files = list(bf.glob(pattern))

    assert len(tar_files) >= 1, (
        f"Expected at least 1 tar.gz file in {paper_dir}, found {len(tar_files)}"
    )

    run_tar_files = sorted([f for f in tar_files if "executed" not in f])
    assert len(run_tar_files) >= 1, (
        f"Expected at least one non-executed tar.gz file in {paper_dir}, found {len(run_tar_files)}"
    )
    tar_file = run_tar_files[-1]  # Take the latest one (last in sorted order)

    # Extract the submission locally and check for expected files
    with tempfile.TemporaryDirectory() as tmp_submission_dir:
        with bf.BlobFile(tar_file, "rb") as f:
            with tarfile.open(fileobj=f) as tar:
                tar.extractall(path=tmp_submission_dir)

        checkpoint_dirs = [
            i for i in bf.listdir(tmp_submission_dir) if bf.isdir(bf.join(tmp_submission_dir, i))
        ]
        assert len(checkpoint_dirs) == 1, (
            f"Expected exactly one checkpoint directory in {tmp_submission_dir}, found {len(checkpoint_dirs)}"
        )
        checkpoint_dir = bf.join(tmp_submission_dir, checkpoint_dirs[0])

        assert_expected_files_exist(checkpoint_dir, agent_id)

        docker_log = bf.join(checkpoint_dir, "logs/docker.log")
        assert bf.exists(docker_log), (
            f"Docker log file not found for agent {agent_id} at {docker_log}"
        )

        with bf.BlobFile(docker_log, "rb") as f:
            log_content = f.read().decode("utf-8")
            assert "Docker version" in log_content, (
                f"Docker version check failed for agent {agent_id}"
            )
            if agent_id == "dummy":  # only dummy agent pulls hello-world image and lists containers
                assert "Hello from Docker!" in log_content, (
                    f"Failed to run container inside container for agent {agent_id}"
                )
                assert "CONTAINER ID" in log_content, (
                    f"Failed to list containers for agent {agent_id}"
                )


def check_group_log_for_errors(run_group_dir: str) -> None:
    """Check the group.log file for error messages."""
    group_log = bf.join(run_group_dir, "group.log")
    assert bf.exists(group_log), f"Expected group.log file to exist at {group_log}"

    with bf.BlobFile(group_log, "rb") as f:
        log_content = f.read().decode("utf-8").lower()
        assert "error" not in log_content, f"Found error message in group.log:\n{log_content}"


def setup_cluster_config(image: str) -> LocalConfig:
    cluster_config = LocalConfig(image=image, pull_from_registry=False)
    return cluster_config


def setup_solver(agent_id: str) -> ExternalPythonCodingSolver:
    image = AGENT_ID_TO_IMAGE[agent_id]
    cluster_config = setup_cluster_config(image)
    return ExternalPythonCodingSolver(agent_id=agent_id, cluster_config=cluster_config)


def setup_reproduction_config(skip_reproduction: bool = True) -> ReproductionConfig:
    image = "pb-reproducer:latest"
    cluster_config = setup_cluster_config(image)
    return ReproductionConfig(
        skip_reproduction=skip_reproduction,
        cluster_config=cluster_config,
    )


def setup_judge_config(skip_grading: bool = True, scaffold: str = "dummy") -> JudgeConfig:
    image = "pb-env:latest"
    cluster_config = setup_cluster_config(image)
    return JudgeConfig(
        scaffold=scaffold,
        grade=not skip_grading,
        cluster_config=cluster_config,
        completer_config=OpenAITurnCompleter.Config(model="gpt-4o-mini"),  # cheap and quick
    )


def create_fake_submission(run_dir: str) -> None:
    """
    Creates a fake submission locally, compresses it, then write it to the run dir. The submission
    contains a reproduce.sh script that creates a hello_world file.
    """
    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_path = Path(tmp_dir_str)

        # build submission directory
        submission_dir = tmp_path / "submission"
        submission_dir.mkdir(parents=True, exist_ok=True)

        reproduce_path = submission_dir / "reproduce.sh"
        reproduce_path.write_text("touch hello_world")
        reproduce_path.chmod(reproduce_path.stat().st_mode | 0o111)

        # pack it
        tar_name = "submission.tar.gz"
        with tarfile.open(tmp_path / tar_name, "w:gz") as tar:
            tar.add(submission_dir, arcname="submission")

        # destination
        timestamp = time.strftime("%Y-%m-%dT%H-%M-%S-%Z", time.gmtime())
        dest_dir = bf.join(run_dir, "submissions", timestamp)
        bf.makedirs(dest_dir)

        tar_path = bf.join(dest_dir, tar_name)
        with open(tmp_path / tar_name, "rb") as f:
            with bf.BlobFile(tar_path, "wb") as f_out:
                f_out.write(f.read())

        with bf.BlobFile(tar_path, "wb") as out_f, open(tmp_path / tar_name, "rb") as in_f:
            out_f.write(in_f.read())

        # Upload status.json
        with bf.BlobFile(bf.join(run_dir, "status.json"), "wb") as f:
            f.write(json.dumps({}).encode("utf-8"))

        assert bf.exists(tar_path), f"Fake submission file not found at {tar_path}"
