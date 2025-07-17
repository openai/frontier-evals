import os
import shutil
import subprocess
import tempfile

from paperbench.utils import get_paperbench_data_dir

# Configuration for each benchmark
REPOS = {
    "rice/0": {
        "repo": "https://github.com/chengzelei/RICE",
        "commit": "4962aeb46d6000ebdf5ecbc5b46fff0d0505da59",
        "output_dir": "rice/0",
    },
}


def clone_repo(repo_url: str, commit_hash: str, output_dir: str) -> str | None:
    """Clone a repository at a specific commit.

    Args:
        repo_url: URL of the git repository
        commit_hash: Specific commit to checkout
        output_dir: Directory to clone into

    Returns:
        Path to cloned repo directory or None if failed
    """
    try:
        # Clone the repository
        subprocess.run(
            ["git", "clone", repo_url, output_dir],
            check=True,
            capture_output=True,
        )

        # Checkout specific commit
        subprocess.run(
            ["git", "checkout", commit_hash],
            cwd=output_dir,
            check=True,
            capture_output=True,
        )

        # Remove .git directory
        shutil.rmtree(os.path.join(output_dir, ".git"))
        return output_dir

    except subprocess.CalledProcessError as e:
        print(f"Failed to clone/checkout repo: {e}")
        return None


def create_tar(source_dir: str, output_path: str) -> bool:
    """Create an uncompressed tar archive from a directory.

    Args:
        source_dir: Directory to create tar from
        output_path: Path to output tar file

    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create a temporary directory to hold the submission folder
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create submission directory inside temp_dir
            submission_dir = os.path.join(temp_dir, "submission")
            shutil.copytree(source_dir, submission_dir)

            # Create tar from temp_dir, which will include the submission folder
            subprocess.run(
                ["tar", "-cf", output_path, "-C", temp_dir, "submission"],
                check=True,
                capture_output=True,
            )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to create tar: {e}")
        return False


def process_benchmark(repo_url: str, commit_hash: str, output_dir: str) -> bool:
    """Process a single benchmark by cloning and creating tar.

    Args:
        repo_url: URL of the git repository
        commit_hash: Specific commit to checkout
        output_dir: Directory to place the final submission.tar

    Returns:
        True if successful, False otherwise
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Cloning {repo_url} at commit {commit_hash}")

        cloned_dir = clone_repo(repo_url, commit_hash, temp_dir)
        if not cloned_dir:
            return False

        output_path = os.path.join(output_dir, "submission.tar")
        if create_tar(temp_dir, output_path):
            print(f"Created {output_path}")
            return True
        return False


def main() -> None:
    data_dir = get_paperbench_data_dir()

    for benchmark, config in REPOS.items():
        print(f"\nProcessing {benchmark}...")
        output_dir = os.path.join(data_dir, "judge_eval", config["output_dir"])
        success = process_benchmark(
            config["repo"],
            config["commit"],
            output_dir,
        )
        if not success:
            print(f"Failed to process {benchmark}")


if __name__ == "__main__":
    main()
