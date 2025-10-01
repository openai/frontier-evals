from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import structlog.stdlib
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, create_repo, upload_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
DEFAULT_CARD_PATH = DEFAULT_PAPERS_DIR / "README.md"
DEFAULT_YAML_CARD_PATH = DEFAULT_PAPERS_DIR / "dataset_card.yaml"

logger = structlog.stdlib.get_logger(component=__name__)


def count_rubric_tasks(rubric: dict) -> int:
    count = 1
    for subtask in rubric.get("sub_tasks", []):
        count += count_rubric_tasks(subtask)
    return count


def extract_paper_metadata(paper_dir: Path, repo_id: str) -> dict:
    paper_id = paper_dir.name
    row = {}

    with open(paper_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
        row["id"] = config.get("id", paper_id)
        row["title"] = config.get("title", "")

    with open(paper_dir / "blacklist.txt") as f:
        row["blacklisted_sites"] = [
            line.strip() for line in f if line.strip() if line.strip() != "none"
        ]

    with open(paper_dir / "rubric.json") as f:
        row["num_rubric_tasks"] = count_rubric_tasks(json.load(f))

    # Collect all files in the paper directory
    reference_files = []
    reference_file_urls = []
    reference_file_hf_uris = []

    for file_path in sorted(paper_dir.rglob("*")):
        if file_path.is_file():
            relative_path = file_path.relative_to(paper_dir.parent)
            relative_path_str = str(relative_path)

            reference_files.append(relative_path_str)
            reference_file_urls.append(
                f"https://huggingface.co/datasets/{repo_id}/resolve/main/{relative_path_str}"
            )
            reference_file_hf_uris.append(f"hf://datasets/{repo_id}@main/{relative_path_str}")

    row["reference_files"] = reference_files
    row["reference_file_urls"] = reference_file_urls
    row["reference_file_hf_uris"] = reference_file_hf_uris

    return row


def build_manifest(papers_dir: Path, repo_id: str) -> list[dict]:
    rows = []
    for paper_dir in sorted(papers_dir.iterdir()):
        if not paper_dir.is_dir() or paper_dir.name.startswith("."):
            continue
        rows.append(extract_paper_metadata(paper_dir, repo_id))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload the PaperBench papers dataset to the Hugging Face Hub.",
    )
    parser.add_argument(
        "repo",
        help=(
            "Target dataset repository. Accepts either a full '<namespace>/<name>' "
            "or just the repository name (defaults to your user namespace)."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the Hugging Face dataset repository as private.",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch or revision to push to (default: main).",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload PaperBench papers dataset",
        help="Custom commit message for the dataset upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: call login here

    api = HfApi()
    user_info = api.whoami()
    username = user_info["name"]
    repo_id = f"{username}/{args.repo}"

    try:
        api.repo_info(repo_id, repo_type="dataset")
        logger.info(f"Repository {repo_id} already exists")
    except:  # noqa: E722
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        logger.info(f"Created repository: {repo_id}")

    # Build manifest for dataset viewer
    logger.info("Building dataset manifest...")
    manifest_rows = build_manifest(DEFAULT_PAPERS_DIR, repo_id)
    logger.info(f"Found {len(manifest_rows)} papers")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy all paper directories
        logger.info("Copying paper directories...")
        for paper_dir in DEFAULT_PAPERS_DIR.iterdir():
            if paper_dir.is_dir() and not paper_dir.name.startswith("."):
                shutil.copytree(paper_dir, temp_path / paper_dir.name)

        data_dir = temp_path / "data"
        data_dir.mkdir(exist_ok=True)

        # Create and save dataset as parquet
        logger.info("Creating dataset parquet file...")
        dataset = Dataset.from_list(manifest_rows)
        parquet_path = data_dir / "train-00000-of-00001.parquet"
        dataset.to_parquet(str(parquet_path))
        logger.info(f"Dataset preview schema: {dataset.column_names}")

        # Upload the prepared directory
        logger.info(f"Uploading to {repo_id}...")
        try:
            api.upload_folder(
                folder_path=str(temp_path),
                repo_id=repo_id,
                repo_type="dataset",
                revision=args.branch,
                commit_message=args.commit_message,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to upload dataset folder: {exc}")
            sys.exit(1)

    # Upload README card
    logger.info("Uploading README.md...")
    readme_content = DEFAULT_CARD_PATH.read_text(encoding="utf-8")

    upload_file(
        path_or_fileobj=io.BytesIO(readme_content.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        revision=args.branch,
        commit_message=args.commit_message,
    )
    logger.info("README.md uploaded successfully")

    yaml_content = DEFAULT_YAML_CARD_PATH.read_text(encoding="utf-8")
    upload_file(
        path_or_fileobj=io.BytesIO(yaml_content.encode("utf-8")),
        path_in_repo="dataset_card.yaml",
        repo_id=repo_id,
        repo_type="dataset",
        revision=args.branch,
        commit_message=args.commit_message,
    )
    logger.info("dataset_card.yaml uploaded successfully")

    logger.info("Dataset upload complete.")
    logger.info(f"View at: https://huggingface.co/datasets/{repo_id}")
    ensure_upload(repo_id)


def ensure_upload(repo_id: str = "josancamon/paperbench", paper_idx: int = 0):
    logger.info("Ensuring upload...")
    dataset = load_dataset(repo_id)
    api = HfApi()
    paper_id = dataset["train"][paper_idx]["id"]
    downloaded_paths = []
    for file in dataset["train"][paper_idx]["reference_files"]:
        local_path = api.hf_hub_download(repo_id=repo_id, filename=file, repo_type="dataset")
        downloaded_paths.append(local_path)

    paper_path = Path(downloaded_paths[0]).parent
    logger.info(f"Downloaded paper and rubric for {paper_id} successfully to: {paper_path}")


if __name__ == "__main__":
    main()
