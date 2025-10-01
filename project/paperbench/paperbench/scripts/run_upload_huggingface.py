from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file, whoami
import structlog.stdlib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
DEFAULT_CARD_PATH = DEFAULT_PAPERS_DIR / "HUGGINGFACE_CARD.md"

logger = structlog.stdlib.get_logger(component=__name__)


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
    parser.add_argument(
        "--card-path",
        type=Path,
        default=DEFAULT_CARD_PATH,
        help="Path to the dataset card markdown file (default: data/papers/HUGGINGFACE_CARD.md).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    card_path = args.card_path.resolve()

    if not card_path.exists() or not card_path.is_file():
        logger.error(f"Card file not found: {card_path}")
        sys.exit(1)

    HF_TOKEN = os.getenv("HF_TOKEN")
    if not HF_TOKEN:
        logger.error("No Hugging Face token found. Login with 'huggingface-cli login'")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)
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
            token=HF_TOKEN,
            private=args.private,
            exist_ok=True,
        )
        logger.info(f"Created repository: {repo_id}")

    try:
        api.upload_folder(
            folder_path=str(DEFAULT_PAPERS_DIR),
            repo_id=repo_id,
            repo_type="dataset",
            revision=args.branch,
            commit_message=args.commit_message,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to upload dataset folder: {exc}")
        sys.exit(1)

    try:
        readme_content = card_path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to read card file: {exc}")
        sys.exit(1)

    try:
        upload_file(
            path_or_fileobj=io.BytesIO(readme_content.encode("utf-8")),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            revision=args.branch,
            commit_message=args.commit_message,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to upload README.md: {exc}")
        sys.exit(1)

    logger.info("Dataset upload complete.")


def verify_upload(api: HfApi, repo_id: str) -> None:
    """Verify that the dataset has been uploaded correctly."""
    repo_info = api.repo_info(repo_id, repo_type="dataset")
    print(repo_info.__dict__.keys())


if __name__ == "__main__":
    main()
    # verify_upload(HfApi(token=os.getenv("HF_TOKEN")), "josancamon/paperbench")
