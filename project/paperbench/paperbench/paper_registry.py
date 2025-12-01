from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog.stdlib

from paperbench.utils import get_paperbench_data_dir, load_yaml_dict

logger = structlog.stdlib.get_logger(component=__name__)


@dataclass(frozen=True)
class Paper:
    id: str
    title: str
    paper_pdf: Path
    paper_md: Path
    addendum: Path
    judge_addendum: Path
    assets: Path
    blacklist: Path
    rubric: Path

    def __post_init__(self) -> None:
        assert isinstance(self.id, str), "Paper id must be a string."
        assert isinstance(self.title, str), "Paper title must be a string."
        assert isinstance(self.paper_pdf, Path), "Paper PDF must be a Path."
        assert isinstance(self.paper_md, Path), "Paper MD must be a Path."
        assert isinstance(self.addendum, Path), "Addendum must be a Path."
        assert isinstance(self.judge_addendum, Path), "Judge addendum must be a Path."
        assert isinstance(self.assets, Path), "Assets must be a Path."
        assert isinstance(self.rubric, Path), "Rubric must be a Path."
        assert isinstance(self.blacklist, Path), "Blacklist must be a Path."
        assert len(self.id) > 0, "Paper id cannot be empty."
        assert len(self.title) > 0, "Paper title cannot be empty."

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Paper:
        try:
            return Paper(
                id=data["id"],
                title=data["title"],
                paper_pdf=data["paper_pdf"],
                paper_md=data["paper_md"],
                addendum=data["addendum"],
                judge_addendum=data["judge_addendum"],
                assets=data["assets"],
                rubric=data["rubric"],
                blacklist=data["blacklist"],
            )
        except KeyError as e:
            raise ValueError(f"Missing key in paper config! {e}") from e


class PaperRegistry:
    def get_paper(self, paper_id: str) -> Paper:
        """Fetch the paper from the registry."""

        config_path = self.get_papers_dir() / paper_id / "config.yaml"
        config = load_yaml_dict(config_path)

        paper_pdf = self.get_papers_dir() / paper_id / "paper.pdf"
        paper_md = self.get_papers_dir() / paper_id / "paper.md"
        addendum = self.get_papers_dir() / paper_id / "addendum.md"
        judge_addendum = self.get_papers_dir() / paper_id / "judge.addendum.md"
        assets = self.get_papers_dir() / paper_id / "assets"
        rubric = self.get_papers_dir() / paper_id / "rubric.json"
        blacklist = self.get_papers_dir() / paper_id / "blacklist.txt"
        return Paper.from_dict(
            {
                **config,
                "paper_pdf": paper_pdf,
                "paper_md": paper_md,
                "addendum": addendum,
                "judge_addendum": judge_addendum,
                "assets": assets,
                "rubric": rubric,
                "blacklist": blacklist,
            }
        )

    def get_papers_dir(self) -> Path:
        """Retrieves the papers directory within the registry."""

        return get_paperbench_data_dir() / "papers"

    def list_paper_ids(self) -> list[str]:
        """List all paper IDs available in the registry, sorted alphabetically."""

        paper_configs = self.get_papers_dir().rglob("config.yaml")
        paper_ids = [f.parent.stem for f in sorted(paper_configs)]

        return paper_ids


paper_registry = PaperRegistry()
