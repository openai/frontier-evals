import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import blobfile as bf
import structlog.stdlib
from unidecode import unidecode

from paperbench.paper_registry import Paper

logger = structlog.stdlib.get_logger(component=__name__)


@dataclass
class ViolationContext:
    """A violation with surrounding context from the log file."""

    line_number: int  # 1-indexed line number where violation was found
    violation: str  # The actual violation (blacklisted URL)
    context: list[str]  # Lines of context around the violation
    context_start: int  # 1-indexed line number where context starts

    def to_dict(self) -> dict[str, Any]:
        """Convert the violation context to a dictionary."""
        return {
            "line_number": self.line_number,
            "violation": self.violation,
            "context": self.context,
            "context_start": self.context_start,
        }


@dataclass
class MonitorResult:
    """Result from monitoring an agent's logs."""

    violations: list[ViolationContext]  # List of violations found
    explanation: str  # Explanation of violations or why there were none
    log_file: str  # blobfile path to the log file that was checked

    def to_dict(self) -> dict[str, Any]:
        """Convert the monitor result to a dictionary."""
        return {
            "violations": [v.to_dict() for v in self.violations],
            "explanation": self.explanation,
            "log_file": self.log_file,
        }


class Monitor(ABC):
    """Base class for monitoring agent behavior through logs."""

    def __init__(
        self,
        paper: Paper,
    ):
        self.paper = paper

        # Load blacklist from paper
        if not self.paper.blacklist.exists():
            logger.warning(f"Blacklist file {self.paper.blacklist} does not exist!")
            self.blacklist = []
        else:
            self.blacklist = self.paper.blacklist.read_text().splitlines()
            # Remove empty lines and comments
            self.blacklist = [
                line.strip() for line in self.blacklist if line.strip() and not line.startswith("#")
            ]
            # Treat 'none' as an empty blacklist
            if self.blacklist == ["none"]:
                self.blacklist = []

    @abstractmethod
    def check_log(self, log_file: str) -> MonitorResult:
        """Check a log file for violations of monitoring rules."""
        raise NotImplementedError()


class BasicMonitor(Monitor):
    """Simple implementation that checks for occurrences of blacklisted terms with git clone, curl, or wget commands in agent logs."""

    def _normalize_url(self, url: str) -> str:
        """Normalize URL by removing protocol, parameters, and anchors."""
        # Remove protocol (http:// or https://)
        url = re.sub(r"^https?://", "", url)
        # Remove query parameters and anchors
        url = re.sub(r"[?#].*$", "", url)
        return url.strip()

    def _get_context_lines(
        self, lines: list[str], violation_line: int, context_size: int = 3
    ) -> tuple[list[str], int]:
        """Get context lines around a violation.

        Args:
            lines: All lines from the log file
            violation_line: 1-indexed line number where violation was found
            context_size: Number of lines before and after to include

        Returns:
            Tuple of (context lines, context start line number)
        """
        start = max(0, violation_line - context_size - 1)  # -1 to convert to 0-indexed
        end = min(len(lines), violation_line + context_size)  # No -1 needed since end is exclusive
        return lines[start:end], start + 1  # +1 to convert back to 1-indexed

    def _contains_command(self, context: list[str]) -> bool:
        """Check if the context contains git clone, curl, or wget commands.

        Args:
            context: List of context lines

        Returns:
            True if the context contains one of the targeted commands
        """
        commands = ["git clone", "curl", "wget"]

        for line in context:
            for cmd in commands:
                if cmd in line:
                    return True
        return False

    def check_log(self, log_file: str) -> MonitorResult:
        """Check a log file for violations of the blacklist.

        Args:
            log_file: blobfile path to the log file to check

        Returns:
            MonitorResult containing any violations found
        """
        if not bf.exists(log_file):
            return MonitorResult(
                violations=[],
                explanation=f"Log file not found at {log_file}",
                log_file=log_file,
            )

        if not self.blacklist:
            return MonitorResult(
                violations=[],
                explanation="No blacklist entries - all URLs allowed",
                log_file=log_file,
            )

        violations = []
        with bf.BlobFile(log_file, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue

                for blacklisted in self.blacklist:
                    normalized_blacklist = self._normalize_url(blacklisted)
                    if normalized_blacklist in line:
                        context_lines, context_start = self._get_context_lines(lines, i)
                        # Clean context lines of unicode characters
                        cleaned_context = [self._clean_unicode(line) for line in context_lines]

                        # Only consider it a violation if it appears with git clone, curl, or wget
                        if self._contains_command(cleaned_context):
                            violation_context = ViolationContext(
                                line_number=i,
                                violation=blacklisted,
                                context=cleaned_context,
                                context_start=context_start,
                            )

                            violations.append(violation_context)
                            logger.warning(
                                f"Found blacklisted term '{blacklisted}' with command at line {i} in {log_file}"
                            )

        if violations:
            explanation = (
                f"Found {len(violations)} violations of blacklist with git clone/curl/wget commands"
            )
        else:
            explanation = "No violations found with git clone/curl/wget commands"

        return MonitorResult(
            violations=violations,
            explanation=explanation,
            log_file=log_file,
        )

    def _clean_unicode(self, text: str) -> str:
        """Clean unicode characters from text to improve readability.

        Args:
            text: The text to clean

        Returns:
            Cleaned text with unicode characters transliterated to ASCII
        """
        if not text:
            return ""

        # Remove NULL bytes
        text = re.sub(r"\\u0000", "", text)

        cleaned = unidecode(text)

        return cleaned
