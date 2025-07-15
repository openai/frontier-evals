import re
from pathlib import Path

import structlog.stdlib
from drain3 import TemplateMiner

logger = structlog.stdlib.get_logger(component=__name__)


def get_model_context_window_length(model: str) -> int:
    max_context_window_lengths = {
        "gpt-4o-mini": 128000,
        "gpt-4o-mini-2024-07-18": 128000,
        "gpt-4o": 128000,
        "gpt-4o-2024-08-06": 128000,
        "o1-mini": 128000,
        "o1-mini-2024-09-12": 128000,
        "o1": 200000,
        "o1-2024-12-17": 200000,
        "o3-mini-2024-12-17": 128000,
        "o3-mini-2025-01-31": 200000,
        "o3-mini": 200000,
        "o1-preview": 128000,
        "gpt-4-turbo": 128000,
    }
    if model not in max_context_window_lengths:
        raise ValueError(f"Context window length not defined for model {model}!")
    return max_context_window_lengths[model]


def sanitize_line(line: str) -> str:
    """
    Convert ephemeral bits (timestamps, progress bars, numeric tokens, IPs, etc.)
    into placeholders so that repeated patterns can be more easily detected.
    """

    # Mask ISO8601 Timestamps (e.g. 2025-01-28T18:47:06.1465140Z)
    line = re.sub(r"\d{4}-\d{2}-\d{2}T[0-9:.]+Z", "<TIMESTAMP>", line)

    # Mask typical date/time strings (e.g. 2025-01-28 18:47:06 or 18:47:06)
    line = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", "<DATE>", line)
    line = re.sub(r"\b\d{2}:\d{2}:\d{2}\b", "<TIME>", line)

    # TQDM or other progress bars: remove generic progress bar lines by matching percentage and bar or repeated progress symbols
    if (
        re.search(r"\d+%?\|[█=]+", line)
        or re.search(r"[KMG]?B/s", line)
        or re.search(r"\d+%\s*\|", line)
        or re.search(r"[▏▎▍▌▋▊▉]{2,}", line)
    ):
        line = "<PROGRESS_BAR>"

    # IP addresses  (1-3 digits).(1-3).(1-3).(1-3)
    line = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "<IP>", line)

    # Mask long hex strings (common in commit hashes, container IDs, etc.)
    line = re.sub(r"\b[0-9a-fA-F]{8,}\b", "<HEX>", line)

    return line


def reduce_log(input_string: str) -> str:
    """
    Reduce a multi-line log string to a filtered version with repeated lines collapsed.
    """
    template_miner = TemplateMiner()
    output_lines = []

    previous_cluster_id = None
    repeat_count = 1

    for raw_line in input_string.splitlines():
        original_line = raw_line
        sanitized = sanitize_line(original_line)

        result = template_miner.add_log_message(sanitized)
        cluster_id = result["cluster_id"]

        if previous_cluster_id is None:
            # First line
            output_lines.append(original_line)
            previous_cluster_id = cluster_id
            continue

        if cluster_id == previous_cluster_id:
            repeat_count += 1
        else:
            if repeat_count > 1:
                output_lines.append(f"  (repeated {repeat_count} times)")
            output_lines.append(original_line)
            repeat_count = 1
            previous_cluster_id = cluster_id

    if previous_cluster_id is not None and repeat_count > 1:
        output_lines.append(f"  (repeated {repeat_count} times)")

    return "\n".join(output_lines)


# just to test the process_log fn
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Process log files and collapse repeated lines.")
    parser.add_argument(
        "-i",
        "--input-file",
        type=Path,
        help="Path to the input log file to be cleaned",
    )

    args = parser.parse_args()

    input_text = args.input_file.read_text()

    print(reduce_log(input_text))


def format_file(file_path: Path, file_content: str):
    return f"""<FILE:{file_path}>
{file_content if file_content.strip() else "(FILE IS EMPTY)"}
</FILE:{file_path}>"""
