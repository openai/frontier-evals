"""
llm_analyze_runs.py – Analyze run.log files in a run-group directory and summarize failure reasons.

For each subdirectory under the specified run-group directory, this script extracts the last N lines
from run.log, invokes an OpenAI LLM (model o4-mini by default) to summarize why the run may have failed,
and writes a JSON report mapping run IDs to summaries.

Usage:
    llm_analyze_runs.py <run_group_dir> [--prompt <prompt_template>]
                             [--model <model>] [--n-lines <N>] [--concurrency <C>]
                             [--output <output_file>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import deque
from pathlib import Path
from string import Template

import openai
import tiktoken
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

_OUTPUT_RE = re.compile(r"<OUTPUT>(.*?)</OUTPUT>", re.DOTALL)
_ERROR_RE = re.compile(r"<ERROR>(.*?)</ERROR>", re.DOTALL)


class MdTemplate(Template):
    delimiter = "§"


def parse_response(text: str) -> tuple[bool, str]:
    """Return (is_successful, payload_or_error_reason)."""
    if m := _OUTPUT_RE.search(text):
        return True, m.group(1).strip()
    if m := _ERROR_RE.search(text):
        return False, m.group(1).strip()
    return False, "Malformed response: missing <OUTPUT>/<ERROR> tags."


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def call_openai(client: openai.AsyncClient, model: str, prompt: str) -> str:
    """Send a prompt to the LLM and return the reply text."""
    resp = await client.chat.completions.create(
        model=model, reasoning_effort="high", messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content  # type: ignore[return-value]


async def handle_run(
    semaphore: asyncio.Semaphore,
    client: openai.AsyncClient,
    model: str,
    prompt_tmpl: str,
    run_id: str,
    run_log: Path,
    n_lines: int,
) -> tuple[str, str]:
    """Process one run: extract last lines, call LLM, parse response."""
    async with semaphore:
        lines: deque[str] = deque(maxlen=n_lines)
        with run_log.open() as f:
            for line in f:
                lines.append(line.rstrip("\n"))
        prompt = MdTemplate(prompt_tmpl).substitute(
            run_id=run_id, n_lines=n_lines, log_lines="\n".join(lines)
        )
        try:
            reply = await call_openai(client, model, prompt)
        except (openai.BadRequestError, RetryError):
            try:
                encoding = tiktoken.encoding_for_model(model)
            except Exception:
                encoding = tiktoken.get_encoding("gpt2")
            log_text = "\n".join(lines)
            tokens = encoding.encode(log_text)
            truncated_log = encoding.decode(tokens[-180000:])
            prompt = MdTemplate(prompt_tmpl).substitute(
                run_id=run_id, n_lines=n_lines, log_lines=truncated_log
            )
            reply = await call_openai(client, model, prompt)
        ok, payload = parse_response(reply)
        return run_id, payload if ok else f"ERROR: {payload}"


async def async_main(args: argparse.Namespace) -> None:
    group_dir = Path(args.group_dir).expanduser().resolve()

    prompt_tmpl = Path(args.prompt).read_text()

    client = openai.AsyncClient()
    sem = asyncio.Semaphore(args.concurrency)
    runs = [d for d in group_dir.iterdir() if d.is_dir() and (d / "run.log").exists()]
    if not runs:
        print(f"No run.log files found under {group_dir}")
        return

    coros = [
        handle_run(sem, client, args.model, prompt_tmpl, d.name, d / "run.log", args.n_lines)
        for d in runs
    ]

    summaries: dict[str, str] = {}
    for fut in tqdm(asyncio.as_completed(coros), total=len(coros), desc="Analyzing runs"):
        run_id, summary = await fut
        summaries[run_id] = summary

    out_path = args.output
    out_path.write_text(json.dumps(summaries, indent=2))
    print(f"Saved analysis to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize failure reasons from run.log files in a run-group directory."
    )
    parser.add_argument("group_dir", help="Path to the run-group directory")
    parser.add_argument(
        "--prompt",
        default="scripts/prompts/analyze_failure.md",
        help="Path to the markdown prompt template",
    )
    parser.add_argument("--model", default="o4-mini", help="LLM model to use")
    parser.add_argument(
        "--n-lines",
        type=int,
        default=10,
        help="Number of last log lines to include in the prompt",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=200,
        help="Maximum number of concurrent LLM calls",
    )
    parser.add_argument(
        "--output",
        default="run_failures_summary.json",
        type=Path,
        help="Output file name for the JSON report",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
