#!/usr/bin/env python
"""
llm_edit – bulk-apply an LLM prompt to many source files.

Default layout
    issues/<issue_id>/test.py

For every matching file the script:
1. Reads the file and interpolates its contents into a prompt template
   (placeholder `{input_file}`).
2. Sends the prompt to the OpenAI chat API (model `o3` by default) with
   exponential-back-off retries.
3. Overwrites the file when the reply is wrapped in <OUTPUT>…</OUTPUT>.
   If the reply is wrapped in <NO-OP>…</NO-OP> the file is left unchanged (no operation).
   If the reply is wrapped in <ERROR>…</ERROR> the file is left unchanged and
   the explanation is collected for a JSON report.

Extras
- Async I/O with bounded concurrency.
- Allow / deny lists for issue IDs (via text files, one ID per line).
- Optional `ruff format` pass on the directory after conversion.
- Progress bar via `tqdm`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
from pathlib import Path
from string import Template
from typing import Iterable

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

_OUTPUT_RE = re.compile(r"<OUTPUT>(.*?)</OUTPUT>", re.DOTALL)
_NOOP_RE = re.compile(r"<NO-OP>(.*?)</NO-OP>", re.DOTALL)
_ERROR_RE = re.compile(r"<ERROR>(.*?)</ERROR>", re.DOTALL)


class MdTemplate(Template):
    delimiter = "§"


def load_id_file(path: str | None) -> set[str]:
    """Return a set of IDs from a newline-separated file, or an empty set."""
    if not path:
        return set()
    with Path(path).expanduser().open() as f:
        return {line.strip() for line in f if line.strip()}


def iter_target_files(
    root: Path,
    filename: str,
    whitelist: set[str] | None,
    blacklist: set[str],
) -> Iterable[tuple[str, Path]]:
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        if whitelist and sub.name not in whitelist:
            continue
        if sub.name in blacklist:
            continue
        candidate = sub / filename
        if candidate.exists():
            yield sub.name, candidate


def parse_response(text: str) -> tuple[str, str]:
    """Return (status, payload_or_reason). Status is one of 'OUTPUT', 'NO-OP', or 'ERROR'."""
    if m := _OUTPUT_RE.search(text):
        return "OUTPUT", m.group(1).strip()
    if m := _NOOP_RE.search(text):
        return "NO-OP", m.group(1).strip()
    if m := _ERROR_RE.search(text):
        return "ERROR", m.group(1).strip()
    return "ERROR", "Malformed response: missing <OUTPUT>/<NO-OP>/<ERROR> tags."


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
async def call_openai(client: openai.AsyncClient, model: str, prompt: str) -> str:
    """Send a prompt to the LLM and return the reply text."""
    resp = await client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content  # type: ignore[return-value]


async def handle_file(
    semaphore: asyncio.Semaphore,
    client: openai.AsyncClient,
    model: str,
    prompt_tmpl: str,
    issue_id: str,
    path: Path,
) -> tuple[str, str, str]:
    async with semaphore:
        original = path.read_text()
        prompt = MdTemplate(prompt_tmpl).substitute(input_file=original)
        reply = await call_openai(client, model, prompt)
        status, payload = parse_response(reply)
        if status == "OUTPUT":
            path.write_text(payload)
        return issue_id, status, payload


async def async_main(args: argparse.Namespace) -> None:
    root = Path(args.directory).expanduser().resolve()

    whitelist = load_id_file(args.whitelist)
    blacklist = load_id_file(args.blacklist)
    if whitelist and blacklist:
        print("Both whitelist and blacklist supplied; blacklist takes precedence.")

    prompt_template = Path(args.prompt).read_text()

    targets = list(iter_target_files(root, args.filename, whitelist or None, blacklist))
    if not targets:
        print("No files to process.")
        return

    client = openai.AsyncClient()
    sem = asyncio.Semaphore(args.concurrency)
    coros = [
        handle_file(sem, client, args.model, prompt_template, issue, path)
        for issue, path in targets
    ]

    failures: dict[str, str] = {}
    with tqdm(total=len(coros), unit="file", desc="Processing") as bar:
        for fut in asyncio.as_completed(coros):
            issue, status, msg = await fut
            bar.update(1)
            if status == "OUTPUT":
                label = "OK"
            elif status == "NO-OP":
                label = "NO-OP"
            else:
                label = "FAIL"
            bar.write(f"[{issue}] {label}")
            if status == "ERROR":
                failures[issue] = msg

    if failures:
        report = root / "llm_compile_errors.json"
        report.write_text(json.dumps(failures, indent=2))
        print(f"\nSaved explanations to {report}")

    if args.format:
        subprocess.run(["uv", "run", "ruff", "format", str(root)], check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="issues")
    parser.add_argument("--filename", default="test.py", help="File name to transform.")
    parser.add_argument("--prompt", default="prompt.md", help="Path to prompt template.")
    parser.add_argument("--model", default="o3")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument(
        "--whitelist",
        help="Path to a file with issue IDs to INCLUDE (one per line).",
    )
    parser.add_argument(
        "--blacklist",
        help="Path to a file with issue IDs to EXCLUDE (one per line).",
    )
    parser.add_argument("--format", action="store_true", help="Run ruff format afterwards.")
    return parser.parse_args()


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
