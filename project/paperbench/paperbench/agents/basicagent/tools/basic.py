from pathlib import Path
from typing import Any

from openai.types.responses import FunctionToolParam

from nanoeval.solvers.computer_tasks.code_execution_interface import ComputerInterface
from paperbench.agents.basicagent.tools.base import Tool


class SubmitTool(Tool):
    def name(self) -> str:
        return "submit"

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="Signal that you are completely finished.",
            parameters={
                "type": "object",
                "properties": {
                    "end_message": {
                        "type": "string",
                        "description": "Final message to signal that you are finished.",
                    },
                },
                "required": ["end_message"],
                "additionalProperties": False,
            },
            strict=False,
        )

    async def execute(self, computer: ComputerInterface, *args: Any, **kwargs: Any) -> str:
        """No-op, handled in the agent directly."""
        return "SUBMIT"


class BashTool(Tool):
    def name(self) -> str:
        return "bash"

    async def execute(self, computer: ComputerInterface, cmd: str) -> str:
        result = await computer.send_shell_command(
            cmd=cmd,
        )
        return result.output.decode("utf-8").strip()

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="Use this function to execute bash commands.",
            parameters={
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                },
                "required": ["cmd"],
                "additionalProperties": False,
            },
            strict=False,
        )


class PythonTool(Tool):
    def name(self) -> str:
        return "python-tool"  # "python" is reserved by OpenAI

    async def execute(self, computer: ComputerInterface, code: str) -> str:
        result = await computer.send_shell_command("mktemp -d")
        tmp_dir = result.output.decode("utf-8").strip()
        await computer.upload(code.encode("utf-8"), str(Path(tmp_dir) / "code.py"))
        result = await computer.send_shell_command("python3 code.py")
        return result.output.decode("utf-8").strip()

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description=(
                "Use the python function to execute Python code.\n\n"
                "The Python tool executes single-run Python scripts. Important notes:\n"
                "1. Each execution is independent - no state is preserved between runs\n"
                "2. You must explicitly use print() statements to see any output\n"
                "3. Simply writing expressions (like in notebooks) will not display results\n"
                "4. The script cannot accept interactive input during execution\n"
                "5. Return statements alone won't produce visible output\n"
                "6. All variables and imports are cleared between executions\n"
                "7. Standard output (via print()) is the only way to see results"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code to execute.",
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            strict=False,
        )


class ReadFileChunk(Tool):
    def name(self) -> str:
        return "read_file_chunk"

    async def execute(
        self, computer: ComputerInterface, file: str, start_line: int = 1, max_lines: int = 50
    ) -> str:
        if start_line < 1:
            return "ERROR: start_line must be >= 1"

        if max_lines < 1:
            return "ERROR: max_lines must be >= 1"

        if max_lines > 50:
            return "ERROR: max_lines cannot exceed 50"

        try:
            # Read the file
            result = await computer.send_shell_command(f"cat {file}")
            content = result.output.decode("utf-8").strip()

            # Split into lines
            lines = content.splitlines()

            if start_line > len(lines):
                return f"ERROR: start_line ({start_line}) is beyond the total number of lines ({len(lines)}) in the file."
            # Calculate end line
            end_line = min(start_line + max_lines - 1, len(lines))

            # Get the requested chunk
            chunk = lines[start_line - 1 : end_line]

            # Add line numbers and join
            numbered_lines = [f"{i + start_line}: {line}" for i, line in enumerate(chunk)]

            # Add summary info
            total_lines = len(lines)
            summary = (
                f"File has {total_lines} total lines. Showing lines {start_line} to {end_line}.\n\n"
            )

            return summary + "\n".join(numbered_lines)
        except FileNotFoundError:
            return f"ERROR: File '{file}' not found"
        except Exception as e:
            return f"ERROR: Error reading file: {str(e)}"

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="Read a chunk of lines from a file.",
            parameters={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-indexed).",
                        "default": 1,
                    },
                    "max_lines": {
                        "type": "integer",
                        "description": "Maximum number of lines to read (default: 50, max: 50).",
                        "default": 50,
                    },
                },
                "required": ["file"],
                "additionalProperties": False,
            },
            strict=False,
        )


class SearchFile(Tool):
    def name(self) -> str:
        return "search_file"

    async def execute(
        self,
        computer: ComputerInterface,
        file: str,
        query: str,
        context_lines: int = 2,
        max_matches: int = 5,
        page: int = 1,
    ) -> str:
        if not query:
            return "ERROR: Query cannot be empty."
        if context_lines < 0:
            return "ERROR: context_lines must be >= 0"
        if max_matches < 1:
            return "ERROR: max_matches must be >= 1"
        if page < 1:
            return "ERROR: page must be >= 1"

        try:
            # Read the file
            result = await computer.send_shell_command(f"cat {file}")
            content = result.output.decode("utf-8").strip()

            # Split into lines
            lines = content.splitlines()

            # Find all matches (case-insensitive)
            all_matches = []
            query = query.lower()

            for i, line in enumerate(lines):
                if query in line.lower():
                    # Calculate context range
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)

                    # Get context lines
                    context = []
                    for j in range(start, end):
                        prefix = ">>> " if j == i else "    "  # Highlight matching line
                        context.append(f"{prefix}{j + 1}: {lines[j]}")

                    all_matches.append("\n".join(context))

            if not all_matches:
                return f"No matches found for '{query}' in {file}"

            # Calculate pagination
            total_matches = len(all_matches)
            total_pages = (total_matches + max_matches - 1) // max_matches

            if page > total_pages:
                return f"Invalid page number. There are only {total_pages} pages of results."

            start_idx = (page - 1) * max_matches
            end_idx = min(start_idx + max_matches, total_matches)

            # Get matches for this page
            matches = all_matches[start_idx:end_idx]

            # Build summary with pagination info
            summary = [
                f"Found {total_matches} matches for '{query}' in {file}",
                f"Showing matches {start_idx + 1}-{end_idx} (Page {page} of {total_pages})",
                "",  # Empty line for spacing
            ]

            # Add match index to each result
            numbered_matches = []
            for i, match in enumerate(matches, start=start_idx + 1):
                numbered_matches.append(f"[Match {i} of {total_matches}]\n{match}")

            return "\n\n".join(summary + numbered_matches)
        except FileNotFoundError:
            return f"ERROR: File '{file}' not found"
        except Exception as e:
            return f"ERROR: Error searching file: {str(e)}"

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description=(
                "Search for a keyword or phrase in a file and return matching lines with context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file": {
                        "type": "string",
                        "description": "Path to the file to search.",
                    },
                    "query": {
                        "type": "string",
                        "description": "Text to search for (case-insensitive).",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of lines of context to show before and after each match (default: 2).",
                        "default": 2,
                    },
                    "max_matches": {
                        "type": "integer",
                        "description": "Maximum number of matches to return per page (default: 5).",
                        "default": 5,
                    },
                    "page": {
                        "type": "integer",
                        "description": "Which page of results to return (1-indexed, default: 1).",
                        "default": 1,
                    },
                },
                "required": [
                    "file",
                    "query",
                ],
                "additionalProperties": False,
            },
            strict=False,
        )
