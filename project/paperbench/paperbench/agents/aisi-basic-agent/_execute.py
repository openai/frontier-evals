from inspect_ai.tool import Tool, tool
from inspect_ai.tool._tool_call import ToolCall, ToolCallContent, ToolCallView, ToolCallViewer
from inspect_ai.util import sandbox


# custom viewer for bash and python code blocks
def code_viewer(language: str, code_param: str) -> ToolCallViewer:
    def viewer(tool_call: ToolCall) -> ToolCallView:
        code = tool_call.arguments.get(code_param, None)
        try:
            code = (code or tool_call.function).strip()
        except Exception as e:
            code = f"ERROR: unable to parse tool call: {e}"
        call = ToolCallContent(
            title=language,
            format="markdown",
            content=f"```{language}\n" + code + "\n```\n",
        )
        return ToolCallView(call=call)

    return viewer


@tool(viewer=code_viewer("bash", "cmd"))
def bash(timeout: int | None = None, user: str | None = None) -> Tool:
    """Bash shell command execution tool.

    Execute bash shell commands using a sandbox environment (e.g. "docker").

    Args:
      timeout (int | None): Timeout (in seconds) for command.
      user (str | None): User to execute commands as.

    Returns:
      String with command output (stdout) or command error (stderr).
    """

    async def execute(cmd: str) -> str:
        """
        Use this function to execute bash commands.

        Args:
          cmd (str): The bash command to execute.

        Returns:
          The output of the command.
        """
        # execute the command
        try:
            # stop hanging on stdin for apply_patch
            cmd_specific_timeout = None
            if "apply_patch" in cmd and "<<" in cmd:
                cmd_specific_timeout = 30

            result = await sandbox().exec(
                cmd=["bash", "--login", "-c", cmd],
                timeout=cmd_specific_timeout,
                user=user,
                env={
                    "DEBIAN_FRONTEND": "noninteractive",
                    "GIT_TERMINAL_PROMPT": "0",
                },
            )
            # return output (including stderr if any)
            output = ""
            if result.stderr:
                output = f"{result.stderr}\n"
            return f"{output}{result.stdout}"
        except Exception as e:
            return f"ERROR: unable to execute command: {e}"

    return execute


@tool(viewer=code_viewer("python", "code"))
def python(timeout: int | None = None, user: str | None = None) -> Tool:
    """Python code execution tool.

    Execute Python code using a sandbox environment (e.g. "docker").

    Args:
      timeout (int | None): Timeout (in seconds) for command.
      user (str | None): User to execute commands as.

    Returns:
      String with command output (stdout) or command error (stderr).
    """

    async def execute(code: str) -> str:
        """
        Use the python function to execute Python code.

        The python function will only return you the stdout of the script,
        so make sure to use print to see the output.

        Args:
          code (str): The python code to execute.

        Returns:
          The output of the Python code.
        """
        result = await sandbox().exec(
            cmd=["python3"],
            input=code,
            timeout=timeout,
            user=user,
        )
        # return output (including stderr if any)
        output = ""
        if result.stderr:
            output = f"{result.stderr}\n"
        return f"{output}{result.stdout}"

    return execute
