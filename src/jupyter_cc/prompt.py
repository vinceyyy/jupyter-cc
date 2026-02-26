"""
Prompt building functionality for jupyter_cc.
Handles system prompts and content preparation.
"""

from __future__ import annotations

from pathlib import Path

from .constants import EXECUTE_PYTHON_TOOL_NAME


def get_system_prompt(is_ipython: bool, max_cells: int) -> str:
    """Generate the system prompt for Claude based on environment.

    Args:
        is_ipython: Whether running in IPython (vs Jupyter notebook)
        max_cells: Maximum number of cells Claude can create

    Returns:
        Complete system prompt string
    """
    if is_ipython:
        env = "shared IPython session"
        tool_call_result = (
            f"The {EXECUTE_PYTHON_TOOL_NAME} tool call will populate the next input with the Python code you provide."
        )
        preference = f"""You can only call {EXECUTE_PYTHON_TOOL_NAME} once, since the IPython terminal does not allow for multiple pending code blocks.

The user will see the code block and can choose to execute it or not."""
    else:
        env = "Jupyter notebook"
        tool_call_result = (
            f"Each {EXECUTE_PYTHON_TOOL_NAME} call will create a new cell in the user's Jupyter notebook interface."
        )
        preference = f"""IMPORTANT: Prefer to call {EXECUTE_PYTHON_TOOL_NAME} only ONCE with a short code snippet.
As a last resort, you may call it multiple times to split up a large code block. You can make at most {max_cells} calls per turn (i.e., in response to each user prompt).
The user will be presented with the code blocks one by one.
If the user executes it and it succeeds, the next code block gets created in a new cell.
If the user executes it and it errors, then the error will get reported, but the chain is broken. Assume that the user does not see the subsequent code.
If the user executes a cell that is not the next code block, then the chain will pause until the proper next code block is executed.

If the user asks you to modify code in the current cell, you may do this by using the {EXECUTE_PYTHON_TOOL_NAME} tool EXACTLY ONCE.
Identifying that the current cell is the target is obvious because the code block is included directly in the user's request itself.

If the user asks you to edit/change/modify code in a DIFFERENT cell, inform them that you do not have that capability.
Instead, suggest that they use `%%cc edit this cell to <requested edits>` at the top of the cell they would like to edit.
Respond ONLY with that suggestion. DO NOT create new cells for the request and DEFINITELY DO NOT use the {EXECUTE_PYTHON_TOOL_NAME} tool."""

    system_prompt_preamble = f"""You are operating in a {env}.
You can see the current session state. You can create new code cells using the {EXECUTE_PYTHON_TOOL_NAME} tool.
{tool_call_result}
Never call {EXECUTE_PYTHON_TOOL_NAME} if you can answer the user's question directly with text.
{preference}
"""
    system_prompt_image_capture = """Images from display() calls (matplotlib, seaborn, PIL, etc.) are automatically captured. You do not need any special wrapper — just write normal plotting code and the images will be available to you on the next turn.

You also have two kernel inspection tools available:
- list_variables: Lists all user-defined variables with types and values
- inspect_variable: Gets detailed info about a specific variable (full repr, attributes, shape/columns for DataFrames, etc.)

Use these tools when you need to understand the current kernel state without creating code cells."""

    system_prompt_tool_usage = f"""For any questions you can answer on your own, DO NOT use {EXECUTE_PYTHON_TOOL_NAME}.
Don't forget that you have other built-in tools like Read. Try responding using your built-in tools first without using {EXECUTE_PYTHON_TOOL_NAME}. Your response does not need to invoke {EXECUTE_PYTHON_TOOL_NAME}.
If you want to explain something to the user, do not put your explanation in {EXECUTE_PYTHON_TOOL_NAME}. Just return regular prose.

Examples:
<basic-example>
    <request>Help me understand what's in the dataframe `my_df`</request>
    This request should use {EXECUTE_PYTHON_TOOL_NAME} to generate the code for inspecting the dataframe.
</basic-example>

<no-python-example>
    <request>Explain how cc_jupyter/magics.py works</request>
    This request should not use {EXECUTE_PYTHON_TOOL_NAME} at all, since it can be fulfilled by reading the file.
</no-python-example>

<plot-example>
    <request>Plot the data in `my_df` and save the images to files</request>
    The generated code should create the plot normally with plt.show(). Images are captured automatically.
</plot-example>

IMPORTANT: Do not invoke {EXECUTE_PYTHON_TOOL_NAME} in parallel.
IMPORTANT: Always include a return value or expression at the end of your {EXECUTE_PYTHON_TOOL_NAME} output. Only return values are captured in output cells - print statements are NOT captured.
For example, instead of print(df.head()), use df.head() as the last line.
IMPORTANT: Always provide a short `description` when calling {EXECUTE_PYTHON_TOOL_NAME}. This appears as a comment at the top of the cell so the user knows what the cell does at a glance. Keep it concise (under 80 chars), e.g. "Load CSV and preview first 5 rows" or "Plot monthly revenue with moving average".

If <request> is empty, it is because the user wants you to continue from where you left off in the previous messages."""

    return "\n".join([system_prompt_preamble, system_prompt_image_capture, system_prompt_tool_usage])


def prepare_imported_files_content(imported_files: list[str]) -> str:
    """Prepare content from imported files to include in initial conversation.

    Args:
        imported_files: List of file paths to import

    Returns:
        Formatted string with file contents
    """
    if not imported_files:
        return ""

    files_content = []

    for file_path_str in imported_files:
        file_path = Path(file_path_str)
        if file_path.exists():
            try:
                with file_path.open() as f:
                    content = f.read()
                files_content.append(f"{file_path.name}:\n```\n{content}\n```")
            except Exception:
                pass

    if files_content:
        return (
            "Files imported by the user for your reference. Use this content directly. Don't read them again:\n\n"
            + "\n\n".join(files_content)
        )
    return ""
