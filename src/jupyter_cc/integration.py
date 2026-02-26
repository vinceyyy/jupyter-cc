"""
Jupyter notebook integration for jupyter_cc.
Handles cell creation, code display, and notebook-specific functionality.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .magics import ClaudeCodeMagics


def create_approval_cell(
    parent: "ClaudeCodeMagics",
    code: str,
    request_id: str,
    should_cleanup_prompts: bool,
    tool_use_id: str | None = None,
    description: str = "",
) -> None:
    """Create a cell for user approval of code execution."""
    marker_id = tool_use_id if tool_use_id else request_id
    # Add a [CC] comment describing what this cell does
    if description:
        marker = f"# [CC] {description}"
    else:
        marker = "# [CC]"
    marked_code = f"{marker}\n{code}"

    # Store code and request ID in IPython namespace for access
    if parent.shell is not None:
        # Store the main request ID
        parent.shell.user_ns["_claude_request_id"] = request_id

        # Initialize cell queue if it doesn't exist
        if "_claude_cell_queue" not in parent.shell.user_ns:
            parent.shell.user_ns["_claude_cell_queue"] = []

        # Add to queue with metadata
        cell_info: dict[str, Any] = {
            "code": marked_code,  # Store the marked version
            "original_code": code,  # Store original for reporting
            "tool_use_id": tool_use_id,
            "request_id": request_id,
            "marker_id": marker_id,
            "marker": marker,
            "executed": False,
        }
        parent.shell.user_ns["_claude_cell_queue"].append(cell_info)

    queue_position = len(parent.shell.user_ns["_claude_cell_queue"]) if parent.shell else 0

    if queue_position == 1:
        # Store the code to be prepopulated after the async operation completes
        if parent.shell is not None:
            parent.shell.user_ns["_claude_pending_input"] = marked_code


def adjust_cell_queue_markers(parent: "ClaudeCodeMagics") -> None:
    """Finalize cell markers after all tool calls complete."""
    if parent.shell is None:
        return

    cell_queue = parent.shell.user_ns.get("_claude_cell_queue", [])
    if not cell_queue:
        return

    # Ensure the first cell is set as pending input
    parent.shell.user_ns["_claude_pending_input"] = cell_queue[0]["code"]


def process_cell_queue(parent: "ClaudeCodeMagics") -> None:
    """Process the cell queue after a successful cell execution."""
    if parent.shell is None:
        return

    cell_queue = parent.shell.user_ns.get("_claude_cell_queue", [])
    if not cell_queue:
        return

    # Find the next unexecuted cell
    next_cell_index = None
    for i, cell_info in enumerate(cell_queue):
        if not cell_info["executed"]:
            next_cell_index = i
            # Set this as the next input (use marked code)
            parent.shell.set_next_input(cell_info.get("code", ""))
            break

    if next_cell_index is not None:
        # Only show "Next cell ready" if there are more cells after this one
        remaining = sum(1 for cell in cell_queue[next_cell_index:] if not cell.get("executed", False))
        if remaining > 0:
            from .display import display_status  # lazy import to avoid circular dependency

            next_cell_marker_id = cell_queue[next_cell_index]["marker_id"]
            display_status(
                f"📋 Next cell ready (Claude cell [{next_cell_marker_id}])",
                kind="info",
            )
    elif len(cell_queue) > 1:
        # All cells have been executed
        if all(cell["executed"] for cell in cell_queue):
            from .display import display_status  # lazy import to avoid circular dependency

            # Check if any had exceptions
            had_exceptions = any(cell.get("had_exception", False) for cell in cell_queue)
            if had_exceptions:
                display_status(
                    "⚠️ All of Claude's generated cells processed (some with errors)",
                    kind="warning",
                )
            else:
                display_status(
                    "✅ All of Claude's generated cells have been processed successfully",
                    kind="success",
                )


_is_jupyter_cached: bool | None = None


def is_in_jupyter_notebook() -> bool:
    """Check if we're running in a Jupyter notebook (vs IPython terminal).

    Result is cached since the environment doesn't change during a session.
    """
    global _is_jupyter_cached  # noqa: PLW0603
    if _is_jupyter_cached is None:
        from IPython import get_ipython  # type: ignore[attr-defined]

        ip = get_ipython()
        _is_jupyter_cached = ip is not None and hasattr(ip, "kernel")
    return _is_jupyter_cached
