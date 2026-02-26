"""MCP tools for kernel state inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyio
from claude_agent_sdk import tool

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell

_FILTERED_NAMES = frozenset({"In", "Out", "exit", "quit"})
_REPR_MAX = 100

# Module-level reference set by magics.py on init
_shell: InteractiveShell | None = None


def _get_truncated_repr(value: Any, max_length: int = _REPR_MAX) -> str:
    """Get truncated repr of a value, with fallback for repr failures."""
    try:
        r = repr(value)
        if len(r) > max_length:
            return r[: max_length - 3] + "..."
        return r
    except Exception:
        return f"<{type(value).__name__} object>"


def _filtered_user_vars(shell: InteractiveShell) -> dict[str, Any]:
    """Return user variables, filtering out internals and special names."""
    return {k: v for k, v in shell.user_ns.items() if not k.startswith("_") and k not in _FILTERED_NAMES}


def list_variables_impl(shell: InteractiveShell) -> list[dict[str, str]]:
    """Return a list of all user variables with name, type, and truncated repr.

    Args:
        shell: IPython shell instance with user_ns namespace.

    Returns:
        Sorted list of dicts, each with keys "name", "type", "repr".
    """
    result = []
    for name in sorted(_filtered_user_vars(shell)):
        value = shell.user_ns[name]
        result.append(
            {
                "name": name,
                "type": type(value).__name__,
                "repr": _get_truncated_repr(value),
            }
        )
    return result


@tool(
    "list_variables",
    "List all user-defined variables in the IPython kernel with their types and values",
    {},
)
async def list_variables_tool(args: dict[str, Any]) -> dict[str, Any]:
    """MCP tool: list all user variables."""
    await anyio.lowlevel.checkpoint()
    if _shell is None:
        return {"content": [{"type": "text", "text": "Shell not available"}], "is_error": True}
    variables = list_variables_impl(_shell)
    if not variables:
        text = "No user-defined variables in the kernel."
    else:
        lines = [f"  {v['name']}: {v['type']} = {v['repr']}" for v in variables]
        text = f"{len(variables)} variable(s):\n" + "\n".join(lines)
    return {"content": [{"type": "text", "text": text}]}
