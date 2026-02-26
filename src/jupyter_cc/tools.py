"""MCP tools for kernel state inspection."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

import anyio
from claude_agent_sdk import tool

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell

_FILTERED_NAMES = frozenset({"In", "Out", "exit", "quit"})
_REPR_MAX = 100
_INSPECT_REPR_MAX = 10_000

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


def inspect_variable_impl(shell: InteractiveShell, name: str) -> dict[str, Any]:
    """Return detailed info about a single variable.

    Args:
        shell: IPython shell instance with user_ns namespace.
        name: Variable name to inspect.

    Returns:
        Dict with keys "name", "type", "repr", "attributes", "extras".

    Raises:
        KeyError: If the variable is not found in the kernel namespace.
    """
    if name not in shell.user_ns:
        raise KeyError(f"Variable '{name}' not found in kernel namespace")

    value = shell.user_ns[name]
    type_name = type(value).__name__

    try:
        full_repr = repr(value)
        if len(full_repr) > _INSPECT_REPR_MAX:
            full_repr = full_repr[: _INSPECT_REPR_MAX - 3] + "..."
    except Exception:
        full_repr = f"<{type_name} object>"

    attributes = [a for a in dir(value) if not a.startswith("_")]

    extras: dict[str, Any] = {}
    if hasattr(value, "shape"):
        with suppress(Exception):
            extras["shape"] = str(value.shape)
    if hasattr(value, "columns"):
        with suppress(Exception):
            extras["columns"] = list(value.columns)
    if hasattr(value, "dtypes"):
        with suppress(Exception):
            extras["dtypes"] = str(value.dtypes)
    if isinstance(value, dict):
        extras["length"] = len(value)
        extras["keys"] = list(value.keys())[:50]
    elif isinstance(value, (list, tuple, set, frozenset)):
        extras["length"] = len(value)

    return {
        "name": name,
        "type": type_name,
        "repr": full_repr,
        "attributes": attributes,
        "extras": extras,
    }


@tool(
    "inspect_variable",
    "Get detailed information about a specific variable in the IPython kernel",
    {"name": str},
)
async def inspect_variable_tool(args: dict[str, Any]) -> dict[str, Any]:
    """MCP tool: inspect a single variable in detail."""
    await anyio.lowlevel.checkpoint()
    if _shell is None:
        return {"content": [{"type": "text", "text": "Shell not available"}], "is_error": True}

    name = args.get("name", "")
    if not name:
        return {"content": [{"type": "text", "text": "Parameter 'name' is required"}], "is_error": True}

    try:
        info = inspect_variable_impl(_shell, name)
    except KeyError as e:
        return {"content": [{"type": "text", "text": str(e)}], "is_error": True}

    lines = [
        f"Name: {info['name']}",
        f"Type: {info['type']}",
        f"Value: {info['repr']}",
    ]
    if info["extras"]:
        lines.append("Details:")
        for k, v in info["extras"].items():
            lines.append(f"  {k}: {v}")
    lines.append(f"Attributes: {', '.join(info['attributes'][:30])}")
    if len(info["attributes"]) > 30:
        lines.append(f"  ... and {len(info['attributes']) - 30} more")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}
