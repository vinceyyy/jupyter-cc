"""
jupyter_cc — Jupyter magic for Claude Code.

Provides %cc magic commands for agentic Claude Code integration in notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path

from .constants import HELP_TEXT
from .display import display_status
from .magics import ClaudeCodeMagics
from .watcher import CellWatcher

__version__ = "1.0.0"

__all__ = [
    "ClaudeCodeMagics",
    "load_ipython_extension",
]

DEFAULT_PERMISSIONS = {
    "permissions": {"allow": ["Bash", "Glob", "Grep", "Read", "Edit", "Write", "WebSearch", "WebFetch"]}
}


def _ensure_claude_settings() -> bool:
    """Create .claude/settings.local.json if not present in cwd."""
    cwd = Path.cwd()
    settings_file = cwd / ".claude" / "settings.local.json"
    if not settings_file.exists():
        settings_file.parent.mkdir(exist_ok=True)
        settings_file.write_text(json.dumps(DEFAULT_PERMISSIONS, indent=2))
        display_status(f"✅ Created {settings_file.relative_to(cwd)}", kind="success")
        return True
    return False


def load_ipython_extension(ipython: object) -> None:
    """Load the jupyter_cc extension."""
    from IPython.core.interactiveshell import InteractiveShell

    if not isinstance(ipython, InteractiveShell):
        return

    created = _ensure_claude_settings()

    # Security warning
    warning_lines = [
        "WARNING: Claude has permissions for Bash, Read, Write, Edit, WebSearch, WebFetch",
        "",
        "Claude can execute shell commands, read/write/edit files, and access the web.",
        "Only use in trusted environments.",
        "",
    ]
    if created:
        warning_lines.append("Created .claude/settings.local.json with default permissions.")
    warning_lines.append("Consider removing .claude/settings.local.json when done.")
    display_status("\n".join(warning_lines), kind="warning")

    cell_watcher = CellWatcher(ipython)
    magics = ClaudeCodeMagics(ipython, cell_watcher)
    ipython.register_magics(magics)
    ipython.events.register("pre_run_cell", cell_watcher.pre_run_cell)
    ipython.events.register("post_run_cell", cell_watcher.post_run_cell)

    display_status(HELP_TEXT, kind="info")
