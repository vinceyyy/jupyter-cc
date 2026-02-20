"""
jupyter_cc — Jupyter magic for Claude Code.

Provides %cc magic commands for agentic Claude Code integration in notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path

from .constants import HELP_TEXT
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
        print(f"Created {settings_file.relative_to(cwd)}")
        return True
    return False


def load_ipython_extension(ipython: object) -> None:
    """Load the jupyter_cc extension."""
    from IPython.core.interactiveshell import InteractiveShell

    if not isinstance(ipython, InteractiveShell):
        return

    created = _ensure_claude_settings()

    # Security warning
    print("")
    print("\033[1;31m" + "=" * 80 + "\033[0m")
    print("\033[1;31mWARNING: Claude has permissions for Bash, Read, Write, Edit, WebSearch, WebFetch\033[0m")
    print("")
    print("  Claude can execute shell commands, read/write/edit files, and access the web.")
    print("  Only use in trusted environments.")
    print("")
    if created:
        print("  Created .claude/settings.local.json with default permissions.")
    print("  Consider removing .claude/settings.local.json when done.")
    print("\033[1;31m" + "=" * 80 + "\033[0m")
    print("")

    cell_watcher = CellWatcher(ipython)
    magics = ClaudeCodeMagics(ipython, cell_watcher)
    ipython.register_magics(magics)
    ipython.events.register("pre_run_cell", cell_watcher.pre_run_cell)
    ipython.events.register("post_run_cell", cell_watcher.post_run_cell)

    print(HELP_TEXT)
