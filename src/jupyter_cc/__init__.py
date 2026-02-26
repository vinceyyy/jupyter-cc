"""
jupyter_cc — Jupyter magic for Claude Code.

Provides %cc magic commands for agentic Claude Code integration in notebooks.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .capture import ImageCollector
from .constants import WELCOME_TEXT
from .display import display_status
from .magics import ClaudeCodeMagics
from .watcher import CellWatcher

__version__ = "0.3.0"

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

    # Check that Claude Code CLI is on PATH (doesn't verify it's functional)
    if not shutil.which("claude"):
        display_status(
            "ERROR: Claude Code CLI not found\n"
            "\n"
            "Install it: https://docs.anthropic.com/en/docs/claude-code\n"
            "Then reload: %load_ext jupyter_cc",
            kind="error",
        )
        return

    _ensure_claude_settings()

    # Short permissions warning
    display_status(
        "Claude has Bash, Read, Write, Edit, and web access. Use in trusted environments only.\n"
        "Permissions: .claude/settings.local.json",
        kind="warning",
    )

    cell_watcher = CellWatcher(ipython)
    image_collector = ImageCollector(ipython)
    image_collector.install()
    magics = ClaudeCodeMagics(ipython, cell_watcher, image_collector)
    ipython.register_magics(magics)
    ipython.events.register("pre_run_cell", cell_watcher.pre_run_cell)
    ipython.events.register("post_run_cell", cell_watcher.post_run_cell)

    display_status(WELCOME_TEXT, kind="info")
