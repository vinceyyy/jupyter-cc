"""
Configuration management for jupyter_cc.
Handles all configuration options and command-line argument processing.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .constants import (
    CLEANUP_PROMPTS_TEXT,
    HELP_TEXT,
    QUEUED_EXECUTION_TEXT,
)
from .display import display_status

if TYPE_CHECKING:
    from .watcher import CellWatcher


class ConfigManager:
    """Manages configuration state for Claude Code magic."""

    def __init__(self) -> None:
        """Initialize configuration with defaults."""
        # Cleanup settings
        self.should_cleanup_prompts = False
        # Per-query flag: replace current cell instead of inserting below.
        # Set by %cc_cur and reset after each query completes.
        self.replace_current_cell = False

        # Conversation settings
        self.is_new_conversation: bool = True
        self.is_current_execution_verbose: bool = False

        # Cell limits
        # 3 cells per turn allows multi-step exploration without forcing everything
        # into a single cell. With proper prompting and enforcement, this should be
        # reasonable while still encouraging focused, incremental work.
        self.max_cells = 3
        # Track create_python_cell calls in current conversation
        self.create_python_cell_count = 0

        # Model selection
        self.model = "sonnet"

        # Import tracking
        self.imported_files: list[str] = []

        # Directory permissions
        self.added_directories: list[str] = []

        # MCP configuration
        self.mcp_config_file: str | None = None

        # Cell loading settings
        self.cells_to_load: int = -1  # Default to -1 (load all) for initial %cc
        self.cells_to_load_user_set: bool = False  # Track if explicitly set by user

    @property
    def should_replace_cell(self) -> bool:
        """Whether the current cell should be replaced (cleanup mode or cc_cur)."""
        return self.should_cleanup_prompts or self.replace_current_cell

    def reset_for_new_conversation(self) -> None:
        """Reset settings for a new conversation."""
        self.is_new_conversation = True
        # Reset create_python_cell counter for new conversation
        self.create_python_cell_count = 0
        # For cc_new, default to not loading previous cells (0)
        # But only if user hasn't explicitly set a value
        if not self.cells_to_load_user_set:
            self.cells_to_load = 0

    def handle_cc_options(self, args: Any, cell_watcher: "CellWatcher") -> bool:
        """
        Handle all command-line options for the cc magic command.

        Args:
            args: Parsed arguments from argparse
            cell_watcher: Cell watcher instance for execution detection

        Returns:
            True if any option was handled (meaning the command should return early)
        """
        if args.help:
            display_status(HELP_TEXT, kind="info")
            return True

        if args.clean is not None:
            self.should_cleanup_prompts = args.clean
            maybe_not = "" if self.should_cleanup_prompts else "not "
            display_status(CLEANUP_PROMPTS_TEXT.format(maybe_not=maybe_not), kind="info")
            return True

        # Settings take effect on the next query. If a conversation is already in progress,
        # the user needs to start fresh with %cc_new for the setting to apply.
        pickup_message = (
            "Will apply to the next query."
            if self.is_new_conversation
            else "Use %cc_new to start a new conversation with this setting."
        )

        if args.max_cells is not None:
            old_max_cells = self.max_cells
            self.max_cells = args.max_cells
            display_status(f"📝 Set max_cells from {old_max_cells} to {self.max_cells}. {pickup_message}", kind="info")
            return True

        if args.import_file is not None:
            file_path = Path(args.import_file).expanduser().resolve()

            try:
                with file_path.open() as f:
                    # Try to read first few bytes to check if it's text
                    f.read(1024)

                file_str = str(file_path)
                if file_str not in self.imported_files:
                    self.imported_files.append(file_str)
                    display_status(f"✅ Added {file_path.name} to import list. {pickup_message}", kind="success")
                else:
                    display_status(f"ℹ️ {file_path} is already in the import list.", kind="info")
            except Exception:
                display_status(
                    f"❌ Import failed: {file_path.name} does not exist or is not a plaintext file.",
                    kind="error",
                )
            return True

        if args.add_dir is not None:
            dir_path = Path(args.add_dir).expanduser().resolve()

            if not dir_path.exists():
                display_status(f"❌ Directory not found: {dir_path}", kind="error")
                return True

            if not dir_path.is_dir():
                display_status(f"❌ Path is not a directory: {dir_path}", kind="error")
                return True

            # Add to added directories list if not already there
            dir_str = str(dir_path)
            if dir_str not in self.added_directories:
                self.added_directories.append(dir_str)
                display_status(f"✅ Added {dir_path} to accessible directories. {pickup_message}", kind="success")
            else:
                display_status(f"ℹ️ {dir_path} is already in the accessible directories list.", kind="info")
            return True

        if args.mcp_config is not None:
            config_path = Path(args.mcp_config).expanduser().resolve()

            self.mcp_config_file = str(config_path)
            display_status(f"✅ Set MCP config file to {config_path}. {pickup_message}", kind="success")
            return True

        if args.model is not None:
            self.model = args.model
            display_status(f"✅ Set model to {self.model}. {pickup_message}", kind="success")
            return True

        if args.cells_to_load is not None:
            if args.cells_to_load < -1:
                display_status("❌ Number of cells must be -1 (all), 0 (none), or positive", kind="error")
                return True
            self.cells_to_load = args.cells_to_load
            self.cells_to_load_user_set = True  # Mark as explicitly set by user
            if args.cells_to_load == 0:
                display_status("✅ Disabled loading recent cells when starting new conversations", kind="success")
            elif args.cells_to_load == -1:
                display_status("✅ Will load all available cells when starting new conversations", kind="success")
            else:
                display_status(
                    f"✅ Will load up to {args.cells_to_load} recent cell(s) when starting new conversations",
                    kind="success",
                )
            return True

        # Handle queued execution check
        if cell_watcher.was_execution_probably_queued() and not args.allow_run_all:
            display_status(QUEUED_EXECUTION_TEXT, kind="warning")
            return True

        # No options were handled
        return False

    def get_mcp_servers(self) -> dict[str, Any]:
        """Get the MCP servers configuration from the mcp_config_file if set."""
        mcp_servers: dict[str, Any] = {}

        # If we have an MCP config file, load servers from it
        if self.mcp_config_file:
            try:
                with Path(self.mcp_config_file).open() as f:
                    config_data = json.load(f)
                    if "mcpServers" in config_data and isinstance(config_data["mcpServers"], dict):
                        mcp_servers.update(config_data["mcpServers"])
            except json.JSONDecodeError as e:
                display_status(f"⚠️ Error parsing MCP config file {self.mcp_config_file}: {e}", kind="warning")
            except Exception as e:
                display_status(f"⚠️ Error loading MCP config file {self.mcp_config_file}: {e}", kind="warning")

        return mcp_servers
