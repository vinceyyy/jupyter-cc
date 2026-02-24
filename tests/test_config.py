"""Unit tests for jupyter_cc.config.ConfigManager."""

from __future__ import annotations

from argparse import Namespace

from jupyter_cc.config import ConfigManager


def test_default_values() -> None:
    """ConfigManager has sensible defaults: max_cells=3, model='sonnet', etc."""
    config = ConfigManager()
    assert config.max_cells == 3
    assert config.model == "sonnet"
    assert config.is_new_conversation is True
    assert config.imported_files == []
    assert config.added_directories == []
    assert config.should_cleanup_prompts is False
    assert config.mcp_config_file is None
    assert config.cells_to_load == -1


def test_update_from_args() -> None:
    """Argparse namespace values update config fields via handle_cc_options."""
    config = ConfigManager()

    # Simulate argparse output for --max-cells 5
    args = Namespace(
        help=False,
        clean=None,
        max_cells=5,
        import_file=None,
        add_dir=None,
        mcp_config=None,
        model=None,
        cells_to_load=None,
        allow_run_all=False,
    )

    from unittest.mock import MagicMock

    mock_watcher = MagicMock()
    mock_watcher.was_execution_probably_queued.return_value = False

    handled = config.handle_cc_options(args, mock_watcher)
    assert handled is True
    assert config.max_cells == 5
