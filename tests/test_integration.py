"""Unit tests for jupyter_cc_magic.integration."""

from __future__ import annotations

from unittest.mock import MagicMock

from jupyter_cc_magic.integration import create_approval_cell, process_cell_queue


def _make_parent(mock_shell: MagicMock) -> MagicMock:
    """Create a mock parent (ClaudeCodeMagics) with the given shell."""
    parent = MagicMock()
    parent.shell = mock_shell
    return parent


def test_create_approval_cell_with_description(mock_shell: MagicMock) -> None:
    """Creates cell with '# [CC] description' marker."""
    parent = _make_parent(mock_shell)

    create_approval_cell(
        parent,
        code="print('hi')",
        request_id="req-1",
        should_cleanup_prompts=False,
        tool_use_id="tool-1",
        description="Print a greeting",
    )

    cell_queue = mock_shell.user_ns["_claude_cell_queue"]
    assert len(cell_queue) == 1
    assert cell_queue[0]["code"].startswith("# [CC] Print a greeting")
    assert "print('hi')" in cell_queue[0]["code"]


def test_create_approval_cell_without_description(mock_shell: MagicMock) -> None:
    """Creates cell with just '# [CC]' marker when no description."""
    parent = _make_parent(mock_shell)

    create_approval_cell(
        parent,
        code="x = 1",
        request_id="req-2",
        should_cleanup_prompts=False,
        tool_use_id="tool-2",
        description="",
    )

    cell_queue = mock_shell.user_ns["_claude_cell_queue"]
    assert len(cell_queue) == 1
    assert cell_queue[0]["code"].startswith("# [CC]\n")
    assert "x = 1" in cell_queue[0]["code"]


def test_cell_queue_initialized(mock_shell: MagicMock) -> None:
    """Cell queue list created in user_ns."""
    parent = _make_parent(mock_shell)
    assert "_claude_cell_queue" not in mock_shell.user_ns

    create_approval_cell(
        parent,
        code="pass",
        request_id="req-3",
        should_cleanup_prompts=False,
    )

    assert "_claude_cell_queue" in mock_shell.user_ns
    assert isinstance(mock_shell.user_ns["_claude_cell_queue"], list)


def test_process_cell_queue_sets_next_input(mock_shell: MagicMock) -> None:
    """Next unexecuted cell is set as next input via set_next_input."""
    parent = _make_parent(mock_shell)

    # Set up a queue with one unexecuted cell
    mock_shell.user_ns["_claude_cell_queue"] = [
        {
            "code": "# [CC] step 1\nprint('step 1')",
            "original_code": "print('step 1')",
            "tool_use_id": "tool-1",
            "request_id": "req-1",
            "marker_id": "tool-1",
            "marker": "# [CC] step 1",
            "executed": False,
        },
    ]

    process_cell_queue(parent)

    mock_shell.set_next_input.assert_called_once_with("# [CC] step 1\nprint('step 1')")
