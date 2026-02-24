"""Unit tests for jupyter_cc.display."""

from __future__ import annotations

from jupyter_cc.constants import EXECUTE_PYTHON_TOOL_NAME
from jupyter_cc.display import StreamingDisplay, format_tool_call


def test_format_tool_call_read() -> None:
    """Read tool shows file path."""
    result = format_tool_call("Read", {"file_path": "/home/user/data.csv"})
    assert "Read" in result
    assert "/home/user/data.csv" in result


def test_format_tool_call_bash() -> None:
    """Bash tool shows command."""
    result = format_tool_call("Bash", {"command": "ls -la"})
    assert "Bash" in result
    assert "ls -la" in result


def test_format_tool_call_grep() -> None:
    """Grep shows pattern and path."""
    result = format_tool_call("GrepToolv2", {"pattern": "TODO", "path": "/src"})
    assert "Search" in result
    assert "TODO" in result
    assert "/src" in result


def test_format_tool_call_create_cell() -> None:
    """CreateNotebookCell shows description."""
    result = format_tool_call(
        EXECUTE_PYTHON_TOOL_NAME,
        {"code": "print('hello')", "description": "Print greeting"},
    )
    assert "CreateNotebookCell" in result
    assert "Print greeting" in result


def test_format_tool_call_create_cell_no_description() -> None:
    """Shows just the display name without description."""
    result = format_tool_call(EXECUTE_PYTHON_TOOL_NAME, {"code": "x = 1"})
    assert result == "CreateNotebookCell"


def test_format_tool_call_unknown() -> None:
    """Unknown tool shows just the name."""
    result = format_tool_call("SomeNewTool", {"foo": "bar"})
    assert result == "SomeNewTool"


def test_streaming_display_lifecycle() -> None:
    """start/stop without errors using fallback mode."""
    display = StreamingDisplay()
    # Force fallback mode by starting without Rich Live
    display._fallback = True
    display.start()
    display.set_model("sonnet")
    display.add_text("Hello world")
    display.add_tool_call("Read", {"file_path": "/home/user/test.py"}, "tool-1")
    display.complete_tool_call("tool-1")
    display.set_session_id("session-abc")
    display.stop()
    # No exception means success
