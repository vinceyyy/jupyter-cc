"""Unit tests for jupyter_cc.display."""

from __future__ import annotations

from jupyter_cc.constants import EXECUTE_PYTHON_TOOL_NAME
from jupyter_cc.display import StreamingDisplay, format_tool_call


def _make_fake_widget() -> object:
    """Create a minimal stand-in for ipywidgets.HTML (no real kernel needed)."""
    return type("FakeWidget", (), {"value": "", "layout": type("L", (), {"display": ""})()})()


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
    # Force fallback mode (no ipywidgets)
    display._fallback = True
    display.start()
    display.set_model("sonnet")
    display.add_text("Hello world")
    display.add_tool_call("Read", {"file_path": "/home/user/test.py"}, "tool-1")
    display.complete_tool_call("tool-1")
    display.set_session_id("session-abc")
    display.stop()
    # No exception means success


# ------------------------------------------------------------------
# HTML renderer tests (Jupyter mode)
# ------------------------------------------------------------------


def test_render_jupyter_html_empty() -> None:
    """Empty state renders a waiting message."""
    display = StreamingDisplay(jupyter=True)
    html = display._render_jupyter_html()
    assert "jcc-output" in html
    assert "Thinking" in html


def test_render_jupyter_html_with_model() -> None:
    """Model name appears in header."""
    display = StreamingDisplay(jupyter=True)
    display.set_model("claude-sonnet-4-20250514")
    html = display._render_jupyter_html()
    assert "claude-sonnet-4-20250514" in html
    assert "jcc-header" in html


def test_render_jupyter_html_with_text() -> None:
    """Text blocks are rendered as markdown HTML."""
    display = StreamingDisplay(jupyter=True)
    display.add_text("Hello **world**")
    html = display._render_jupyter_html()
    assert "<strong>world</strong>" in html
    assert "jcc-content" in html


def test_render_jupyter_html_with_tool_calls() -> None:
    """Tool calls show with appropriate CSS classes."""
    display = StreamingDisplay(jupyter=True)
    display.add_tool_call("Read", {"file_path": "/home/user/test.py"}, "t1")
    html = display._render_jupyter_html()
    assert "jcc-tool" in html
    assert "Read" in html
    assert "/home/user/test.py" in html


def test_render_jupyter_html_completed_tool() -> None:
    """Completed tool calls show checkmark."""
    display = StreamingDisplay(jupyter=True)
    display.add_tool_call("Read", {"file_path": "/home/user/test.py"}, "t1")
    display.complete_tool_call("t1")
    html = display._render_jupyter_html()
    assert "\u2713" in html


def test_render_jupyter_html_error() -> None:
    """Errors render with error styling."""
    display = StreamingDisplay(jupyter=True)
    display.show_error("Connection lost")
    html = display._render_jupyter_html()
    assert "jcc-error" in html
    assert "Connection lost" in html


def test_render_jupyter_html_interrupt() -> None:
    """Interrupt notice is shown."""
    display = StreamingDisplay(jupyter=True)
    display.show_interrupt()
    html = display._render_jupyter_html()
    assert "interrupted" in html.lower()


def test_throttled_refresh_skips_rapid_updates() -> None:
    """Rapid calls to _refresh are throttled."""
    import time

    display = StreamingDisplay(jupyter=True)
    display._widget = _make_fake_widget()
    display._last_refresh = 0.0

    display.add_text("first")
    first_html = display._widget.value
    assert first_html != ""

    display._last_refresh = time.monotonic()
    display._text_blocks.append("second")
    display._refresh()
    assert display._widget.value == first_html
    assert display._dirty is True


def test_streaming_display_receives_updates_during_collection() -> None:
    """Verify display methods are called, simulating the inline processing flow."""
    display = StreamingDisplay(jupyter=True)
    display._widget = _make_fake_widget()
    display._last_refresh = 0.0

    # Simulate what the fixed client.py does: call display inline
    display.set_model("claude-sonnet-4-20250514")
    display.add_text("Hello from stream")
    display.add_tool_call("Bash", {"command": "echo hi"}, "tool-1")

    html = display._render_jupyter_html()
    assert "claude-sonnet-4-20250514" in html
    assert "Hello from stream" in html
    assert "Bash" in html
