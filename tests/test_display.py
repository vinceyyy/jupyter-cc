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
    display._items.append(("text", "second"))
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


# ------------------------------------------------------------------
# Ordered rendering tests
# ------------------------------------------------------------------


def test_order_text_then_tool() -> None:
    """Text added before tool call appears first in rendered HTML."""
    display = StreamingDisplay(jupyter=True)
    display.add_text("First message")
    display.add_tool_call("Read", {"file_path": "/test"}, "t1")
    html = display._render_jupyter_html()
    text_pos = html.index("First message")
    tool_pos = html.index("Read(/test)")
    assert text_pos < tool_pos


def test_order_tool_then_text() -> None:
    """Tool call added before text appears first in rendered HTML."""
    display = StreamingDisplay(jupyter=True)
    display.add_tool_call("Bash", {"command": "echo hi"}, "t1")
    display.add_text("After the tool")
    html = display._render_jupyter_html()
    tool_pos = html.index("Bash")
    text_pos = html.index("After the tool")
    assert tool_pos < text_pos


def test_order_interleaved() -> None:
    """Multiple interleaved items render in arrival order."""
    display = StreamingDisplay(jupyter=True)
    display.add_text("text-A")
    display.add_tool_call("Read", {"file_path": "/f1"}, "t1")
    display.add_text("text-B")
    display.add_tool_call("Bash", {"command": "ls"}, "t2")
    html = display._render_jupyter_html()
    positions = [
        html.index("text-A"),
        html.index("Read(/f1)"),
        html.index("text-B"),
        html.index("Bash"),
    ]
    assert positions == sorted(positions)


# ------------------------------------------------------------------
# Thinking block tests
# ------------------------------------------------------------------


def test_thinking_block_rendering() -> None:
    """Thinking blocks render with jcc-thinking class."""
    display = StreamingDisplay(jupyter=True)
    display.add_thinking("Let me analyze this...")
    html = display._render_jupyter_html()
    assert "jcc-thinking" in html
    assert "Let me analyze this..." in html


def test_thinking_block_html_escaped() -> None:
    """Thinking blocks HTML-escape their content."""
    display = StreamingDisplay(jupyter=True)
    display.add_thinking("test <script>alert('xss')</script>")
    html = display._render_jupyter_html()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


# ------------------------------------------------------------------
# Tool completion tests
# ------------------------------------------------------------------


def test_complete_tool_call_via_items() -> None:
    """complete_tool_call finds the entry in _items by tool_id."""
    display = StreamingDisplay(jupyter=True)
    display.add_tool_call("Read", {"file_path": "/a"}, "t1")
    display.add_tool_call("Bash", {"command": "ls"}, "t2")
    display.complete_tool_call("t1")

    html = display._render_jupyter_html()
    # t1 completed: shows checkmark
    assert "\u2713 Read(/a)" in html
    # t2 still active: shows "Tool:" prefix
    assert "Tool: Bash" in html


def test_tool_prefix_not_hourglass() -> None:
    """Active tool calls show 'Tool:' prefix, not hourglass emoji."""
    display = StreamingDisplay(jupyter=True)
    display.add_tool_call("Read", {"file_path": "/test"}, "t1")
    html = display._render_jupyter_html()
    assert "Tool:" in html
    assert "\u23f3" not in html  # No hourglass


# ------------------------------------------------------------------
# nl2br / line break tests
# ------------------------------------------------------------------


def test_nl2br_preserves_single_newlines() -> None:
    """Single newlines within text are preserved as <br> in HTML."""
    display = StreamingDisplay(jupyter=True)
    display.add_text("line one\nline two\nline three")
    html = display._render_jupyter_html()
    assert "<br" in html


# ------------------------------------------------------------------
# Scrollable container tests
# ------------------------------------------------------------------


def test_scrollable_container_css() -> None:
    """The CSS includes max-height and overflow-y for scrollable output."""
    display = StreamingDisplay(jupyter=True)
    css = display._render_css()
    assert "max-height" in css
    assert "overflow-y" in css


# ------------------------------------------------------------------
# Result metadata / footer tests
# ------------------------------------------------------------------


def test_set_result_renders_footer() -> None:
    """set_result stores metadata that renders in a footer."""
    display = StreamingDisplay(jupyter=True)
    display.add_text("done")
    display.set_result(
        duration_ms=2500,
        total_cost_usd=0.0045,
        usage={"input_tokens": 200, "output_tokens": 100},
        num_turns=3,
    )
    html = display._render_jupyter_html()
    assert "jcc-footer" in html
    assert "2.5s" in html
    assert "$0.0045" in html
    assert "300 tokens" in html
    assert "3 turns" in html


def test_set_result_partial_metadata() -> None:
    """Footer renders gracefully with partial metadata."""
    display = StreamingDisplay(jupyter=True)
    display.set_result(duration_ms=1000)
    html = display._render_jupyter_html()
    assert "jcc-footer" in html
    assert "1.0s" in html
    # No cost or tokens
    assert "$" not in html
    assert "tokens" not in html


def test_set_result_single_turn() -> None:
    """Single turn shows '1 turn' (not '1 turns')."""
    display = StreamingDisplay(jupyter=True)
    display.set_result(num_turns=1)
    html = display._render_jupyter_html()
    assert "1 turn" in html
    assert "1 turns" not in html


# ------------------------------------------------------------------
# Fallback mode tests
# ------------------------------------------------------------------


def test_fallback_thinking_block(capsys: object) -> None:
    """Fallback mode prints thinking blocks with [thinking] prefix."""
    import io
    import sys

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured

    display = StreamingDisplay(jupyter=False)
    display._fallback = True
    display.add_thinking("I need to check the file structure")

    sys.stdout = old_stdout
    output = captured.getvalue()
    assert "[thinking]" in output
    assert "I need to check" in output


def test_fallback_tool_prefix(capsys: object) -> None:
    """Fallback mode prints 'Tool:' for active tool calls."""
    import io
    import sys

    captured = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = captured

    display = StreamingDisplay(jupyter=False)
    display._fallback = True
    display.add_tool_call("Read", {"file_path": "/test"}, "t1")

    sys.stdout = old_stdout
    output = captured.getvalue()
    assert "Tool:" in output
    assert "Read" in output
