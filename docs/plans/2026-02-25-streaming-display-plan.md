# Streaming Display Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the batch-render-at-end display with real-time streaming HTML that updates as Claude works, using Jupyter-native styling.

**Architecture:** Rewrite `StreamingDisplay` to render HTML directly (no Rich). `_refresh()` throttle-updates an `ipywidgets.HTML` widget. Fix `client.py` to process messages as they stream instead of collecting first. Remove `rich` dependency, add `markdown`.

**Tech Stack:** ipywidgets (HTML widget), markdown (Python package), CSS variables (JupyterLab theming)

______________________________________________________________________

### Task 1: Update dependencies

**Files:**

- Modify: `pyproject.toml:8-14`

**Step 1: Swap rich for markdown**

In `pyproject.toml`, replace `"rich>=13.0.0"` with `"markdown>=3.7"` in the `dependencies` list.

**Step 2: Sync dependencies**

Run: `uv sync`
Expected: Clean install, no errors.

**Step 3: Verify markdown is importable**

Run: `uv run python -c "import markdown; print(markdown.markdown('**bold**'))"`
Expected: `<p><strong>bold</strong></p>`

**Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: replace rich with markdown for Jupyter-native HTML rendering"
```

______________________________________________________________________

### Task 2: Rewrite display.py — HTML renderer and throttled refresh

This is the core change. Replace Rich rendering with native HTML generation.

**Files:**

- Rewrite: `src/jupyter_cc/display.py`

**Step 1: Write tests for the new HTML renderer**

Add to `tests/test_display.py`:

```python
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
    display.add_tool_call("Read", {"file_path": "/tmp/test.py"}, "t1")
    html = display._render_jupyter_html()
    assert "jcc-tool" in html
    assert "Read" in html
    assert "/tmp/test.py" in html


def test_render_jupyter_html_completed_tool() -> None:
    """Completed tool calls show checkmark."""
    display = StreamingDisplay(jupyter=True)
    display.add_tool_call("Read", {"file_path": "/tmp/test.py"}, "t1")
    display.complete_tool_call("t1")
    html = display._render_jupyter_html()
    assert "✓" in html


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
    assert "interrupted" in html.lower() or "Interrupted" in html


def test_throttled_refresh_skips_rapid_updates() -> None:
    """Rapid calls to _refresh are throttled."""
    display = StreamingDisplay(jupyter=True)
    display._widget = type("FakeWidget", (), {"value": "", "layout": type("L", (), {"display": ""})()})()
    display._last_refresh = 0.0  # ensure first refresh happens

    # First refresh should go through
    display.add_text("first")
    first_html = display._widget.value
    assert first_html != ""

    # Immediate second refresh should be skipped (throttled)
    import time
    display._last_refresh = time.monotonic()  # pretend we just refreshed
    display._text_blocks.append("second")
    display._refresh()
    # Widget value should NOT have changed (still first render)
    assert display._widget.value == first_html
    assert display._dirty is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_display.py -v`
Expected: New tests FAIL (methods don't exist yet or return wrong HTML).

**Step 3: Rewrite display.py**

Replace the entire file. Key changes:

- Remove all `from rich.*` imports
- Add `import html as html_module`, `import time`, `import markdown`
- `_render_jupyter_html()`: build HTML string from state using CSS classes and `--jp-*` variables
- `_render_css()`: return the `<style>` block (called once, cached)
- `_refresh()`: throttled update of `widget.value` in Jupyter mode
- `start()`: create `ipywidgets.HTML` widget (Jupyter) or just set `_fallback=True` (terminal)
- `stop()`: final refresh + set a `_stopped` flag (Jupyter), no-op for fallback
- Remove `_render()` (Rich renderable) and `_render_html_string()` (Rich-to-HTML export)
- Keep `format_tool_call()` and `_ToolCallEntry` unchanged
- Keep `_print_fallback_latest()` for terminal mode

The CSS should use `--jp-*` variables:

```css
.jcc-output { font-family: var(--jp-code-font-family, monospace); font-size: var(--jp-code-font-size, 13px); color: var(--jp-ui-font-color1, #333); padding: 8px 0; }
.jcc-header { color: var(--jp-ui-font-color2, #888); font-size: 0.85em; margin-bottom: 8px; }
.jcc-tool { color: var(--jp-ui-font-color2, #666); font-size: 0.9em; padding: 1px 0; font-family: var(--jp-code-font-family, monospace); }
.jcc-tool.done { opacity: 0.6; }
.jcc-tools { margin-bottom: 8px; }
.jcc-content { line-height: 1.5; }
.jcc-content p { margin: 0.4em 0; }
.jcc-content pre { background: var(--jp-layout-color2, #f5f5f5); padding: 8px 12px; border-radius: 4px; overflow-x: auto; }
.jcc-content code { font-family: var(--jp-code-font-family, monospace); font-size: 0.9em; }
.jcc-content p code { background: var(--jp-layout-color2, #f5f5f5); padding: 1px 4px; border-radius: 3px; }
.jcc-error { color: var(--jp-error-color1, #d32f2f); margin-top: 8px; }
.jcc-interrupt { color: var(--jp-warn-color1, #f57c00); margin-top: 8px; }
.jcc-waiting { color: var(--jp-ui-font-color3, #aaa); font-style: italic; }
```

The spinner in the header uses a CSS animation (already exists as `_CSS_SPINNER_HTML` — keep it but simplified).

**Step 4: Run tests**

Run: `uv run pytest tests/test_display.py -v`
Expected: ALL tests pass.

**Step 5: Lint and type check**

Run: `uv run ruff check src/jupyter_cc/display.py --fix && uv run ruff format src/jupyter_cc/display.py`
Run: `uv run pyright src/jupyter_cc/display.py`
Expected: No errors (warnings OK for pyright).

**Step 6: Commit**

```bash
git add src/jupyter_cc/display.py tests/test_display.py
git commit -m "feat: rewrite display with native HTML streaming, drop Rich"
```

______________________________________________________________________

### Task 3: Fix client.py — stream messages as they arrive

The interrupt-enabled path collects messages then processes. Change to process inline.

**Files:**

- Modify: `src/jupyter_cc/client.py:140-225`

**Step 1: Write a test for inline message processing**

Add to `tests/test_display.py` (or a new `tests/test_client.py` if preferred):

```python
def test_streaming_display_receives_updates_during_collection() -> None:
    """Verify display methods are called, simulating the inline processing flow."""
    display = StreamingDisplay(jupyter=True)
    display._widget = type("FakeWidget", (), {"value": "", "layout": type("L", (), {"display": ""})()})()
    display._last_refresh = 0.0

    # Simulate what the fixed client.py will do: call display inline
    display.set_model("claude-sonnet-4-20250514")
    display.add_text("Hello from stream")
    display.add_tool_call("Bash", {"command": "echo hi"}, "tool-1")

    html = display._render_jupyter_html()
    assert "claude-sonnet-4-20250514" in html
    assert "Hello from stream" in html
    assert "Bash" in html
```

**Step 2: Run test to verify it passes (tests display, not client)**

Run: `uv run pytest tests/test_display.py::test_streaming_display_receives_updates_during_collection -v`
Expected: PASS (this tests the display side; the client change is structural).

**Step 3: Refactor client.py — inline display processing**

In `query_sync()`, the interrupt-enabled path (the `if enable_interrupt:` block starting around line 141):

1. Keep the `messages_to_process` list and `collect_messages()` coroutine structure.
1. Inside `collect_messages()`, after `messages_to_process.append(message)`, add the display processing logic that currently lives in the "Process collected messages" section (lines 188-204).
1. Delete the duplicate "Process collected messages" loop that runs after the task group (lines 188-204).
1. The non-interrupt path (`else:` block, lines 206-226) already processes inline — no change needed there.

The key structural change in `collect_messages()`:

```python
async def collect_messages() -> None:
    nonlocal collection_error
    try:
        async for message in client.receive_response():
            if message is None:
                continue
            messages_to_process.append(message)
            # Process for display IMMEDIATELY
            if isinstance(message, AssistantMessage):
                if hasattr(message, "model") and not has_printed_model:
                    display.set_model(message.model)
                    has_printed_model = True
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        display.add_text(block.text)
                        assistant_messages.append(block.text)
                    elif isinstance(block, ToolUseBlock):
                        display.add_tool_call(block.name, block.input, block.id)
                        tool_calls.append(f"{block.name}: {block.input}")
            elif isinstance(message, ResultMessage):
                if message.session_id and message.session_id != self._session_id:
                    self._session_id = message.session_id
                    display.set_session_id(self._session_id)
                break
    except Exception as exc:
        collection_error = exc
    finally:
        collection_done.set()
```

Then delete the post-collection processing block (lines 188-204 currently starting with `# Process collected messages`).

Note: `has_printed_model` needs to be declared as `nonlocal` in the inner function since `collect_messages` now mutates it.

**Step 4: Remove the Rich import from client.py**

`client.py` only imports `StreamingDisplay` from `.display` — no direct Rich imports. Confirm with grep; nothing to change here.

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass.

**Step 6: Lint and type check**

Run: `uv run ruff check src/jupyter_cc/client.py --fix && uv run ruff format src/jupyter_cc/client.py`
Run: `uv run pyright src/jupyter_cc/client.py`

**Step 7: Commit**

```bash
git add src/jupyter_cc/client.py tests/test_display.py
git commit -m "feat: stream display updates inline during message collection"
```

______________________________________________________________________

### Task 4: Remove Rich from display.py terminal fallback references

After the rewrite in Task 2, there should be no Rich references left. This task is a verification pass.

**Files:**

- Verify: `src/jupyter_cc/display.py`

**Step 1: Verify no Rich imports remain**

Run: `uv run ruff check src/ --fix && uv run ruff format src/`
Run: `grep -r "from rich\|import rich" src/`
Expected: No output (no Rich imports anywhere in src/).

**Step 2: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass.

**Step 3: Verify pyright**

Run: `uv run pyright src/`
Expected: No errors (warnings OK).

**Step 4: Commit if any cleanup was needed**

```bash
git add -A
git commit -m "chore: verify Rich removal, clean up any residual references"
```

______________________________________________________________________

### Task 5: Manual integration test in Jupyter

This task verifies the feature end-to-end in a real Jupyter environment.

**Step 1: Install in development mode**

Run: `uv sync`

**Step 2: Launch Jupyter and test**

Run: `uv run jupyter notebook`

In a new notebook:

```python
%load_ext jupyter_cc
%cc what is 2 + 2?
```

**Verify:**

- [ ] Spinner shows while Claude connects
- [ ] Model name appears in header when response starts
- [ ] Tool calls appear as one-liners with activity indicator
- [ ] Text streams incrementally (not all at once at the end)
- [ ] Final state freezes in place (no duplication, no layout shift)
- [ ] Colors look natural in Jupyter's default light theme
- [ ] Markdown renders properly (bold, code blocks, lists)
- [ ] `--verbose` flag shows tool arguments
- [ ] Interrupt (Ctrl+C / kernel interrupt) shows interrupt notice

**Step 3: Test in dark theme**

Switch JupyterLab to dark theme and verify:

- [ ] CSS variables pick up dark theme colors
- [ ] Text is readable, no inverted highlights
- [ ] Code blocks have appropriate background

**Step 4: Commit any fixes discovered during testing**

```bash
git add -A
git commit -m "fix: adjustments from manual integration testing"
```
