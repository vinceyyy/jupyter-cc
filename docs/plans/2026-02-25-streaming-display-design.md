# Streaming Display Design

## Problem

1. **No streaming**: Jupyter users see only a CSS spinner while Claude runs. All content renders at once when done.
1. **Ugly final output**: Rich's terminal-to-HTML export produces inverted colors and box-drawing that clashes with Jupyter themes.

## Decision Summary

| Decision           | Choice                                                                  |
| ------------------ | ----------------------------------------------------------------------- |
| UX model           | Status panel (model + tool calls) with streaming markdown below         |
| Tool call display  | Inline one-liners: `> ✓ Read(src/main.py)` with spinner then checkmark  |
| Render technology  | `ipywidgets.HTML` widget, updated on each state change                  |
| Final state        | Freeze widget in place (no swap, no duplication)                        |
| Rich dependency    | **Remove entirely** — native HTML for Jupyter, plain print for terminal |
| Markdown rendering | Add `markdown` package for full markdown-to-HTML conversion             |

## Architecture

```
┌─ _execute_prompt() (main thread) ──────────────────────┐
│  display = StreamingDisplay()                           │
│  display.start()  → creates ipywidgets.HTML widget      │
│                                                          │
│  ┌─ background thread (anyio.run) ───────────────────┐  │
│  │  async for message in client.receive_response():  │  │
│  │    display.add_text(block.text)                   │  │
│  │      → _refresh() → widget.value = new HTML       │  │
│  │    display.add_tool_call(name, input, id)         │  │
│  │      → _refresh() → widget.value = new HTML       │  │
│  └───────────────────────────────────────────────────┘  │
│                                                          │
│  display.stop()  → one final refresh, freeze widget     │
└──────────────────────────────────────────────────────────┘
```

### Thread safety

`ipywidgets.HTML.value` assignment from background threads is safe — traitlets posts
comm messages through the kernel's IOPub channel. Rapid updates are throttled to
10/sec max via `time.monotonic()` check in `_refresh()`.

### Throttled refresh

```python
_MIN_REFRESH_INTERVAL = 0.1  # seconds

def _refresh(self) -> None:
    now = time.monotonic()
    if now - self._last_refresh < _MIN_REFRESH_INTERVAL:
        self._dirty = True
        return
    self._last_refresh = now
    self._dirty = False
    if self._jupyter and self._widget is not None:
        self._widget.value = self._render_jupyter_html()
```

`stop()` always performs a final refresh regardless of throttle.

## HTML Rendering

### Output structure

```html
<div class="jcc-output">
  <div class="jcc-header">claude-sonnet-4-20250514</div>
  <div class="jcc-tools">
    <div class="jcc-tool done">✓ Read(src/main.py)</div>
    <div class="jcc-tool active">⠋ Grep(pattern: "TODO")</div>
  </div>
  <div class="jcc-content">
    <!-- markdown converted to HTML -->
    <p>Looking at your code, I found 3 TODOs...</p>
  </div>
  <div class="jcc-error">Connection lost</div>
</div>
```

### Styling

Uses JupyterLab CSS variables with sensible fallbacks:

```css
.jcc-output {
  font-family: var(--jp-code-font-family, monospace);
  font-size: var(--jp-code-font-size, 13px);
  color: var(--jp-ui-font-color1, #333);
  padding: 8px 0;
}
.jcc-header {
  color: var(--jp-ui-font-color2, #888);
  font-size: 0.85em;
  margin-bottom: 4px;
}
.jcc-tool {
  color: var(--jp-ui-font-color2, #666);
  font-size: 0.9em;
  padding: 1px 0;
}
.jcc-tool.done { opacity: 0.7; }
.jcc-content p { margin: 0.5em 0; }
.jcc-error { color: var(--jp-error-color1, #d32f2f); }
```

Works in JupyterLab, Notebook 7+, VS Code Jupyter, and Colab (via fallback values).

### Markdown conversion

```python
import markdown

def _md_to_html(text: str) -> str:
    return markdown.markdown(text, extensions=["fenced_code", "tables", "codehilite"])
```

## Client Streaming Fix

The interrupt-enabled path in `client.py` currently collects all messages, then
processes them after the stream ends. This must change to process-as-you-go:

**Before** (broken for streaming):

```python
messages_to_process = []
async for message in client.receive_response():
    messages_to_process.append(message)  # collect
for message in messages_to_process:      # then process
    display.add_text(...)
```

**After** (stream as they arrive):

```python
async for message in client.receive_response():
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                display.add_text(block.text)    # immediate
            elif isinstance(block, ToolUseBlock):
                display.add_tool_call(...)      # immediate
    # also append to messages list for return value
    messages_to_process.append(message)
```

Interrupt handling unchanged — parallel task checks `_interrupt_requested` and
cancels the scope.

## Files Changed

| File             | Change                                                                                                                                                      |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `display.py`     | Replace Rich rendering with native HTML. `_refresh()` updates widget. Add `_render_jupyter_html()`, markdown conversion, throttle. Remove all Rich imports. |
| `client.py`      | Move display calls inside message stream loop. Remove Rich import.                                                                                          |
| `pyproject.toml` | Remove `rich>=13.0.0`, add `markdown>=3.0`                                                                                                                  |

### Unchanged

- `magics.py` — already creates display in main thread, passes to background
- `integration.py` — cell creation logic unrelated
- Public API of `StreamingDisplay` — same methods, different internal rendering
