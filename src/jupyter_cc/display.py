"""
Streaming display for Claude responses.

In Jupyter notebooks:
  - Shows a CSS spinner while loading
  - Live-updates HTML via IPython's display_id mechanism with throttled refresh
  - Renders markdown text, tool calls, errors via native HTML with theme variables
  - Theme-adaptive CSS: JupyterLab (--jp-*), VS Code (--vscode-*), and safe fallbacks

In terminals:
  - Falls back to plain print()
"""

import html as html_module
import logging
import threading
import time
from typing import Any

import markdown

from .constants import EXECUTE_PYTHON_TOOL_NAME

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Styled status messages (consistent HTML output in Jupyter)
# ---------------------------------------------------------------------------

# Accent colors for each message kind.
# These are hardcoded since display_status() uses inline styles (can't scope
# via body[data-vscode-theme-kind]). The colors work on both light/dark.
_STATUS_ACCENT = {
    "success": "#4caf50",
    "warning": "#f57c00",
    "error": "#d32f2f",
    "info": "#4a90d9",
}


def display_status(message: str, *, kind: str = "info") -> None:
    """Display a styled status message.

    In Jupyter notebooks, renders as an HTML div with a coloured left-border
    accent that matches the existing ``.jcc-*`` theme.  In terminals, falls
    back to a plain ``print()``.

    Args:
        message: The message text (may be multi-line).
        kind: One of ``"success"``, ``"warning"``, ``"error"``, ``"info"``.
    """
    try:
        from IPython import get_ipython  # type: ignore[attr-defined]
        from IPython.display import HTML, display

        ip = get_ipython()
        if ip is not None and hasattr(ip, "kernel"):
            accent = _STATUS_ACCENT.get(kind, _STATUS_ACCENT["info"])
            escaped = html_module.escape(message.strip())
            html_str = (
                f'<div style="border-left:3px solid {accent};'
                "padding:6px 12px;margin:4px 0;"
                "font-family:inherit;font-size:13px;color:inherit;"
                f'white-space:pre-wrap">{escaped}</div>'
            )
            display(HTML(html_str))
            return
    except ImportError:
        pass
    print(message, flush=True)


# Inline-styled spinner shown before the full CSS loads.
# Uses neutral values since inline styles can't scope via body[data-vscode-theme-kind].
_CSS_SPINNER_HTML = (
    '<div style="display:flex;align-items:center;gap:8px;padding:4px 0;'
    "font-family:inherit;font-size:13px;"
    'color:color-mix(in srgb, currentColor, transparent 35%)">'
    '<div style="width:14px;height:14px;'
    "border:2px solid rgba(128,128,128,0.2);"
    "border-top:2px solid #4a90d9;"
    'border-radius:50%;animation:jcc-spin .8s linear infinite"></div>'
    "<span>Running&hellip;</span></div>"
    "<style>@keyframes jcc-spin{0%{transform:rotate(0deg)}"
    "100%{transform:rotate(360deg)}}</style>"
)

# Minimum interval between widget refreshes (seconds)
_REFRESH_INTERVAL = 0.1


def format_tool_call(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Format tool calls to match Claude CLI style with meaningful details."""
    tool_display_names = {
        "LS": "List",
        "GrepToolv2": "Search",
        EXECUTE_PYTHON_TOOL_NAME: "CreateNotebookCell",
    }

    display_name = tool_display_names.get(tool_name, tool_name)

    match tool_name:
        case "Read":
            file_path = tool_input.get("file_path", "")
            parts = [f"{display_name}({file_path})"]
            if "offset" in tool_input:
                parts.append(f"offset: {tool_input['offset']}")
            if "limit" in tool_input:
                parts.append(f"limit: {tool_input['limit']}")
            return " ".join(parts)

        case "LS":
            path = tool_input.get("path", "")
            return f"{display_name}({path})"

        case "GrepToolv2":
            pattern = tool_input.get("pattern", "")
            parts = [f'{display_name}(pattern: "{pattern}"']
            path = tool_input.get("path")
            if path:
                parts.append(f'path: "{path}"')
            if "glob" in tool_input:
                parts.append(f'glob: "{tool_input["glob"]}"')
            if "type" in tool_input:
                parts.append(f'type: "{tool_input["type"]}"')
            if tool_input.get("output_mode") and tool_input["output_mode"] != "files_with_matches":
                parts.append(f'output_mode: "{tool_input["output_mode"]}"')
            if "head_limit" in tool_input:
                parts.append(f"head_limit: {tool_input['head_limit']}")
            return ", ".join(parts) + ")"

        case "Bash":
            command = tool_input.get("command", "")
            return f'{display_name}("{command}")'

        case "Write" | "Edit" | "MultiEdit":
            file_path = tool_input.get("file_path", "")
            return f"{display_name}({file_path})"

        case "Glob":
            pattern = tool_input.get("pattern", "")
            path = tool_input.get("path", "")
            if path:
                return f'{display_name}(pattern: "{pattern}", path: "{path}")'
            return f'{display_name}("{pattern}")'

        case "WebFetch":
            url = tool_input.get("url", "")
            return f'{display_name}("{url}")'

        case "WebSearch":
            query = tool_input.get("query", "")
            return f'{display_name}("{query}")'

        case "TodoWrite":
            todos = tool_input.get("todos", [])
            return f"{display_name}({len(todos)} items)"

        case _:
            if tool_name == EXECUTE_PYTHON_TOOL_NAME:
                description = tool_input.get("description", "")
                if description:
                    return f'{display_name}("{description}")'
            return display_name


class _ToolCallEntry:
    """Internal state for a single tool call being displayed."""

    def __init__(self, display_text: str, tool_id: str) -> None:
        self.display_text = display_text
        self.tool_id = tool_id
        self.completed = False


class StreamingDisplay:
    """
    Display for Claude responses.

    Jupyter mode:
      - CSS spinner initially, then live-updated HTML via display_id
      - Throttled refresh to avoid excessive DOM updates

    Fallback:
      - Plain print() for terminals

    Must be created and start()'d from the main IPython thread.
    State-mutating methods (add_text, add_tool_call, etc.) are safe from any thread.
    """

    def __init__(self, *, verbose: bool = False, jupyter: bool | None = None, replace_mode: bool = False) -> None:
        self._verbose = verbose
        self._replace_mode = replace_mode
        self._model: str | None = None
        # Single ordered list: ("text", str) | ("tool", _ToolCallEntry) | ("thinking", str)
        self._items: list[tuple[str, Any]] = []
        self._session_id: str | None = None
        self._error: str | None = None
        self._interrupted = False
        self._result_meta: dict[str, Any] | None = None
        self._cells_created: int = 0
        self._stopped = False
        # Auto-detect only works from the main IPython thread.
        if jupyter is not None:
            self._jupyter = jupyter
        else:
            from .integration import is_in_jupyter_notebook

            self._jupyter = is_in_jupyter_notebook()
        self._fallback = False

        # display_id for updatable HTML output (created in start())
        self._display_id: str | None = None

        # Throttling state (guarded by _refresh_lock for cross-thread safety)
        self._refresh_lock = threading.Lock()
        self._last_refresh = 0.0
        self._dirty = False
        self._pending_timer: threading.Timer | None = None

        # CSS cache
        self._css_cache: str | None = None

    def start(self) -> None:
        """Start the live display. Must be called from the main IPython thread."""
        if self._jupyter:
            try:
                import uuid

                from IPython.display import HTML, display

                self._display_id = f"jcc-{uuid.uuid4().hex[:8]}"
                display(HTML(_CSS_SPINNER_HTML), display_id=self._display_id)
            except Exception:
                logger.debug("display_id unavailable, falling back to print()", exc_info=True)
                self._jupyter = False
                self._fallback = True
                print(
                    "[jupyter-cc] Rich display unavailable — using plain text output.",
                    flush=True,
                )
            return

        # Terminal mode: plain print fallback
        self._fallback = True

    def stop(self) -> None:
        """Stop the live display, render final output (removes spinner)."""
        self._stopped = True
        if self._pending_timer is not None:
            self._pending_timer.cancel()
            self._pending_timer = None
        if self._jupyter and self._display_id is not None:
            self._refresh(force=True)
            return

        # Terminal fallback: no-op (all output already printed incrementally)

    def set_model(self, model: str) -> None:
        """Set the model name shown in the header."""
        self._model = model
        self._refresh()

    def add_text(self, text: str) -> None:
        """Append a text block to the display."""
        self._items.append(("text", text))
        self._refresh()

    def add_tool_call(self, tool_name: str, tool_input: dict[str, Any], tool_id: str) -> None:
        """Add an active tool call (shown with "Tool:" prefix)."""
        display_text = format_tool_call(tool_name, tool_input)
        entry = _ToolCallEntry(display_text, tool_id)
        self._items.append(("tool", entry))
        if tool_name == EXECUTE_PYTHON_TOOL_NAME:
            self._cells_created += 1
        if self._verbose:
            entry.display_text += f"\n  Arguments: {tool_input}"
        self._refresh()

    def complete_tool_call(self, tool_id: str) -> None:
        """Mark a tool call as completed (prefix -> checkmark)."""
        for kind, item in self._items:
            if kind == "tool" and item.tool_id == tool_id:
                item.completed = True
                break
        self._refresh()

    def add_thinking(self, text: str) -> None:
        """Append a thinking block to the display."""
        self._items.append(("thinking", text))
        self._refresh()

    def set_result(
        self,
        *,
        duration_ms: int = 0,
        total_cost_usd: float | None = None,
        usage: dict[str, Any] | None = None,
        num_turns: int = 0,
    ) -> None:
        """Store result metadata shown in the footer."""
        self._result_meta = {
            "duration_ms": duration_ms,
            "total_cost_usd": total_cost_usd,
            "usage": usage,
            "num_turns": num_turns,
        }
        self._refresh()

    def set_session_id(self, session_id: str) -> None:
        """Set the session ID shown in the footer."""
        self._session_id = session_id
        self._refresh()

    def show_error(self, error_text: str) -> None:
        """Display an error message."""
        self._error = error_text
        self._refresh()

    def show_interrupt(self) -> None:
        """Display an interrupt notice."""
        self._interrupted = True
        self._refresh()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _refresh(self, *, force: bool = False) -> None:
        """Push the latest render to the display.

        In Jupyter mode, updates the widget HTML with throttling.
        In fallback mode, prints incrementally.
        """
        if self._fallback:
            self._print_fallback_latest()
            return

        if not self._jupyter or self._display_id is None:
            return

        with self._refresh_lock:
            now = time.monotonic()
            if not force and (now - self._last_refresh) < _REFRESH_INTERVAL:
                self._dirty = True
                if self._pending_timer is None:
                    self._pending_timer = threading.Timer(_REFRESH_INTERVAL, self._deferred_refresh)
                    self._pending_timer.daemon = True
                    self._pending_timer.start()
                return

            from IPython.display import HTML, update_display

            update_display(HTML(self._render_jupyter_html()), display_id=self._display_id)
            self._last_refresh = now
            self._dirty = False

    def _deferred_refresh(self) -> None:
        """Called by timer to flush pending dirty state."""
        with self._refresh_lock:
            self._pending_timer = None
        if self._dirty:
            self._refresh()

    def _render_jupyter_html(self) -> str:
        """Build full HTML string from current state, preserving arrival order."""
        parts: list[str] = []

        # CSS (includes spinner keyframes)
        parts.append(self._render_css())

        parts.append('<div class="jcc-output">')

        # Header with model name
        if self._model:
            parts.append(f'<div class="jcc-header">Using model: {html_module.escape(self._model)}</div>')

        # Scrollable body with distinct background for SDK information
        parts.append('<div class="jcc-body">')

        # Items in arrival order
        for kind, item in self._items:
            if kind == "text":
                parts.append(f'<div class="jcc-content">{self._md_to_html(item)}</div>')
            elif kind == "tool":
                escaped_text = html_module.escape(item.display_text)
                if item.completed:
                    parts.append(f'<div class="jcc-tool done">\u2713 {escaped_text}</div>')
                else:
                    parts.append(f'<div class="jcc-tool">Tool: {escaped_text}</div>')
            elif kind == "thinking":
                parts.append(f'<div class="jcc-thinking">{html_module.escape(item)}</div>')

        # Error
        if self._error:
            parts.append(f'<div class="jcc-error">{html_module.escape(self._error)}</div>')

        # Interrupt
        if self._interrupted:
            parts.append('<div class="jcc-interrupt">Interrupted by user</div>')

        # Empty state
        has_content = self._model or self._items or self._error or self._interrupted
        if not has_content:
            parts.append('<div class="jcc-waiting">Thinking...</div>')

        # Spinner at bottom of body while still running
        if not self._stopped:
            parts.append('<div class="jcc-spinner"><div class="jcc-spinner-dot"></div><span>Running\u2026</span></div>')

        parts.append("</div>")  # close .jcc-body

        # Result metadata footer — only after processing completes
        if self._stopped and (self._result_meta or self._cells_created):
            parts.append(self._render_footer())

        parts.append("</div>")  # close .jcc-output
        return "".join(parts)

    def _render_footer(self) -> str:
        """Render result metadata footer (duration, tokens, turns, cell hint)."""
        meta = self._result_meta
        segments: list[str] = []
        if not meta and not self._cells_created:
            return ""
        if meta:
            duration_ms = meta.get("duration_ms", 0)
            if duration_ms:
                secs = duration_ms / 1000
                segments.append(f"{secs:.1f}s")
            usage = meta.get("usage")
            if usage:
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                if input_tokens or output_tokens:
                    segments.append(f"{input_tokens + output_tokens:,} tokens")
            num_turns = meta.get("num_turns", 0)
            if num_turns:
                segments.append(f"{num_turns} turn{'s' if num_turns != 1 else ''}")
        if self._cells_created:
            n = self._cells_created
            if self._replace_mode:
                if n == 1:
                    segments.append("\u2191 code cell replaced above")
                else:
                    segments.append(f"\u2191 code cell replaced above \u00b7 \u2193 {n - 1} more below")
            else:
                segments.append(f"\u2193 {n} code cell{'s' if n != 1 else ''} created below")
        if not segments:
            return ""
        return f'<div class="jcc-footer">{" \u00b7 ".join(segments)}</div>'

    def _render_css(self) -> str:
        """Return <style> block with all .jcc-* classes. Cached after first call.

        Two-layer design:

        **Base layer** — neutral ``inherit``/``color-mix()``/``rgba()`` values.
        In JupyterLab, the parent output cell is styled with ``--jp-*`` vars,
        so ``inherit`` picks up correct themed colors automatically.
        In bare environments, ``color-mix()`` derives proportional grays from
        ``currentColor`` for visual hierarchy, and ``rgba()`` provides subtle
        backgrounds/borders that work on any base color.

        **VS Code layer** — scoped via ``body[data-vscode-theme-kind]``.
        VS Code's notebook output iframe exposes ``--vscode-*`` CSS variables
        on ``<html>`` and a ``data-vscode-theme-kind`` attribute on ``<body>``.
        This layer explicitly reads ``--vscode-*`` for correct theme colors.
        """
        if self._css_cache is not None:
            return self._css_cache

        # fmt: off
        # --- Shorthand aliases ---
        # Base: neutral values (JupyterLab inherits from themed parents)
        text1 = "inherit"
        text2 = "color-mix(in srgb, currentColor, transparent 35%)"
        text3 = "color-mix(in srgb, currentColor, transparent 50%)"
        bg1   = "rgba(128,128,128,0.04)"
        bg2   = "rgba(128,128,128,0.08)"
        brd   = "rgba(128,128,128,0.25)"
        brd_l = "rgba(128,128,128,0.2)"
        brand = "#4a90d9"
        err   = "#d32f2f"
        warn  = "#f57c00"
        font  = "inherit"
        mono  = "monospace"
        # VS Code: explicit --vscode-* variables (available in output iframe)
        vs       = "body[data-vscode-theme-kind]"
        vs_fg    = "var(--vscode-foreground)"
        vs_fg2   = "var(--vscode-descriptionForeground)"
        vs_fg3   = "var(--vscode-disabledForeground)"
        vs_bg    = "var(--vscode-editor-background)"
        vs_bg2   = "var(--vscode-textCodeBlock-background)"
        vs_brd   = "var(--vscode-editorWidget-border)"
        vs_brand = "var(--vscode-focusBorder)"
        vs_err   = "var(--vscode-errorForeground)"
        vs_warn  = "var(--vscode-editorWarning-foreground)"
        vs_font  = "var(--vscode-font-family)"
        vs_mono  = "var(--vscode-editor-font-family)"
        # fmt: on

        self._css_cache = (
            "<style>"
            "@keyframes jcc-spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}"
            # --- Base layer ---
            f".jcc-output {{ font-family: {font}; font-size: 13px;"
            f" color: {text1}; padding: 8px 0; }}"
            f".jcc-header {{ color: {text2}; font-size: 0.85em; margin-bottom: 8px; }}"
            f".jcc-body {{ background: {bg1}; border: 1px solid {brd};"
            " border-radius: 4px; padding: 8px 12px; max-height: 400px; overflow-y: auto; }"
            f".jcc-tool {{ color: {text2}; font-size: 0.9em; padding: 1px 0; font-family: {mono}; }}"
            ".jcc-tool.done { opacity: 0.6; }"
            f".jcc-thinking {{ color: {text3}; font-style: italic; font-size: 0.85em;"
            " padding: 2px 0; white-space: pre-wrap; }"
            ".jcc-content { line-height: 1.5; }"
            ".jcc-content p { margin: 0.4em 0; }"
            f".jcc-content pre {{ background: {bg2}; padding: 8px 12px; border-radius: 4px; overflow-x: auto; }}"
            f".jcc-content code {{ font-family: {mono}; font-size: 0.9em; }}"
            f".jcc-content p code {{ background: {bg2}; padding: 1px 4px; border-radius: 3px; }}"
            f".jcc-error {{ color: {err}; margin-top: 8px; }}"
            f".jcc-interrupt {{ color: {warn}; margin-top: 8px; }}"
            f".jcc-waiting {{ color: {text3}; font-style: italic; }}"
            f".jcc-spinner {{ display: flex; align-items: center; gap: 8px; padding: 6px 0; color: {text2}; font-size: 0.85em; }}"
            f".jcc-spinner-dot {{ width: 12px; height: 12px; border: 2px solid {brd_l};"
            f" border-top: 2px solid {brand}; border-radius: 50%; animation: jcc-spin .8s linear infinite; }}"
            f".jcc-footer {{ color: {text3}; font-size: 0.8em; margin-top: 6px; }}"
            # --- VS Code layer (higher specificity via body[...] ancestor) ---
            f"{vs} .jcc-output {{ color: {vs_fg}; font-family: {vs_font}; }}"
            f"{vs} .jcc-header {{ color: {vs_fg2}; }}"
            f"{vs} .jcc-body {{ background: {vs_bg}; border-color: {vs_brd}; }}"
            f"{vs} .jcc-tool {{ color: {vs_fg2}; font-family: {vs_mono}; }}"
            f"{vs} .jcc-thinking {{ color: {vs_fg3}; }}"
            f"{vs} .jcc-content pre {{ background: {vs_bg2}; }}"
            f"{vs} .jcc-content code {{ font-family: {vs_mono}; }}"
            f"{vs} .jcc-content p code {{ background: {vs_bg2}; }}"
            f"{vs} .jcc-error {{ color: {vs_err}; }}"
            f"{vs} .jcc-interrupt {{ color: {vs_warn}; }}"
            f"{vs} .jcc-waiting {{ color: {vs_fg3}; }}"
            f"{vs} .jcc-spinner {{ color: {vs_fg2}; }}"
            f"{vs} .jcc-spinner-dot {{ border-color: {vs_brd}; border-top-color: {vs_brand}; }}"
            f"{vs} .jcc-footer {{ color: {vs_fg3}; }}"
            "</style>"
        )
        return self._css_cache

    # Security note: markdown output is not sanitized for HTML injection.
    # In Jupyter, the kernel already has full code execution access, so
    # injected HTML in widget output is not an escalation of privilege.
    def _md_to_html(self, text: str) -> str:
        """Convert markdown text to HTML."""
        return markdown.markdown(text, extensions=["fenced_code", "tables", "nl2br"])

    # ------------------------------------------------------------------
    # Fallback: plain print for environments where nothing else works
    # ------------------------------------------------------------------

    def _print_fallback_latest(self) -> None:
        """Print only the most recently added item (avoids duplicating earlier output)."""
        if self._model and not self._items:
            print(f"Using model: {self._model}", flush=True)
        if self._items:
            kind, item = self._items[-1]
            if kind == "text":
                print(item, flush=True)
            elif kind == "tool":
                prefix = "  \u2713" if item.completed else "  Tool:"
                print(f"{prefix} {item.display_text}", flush=True)
            elif kind == "thinking":
                print(f"  [thinking] {item[:80]}{'...' if len(item) > 80 else ''}", flush=True)
        if self._interrupted:
            print("Query interrupted by user", flush=True)
        if self._error:
            print(f"Error: {self._error}", flush=True)
        if self._session_id:
            print(f"Session: {self._session_id}", flush=True)
