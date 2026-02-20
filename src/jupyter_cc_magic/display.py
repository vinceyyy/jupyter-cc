"""
Rich streaming display for Claude responses.

In Jupyter notebooks, uses ipywidgets.HTML for thread-safe in-place updates.
A pure-CSS spinner runs independently to show the cell is active, while the
content area updates as SDK events arrive. This avoids all the threading issues
with IPython DisplayHandle (which loses cell association from background threads).

In terminals, uses Rich Live for ANSI-based live rendering.
Falls back to plain print() if neither works.

IMPORTANT: In Jupyter, the StreamingDisplay must be created and start()'d from
the main IPython thread so the widgets are displayed in the correct cell output.
After start(), add_text / add_tool_call / etc. are safe to call from any thread
because ipywidgets property assignments are thread-safe.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .constants import EXECUTE_PYTHON_TOOL_NAME
from .integration import is_in_jupyter_notebook

logger = logging.getLogger(__name__)

# Braille spinner frames for active tool calls — cycled on each render
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# How often the Jupyter refresh thread redraws (seconds).
# ~7 FPS — enough for a smooth braille spinner without excessive overhead.
_JUPYTER_REFRESH_INTERVAL = 0.15

# Pure-CSS spinner shown above the content panel while running.
# Runs entirely in the browser — no Python-side refresh needed.
_CSS_SPINNER_HTML = (
    '<div style="display:flex;align-items:center;gap:8px;padding:4px 0;'
    'font-family:sans-serif;font-size:13px;color:#888">'
    '<div style="width:14px;height:14px;border:2px solid #e0e0e0;'
    "border-top:2px solid #4a90d9;border-radius:50%;"
    'animation:jcc-spin .8s linear infinite"></div>'
    "<span>Running&hellip;</span></div>"
    "<style>@keyframes jcc-spin{0%{transform:rotate(0deg)}"
    "100%{transform:rotate(360deg)}}</style>"
)


def format_tool_call(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Format tool calls to match Claude CLI style with meaningful details."""
    tool_display_names = {
        "LS": "List",
        "GrepToolv2": "Search",
        EXECUTE_PYTHON_TOOL_NAME: "CreateNotebookCell",
    }

    display_name = tool_display_names.get(tool_name, tool_name)

    if tool_name == "Read":
        file_path = tool_input.get("file_path", "")
        parts = [f"{display_name}({file_path})"]
        if "offset" in tool_input:
            parts.append(f"offset: {tool_input['offset']}")
        if "limit" in tool_input:
            parts.append(f"limit: {tool_input['limit']}")
        return " ".join(parts)

    if tool_name == "LS":
        path = tool_input.get("path", "")
        return f"{display_name}({path})"

    if tool_name == "GrepToolv2":
        pattern = tool_input.get("pattern", "")
        parts = [f'{display_name}(pattern: "{pattern}"']
        path = tool_input.get("path")
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

    if tool_name == "Bash":
        command = tool_input.get("command", "")
        return f'{display_name}("{command}")'

    if tool_name in ["Write", "Edit", "MultiEdit"]:
        file_path = tool_input.get("file_path", "")
        return f"{display_name}({file_path})"

    if tool_name == "Glob":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        if path:
            return f'{display_name}(pattern: "{pattern}", path: "{path}")'
        return f'{display_name}("{pattern}")'

    if tool_name == "WebFetch":
        url = tool_input.get("url", "")
        return f'{display_name}("{url}")'

    if tool_name == "WebSearch":
        query = tool_input.get("query", "")
        return f'{display_name}("{query}")'

    if tool_name == "TodoWrite":
        todos = tool_input.get("todos", [])
        return f"{display_name}({len(todos)} items)"

    if tool_name == EXECUTE_PYTHON_TOOL_NAME:
        description = tool_input.get("description", "")
        if description:
            return f'{display_name}("{description}")'
        return display_name

    return display_name


class _ToolCallEntry:
    """Internal state for a single tool call being displayed."""

    def __init__(self, display_text: str, tool_id: str) -> None:
        self.display_text = display_text
        self.tool_id = tool_id
        self.completed = False


class StreamingDisplay:
    """
    Streaming display for Claude responses.

    Jupyter mode:
      - CSS spinner widget (always animating, separate from content)
      - ipywidgets.HTML content widget (Rich-rendered, updated by refresh thread)
      - VBox stacks spinner above content
      - Refresh thread updates content_widget.value (thread-safe via ipywidgets)

    Terminal mode:
      - Rich Live for ANSI-based in-place rendering

    Fallback:
      - Plain print()

    Must be created and start()'d from the main IPython thread.
    After that, state-mutating methods are safe from any thread.
    """

    def __init__(self, *, verbose: bool = False, jupyter: bool | None = None) -> None:
        self._verbose = verbose
        self._model: str | None = None
        self._text_blocks: list[str] = []
        self._tool_calls: list[_ToolCallEntry] = []
        self._session_id: str | None = None
        self._error: str | None = None
        self._interrupted = False
        self._spinner_tick = 0
        self._live: Any | None = None  # rich.live.Live or None
        # Auto-detect only works from the main IPython thread.
        self._jupyter = jupyter if jupyter is not None else is_in_jupyter_notebook()
        self._fallback = False

        # Jupyter widgets (created in start())
        self._spinner_widget: Any | None = None  # ipywidgets.HTML
        self._content_widget: Any | None = None  # ipywidgets.HTML
        self._container: Any | None = None  # ipywidgets.VBox

        # Jupyter periodic refresh state
        self._dirty = False
        self._stop_event: threading.Event | None = None
        self._refresh_thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the live display. Must be called from the main IPython thread."""
        if self._jupyter:
            try:
                import ipywidgets as widgets
                from IPython.display import display

                # CSS spinner: runs entirely in the browser, always animated
                self._spinner_widget = widgets.HTML(value=_CSS_SPINNER_HTML)
                # Content: Rich-rendered HTML, updated by the refresh thread
                self._content_widget = widgets.HTML(value="")
                self._container = widgets.VBox([self._spinner_widget, self._content_widget])
                display(self._container)

                # Start a daemon thread that periodically renders content.
                # Only this thread writes to _content_widget.value.
                self._stop_event = threading.Event()
                self._refresh_thread = threading.Thread(
                    target=self._jupyter_periodic_refresh,
                    daemon=True,
                    name="jupyter-cc-display-refresh",
                )
                self._refresh_thread.start()
            except Exception:
                logger.debug("ipywidgets unavailable, falling back to print()", exc_info=True)
                self._jupyter = False
                self._fallback = True
            return

        try:
            from rich.live import Live

            self._live = Live(
                self._render(),
                refresh_per_second=12,
                transient=False,
            )
            self._live.start()
        except Exception:
            logger.debug("Rich Live display unavailable, falling back to print()")
            self._fallback = True

    def stop(self) -> None:
        """Stop the live display, leaving final output visible."""
        if self._jupyter:
            # Shut down the periodic refresh thread
            if self._stop_event is not None:
                self._stop_event.set()
            if self._refresh_thread is not None:
                self._refresh_thread.join(timeout=2)
                self._refresh_thread = None
            # Hide the CSS spinner
            if self._spinner_widget is not None:
                self._spinner_widget.layout.display = "none"
            # Final content render
            if self._content_widget is not None:
                try:
                    self._content_widget.value = self._render_html_string()
                except Exception:
                    logger.debug("Error rendering final Jupyter content", exc_info=True)
            return

        if self._live is not None:
            try:
                self._live.update(self._render())
                self._live.stop()
            except Exception:
                logger.debug("Error stopping Rich Live display", exc_info=True)
            self._live = None

    def set_model(self, model: str) -> None:
        """Set the model name shown in the header."""
        self._model = model
        self._refresh()

    def add_text(self, text: str) -> None:
        """Append a text block to the display."""
        self._text_blocks.append(text)
        self._refresh()

    def add_tool_call(self, tool_name: str, tool_input: dict[str, Any], tool_id: str) -> None:
        """Add an active tool call (shown with spinner indicator)."""
        display_text = format_tool_call(tool_name, tool_input)
        entry = _ToolCallEntry(display_text, tool_id)
        self._tool_calls.append(entry)
        if self._verbose:
            entry.display_text += f"\n  Arguments: {tool_input}"
        self._refresh()

    def complete_tool_call(self, tool_id: str) -> None:
        """Mark a tool call as completed (spinner -> checkmark)."""
        for entry in self._tool_calls:
            if entry.tool_id == tool_id:
                entry.completed = True
                break
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

    def _refresh(self) -> None:
        """Push the latest render to the display."""
        if self._fallback:
            self._print_fallback_latest()
            return

        if self._jupyter:
            # In Jupyter mode, just mark dirty. The periodic refresh thread
            # handles widget updates to keep the braille spinner animated
            # and to batch rapid state changes.
            self._dirty = True
            return

        if self._live is not None:
            try:
                self._spinner_tick = (self._spinner_tick + 1) % len(_SPINNER_FRAMES)
                self._live.update(self._render())
            except Exception:
                logger.debug("Error refreshing Rich Live display", exc_info=True)

    def _jupyter_periodic_refresh(self) -> None:
        """Daemon thread: periodically updates the Jupyter content widget."""
        while self._stop_event is not None and not self._stop_event.wait(_JUPYTER_REFRESH_INTERVAL):
            # Refresh when state changed OR when there are active tool calls (braille spinner)
            has_active_tools = any(not e.completed for e in self._tool_calls)
            if self._dirty or has_active_tools:
                self._dirty = False
                self._spinner_tick = (self._spinner_tick + 1) % len(_SPINNER_FRAMES)
                try:
                    if self._content_widget is not None:
                        self._content_widget.value = self._render_html_string()
                except Exception:
                    logger.debug("Error in Jupyter periodic refresh", exc_info=True)
                    break

    def _render(self) -> Any:
        """Build the Rich renderable for the current state."""
        from rich.console import Group
        from rich.markdown import Markdown
        from rich.panel import Panel
        from rich.text import Text

        parts: list[Any] = []

        # Model header
        if self._model:
            parts.append(Text(f"Model: {self._model}", style="bold cyan"))
            parts.append(Text(""))

        # Text blocks
        for block in self._text_blocks:
            parts.append(Markdown(block))
            parts.append(Text(""))

        # Tool calls
        for entry in self._tool_calls:
            if entry.completed:
                indicator = Text("  \u2713 ", style="bold green")
            else:
                frame = _SPINNER_FRAMES[self._spinner_tick % len(_SPINNER_FRAMES)]
                indicator = Text(f"  {frame} ", style="bold yellow")
            line = Text.assemble(indicator, entry.display_text)
            parts.append(line)

        # Interrupt notice
        if self._interrupted:
            parts.append(Text(""))
            parts.append(Text("Query interrupted by user", style="bold yellow"))

        # Error
        if self._error:
            parts.append(Text(""))
            parts.append(Text(f"Error: {self._error}", style="bold red"))

        # Session ID footer
        if self._session_id:
            parts.append(Text(""))
            parts.append(Text(f"Session: {self._session_id}", style="dim"))

        if not parts:
            parts.append(Text("Waiting for response...", style="dim italic"))

        return Panel(Group(*parts), title="Claude", border_style="blue", expand=True)

    def _render_html_string(self) -> str:
        """Render content to an HTML string for the Jupyter ipywidgets.HTML widget."""
        from rich.console import Console

        console = Console(record=True, width=120, force_jupyter=False, force_terminal=True)
        console.print(self._render())
        html = console.export_html(inline_styles=True)
        return f'<div style="font-family: monospace; font-size: 13px;">{html}</div>'

    # ------------------------------------------------------------------
    # Fallback: plain print for environments where nothing else works
    # ------------------------------------------------------------------

    def _print_fallback_latest(self) -> None:
        """Print only the most recently added item (avoids duplicating earlier output)."""
        if self._model and len(self._text_blocks) == 0 and len(self._tool_calls) == 0:
            print(f"Model: {self._model}", flush=True)
        if self._text_blocks:
            print(self._text_blocks[-1], flush=True)
        if self._tool_calls:
            entry = self._tool_calls[-1]
            if not entry.completed:
                print(f"  ... {entry.display_text}", flush=True)
            else:
                print(f"  \u2713 {entry.display_text}", flush=True)
        if self._interrupted:
            print("Query interrupted by user", flush=True)
        if self._error:
            print(f"Error: {self._error}", flush=True)
        if self._session_id:
            print(f"Session: {self._session_id}", flush=True)
