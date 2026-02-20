"""
Claude API client integration for jupyter_cc.
Handles streaming queries and message processing by creating fresh ClaudeSDKClient instances.
"""

from __future__ import annotations

import contextlib
import logging
import traceback
from typing import TYPE_CHECKING, Any

import anyio
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)

from .constants import EXECUTE_PYTHON_TOOL_NAME
from .integration import is_in_jupyter_notebook

if TYPE_CHECKING:
    from .magics import ClaudeCodeMagics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patch SDK message parser to skip unknown message types (e.g. rate_limit_event)
# instead of raising MessageParseError.  The SDK (v0.1.39) hard-fails on any
# type it doesn't recognise, but the API can introduce new informational event
# types at any time.  We return None for unknown types and filter them out
# during iteration.
# ---------------------------------------------------------------------------
def _patch_sdk_message_parser() -> None:
    try:
        from claude_agent_sdk._internal import message_parser as _mp

        _original_parse = _mp.parse_message

        def _lenient_parse(data: dict[str, Any]) -> Any:
            try:
                return _original_parse(data)
            except Exception:
                msg_type = data.get("type", "<unknown>") if isinstance(data, dict) else "<non-dict>"
                logger.debug("Skipping unrecognised SDK message type: %s", msg_type)
                return None

        _mp.parse_message = _lenient_parse  # type: ignore[assignment]
    except Exception:
        logger.debug("Could not patch SDK message parser – unknown types will still raise")


_patch_sdk_message_parser()

MARKDOWN_PATTERNS = [
    "```",  # Code blocks
    "`",  # Inline code
    "    ",  # Indented code blocks
    "\t",  # Indented code blocks
    "**",  # Bold
    "##",  # Headers (checking for at least level 2)
    "](",  # Links/images
    "---",  # Tables
    ">",  # Blockquotes
    "~~",  # Strikethrough
]


def _display_claude_message_with_markdown(text: str) -> None:
    """Display a Claude message with markdown rendering if relevant and available."""
    claude_message = f"💭 Claude: {text}"

    # IPython displays markdown as <IPython.core.display.Markdown object>
    if not is_in_jupyter_notebook():
        print(claude_message, flush=True)
        return

    # Simple check: if text has any markdown elements, use markdown display
    has_markdown = any(pattern in text for pattern in MARKDOWN_PATTERNS)
    if not has_markdown:
        print(claude_message, flush=True)
        return

    try:
        from IPython.display import Markdown, display

        display(Markdown(claude_message))
    except ImportError:
        print(claude_message, flush=True)


def _format_tool_call(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Format tool calls to match Claude CLI style with meaningful details."""
    # Map tool names to their user-facing names
    tool_display_names = {
        "LS": "List",
        "GrepToolv2": "Search",
        EXECUTE_PYTHON_TOOL_NAME: "CreateNotebookCell",
    }

    display_name = tool_display_names.get(tool_name, tool_name)

    # Format based on tool type with most relevant info
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

        # Add path if not current directory
        path = tool_input.get("path")
        parts.append(f'path: "{path}"')

        # Add other relevant options
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

    return display_name


class ClaudeClientManager:
    """Manages ClaudeSDKClient instances for Jupyter magic, creating fresh clients per query."""

    def __init__(self) -> None:
        """Initialize the client manager."""
        self._session_id: str | None = None
        self._interrupt_requested: bool = False
        self._current_client: ClaudeSDKClient | None = None

    async def query_sync(
        self,
        prompt: str | list[dict[str, Any]],
        options: ClaudeAgentOptions,
        is_new_conversation: bool,
        verbose: bool = False,
        enable_interrupt: bool = True,
    ) -> tuple[list[str], list[str]]:
        """
        Send a query and collect all responses synchronously.
        Creates a new ClaudeSDKClient for each query.

        Args:
            prompt: The prompt to send to Claude (string or list of content blocks)
            options: Claude Code options to use for this query
            is_new_conversation: Whether this is a new conversation
            verbose: Whether to show verbose output
            enable_interrupt: If True, enables interrupt handling

        Returns:
            Tuple of (assistant_messages, tool_calls)
        """
        # Ensure we have an async checkpoint at the start
        await anyio.lowlevel.checkpoint()

        tool_calls: list[str] = []
        assistant_messages: list[str] = []
        self._interrupt_requested = False

        # If we have a stored session ID and this is not a new conversation, use it for resumption
        # But only if the options don't already have a resume value set
        if self._session_id and not is_new_conversation:
            if not options.resume:
                options.resume = self._session_id
            # Also set continue_conversation to true when resuming
            options.continue_conversation = True

        # Create a new client for this query — context manager handles connect/disconnect
        try:
            async with ClaudeSDKClient(options=options) as client:
                self._current_client = client

                # Send the query based on prompt type
                if isinstance(prompt, list):
                    # Structured content with images — plain async generator
                    async def content_generator():  # type: ignore[override]
                        yield {
                            "type": "user",
                            "message": {"role": "user", "content": prompt},
                            "parent_tool_use_id": None,
                        }

                    await client.query(content_generator())
                else:
                    # Simple string prompt
                    await client.query(prompt)

                # Process responses
                has_printed_model = not is_new_conversation

                # If interrupt support is enabled, we need to handle messages differently
                if enable_interrupt:
                    # Collect messages with interrupt checking.
                    # Exceptions inside collect_messages are captured (not raised)
                    # to prevent anyio from wrapping them in an ExceptionGroup.
                    messages_to_process: list[Any] = []
                    collection_error: Exception | None = None
                    collection_done = anyio.Event()

                    async with anyio.create_task_group() as tg:

                        async def collect_messages() -> None:
                            nonlocal collection_error
                            try:
                                async for message in client.receive_response():
                                    if message is None:
                                        continue  # Skipped by patched parser (unknown type)
                                    messages_to_process.append(message)
                                    if isinstance(message, ResultMessage):
                                        break
                            except Exception as exc:
                                # Catch SDK/connection errors but let CancelledError
                                # propagate — that's how anyio signals scope cancellation.
                                collection_error = exc
                            finally:
                                collection_done.set()

                        tg.start_soon(collect_messages)

                        # Monitor for interrupts
                        while True:
                            if self._interrupt_requested:
                                tg.cancel_scope.cancel()
                                await client.interrupt()
                                print("\n⚠️ Query interrupted by user", flush=True)
                                break

                            # Check if we're done (success or error)
                            if collection_done.is_set():
                                break

                            await anyio.sleep(0.05)

                    # Re-raise any error that occurred during message collection
                    if collection_error is not None:
                        raise collection_error

                    # Process collected messages
                    for message in messages_to_process:
                        if isinstance(message, AssistantMessage):
                            if hasattr(message, "model") and not has_printed_model:
                                print(f"🧠 Claude model: {message.model}", flush=True)
                                has_printed_model = True
                            for block in message.content:
                                if isinstance(block, TextBlock) and block.text.strip():
                                    _display_claude_message_with_markdown(block.text)
                                    assistant_messages.append(block.text)
                                elif isinstance(block, ToolUseBlock):
                                    tool_display = _format_tool_call(block.name, block.input)
                                    print(f"⏺ {tool_display}", flush=True)
                                    if verbose:
                                        print(f"  ⎿  Arguments: {block.input}", flush=True)
                                    tool_calls.append(f"{block.name}: {block.input}")
                        elif isinstance(message, ResultMessage):
                            # Extract and store session ID from result
                            if message.session_id and message.session_id != self._session_id:
                                self._session_id = message.session_id
                                print(
                                    f"📍 Claude Code Session ID: {self._session_id}",
                                    flush=True,
                                )
                else:
                    # Simple mode without interrupt support
                    async for message in client.receive_response():
                        if message is None:
                            continue  # Skipped by patched parser (unknown type)
                        if isinstance(message, AssistantMessage):
                            if hasattr(message, "model") and not has_printed_model:
                                print(f"🧠 Claude model: {message.model}", flush=True)
                                has_printed_model = True
                            for block in message.content:
                                if isinstance(block, TextBlock) and block.text.strip():
                                    _display_claude_message_with_markdown(block.text)
                                    assistant_messages.append(block.text)
                                elif isinstance(block, ToolUseBlock):
                                    tool_display = _format_tool_call(block.name, block.input)
                                    print(f"\n⏺ {tool_display}", flush=True)
                                    if verbose:
                                        print(f"  ⎿  Arguments: {block.input}", flush=True)
                                    tool_calls.append(f"{block.name}: {block.input}")
                        elif isinstance(message, ResultMessage):
                            # Extract and store session ID from result
                            if message.session_id and message.session_id != self._session_id:
                                self._session_id = message.session_id
                                print(
                                    f"\n📍 Claude Code Session ID: {self._session_id}",
                                    flush=True,
                                )
                            break

        except Exception as e:
            # Unwrap ExceptionGroup to get the actual error(s)
            errors = list(e.exceptions) if isinstance(e, ExceptionGroup) else [e]

            for err in errors:
                error_type_str = str(type(err))
                error_msg_str = str(err)
                if any(
                    token in error_type_str or token in error_msg_str
                    for token in [
                        "BrokenResourceError",
                        "BrokenPipeError",
                        "ClosedResourceError",
                    ]
                ):
                    if not self._interrupt_requested:
                        print(
                            "\n⚠️ Connection was lost. A new connection will be created automatically.",
                            flush=True,
                        )
                else:
                    print(f"\n❌ Error during Claude execution: {err!s}", flush=True)
                    traceback.print_exception(type(err), err, err.__traceback__)
        finally:
            self._current_client = None

        return assistant_messages, tool_calls

    async def handle_interrupt(self) -> None:
        """Send an interrupt signal to the current client if one exists."""
        self._interrupt_requested = True
        if self._current_client is not None:
            with contextlib.suppress(Exception):
                await self._current_client.interrupt()
        await anyio.lowlevel.checkpoint()

    def reset_session(self) -> None:
        """Clear the stored session ID to start a new conversation."""
        self._session_id = None

    @property
    def session_id(self) -> str | None:
        """Get the current session ID if available."""
        return self._session_id


async def run_streaming_query(
    parent: ClaudeCodeMagics,
    prompt: str | list[dict[str, Any]],
    options: ClaudeAgentOptions,
    verbose: bool,
) -> None:
    """
    Run Claude query with real-time message streaming using a fresh client.
    This function maintains compatibility with the existing interface.
    """
    # Ensure client manager exists
    if not hasattr(parent, "_client_manager") or parent._client_manager is None:
        parent._client_manager = ClaudeClientManager()

    # Run the query with a fresh client
    await parent._client_manager.query_sync(prompt, options, parent._config_manager.is_new_conversation, verbose)

    # Update last output line
    parent._history_manager.update_last_output_line()
