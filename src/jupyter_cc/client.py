"""
Claude API client integration for jupyter_cc.
Handles streaming queries and message processing by creating fresh ClaudeSDKClient instances.
"""

import contextlib
import logging
from typing import Any

import anyio
from anyio import BrokenResourceError, ClosedResourceError
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from .display import StreamingDisplay

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
        *,
        display: StreamingDisplay | None = None,
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
            display: Pre-created StreamingDisplay. When provided, the caller is
                responsible for start() and stop(). When None, this method creates
                and manages its own display.

        Returns:
            Tuple of (assistant_messages, tool_calls)
        """
        # Ensure we have an async checkpoint at the start
        await anyio.lowlevel.checkpoint()

        tool_calls: list[str] = []
        assistant_messages: list[str] = []
        self._interrupt_requested = False

        # If no external display provided, create and manage our own
        owns_display = display is None
        if owns_display:
            display = StreamingDisplay(verbose=verbose)
            display.start()

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
                assert display is not None  # noqa: S101  # Guaranteed by owns_display logic above

                async def process_messages() -> None:
                    """Iterate over streamed messages and update display + result lists."""
                    nonlocal has_printed_model
                    async for message in client.receive_response():
                        if message is None:
                            continue  # Skipped by patched parser (unknown type)

                        # Log every message type for debugging (visible with %cc --verbose
                        # or logging.getLogger("jupyter_cc").setLevel(logging.DEBUG))
                        logger.debug("SDK message: %s", type(message).__name__)

                        if isinstance(message, AssistantMessage):
                            if hasattr(message, "model") and not has_printed_model:
                                display.set_model(message.model)
                                has_printed_model = True
                            for block in message.content:
                                logger.debug("  block: %s", type(block).__name__)
                                if isinstance(block, TextBlock) and block.text.strip():
                                    display.add_text(block.text)
                                    assistant_messages.append(block.text)
                                elif isinstance(block, ToolUseBlock):
                                    display.add_tool_call(block.name, block.input, block.id)
                                    tool_calls.append(f"{block.name}: {block.input}")
                                elif isinstance(block, ThinkingBlock) and block.thinking.strip():
                                    display.add_thinking(block.thinking)
                                elif isinstance(block, ToolResultBlock):
                                    logger.debug("  ToolResultBlock in assistant (tool_use_id=%s)", block.tool_use_id)
                        elif isinstance(message, UserMessage):
                            # UserMessage contains tool results — mark tool calls as completed
                            if isinstance(message.content, list):
                                for block in message.content:
                                    if isinstance(block, ToolResultBlock):
                                        display.complete_tool_call(block.tool_use_id)
                        elif isinstance(message, ResultMessage):
                            if message.session_id and message.session_id != self._session_id:
                                self._session_id = message.session_id
                                display.set_session_id(self._session_id)
                            display.set_result(
                                duration_ms=message.duration_ms,
                                total_cost_usd=message.total_cost_usd,
                                usage=message.usage,
                                num_turns=message.num_turns,
                            )
                            break
                        else:
                            # SystemMessage, StreamEvent, or future types — log but don't render
                            logger.debug("  unhandled message: %r", message)

                if enable_interrupt:
                    # Collect messages with interrupt checking.
                    # Exceptions inside collect_messages are captured (not raised)
                    # to prevent anyio from wrapping them in an ExceptionGroup.
                    collection_error: Exception | None = None
                    collection_done = anyio.Event()

                    async with anyio.create_task_group() as tg:

                        async def collect_messages() -> None:
                            nonlocal collection_error
                            try:
                                await process_messages()
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
                                display.show_interrupt()
                                break

                            # Check if we're done (success or error)
                            if collection_done.is_set():
                                break

                            await anyio.sleep(0.05)

                    # Re-raise any error that occurred during message collection
                    if collection_error is not None:
                        raise collection_error
                else:
                    await process_messages()

        except Exception as e:
            # Unwrap ExceptionGroup to get the actual error(s)
            errors = list(e.exceptions) if isinstance(e, ExceptionGroup) else [e]

            for err in errors:
                if isinstance(err, (BrokenPipeError, ConnectionError, BrokenResourceError, ClosedResourceError)):
                    if not self._interrupt_requested:
                        display.show_error("Connection was lost. A new connection will be created automatically.")
                else:
                    display.show_error(str(err))
                    logger.exception("Unexpected error during query", exc_info=err)
        finally:
            if owns_display:
                display.stop()
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
