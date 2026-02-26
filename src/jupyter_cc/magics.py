"""
IPython magic for Claude Code using SDK's in-process MCP server.

Uses the Claude Code SDK's @tool decorator and create_sdk_mcp_server to
provide direct in-process tool execution: Claude -> SDK Tool -> Jupyter.
"""

import argparse
import contextlib
import logging
import queue
import signal
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
from claude_agent_sdk import (
    ClaudeAgentOptions,
    McpServerConfig,
    create_sdk_mcp_server,
    tool,
)
from IPython.core.magic import Magics, line_cell_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring

from .capture import (
    extract_images_from_captured,
    format_images_summary,
)
from .client import ClaudeClientManager
from .config import ConfigManager
from .display import display_status
from .history import HistoryManager
from .integration import (
    adjust_cell_queue_markers,
    create_approval_cell,
    is_in_jupyter_notebook,
    process_cell_queue,
)
from .prompt import get_system_prompt, prepare_imported_files_content
from .variables import VariableTracker

if TYPE_CHECKING:
    from types import FrameType

    from IPython.core.interactiveshell import InteractiveShell

    from .watcher import CellWatcher

logger = logging.getLogger(__name__)

# Global variables
_magic_instance: "ClaudeCodeMagics | None" = None


async def _tool_error(message: str) -> dict[str, Any]:
    """Build an error response for an SDK tool call."""
    await anyio.lowlevel.checkpoint()
    return {"content": [{"type": "text", "text": message}], "is_error": True}


async def _tool_success(message: str) -> dict[str, Any]:
    """Build a success response for an SDK tool call."""
    await anyio.lowlevel.checkpoint()
    return {"content": [{"type": "text", "text": message}]}


@tool(
    "create_python_cell",
    "Create a cell with Python code in the IPython environment",
    {"code": str, "description": str},
)
async def execute_python_tool(args: dict[str, Any]) -> dict[str, Any]:
    """Handle create_python_cell tool calls - create cells and return immediately."""
    if _magic_instance is None:
        return await _tool_error("❌ Magic instance not initialized")

    code = args.get("code", "")
    description = args.get("description", "")
    if not code:
        return await _tool_error("❌ No code provided")

    # Check if max_cells limit has been reached
    if _magic_instance._config_manager.create_python_cell_count >= _magic_instance._config_manager.max_cells:
        return await _tool_error(
            f"❌ Maximum number of cells ({_magic_instance._config_manager.max_cells}) reached for this turn. "
            "Please wait for the user to provide additional input before creating more cells."
        )

    # Generate tool_use_id for tracking
    tool_use_id = str(uuid.uuid4())

    try:
        # Get or create request ID
        request_id = _magic_instance.current_request_id
        if not request_id:
            request_id = str(uuid.uuid4())
            _magic_instance.current_request_id = request_id

        # Initialize the request if it doesn't exist
        if request_id not in _magic_instance.pending_requests:
            _magic_instance.pending_requests[request_id] = {}

        # Create cell in IPython
        _magic_instance._create_approval_cell(code, request_id, tool_use_id, description)

        # Increment the counter after successful cell creation
        _magic_instance._config_manager.create_python_cell_count += 1

        return await _tool_success(
            "✅ Code cell created. Waiting for user to review and execute. The user will run %cc when ready to proceed."
        )

    except Exception as e:
        logger.exception("Error in create_python_cell tool")
        return await _tool_error(f"❌ Error creating cells: {e}")


@magics_class
class ClaudeCodeMagics(Magics):
    """IPython magic for Claude Code with direct SDK integration."""

    def __init__(self, shell: "InteractiveShell", cell_watcher: "CellWatcher") -> None:
        super().__init__(shell)
        global _magic_instance  # noqa: PLW0603
        _magic_instance = self

        self.cell_watcher = cell_watcher
        # Initialize delegated components
        self._variable_tracker = VariableTracker(shell)
        self._history_manager = HistoryManager(shell)
        self._config_manager = ConfigManager()

        # Request tracking for cell-based flow
        # request_id -> request data
        self.pending_requests: dict[str, dict[str, Any]] = {}

        # Track the current request ID for the batch of tool calls
        self.current_request_id: str | None = None

        # Claude client manager for persistent connections
        # Will be initialized on first use
        self._client_manager: ClaudeClientManager | None = None

        # Create SDK MCP server once — the tool config is static
        self._sdk_server = create_sdk_mcp_server(
            name="jupyter_executor",
            version="1.0.0",
            tools=[execute_python_tool],
        )

        # Register post-execution hook to process cell queue
        if shell is not None:
            shell.events.register("post_run_cell", self._post_run_cell_hook)

        # Don't let ipython intercept question marks at the end of line magic
        from IPython.core.inputtransformer2 import EscapedCommand, HelpEnd

        HelpEnd.priority = EscapedCommand.priority + 1

    def _create_approval_cell(
        self, code: str, request_id: str, tool_use_id: str | None = None, description: str = ""
    ) -> None:
        """Create a cell for user approval of code execution."""
        create_approval_cell(self, code, request_id, self._config_manager.should_replace_cell, tool_use_id, description)

    def _post_run_cell_hook(self, result: Any) -> None:
        """Hook that runs after each cell execution to process the queue."""
        if self.shell is None:
            return

        # Check if we have a cell queue
        cell_queue: list[dict[str, Any]] = self.shell.user_ns.get("_claude_cell_queue", [])
        if not cell_queue:
            return

        # Get the last executed code
        last_input = self.shell.user_ns.get("In", [""])[-1] if "In" in self.shell.user_ns else ""

        # Find the next unexecuted cell in the queue
        next_expected_index: int | None = None
        next_expected_marker: str | None = None
        next_expected_marker_id: str | None = None
        for i, cell_info in enumerate(cell_queue):
            if not cell_info["executed"]:
                next_expected_index = i
                next_expected_marker = cell_info.get("marker", "")
                next_expected_marker_id = cell_info["marker_id"]
                break

        # Check if the executed code contains the expected marker
        executed_expected = False
        if next_expected_marker and last_input.startswith(next_expected_marker):
            executed_expected = True
            # Mark this cell as executed
            if next_expected_index is not None:
                cell_queue[next_expected_index]["executed"] = True
                cell_queue[next_expected_index]["had_exception"] = not result.success if result else False
                if result and not result.success and result.error_in_exec:
                    # Store the exception information
                    cell_queue[next_expected_index]["error"] = {
                        "type": type(result.error_in_exec).__name__,
                        "message": str(result.error_in_exec),
                    }

        # Only process the queue if the expected cell was executed successfully
        if executed_expected and result and result.success:
            # Set up the next cell
            process_cell_queue(self)
        elif executed_expected and result and not result.success:
            # Expected cell was executed but failed - notify user but don't set up next cell
            remaining = sum(1 for cell in cell_queue if not cell.get("executed", False))
            if remaining > 0:
                display_status(
                    f"⚠️ Execution failed. {remaining} cell(s) remaining in queue.\n"
                    "Run %cc to continue with the error in context, or %cc_new to start fresh.",
                    kind="warning",
                )
        elif not executed_expected and next_expected_marker:
            # User executed something else - check if it's a different queued cell
            for cell_info in cell_queue:
                marker = cell_info.get("marker", "")
                if marker and last_input.startswith(marker):
                    marker_id = cell_info["marker_id"]
                    display_status(
                        f"⚠️ Claude cell [{marker_id}] executed out of order. Expected Claude cell [{next_expected_marker_id}] to run next.\n"
                        "Run the expected cell to continue the automatic queue, or use %cc to report results.",
                        kind="warning",
                    )
                    break

    def __del__(self) -> None:
        """Clean up resources when magic is destroyed."""
        global _magic_instance  # noqa: PLW0603
        _magic_instance = None

        # Unregister post-execution hook
        if self.shell is not None:
            with contextlib.suppress(Exception):
                self.shell.events.unregister("post_run_cell", self._post_run_cell_hook)

        self._client_manager = None

    def _execute_prompt(
        self,
        prompt: str,
        previous_execution: str = "",
        captured_images: list[dict[str, Any]] | None = None,
        verbose: bool = False,
    ) -> None:
        """
        Implementation of claude_local functionality with support for captured images.

        Args:
            prompt: User prompt
            previous_execution: Previous execution context
            captured_images: Optional list of captured images from previous execution
            verbose: Whether to show verbose output
        """
        if captured_images is None:
            captured_images = []

        # Generate a new request ID for this batch of tool calls
        self.current_request_id = str(uuid.uuid4())

        # Reset create_python_cell_count for this turn
        self._config_manager.create_python_cell_count = 0

        # Check for captured output with images
        if self.shell is not None and "_claude_captured_output" in self.shell.user_ns:
            captured_output = self.shell.user_ns["_claude_captured_output"]
            captured_images = extract_images_from_captured(captured_output)

            # Clean up the captured output variable
            del self.shell.user_ns["_claude_captured_output"]

        # Get current variables for context
        variables_info = self._variable_tracker.get_variables_info()

        # Capture any shell output since last call
        # Skip this if we're loading cells for a new conversation to avoid duplication
        shell_output = ""
        if not self._config_manager.is_new_conversation:
            shell_output = self._history_manager.get_shell_output_since_last()

        # Build enhanced prompt with conversation history
        enhanced_prompt_text = f"""
Your client's request is <request>{prompt}</request>

{variables_info}
{previous_execution}
"""

        # Add shell output if present
        if shell_output:
            enhanced_prompt_text += shell_output

        # Prepend context for new conversations
        if self._config_manager.is_new_conversation:
            context_parts = []

            # Add imported files content
            if self._config_manager.imported_files:
                imported_content = prepare_imported_files_content(self._config_manager.imported_files)
                if imported_content:
                    context_parts.append(imported_content)

            # Add last executed cells if requested
            if self._config_manager.cells_to_load != 0:  # Load cells if not explicitly disabled (0)
                last_cells_content = self._history_manager.get_last_executed_cells(self._config_manager.cells_to_load)
                if last_cells_content:
                    context_parts.append(last_cells_content)

            if context_parts:
                enhanced_prompt_text = "\n\n".join(context_parts) + "\n\n" + enhanced_prompt_text

        # Build the prompt content - either as string or structured with images
        enhanced_prompt: str | list[dict[str, Any]]
        if captured_images:
            display_status(format_images_summary(captured_images), kind="info")

            # Build structured content with images
            content_blocks: list[dict[str, Any]] = []

            # Add images first
            for img in captured_images:
                content_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img["format"],
                            "data": img["data"],
                        },
                    }
                )

            # Add text content
            content_blocks.append({"type": "text", "text": enhanced_prompt_text})

            enhanced_prompt = content_blocks
        else:
            enhanced_prompt = enhanced_prompt_text

        mcp_servers: dict[str, McpServerConfig] = {"jupyter": self._sdk_server}
        additional_mcp_servers = self._config_manager.get_mcp_servers()
        if additional_mcp_servers:
            mcp_servers.update(additional_mcp_servers)

        options = ClaudeAgentOptions(
            allowed_tools=[
                "Bash",
                "LS",
                "Grep",
                "Read",
                "Edit",
                "MultiEdit",
                "Write",
                "WebSearch",
                "WebFetch",
                "mcp__jupyter__create_python_cell",
            ],
            model=self._config_manager.model,
            mcp_servers=mcp_servers,
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": get_system_prompt(
                    is_ipython=not is_in_jupyter_notebook(),
                    max_cells=self._config_manager.max_cells,
                ),
            },
            setting_sources=["user", "project", "local"],
            add_dirs=[Path(d) for d in self._config_manager.added_directories]
            if self._config_manager.added_directories
            else [],
        )

        # If we have an existing session ID from the client manager, use it to resume the conversation
        if self._client_manager is not None and self._client_manager.session_id:
            options.resume = self._client_manager.session_id

        # Create display in the main thread so display_id is associated
        # with the current cell output. The background thread only mutates
        # state; stop() renders the final result from the main thread.
        from .display import StreamingDisplay

        display = StreamingDisplay(verbose=verbose, replace_mode=self._config_manager.should_replace_cell)
        display.start()

        # Run the query with streaming
        # Simple approach: always use a thread to avoid anyio.run() nesting issues
        exception_queue: queue.Queue[Exception] = queue.Queue()

        def run_in_thread() -> None:
            try:
                # This always works because the thread has its own context
                # anyio.run() takes a no-arg async callable, so wrap with a lambda
                anyio.run(
                    lambda: self._run_streaming_query(enhanced_prompt, options, verbose, display),
                )
            except Exception as e:
                exception_queue.put(e)

        # Run in a separate thread to avoid event loop nesting issues
        thread = threading.Thread(target=run_in_thread)
        thread.start()

        # Set up interrupt handler that sends interrupt signal to Claude
        original_handler = None

        def interrupt_handler(signum: int, frame: "FrameType | None") -> None:
            # Send interrupt signal to Claude client if one exists
            if self._client_manager is not None:
                # Handle interrupt in a separate thread to avoid nesting anyio.run()
                # and to keep the signal handler signal-safe (no IPython display / ZMQ).
                def handle_interrupt() -> None:
                    display_status("Interrupting Claude Code", kind="warning")
                    if self._client_manager is not None:
                        with contextlib.suppress(Exception):
                            anyio.run(self._client_manager.handle_interrupt)  # no-args async callable

                interrupt_thread = threading.Thread(target=handle_interrupt)
                interrupt_thread.start()

        # Install our handler temporarily
        try:
            original_handler = signal.signal(signal.SIGINT, interrupt_handler)
            thread.join()
        finally:
            if original_handler is not None:
                signal.signal(signal.SIGINT, original_handler)
            display.stop()

        # Check for exceptions
        if not exception_queue.empty():
            raise exception_queue.get()

        adjust_cell_queue_markers(self)

        # After the async operation completes, set any pending input for terminal
        if self.shell is not None and "_claude_pending_input" in self.shell.user_ns:
            pending_input = self.shell.user_ns["_claude_pending_input"]
            del self.shell.user_ns["_claude_pending_input"]
            with contextlib.suppress(Exception):
                # Now we can safely set the next input
                self.shell.set_next_input(
                    pending_input,
                    replace=self._config_manager.should_replace_cell,
                )

        self._config_manager.is_new_conversation = False
        self._config_manager.replace_current_cell = False

    async def _run_streaming_query(
        self,
        prompt: str | list[dict[str, Any]],
        options: ClaudeAgentOptions,
        verbose: bool,
        display: Any = None,
    ) -> None:
        """Run Claude query with real-time message streaming."""
        if self._client_manager is None:
            self._client_manager = ClaudeClientManager()

        self._config_manager.is_current_execution_verbose = verbose
        await self._client_manager.query_sync(
            prompt, options, self._config_manager.is_new_conversation, verbose, display=display
        )
        self._config_manager.is_current_execution_verbose = False
        self._history_manager.update_last_output_line()

    def _claude_continue_impl(self, request_id: str, additional_prompt: str = "", verbose: bool = False) -> str:
        cell_queue: list[dict[str, Any]] = self.shell.user_ns.get("_claude_cell_queue", []) if self.shell else []

        if verbose:
            executed_count = sum(1 for cell in cell_queue if cell.get("executed", False))
            total_count = len(cell_queue)

            display_status(
                f"📊 Cell execution summary: {executed_count} of {total_count} cells executed",
                kind="info",
            )

        execution_results = []
        for i, cell in enumerate(cell_queue):
            tool_use_id = cell.get("tool_use_id", "")
            code = cell.get("original_code", cell.get("code", ""))
            executed = cell.get("executed", False)
            had_exception = cell.get("had_exception", False)
            error_info = cell.get("error", None)
            output = None

            # If executed, try to get output from history
            if executed:
                try:
                    if self.shell is not None:
                        history = self._history_manager.get_history_range(
                            start=-10,  # Look at last 10 entries to find all cells
                            stop=None,
                        )

                        for _session_id, _line_num, (
                            input_code,
                            output_result,
                        ) in history:
                            if input_code.strip() == code.strip():
                                if output_result is not None:
                                    output = str(output_result)
                                break
                except Exception:
                    logger.debug("Failed to retrieve output for cell %d", i, exc_info=True)

            # Build result summary
            if tool_use_id:
                result_entry = f"Tool use {tool_use_id}: "
            else:
                result_entry = f"Code cell {i + 1}: "

            if executed:
                if had_exception:
                    if error_info:
                        result_entry += f"Executed but encountered {error_info['type']}: {error_info['message']}"
                    else:
                        result_entry += "Executed but encountered an error"
                elif output:
                    result_entry += f"Executed successfully with output:\n{output}"
                else:
                    result_entry += "Executed successfully (no output)"
            else:
                result_entry += "Not executed by user"

            execution_results.append(result_entry)

        continue_prompt = "Previous code execution results for requested code cells:\n" + "\n\n".join(execution_results)

        additional_prompt = additional_prompt.strip()
        if not additional_prompt:
            additional_prompt = "Please continue with the task."

        display_status("✅ Continuing Claude session with execution results...", kind="success")

        # Clean up namespace
        if self.shell is not None:
            if "_claude_request_id" in self.shell.user_ns:
                del self.shell.user_ns["_claude_request_id"]

            # Clear the cell queue
            if "_claude_cell_queue" in self.shell.user_ns:
                del self.shell.user_ns["_claude_cell_queue"]

        # Clean up the pending request
        if request_id in self.pending_requests:
            del self.pending_requests[request_id]

        # Run a new Claude query with the execution results
        self._execute_prompt(additional_prompt, continue_prompt)

        return request_id  # Return for cell deletion logic

    def _handle_cc_options(self, args: Any) -> bool:
        """
        Handle all command-line options for the cc magic command.
        Returns True if any option was handled (meaning the command should return early).
        """
        return self._config_manager.handle_cc_options(args, self.cell_watcher)

    @line_cell_magic
    @magic_arguments()
    @argument("--verbose", "-v", action="store_true", help="Show verbose output")
    @argument(
        "--allow-run-all",
        "-a",
        action="store_true",
        help="Allow this cell to call Claude when Run All is used",
    )
    @argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Persistent setting for this session to automatically clean up prompt cells with Claude's code cells",
    )
    @argument(
        "--max-cells",
        type=int,
        default=None,
        help="Maximum number of code cells Claude can create (default: 3)",
    )
    @argument(
        "--help",
        "-h",
        action="store_true",
        help="Show available options and usage information",
    )
    @argument(
        "--import",
        type=str,
        dest="import_file",
        help="Add a file to be included in initial conversation messages",
    )
    @argument(
        "--add-dir",
        type=str,
        dest="add_dir",
        help="Add a directory to Claude's accessible directories",
    )
    @argument(
        "--mcp-config",
        type=str,
        dest="mcp_config",
        help="Path to a .mcp.json file containing MCP server configurations",
    )
    @argument(
        "--cells-to-load",
        type=int,
        dest="cells_to_load",
        help="Maximum number of recent cells to load when starting a new conversation (default: 0)",
    )
    @argument(
        "--model",
        type=str,
        dest="model",
        help="Model to use for Claude Code (default: sonnet)",
    )
    def cc(self, line: str, cell: str | None = None) -> None:
        """
        Run Claude Code with full agentic loop.

        This combines the functionality of claude_local and claude_continue:
        - If there's a pending code execution, it continues with the results
        - Otherwise, it starts/continues a conversation with Claude

        Usage as line magic:
            %cc
            %cc additional instructions here
            %cc --verbose
            %cc --help

        Usage as cell magic:
            %%cc
            Create a pandas DataFrame and plot it

            %%cc --verbose
            Show detailed tool call information

        Note: If your prompt ends with '?', use cell magic (%%cc) instead of line magic (%cc)
        to avoid IPython's help system interference.
        """
        if cell is not None:
            line = line + "\n" + cell

        # Parse arguments and prompt
        args, prompt = self._parse_args_and_prompt(line, self.cc)

        # Handle all command-line options (returns True if we should exit early)
        if self._handle_cc_options(args):
            return

        # If no prompt provided after handling options, return
        if not prompt:
            return

        # Check if there's a pending request
        request_id = self.shell.user_ns.get("_claude_request_id") if self.shell is not None else None

        if request_id:
            # There's a pending code execution - continue with it
            self._claude_continue_impl(request_id, prompt, args.verbose)
        else:
            # Clear any remaining cell queue when starting a new prompt
            if self.shell is not None and "_claude_cell_queue" in self.shell.user_ns:
                cell_queue = self.shell.user_ns["_claude_cell_queue"]
                if cell_queue:
                    unexecuted = sum(1 for cell in cell_queue if not cell.get("executed", False))
                    if unexecuted > 0:
                        display_status(
                            f"⚠️ Clearing {unexecuted} unexecuted cells from previous request",
                            kind="warning",
                        )
                del self.shell.user_ns["_claude_cell_queue"]

            self._execute_prompt(prompt, verbose=args.verbose)

    @line_cell_magic
    def ccn(self, line: str, cell: str | None = None) -> None:
        """An alias for %cc_new"""
        self.cc_new(line, cell)

    @line_cell_magic
    def ccc(self, line: str, cell: str | None = None) -> None:
        """An alias for %cc_cur"""
        self.cc_cur(line, cell)

    @line_cell_magic
    def cc_cur(self, line: str, cell: str | None = None) -> None:
        """
        Run Claude Code and replace the current cell with generated code.

        Like %cc, but replaces the prompt cell instead of inserting a new cell below.
        Useful when you want Claude to overwrite the current cell in-place.

        Usage as line magic:
            %cc_cur refactor this code
            %cc_cur --verbose rewrite using list comprehension

        Usage as cell magic:
            %%cc_cur
            Replace this cell with a function that computes the mean
        """
        self._config_manager.replace_current_cell = True
        self.cc(line, cell)

    @line_cell_magic
    def cc_new(self, line: str, cell: str | None = None) -> None:
        """
        Run Claude Code with a fresh conversation (no history).

        Usage as line magic:
            %cc_new
            %cc_new Analyze this data from scratch
            %cc_new --verbose Start a new analysis

        Usage as cell magic:
            %%cc_new
            Analyze this data from scratch

            %%cc_new --verbose
            Start a new analysis with detailed output
        """
        if cell is not None:
            line = line + "\n" + cell

        # Parse arguments and prompt
        args, prompt = self._parse_args_and_prompt(line, self.cc)

        # Handle all command-line options (returns True if we should exit early)
        if self._handle_cc_options(args):
            return

        if not prompt:
            raise ValueError("A prompt must be provided to start the conversation.")

        # Reset shell output tracking to 0 so new conversation sees all shell history
        self._history_manager.reset_output_tracking()

        # Reset variable tracking for new conversation
        self._variable_tracker.reset()

        # Reset configuration for new conversation
        self._config_manager.reset_for_new_conversation()

        # Clear any remaining cell queue since we're starting fresh
        if self.shell is not None and "_claude_cell_queue" in self.shell.user_ns:
            del self.shell.user_ns["_claude_cell_queue"]

        # Reset session in the client manager to force new conversation
        if self._client_manager is not None:
            self._client_manager.reset_session()
        else:
            # Create a new client manager if needed
            self._client_manager = ClaudeClientManager()

        # Now run as normal
        self._config_manager.is_new_conversation = True
        self._execute_prompt(prompt, verbose=args.verbose)

    def _parse_args_and_prompt(self, line: str, magic_func: Any) -> tuple[Any, str]:
        """Parse arguments and prompt from magic command line.

        Returns tuple of (args, prompt).
        """
        parts = line.split(None, 1) if line else []  # Split into at most 2 parts

        if not parts:
            # Empty line
            return parse_argstring(magic_func, ""), ""

        # Check if first part looks like an argument
        if parts[0].startswith("-"):
            # First part is an argument
            first_part = parts[0]
            remaining = parts[1] if len(parts) > 1 else ""

            # Get all value-taking args
            value_taking_args = []
            for action in magic_func.parser._actions:
                # Skip help action and positional arguments
                if action.option_strings and action.nargs != 0:
                    # Store both short and long forms
                    value_taking_args.extend(action.option_strings)

            # Special handling for arguments that take values
            if first_part in value_taking_args:
                # These arguments need a value, check if we have more parts
                value_and_prompt = remaining.split(None, 1) if remaining else []
                if value_and_prompt:
                    # We have a value
                    args_str = f"{first_part} {value_and_prompt[0]}"
                    prompt = value_and_prompt[1] if len(value_and_prompt) > 1 else ""
                else:
                    # No value provided, let argparse handle the error
                    args_str = first_part
                    prompt = ""
            else:
                # Boolean flag or other argument type
                args_str = first_part
                prompt = remaining

            return parse_argstring(magic_func, args_str), prompt
        else:
            # First part is not an argument, entire line is the prompt
            return parse_argstring(magic_func, ""), line
