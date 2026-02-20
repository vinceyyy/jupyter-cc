# CLAUDE.md

## Commands

```bash
uv sync                        # Install all dependencies
uv run ruff check src/ --fix   # Lint (auto-fix)
uv run ruff format src/        # Format
uv run pyright src/            # Type check (warnings OK, errors not)
uv run pytest                  # Run tests
uv build                       # Build wheel/sdist
```

## Project Structure

```
src/jupyter_cc_magic/
├── __init__.py      # Extension entry point: load_ipython_extension(), permissions setup
├── magics.py        # Core: %cc, %cc_new, %cc_cur magic commands, MCP server setup
├── client.py        # SDK client lifecycle, streaming, interrupt handling, message display
├── config.py        # CLI options (--model, --add-dir, --import, --mcp-config, etc.)
├── prompt.py        # System prompt construction per environment (Jupyter vs IPython)
├── integration.py   # Cell creation, queue management, env detection
├── history.py       # IPython cell history tracking
├── variables.py     # Session variable change detection
├── capture.py       # Image extraction from cell outputs
├── watcher.py       # "Run All" detection via timing heuristics
├── constants.py     # Help text, tool names
└── py.typed         # PEP 561 marker
```

## Architecture

### Extension Lifecycle

1. `%load_ext jupyter_cc_magic` -> `__init__.py:load_ipython_extension()`
2. Creates `.claude/settings.local.json` with default permissions (Bash, Read, Write, etc.)
3. Registers `ClaudeCodeMagics` and `CellWatcher` hooks

### Query Flow

1. `%cc <prompt>` -> `magics.py:_execute_prompt()`
2. Builds enhanced prompt with variables, history, imported files, images
3. Creates `ClaudeAgentOptions` with MCP server, allowed tools, `add_dirs`
4. `setting_sources=["user", "project", "local"]` -- SDK auto-reads `~/.claude/` and `.claude/`
5. Runs in a thread to avoid event loop nesting: `anyio.run(query)`
6. `client.py`: `async with ClaudeSDKClient(options) as client:` -- one client per query
7. Streams responses, displays messages/tool calls, extracts session ID for continuity
8. Tool calls to `create_python_cell` -> creates notebook cells for user approval

### Key Patterns

- **Session continuity**: `session_id` stored on `ClaudeClientManager`, passed via `options.resume`
- **Interrupt handling**: SIGINT -> `client.interrupt()` via separate thread + anyio task group
- **Cell queue**: Tool calls create cells that get queued; `post_run_cell` hook processes them in order
- **Markdown display**: Messages with markdown patterns rendered via `IPython.display.Markdown`
- **SDK message parser patch**: Lenient parsing to skip unknown message types (e.g. `rate_limit_event`)

## Constraints

- Uses `anyio` for async (not trio). SDK v0.1.39+ uses anyio internally
- `ClaudeSDKClient` as context manager, fresh client per query
- Python >=3.13, line-length 120
- Some files have relaxed linting (E501, F841) due to upstream style -- see `pyproject.toml`
