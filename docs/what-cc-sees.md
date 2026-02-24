# What Claude Sees When You Run `%cc`

## Overview

When you run `%cc analyze my dataframe`, Claude does not just receive the string `"analyze my dataframe"`. jupyter-cc augments your prompt with the current kernel state -- variable changes, recently executed cells, captured images -- before sending it to the Claude Agent SDK. This document explains exactly what Claude receives and how each piece is constructed.

## The Enhanced Prompt

Every `%cc` call builds an enhanced prompt with this structure:

```
Your client's request is <request>{your prompt here}</request>

{variable_changes}
{previous_execution_results}
{recent_cell_history}
```

For new conversations (first `%cc` or after `%cc_new`), imported files and loaded cell history are prepended before the prompt.

Here is a concrete example of what Claude actually sees:

```
Files imported by the user for your reference. Use this content directly. Don't read them again:

config.yaml:
```

database:
host: localhost
port: 5432

```

Last executed cells from this session:

<cell-in-3>
import pandas as pd
df = pd.read_csv("sales.csv")
</cell-in-3>

<cell-in-4>
df.head()
</cell-in-4>
<cell-out-4>
   date       revenue
0  2024-01    15000
1  2024-02    18000
</cell-out-4>

Your client's request is <request>Plot monthly revenue trends</request>

Variable changes in IPython session:
New variables:
  + pd: module = <module 'pandas' from '/...'>
  + df: DataFrame =    date       revenue\n0  2024-01    15000\n1  2024-02    18000\n2  2024-...

No recent cell executions since last interaction.
```

Each section is explained below.

## Variable Tracking

**Source**: `src/jupyter_cc/variables.py` -- `VariableTracker.get_variables_info()`

Claude sees only what **changed** since the last `%cc` call, not a full dump of every variable. Changes are grouped into three categories:

```
Variable changes in IPython session:
New variables:
  + df: DataFrame =    date       revenue\n0  2024-01    15000\n1  2024-02    1800...
  + model: LinearRegression = LinearRegression()

Modified variables:
  ~ threshold: float = 0.85

Removed variables:
  - temp_data
```

**How it works:**

- After each `%cc` call, `VariableTracker` snapshots the `repr()` of every user variable.
- On the next `%cc` call, it compares the current state to the snapshot and reports the diff.
- Values are truncated to **100 characters** via `repr()`. Longer values get `...` appended.
- If nothing changed, Claude sees: `"No variable changes detected since last interaction."`

**Filtered out (never sent to Claude):**

| Filter                  | Examples                    |
| ----------------------- | --------------------------- |
| Names starting with `_` | `_temp`, `__name__`, `_i3`  |
| IPython internals       | `In`, `Out`, `exit`, `quit` |

**Edge cases:**

- Objects whose `repr()` raises an exception show as `<ClassName object>`.
- `%cc_new` resets the tracker, so the next call treats all variables as "new".

## Cell History

**Source**: `src/jupyter_cc/history.py` -- `HistoryManager`

Claude sees cells executed since the last `%cc` call, formatted as XML:

```
Recent IPython cell executions (Note: Only return values are captured, print statements are not shown):
<cell-in-5>
df.describe()
</cell-in-5>
<cell-out-5>
         revenue
count   12.000000
mean  16500.000000
std    2100.000000
</cell-out-5>

<cell-in-6>
df["revenue"].sum()
</cell-in-6>
<cell-out-6>
198000
</cell-out-6>
```

**Key details:**

- Input code is wrapped in `<cell-in-N>` tags (N is the IPython execution number).
- Return values (the `Out[N]` values) go in `<cell-out-N>` tags.
- **Print statements are NOT captured.** Only the cell's return value appears. If a cell ends with `print(x)`, the output tag is absent. Use `x` as the last line instead.
- Magic commands (`get_ipython().run_cell_magic(...)`) are filtered out to avoid self-referential noise.
- The history manager tracks `last_output_line` to know where to start capturing after each interaction.

**For new conversations (`%cc_new`):**

- By default, `%cc_new` loads **no** previous cells (`cells_to_load = 0`).
- The first `%cc` in a session loads **all** available cells (`cells_to_load = -1`).
- You can override this with `--cells-to-load <N>`:
  - `--cells-to-load 5` -- load the last 5 cells
  - `--cells-to-load -1` -- load all available history
  - `--cells-to-load 0` -- load nothing

## Image Capture

**Source**: `src/jupyter_cc/capture.py` -- `extract_images_from_captured()`

When code uses IPython's `capture_output()` pattern with the special variable `_claude_captured_output`, images are extracted and sent to Claude as base64-encoded content blocks alongside the text prompt.

**How it works:**

1. Claude generates code that wraps plotting in `capture_output()`:
   ```python
   from IPython.utils.capture import capture_output

   with capture_output() as _claude_captured_output:
       plt.plot([1, 2, 3])
       plt.show()

   for output in _claude_captured_output.outputs:
       display(output)
   ```
1. When the user runs this cell and then calls `%cc`, jupyter-cc detects `_claude_captured_output` in the namespace.
1. `extract_images_from_captured()` extracts image data from each output object.
1. Images are sent as structured content blocks (not embedded in text):
   ```python
   [
       {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}},
       {"type": "text", "text": "Your client's request is <request>...</request>\n..."},
   ]
   ```
1. The `_claude_captured_output` variable is deleted from the namespace after extraction.

**Supported formats:** `image/png`, `image/jpeg`, `image/jpg`, `image/svg+xml`

Claude can see and reference these images in its response -- for example, suggesting changes to a plot's styling based on what it actually looks like.

## System Prompt

**Source**: `src/jupyter_cc/prompt.py` -- `get_system_prompt()`

The system prompt is appended to the Claude Code preset (`"claude_code"`) and varies based on the environment:

**In a Jupyter notebook:**

- Tells Claude it is "operating in a Jupyter notebook"
- Explains that `create_python_cell` creates new cells in the notebook interface
- Limits cell creation to `max_cells` per turn (default: 3)
- Instructs Claude to prefer creating only ONE cell with a short snippet
- Explains the cell queue behavior: cells are presented one by one, errors break the chain
- Tells Claude it cannot edit cells other than the current one -- suggests the `%%cc edit` pattern instead

**In an IPython terminal:**

- Tells Claude it is in a "shared IPython session"
- Limits to exactly ONE `create_python_cell` call (terminal does not support multiple pending blocks)

**Common instructions (both environments):**

- Use `capture_output() as _claude_captured_output` for any image-generating code (matplotlib, seaborn, PIL, etc.)
- Prefer text answers -- do not use `create_python_cell` when a direct answer suffices
- Try built-in tools (Read, Grep, etc.) before reaching for `create_python_cell`
- Always include a return value as the last line (not `print()`)
- Do not invoke `create_python_cell` in parallel
- Always provide a short `description` parameter (appears as a comment atop the cell)
- If the request is empty, continue from where Claude left off

## Available Tools

Claude has access to these tools during a `%cc` session:

| Tool                               | Purpose                             |
| ---------------------------------- | ----------------------------------- |
| `Bash`                             | Run shell commands                  |
| `LS`                               | List directory contents             |
| `Grep`                             | Search file contents                |
| `Read`                             | Read file contents                  |
| `Edit`                             | Edit a file (single replacement)    |
| `MultiEdit`                        | Edit a file (multiple replacements) |
| `Write`                            | Write or create files               |
| `WebSearch`                        | Search the web                      |
| `WebFetch`                         | Fetch and read web pages            |
| `mcp__jupyter__create_python_cell` | Create a code cell in the notebook  |

The `create_python_cell` tool accepts two parameters:

- `code` (str) -- the Python code for the cell
- `description` (str) -- a short comment that appears at the top of the cell

**Additional MCP servers**: If you configure `--mcp-config <path>`, servers from that `.mcp.json` file are merged into the available tools. The config file format follows the standard `{"mcpServers": {...}}` schema.

## Session State

**Source**: `src/jupyter_cc/client.py` -- `ClaudeClientManager`

jupyter-cc maintains conversation continuity across `%cc` calls:

- **Session ID**: After each query, the SDK returns a `session_id` in the `ResultMessage`. `ClaudeClientManager` stores this.
- **Resumption**: On the next `%cc` call, the stored session ID is passed via `options.resume`, and `options.continue_conversation = True`. Claude picks up where it left off.
- **Fresh start**: `%cc_new` calls `client_manager.reset_session()`, clearing the session ID. The next query starts a new conversation with no history from previous turns.

This means Claude remembers the full conversation within a session. You can say `%cc now filter to only Q4 data` and Claude knows what "the data" refers to from prior context.

## SDK Configuration

**Source**: `src/jupyter_cc/magics.py` -- `ClaudeAgentOptions` construction

The `ClaudeAgentOptions` object configures the Claude Agent SDK:

```python
options = ClaudeAgentOptions(
    allowed_tools=[...],          # See "Available Tools" above
    model="sonnet",               # Configurable via --model
    mcp_servers={...},            # jupyter executor + any --mcp-config servers
    system_prompt={
        "type": "preset",
        "preset": "claude_code",  # Full Claude Code system prompt
        "append": "...",          # jupyter-cc-specific additions (see "System Prompt")
    },
    setting_sources=["user", "project", "local"],
    add_dirs=[...],               # From --add-dir options
)
```

**`setting_sources`** tells the SDK to automatically read:

- `"user"` -- `~/.claude/CLAUDE.md` and `~/.claude/skills/`
- `"project"` -- `.claude/CLAUDE.md` and `.claude/skills/` in the working directory
- `"local"` -- `.claude/settings.local.json` for tool permissions

This means your project's `CLAUDE.md` instructions and skills are automatically available to Claude without any extra configuration.

**`add_dirs`** gives Claude file access to directories beyond the working directory. Set via `%cc --add-dir /path/to/data`.

## Imported Files

**Source**: `src/jupyter_cc/prompt.py` -- `prepare_imported_files_content()`

The `--import` option includes file contents in the first message of a new conversation:

```python
%cc --import config.yaml
%cc --import schema.sql
%cc_new Analyze the database schema and suggest optimizations
```

**What Claude sees:**

```
Files imported by the user for your reference. Use this content directly. Don't read them again:

config.yaml:
```

database:
host: localhost
port: 5432

```

schema.sql:
```

CREATE TABLE orders (
id SERIAL PRIMARY KEY,
...
);

```
```

**Key details:**

- Files are read at query time (not import time), so edits between `--import` and the query are picked up.
- Only included in **new conversations** (first `%cc` or after `%cc_new`). Subsequent `%cc` calls in the same session do not re-send them since Claude already has the context.
- The instruction "Don't read them again" prevents Claude from wasting a `Read` tool call on a file it already has.
- Files that do not exist or cannot be read are silently skipped.

## What Claude Does NOT See

Understanding what is excluded helps avoid confusion:

| Excluded                                        | Why                                                                                                                                                                                 |
| ----------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `print()` output                                | IPython only captures return values in `Out[N]`, not stdout. Use `df.head()` instead of `print(df.head())`.                                                                         |
| Variables starting with `_`                     | Filtered out to avoid noise from IPython temporaries (`_`, `_i`, `__`, `_oh`, etc.).                                                                                                |
| IPython built-ins (`In`, `Out`, `exit`, `quit`) | Always filtered from variable tracking.                                                                                                                                             |
| Cell execution errors                           | Errors do not produce `Out[N]` values. Claude sees the cell input but not the traceback. (Exception: the `_claude_continue_impl` flow does report errors for cells Claude created.) |
| Kernel metadata                                 | Python version, installed packages, kernel name -- none of this is sent. Claude can discover it via tool calls if needed.                                                           |
| Other notebooks or sessions                     | Each notebook has its own `ClaudeClientManager` and `VariableTracker`. There is no cross-notebook state.                                                                            |
| Cells from before `%load_ext`                   | History tracking starts when the extension loads. Earlier cells are only available if `--cells-to-load` pulls them from IPython's history database.                                 |
