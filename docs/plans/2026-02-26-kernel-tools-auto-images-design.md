# Kernel State Tools & Automatic Image Capture

Two features in a single PR that improve how CC observes the Jupyter kernel.

## Feature 1: Kernel State Tools

Two new MCP tools registered alongside `create_python_cell`.

### `list_variables`

Returns a full snapshot of all user variables (not a diff).

- No parameters
- Filters out `_`-prefixed names and IPython builtins (`In`, `Out`, `exit`, `quit`)
- Returns: name, type, truncated repr (100 chars) for each variable
- Reads from `shell.user_ns` directly — no code execution

### `inspect_variable`

Returns detailed info about a single variable.

- Parameter: `name` (str)
- Returns:
  - Type name
  - Full `repr()` (up to 10,000 chars)
  - `dir()` listing of public attributes
  - Type-specific extras: shape/columns/dtypes for DataFrames, length for lists/dicts, keys for dicts
- Errors clearly if the variable doesn't exist

### Rationale

The existing variable diff in the prompt is useful as automatic context, but CC has no way to dig deeper
mid-conversation. These tools let CC query on-demand without creating code cells.

## Feature 2: Automatic Image Capture

Replace the explicit `capture_output()` pattern with transparent interception of `display()` calls.

### Approach

Wrap `shell.display_pub.publish()` to intercept image MIME types while passing them through to the
notebook normally.

### `ImageCollector` class (in `capture.py`)

- Installed during extension load — wraps `shell.display_pub.publish`
- On each `publish()` call, checks for `image/png`, `image/jpeg`, `image/svg+xml` in the data dict
- Stores images in a buffer, tagged with the cell execution count
- Only images produced since the last `%cc` call are delivered
- Cap: 20 images max per window — oldest dropped if exceeded

### Delivery

- `_execute_prompt()` drains the collector instead of checking for `_claude_captured_output`
- Images become structured content blocks (same format as today's `capture_output()` path)
- Buffer cleared after drain

### Cleanup

- Remove `capture_output()` instructions from system prompt
- Remove `_claude_captured_output` detection from `_execute_prompt()`
- Remove `format_images_summary` and `extract_images_from_captured` (replaced by collector)

## Files Changed

| File                          | Change                                                                                      |
| ----------------------------- | ------------------------------------------------------------------------------------------- |
| `src/jupyter_cc/tools.py`     | **New** — `list_variables` and `inspect_variable` tool functions                            |
| `src/jupyter_cc/capture.py`   | Rewrite — `ImageCollector` class replaces old extraction helpers                            |
| `src/jupyter_cc/magics.py`    | Register new tools, replace `_claude_captured_output` with collector drain                  |
| `src/jupyter_cc/__init__.py`  | Install `ImageCollector` on display publisher during extension load                         |
| `src/jupyter_cc/prompt.py`    | Remove `capture_output()` instructions, note automatic capture                              |
| `src/jupyter_cc/constants.py` | Update tool name constant if needed                                                         |
| `CLAUDE.md`                   | Add `tools.py` to structure, update `capture.py` description                                |
| `docs/what-cc-sees.md`        | Rewrite Image Capture section, add Kernel State Tools section, update Available Tools table |
| `tests/`                      | Tests for new tools and image collector                                                     |

## Non-goals

- No arbitrary expression eval — CC uses `create_python_cell` for that
- No cross-notebook image sharing
- No video/animation capture
