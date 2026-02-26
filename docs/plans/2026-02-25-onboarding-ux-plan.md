# Onboarding UX Improvement — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the verbose 60-line onboarding output with a CLI availability check, a short permissions warning, and a concise welcome message.

**Architecture:** Add `shutil.which("claude")` check at the top of `load_ipython_extension()` with early return on failure. Replace `HELP_TEXT` display at load time with a new `WELCOME_TEXT` constant. `HELP_TEXT` stays for `%cc --help`.

**Tech Stack:** Python stdlib (`shutil.which`), IPython display

______________________________________________________________________

### Task 1: Add WELCOME_TEXT constant

**Files:**

- Modify: `src/jupyter_cc/constants.py:6-56`

**Step 1: Add WELCOME_TEXT before HELP_TEXT**

Add a new constant at line 6 (before `HELP_TEXT`):

```python
WELCOME_TEXT = """\
jupyter_cc ready!

  %cc <prompt>       Send a prompt to Claude
  %%cc <prompt>      Multi-line prompt
  %cc_new            Start a new conversation
  %cc_cur            Replace prompt cell with Claude's code

  %cc --help         Show all options"""
```

**Step 2: Verify the file is valid Python**

Run: `uv run python -c "from jupyter_cc.constants import WELCOME_TEXT, HELP_TEXT; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/jupyter_cc/constants.py
git commit -m "feat: add concise WELCOME_TEXT constant for onboarding"
```

______________________________________________________________________

### Task 2: Add CLI check and rewrite load_ipython_extension

**Files:**

- Modify: `src/jupyter_cc/__init__.py:1-70`

**Step 1: Update imports (line 9-14)**

Replace:

```python
import json
from pathlib import Path

from .constants import HELP_TEXT
from .display import display_status
from .magics import ClaudeCodeMagics
from .watcher import CellWatcher
```

With:

```python
import json
import shutil
from pathlib import Path

from .constants import WELCOME_TEXT
from .display import display_status
from .magics import ClaudeCodeMagics
from .watcher import CellWatcher
```

**Step 2: Rewrite load_ipython_extension (lines 41-69)**

Replace the entire function body with:

```python
def load_ipython_extension(ipython: object) -> None:
    """Load the jupyter_cc extension."""
    from IPython.core.interactiveshell import InteractiveShell

    if not isinstance(ipython, InteractiveShell):
        return

    # Check that Claude Code CLI is installed
    if not shutil.which("claude"):
        display_status(
            "ERROR: Claude Code CLI not found\n"
            "\n"
            "Install it: https://docs.anthropic.com/en/docs/claude-code\n"
            "Then reload: %load_ext jupyter_cc",
            kind="error",
        )
        return

    _ensure_claude_settings()

    # Short permissions warning
    display_status(
        "Claude has Bash, Read, Write, Edit, and web access. Use in trusted environments only.\n"
        "Permissions: .claude/settings.local.json",
        kind="warning",
    )

    cell_watcher = CellWatcher(ipython)
    magics = ClaudeCodeMagics(ipython, cell_watcher)
    ipython.register_magics(magics)
    ipython.events.register("pre_run_cell", cell_watcher.pre_run_cell)
    ipython.events.register("post_run_cell", cell_watcher.post_run_cell)

    display_status(WELCOME_TEXT, kind="info")
```

Note: `_ensure_claude_settings()` return value is no longer used — the settings creation message from `display_status` inside that function is sufficient.

**Step 3: Verify import and syntax**

Run: `uv run python -c "from jupyter_cc import load_ipython_extension; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/jupyter_cc/__init__.py
git commit -m "feat: add CLI check, shorten onboarding messages"
```

______________________________________________________________________

### Task 3: Update tests

**Files:**

- Modify: `tests/test_magic.py:26-38`

**Step 1: Update test_extension_loads assertion**

Change line 38 from:

```python
    assert "jupyter_cc loaded" in output
```

To:

```python
    assert "jupyter_cc ready" in output
```

**Step 2: Run the tests**

Run: `uv run pytest tests/test_magic.py -v`
Expected: All tests pass. `test_extension_loads` passes with the new assertion. `test_help` still passes because `HELP_TEXT` is still shown via `%cc --help`.

**Step 3: Commit**

```bash
git add tests/test_magic.py
git commit -m "test: update onboarding assertion to match new welcome text"
```

______________________________________________________________________

### Task 4: Lint, type-check, final verification

**Step 1: Lint and format**

Run: `uv run ruff check src/ --fix && uv run ruff format src/`
Expected: No errors (unused `HELP_TEXT` import was already removed in Task 2)

**Step 2: Type-check**

Run: `uv run pyright src/`
Expected: No new errors

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests pass

**Step 4: Commit any lint fixes**

```bash
git add -A
git commit -m "chore: lint and format"
```
