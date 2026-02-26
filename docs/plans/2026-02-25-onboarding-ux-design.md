# Onboarding UX Improvement

## Problem

Loading the extension dumps ~60 lines of text: a 6-line security warning followed by a 50-line help message. Users need to know whether Claude Code is set up correctly and how to get started — not a reference manual.

## Design

### Approach: Sequential checks with early exit

1. **CLI validation** — `shutil.which("claude")`. If missing, show an error with install link and return without registering magics.
1. **Settings file** — Create `.claude/settings.local.json` if absent (unchanged).
1. **Permissions warning** — Shortened to 2 lines: what tools Claude has access to, and where to find the settings file.
1. **Welcome message** — ~8 lines: the 3 core commands (`%cc`, `%cc_new`, `%cc_cur`) plus `%cc --help` hint. Replaces the 50-line `HELP_TEXT` dump.

### Error state (CLI not found)

```
ERROR: Claude Code CLI not found

Install it: https://docs.anthropic.com/en/docs/claude-code
Then reload: %load_ext jupyter_cc
```

Single `display_status(..., kind="error")`, then early return (no magics registered).

### Permissions warning (shortened)

```
⚠ Claude has Bash, Read, Write, Edit, and web access. Use in trusted environments only.
  Permissions: .claude/settings.local.json
```

### Welcome message (replaces HELP_TEXT at load time)

```
jupyter_cc ready!

  %cc <prompt>       Send a prompt to Claude
  %%cc <prompt>      Multi-line prompt
  %cc_new            Start a new conversation
  %cc_cur            Replace prompt cell with Claude's code

  %cc --help         Show all options
```

The full help text (current `HELP_TEXT`) stays available via `%cc --help`.

## Files changed

- `src/jupyter_cc/__init__.py` — Add CLI check, shorten warning, replace HELP_TEXT with welcome message
- `src/jupyter_cc/constants.py` — Add `WELCOME_TEXT`, keep `HELP_TEXT` for `--help`
- `tests/test_magic.py` — Update assertion from "jupyter_cc loaded" to "jupyter_cc ready"
