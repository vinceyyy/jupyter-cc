# Kernel State Tools & Automatic Image Capture — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Give CC on-demand kernel variable inspection (two new MCP tools) and automatic image capture from all `display()` calls (replacing the explicit `capture_output()` pattern).

**Architecture:** Two new MCP tools (`list_variables`, `inspect_variable`) in a new `tools.py` module, registered on the existing SDK MCP server. A new `ImageCollector` class wraps `shell.display_pub.publish` to transparently intercept images. The old `capture_output()` code path is removed entirely.

**Tech Stack:** Python 3.13, claude-agent-sdk `@tool` decorator, IPython `DisplayPublisher`, pytest

______________________________________________________________________

### Task 1: Create `list_variables` tool

**Files:**

- Create: `src/jupyter_cc/tools.py`
- Test: `tests/test_tools.py`

**Step 1: Write failing tests**

```python
# tests/test_tools.py
"""Unit tests for jupyter_cc.tools — kernel state tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jupyter_cc.tools import list_variables_impl, inspect_variable_impl


@pytest.fixture
def mock_shell() -> MagicMock:
    shell = MagicMock()
    shell.user_ns = {}
    return shell


class TestListVariables:
    def test_empty_namespace(self, mock_shell: MagicMock) -> None:
        result = list_variables_impl(mock_shell)
        assert result == []

    def test_lists_user_variables(self, mock_shell: MagicMock) -> None:
        mock_shell.user_ns["x"] = 42
        mock_shell.user_ns["name"] = "hello"
        result = list_variables_impl(mock_shell)
        names = [v["name"] for v in result]
        assert "x" in names
        assert "name" in names

    def test_includes_type_and_repr(self, mock_shell: MagicMock) -> None:
        mock_shell.user_ns["x"] = 42
        result = list_variables_impl(mock_shell)
        var = next(v for v in result if v["name"] == "x")
        assert var["type"] == "int"
        assert var["repr"] == "42"

    def test_filters_underscore_and_builtins(self, mock_shell: MagicMock) -> None:
        mock_shell.user_ns["_private"] = 1
        mock_shell.user_ns["In"] = []
        mock_shell.user_ns["Out"] = {}
        mock_shell.user_ns["exit"] = None
        mock_shell.user_ns["quit"] = None
        mock_shell.user_ns["visible"] = "yes"
        result = list_variables_impl(mock_shell)
        names = [v["name"] for v in result]
        assert names == ["visible"]

    def test_truncates_long_repr(self, mock_shell: MagicMock) -> None:
        mock_shell.user_ns["big"] = "a" * 200
        result = list_variables_impl(mock_shell)
        var = next(v for v in result if v["name"] == "big")
        assert len(var["repr"]) <= 100
        assert var["repr"].endswith("...")
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools.py -v`
Expected: ImportError — `jupyter_cc.tools` does not exist yet

**Step 3: Implement `list_variables_impl` and the SDK tool wrapper**

```python
# src/jupyter_cc/tools.py
"""MCP tools for kernel state inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import anyio
from claude_agent_sdk import tool

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell

_FILTERED_NAMES = frozenset({"In", "Out", "exit", "quit"})
_REPR_MAX = 100

# Module-level reference set by magics.py on init
_shell: InteractiveShell | None = None


def _get_truncated_repr(value: Any, max_length: int = _REPR_MAX) -> str:
    try:
        r = repr(value)
        if len(r) > max_length:
            return r[: max_length - 3] + "..."
        return r
    except Exception:
        return f"<{type(value).__name__} object>"


def _filtered_user_vars(shell: InteractiveShell) -> dict[str, Any]:
    """Return user variables, filtering out internals."""
    return {
        k: v
        for k, v in shell.user_ns.items()
        if not k.startswith("_") and k not in _FILTERED_NAMES
    }


def list_variables_impl(shell: InteractiveShell) -> list[dict[str, str]]:
    """Return a list of all user variables with name, type, and truncated repr."""
    result = []
    for name in sorted(_filtered_user_vars(shell)):
        value = shell.user_ns[name]
        result.append({
            "name": name,
            "type": type(value).__name__,
            "repr": _get_truncated_repr(value),
        })
    return result


@tool(
    "list_variables",
    "List all user-defined variables in the IPython kernel with their types and values",
    {},
)
async def list_variables_tool(args: dict[str, Any]) -> dict[str, Any]:
    """MCP tool: list all user variables."""
    await anyio.lowlevel.checkpoint()
    if _shell is None:
        return {"content": [{"type": "text", "text": "Shell not available"}], "is_error": True}
    variables = list_variables_impl(_shell)
    if not variables:
        text = "No user-defined variables in the kernel."
    else:
        lines = [f"  {v['name']}: {v['type']} = {v['repr']}" for v in variables]
        text = f"{len(variables)} variable(s):\n" + "\n".join(lines)
    return {"content": [{"type": "text", "text": text}]}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools.py::TestListVariables -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/jupyter_cc/tools.py tests/test_tools.py
git commit -m "feat: add list_variables kernel tool"
```

______________________________________________________________________

### Task 2: Create `inspect_variable` tool

**Files:**

- Modify: `src/jupyter_cc/tools.py`
- Test: `tests/test_tools.py`

**Step 1: Write failing tests**

Append to `tests/test_tools.py`:

```python
class TestInspectVariable:
    def test_inspect_basic_variable(self, mock_shell: MagicMock) -> None:
        mock_shell.user_ns["x"] = 42
        result = inspect_variable_impl(mock_shell, "x")
        assert result["name"] == "x"
        assert result["type"] == "int"
        assert result["repr"] == "42"
        assert "attributes" in result

    def test_inspect_nonexistent_variable(self, mock_shell: MagicMock) -> None:
        with pytest.raises(KeyError, match="not found"):
            inspect_variable_impl(mock_shell, "missing")

    def test_inspect_full_repr_not_truncated(self, mock_shell: MagicMock) -> None:
        mock_shell.user_ns["big"] = "a" * 500
        result = inspect_variable_impl(mock_shell, "big")
        # Full repr up to 10000 chars, not truncated at 100
        assert len(result["repr"]) > 100

    def test_inspect_dict_shows_keys(self, mock_shell: MagicMock) -> None:
        mock_shell.user_ns["d"] = {"a": 1, "b": 2, "c": 3}
        result = inspect_variable_impl(mock_shell, "d")
        assert result["extras"]["length"] == 3
        assert result["extras"]["keys"] == ["a", "b", "c"]

    def test_inspect_list_shows_length(self, mock_shell: MagicMock) -> None:
        mock_shell.user_ns["lst"] = [1, 2, 3, 4, 5]
        result = inspect_variable_impl(mock_shell, "lst")
        assert result["extras"]["length"] == 5
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools.py::TestInspectVariable -v`
Expected: ImportError for `inspect_variable_impl`

**Step 3: Implement `inspect_variable_impl` and SDK tool wrapper**

Add to `src/jupyter_cc/tools.py`:

```python
_INSPECT_REPR_MAX = 10_000


def inspect_variable_impl(shell: InteractiveShell, name: str) -> dict[str, Any]:
    """Return detailed info about a single variable."""
    if name not in shell.user_ns:
        raise KeyError(f"Variable '{name}' not found in kernel namespace")

    value = shell.user_ns[name]
    type_name = type(value).__name__

    # Full repr (up to 10k chars)
    try:
        full_repr = repr(value)
        if len(full_repr) > _INSPECT_REPR_MAX:
            full_repr = full_repr[: _INSPECT_REPR_MAX - 3] + "..."
    except Exception:
        full_repr = f"<{type_name} object>"

    # Public attributes
    attributes = [a for a in dir(value) if not a.startswith("_")]

    # Type-specific extras
    extras: dict[str, Any] = {}
    if hasattr(value, "shape"):
        try:
            extras["shape"] = str(value.shape)
        except Exception:
            pass
    if hasattr(value, "columns"):
        try:
            extras["columns"] = list(value.columns)
        except Exception:
            pass
    if hasattr(value, "dtypes"):
        try:
            extras["dtypes"] = str(value.dtypes)
        except Exception:
            pass
    if isinstance(value, dict):
        extras["length"] = len(value)
        extras["keys"] = list(value.keys())[:50]  # Cap at 50 keys
    elif isinstance(value, (list, tuple, set, frozenset)):
        extras["length"] = len(value)

    return {
        "name": name,
        "type": type_name,
        "repr": full_repr,
        "attributes": attributes,
        "extras": extras,
    }


@tool(
    "inspect_variable",
    "Get detailed information about a specific variable in the IPython kernel",
    {"name": str},
)
async def inspect_variable_tool(args: dict[str, Any]) -> dict[str, Any]:
    """MCP tool: inspect a single variable in detail."""
    await anyio.lowlevel.checkpoint()
    if _shell is None:
        return {"content": [{"type": "text", "text": "Shell not available"}], "is_error": True}

    name = args.get("name", "")
    if not name:
        return {"content": [{"type": "text", "text": "Parameter 'name' is required"}], "is_error": True}

    try:
        info = inspect_variable_impl(_shell, name)
    except KeyError as e:
        return {"content": [{"type": "text", "text": str(e)}], "is_error": True}

    lines = [
        f"Name: {info['name']}",
        f"Type: {info['type']}",
        f"Value: {info['repr']}",
    ]
    if info["extras"]:
        lines.append("Details:")
        for k, v in info["extras"].items():
            lines.append(f"  {k}: {v}")
    lines.append(f"Attributes: {', '.join(info['attributes'][:30])}")
    if len(info["attributes"]) > 30:
        lines.append(f"  ... and {len(info['attributes']) - 30} more")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add src/jupyter_cc/tools.py tests/test_tools.py
git commit -m "feat: add inspect_variable kernel tool"
```

______________________________________________________________________

### Task 3: Register tools in MCP server and allowed_tools

**Files:**

- Modify: `src/jupyter_cc/magics.py`

**Step 1: Wire the new tools into the MCP server**

In `magics.py`, add imports and register the tools:

```python
# At top of magics.py, add import:
from .tools import _shell as _tools_shell_ref  # noqa: unused, side-effect
from .tools import inspect_variable_tool, list_variables_tool
import jupyter_cc.tools as _tools_module
```

In `ClaudeCodeMagics.__init__`, after `self._sdk_server = create_sdk_mcp_server(...)`:

```python
# Set the shell reference for the tools module
_tools_module._shell = shell

# Create SDK MCP server with all tools
self._sdk_server = create_sdk_mcp_server(
    name="jupyter_executor",
    version="1.0.0",
    tools=[execute_python_tool, list_variables_tool, inspect_variable_tool],
)
```

In `_execute_prompt`, add to `allowed_tools`:

```python
"mcp__jupyter__list_variables",
"mcp__jupyter__inspect_variable",
```

**Step 2: Run existing tests to verify nothing broke**

Run: `uv run pytest -v`
Expected: All existing tests PASS

**Step 3: Commit**

```bash
git add src/jupyter_cc/magics.py
git commit -m "feat: register kernel tools on MCP server"
```

______________________________________________________________________

### Task 4: Create `ImageCollector` class

**Files:**

- Rewrite: `src/jupyter_cc/capture.py`
- Create: `tests/test_capture.py`

**Step 1: Write failing tests**

```python
# tests/test_capture.py
"""Unit tests for jupyter_cc.capture — ImageCollector."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jupyter_cc.capture import ImageCollector


@pytest.fixture
def mock_shell() -> MagicMock:
    shell = MagicMock()
    shell.display_pub = MagicMock()
    shell.display_pub.publish = MagicMock()
    return shell


@pytest.fixture
def collector(mock_shell: MagicMock) -> ImageCollector:
    return ImageCollector(mock_shell)


class TestImageCollector:
    def test_install_wraps_publish(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        original = mock_shell.display_pub.publish
        collector.install()
        assert mock_shell.display_pub.publish is not original

    def test_uninstall_restores_publish(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        original = mock_shell.display_pub.publish
        collector.install()
        collector.uninstall()
        assert mock_shell.display_pub.publish is original

    def test_captures_png_image(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        collector.install()
        # Simulate a display() call with PNG data
        mock_shell.display_pub.publish(
            data={"image/png": "base64data==", "text/plain": "<Figure>"},
            metadata={},
        )
        images = collector.drain()
        assert len(images) == 1
        assert images[0]["format"] == "image/png"
        assert images[0]["data"] == "base64data=="

    def test_captures_multiple_formats(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        collector.install()
        mock_shell.display_pub.publish(
            data={"image/png": "png_data", "image/svg+xml": "<svg/>"},
            metadata={},
        )
        images = collector.drain()
        # One display call with two image formats = two captured images
        assert len(images) == 2

    def test_ignores_non_image_data(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        collector.install()
        mock_shell.display_pub.publish(
            data={"text/html": "<table>...</table>"},
            metadata={},
        )
        images = collector.drain()
        assert len(images) == 0

    def test_drain_clears_buffer(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        collector.install()
        mock_shell.display_pub.publish(
            data={"image/png": "data1"},
            metadata={},
        )
        collector.drain()
        assert collector.drain() == []

    def test_cap_at_20_images(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        collector.install()
        for i in range(25):
            mock_shell.display_pub.publish(
                data={"image/png": f"data_{i}"},
                metadata={},
            )
        images = collector.drain()
        assert len(images) == 20
        # Should keep the last 20 (most recent)
        assert images[0]["data"] == "data_5"
        assert images[-1]["data"] == "data_24"

    def test_passthrough_to_original(self, collector: ImageCollector, mock_shell: MagicMock) -> None:
        original = mock_shell.display_pub.publish
        collector.install()
        mock_shell.display_pub.publish(
            data={"image/png": "data"},
            metadata={"isolated": True},
        )
        # Original should have been called with same args
        original.assert_called_once_with(
            data={"image/png": "data"},
            metadata={"isolated": True},
        )
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_capture.py -v`
Expected: ImportError for `ImageCollector`

**Step 3: Rewrite `capture.py`**

```python
# src/jupyter_cc/capture.py
"""Automatic image capture from IPython display() calls."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell

logger = logging.getLogger(__name__)

_IMAGE_FORMATS = frozenset({"image/png", "image/jpeg", "image/jpg", "image/svg+xml"})
_MAX_IMAGES = 20


class ImageCollector:
    """Intercepts images from display() calls by wrapping DisplayPublisher.publish."""

    def __init__(self, shell: InteractiveShell) -> None:
        self._shell = shell
        self._images: list[dict[str, Any]] = []
        self._original_publish: Any = None

    def install(self) -> None:
        """Wrap shell.display_pub.publish to intercept images."""
        self._original_publish = self._shell.display_pub.publish

        @functools.wraps(self._original_publish)
        def _capturing_publish(data: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None, **kwargs: Any) -> Any:
            # Intercept image data
            if data:
                for fmt in _IMAGE_FORMATS:
                    if fmt in data:
                        self._images.append({
                            "format": fmt,
                            "data": data[fmt],
                            "metadata": metadata or {},
                        })
                        # Cap at max
                        if len(self._images) > _MAX_IMAGES:
                            self._images = self._images[-_MAX_IMAGES:]

            # Always pass through to original
            return self._original_publish(data=data, metadata=metadata, **kwargs)

        self._shell.display_pub.publish = _capturing_publish

    def uninstall(self) -> None:
        """Restore original publish method."""
        if self._original_publish is not None:
            self._shell.display_pub.publish = self._original_publish
            self._original_publish = None

    def drain(self) -> list[dict[str, Any]]:
        """Return all captured images and clear the buffer."""
        images = self._images
        self._images = []
        return images

    def format_summary(self, images: list[dict[str, Any]]) -> str:
        """Create a text summary of captured images."""
        if not images:
            return ""
        lines = [f"Captured {len(images)} image(s) from cell execution:"]
        for i, img in enumerate(images, 1):
            lines.append(f"  {i}. {img['format']}")
        return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_capture.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/jupyter_cc/capture.py tests/test_capture.py
git commit -m "feat: add ImageCollector for automatic display() image capture"
```

______________________________________________________________________

### Task 5: Install `ImageCollector` in extension lifecycle

**Files:**

- Modify: `src/jupyter_cc/__init__.py`
- Modify: `src/jupyter_cc/magics.py`

**Step 1: Install collector in `__init__.py`**

In `load_ipython_extension`, after creating `magics`:

```python
from .capture import ImageCollector

image_collector = ImageCollector(ipython)
image_collector.install()
magics = ClaudeCodeMagics(ipython, cell_watcher, image_collector)
```

Update `ClaudeCodeMagics.__init__` to accept `image_collector`:

```python
def __init__(self, shell, cell_watcher, image_collector):
    ...
    self._image_collector = image_collector
```

**Step 2: Replace `_claude_captured_output` detection in `_execute_prompt`**

In `magics.py:_execute_prompt`, replace the `_claude_captured_output` block (lines ~272-277) with:

```python
# Drain any images captured since last %cc call
captured_images = self._image_collector.drain()
```

Remove the old imports from capture.py (`extract_images_from_captured`, `format_images_summary`) and use the collector's `format_summary` instead.

**Step 3: Run all tests**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/jupyter_cc/__init__.py src/jupyter_cc/magics.py
git commit -m "feat: wire ImageCollector into extension lifecycle, remove capture_output pattern"
```

______________________________________________________________________

### Task 6: Update system prompt — remove `capture_output()` instructions

**Files:**

- Modify: `src/jupyter_cc/prompt.py`
- Modify: `tests/test_prompt.py`

**Step 1: Write failing test**

Add to `tests/test_prompt.py`:

```python
def test_system_prompt_no_capture_output_instructions() -> None:
    """System prompt should NOT contain capture_output instructions."""
    prompt = get_system_prompt(is_ipython=False, max_cells=3)
    assert "capture_output" not in prompt
    assert "_claude_captured_output" not in prompt


def test_system_prompt_mentions_automatic_image_capture() -> None:
    """System prompt should mention that images are captured automatically."""
    prompt = get_system_prompt(is_ipython=False, max_cells=3)
    assert "automatically" in prompt.lower() or "automatic" in prompt.lower()
```

**Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_prompt.py -v`
Expected: First test FAILS (capture_output still present), second FAILS (no mention of automatic)

**Step 3: Update `prompt.py`**

Replace the `system_prompt_image_capture` block with:

```python
system_prompt_image_capture = """Images from display() calls (matplotlib, seaborn, PIL, etc.) are automatically captured. You do not need any special wrapper — just write normal plotting code and the images will be available to you on the next turn.

You also have two kernel inspection tools available:
- list_variables: Lists all user-defined variables with types and values
- inspect_variable: Gets detailed info about a specific variable (full repr, attributes, shape/columns for DataFrames, etc.)

Use these tools when you need to understand the current kernel state without creating code cells."""
```

**Step 4: Run all prompt tests**

Run: `uv run pytest tests/test_prompt.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/jupyter_cc/prompt.py tests/test_prompt.py
git commit -m "feat: update system prompt for auto image capture and kernel tools"
```

______________________________________________________________________

### Task 7: Update all documentation

**Files:**

- Modify: `CLAUDE.md`
- Modify: `docs/what-cc-sees.md`

**Step 1: Update `CLAUDE.md`**

In the Project Structure section, add `tools.py` and update `capture.py` description:

```
├── tools.py         # MCP tools: list_variables, inspect_variable
├── capture.py       # Automatic image capture from display() calls
```

**Step 2: Update `docs/what-cc-sees.md`**

Major updates needed:

1. **Image Capture section**: Rewrite to describe automatic capture via `ImageCollector`. Remove the `capture_output()` pattern. Explain that images from any `display()` call are transparently collected and sent on the next `%cc` call. Mention the 20-image cap.

1. **Add new section: "Kernel State Tools"**: Document `list_variables` and `inspect_variable` — what they return, when CC uses them, how they differ from the variable diff in the prompt.

1. **Available Tools table**: Add two new rows for `mcp__jupyter__list_variables` and `mcp__jupyter__inspect_variable`.

1. **"What Claude Does NOT See" table**: Update the "Kernel metadata" row — CC still doesn't receive Python version, installed packages, or kernel name automatically, but can now query variable state on-demand via tools. Remove the "Cell execution errors" note about `_claude_continue_impl` if it references `_claude_captured_output`.

**Step 3: Commit**

```bash
git add CLAUDE.md docs/what-cc-sees.md
git commit -m "docs: update CLAUDE.md and what-cc-sees.md for kernel tools and auto images"
```

______________________________________________________________________

### Task 8: Lint, type-check, and final test pass

**Files:**

- Possibly modify: `pyproject.toml` (if `tools.py` needs lint relaxation)

**Step 1: Run linter**

Run: `uv run ruff check src/ --fix`
Expected: No errors (or fix any that appear)

**Step 2: Run formatter**

Run: `uv run ruff format src/`

**Step 3: Run type checker**

Run: `uv run pyright src/`
Expected: No errors (warnings OK)

**Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS

**Step 5: Commit any fixes**

```bash
git add -A
git commit -m "chore: lint, format, and type-check fixes"
```
