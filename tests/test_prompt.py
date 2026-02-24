"""Unit tests for jupyter_cc.prompt."""

from __future__ import annotations

from pathlib import Path

from jupyter_cc.constants import EXECUTE_PYTHON_TOOL_NAME
from jupyter_cc.prompt import get_system_prompt, prepare_imported_files_content


def test_system_prompt_jupyter() -> None:
    """Jupyter prompt contains 'Jupyter notebook' and tool name."""
    prompt = get_system_prompt(is_ipython=False, max_cells=3)
    assert "Jupyter notebook" in prompt
    assert EXECUTE_PYTHON_TOOL_NAME in prompt


def test_system_prompt_ipython() -> None:
    """IPython prompt contains 'IPython session'."""
    prompt = get_system_prompt(is_ipython=True, max_cells=3)
    assert "IPython session" in prompt


def test_system_prompt_max_cells() -> None:
    """System prompt respects the max_cells parameter."""
    prompt = get_system_prompt(is_ipython=False, max_cells=7)
    assert "7" in prompt


def test_system_prompt_description_instruction() -> None:
    """System prompt contains instruction about providing a description."""
    prompt = get_system_prompt(is_ipython=False, max_cells=3)
    assert "description" in prompt.lower()


def test_prepare_imported_files_empty() -> None:
    """Empty list returns empty string."""
    result = prepare_imported_files_content([])
    assert result == ""


def test_prepare_imported_files_with_content(tmp_path: Path) -> None:
    """Reads file and includes content."""
    test_file = tmp_path / "data.csv"
    test_file.write_text("col1,col2\n1,2\n3,4")

    result = prepare_imported_files_content([str(test_file)])
    assert "data.csv" in result
    assert "col1,col2" in result
    assert "1,2" in result
