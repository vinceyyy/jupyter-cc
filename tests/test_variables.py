"""Unit tests for jupyter_cc_magic.variables.VariableTracker."""

from __future__ import annotations

from unittest.mock import MagicMock

from jupyter_cc_magic.variables import VariableTracker


def test_no_variables(variable_tracker: VariableTracker) -> None:
    """Empty namespace returns 'no user-defined variables' message."""
    result = variable_tracker.get_variables_info()
    assert "no user-defined variables" in result.lower()


def test_new_variables_detected(variable_tracker: VariableTracker, mock_shell: MagicMock) -> None:
    """Adding vars to user_ns shows them as 'New variables'."""
    mock_shell.user_ns["x"] = 42
    mock_shell.user_ns["name"] = "hello"

    result = variable_tracker.get_variables_info()
    assert "New variables" in result
    assert "x" in result
    assert "name" in result
    assert "42" in result


def test_modified_variables_detected(variable_tracker: VariableTracker, mock_shell: MagicMock) -> None:
    """Changing a var shows it as 'Modified'."""
    mock_shell.user_ns["x"] = 10
    variable_tracker.get_variables_info()  # Capture initial state

    mock_shell.user_ns["x"] = 99
    result = variable_tracker.get_variables_info()
    assert "Modified" in result
    assert "x" in result
    assert "99" in result


def test_removed_variables_detected(variable_tracker: VariableTracker, mock_shell: MagicMock) -> None:
    """Removing a var shows it as 'Removed'."""
    mock_shell.user_ns["temp"] = "will be removed"
    variable_tracker.get_variables_info()  # Capture initial state

    del mock_shell.user_ns["temp"]
    result = variable_tracker.get_variables_info()
    assert "Removed" in result
    assert "temp" in result


def test_internal_variables_filtered(variable_tracker: VariableTracker, mock_shell: MagicMock) -> None:
    """Vars starting with _ and special names (In, Out, exit, quit) are excluded."""
    mock_shell.user_ns["_private"] = 1
    mock_shell.user_ns["__dunder"] = 2
    mock_shell.user_ns["In"] = []
    mock_shell.user_ns["Out"] = {}
    mock_shell.user_ns["exit"] = None
    mock_shell.user_ns["quit"] = None
    mock_shell.user_ns["visible"] = "yes"

    result = variable_tracker.get_variables_info()
    assert "_private" not in result
    assert "__dunder" not in result
    # "In" and "Out" should not appear as variable names in the output
    # (the word "In" may appear in other contexts like "IPython", so check for the variable format)
    assert "+ In:" not in result
    assert "+ Out:" not in result
    assert "+ exit:" not in result
    assert "+ quit:" not in result
    assert "visible" in result


def test_truncated_repr(variable_tracker: VariableTracker) -> None:
    """Long values get truncated to 100 chars."""
    long_value = "a" * 200
    result = variable_tracker.get_truncated_repr(long_value)
    assert len(result) <= 100
    assert result.endswith("...")


def test_reset_clears_state(variable_tracker: VariableTracker, mock_shell: MagicMock) -> None:
    """reset() clears tracking so previously seen variables appear as new again."""
    mock_shell.user_ns["x"] = 42
    variable_tracker.get_variables_info()  # Capture initial state

    variable_tracker.reset()

    # After reset, x should be reported as new again (not modified)
    result = variable_tracker.get_variables_info()
    assert "New variables" in result
    assert "x" in result
