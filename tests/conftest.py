"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jupyter_cc_magic.variables import VariableTracker


@pytest.fixture
def mock_shell() -> MagicMock:
    """A mock IPython InteractiveShell with user_ns, set_next_input, and events."""
    shell = MagicMock()
    shell.user_ns = {}
    shell.set_next_input = MagicMock()
    shell.events = MagicMock()
    return shell


@pytest.fixture
def variable_tracker(mock_shell: MagicMock) -> VariableTracker:
    """A VariableTracker initialized with the mock shell."""
    return VariableTracker(mock_shell)
