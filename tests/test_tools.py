"""Unit tests for jupyter_cc.tools -- kernel state tools."""

from __future__ import annotations

from unittest.mock import MagicMock

from jupyter_cc.tools import list_variables_impl


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
