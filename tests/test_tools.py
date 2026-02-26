"""Unit tests for jupyter_cc.tools -- kernel state tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jupyter_cc.tools import inspect_variable_impl, list_variables_impl


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
