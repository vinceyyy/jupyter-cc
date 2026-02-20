"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_fixture():
    """Example fixture that can be used across tests."""
    return {"key": "value"}
