"""Example test to verify package imports correctly."""

from jupyter_cc import __version__


def test_version():
    assert __version__ == "1.0.0"
