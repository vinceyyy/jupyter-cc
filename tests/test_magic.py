"""Automated smoke tests for jupyter_cc."""

import subprocess
import sys
import tempfile
from pathlib import Path


def run_ipython(code: str, timeout: int = 120) -> str:
    """Run code in IPython and return combined stdout+stderr."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        result = subprocess.run(
            [sys.executable, "-u", f.name],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**__import__("os").environ, "CLAUDECODE": ""},  # Clear nested session detection
        )
    # Remove temp file
    Path(f.name).unlink(missing_ok=True)
    return result.stdout + result.stderr


def test_extension_loads():
    """Extension loads without error."""
    output = run_ipython("""
import IPython
ip = IPython.get_ipython()
if ip is None:
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    ip = TerminalInteractiveShell.instance()
ip.run_line_magic("load_ext", "jupyter_cc")
print("PASS: Extension loaded")
""")
    assert "PASS: Extension loaded" in output
    assert "jupyter_cc ready" in output


def test_version():
    """Version is correct."""
    output = run_ipython("""
from jupyter_cc import __version__
print(f"Version: {__version__}")
assert __version__ == "0.3.0", f"Expected 0.3.0, got {__version__}"
print("PASS")
""")
    assert "PASS" in output


def test_help():
    """Help text displays correctly."""
    output = run_ipython("""
import IPython
ip = IPython.get_ipython()
if ip is None:
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    ip = TerminalInteractiveShell.instance()
ip.run_line_magic("load_ext", "jupyter_cc")
ip.run_line_magic("cc", "--help")
print("PASS")
""")
    assert "%cc" in output
    assert "%cc_new" in output
    assert "%cc_cur" in output
    assert "PASS" in output


def test_config_options():
    """Config options work."""
    output = run_ipython("""
import IPython
ip = IPython.get_ipython()
if ip is None:
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    ip = TerminalInteractiveShell.instance()
ip.run_line_magic("load_ext", "jupyter_cc")
ip.run_line_magic("cc", "--max-cells 5")
ip.run_line_magic("cc", "--model sonnet")
ip.run_line_magic("cc", "--cells-to-load 3")
print("PASS")
""")
    assert "max_cells" in output
    assert "model" in output
    assert "PASS" in output


def test_magics_registered():
    """All magic commands are registered."""
    output = run_ipython("""
import IPython
ip = IPython.get_ipython()
if ip is None:
    from IPython.terminal.interactiveshell import TerminalInteractiveShell
    ip = TerminalInteractiveShell.instance()
ip.run_line_magic("load_ext", "jupyter_cc")
magics = ip.magics_manager.magics
line_magics = list(magics.get("line", {}).keys())
for name in ["cc", "cc_new", "ccn", "cc_cur", "ccc"]:
    assert name in line_magics, f"Missing magic: {name}"
    print(f"  Found: %{name}")
print("PASS: All magics registered")
""")
    assert "PASS: All magics registered" in output
