"""
Utility Modules
--------------
Utility functions for the application.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple


def run_command(command: list, cwd: Path = None) -> Tuple[str, str, int, str]:
    """Run a command and return stdout, stderr, and exit code."""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout, result.stderr, result.returncode, " ".join(command)
    except Exception as e:
        return "", str(e), -1, " ".join(command)
