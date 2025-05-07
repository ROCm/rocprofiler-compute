"""
Specialized Widget Modules
-------------------------
Contains custom widget implementations for the application.
"""

import sys
import traceback
from io import StringIO
from pathlib import Path

import pandas as pd
from textual.events import MouseDown, MouseMove, MouseUp
from textual.widgets import DirectoryTree, Static


class FolderOnlyDirectory(DirectoryTree):
    """Directory tree that only shows folders."""

    def filter_paths(self, paths):
        """Filter to only show directories."""
        return [path for path in paths if path.is_dir()]
