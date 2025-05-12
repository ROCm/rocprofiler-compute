"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from pathlib import Path
from typing import Any, Dict

from textual.containers import Vertical
from textual.widgets import Button, Label
from widgets.collapsibles import (
    build_kernel_section,
    build_source_section,
    build_summary_section,
    build_sysinfo_section,
)
from widgets.directory_tree import FolderOnlyDirectory


class Browser(Vertical):
    """Left panel with directory tree and controls."""

    def __init__(self, start_path: Path):
        """Initialize the left panel."""
        super().__init__()
        self.start_path = start_path

    def compose(self):
        """Compose the left panel."""
        yield Label("Current Working Directory")
        yield FolderOnlyDirectory(self.start_path, id="dir-tree")
        with Vertical():
            yield Button("Analyze", id="analyze")

    def on_mount(self):
        self.border_title = "EXPLORER"
        self.add_class("section")
