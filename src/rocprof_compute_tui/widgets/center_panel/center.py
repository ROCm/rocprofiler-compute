"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from pathlib import Path
from typing import Any, Dict

from textual.containers import ScrollableContainer
from textual.widgets import Label
from widgets.collapsibles import (
    build_kernel_section,
    build_source_section,
    build_summary_section,
    build_sysinfo_section,
)


class CenterPanel(ScrollableContainer):
    """Center panel with analysis results."""

    def __init__(self):
        """Initialize the center panel."""
        super().__init__()
        self.dfs = {}

    def compose(self):
        """Compose the initial center panel state."""
        yield Label(
            "Select a workload directory to run analysis and view results",
            classes="placeholder",
        )

    def update_results(self, dfs: Dict[str, Any]) -> None:
        """Update the center panel with analysis results."""
        self.dfs = dfs
        self.remove_children()

        # Create and mount widgets one by one to preserve order
        try:
            # Main header
            self.mount(Label("Analysis Results", classes="main-header"))

            # Use the builder functions that match the original structure
            self.mount(build_summary_section(dfs))
            self.mount(build_sysinfo_section(dfs))
            self.mount(build_kernel_section(dfs))
            self.mount(build_source_section(dfs))

        except Exception as e:
            self.mount(Label(f"Error displaying results: {str(e)}", classes="error"))

    def on_mount(self):
        self.add_class("section")
