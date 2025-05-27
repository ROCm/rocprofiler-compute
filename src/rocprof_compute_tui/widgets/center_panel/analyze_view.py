"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from typing import Any, Dict

from textual.containers import ScrollableContainer
from textual.widgets import Label

from rocprof_compute_tui.widgets.collapsibles import (
    build_kernel_section,
    build_source_section,
    build_summary_section,
    build_sysinfo_section,
)


class AnalyzeView(ScrollableContainer):
    """Center panel with analysis results."""

    def __init__(self):
        """Initialize the center panel."""
        super().__init__(id="analyze-view")
        self.dfs = {}

    def compose(self):
        """Compose the initial center panel state."""
        yield Label(
            "Open a workload directory to run analysis and view results",
            classes="placeholder",
        )

    def update_results(self, dfs: Dict[str, Any]) -> None:
        """Update the center panel with analysis results."""
        self.dfs = dfs
        self.remove_children()

        # Create and mount widgets one by one to preserve order
        try:
            self.mount(build_summary_section(self.dfs))
            self.mount(build_sysinfo_section(self.dfs))
            self.mount(build_kernel_section(self.dfs))
            self.mount(build_source_section(self.dfs))

        except Exception as e:
            self.mount(Label(f"Error displaying results: {str(e)}", classes="error"))

    def update_view(self, message: str, log_level: str) -> None:
        self.remove_children()
        try:
            self.mount(Label(f"{message}", classes=log_level))
        except Exception as e:
            self.mount(Label(f"Error displaying results: {str(e)}", classes="error"))
