"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from pathlib import Path
from typing import Any, Dict

from textual.containers import ScrollableContainer, Vertical
from textual.widgets import Button, Label, TabbedContent, TabPane, TextArea
from widgets.collapsibles import (
    build_kernel_section,
    build_source_section,
    build_summary_section,
    build_sysinfo_section,
)
from widgets.directory_tree import FolderOnlyDirectory
from widgets.tabbed_content import PostingTabbedContent


class LeftPanel(Vertical):
    """Left panel with directory tree and controls."""

    def __init__(self, start_path: Path):
        """Initialize the left panel."""
        super().__init__()
        self.start_path = start_path

    def compose(self):
        """Compose the left panel."""
        yield Label("Directory Explorer")
        yield FolderOnlyDirectory(self.start_path, id="dir-tree")
        with Vertical():
            yield Button("Analyze", id="analyze")

    def on_mount(self):
        self.add_class("section")


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


class RightPanel(Vertical):
    """Right panel for additional tools."""

    def __init__(self):
        """Initialize the right panel."""
        super().__init__()

    def compose(self):
        """Compose the right panel."""
        yield Label("🚧 Under Construction")

    def _on_mount(self):
        self.border_title = "🚧 Under Construction"
        self.add_class("section")

class BottomPanel(Vertical):
    """Bottom panel with tabbed output areas."""

    def __init__(self):
        """Initialize the bottom panel."""

        super().__init__()

        # Create text areas as instance attributes
        self.tips_area = TextArea(id="tips-text", read_only=True)
        self.output_area = TextArea(id="output-text", read_only=True)
        self.terminal_area = TextArea(id="terminal-text", read_only=True)

        # Set initial tab
        self.default_tab = "tab-output"

    def compose(self):
        with PostingTabbedContent(initial="tab-output"):
            with TabPane("TIPS", id="tab-tips"):
                yield (self.tips_area)

            with TabPane("OUTPUT", id="tab-output"):
                yield (self.output_area)

            with TabPane("TERMINAL", id="tab-terminal"):
                yield (self.terminal_area)

    def on_mount(self):
        self.add_class("section")