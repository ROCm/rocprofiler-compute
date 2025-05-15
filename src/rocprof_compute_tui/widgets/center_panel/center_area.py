"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from textual.containers import Vertical
from textual.widgets import Label, TabPane
from widgets.center_panel.analyze_view import AnalyzeView
from widgets.tabbed_content import TabsTabbedContent


class CenterPanel(Vertical):
    """
    The response area.
    """

    COMPONENT_CLASSES = {
        "border-title-status",
    }

    def __init__(self):
        """Initialize the bottom panel."""

        super().__init__()

        self.analyze_view = AnalyzeView()

        # Set initial tab
        self.default_tab = "center-analyze"

    def compose(self):
        with TabsTabbedContent(initial="tab-analyze"):
            with TabPane("Analyze Results", id="tab-analyze"):
                yield self.analyze_view

            with TabPane("View 1", id="tab-1"):
                yield Label("🚧 Under Construction")

            with TabPane("View 2", id="tab-2"):
                yield Label("🚧 Under Construction")

    def on_mount(self) -> None:
        self.border_title = "CENTER TABS"
        self.add_class("section")
