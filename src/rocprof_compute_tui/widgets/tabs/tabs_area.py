"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from textual.containers import Vertical
from textual.widgets import TabPane, TextArea
from widgets.tabbed_content import TabsTabbedContent


class TabsArea(Vertical):
    """
    The response area.
    """

    COMPONENT_CLASSES = {
        "border-title-status",
    }

    def __init__(self):
        """Initialize the bottom panel."""

        super().__init__()

        # Create text areas as instance attributes
        self.tips_area = TextArea(id="tips-text", read_only=True)
        self.output_area = TextArea(id="output-text", read_only=True)
        self.terminal_area = TextArea(id="terminal-text")

        # Set initial tab
        self.default_tab = "tab-output"

    def compose(self):
        with TabsTabbedContent(initial="tab-output"):
            with TabPane("TIPS", id="tab-tips"):
                yield (self.tips_area)

            with TabPane("OUTPUT", id="tab-output"):
                yield (self.output_area)

            with TabPane("TERMINAL", id="tab-terminal"):
                yield (self.terminal_area)

    def on_mount(self) -> None:
        self.border_title = "BOTTOM TABS"
        self.add_class("section")
