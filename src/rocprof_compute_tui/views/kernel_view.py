"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from typing import Any, Dict, List, Optional

from textual import on
from textual.containers import Container, VerticalScroll
from textual.widgets import Label, RadioButton, RadioSet

from config import rocprof_compute_home
from rocprof_compute_tui.widgets.collapsibles import build_all_sections


class KernelView(Container):
    """Center panel with analysis results split into two scrollable sections."""

    DEFAULT_CSS = """
    KernelView {
        layout: vertical;
    }

    #top-container {
        height: 1fr;
        border: none;
    }

    #bottom-container {
        height: 7fr;
        border: none;
    }
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(id="kernel-view")
        self.dfs = {}
        self.top_kernel = []

        if rocprof_compute_home:
            config_path = (
                rocprof_compute_home
                / "rocprof_compute_tui"
                / "utils"
                / "kernel_view_config.yaml"
            )
        self.config_path = config_path

        self.current_selection = None

    def compose(self):
        """
        Compose the split panel layout with two scrollable containers.
        """
        with VerticalScroll(id="top-container"):
            yield Label(
                "Open a workload directory to run analysis and view kernel selection",
                classes="placeholder",
            )

        with VerticalScroll(id="bottom-container"):
            # empty on init
            pass

    def update_results(self, dfs: Dict[str, Any], top_kernel: List[Dict]) -> None:
        """
        Update both containers with analysis results.
        """
        self.dfs = dfs
        self.top_kernel = top_kernel

        # Update top container with radio set
        top_container = self.query_one("#top-container", VerticalScroll)
        top_container.remove_children()

        if self.dfs and self.top_kernel:
            top_container.mount(Label("Select a kernel to view detailed analysis."))
            try:
                selector = self.build_selector()
                top_container.mount(selector)
            except Exception as e:
                top_container.mount(
                    Label(f"Error displaying kernel list: {str(e)}", classes="error")
                )
        else:
            top_container.mount(Label("No kernels available", classes="placeholder"))

        # Clear bottom container until selection is made
        bottom_container = self.query_one("#bottom-container", VerticalScroll)
        bottom_container.remove_children()
        bottom_container.mount(
            Label(
                "Select a kernel from above to view detailed analysis",
                classes="placeholder",
            )
        )

    def reload_config(self, config_path: str = None) -> None:
        """
        Reload the configuration and update the view.
        """
        if config_path:
            self.config_path = config_path

        if self.dfs:
            self.update_results(self.dfs, self.top_kernel)

    def build_selector(self):
        """Build the radio set for kernel selection."""
        radio_buttons = []
        for kernel in self.top_kernel:
            radio_buttons.append(RadioButton(kernel["Kernel_Name"]))

        selector = RadioSet(*radio_buttons)
        return selector

    @on(RadioSet.Changed)
    def on_radio_changed(self, event: RadioSet.Changed) -> None:
        """Handle radio button selection and update bottom container."""
        if event.pressed:
            selected_kernel = event.pressed.label.plain
            self.current_selection = selected_kernel
            self._update_bottom_content()

    def _update_bottom_content(self):
        """Update the bottom container with detailed analysis for selected kernel."""
        bottom_container = self.query_one("#bottom-container", VerticalScroll)
        bottom_container.remove_children()

        if self.dfs and self.current_selection:
            # Check if current_selection exists in dfs
            if self.current_selection in self.dfs:
                filtered_dfs = self.dfs[self.current_selection]

                try:
                    sections = build_all_sections(filtered_dfs, self.config_path)
                    for section in sections:
                        bottom_container.mount(section)
                except Exception as e:
                    bottom_container.mount(
                        Label(f"Error displaying results: {str(e)}", classes="error")
                    )
            else:
                bottom_container.mount(
                    Label(
                        f"No data available for kernel: {self.current_selection}",
                        classes="error",
                    )
                )
        else:
            bottom_container.mount(
                Label("Select a kernel to view detailed analysis", classes="placeholder")
            )
