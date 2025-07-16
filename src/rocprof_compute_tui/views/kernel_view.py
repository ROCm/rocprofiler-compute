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
        margin-top: 1;
    }

    #bottom-container {
        height: 4fr;
        border: none;
        margin-top: 2;
    }

    .kernel-table-header {
        background: $primary;
        color: $text;
        text-style: bold;
        padding: 0 1;
        offset: 5 0;
        margin-top: 1;
    }

    .kernel-row {
        padding: 0 1;
        border-bottom: solid $border;
    }

    RadioSet {
        border: solid $border;
    }
    """

    def __init__(self, config_path: Optional[str] = None):
        super().__init__(id="kernel-view")
        self.status_label = None
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

        self.keys = None
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
                header = self.build_header()
                top_container.mount(header)
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

    def update_view(self, message: str, log_level: str) -> None:
        """
        Update the view with a status message.
        """
        if self.status_label is None:
            self.status_label = Label(f"{message}", classes=log_level)
            self.mount(self.status_label)
        else:
            # Update existing label
            self.status_label.update(f"{message}")
            self.status_label.set_classes(log_level)

    def reload_config(self, config_path: str = None) -> None:
        """
        Reload the configuration and update the view.
        """
        if config_path:
            self.config_path = config_path

        if self.dfs:
            self.update_results(self.dfs, self.top_kernel)

    def build_header(self):
        if not self.top_kernel:
            return Label()

        all_keys = set()

        for kernel in self.top_kernel:
            all_keys.update(kernel.keys())

        self.keys = sorted(all_keys)

        if "Kernel_Name" in self.keys:
            self.keys.remove("Kernel_Name")
            self.keys.insert(0, "Kernel_Name")

        header_text = " | ".join(f"{key:25}" for key in self.keys)
        header_label = Label(header_text, classes="kernel-table-header")

        return header_label

    def build_selector(self):
        """Build the radio set for kernel selection."""
        if not self.top_kernel:
            return RadioSet()

        radio_buttons = []

        # Create radio buttons with formatted kernel data
        for i, kernel in enumerate(self.top_kernel):
            row_data = []
            for key in self.keys:
                value = str(kernel.get(key, "N/A"))
                if len(value) > 18:
                    value = value[:15] + "..."
                row_data.append(f"{value:25}")

            row_text = " | ".join(row_data)
            radio_button = RadioButton(row_text, id=f"kernel-{i}")
            radio_button.kernel_data = kernel
            radio_buttons.append(radio_button)

        selector = RadioSet(*radio_buttons)

        return selector

    @on(RadioSet.Changed)
    def on_radio_changed(self, event: RadioSet.Changed) -> None:
        """Handle radio button selection and update bottom container."""
        if event.pressed:
            kernel_data = getattr(event.pressed, "kernel_data", None)
            if kernel_data and "Kernel_Name" in kernel_data:
                selected_kernel = kernel_data["Kernel_Name"]
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
