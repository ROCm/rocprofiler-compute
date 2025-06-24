"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

from typing import Any, Dict, List

from textual.containers import ScrollableContainer
from textual.widgets import Label, Select

from rocprof_compute_tui.widgets.collapsibles import build_all_sections


class KernelView(ScrollableContainer):
    """Center panel with analysis results."""

    def __init__(
        self, config_path: str = "src/rocprof_compute_tui/utils/analyze_config.yaml"
    ):
        super().__init__(id="kernel-view")
        self.dfs = {}
        self.top_kernel = []
        self.config_path = config_path
        self.current_selection = None

    def compose(self):
        """
        Compose the initial center panel state.
        """
        yield Label(
            "Open a workload directory to run analysis and view per kernel results",
            classes="placeholder",
        )

    def update_results(self, dfs: Dict[str, Any], top_kernerl: List[Dict]) -> None:
        """
        Update the center panel with analysis results.
        """
        self.dfs = dfs
        self.top_kernel = top_kernerl
        self.remove_children()

        try:
            section = self.build_selector()

            self.mount(section)

        except Exception as e:
            self.mount(Label(f"Error displaying results: {str(e)}", classes="error"))

    def update_view(self, message: str, log_level: str) -> None:
        """
        Update the view with a status message.
        """
        self.remove_children()
        try:
            self.mount(Label(f"{message}", classes=log_level))
        except Exception as e:
            self.mount(Label(f"Error displaying results: {str(e)}", classes="error"))

    def reload_config(self, config_path: str = None) -> None:
        """
        Reload the configuration and update the view.
        """
        if config_path:
            self.config_path = config_path

        if self.dfs:
            self.update_results(self.dfs)

    def build_selector(self):
        kernel_names = [kernel["Kernel_Name"] for kernel in self.top_kernel]
        selector = Select.from_values(kernel_names)
        return selector

    def on_select_changed(self, event: Select.Changed):
        self.current_selection = event.value
        self._update_displayed_content()

    def _update_displayed_content(self):
        self.remove_children()

        section = self.build_selector()
        self.mount(section)

        if self.dfs:
            if self.current_selection:
                filtered_dfs = self.dfs[self.current_selection]

            try:
                sections = build_all_sections(filtered_dfs, self.config_path)
                for section in sections:
                    self.mount(section)
            except Exception as e:
                self.mount(Label(f"Error displaying results: {str(e)}", classes="error"))
