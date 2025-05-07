"""
ROCm Compute Profiler TUI - Main Application
-------------------------------------------
This module contains the main application for the rocprof-compute tool.
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header
from views.main_view import MainView

from config import APP_TITLE, VERSION


class RocprofTUIApp(App):
    """Main application for the performance analysis tool."""

    TITLE = f"{APP_TITLE} v{VERSION}"
    SUB_TITLE = "Workload Analyze Tool"

    CSS_PATH = "assets/style.css"
    BINDINGS = [
        Binding(key="q", action="quit", description="Quit"),
        Binding(key="r", action="refresh", description="Refresh"),
        Binding(key="a", action="analyze", description="Analyze"),
    ]

    def __init__(self):
        """Initialize the application."""
        super().__init__()
        self.main_view = MainView()

    def compose(self) -> ComposeResult:
        """Compose the application layout."""
        yield Header()
        yield self.main_view
        yield Footer()

    def action_analyze(self) -> None:
        """Run analysis on the selected directory."""
        self.main_view.run_analysis()

    def action_refresh(self) -> None:
        """Refresh the view."""
        self.main_view.refresh_view()


if __name__ == "__main__":
    app = RocprofTUIApp()
    app.run()
