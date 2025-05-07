"""
Main View Module
---------------
Contains the main view layout and organization for the application.
"""

import sys
from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Button
from widgets.directory_tree import FolderOnlyDirectory
from widgets.panels import BottomPanel, CenterPanel, LeftPanel, RightPanel
from widgets.splitter import HorizontalSplitter

from config import DEFAULT_START_PATH, SECTIONS_TO_SKIP
from utils.tui_utils import analyze_runner, get_table_dfs


class MainView(Horizontal):
    """Main view layout for the application."""

    selected_path = reactive(None)
    dfs = reactive({})

    def __init__(self):
        """Initialize the main view."""
        super().__init__(id="main-container")
        self.start_path = (
            # NOTE: is cwd the best choice?
            Path.cwd()
            if DEFAULT_START_PATH is None
            else Path(DEFAULT_START_PATH)
        )

        # Set up output redirect
        self.terminal_output = ""
        self._old_stdout = sys.stdout
        sys.stdout = self

    def write(self, text):
        """Handle stdout writes for terminal output."""
        self.terminal_output += text
        if hasattr(self, "terminal") and self.terminal:
            self.terminal.text = self.terminal_output
        return len(text)

    def flush(self):
        """Required for stdout compatibility."""
        pass

    def compose(self) -> ComposeResult:
        """Compose the main view layout."""
        # Left Panel - Directory navigation and controls
        yield LeftPanel(self.start_path)

        # Center Container - Holds both analysis results and output tabs
        with Vertical(id="center-container"):
            # Center Panel - Analysis results display
            yield CenterPanel()

            # Bottom Panel - Output, terminal, and tips
            bottom_panel = BottomPanel()
            yield bottom_panel

            # Store references to text areas
            self.tooltips = bottom_panel.tips_area
            self.output = bottom_panel.output_area
            self.terminal = bottom_panel.terminal_area

        # Right Panel - Additional tools/features
        yield RightPanel()

    @on(FolderOnlyDirectory.DirectorySelected)
    def on_directory_selected(self, event: FolderOnlyDirectory.DirectorySelected) -> None:
        """Handle directory selection events."""
        self.selected_path = event.path
        self.output.text += f"\nSelected directory: {self.selected_path}"

    @on(Button.Pressed, "#analyze")
    def on_analyze_button(self, event: Button.Pressed) -> None:
        """Handle analyze button press."""
        self.run_analysis()

    @work(thread=True)
    def run_analysis(self) -> None:
        """Run analysis on the selected directory."""
        if not self.selected_path:
            self.output.text += "\nNo directory selected for analysis."
            return

        try:
            self.output.text += f"\nRunning analysis on: {self.selected_path}"
            stdout_output, stderr_output, exit_code, cmd_str = analyze_runner(
                self.selected_path
            )

            if exit_code == 0:
                self.output.text += f"\nLoading analysis data..."
                self.dfs = get_table_dfs()
                self.app.call_from_thread(self.refresh_results)
                self.output.text += f"\nAnalysis completed successfully"
            else:
                self.output.text += f"\nAnalysis failed: {stderr_output}"
                self.output.text += f"\nCommand: {cmd_str}"
        except Exception as e:
            self.output.text += f"\nAnalysis error: {str(e)}"

    def refresh_results(self) -> None:
        """Refresh analysis results in the center panel."""
        center_panel = self.query_one(CenterPanel)
        center_panel.update_results(self.dfs)

    def refresh_view(self) -> None:
        """Refresh the entire view."""
        self.output.text += "\nRefreshing view..."
        if self.dfs:
            self.refresh_results()
