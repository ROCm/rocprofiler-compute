"""
Main View Module
---------------
Contains the main view layout and organization for the application.
"""

from pathlib import Path

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from widgets.center_panel.center_area import CenterPanel
from widgets.collapsibles import DataTable
from widgets.menu_bar.menu_bar import MenuBar
from widgets.right_panel.right import RightPanel
from widgets.tabs.tabs_area import TabsArea

from config import DEFAULT_START_PATH
from utils.tui_utils import Logger, LogLevel, analyze_runner, get_table_dfs


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

        self.logger = Logger()
        self.logger.info("MainView initialized", update_ui=False)

    def flush(self):
        """Required for stdout compatibility."""
        pass

    def compose(self) -> ComposeResult:
        """Compose the main view layout."""
        self.logger.info("Composing main view layout", update_ui=False)
        yield MenuBar()

        # Center Container - Holds both analysis results and output tabs
        with Horizontal(id="center-container"):
            with Vertical(id="activity-container"):
                # Center Panel - Analysis results display
                center_panel = CenterPanel()
                yield center_panel

                self.center = center_panel

                # Bottom Panel - Output, terminal, and tips
                tabs = TabsArea()
                yield tabs

                # Store references to text areas
                self.tooltips = tabs.tips_area
                self.output = tabs.output_area

                # Now set the output area for the logger
                self.logger.set_output_area(self.output)
                self.logger.info("Main view layout composed")

            # Right Panel - Additional tools/features
            yield RightPanel()

    @on(DataTable.CellSelected)
    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        table = event.data_table
        row_idx = event.coordinate.row

        self.logger.info(f"Cell selected at row {row_idx}")

        try:
            row_data = table.get_row_at(row_idx)
            content = f"Selected Row {row_idx}:\n"
            content += "\n".join(f"{val}" for val in row_data)

            self.tooltips.text = content
            self.logger.info(f"Row {row_idx} data displayed in tooltips")

        except Exception as e:
            error_msg = f"Error displaying row {row_idx}: {str(e)}"
            table.add_column("Error")
            table.add_row(str(e))
            self.tooltips.text = error_msg
            self.logger.error(error_msg)

        self.run_analysis()

    @work(thread=True)
    def run_analysis(self) -> None:
        if not self.selected_path:
            error_msg = "No directory selected for analysis"
            self._update_view(error_msg, LogLevel.ERROR)
            self.logger.error(error_msg)
            return

        try:
            self.logger.info(f"Starting analysis on: {self.selected_path}")
            self._update_view(
                f"Running analysis on: {self.selected_path}", LogLevel.SUCCESS
            )

            # Run analysis and capture results
            stdout_output, stderr_output, exit_code, cmd_str = analyze_runner(
                self.selected_path
            )

            if exit_code == 0:
                self._update_view("Loading analysis data...", LogLevel.SUCCESS)
                self.logger.info(
                    "Analysis command executed successfully, loading data..."
                )

                # Log command output if available
                if stdout_output:
                    self.logger.info(f"Analysis command output: {stdout_output}")

                # Load data with error handling
                try:
                    self.dfs = get_table_dfs()
                    if not self.dfs:
                        warning_msg = "Analysis completed but no data was returned"
                        self._update_view(warning_msg, LogLevel.WARNING)
                        self.logger.warning(warning_msg)
                    else:
                        self.app.call_from_thread(self.refresh_results)
                        self.logger.success("Analysis completed successfully")
                except Exception as data_error:
                    error_msg = f"Error loading analysis data: {str(data_error)}"
                    self._update_view(error_msg, LogLevel.ERROR)
                    self.logger.error(error_msg)
            else:
                error_msg = f"Analysis failed with exit code {exit_code}: {stderr_output}"
                command_info = f"Running command: {cmd_str}"
                self._update_view(f"{error_msg}\n{command_info}", LogLevel.ERROR)
                self.logger.error(f"{error_msg}\n{command_info}")
        except Exception as e:
            self.logger.error(f"Unexpected error during analysis: {str(e)}")

    def _update_view(self, message: str, log_level: LogLevel) -> None:
        try:
            # Use call_from_thread to safely update UI from background thread
            self.app.call_from_thread(self._safe_update_view, message, log_level)
        except Exception as e:
            # Capture errors that might occur when scheduling the UI update
            self.logger.error(f"View update scheduling error: {str(e)}")

    def _safe_update_view(self, message: str, log_level: LogLevel) -> None:
        try:
            analyze_view = self.query_one("#analyze-view")
            if analyze_view:
                analyze_view.update_view(message, log_level)
            else:
                self.logger.warning("Analysis view not found when updating log")
        except Exception as e:
            self.logger.error(f"Log update error: {str(e)}")

    def refresh_results(self) -> None:
        try:
            self.logger.info("Refreshing analysis results")
            analyze_view = self.query_one("#analyze-view")
            if not analyze_view:
                self.logger.error("Analysis view not found")
                return

            if not hasattr(self, "dfs") or self.dfs is None:
                self.logger.error("No analysis data available to display")
                return

            analyze_view.update_results(self.dfs)
            self.logger.success("Results displayed successfully")
        except Exception as e:
            self.logger.error(f"Error refreshing results: {str(e)}")

    def refresh_view(self) -> None:
        self.logger.info("Refreshing view...")
        if self.dfs:
            self.refresh_results()
        else:
            self.logger.warning("No data available for refresh")
