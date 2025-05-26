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

from rocprof_compute_tui.analysis_tui import tui_analysis
from rocprof_compute_tui.config import DEFAULT_START_PATH
from rocprof_compute_tui.utils.tui_utils import (
    Logger,
    LogLevel,
)
from rocprof_compute_tui.widgets.center_panel.center_area import CenterPanel
from rocprof_compute_tui.widgets.collapsibles import DataTable
from rocprof_compute_tui.widgets.menu_bar.menu_bar import MenuBar
from rocprof_compute_tui.widgets.right_panel.right import RightPanel
from rocprof_compute_tui.widgets.tabs.tabs_area import TabsArea
from utils import file_io


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

                # Bottom Panel - Output, terminal, and metric description
                tabs = TabsArea()
                yield tabs

                # Store references to text areas
                self.metric_description = tabs.description_area
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

            self.metric_description.text = content
            self.logger.info(f"Row {row_idx} data displayed in metric_description")

        except Exception as e:
            error_msg = f"Error displaying row {row_idx}: {str(e)}"
            table.add_column("Error")
            table.add_row(str(e))
            self.metric_description.text = error_msg
            self.logger.error(error_msg)

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

            analyzer = tui_analysis(self.app.args, self.app.supported_archs)
            analyzer.sanitize()

            sys_info = file_io.load_sys_info(
                Path(self.selected_path).joinpath("sysinfo.csv")
            )

            sys_info = sys_info.iloc[0].to_dict()
            self.app.load_soc_specs(sys_info)

            analyzer.set_soc(self.app.soc)
            analyzer.pre_processing()
            self.dfs = analyzer.run_analysis()

        except Exception as e:
            self.logger.error(f"Unexpected error during analysis: {str(e)}")
            self._update_view(
                f"Unexpected error during analysis: {str(e)}", LogLevel.ERROR
            )

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
