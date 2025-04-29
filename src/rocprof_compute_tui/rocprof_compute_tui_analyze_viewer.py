import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from mem_chart import plot_mem_chart
from textual import on, work
from textual.app import ComposeResult
from textual.containers import (
    Horizontal,
    ScrollableContainer,
    Vertical,
    VerticalScroll,
)
from textual.screen import Screen
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Label,
    Static,
    TabbedContent,
    TabPane,
    TextArea,
)
from tui_plots import RooflinePlot
from tui_utils import get_table_dfs

SECTIONS_TO_SKIP = [
    "0. Top Stats",
    "1. System Info",
    "2. System Speed-of-Light",
    "3. Memory Chart",
]


class FolderOnlyDirectory(DirectoryTree):
    def filter_paths(self, paths):
        return [path for path in paths if path.is_dir()]


class AnalysisScreen(Screen):

    CSS = """
    Screen {
        background: $surface;
        color: $text;
    }

    /* Main container grid layout */
    #main-container {
        layout: grid;
        grid-size: 3 2;
        grid-columns: 1fr 7fr 1fr;
        grid-rows: 1fr auto;
        width: 100%;
        height: 100%;
        padding: 1;
        grid-gutter: 1;
    }

    #center-container {
        layout: grid;
        grid-size: 1 2;
        grid-rows: 5fr 1fr;
        height: 100%;
    }

    /* Panel styling */
    #left-panel, #right-panel, #center-panel, #bottom-panel {
        border: solid $primary;
        background: $surface-darken-1;
    }

    /* Directory Tree */
    DirectoryTree {
        height: 1fr;
    }

    .tree-folder {
        color: $accent;
        text-style: bold;
    }

    /* Buttons */
    Button {
        width: 1fr;
        margin: 1;
    }

    Button:hover {
        background: $primary-lighten-1;
    }

    /* Collapsible sections */
    Collapsible {
    }

    .collapsible-title {
        background: $surface-darken-2;
        text-style: bold;
    }

    .collapsible-content {
        background: $surface;
    }

    #summary-section {
        height: 1;
        margin: 0;
    }

    .summary-grid {
    }

    .header-cell {
        text-style: bold;
        color: cyan;
        padding: 0 1;
        height: 1;
        margin: 0;
    }

    .data-cell {
        color: magenta;
        padding: 0 1;
        height: 1;
        margin: 0;
    }

    .row {
        width: 1fr;
        height: 1;
        margin: 0;
    }

    /* DataTables */
    DataTable {
        height: auto;
    }

    .datatable-header {
        background: $surface-darken-2;
        text-style: bold;
    }

    .datatable-row {
        background: $surface;
    }

    .datatable-row-odd {
        background: $surface-darken-1;
    }

    /* Terminal output */
    #terminal-log {
        height: 100%;
        background: $surface-darken-3;
        padding: 1;
    }

    .roofline-plot {
        padding: 1;
        width: auto;
        height: auto;
        background: $surface;
        color: $text;
    }

    .mem-chart {
        border: solid $accent;
        padding: 0;
        width: auto;
        height: auto;
        overflow-y: auto;
        overflow-x:auto;
        background: $surface;
        color: $text;
    }

    /* Debug view styling */
    .debug-view {
        border: solid $error;
        padding: 1;
        width: 100%;
    }

    /* Status classes */
    .error {
        color: $error;
        text-style: bold;
    }

    .warning {
        color: $warning;
    }

    .success {
        color: $success;
    }
    """

    def __init__(self, dfs: Dict[str, Any] = None):
        super().__init__()
        self.dfs = dfs or {}
        self.selected_path = Path.cwd()
        self.tooltips = TextArea(read_only=True)
        self.logs = TextArea(read_only=True)
        sys.stdout = self  # Redirect stdout

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

        with Horizontal(id="main-container"):
            # Left Panel
            with Vertical(id="left-panel"):
                yield Label("Directory Explorer")
                yield FolderOnlyDirectory(self.selected_path)
                with Horizontal():
                    yield Button("Analyze", id="analyze")
                    yield Button("Refresh", id="refresh")

            with Vertical(id="center-container"):
                # Center Panel
                with ScrollableContainer(id="center-panel"):
                    yield from self._compose_initial_state()

                # Bottom Row → TabbedContent
                with TabbedContent(initial="tab-tips", id="bottom-panel"):
                    with TabPane("Tips", id="tab-tips"):
                        yield (self.tooltips)
                    with TabPane("Terminal Output", id="tab-terminal"):
                        yield (self.logs)

            with Vertical(id="right-panel"):
                yield Label("🚧 Under Construction")

    def _compose_initial_state(self) -> ComposeResult:
        """Initial empty state before analysis"""
        yield Label(
            "Select a workload directory to run analysis and view results",
            classes="placeholder",
        )

    def refresh_results(self) -> None:
        """Safe refresh with proper mounting sequence"""
        center_panel = self.query_one("#center-panel")
        center_panel.remove_children()

        # Create all widgets first
        widgets = list(self._compose_results())

        # Mount only after full construction
        center_panel.mount(*widgets)

    def _compose_results(self) -> ComposeResult:
        """Main results composition"""
        try:
            yield Label("Analysis Results")
            yield self._build_summary_section()
            yield self._build_sysinfo_section()
            yield self._build_kernel_section()

        except Exception as e:
            self.logs.text = f"Display Error: {str(e)}"

    def _build_summary_section(self) -> Collapsible:
        """Build complete collapsible section"""
        df = self.dfs["0. Top Stats"]["0.1 Top Kernels"]

        summary = Collapsible(
            Label("Top Kernels by Duration (ns):", classes="section-header"),
            self._create_table(df),
            title="📊 Kernel Summary",
            collapsed=True,
        )

        summary.add_class("summary-section")
        return summary

    def _build_sysinfo_section(self) -> ComposeResult:
        """Build the kernels section with hierarchical data"""
        sysinf_children = []

        ########################################
        # 1. System Speed of Light
        ########################################
        df = self.dfs["2. System Speed-of-Light"]["2.1 Speed-of-Light"]
        sysinf_children.append(
            Collapsible(
                self._create_table(df),
                title="System Speed-of-Light",
                collapsed=True,
            ),
        )

        ########################################
        # 2. Roofline
        ########################################
        sysinf_children.append(
            Collapsible(
                VerticalScroll(RooflinePlot()),
                title="Roofline",
                collapsed=True,
                id="roofline-plot",
            )
        )

        ########################################
        # 3. Memory Chart (ANSI Art)
        ########################################
        df = self.dfs["3. Memory Chart"]["3.1 Memory Chart"]
        sysinf_children.append(
            Collapsible(
                self._create_mem_chart(df),
                title="Memory Chart",
                collapsed=True,
            ),
        )

        sysinfo = Collapsible(
            *sysinf_children, title="⚡ System Information", collapsed=True
        )
        sysinfo.add_class("sysinfo-section")
        return sysinfo

    def _build_kernel_section(self) -> Collapsible:
        """Build the detailed metrics section"""
        children = [Label("Performance Metrics", classes="section-header")]

        for section_name, subsections in self.dfs.items():
            if section_name in SECTIONS_TO_SKIP:
                continue
            kernel_children = []
            for subsection_name, df in subsections.items():
                kernel_children.append(
                    Collapsible(
                        self._create_table(df), title=subsection_name, collapsed=True
                    )
                )
            children.append(
                Collapsible(*kernel_children, title=section_name, collapsed=True)
            )

        kernels = Collapsible(*children, title="🔍 Kernels", collapsed=True)
        kernels.add_class("kernels-section")
        return kernels

    def _create_table(self, df: pd.DataFrame) -> DataTable:
        table = DataTable(zebra_stripes=True)

        str_columns = [str(col) for col in df.columns]
        table.add_columns(*str_columns)

        table.add_rows([tuple(str(x) for x in row) for row in df.itertuples(index=False)])

        return table

    def on_data_table_cell_selected(self, event: DataTable.CellSelected):
        table = event.data_table
        row_idx = event.coordinate.row

        try:
            row_data = table.get_row_at(row_idx)
            content = f"Selected Row {row_idx}:\n"
            content += "\n".join(f"{val}" for val in row_data)

            # Show it in the TextArea
            self.tooltips.text = content

        except Exception as e:
            self.tooltips.text = f"Error displaying row {str(row_idx)}: {str(e)}"

    def _create_mem_chart(self, df: pd.DataFrame) -> Static:
        """Create memory chart visualization"""
        try:
            # Prepare data
            metric_dict = df[["Metric", "Value"]].set_index("Metric").to_dict()["Value"]
            self.logs.text = f"Metrics: {metric_dict}"

            import sys
            from io import StringIO

            original_stdout = sys.stdout
            string_buffer = StringIO()
            sys.stdout = string_buffer

            try:
                result = plot_mem_chart("", "per_kernel", metric_dict)
                stdout_output = string_buffer.getvalue()

                if stdout_output:
                    self.logs.text += (
                        f"\nGot output from stdout: {len(stdout_output)} chars"
                    )
                    plot_str = stdout_output
                elif result:
                    self.logs.text += f"\nGot returned value: {len(str(result))} chars"
                    plot_str = str(result)
                else:
                    self.logs.text += "\nNo output captured from either method"
                    plot_str = "No chart data generated"
            finally:
                sys.stdout = original_stdout

            # Wrap the Static widget in a Scroll container
            return Static(
                plot_str,
                markup=False,
                classes="mem-chart",
                shrink=False,
            )

        except Exception as e:
            self.logs.text = f"Memory chart error: {str(e)}\n{type(e)}"
            import traceback

            self.logs.text += f"\n{traceback.format_exc()}"
            return Static(f"Error: {str(e)}", classes="error")

    @on(Button.Pressed, "#analyze")
    def on_analyze(self):
        self.run_analysis(self.selected_path)

    @work(thread=True)
    def run_analysis(self, path: Path):
        """Your analysis logic here"""
        try:
            # Simulate analysis
            self.dfs = get_table_dfs()
            self.app.call_from_thread(self.refresh_results)
        except Exception as e:
            self.logs.text = f"Analysis failed: {e}"
