from textual.app import ComposeResult
from textual.containers import VerticalScroll, Horizontal, Vertical
from textual.widgets import (
    DirectoryTree,
    Static,
    Header,
    Footer,
    Button,
    Collapsible,
    Label,
    DataTable,
    RichLog,
)
import pandas as pd
from pathlib import Path
import sys
from typing import Dict, Any
from textual.screen import Screen
from textual import on, events, work

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
        grid-columns: 1fr 3fr 1fr;
        grid-rows: 1fr auto;
        width: 100%;
        height: 100%;
        padding: 1;
        grid-gutter: 1;
    }

    /* Panel styling */
    #left-panel, #right-panel, #center-panel, #terminal-panel {
        border: solid $primary;
        background: $surface-darken-1;
    }

    /* Directory Tree */
    DirectoryTree {
        height: 1fr;
    }

    .tree--folder {
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
        margin: 1 0;
    }

    .collapsible-title {
        background: $surface-darken-2;
        padding: 1;
        text-style: bold;
    }

    .collapsible-content {
        padding: 1;
        background: $surface;
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
        self.terminal_log = RichLog(id="terminal-log")
        sys.stdout = self  # Redirect stdout

    def write(self, text: str):
        self.terminal_log.write(text.strip())

    def flush(self):
        pass

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
                with VerticalScroll(id="center-panel"):
                    yield from self._compose_initial_state()

                # Bottom Row - Terminal
                with Vertical(id="terminal-panel"):
                    yield Label("Terminal Output")
                    yield self.terminal_log
            with Vertical(id="right-panel"):
                yield Label("Toolbox")

    def _compose_initial_state(self) -> ComposeResult:
        """Initial empty state before analysis"""
        yield Label("Run analysis to view results", classes="placeholder")

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
            yield Label(f"Display Error: {str(e)}", classes="error")

    def _build_summary_section(self) -> Collapsible:
        """Build complete collapsible section"""
        df = self.dfs["0. Top Stats"]["0.1 Top Kernels"]
        # Create all child widgets first
        summary_children = [
            Label("Top Kernels by Duration (ns):", classes="section-header")
        ]

        # Build collapsible with pre-constructed children
        summary_children.extend(
            [
                Vertical(
                    self._df_to_rich_view(df),
                )
            ]
        )
        summary = Collapsible(
            *summary_children, title="📊 Kernel Summary", collapsed=True
        )
        summary.add_class("summary-section")
        return summary

    def _build_sysinfo_section(self) -> ComposeResult:
        """Build the kernels section with hierarchical data"""
        sysinf_children = []

        df = self.dfs["2. System Speed-of-Light"]["2.1 Speed-of-Light"]
        sysinf_children.append(
            Collapsible(
                self._df_to_rich_view(df),
                title="System Speed-of-Light",
                collapsed=True,
            ),
        )

        df = self.dfs["3. Memory Chart"]["3.1 Memory Chart"]
        sysinf_children.append(
            Collapsible(
                self._df_to_rich_view(df),
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
                        self._df_to_rich_view(df), title=subsection_name, collapsed=True
                    )
                )
            children.append(
                Collapsible(*kernel_children, title=section_name, collapsed=True)
            )

        kernels = Collapsible(*children, title="🔍 Kernels", collapsed=True)
        kernels.add_class("kernels-section")
        return kernels

    def _create_table(self, df: pd.DataFrame) -> DataTable:
        """Safely create populated DataTable"""
        table = DataTable()

        try:
            if not df.empty:
                columns = list(df.columns)
                table.add_columns(*columns)

                for i in range(len(df)):
                    row = df.iloc[i]
                    table.add_row(
                        *[
                            (
                                str(row[col])
                                if col in df.columns and not pd.isna(row[col])
                                else ""
                            )
                            for col in columns
                        ]
                    )
            else:
                table.add_column("Info")
                table.add_row("No data available")

        except Exception as e:
            table.add_column("Error")
            table.add_row(str(e))

        return table

    def _df_to_rich_view(self, df: pd.DataFrame) -> Static:
        """Convert DataFrame to a beautiful rich table with proper alignment"""

        from rich.table import Table
        from rich.box import SQUARE

        # Create a Rich Table
        rich_table = Table(
            box=SQUARE,
            show_header=True,
            header_style="bold cyan",
            show_lines=True,
            expand=True,
        )

        # Add columns
        for col in df.columns:
            # Right-align numeric columns, left-align others
            justify = "right" if pd.api.types.is_numeric_dtype(df[col]) else "left"
            rich_table.add_column(str(col), justify=justify, style="magenta")

        # Add rows
        for row in df.itertuples(index=False):
            rich_table.add_row(*[str(x) for x in row])

        return Static(rich_table)

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
            print(f"Analysis failed: {e}")

    def on_unmount(self):
        sys.stdout = sys.__stdout__
