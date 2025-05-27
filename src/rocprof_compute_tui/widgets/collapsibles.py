"""
Collapsible Section Widgets
-------------------------
Contains collapsible section builders that match the original structure.
"""

from typing import Any, Dict

import pandas as pd
from textual.containers import VerticalScroll
from textual.widgets import Collapsible, DataTable, Label

from rocprof_compute_tui.config import SECTIONS_TO_SKIP
from rocprof_compute_tui.widgets.charts import MemoryChart, RooflinePlot


def create_table(df: pd.DataFrame) -> DataTable:
    """Create a data table from a DataFrame."""
    table = DataTable(zebra_stripes=True)

    # Clean the DataFrame - remove NaN and empty cells
    df = df.dropna(how="any")
    df = df[~df.apply(lambda row: row.astype(str).str.strip().eq("").any(), axis=1)]

    # Add columns and rows
    str_columns = [str(col) for col in df.columns]
    table.add_columns(*str_columns)
    table.add_rows([tuple(str(x) for x in row) for row in df.itertuples(index=False)])

    return table


def build_summary_section(dfs: Dict[str, Any]) -> Collapsible:
    summary_children = []

    try:
        # Top Kernels section
        df = dfs["0. Top Stats"]["0.1 Top Kernels"]["df"]
        summary_children.append(
            Collapsible(
                Label("Top Kernels by Duration (ns):", classes="section-header"),
                create_table(df),
                title="Top Kernels",
                collapsed=True,
            )
        )
    except (KeyError, Exception) as e:
        summary_children.append(
            Label(f"Top Kernels data not available: {str(e)}", classes="warning")
        )

    try:
        # Dispatch List
        df = dfs["0. Top Stats"]["0.2 Dispatch List"]["df"]
        summary_children.append(
            Collapsible(
                create_table(df),
                title="Dispatch List",
                collapsed=True,
            )
        )
    except (KeyError, Exception) as e:
        summary_children.append(
            Label(f"Dispatch List data not available: {str(e)}", classes="warning")
        )

    try:
        # System Info
        df = dfs["1. System Info"]["1.1"]["df"]
        summary_children.append(
            Collapsible(
                create_table(df),
                title="System Info",
                collapsed=True,
            )
        )
    except (KeyError, Exception) as e:
        summary_children.append(
            Label(f"System Info data not available: {str(e)}", classes="warning")
        )

    # Create and return the top-level collapsible
    summary = Collapsible(*summary_children, title="📊 Summaries", collapsed=True)
    summary.add_class("summary-section")
    return summary


def build_sysinfo_section(dfs: Dict[str, Any]) -> Collapsible:
    sysinf_children = []

    try:
        # Speed-of-Light section
        df = dfs["2. System Speed-of-Light"]["2.1 Speed-of-Light"]["df"]
        sysinf_children.append(
            Collapsible(
                create_table(df),
                title="System Speed-of-Light",
                collapsed=True,
            )
        )
    except (KeyError, Exception) as e:
        sysinf_children.append(
            Label(f"Speed-of-Light data not available: {str(e)}", classes="warning")
        )

    # Roofline section
    sysinf_children.append(
        Collapsible(
            VerticalScroll(RooflinePlot()),
            title="Roofline",
            collapsed=True,
            id="roofline-plot",
        )
    )

    try:
        # Memory Chart section
        df = dfs["3. Memory Chart"]["3.1 Memory Chart"]["df"]
        sysinf_children.append(
            Collapsible(
                MemoryChart(df),
                title="Memory Chart",
                collapsed=True,
            )
        )
    except (KeyError, Exception) as e:
        sysinf_children.append(
            Label(f"Memory Chart data not available: {str(e)}", classes="warning")
        )

    # Create and return the top-level collapsible
    sysinfo = Collapsible(
        *sysinf_children, title="⚡ High Level Analysis", collapsed=True
    )
    sysinfo.add_class("sysinfo-section")
    return sysinfo


def build_kernel_section(dfs: Dict[str, Any]) -> Collapsible:
    children = []

    try:
        for section_name, subsections in dfs.items():
            if section_name in SECTIONS_TO_SKIP:
                continue

            kernel_children = []
            for subsection_name, df in subsections.items():
                df = df["df"]
                kernel_children.append(
                    Collapsible(create_table(df), title=subsection_name, collapsed=True)
                )

            if kernel_children:
                children.append(
                    Collapsible(*kernel_children, title=section_name, collapsed=True)
                )
    except Exception as e:
        children.append(Label(f"Error in Kernel Section: {str(e)}", classes="error"))

    # Create and return the top-level collapsible
    kernels = Collapsible(*children, title="🔍 Detailed Block Analysis", collapsed=True)
    kernels.add_class("kernels-section")
    return kernels


def build_source_section(dfs: Dict[str, Any]) -> Collapsible:
    children = [Label("🚧 Under Construction", classes="section-header")]

    # Create and return the top-level collapsible
    sources = Collapsible(*children, title="🚧 Source Level Analysis/PC Sampling", collapsed=True)
    sources.add_class("source-section")
    return sources
