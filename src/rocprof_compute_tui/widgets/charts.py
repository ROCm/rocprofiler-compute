"""
Visualization Widget Modules
--------------------------
Contains custom visualization widgets for the application.
"""

from __future__ import annotations

import sys
import traceback
from io import StringIO

import pandas as pd
from textual.widgets import Static
from textual_plotext import PlotextPlot

from utils.mem_chart import plot_mem_chart


class RooflinePlot(PlotextPlot):
    """Roofline plot visualization widget."""

    DEFAULT_CSS = """
    RooflinePlot {
        padding: 1;
        width: auto;
        height: auto;
        background: $surface;
        color: $text;
        border: solid $accent;
    }
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.styles.height = "75%"
        self.styles.width = "75%"
        self.plot_initialized = False
        self.plt.theme("pro")

    def on_mount(self):
        self.refresh_plot()

    def refresh_plot(self):
        # Get the current size of this widget
        plot_width, plot_height = self.size

        # Configure the plot size
        self.plt.plot_size(plot_width, plot_height)
        self.create_roofline()
        self.plot_initialized = True

    def on_resize(self):
        if self.plot_initialized:
            self.refresh_plot()

    def create_roofline(self):
        """Generate the roofline plot data and visualization."""
        # Roofline model parameters
        peak_performance = 1000.0  # GFLOPS
        memory_bandwidth = 100.0  # GB/s

        # For memory-bound region (diagonal line)
        x_mem = [0.1, 0.5, 1.0, 5.0, 10.0]
        y_mem = [x * memory_bandwidth for x in x_mem]

        # For compute-bound region (horizontal line)
        x_comp = [10.0, 20.0, 50.0, 100.0]
        y_comp = [peak_performance] * len(x_comp)

        # Example workloads with safe values
        workloads = [
            (0.5, 45.0, "Workload A"),  # Memory bound
            (2.0, 180.0, "Workload B"),  # Memory bound
            (15.0, 950.0, "Workload C"),  # Compute bound
            (30.0, 980.0, "Workload D"),  # Compute bound
        ]

        # Clear the plot and set properties
        self.plt.clear_figure()
        self.plt.title("Roofline Model (🚧 Under Construction)")
        self.plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
        self.plt.ylabel("Performance (GFLOP/sec)")

        # Plot memory-bound and compute-bound lines
        self.plt.plot(x_mem, y_mem, label="Memory Bound")
        self.plt.plot(x_comp, y_comp, label="Compute Bound")

        # Add workload points
        workload_x = [w[0] for w in workloads]
        workload_y = [w[1] for w in workloads]
        workload_names = [w[2] for w in workloads]

        # Plot workload points one by one to avoid errors
        for i in range(len(workload_x)):
            self.plt.scatter([workload_x[i]], [workload_y[i]], label=workload_names[i])

        # Set a reasonable view range
        self.plt.xlim(0, 100)
        self.plt.ylim(0, 1100)

        # Draw the plot
        self.refresh()


class MemoryChart(Static):
    """Memory chart visualization widget."""

    DEFAULT_CSS = """
    MemoryChart {
        border: solid $accent;
        padding: 0;
        width: auto;
        height: auto;
        overflow-y: auto;
        overflow-x: auto;
        background: $surface;
        color: $text;
    }
    """

    def __init__(self, df: pd.DataFrame, **kwargs):
        """Initialize the memory chart."""
        super().__init__("", classes="mem-chart", **kwargs)
        self.df = df

        # Generate the chart content on initialization
        try:
            # Prepare data
            metric_dict = (
                self.df[["Metric", "Value"]].set_index("Metric").to_dict()["Value"]
            )

            # Capture stdout
            original_stdout = sys.stdout
            string_buffer = StringIO()
            sys.stdout = string_buffer

            try:
                # Generate the chart
                result = plot_mem_chart("", "per_kernel", metric_dict)
                stdout_output = string_buffer.getvalue()

                if stdout_output:
                    plot_str = stdout_output
                elif result:
                    plot_str = str(result)
                else:
                    plot_str = "No chart data generated"
            finally:
                sys.stdout = original_stdout

            # Set the content
            self.update(plot_str)

        except Exception as e:
            error_message = f"Memory chart error: {str(e)}\n{traceback.format_exc()}"
            self.update(f"Error: {str(e)}")
