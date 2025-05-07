# The following code borrows heavily from the Plotext readme files, and as
# such has some variables names and the like that would generally annoy
# pylint; so here we ask pylint to ease up for this particular file.
#
# Also, because it's common with Textual to introduce a property in
# `on_mount`, we ask pylint to ease up on that too.
#
# pylint:disable=invalid-name, attribute-defined-outside-init

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Label, Rule
from textual_plotext import PlotextPlot


class ExamplesPane(VerticalScroll):
    """Base class for a pane of examples."""

    DEFAULT_CSS = """
    ExamplesPane {
        scrollbar-gutter: stable;
    }

    ExamplesPane Label {
        margin: 1;
        height: 50%;
        width: 1fr;
        text-align: center;
    }
    """

    def examples(self, source: str, examples: list[PlotextPlot]) -> ComposeResult:
        """Provide the composed examples.

        Args:
            source: The source for the examples within the Plotext readme files.
            examples: A list of the example widgets.
        """
        yield Label(
            "Examples taken from https://github.com/piccolomo/"
            f"plotext/blob/master/readme/{source}.md"
        )
        for example in examples:
            yield example
            yield Rule()


class BasicPlots(ExamplesPane):
    """Examples from the basic section of the Plotext documentation."""

    class ScatterPlot(PlotextPlot):
        """https://github.com/piccolomo/plotext/blob/master/readme/basic.md#scatter-plot"""

        def on_mount(self) -> None:
            """Set up the plot."""
            self.plt.scatter(self.plt.sin())
            self.plt.title("Scatter Plot")

    class LogPlot(PlotextPlot):
        """https://github.com/piccolomo/plotext/blob/master/readme/basic.md#log-plot"""

        def on_mount(self) -> None:
            """Set up the plot."""
            self.plt.plot(self.plt.sin(periods=2, length=10**4))
            self.plt.xscale("log")
            self.plt.yscale("linear")
            self.plt.grid(0, 1)
            self.plt.title("Logarithmic Plot")
            self.plt.xlabel("logarithmic scale")
            self.plt.ylabel("linear scale")
            # A slightly hacky workaround for what seems to be a bug in
            # Plotext itself when it comes to log scales. We force a build
            # of the data...
            _ = self.plt.build()
            # ...then put the scale back to linear so it doesn't try and
            # apply log again and again on each render.
            self.plt.xscale("linear")

    class MultipleDataSets(PlotextPlot):
        """https://github.com/piccolomo/plotext/blob/master/readme/basic.md#multiple-data-sets"""

        def on_mount(self) -> None:
            """Set up the plot."""
            self.plt.plot(self.plt.sin(), label="plot")
            self.plt.scatter(self.plt.sin(phase=-1), label="scatter")
            self.plt.title("Multiple Data Set")

    def compose(self) -> ComposeResult:
        """Compose the child widgets."""
        return self.examples(
            "basic",
            [
                self.ScatterPlot(),
                self.LogPlot(),
                self.MultipleDataSets(),
            ],
        )


class BarPlots(ExamplesPane):
    """Examples from the bar plots section of the Plotext documentation."""

    class VerticalBarPlot(PlotextPlot):
        """https://github.com/piccolomo/plotext/blob/master/readme/bar.md#vertical-bar-plot"""

        def on_mount(self) -> None:
            """Set up the plot."""
            pizzas = ["Sausage", "Pepperoni", "Mushrooms", "Cheese", "Chicken", "Beef"]
            percentages = [14, 36, 11, 8, 7, 4]
            self.plt.bar(pizzas, percentages)
            self.plt.title("Most Favored Pizzas in the World")

    class HorizontalBarPlot(PlotextPlot):
        """https://github.com/piccolomo/plotext/blob/master/readme/bar.md#horizontal-bar-plot"""

        def on_mount(self) -> None:
            """Set up the plot."""
            pizzas = ["Sausage", "Pepperoni", "Mushrooms", "Cheese", "Chicken", "Beef"]
            percentages = [14, 36, 11, 8, 7, 4]
            self.plt.bar(
                pizzas, percentages, orientation="horizontal", width=3 / 5
            )  # or in short orientation = 'h'
            self.plt.title("Most Favoured Pizzas in the World")

    class MultipleBarPlot(PlotextPlot):
        """https://github.com/piccolomo/plotext/blob/master/readme/bar.md#multiple-bar-plot"""

        def on_mount(self) -> None:
            """Set up the plot."""
            pizzas = ["Sausage", "Pepperoni", "Mushrooms", "Cheese", "Chicken", "Beef"]
            male_percentages = [14, 36, 11, 8, 7, 4]
            female_percentages = [12, 20, 35, 15, 2, 1]
            self.plt.multiple_bar(
                pizzas, [male_percentages, female_percentages]
            )  # , labels = ["men", "women"])
            self.plt.title("Most Favored Pizzas in the World by Gender")

    def compose(self) -> ComposeResult:
        """Compose the child widgets."""
        return self.examples(
            "bar",
            [
                self.VerticalBarPlot(),
                self.HorizontalBarPlot(),
                self.MultipleBarPlot(),
            ],
        )


class SpecialPlots(ExamplesPane):
    """Examples from the special plots section of the Plotext documentation."""

    class StreamingDataPlot(PlotextPlot):
        """https://github.com/piccolomo/plotext/blob/master/readme/special.md#streaming-data"""

        def on_mount(self) -> None:
            """Set up the initial conditions for the 'streaming' data."""
            self.frame = 0
            self.plt.title("Streaming Data")
            self.set_interval(0.25, self.plot)

        def plot(self) -> None:
            """Plot the current frame of the stream."""
            self.plt.clear_data()
            self.plt.scatter(
                self.plt.sin(periods=2, length=1_000, phase=(2 * self.frame) / 50)
            )
            self.refresh()
            self.frame += 1

    def compose(self) -> ComposeResult:
        """Compose the child widgets."""
        return self.examples(
            "special",
            [
                self.StreamingDataPlot(),
            ],
        )


class RooflinePlot(PlotextPlot):

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
