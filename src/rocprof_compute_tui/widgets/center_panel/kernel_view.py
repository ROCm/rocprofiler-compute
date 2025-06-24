"""
Panel Widget Modules
-------------------
Contains the panel widgets used in the main layout.
"""

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate
from textual.containers import ScrollableContainer
from textual.widgets import Label

from config import HIDDEN_COLUMNS, HIDDEN_SECTIONS
from utils import mem_chart, parser
from utils.logger import console_log, console_warning
from utils.utils import convert_metric_id_to_panel_idx


class KernelView(ScrollableContainer):
    """Center panel with analysis results."""

    def __init__(self):
        super().__init__(id="kernel-view")
        self.dfs = {}

    def compose(self):
        """
        Compose the initial center panel state.
        """
        yield Label(
            "Open a workload directory to run analysis and view results",
            classes="placeholder",
        )


def string_multiple_lines(source, width, max_rows):
    """
    Adjust string with multiple lines by inserting '\n'
    """
    idx = 0
    lines = []
    while idx < len(source) and len(lines) < max_rows:
        lines.append(source[idx : idx + width])
        idx += width

    if idx < len(source):
        last = lines[-1]
        lines[-1] = last[0:-3] + "..."
    return "\n".join(lines)


def get_table_string(df, transpose=False, decimal=2):
    return tabulate(
        df.transpose() if transpose else df,
        headers="keys",
        tablefmt="fancy_grid",
        floatfmt="." + str(decimal) + "f",
    )


def evaluate_metric(metric_formula, kernel_data, kernel_idx, run_data):
    """
    Evaluate a metric formula for a specific kernel using its performance counter data.

    Args:
        metric_formula: The formula/expression string for the metric
        kernel_data: Performance counter data for the specific kernel (pandas Series)
        kernel_idx: Index of the kernel in the dataframe
        run_data: The run data object containing all necessary information

    Returns:
        The evaluated metric value
    """
    try:
        import re

        # Create evaluation context
        context = {"__builtins__": {}}

        # Add numpy functions that might be used in formulas
        context["np"] = np
        context["pd"] = pd

        # Add SQL-like functions that work with scalars (since we're evaluating per kernel)
        context["ROUND"] = lambda x, decimals=0: (
            round(float(x), int(decimals)) if not pd.isna(_to_scalar(x)) else None
        )
        context["AVG"] = lambda x: float(
            x
        )  # For single kernel, AVG is just the value itself
        context["SUM"] = lambda x: float(
            x
        )  # For single kernel, SUM is just the value itself
        context["MIN"] = lambda x: float(
            x
        )  # For single kernel, MIN is just the value itself
        context["MAX"] = lambda x: float(
            x
        )  # For single kernel, MAX is just the value itself

        # Extract system info variables (ammolite__ prefixed variables)
        if hasattr(run_data, "sys_info"):
            sys_info = run_data.sys_info

            # Add all ammolite__ variables from sys_info
            ammolite_vars = {
                "ammolite__se_per_gpu": (
                    int(_to_scalar(sys_info.se_per_gpu))
                    if hasattr(sys_info, "se_per_gpu")
                    and not np.isnan(_to_scalar(sys_info.se_per_gpu))
                    else 0
                ),
                "ammolite__pipes_per_gpu": (
                    int(_to_scalar(sys_info.pipes_per_gpu))
                    if hasattr(sys_info, "pipes_per_gpu")
                    and not np.isnan(_to_scalar(sys_info.pipes_per_gpu))
                    else 0
                ),
                "ammolite__cu_per_gpu": (
                    int(_to_scalar(sys_info.cu_per_gpu))
                    if hasattr(sys_info, "cu_per_gpu")
                    and not np.isnan(_to_scalar(sys_info.cu_per_gpu))
                    else 0
                ),
                "ammolite__simd_per_cu": (
                    int(_to_scalar(sys_info.simd_per_cu))
                    if hasattr(sys_info, "simd_per_cu")
                    and not np.isnan(_to_scalar(sys_info.simd_per_cu))
                    else 0
                ),
                "ammolite__sqc_per_gpu": (
                    int(_to_scalar(sys_info.sqc_per_gpu))
                    if hasattr(sys_info, "sqc_per_gpu")
                    and not np.isnan(_to_scalar(sys_info.sqc_per_gpu))
                    else 0
                ),
                "ammolite__lds_banks_per_cu": (
                    int(_to_scalar(sys_info.lds_banks_per_cu))
                    if hasattr(sys_info, "lds_banks_per_cu")
                    and not np.isnan(_to_scalar(sys_info.lds_banks_per_cu))
                    else 0
                ),
                "ammolite__cur_sclk": (
                    float(_to_scalar(sys_info.cur_sclk))
                    if hasattr(sys_info, "cur_sclk")
                    and not np.isnan(_to_scalar(sys_info.cur_sclk))
                    else 0
                ),
                "ammolite__cur_mclk": (
                    float(_to_scalar(sys_info.cur_mclk))
                    if hasattr(sys_info, "cur_mclk")
                    and not np.isnan(_to_scalar(sys_info.cur_mclk))
                    else 0
                ),
                "ammolite__max_mclk": (
                    float(_to_scalar(sys_info.max_mclk))
                    if hasattr(sys_info, "max_mclk")
                    and not np.isnan(_to_scalar(sys_info.max_mclk))
                    else 0
                ),
                "ammolite__max_sclk": (
                    float(_to_scalar(sys_info.max_sclk))
                    if hasattr(sys_info, "max_sclk")
                    and not np.isnan(_to_scalar(sys_info.max_sclk))
                    else 0
                ),
                "ammolite__max_waves_per_cu": (
                    int(_to_scalar(sys_info.max_waves_per_cu))
                    if hasattr(sys_info, "max_waves_per_cu")
                    and not np.isnan(_to_scalar(sys_info.max_waves_per_cu))
                    else 0
                ),
                "ammolite__num_hbm_channels": (
                    float(_to_scalar(sys_info.num_hbm_channels))
                    if hasattr(sys_info, "num_hbm_channels")
                    and not np.isnan(_to_scalar(sys_info.num_hbm_channels))
                    else 0
                ),
                "ammolite__num_xcd": (
                    int(_to_scalar(sys_info.num_xcd))
                    if hasattr(sys_info, "num_xcd")
                    and not np.isnan(_to_scalar(sys_info.num_xcd))
                    else 0
                ),
                "ammolite__wave_size": (
                    int(_to_scalar(sys_info.wave_size))
                    if hasattr(sys_info, "wave_size")
                    and not np.isnan(_to_scalar(sys_info.wave_size))
                    else 0
                ),
            }
            context.update(ammolite_vars)

            # Add calculated variables (if they exist in run_data)
            if hasattr(sys_info, "total_l2_chan"):
                context["ammolite__total_l2_chan"] = (
                    float(sys_info.total_l2_chan)
                    if not np.isnan(_to_scalar(sys_info.total_l2_chan))
                    else 0
                )

        # Add built-in derived variables if they exist
        if hasattr(run_data, "ammolite__build_in"):
            for key, value in run_data.ammolite__build_in.items():
                if value is not None:
                    # For individual kernel evaluation, use scalar values
                    if isinstance(value, pd.Series):
                        # If it's a series, get the value for this kernel
                        try:
                            val = value.iloc[kernel_idx]
                            # Ensure it's a scalar
                            if hasattr(val, "item"):
                                val = val.item()
                            context[f"ammolite__{key}"] = (
                                float(val) if not pd.isna(_to_scalar(val)) else 0
                            )
                        except:
                            context[f"ammolite__{key}"] = 0
                    else:
                        context[f"ammolite__{key}"] = (
                            float(value) if not pd.isna(_to_scalar(value)) else 0
                        )

        # Replace $ prefixed variables with ammolite__ prefixed ones
        modified_formula = metric_formula
        dollar_var_pattern = r"\$(\w+)"
        dollar_vars = re.findall(dollar_var_pattern, modified_formula)
        for var_name in dollar_vars:
            ammolite_var_name = f"ammolite__{var_name}"
            if ammolite_var_name in context:
                modified_formula = modified_formula.replace(
                    f"${var_name}", ammolite_var_name
                )
            else:
                console_warning(f"Built-in variable ${var_name} not found in context")
                return None

        # Handle raw_pmc_df references in the formula
        raw_pmc_pattern = r"raw_pmc_df\['pmc_perf'\]\['(\w+)'\]"
        matches = re.findall(raw_pmc_pattern, modified_formula)

        for counter_name in matches:
            if counter_name in kernel_data.index:
                value = kernel_data[counter_name]
                value = _to_scalar(value)
                # Ensure value is scalar
                if hasattr(value, "item"):
                    value = value.item()
                if pd.notna(value):
                    modified_formula = modified_formula.replace(
                        f"raw_pmc_df['pmc_perf']['{counter_name}']", str(float(value))
                    )
                else:
                    modified_formula = modified_formula.replace(
                        f"raw_pmc_df['pmc_perf']['{counter_name}']", "0.0"
                    )
            else:
                console_warning(
                    f"Performance counter '{counter_name}' not found in kernel data"
                )
                return None

        # Also handle direct counter references (without raw_pmc_df prefix)
        # First, get all variable names from the formula
        remaining_vars = set()
        try:
            # Use regex to find all word boundaries that look like variable names
            var_pattern = r"\b([A-Za-z_]\w*)\b"
            all_vars = re.findall(var_pattern, modified_formula)

            for var_name in all_vars:
                # Skip if it's already in context, a function name, or starts with ammolite__
                if (
                    var_name not in context
                    and var_name not in ["ROUND", "AVG", "SUM", "MIN", "MAX", "np", "pd"]
                    and not var_name.startswith("ammolite__")
                ):
                    remaining_vars.add(var_name)
        except:
            pass

        # Add counter values to context
        for var_name in remaining_vars:
            if var_name in kernel_data.index:
                value = kernel_data[var_name]
                value = _to_scalar(value)
                # Ensure value is scalar
                if hasattr(value, "item"):
                    value = value.item()
                if pd.notna(value):
                    context[var_name] = float(value)
                else:
                    context[var_name] = 0.0

        # Debug: print the modified formula and context
        # console_log(f"Modified formula: {modified_formula}")
        # console_log(f"Context keys: {list(context.keys())}")

        # Compile and evaluate the modified formula
        compiled_expr = compile(modified_formula, "<metric>", "eval")
        result = eval(compiled_expr, {"__builtins__": {}}, context)

        # Ensure result is scalar
        if hasattr(result, "item"):
            result = result.item()

        # Handle NaN results
        if pd.isna(_to_scalar(result)):
            return None

        return float(result)

    except ZeroDivisionError:
        return float("inf")  # or 0.0, depending on your preference
    except Exception as e:
        console_warning(f"Failed to evaluate metric '{metric_formula}': {e}")
        return None


def show_all(args, runs, archConfigs, output, profiling_config, roof_plot=None):
    """
    Show all panels with metric data for individual kernels.
    """
    filter_panel_ids = [
        convert_metric_id_to_panel_idx(section)
        for section in [
            name
            for name, type in profiling_config.get("filter_blocks", {}).items()
            if type == "metric_id"
        ]
    ]

    # Get the first (and only) run to extract kernel information
    run_name, run_data = next(iter(runs.items()))

    # Get list of kernels
    if (
        "pmc_perf" not in run_data.raw_pmc
        or "Kernel_Name" not in run_data.raw_pmc["pmc_perf"]
    ):
        console_warning("No kernel data found in pmc_perf")
        return

    kernel_names = run_data.raw_pmc["pmc_perf"]["Kernel_Name"].tolist()

    # Iterate through kernels
    for kernel_idx, kernel_name in enumerate(kernel_names):
        print(f"\n{'='*80}", file=output)
        print(f"Kernel {kernel_idx}: {kernel_name}", file=output)
        print(f"{'='*80}", file=output)

        # Get performance counter data for this specific kernel
        kernel_perf_data = run_data.raw_pmc["pmc_perf"].iloc[kernel_idx]

        for panel_id, panel in archConfigs.panel_configs.items():
            # show roofline
            if panel_id == 400 and roof_plot:
                show_roof_plot(roof_plot)

            # Skip panels that don't support baseline comparison
            if panel_id in HIDDEN_SECTIONS:
                continue

            # Skip if panel filtering is active and this panel is not included
            if (
                not args.filter_metrics
                and filter_panel_ids
                and panel_id not in filter_panel_ids
                and panel_id > 100
            ):
                continue

            ss = ""  # store content of all data_source from one panel

            for data_source in panel["data source"]:
                for type, table_config in data_source.items():
                    # Skip if table filtering is active and this table is not included
                    if (
                        not args.filter_metrics
                        and filter_panel_ids
                        and table_config["id"] not in filter_panel_ids
                        and panel_id not in filter_panel_ids
                        and panel_id > 100
                    ):
                        table_id_str = (
                            str(table_config["id"] // 100)
                            + "."
                            + str(table_config["id"] % 100)
                        )
                        console_log(
                            f"Not showing table not selected during profiling: {table_id_str} {table_config['title']}"
                        )
                        continue

                    # For metric tables, calculate metrics for this kernel
                    if type == "metric_table" and "metric" in table_config:
                        df_data = []

                        for metric_name, metric_info in table_config["metric"].items():
                            if "value" in metric_info:
                                # Calculate metric value for this kernel
                                metric_value = evaluate_metric(
                                    metric_info["value"],
                                    kernel_perf_data,
                                    kernel_idx,
                                    run_data,
                                )
                                metric_value = _round2(metric_value)

                                if metric_value is None:
                                    continue

                                # Add metric to dataframe
                                row_data = {
                                    "Metric": metric_name,
                                    "Value": (
                                        metric_value
                                    ),
                                }

                                df_data.append(row_data)

                        if df_data:
                            df = pd.DataFrame(df_data)

                            # Build table subtitle
                            table_id_str = (
                                str(table_config["id"] // 100)
                                + "."
                                + str(table_config["id"] % 100)
                            )

                            if "title" in table_config and table_config["title"]:
                                ss += table_id_str + " " + table_config["title"] + "\n"

                            # Handle transpose if needed
                            transpose = (
                                "columnwise" in table_config
                                and table_config["columnwise"] == True
                            )

                            # Add table to output
                            ss += (
                                get_table_string(
                                    df, transpose=transpose, decimal=args.decimal
                                )
                                + "\n"
                            )

                    # For raw CSV tables, we might want to show kernel-specific data if available
                    elif type == "raw_csv_table":
                        # This would need to be implemented based on your specific raw CSV structure
                        # For now, we'll skip raw CSV tables for individual kernels
                        pass

            # Print panel content if any
            if ss:
                print("\n" + "-" * 80, file=output)
                print(str(panel_id // 100) + ". " + panel["title"], file=output)
                print(ss, file=output)

        # Optional: limit number of kernels displayed
        if args.max_stat_num and kernel_idx >= args.max_stat_num - 1:
            print(f"\n(Showing first {args.max_stat_num} kernels only)", file=output)
            break


def show_roof_plot(roof_plot):
    # TODO: short term solution to display roofline plot
    print("\n" + "-" * 80)
    print("4. Roofline")
    print("4.1 Roofline")
    print(roof_plot)


def show_kernel_stats(args, runs, archConfigs, output):
    """
    Show the kernels and dispatches from "Top Stats" section.
    """

    df = pd.DataFrame()
    for panel_id, panel in archConfigs.panel_configs.items():
        for data_source in panel["data source"]:
            for type, table_config in data_source.items():
                for run, data in runs.items():
                    df = pd.DataFrame()
                    single_df = data.dfs[table_config["id"]]
                    # NB:
                    #   For pmc_kernel_top.csv, have to sort here if not
                    #   sorted when load_table_data.
                    if table_config["id"] == 1:
                        print("\n" + "-" * 80, file=output)
                        print(
                            "Detected Kernels (sorted descending by duration)",
                            file=output,
                        )
                        df = pd.concat([df, single_df["Kernel_Name"]], axis=1)

                    if table_config["id"] == 2:
                        print("\n" + "-" * 80, file=output)
                        print("Dispatch list", file=output)
                        df = single_df

                    print(
                        get_table_string(df, transpose=False, decimal=args.decimal),
                        file=output,
                    )


def _to_scalar(v):
    if isinstance(v, pd.Series):
        v = v.dropna()
        return v.iloc[0] if len(v) else np.nan
    if isinstance(v, (np.ndarray, list, tuple)):
        return v[0] if len(v) else np.nan
    return v


def _is_na(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().upper() == "N/A"
    # covers float('nan'), numpy.nan, pandas NA scalars, etc.
    return isinstance(v, (float, np.floating)) and math.isnan(v)


def _round2(v):
    return None if _is_na(v) else round(float(v), 2)
