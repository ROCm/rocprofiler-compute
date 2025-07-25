import ast
import logging
import math
import re
import warnings
from collections import defaultdict
from datetime import datetime
from enum import Enum

import astunparse
import numpy as np
import pandas as pd

import config
from utils.parser import (
    CodeTransformer,
    supported_denom,
    to_avg,
    to_concat,
    to_int,
    to_max,
    to_median,
    to_min,
    to_mod,
    to_quantile,
    to_round,
    to_std,
)
from utils.utils import convert_metric_id_to_panel_idx


class LogLevel(str, Enum):
    """Log levels for consistent logging."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


class Logger:
    """Centralized logging handler for the application."""

    def __init__(self, output_area=None):
        """
        Initialize the logger.
        """
        self.output_area = output_area
        self._setup_logger()

    def _setup_logger(self):
        """
        Setup the Python logger with proper formatting.
        """
        self.logger = logging.getLogger("app")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def set_output_area(self, output_area):
        """
        Set or update the output area for displaying logs.
        """
        self.output_area = output_area

    def log(self, message, level=LogLevel.INFO, update_ui=True):
        """
        Log a message with the specified level.
        """
        level_map = {
            LogLevel.INFO: logging.INFO,
            LogLevel.SUCCESS: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
        }

        self.logger.log(level_map[level], message)

        timestamp = datetime.now().strftime("%H:%M:%S")

        if update_ui and self.output_area:
            if level == LogLevel.ERROR:
                formatted_msg = f"[{timestamp}] [ERROR] {message}"
            elif level == LogLevel.WARNING:
                formatted_msg = f"[{timestamp}] [WARNING] {message}"
            elif level == LogLevel.SUCCESS:
                formatted_msg = f"[{timestamp}] [SUCCESS] {message}"
            else:
                formatted_msg = f"[{timestamp}] [INFO] {message}"

            if hasattr(self.output_area, "text"):
                current_text = self.output_area.text
                self.output_area.text = (
                    f"{current_text}\n{formatted_msg}" if current_text else formatted_msg
                )
                # HACK: moving curson to end of outpu (Is there a better way to achieve this?)
                self.output_area.cursor_location = (999999, 0)

    def info(self, message, update_ui=True):
        self.log(message, LogLevel.INFO, update_ui)

    def success(self, message, update_ui=True):
        self.log(message, LogLevel.SUCCESS, update_ui)

    def warning(self, message, update_ui=True):
        self.log(message, LogLevel.WARNING, update_ui)

    def error(self, message, update_ui=True):
        self.log(message, LogLevel.ERROR, update_ui)


def build_eval_string(equation):
    """
    Convert user defined equation string to eval executable string
    For example,
        input: AVG(100  * SQ_ACTIVE_INST_SCA / ( GRBM_GUI_ACTIVE * $numCU ))
        output: to_avg(100 * kernel_data["SQ_ACTIVE_INST_SCA"] / \
                 (kernel_data["GRBM_GUI_ACTIVE"] * numCU))
        input: AVG(((TCC_EA_RDREQ_LEVEL_31 / TCC_EA_RDREQ_31) if (TCC_EA_RDREQ_31 != 0) else (0)))
        output: to_avg((kernel_data["TCC_EA_RDREQ_LEVEL_31"] / kernel_data["TCC_EA_RDREQ_31"]).where(kernel_data["TCC_EA_RDREQ_31"] != 0, 0))
        We can not handle the below for now,
        input: AVG((0 if (TCC_EA_RDREQ_31 == 0) else (TCC_EA_RDREQ_LEVEL_31 / TCC_EA_RDREQ_31)))
        But potential workaound is,
        output: to_avg(kernel_data["TCC_EA_RDREQ_31"].where(kernel_data["TCC_EA_RDREQ_31"] == 0, kernel_data["TCC_EA_RDREQ_LEVEL_31"] / kernel_data["TCC_EA_RDREQ_31"]))
    """

    if not equation:
        return ""

    s = str(equation)
    s = re.sub(r"\$", "ammolite__", s)

    ast_node = ast.parse(s)
    transformer = CodeTransformer()
    transformer.visit(ast_node)

    s = astunparse.unparse(ast_node)

    s = re.sub(r"\'\]\[(\d+)\]", r"[\g<1>]']", s)
    s = re.sub(r"raw_pmc_df\['(.*?)']", r'kernel_data.get("\1")', s)
    s = re.sub(
        r'kernel_data\.get\("([^"]*Timestamp)"\)', r'kernel_data.get("\1").iloc[0]', s
    )
    s = re.sub(r"\.where\(([^,]+),\s*([^)]+)\)", r", \1, \2)", s)
    s = re.sub(r"([^,\s]+), ([^,]+), ([^)]+)\)", r"safe_where(\1, \2, \3)", s)

    return s


def safe_where(series_data, condition, else_value):
    """
    Safely apply where condition, handling both Series and scalar conditions.
    """
    if pd.isna(condition) or condition is None:
        return else_value

    if np.isscalar(condition):
        if condition:
            return series_data
        else:
            return else_value

    return series_data.where(condition, else_value)


def evaluate_metric(
    metric_formula, kernel_data, kernel_idx, run_data, normalization_unit="per_kernel"
):
    """
    Evaluate a metric formula for a specific kernel using its performance counter data.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            if not metric_formula or metric_formula == "None":
                return None

            if "$denom" in metric_formula:
                metric_formula = metric_formula.replace(
                    "$denom", supported_denom.get(normalization_unit, "1")
                )

            context = {"__builtins__": {}}
            context["np"] = np
            context["pd"] = pd

            context.update(
                {
                    "to_min": to_min,
                    "to_max": to_max,
                    "to_avg": to_avg,
                    "to_median": to_median,
                    "to_std": to_std,
                    "to_int": to_int,
                    "to_round": to_round,
                    "to_quantile": to_quantile,
                    "to_mod": to_mod,
                    "to_concat": to_concat,
                    "safe_where": safe_where,
                }
            )

            if hasattr(run_data, "sys_info"):
                sys_info = run_data.sys_info

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

                # Add total_l2_chan
                if hasattr(sys_info, "total_l2_chan"):
                    total_l2_chan_value = _to_scalar(sys_info.total_l2_chan)
                    ammolite_vars["ammolite__total_l2_chan"] = (
                        float(total_l2_chan_value)
                        if not np.isnan(total_l2_chan_value)
                        else 0
                    )

                context.update(ammolite_vars)

            if "pmc_perf" in run_data.raw_pmc and hasattr(
                run_data.raw_pmc["pmc_perf"], "GRBM_GUI_ACTIVE"
            ):
                if "GRBM_GUI_ACTIVE" in kernel_data.columns:
                    grbm_gui_active = _to_scalar(kernel_data["GRBM_GUI_ACTIVE"].iloc[0])
                    if (
                        pd.notna(grbm_gui_active)
                        and context.get("ammolite__num_xcd", 0) > 0
                    ):
                        context["ammolite__GRBM_GUI_ACTIVE_PER_XCD"] = (
                            grbm_gui_active / context["ammolite__num_xcd"]
                        )
                    else:
                        context["ammolite__GRBM_GUI_ACTIVE_PER_XCD"] = 0
                else:
                    context["ammolite__GRBM_GUI_ACTIVE_PER_XCD"] = 0

                if "GRBM_COUNT" in kernel_data.columns:
                    grbm_count = _to_scalar(kernel_data["GRBM_COUNT"].iloc[0])
                    if pd.notna(grbm_count) and context.get("ammolite__num_xcd", 0) > 0:
                        context["ammolite__GRBM_COUNT_PER_XCD"] = (
                            grbm_count / context["ammolite__num_xcd"]
                        )
                    else:
                        context["ammolite__GRBM_COUNT_PER_XCD"] = 0
                else:
                    context["ammolite__GRBM_COUNT_PER_XCD"] = 0

                if "GRBM_SPI_BUSY" in kernel_data.columns:
                    grbm_spi_busy = _to_scalar(kernel_data["GRBM_SPI_BUSY"].iloc[0])
                    if (
                        pd.notna(grbm_spi_busy)
                        and context.get("ammolite__num_xcd", 0) > 0
                    ):
                        context["ammolite__GRBM_SPI_BUSY_PER_XCD"] = (
                            grbm_spi_busy / context["ammolite__num_xcd"]
                        )
                    else:
                        context["ammolite__GRBM_SPI_BUSY_PER_XCD"] = 0
                else:
                    context["ammolite__GRBM_SPI_BUSY_PER_XCD"] = 0

            # Calculate numActiveCUs for this kernel
            if (
                "SQ_BUSY_CU_CYCLES" in kernel_data.columns
                and context.get("ammolite__GRBM_GUI_ACTIVE_PER_XCD", 0) > 0
            ):
                sq_busy = _to_scalar(kernel_data["SQ_BUSY_CU_CYCLES"].iloc[0])
                if pd.notna(sq_busy):
                    max_waves = context.get("ammolite__max_waves_per_cu", 1)
                    cu_per_gpu = context.get("ammolite__cu_per_gpu", 1)
                    grbm_active = context.get("ammolite__GRBM_GUI_ACTIVE_PER_XCD", 1)

                    val = round((4 * sq_busy) / grbm_active, 0) if grbm_active > 0 else 0
                    active_cus = min(
                        (val / max_waves * 8) + min(val % max_waves, 8), cu_per_gpu
                    )
                    context["ammolite__numActiveCUs"] = int(active_cus)
                else:
                    context["ammolite__numActiveCUs"] = 0
            else:
                context["ammolite__numActiveCUs"] = 0

            # TODO: Calculate kernelBusyCycles for this kernel

            # Calculate hbmBandwidth for this kernel
            max_mclk = context.get("ammolite__max_mclk", 0)
            num_hbm_channels = context.get("ammolite__num_hbm_channels", 0)
            if max_mclk > 0 and num_hbm_channels > 0:
                context["ammolite__hbmBandwidth"] = max_mclk / (
                    1000 * 32 * num_hbm_channels
                )
            else:
                context["ammolite__hbmBandwidth"] = 0

            ammolite__se_per_gpu = context["ammolite__se_per_gpu"]
            ammolite__pipes_per_gpu = context["ammolite__pipes_per_gpu"]
            ammolite__cu_per_gpu = context["ammolite__cu_per_gpu"]
            ammolite__simd_per_cu = context["ammolite__simd_per_cu"]
            ammolite__sqc_per_gpu = context["ammolite__sqc_per_gpu"]
            ammolite__lds_banks_per_cu = context["ammolite__lds_banks_per_cu"]
            ammolite__cur_sclk = context["ammolite__cur_sclk"]
            ammolite__cur_mclk = context["ammolite__cur_mclk"]
            ammolite__max_mclk = context["ammolite__max_mclk"]
            ammolite__max_sclk = context["ammolite__max_sclk"]
            ammolite__max_waves_per_cu = context["ammolite__max_waves_per_cu"]
            ammolite__num_hbm_channels = context["ammolite__num_hbm_channels"]
            ammolite__num_xcd = context["ammolite__num_xcd"]
            ammolite__wave_size = context["ammolite__wave_size"]
            ammolite__total_l2_chan = context["ammolite__total_l2_chan"]
            ammolite__GRBM_GUI_ACTIVE_PER_XCD = context[
                "ammolite__GRBM_GUI_ACTIVE_PER_XCD"
            ]
            ammolite__GRBM_COUNT_PER_XCD = context["ammolite__GRBM_COUNT_PER_XCD"]
            ammolite__GRBM_SPI_BUSY_PER_XCD = context["ammolite__GRBM_SPI_BUSY_PER_XCD"]
            ammolite__numActiveCUs = context["ammolite__numActiveCUs"]
            ammolite__hbmBandwidth = context["ammolite__hbmBandwidth"]

            s = build_eval_string(metric_formula)

            try:
                result = eval(compile(s, "<string>", "eval"))
                if hasattr(result, "item"):
                    result = result.item()
                if pd.isna(_to_scalar(result)):
                    return None
                return float(result)
            except TypeError as e:
                result = None
            except AttributeError as ae:
                if ae == "'NoneType' object has no attribute 'get'":
                    result = None
            except Exception as e:
                print(f"something is wrong 6: {str(e)}")
                print(f"Failed expression: {s}")
                return None
        except Exception as e:
            return None


def process_per_kernel_panels_to_dataframes(
    args, runs, archConfigs, profiling_config, roof_plot=None, debug=False
):
    """
    Process panel data into pandas DataFrames.
    Returns a nested dictionary structure with DataFrames and tui_style information.

    Returns:
        Dict[str, Dict[str, Dict[str, Any]]]: Nested structure {
            "kernel_name": {
                "section_name": {
                    "subsection_name": {
                        "df": DataFrame,
                        "tui_style": dict or None
                    }
                }
            }
        }
    """
    result_structure = defaultdict(dict)

    comparable_columns = build_comparable_columns(args.time_unit)
    filter_panel_ids = profiling_config.get("filter_blocks", [])
    if isinstance(filter_panel_ids, dict):
        # For backward compatibility
        filter_panel_ids = [
            name for name, type in filter_panel_ids.items() if type == "metric_id"
        ]
    filter_panel_ids = [
        int(convert_metric_id_to_panel_info(metric_id)[0])
        for metric_id in filter_panel_ids
    ]

    run_name, run_data = next(iter(runs.items()))

    if (
        "pmc_perf" not in run_data.raw_pmc
        or "Kernel_Name" not in run_data.raw_pmc["pmc_perf"]
    ):
        return result_structure

    kernel_names = run_data.raw_pmc["pmc_perf"]["Kernel_Name"].tolist()

    for kernel_idx, kernel_name in enumerate(kernel_names):
        result_structure[kernel_name] = {}

        kernel_perf_data = run_data.raw_pmc["pmc_perf"].iloc[[kernel_idx]]

        for panel_id, panel in archConfigs.panel_configs.items():
            if panel_id in config.HIDDEN_SECTIONS:
                continue

            if (
                not args.filter_metrics
                and filter_panel_ids
                and panel_id not in filter_panel_ids
                and panel_id > 100
            ):
                continue

            section_name = f"{panel_id // 100}. {panel['title']}"

            if section_name not in result_structure[kernel_name]:
                result_structure[kernel_name][section_name] = {}

            for data_source in panel["data source"]:
                for data_type, table_config in data_source.items():
                    if not isinstance(table_config, dict) or "id" not in table_config:
                        continue
                    if (
                        data_type != "metric_table"
                        or "metric" not in table_config
                        or "header" not in table_config
                    ):
                        continue

                    table_id = table_config["id"]
                    table_title = table_config.get("title", "")
                    table_id_str = f"{table_id // 100}.{table_id % 100}"

                    if table_title:
                        subsection_name = f"{table_id_str} {table_title}"
                    else:
                        subsection_name = table_id_str

                    df = generate_subsection_df(
                        table_config, kernel_perf_data, kernel_idx, run_data, debug=debug
                    )

                    if df is not None and not df.empty:
                        if table_config.get("columnwise", False) == True:
                            df = df.transpose()

                        result_structure[kernel_name][section_name][subsection_name] = {
                            "df": df,
                            "tui_style": table_config.get("tui_style", None),
                        }

            if (
                section_name in result_structure[kernel_name]
                and not result_structure[kernel_name][section_name]
            ):
                del result_structure[kernel_name][section_name]

    for kernel_section in list(result_structure.keys()):
        if not result_structure[kernel_section]:
            del result_structure[kernel_section]

    return result_structure


def generate_subsection_df(
    table_config, kernel_perf_data, kernel_idx, run_data, debug=False
):
    """
    Generate a DataFrame for a subsection based on table_config.
    """

    if "metric" not in table_config or "header" not in table_config:
        return None

    header = table_config["header"]
    metrics = table_config["metric"]
    table_id = table_config.get("id", 0)

    df_data = []

    for metric_idx, (metric_key, metric_values) in enumerate(metrics.items()):
        major_id = table_id // 100
        minor_id = table_id % 100
        metric_id = f"{major_id}.{minor_id}.{metric_idx}"

        row_data = {"Metric_ID": metric_id}

        base_expression_value = None
        if isinstance(metric_values, dict):
            expr_keys = ["expr", "expression", "value", "formula"]
            for expr_key in expr_keys:
                if expr_key in metric_values:
                    formula = metric_values[expr_key]

                    if formula is not None:
                        base_expression_value = evaluate_metric(
                            formula, kernel_perf_data, kernel_idx, run_data, "per_kernel"
                        )

                        if base_expression_value is not None:
                            base_expression_value = _round2(base_expression_value)
                    break

        for header_key, column_name in header.items():
            if header_key == "metric":
                try:
                    numeric_value = float(metric_key)
                    row_data[column_name] = numeric_value
                except ValueError:
                    row_data[column_name] = metric_key
            elif header_key in ["unit", "units", "tips"] and isinstance(
                metric_values, dict
            ):
                if header_key in metric_values:
                    value = metric_values[header_key]
                    if header_key == "unit" and isinstance(value, str):
                        value = value.capitalize()
                    row_data[column_name] = value
                else:
                    row_data[column_name] = None
            elif header_key in ["peak"]:
                # TODO
                continue
            elif header_key in metric_values:
                formula = metric_values[header_key]

                if formula is None:
                    evaluated_value = None
                else:
                    evaluated_value = evaluate_metric(
                        formula, kernel_perf_data, kernel_idx, run_data, "per_kernel"
                    )

                    if evaluated_value is not None:
                        evaluated_value = _round2(evaluated_value)

                row_data[column_name] = evaluated_value
            else:
                if base_expression_value is not None and column_name in [
                    "Min",
                    "Q1",
                    "Median",
                    "Q3",
                    "Max",
                    "Expression",
                ]:
                    row_data[column_name] = base_expression_value
                else:
                    row_data[column_name] = None
        df_data.append(row_data)

    if df_data:
        df = pd.DataFrame(df_data)
        if "Metric_ID" in df.columns:
            df.set_index("Metric_ID", inplace=True)
        return df

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


def convert_metric_id_to_panel_info(metric_id):
    """
    Convert metric id into panel information.
    Output is a tuples of the form (file_id, panel_id, metric_id).

    For example:

    Input: "2"
    Output: ("0200", None, None)

    Input: "11"
    Output: ("1100", None, None)

    Input: "11.1"
    Output: ("1100", 1101, None)

    Input: "11.1.1"
    Output: ("1100", 1101, 1)

    Raises exception for invalid metric id.
    """
    tokens = metric_id.split(".")
    if 0 < len(tokens) < 4:
        # File id
        file_id = str(int(tokens[0]))
        # 4 -> 04
        if len(file_id) < 2:
            file_id = f"0{file_id}"
        # Multiply integer by 100
        file_id = f"{file_id}00"
        # Panel id
        if len(tokens) > 1:
            panel_id = int(tokens[0]) * 100
            panel_id += int(tokens[1])
        else:
            panel_id = None
        # Metric id
        if len(tokens) > 2:
            metric_id = int(tokens[2])
        else:
            metric_id = None
        return (file_id, panel_id, metric_id)
    else:
        return None


def get_top_kernels_and_dispatch_ids(runs):
    if not runs:
        return None

    base_run = next(iter(runs.values()))
    if not hasattr(base_run, "dfs"):
        return None

    top_kernel_df = base_run.dfs.get(1)
    dispatch_id_df = base_run.dfs.get(2)

    if top_kernel_df is None or dispatch_id_df is None:
        return None

    merged_df = pd.merge(
        top_kernel_df, dispatch_id_df, on="Kernel_Name", how="outer"
    ).sort_values("Pct", ascending=False)
    return merged_df.to_dict("records")


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
    return isinstance(v, (float, np.floating)) and math.isnan(v)


def _round2(v):
    return None if _is_na(v) else round(float(v), 2)
