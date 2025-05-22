##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tabulate import tabulate

from utils import parser
from utils.logger import console_log, console_warning
from utils.utils import convert_metric_id_to_panel_idx

HIDDEN_COLUMNS = ["Tips", "coll_level"]
HIDDEN_SECTIONS = [1900, 2000]


class PanelProcessor:
    """Handles processing of individual panels and their data sources."""

    def __init__(
        self, args, runs: Dict, comparable_columns: List, filter_panel_ids: List
    ):
        self.args = args
        self.runs = runs
        self.comparable_columns = comparable_columns
        self.filter_panel_ids = filter_panel_ids
        self.base_run, self.base_data = next(iter(runs.items()))

    def should_skip_table(self, table_config: Dict, panel_id: int) -> bool:
        """Determine if a table should be skipped based on filtering rules."""
        if (
            not self.args.filter_metrics
            and self.filter_panel_ids
            and table_config["id"] not in self.filter_panel_ids
            and panel_id not in self.filter_panel_ids
            and panel_id > 100
        ):

            table_id_str = self._get_table_id_string(table_config["id"])
            console_log(
                f"Not showing table not selected during profiling: "
                f"{table_id_str} {table_config['title']}"
            )
            return True
        return False

    def _get_table_id_string(self, table_id: int) -> str:
        """Generate formatted table ID string."""
        return f"{table_id // 100}.{table_id % 100}"

    def _should_include_column(self, header: str, table_type: str) -> bool:
        """Check if a column should be included in the output."""
        if header in HIDDEN_COLUMNS:
            return False

        if self.args.cols and table_type != "raw_csv_table":
            base_df = self.base_data.dfs[list(self.base_data.dfs.keys())[0]]
            return base_df.columns.get_loc(header) in self.args.cols

        return True

    def _process_kernel_name_column(
        self, base_df: pd.DataFrame, table_config: Dict
    ) -> pd.Series:
        """Process kernel name column with appropriate line wrapping."""
        if table_config["source"] == "pmc_kernel_top.csv":
            return base_df["Kernel_Name"].apply(lambda x: string_multiple_lines(x, 40, 3))
        else:
            return base_df["Kernel_Name"].apply(lambda x: string_multiple_lines(x, 80, 4))

    def _calculate_percentage_diff(
        self, base_values: pd.Series, current_values: pd.Series, header: str
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate percentage difference between base and current values."""
        # Convert to float, replacing empty strings with 0
        base_float = pd.to_numeric(base_values.replace("", 0), errors="coerce").fillna(0)
        current_float = pd.to_numeric(
            current_values.replace("", 0), errors="coerce"
        ).fillna(0)

        absolute_diff = (current_float - base_float).round(self.args.decimal)
        percentage_diff = (absolute_diff / base_float.replace(0, 1)) * 100

        if self.args.verbose >= 2:
            console_log("---------", header, percentage_diff)

        return absolute_diff, percentage_diff.round(self.args.decimal)

    def _format_value_with_percentage(
        self, current_values: pd.Series, percentage_diff: pd.Series
    ) -> pd.Series:
        """Format values with percentage change."""
        current_rounded = pd.to_numeric(current_values, errors="coerce").round(
            self.args.decimal
        )
        return current_rounded.astype(str) + " (" + percentage_diff.astype(str) + "%)"

    def _check_threshold_violations(
        self,
        header: str,
        percentage_diff: pd.Series,
        absolute_diff: pd.Series,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Check for threshold violations and add warnings."""
        if (
            header in ["Value", "Count", "Avg"]
            and percentage_diff.abs().gt(self.args.report_diff).any()
        ):

            df = df.copy()
            df["Abs Diff"] = absolute_diff

            if self.args.report_diff:
                violation_idx = percentage_diff.index[
                    percentage_diff.abs() > self.args.report_diff
                ]
                console_warning(
                    f"Dataframe diff exceeds {self.args.report_diff}% threshold requirement\n"
                    f"See metric {violation_idx.to_numpy()}"
                )
                console_warning(df)

        return df

    def _process_comparable_column(
        self, header: str, table_config: Dict, table_type: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Process columns that can be compared across runs."""
        for run, data in self.runs.items():
            current_df = data.dfs[table_config["id"]]

            if table_type == "raw_csv_table" or (
                table_type == "metric_table" and header not in HIDDEN_COLUMNS
            ):

                if run != self.base_run:
                    base_df = self.base_data.dfs[table_config["id"]]
                    absolute_diff, percentage_diff = self._calculate_percentage_diff(
                        base_df[header], current_df[header], header
                    )

                    formatted_values = self._format_value_with_percentage(
                        current_df[header], percentage_diff
                    )
                    df = pd.concat([df, formatted_values], axis=1)

                    df = self._check_threshold_violations(
                        header, percentage_diff, absolute_diff, df
                    )
                else:
                    # Base run - just round the values
                    base_df = self.base_data.dfs[table_config["id"]]
                    rounded_values = (
                        pd.to_numeric(base_df[header].replace("", 0), errors="coerce")
                        .round(self.args.decimal)
                        .fillna(base_df[header])
                    )

                    df = pd.concat([df, rounded_values], axis=1)

        return df

    def _process_non_comparable_column(
        self, header: str, table_config: Dict, table_type: str, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Process columns that cannot be compared across runs."""
        base_df = self.base_data.dfs[table_config["id"]]

        if (
            table_type == "raw_csv_table"
            and table_config["source"] in ["pmc_kernel_top.csv", "pmc_dispatch_info.csv"]
            and header == "Kernel_Name"
        ):

            adjusted_name = self._process_kernel_name_column(base_df, table_config)
            df = pd.concat([df, adjusted_name], axis=1)

        elif table_type == "raw_csv_table" and header == "Info":
            for run, data in self.runs.items():
                current_df = data.dfs[table_config["id"]]
                df = pd.concat([df, current_df[header]], axis=1)
        else:
            df = pd.concat([df, base_df[header]], axis=1)

        return df

    def process_data_source(self, data_source: Dict, panel_id: int) -> str:
        """Process a single data source and return formatted string."""
        content = ""

        for table_type, table_config in data_source.items():
            # If block filtering was used during analysis, then dont use profiling config
            # If block filtering was used in profiling config, only show those panels
            # If block filtering not used in profiling config, show all panels
            # Skip this table if table id or panel id is not present in block filters
            # However, always show panel id <= 100
            if self.should_skip_table(table_config, panel_id):
                continue

            base_df = self.base_data.dfs[table_config["id"]]
            df = pd.DataFrame(index=base_df.index)

            # Process each column
            for header in base_df.columns:
                if not self._should_include_column(header, table_type):
                    continue

                if header in self.comparable_columns:
                    df = self._process_comparable_column(
                        header, table_config, table_type, df
                    )
                else:
                    df = self._process_non_comparable_column(
                        header, table_config, table_type, df
                    )

            if not df.empty:
                table_content = self._format_table_output(df, table_config, table_type)
                if table_content:
                    content += table_content

        return content

    def _format_table_output(
        self, df: pd.DataFrame, table_config: Dict, table_type: str
    ) -> str:
        """Format table for output and handle file saving."""
        table_id_str = self._get_table_id_string(table_config["id"])

        # Check for empty columns
        if self._has_empty_columns(df):
            title = table_config.get("title", "")
            console_log(f"Not showing table with empty column(s): {table_id_str} {title}")
            return ""

        content = ""

        # Add title if present
        if table_config.get("title"):
            content += f"{table_id_str} {table_config['title']}\n"

        # Save to file if requested
        if self.args.df_file_dir:
            self._save_table_to_file(df, table_id_str, table_config)

        # Limit rows for certain table types
        if table_type == "raw_csv_table" and table_config.get("source") in [
            "pmc_kernel_top.csv",
            "pmc_dispatch_info.csv",
        ]:
            df = df.head(self.args.max_stat_num)

        # Determine if transpose is needed
        transpose = table_type != "raw_csv_table" and table_config.get(
            "columnwise", False
        )

        content += (
            get_table_string(df, transpose=transpose, decimal=self.args.decimal) + "\n"
        )

        return content

    def _has_empty_columns(self, df: pd.DataFrame) -> bool:
        """Check if dataframe has any completely empty columns."""
        return any(
            df.replace("", None).iloc[:, col_idx].isnull().all()
            for col_idx in range(len(df.columns))
        )

    def _save_table_to_file(
        self, df: pd.DataFrame, table_id_str: str, table_config: Dict
    ) -> None:
        """Save table to CSV file."""
        path = Path(self.args.df_file_dir)
        path.mkdir(exist_ok=True)

        if path.is_dir():
            filename = table_id_str
            if table_config.get("title"):
                filename += f"_{table_config['title']}"

            filepath = path / f"{filename.replace(' ', '_')}.csv"
            df.to_csv(filepath, index=False)


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


def show_all(args, runs, archConfigs, output, profiling_config):
    """
    Show all panels with their data in plain text mode.
    """
    comparable_columns = parser.build_comparable_columns(args.time_unit)
    filter_panel_ids = filter_panel_ids = _build_filter_panel_ids(profiling_config)

    processor = PanelProcessor(args, runs, comparable_columns, filter_panel_ids)

    # Process each panel
    for panel_id, panel in archConfigs.panel_configs.items():
        # Skip panels that don't support baseline comparison
        if panel_id in HIDDEN_SECTIONS:
            continue

        panel_content = ""  # store content of all data_source from one pannel

        for data_source in panel["data source"]:
            content = processor.process_data_source(data_source, panel_id)
            panel_content += content
        # Output panel if it has content
        if panel_content:
            _output_panel(panel_id, panel, panel_content, output)


def _build_filter_panel_ids(profiling_config: Dict) -> List[int]:
    """Build list of panel IDs to filter based on profiling config."""
    filter_blocks = profiling_config.get("filter_blocks", {})
    metric_sections = [
        name for name, type_val in filter_blocks.items() if type_val == "metric_id"
    ]

    return [convert_metric_id_to_panel_idx(section) for section in metric_sections]


def _output_panel(panel_id: int, panel_config: Dict, content: str, output) -> None:
    """Output formatted panel content."""
    print("\n" + "-" * 80, file=output)
    print(f"{panel_id // 100}. {panel_config['title']}", file=output)
    print(content, file=output)


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
