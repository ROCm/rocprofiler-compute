import logging
import os
import re
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd

from utils.rocprof_compute_cmd import RocprofRunner


class LogLevel(str, Enum):
    """Log levels for consistent logging."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"  # Maintained for UI compatibility


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
            LogLevel.SUCCESS: logging.INFO,  # Success is treated as INFO in Python logging
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
        }

        # Log to Python logger
        self.logger.log(level_map[level], message)

        timestamp = datetime.now().strftime("%H:%M:%S")

        if update_ui and self.output_area:
            if level == LogLevel.ERROR:
                formatted_msg = f"[{timestamp}] [ERROR] {message}"
            elif level == LogLevel.WARNING:
                formatted_msg = f"[{timestamp}] [WARNING] {message}"
            elif level == LogLevel.SUCCESS:
                formatted_msg = f"[{timestamp}] [SUCCESS] {message}"
            else:  # INFO
                formatted_msg = f"[{timestamp}] [INFO] {message}"

            # Append to output area
            if hasattr(self.output_area, "text"):
                current_text = self.output_area.text
                self.output_area.text = (
                    f"{current_text}\n{formatted_msg}" if current_text else formatted_msg
                )

    def info(self, message, update_ui=True):
        self.log(message, LogLevel.INFO, update_ui)

    def success(self, message, update_ui=True):
        self.log(message, LogLevel.SUCCESS, update_ui)

    def warning(self, message, update_ui=True):
        self.log(message, LogLevel.WARNING, update_ui)

    def error(self, message, update_ui=True):
        self.log(message, LogLevel.ERROR, update_ui)


def split_table_line(line):
    """
    Splits a table row line into a list of cell strings (trimmed). For example:

    │    │ Kernel_Name                              │   Count │ ...
    """

    cells = line.split("│")
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return [cell.strip() for cell in cells]


def parse_ascii_table(table_lines):
    """
    Given a list of lines belonging to one ASCII table (including border rows),
    return a tuple (header, data_rows) where header is a list of column names and
    data_rows is a list of rows (each a list of cell strings).

    Skips border/separator lines and also checks for continuation
    rows (which have an empty first cell). Continuation rows get merged into the previous row.
    """

    header = None
    data_rows = []

    for line in table_lines:
        if re.match(r"^[╒╞╘├└─]+", line):
            continue
        if "│" not in line:
            continue

        cells = split_table_line(line)

        if header is None:
            header = cells
            continue

        if cells and cells[0] == "":
            if data_rows:  # There should be at least one row already.
                for i, cell in enumerate(cells):
                    if cell:
                        data_rows[-1][i] += " " + cell
            else:
                continue
        else:
            data_rows.append(cells)
    return header, data_rows


def parse_file(filename):
    """
    Returns nested structure:
    {
        "0. Top Stats": {
            "0.1 Top Kernels": {header: [...], data: [...]},
            "0.2 Dispatch List": {header: [...], data: [...]}
        },
        "1. System Info": {
            "1.1 System Information": {header: [...], data: [...]}
        },
        ...
    }
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    sections = {}
    current_section = None
    current_subsection = None
    table_lines = []
    in_table = False

    for line in lines:
        line = line.rstrip("\n")

        # Skip separator lines
        if line.startswith(
            "--------------------------------------------------------------------------------"
        ):
            continue

        # Check for section header (e.g., "0. Top Stats")
        section_match = re.match(r"^\s*(\d+\. .+)$", line)
        if section_match:
            current_section = section_match.group(1).strip()
            sections[current_section] = {}
            continue

        # Check for subsection header (e.g., "0.1 Top Kernels")
        # FIXME: 1. System Info is an exception, no subsection
        subsection_match = re.match(r"^\s*(\d+\.\d+ .+)$", line)
        if subsection_match:
            current_subsection = subsection_match.group(1).strip()
            if current_section is None:
                current_section = "Uncategorized"
                sections[current_section] = {}
            continue

        # Table parsing logic
        if line.startswith("╒"):
            in_table = True
            table_lines = [line]
            continue

        if in_table:
            table_lines.append(line)
            if line.startswith("╘"):
                if current_section and current_subsection:
                    header, data = parse_ascii_table(table_lines)
                    sections[current_section][current_subsection] = {
                        "header": header,
                        "data": data,
                    }
                in_table = False
                table_lines = []

    return sections


def get_table_dfs():
    filename = str(Path(os.getcwd()).joinpath("analyze_output.csv"))
    sections_info = parse_file(filename)

    # Convert to DataFrames while maintaining nested structure
    section_dfs = {}
    for section_name, subsections in sections_info.items():
        section_dfs[section_name] = {}
        for subsection_name, table_data in subsections.items():
            if table_data and table_data["data"]:
                try:
                    df = pd.DataFrame(table_data["data"], columns=table_data["header"])
                    section_dfs[section_name][subsection_name] = df
                except Exception as e:
                    print(f"Error creating DataFrame for {subsection_name}: {e}")
                    continue

    return section_dfs


def analyze_runner(workload_path):
    filename = str(Path(os.getcwd()).joinpath("analyze_output.csv"))

    runner = RocprofRunner()

    stdout_output, stderr_output, exit_code, cmd_str = runner.run_analyze(
        input_dir=workload_path, output_file=filename
    )

    return stdout_output, stderr_output, exit_code, cmd_str
