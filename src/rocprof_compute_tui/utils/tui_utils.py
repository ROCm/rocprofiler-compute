import os
import re
from pathlib import Path

import pandas as pd

from utils.rocprof_compute_cmd import RocprofRunner


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
