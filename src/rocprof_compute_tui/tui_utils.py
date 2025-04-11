import re
import pandas as pd


def parse_section_header(line):
    """
    Parse a line that looks like a table header, e.g.
       "0.1 Top Kernels"
    Returns a tuple (section, name) if the line matches, or None.
    The regex here requires at least one dot followed by digits.
    """

    m = re.match(r"^\s*(\d+(?:\.\d+)+)\s+(.+)$", line)
    if m:
        section = m.group(1).strip()
        name = m.group(2).strip()
        return section, name
    return None


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
    Read the file and parse its content.
    Detects headers (like “0.1 Top Kernels”) and
    ASCII table (delimited by a top border starting with "╒" and bottom border with "╘").
    Returns a list of dicts with keys: section, name, header, data.
    """
    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    tables = []
    current_section = None
    current_table_name = None
    table_lines = []
    in_table = False

    for line in lines:
        line = line.rstrip("\n")
        sec = parse_section_header(line)
        if sec:
            current_section, current_table_name = sec

        if line.startswith("╒"):
            in_table = True
            table_lines = [line]
            continue

        if in_table:
            table_lines.append(line)
            # When reaching a bottom border (starting with "╘"), assume table is complete.
            if line.startswith("╘"):
                header, data = parse_ascii_table(table_lines)
                tables.append(
                    {
                        "section": current_section,
                        "name": current_table_name,
                        "header": header,
                        "data": data,
                    }
                )
                in_table = False
                table_lines = []
    return tables


def section_key(section_str):
    """
    Convert a section string like "0.1" or "2.1"
    into a tuple of integers for sorting.
    """
    parts = section_str.split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        # if conversion fails, fallback to the original string
        return section_str


def get_table_dfs():
    # FIXME: update the path!!!
    filename = "/home/xuchen/dev/rocprofiler-compute/TUI_OUTPUT.txt"
    tables_info = parse_file(filename)

    tables_info.sort(key=lambda t: section_key(t["section"]))

    table_dfs = {}
    for table in tables_info:
        df = pd.DataFrame(table["data"], columns=table["header"])
        key = f"{table['section']} {table['name']}"
        table_dfs[key] = df

    return table_dfs
