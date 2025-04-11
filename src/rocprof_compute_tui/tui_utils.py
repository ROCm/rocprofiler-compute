import re
import pandas as pd

def parse_section_header(line):
    """
    Parse a line that looks like a table header, e.g.
       "0.1 Top Kernels"
    Returns a tuple (section, name) if the line matches, or None.
    The regex here requires at least one dot followed by digits.
    """
    m = re.match(r'^\s*(\d+(?:\.\d+)+)\s+(.+)$', line)
    if m:
        section = m.group(1).strip()
        name = m.group(2).strip()
        return section, name
    return None

def split_table_line(line):
    """
    Splits a table row line (which is bounded by vertical bar characters)
    into a list of cell strings (trimmed). For example:

    │    │ Kernel_Name                              │   Count │ ...

    The function removes the first and last empty pieces.
    """
    # split by the vertical bar; the row always starts and ends with "│"
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

    This function skips border/separator lines and also checks for continuation
    rows (which have an empty first cell). Continuation rows get merged into the previous row.
    """
    header = None
    data_rows = []

    for line in table_lines:
        # Skip lines that are just borders or separators:
        if re.match(r'^[╒╞╘├└─]+', line):
            continue
        # Only process lines containing the vertical bar
        if "│" not in line:
            continue

        cells = split_table_line(line)

        # The very first valid row (after the top border) is assumed to be the header.
        if header is None:
            header = cells
            continue

        # For data rows, check if it is a continuation line:
        # (Assuming that continuation lines have an empty first cell)
        if cells and cells[0] == "":
            # Merge cells with the last appended row.
            # For each column cell that is not empty, append it to the corresponding cell.
            if data_rows:  # There should be at least one row already.
                for i, cell in enumerate(cells):
                    if cell:  # Only if this cell has additional content, merge it.
                        # Append with a space (you can adjust the separator as needed)
                        data_rows[-1][i] += " " + cell
            else:
                # If by any chance there's no previous row, just skip.
                continue
        else:
            data_rows.append(cells)
    return header, data_rows

def parse_file(filename):
    """
    Read the file and parse its content.
    It detects table section headers (like “0.1 Top Kernels”) and then the following
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
        # Look for section header lines. They match things like "0.1 Top Kernels" or "2.1 Speed-of-Light".
        sec = parse_section_header(line)
        if sec:
            # Update the current section and table name.
            current_section, current_table_name = sec

        # Detect start of a table (which always starts with the top border, using the box-drawing char "╒")
        if line.startswith("╒"):
            in_table = True
            table_lines = [line]
            continue

        if in_table:
            table_lines.append(line)
            # When reaching a bottom border (starting with "╘"), assume table is complete.
            if line.startswith("╘"):
                header, data = parse_ascii_table(table_lines)
                tables.append({
                    "section": current_section,
                    "name": current_table_name,
                    "header": header,
                    "data": data
                })
                in_table = False
                table_lines = []
    return tables

def section_key(section_str):
    """
    Helper function to convert a section string like "0.1" or "2.1"
    into a tuple of integers for sorting.
    """
    parts = section_str.split(".")
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        # if conversion fails, fallback to the original string
        return section_str

def get_table_dfs():
    filename = "/home/xuchen/dev/rocprofiler-compute/TUI_OUTPUT.txt"
    tables_info = parse_file(filename)

    # Sort tables based on section number, using our helper function
    tables_info.sort(key=lambda t: section_key(t["section"]))

    # Create a dictionary of pandas DataFrames.
    # The key is created from the section and name (e.g. "0.1 Top Kernels")
    table_dfs = {}
    for table in tables_info:
        # Create a DataFrame from the extracted header and data rows.
        df = pd.DataFrame(table["data"], columns=table["header"])
        key = f"{table['section']} {table['name']}"
        table_dfs[key] = df

    return table_dfs
