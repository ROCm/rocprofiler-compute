"""
Configuration Module
-------------------
Central configuration for the application.
"""

# Application settings
APP_TITLE = "ROCm Compute Profiler TUI"
VERSION = "1.0.0"

# Analysis settings
SECTIONS_TO_SKIP = [
    "0. Top Stats",
    "1. System Info",
    "2. System Speed-of-Light",
    "3. Memory Chart",
]

# Widget configurations
DEFAULT_COLLAPSIBLE_STATE = True  # True = collapsed by default

# File paths
DEFAULT_START_PATH = None  # None uses cwd