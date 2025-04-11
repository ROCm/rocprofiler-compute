from pathlib import Path
import pandas as pd
from textual.app import App, ComposeResult
from textual.widgets import (
    Button,
    Static,
    DirectoryTree,
    TabbedContent,
    TabPane,
    Log,
    Markdown,
)
from textual.containers import Vertical
from textual.screen import Screen
from textual.events import Key
from textual import on
import datetime
import logging

from rocprof_compute_cmd import RocprofRunner
from tui_utils import get_table_dfs


# -------------------------------------------------------------------
# Main Menu View
# -------------------------------------------------------------------


class MainMenuView(Static):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focused_button_index = 0

    def compose(self) -> ComposeResult:
        with Vertical():
            self.profile_button = Button("Profile", id="profile", variant="default")
            self.analyze_button = Button("Analyze", id="analyze", variant="default")
            self.quit_button = Button("Quit", id="quit", variant="error")

            yield self.profile_button
            yield self.analyze_button
            yield self.quit_button

    def on_mount(self) -> None:
        self.buttons = [self.profile_button, self.analyze_button, self.quit_button]
        self.buttons[self.focused_button_index].focus()

    def on_key(self, event: Key) -> None:
        if event.key == "up":
            # move index up (decrement), wrap around if needed
            self.focused_button_index = (self.focused_button_index - 1) % len(
                self.buttons
            )
            self.buttons[self.focused_button_index].focus()
        elif event.key == "down":
            # move index down (increment), wrap around
            self.focused_button_index = (self.focused_button_index + 1) % len(
                self.buttons
            )
            self.buttons[self.focused_button_index].focus()


# -------------------------------------------------------------------
# Profile Screen (#TODO: Placeholder)
# -------------------------------------------------------------------


class ProfileStubScreen(Screen):
    """Minimal placeholder screen for Profile functionality."""

    def on_key(self, event: Key) -> None:
        if event.key == "q":
            self.app.pop_screen()


# -------------------------------------------------------------------
# Analyze Screen (Directory Navigator)
# -------------------------------------------------------------------


class AnalyzeScreen(Screen):

    def __init__(self, start_path=None) -> None:
        super().__init__()
        if start_path is None:
            start_path = Path.home()  # default to user's home directory
        self.start_path = Path(start_path)
        self.selected_item: Path | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            "Analyze Screen\n"
            "• Navigate with arrow keys\n"
            "• Press Space or Enter to expand/collapse directories\n"
            "• Press 's' to select the currently highlighted item for analysis\n"
            "• Press 'q' to go back to the main menu\n",
            id="analyze-header",
        )
        # DirectoryTree automatically displays a file tree
        self.dir_tree = DirectoryTree(self.start_path, id="dir-tree")
        yield self.dir_tree

    @on(DirectoryTree.DirectorySelected, "#dir-tree")
    def directory_chosen(self, event: DirectoryTree.DirectorySelected) -> None:
        """When a directory is highlighted by the user."""
        self.selected_item = event.path
        self.app.log(f"Selected directory: {event.path}")

    @on(DirectoryTree.FileSelected, "#dir-tree")
    def file_chosen(self, event: DirectoryTree.FileSelected) -> None:
        """When a file is highlighted by the user."""
        self.selected_item = event.path
        self.app.log(f"Selected file: {event.path}")

    def on_key(self, event: Key) -> None:
        if event.key.lower() == "q":
            # Return to the main menu
            self.app.pop_screen()
        elif event.key.lower() == "s":
            if self.selected_item is not None:
                # Here is where you'd integrate your CLI logic or data parsing logic
                # For demonstration, we just move to AnalysisResultsScreen
                logging.info("Workload path selected:" + str(self.selected_item))
                self.app.push_screen(AnalysisResultsScreen(self.selected_item))
            else:
                logging.error("No item is currently selected for analysis.")


# -------------------------------------------------------------------
# Analysis Results Screen
# -------------------------------------------------------------------


class AnalysisResultsScreen(Screen):
    """
    Displays different analysis views of the selected directory/file.
    For demonstration, we'll show:
    - A "Summary" view
    - A "DataFrame" view
    - A "Logs" view
    You can expand or rename these as needed.
    """

    def __init__(self, selected_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_path = selected_path
        self.return_msg = "Analyze run failed."

        ##############################
        # FIXME CLI integration
        ##############################

        self.exit_code = analyze_runner(selected_path)

        if self.exit_code == 0:
            self.return_msg = "Analyze run completed."

        self.table_dfs = get_table_dfs()

    def compose(self) -> ComposeResult:
        yield Static(
            f"Analysis Results Screen\n"
            f"Selected Path: {self.selected_path}\n"
            f"{self.return_msg}\n"
            "Use the tabs below to switch between analysis views.\n"
            "[Press 'q' to go back]",
            id="analysis-header",
            markup=False,
        )

        # TabbedContent automatically wraps TabPanes into a tab-like interface
        with TabbedContent():
            # Summary tab
            with TabPane("Summary", id="summary-tab"):
                yield self.build_summary_view()

            # DataFrame tab
            with TabPane("DataFrame", id="df-tab"):
                yield self.build_dataframe_view()

            # Logs or details tab
            with TabPane("Logs"):
                yield self.build_logs_view()

    def build_summary_view(self) -> Static:
        """
        For demonstration, show a markdown-like summary or
        put a textual widget describing high-level results.
        """
        # If you have a function get_summary_for_path(self.selected_path),
        # you could call that here to get the actual data.
        summary_text = (
            f"# Summary of Analysis\n\n"
            f"This is a placeholder summary for `{self.selected_path}`.\n\n"
            f"Add your real logic to parse the files, run your CLI,\n"
            f"and extract key metrics."
        )
        return Markdown(summary_text)

    def build_dataframe_view(self) -> Static:
        md_content = ""

        for key, df in self.table_dfs.items():
            md_content += f"### {key}\n\n"
            md_content += "```\n" + df.to_string(index=False) + "\n```\n\n"

        return Markdown(md_content)

    def build_logs_view(self) -> Static:
        """
        A placeholder for any log or detailed data you want to show.
        Could be a large text area or a scrolled container of lines.
        """
        logs_text = (
            f"Here, you could load logs from {self.selected_path}.\n"
            "For now, it's just placeholder text."
        )
        return Markdown(logs_text)

    def on_key(self, event: Key) -> None:
        """Press 'q' to go back to the main menu."""
        if event.key == "q":
            self.app.pop_screen()


# -------------------------------------------------------------------
# Main Application
# -------------------------------------------------------------------


class RocProfTUI(App):
    CSS = """
    MainMenuView {
        align: center middle;
        height: 100%;
        width: 100%;
        content-align: center middle;
    }

    Button {
        margin: 1;
        width: 24;
    }

    #analyze-header {
        margin: 1;
    }

    #dir-tree {
        margin: 1;
        border: round $accent;
        height: 1fr;
    }

    #analysis-header {
        margin: 1;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield MainMenuView()
        yield Static("Press ESC to quit at any time.")

    def on_key(self, event: Key) -> None:
        """Global key handler to capture escape and quit."""
        if event.key == "escape":
            self.exit()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "profile":
            self.push_screen(ProfileStubScreen())
        elif button_id == "analyze":
            self.push_screen(AnalyzeScreen())
        elif button_id == "quit":
            self.exit()


def analyze_runner(workload_path):
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/home/xuchen/dev/rocprofiler-compute/TUI_OUTPUT.txt"

    runner = RocprofRunner()

    exit_code = runner.run_analyze(input_dir=workload_path, output_file=filename)

    if exit_code == 0:
        logging.info("run_analyze WORKED")
    else:
        logging.error("SOMETHING IS WRONGGGGG")

    return exit_code


if __name__ == "__main__":
    app = RocProfTUI()
    app.run()
