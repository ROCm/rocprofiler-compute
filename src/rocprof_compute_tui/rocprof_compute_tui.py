from home_viewer import MainMenuView
from rocprof_compute_tui_analyze_viewer import AnalysisScreen
from textual import on
from textual.app import App, ComposeResult
from textual.events import Key
from textual.screen import Screen
from textual.widgets import (
    Button,
    DirectoryTree,
    Header,
    Static,
)


class FolderOnlyDirectory(DirectoryTree):
    def filter_paths(self, paths):
        return [path for path in paths if path.is_dir()]


class ProfileStubScreen(Screen):
    """Placeholder screen for Profile"""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("Profile configuration will go here", id="profile-content")
        yield Button("Back", id="back-button")

    @on(Button.Pressed, "#back-button")
    def back_to_main(self):
        self.app.pop_screen()


class RocProfTUI(App):
    """Main application class"""

    CSS = """
    Screen {
        layout: vertical;
    }

    #analyze-header, #profile-content {
        margin: 1;
        padding: 1;
    }

    #dir-tree {
        margin: 1;
        border: round $accent;
        height: 1fr;
    }

    Button {
        margin: 1;
        width: 24;
    }
    """

    SCREENS = {
        "home": MainMenuView,
        "profile": ProfileStubScreen,
        "test": AnalysisScreen,
    }

    def on_mount(self) -> None:
        self.title = "ROCm Compute Profiler"
        self.push_screen("home")

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.exit()


if __name__ == "__main__":
    app = RocProfTUI()
    app.run()
