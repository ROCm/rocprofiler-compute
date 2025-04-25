from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import (
    Button,
    Header,
    Footer,
    Static,
    ListView,
    ListItem,
    Label,
    Markdown,
)
from textual.screen import Screen
from textual import on, events, work
from textual.events import Key
import datetime
import logging
import os


class RecentItem(ListItem):
    """Custom list item for recent with timestamp"""
    def __init__(self, path: str, timestamp: str):
        super().__init__()
        self.path = path
        self.timestamp = timestamp
        self.path_display = os.path.basename(path)

    def compose(self) -> ComposeResult:
        yield Label(f"[b]{self.path_display}[/b]\n[dim]{self.timestamp}[/dim]")

class MainMenuView(Screen):
    """Professional main menu with multiple panels"""

    CSS = """
    #main-container {
        layout: grid;
        grid-size: 4 4;
        grid-columns: 1fr 1fr;
        grid-rows: auto 1fr;
        padding: 1;
    }

    #welcome-panel {
        column-span: 4;
        row-span: 1;
        height: auto;
        border: round $primary;
        padding: 1;
    }

    .panel-header {
        text-style: bold;
        color: $primary;
        width: 100%;
        text-align: center;
        padding: 1;
    }

    .panel-title {
        text-style: bold;
        color: white;
        width: 100%;
        text-align: center;
    }

    #recent-profiles {
        row-span: 3;
        height: 100%;
        border: round $primary;
        margin: 1;
    }

    #quick-actions {
        column-span: 3;
        row-span: 3;
        height: 100%;
        border: round $primary;
        margin: 1;
    }

    #profile-list {
        height: 1;
        margin: 1;
    }

    Button {
        width: 100%;
        margin: 1;
    }

    Button:focus {
        background: $accent;
    }

    #last-updated {
        text-align: center;
        padding: 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header()
        yield Footer()

        with Container(id="main-container"):
            # Welcome panel (top)
            with Vertical(id="welcome-panel"):
                yield Label("ROCm Compute Profiler", classes="panel-header")
                yield Static("Ready to profile", id="last-updated")

            # Recent profiles panel
            with Vertical(id="recent-profiles"):
                yield Label("Recent Analyzes", classes="panel-title")
                self.recent_analyzes = ListView(id="analyze-list")
                yield self.recent_analyzes

                with Vertical():
                    yield Button("Open Profile", id="open-profile", disabled=True)
                    yield Button("Clear History", id="clear-history", variant="error", disabled=True)

            # Quick actions panel
            with Vertical(id="quick-actions"):
                yield Label("Quick Actions", classes="panel-title")
                yield Button("New Analyze (Demo)", id="new-profile", variant="success")

                # TODO
                yield (Markdown("🚧 Under Construction"))

                # yield Button("Compare Runs", id="compare-runs", disabled=True)
                # yield Button("Settings", id="settings", disabled=True)

        # Load initial data
        self.load_recent_profiles()
        self.load_recent_analyzes()

    def on_mount(self) -> None:
        self.set_interval(1, self.update_clock)

    def update_clock(self) -> None:
        """Update the clock every second"""
        self.query_one("#last-updated", Static).update(f"Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    @work(thread=True)
    def load_recent_profiles(self) -> None:
        """Load recent profile history from disk"""
        try:
            # Simulated data - replace with actual profile loading
            recent_profiles = [
                ("/home/xuchen/dev/rocprofiler-compute/sample/vmem", "2024-05-20 14:30:22"),
            ]

            def update_ui():
                self.recent_profiles.clear()
                for path, timestamp in recent_profiles:
                    self.recent_profiles.append(RecentItem(path, timestamp))

            self.app.call_from_thread(update_ui)
        except Exception as e:
            logging.error(f"Error loading profiles: {e}")

    @work(thread=True)
    def load_recent_analyzes(self) -> None:
        """Load recent analyze history from disk"""
        try:
            # Simulated data - replace with actual profile loading
            recent_analyzes = [
                ("/home/xuchen/dev/rocprofiler-compute/workloads/vmem/MI300X_A1", "2024-05-20 14:30:22"),
            ]

            def update_ui():
                self.recent_analyzes.clear()
                for path, timestamp in recent_analyzes:
                    self.recent_analyzes.append(RecentItem(path, timestamp))

            self.app.call_from_thread(update_ui)
        except Exception as e:
            logging.error(f"Error loading analyze: {e}")

    @on(ListView.Selected, "#profile-list")
    def on_profile_selected(self, event: ListView.Selected) -> None:
        """Handle profile selection"""
        if isinstance(event.item, RecentItem):
            self.notify(f"Selected Profile: {event.item.path_display}")


    @on(Button.Pressed, "#open-profile")
    def on_open_profile(self) -> None:
        """Handle open profile action"""
        self.app.push_screen("analyze")

    @on(Button.Pressed, "#analyze-results")
    def on_analyze_results(self) -> None:
        """Handle analyze results action"""
        self.app.push_screen("analyze")

    @on(Button.Pressed, "#new-profile")
    def on_new_profile(self) -> None:
        """Handle new profile action"""
        self.app.push_screen("test")

    @on(Key)
    def on_key(self, event: Key) -> None:
        """Global key handling"""
        if event.key == "escape":
            self.app.exit()