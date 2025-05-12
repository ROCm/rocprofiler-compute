import time
from typing import Any, Literal

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Label


class OutputTab(VerticalScroll):
    DEFAULT_CSS = """\
        OutputTab {
            padding: 0 2;
        }
    """

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes, disabled=disabled)
        # Initialize with a non-empty string to confirm it works
        self._log_content: str = ""

    def compose(self) -> ComposeResult:
        self.can_focus = False
        # Add debug message
        if self._log_content:
            # Don't clear the log here
            yield Label(self._log_content)

    async def log_event(self, log) -> None:
        if log:
            if self._log_content:
                self._log_content += f"\n{log}"
            else:
                self._log_content = log
        await self.recompose()
