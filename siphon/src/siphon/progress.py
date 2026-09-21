"""Reporter interface so analysis code stays decoupled from the terminal UI."""
from __future__ import annotations

from typing import Iterable, TypeVar

T = TypeVar("T")


class Reporter:
    """No-op reporter base class; subclass to actually display progress."""

    def stage(self, message: str) -> None:
        """A new phase of work started (no-op here)."""

    def detail(self, message: str) -> None:
        """Something worth mentioning within the current phase (no-op here)."""

    def track(self, items: Iterable[T], description: str, total: int | None = None) -> Iterable[T]:
        """Wrap an iterable to report progress over it; default just passes items through."""
        return items
