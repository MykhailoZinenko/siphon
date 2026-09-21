"""Shared rich-based presentation layer: header, numbered stages, tables, footer."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                           TextColumn, TimeElapsedColumn)
from rich.table import Table

from . import runs
from .progress import Reporter

console = Console()


def _progress() -> Progress:
    """Build a rich Progress bar with the columns this UI always uses."""
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


class Session(Reporter):
    """Header, numbered stages, tables, footer - and the run folder."""

    def __init__(self, title: str, *, subtitle: Sequence[str] = (),
                 sources=None, hardware: bool = False, total_stages: int | None = None):
        """Print the header panel and, if sources are given, open a run folder + log."""
        self.title = title
        self.hardware = hardware
        self.total = total_stages
        self.n = 0
        self.run = runs.create(title, sources) if sources is not None else None  # snapshot configs into runs/
        from .logging_setup import setup
        self.log = setup(self.run.log) if self.run else None

        body = [f"[bold]{title}[/bold]"]
        body += [f"[dim]{line}[/dim]" for line in subtitle]
        if hardware:
            body.append("[yellow]touches the device: flashes and captures[/yellow]")
        if self.run:
            body.append(f"[dim]run: {self.run.root.name}[/dim]")
        console.print(Panel.fit("\n".join(body),
                                border_style="yellow" if hardware else "cyan"))

    # -- Reporter interface -------------------------------------------------
    def stage(self, message: str) -> None:
        """Print a new numbered top-level stage."""
        self.n += 1
        counter = f"{self.n}/{self.total}" if self.total else str(self.n)
        self._print(f"[bold cyan]{counter}[/bold cyan] {message}")

    def detail(self, message: str) -> None:
        """Print an indented detail line under the current stage."""
        self._print(f"    [dim]{message}[/dim]")


    def track(self, items: Iterable, description: str, total: int | None = None) -> Iterable:
        """Yield items while driving a progress bar task; passthrough if no bar is open."""
        if self._bar is None:
            yield from items
            return
        task = self._bar.add_task(f"    {description}", total=total)
        for item in items:
            yield item
            self._bar.advance(task)
        self._bar.remove_task(task)

    @property
    def sub(self) -> Reporter:
        """A nested reporter whose stages print as this session's details, not new top-level counts."""
        return _Nested(self)

    # -- plumbing -----------------------------------------------------------
    _bar: Progress | None = None

    def _print(self, text: str) -> None:
        """Print to the active progress bar's console (or the plain console), and log it."""
        (self._bar.console if self._bar else console).print(text)
        if self.log:
            from rich.text import Text
            self.log.info(Text.from_markup(text).plain.strip())  # strip markup for the log file

    def __enter__(self) -> "Session":
        """Open the progress bar for the duration of the `with` block."""
        self._bar = _progress()
        self._bar.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        """Close the progress bar."""
        bar, self._bar = self._bar, None
        bar.__exit__(*exc)

    # -- output -------------------------------------------------------------
    def table(self, columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        """Print a table via the module-level helper."""
        table(columns, rows)

    def done(self, summary: str, *, ok: bool = True, artefact: Path | None = None) -> None:
        """Print the final summary line, coloured by success, with an optional artefact path."""
        colour = "green" if ok else "yellow"
        tail = f"   [dim]{short(artefact)}[/dim]" if artefact else ""
        console.print(f"\n[{colour} bold]{summary}[/{colour} bold]{tail}")
        if self.log:
            self.log.info(summary)


class _Nested(Reporter):
    """Reporter that forwards stages and details to the parent session's detail line."""

    def __init__(self, parent: Session):
        self.parent = parent

    def stage(self, message: str) -> None:
        """Forward as a detail line, not a new top-level stage."""
        self.parent.detail(message)

    def detail(self, message: str) -> None:
        """Forward to the parent's detail line."""
        self.parent.detail(message)

    def track(self, items, description, total=None):
        """Forward to the parent's progress bar."""
        return self.parent.track(items, description, total=total)


def short(path: Path) -> str:
    """Path relative to the cwd if possible, else the path unchanged."""
    try:
        return str(Path(path).resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def table(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """Print a plain rich table with a right-aligned columns after the first."""
    t = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    for i, c in enumerate(columns):
        t.add_column(c, justify="left" if i == 0 else "right")
    for r in rows:
        t.add_row(*[str(v) for v in r])
    console.print(t)


def mark(ok: bool) -> str:
    """Coloured ok/NO marker for tables."""
    return "[green]ok[/green]" if ok else "[red]NO[/red]"


def banner(title: str, subtitle: Sequence[str] = ()) -> None:
    """Header panel only, for commands that don't run stages (just inspect things)."""
    body = [f"[bold]{title}[/bold]"] + [f"[dim]{l}[/dim]" for l in subtitle]
    console.print(Panel.fit("\n".join(body), border_style="cyan"))
